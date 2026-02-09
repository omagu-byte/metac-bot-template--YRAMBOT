import argparse
import asyncio
import logging
import os
import re
import textwrap
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Union, Optional, Iterable, cast

from tavily import TavilyClient

from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    PredictedOption,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

logger = logging.getLogger("Yrambot")

EPS = 1e-12


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def clamp_prob(p: float, lo: float = 1e-6, hi: float = 1.0 - 1e-6) -> float:
    return max(lo, min(hi, float(p)))


def logit(p: float) -> float:
    p = clamp_prob(p)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def pool_binary_logit(ps: Iterable[float]) -> float:
    ps = [clamp_prob(p) for p in ps]
    if not ps:
        return 0.5
    return clamp_prob(sigmoid(sum(logit(p) for p in ps) / len(ps)))


def conservative_shrink_to_50(p: float, disagreement: float, k: float) -> float:
    w = 1.0 / (1.0 + k * max(0.0, disagreement))
    return clamp_prob(0.5 + w * (p - 0.5))


def temperature_transform(probs: dict[str, float], temperature: float) -> dict[str, float]:
    temperature = max(0.05, float(temperature))
    powered = {k: max(EPS, v) ** (1.0 / temperature) for k, v in probs.items()}
    s = sum(powered.values())
    return {k: v / s for k, v in powered.items()} if s > 0 else probs


_ARITH_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*$")


def _safe_eval(expr: str) -> Optional[float]:
    m = _ARITH_RE.match(expr)
    if not m:
        return None
    a, op, b = float(m[1]), m[2], float(m[3])
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/" and abs(b) > 1e-12:
        return a / b
    return None


def sanitize_numeric_json(text: str) -> str:
    def repl(m: re.Match) -> str:
        raw = m.group(2)
        v = _safe_eval(raw)
        return m.group(1) + (str(v) if v is not None else raw)

    return re.sub(r'("percentile"\s*:\s*)([^,\]\}\n]+)', repl, text)


def enforce_monotonic_percentiles(ps: list[Percentile]) -> list[Percentile]:
    ps = sorted(ps, key=lambda p: p.percentile)
    out = []
    last = -1.0
    for p in ps:
        pct = max(p.percentile, last + 1e-6)
        pct = min(1.0, pct)
        out.append(Percentile(percentile=pct, value=p.value))
        last = pct
    for i in range(1, len(out)):
        if out[i].value < out[i - 1].value:
            out[i] = Percentile(out[i].percentile, out[i - 1].value)
    return out


def clamp_to_bounds(v: float, q: NumericQuestion) -> float:
    if getattr(q, "minimum", None) is not None:
        v = max(v, q.minimum)
    if getattr(q, "maximum", None) is not None:
        v = min(v, q.maximum)
    return v


def build_tavily_query(q: MetaculusQuestion) -> str:
    s = re.sub(r"http\S+", "", q.question_text)
    s = re.sub(r"\s+", " ", s).strip()
    return textwrap.shorten(s, 380)


class Yrambot(ForecastBot):
    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5.2",
            "forecaster": "openrouter/openai/gpt-5.2",
            "parser": "openrouter/openai/gpt-4.1-mini",
            "summarizer": "openrouter/openai/gpt-4.1-mini",
        }

    async def _one_forecast_sample(
        self, question: MetaculusQuestion, research: str
    ):
        prompt = clean_indents(f"""
        You are Yrambot, a quantitative forecaster.

        IMPORTANT FOR NUMERIC QUESTIONS:
        - Output MUST be valid JSON
        - Use ONLY numeric literals
        - DO NOT use arithmetic (e.g. 0.5-0.25)
        - Percentiles must be numbers in [0,1]

        Question:
        {question.question_text}

        Research:
        {research}

        Output must match the required schema exactly.
        """).strip()

        llm = self.get_llm("forecaster", "llm")
        parser = self.get_llm("parser", "llm")
        summarizer = self.get_llm("summarizer", "llm")

        raw = await llm.invoke(prompt)

        try:
            if isinstance(question, NumericQuestion):
                return await structure_output(raw, list[Percentile], model=parser)
            if isinstance(question, BinaryQuestion):
                p = await structure_output(raw, BinaryPrediction, model=parser)
                return clamp_prob(p.prediction_in_decimal)
            pol = await structure_output(raw, PredictedOptionList, model=parser)
            return pol
        except Exception:
            fixed = sanitize_numeric_json(str(raw))
            return await structure_output(fixed, list[Percentile], model=parser)

    async def _run_forecast_on_numeric(
        self, q: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        samples = [await self._one_forecast_sample(q, research) for _ in range(3)]
        proc = []
        for s in samples:
            vals = []
            for p in s:
                vals.append(
                    Percentile(
                        percentile=float(p.percentile),
                        value=clamp_to_bounds(float(p.value), q),
                    )
                )
            proc.append(enforce_monotonic_percentiles(vals))

        grid = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

        def interp(ps, g):
            for i in range(1, len(ps)):
                if g <= ps[i].percentile:
                    p0, p1 = ps[i - 1], ps[i]
                    t = (g - p0.percentile) / (p1.percentile - p0.percentile)
                    return p0.value + t * (p1.value - p0.value)
            return ps[-1].value

        pooled = []
        for g in grid:
            vals = [interp(p, g) for p in proc]
            pooled.append(Percentile(g, sum(vals) / len(vals)))

        pooled = enforce_monotonic_percentiles(pooled)
        dist = NumericDistribution.from_question(pooled, q)

        return ReasonedPrediction(
            prediction_value=dist,
            reasoning="Yrambot numeric forecast with self-consistency and parser-safe repair.",
        )


async def main():
    bot = Yrambot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
    )
    client = MetaculusClient()
    await bot.forecast_on_tournament("market-pulse-26q1")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - Yrambot - %(levelname)s - %(message)s",
    )
    asyncio.run(main())
