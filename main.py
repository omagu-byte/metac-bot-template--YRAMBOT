import argparse
import asyncio
import logging
import os
import re
import textwrap
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Union, Optional, Iterable

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

logger = logging.getLogger(__name__)

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


def pool_binary_logit(ps: Iterable[float], weights: Optional[Iterable[float]] = None) -> float:
    ps = [clamp_prob(p) for p in ps]
    if not ps:
        return 0.5
    if weights is None:
        avg = sum(logit(p) for p in ps) / len(ps)
    else:
        ws = list(weights)
        if len(ws) != len(ps) or sum(ws) <= 0:
            avg = sum(logit(p) for p in ps) / len(ps)
        else:
            s = sum(ws)
            avg = sum(w * logit(p) for w, p in zip(ws, ps)) / s
    return clamp_prob(sigmoid(avg))


def conservative_shrink_to_50(p: float, disagreement: float, k: float) -> float:
    # Shrink more when models disagree; reduces overconfidence penalties on Brier/log.
    w = 1.0 / (1.0 + k * max(0.0, disagreement))
    return clamp_prob(0.5 + w * (p - 0.5))


def temperature_transform(probs: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    temperature = max(0.05, float(temperature))
    safe = {k: max(EPS, float(v)) for k, v in probs.items()}
    exp_ = 1.0 / temperature
    powered = {k: v ** exp_ for k, v in safe.items()}
    s = sum(powered.values())
    if s <= 0:
        n = len(probs) or 1
        return {k: 1.0 / n for k in probs}
    return {k: v / s for k, v in powered.items()}


def enforce_monotonic_percentiles(percentiles: list[Percentile]) -> list[Percentile]:
    if not percentiles:
        return percentiles
    ps = sorted(percentiles, key=lambda p: p.percentile)
    ps = [Percentile(percentile=max(0.0, min(1.0, float(p.percentile))), value=float(p.value)) for p in ps]
    fixed = []
    last_pct = -1.0
    for p in ps:
        pct = p.percentile
        if pct <= last_pct:
            pct = min(1.0, last_pct + 1e-6)
        fixed.append(Percentile(percentile=pct, value=p.value))
        last_pct = pct
    vals = [p.value for p in fixed]
    for i in range(1, len(vals)):
        if vals[i] < vals[i - 1]:
            vals[i] = vals[i - 1]
    return [Percentile(percentile=p.percentile, value=vals[i]) for i, p in enumerate(fixed)]


def clamp_to_bounds(value: float, question: NumericQuestion) -> float:
    low = getattr(question, "minimum", float("-inf"))
    high = getattr(question, "maximum", float("inf"))
    if low != float("-inf"):
        value = max(low, value)
    if high != float("inf"):
        value = min(high, value)
    return value


def build_tavily_query(question: MetaculusQuestion, max_chars: int = 397) -> str:
    q = re.sub(r"http\S+", "", question.question_text.strip())
    q = re.sub(r"\s+", " ", q).strip()
    return textwrap.shorten(q, width=max_chars, placeholder="…")


def _extract_close_dt(question: MetaculusQuestion) -> Optional[datetime]:
    for attr in ("close_time", "close_datetime", "close_date", "close_at", "closeTime", "closeDate"):
        v = getattr(question, attr, None)
        if v is None:
            continue
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        if isinstance(v, str):
            try:
                dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except Exception:
                continue
    return None


async def with_timeout(coro, seconds: float, label: str):
    try:
        return await asyncio.wait_for(coro, timeout=max(1.0, float(seconds)))
    except asyncio.TimeoutError:
        raise TimeoutError(f"{label} timed out after {seconds}s")


@dataclass
class ResearchItem:
    title: str
    url: str
    snippet: str
    published_date: Optional[str] = None


def format_research(today_str: str, items: list[ResearchItem]) -> str:
    if not items:
        return f"--- RESEARCH ({today_str}) ---\nNo relevant results found."
    lines = []
    for i, it in enumerate(items, start=1):
        date_part = f" ({it.published_date})" if it.published_date else ""
        lines.append(f"[{i}] {it.title}{date_part}\nURL: {it.url}\n{it.snippet}")
    return f"--- RESEARCH ({today_str}) ---\n" + "\n\n".join(lines)


class Yrambot(ForecastBot):
    _max_concurrent_questions = int(os.getenv("MAX_CONCURRENT_QUESTIONS", "2"))
    _structure_output_validation_samples = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._concurrency_limiter = asyncio.Semaphore(self._max_concurrent_questions)
        self._mc_temperature = float(os.getenv("MC_TEMPERATURE", "1.0"))
        self._disagreement_shrink_k = float(os.getenv("DISAGREE_SHRINK_K", "8.0"))
        self._deadline_buffer_seconds = int(os.getenv("DEADLINE_BUFFER_SECONDS", "120"))
        self._research_timeout = float(os.getenv("RESEARCH_TIMEOUT_SECONDS", "35"))
        self._llm_timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
        self._tavily_retries = int(os.getenv("TAVILY_RETRIES", "2"))

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            today_str = now_utc().strftime("%Y-%m-%d")
            query = build_tavily_query(question)
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                return f"--- RESEARCH ({today_str}) ---\nResearch unavailable (missing TAVILY_API_KEY)."
            tavily_client = TavilyClient(api_key=api_key)

            async def _attempt():
                def _do_search():
                    return tavily_client.search(query=query, search_depth="advanced", max_results=6)
                return await asyncio.to_thread(_do_search)

            last_err = None
            for i in range(self._tavily_retries + 1):
                try:
                    resp = await with_timeout(_attempt(), self._research_timeout, "research")
                    res = resp.get("results", []) or []
                    items: list[ResearchItem] = []
                    for r in res:
                        title = str(r.get("title", "")).strip()
                        url = str(r.get("url", "")).strip()
                        content = str(r.get("content", "")).strip()
                        snippet = content[:400] + ("…" if len(content) > 400 else "")
                        published_date = r.get("published_date") or r.get("publishedDate") or None
                        if title and (snippet or url):
                            items.append(ResearchItem(title=title, url=url, snippet=snippet, published_date=published_date))
                    logger.info(
                        "Research Q%s query=%r results=%d",
                        getattr(question, "id", "?"),
                        query,
                        len(items),
                    )
                    return format_research(today_str, items)
                except Exception as e:
                    last_err = e
                    await asyncio.sleep(0.6 * (2 ** i))
            logger.warning("Research failed Q%s: %s", getattr(question, "id", "?"), last_err)
            return f"--- RESEARCH ({today_str}) ---\nResearch unavailable (error)."

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        p1 = await self._run_model_forecast("forecaster_1", question, research)
        p2 = await self._run_model_forecast("forecaster_2", question, research)
        pooled = pool_binary_logit([p1, p2])
        disagreement = abs(float(p1) - float(p2))
        final_p = conservative_shrink_to_50(pooled, disagreement, self._disagreement_shrink_k)
        reasoning = clean_indents(f"""
        Research:
        {research}

        Model outputs:
        - forecaster_1: {p1:.1%}
        - forecaster_2: {p2:.1%}

        Aggregation:
        - log-odds pooling then disagreement shrink -> {final_p:.1%}
        """).strip()
        return ReasonedPrediction(prediction_value=final_p, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        p1 = await self._run_model_forecast("forecaster_1", question, research)
        p2 = await self._run_model_forecast("forecaster_2", question, research)

        avg: dict[str, float] = {}
        for opt in question.options:
            v1 = next((o.probability for o in p1.predicted_options if o.option_name == opt), 0.0)
            v2 = next((o.probability for o in p2.predicted_options if o.option_name == opt), 0.0)
            avg[opt] = max(0.0, float(v1) + float(v2)) / 2.0

        s = sum(avg.values())
        if s <= 0:
            uniform = 1.0 / max(1, len(question.options))
            avg = {opt: uniform for opt in question.options}
        else:
            avg = {k: v / s for k, v in avg.items()}

        final_probs = temperature_transform(avg, temperature=self._mc_temperature)
        final_options = [PredictedOption(option_name=opt, probability=final_probs.get(opt, 0.0)) for opt in question.options]
        prediction_list = PredictedOptionList(predicted_options=final_options)
        reasoning = clean_indents(f"""
        Research:
        {research}

        Aggregation:
        - avg probs + global temperature T={self._mc_temperature:.2f}
        """).strip()
        return ReasonedPrediction(prediction_value=prediction_list, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        p1 = await self._run_model_forecast("forecaster_1", question, research)
        p2 = await self._run_model_forecast("forecaster_2", question, research)

        if not isinstance(p1, list) or not isinstance(p2, list):
            return self._numeric_fallback(question, research)

        def normalize_level(level: float) -> float:
            level = float(level)
            return level / 100.0 if level > 1.0 else level

        def process(raw: list[Percentile]) -> list[Percentile]:
            out: list[Percentile] = []
            for p in raw:
                pct = normalize_level(p.percentile)
                val = clamp_to_bounds(float(p.value), question)
                out.append(Percentile(percentile=pct, value=val))
            out = enforce_monotonic_percentiles(out)
            dedup: dict[float, float] = {}
            for pp in out:
                dedup[pp.percentile] = pp.value
            return [Percentile(percentile=k, value=v) for k, v in sorted(dedup.items())]

        try:
            p1_proc = process(p1)
            p2_proc = process(p2)
        except Exception:
            return self._numeric_fallback(question, research)

        if len(p1_proc) < 2 or len(p2_proc) < 2:
            return self._numeric_fallback(question, research)

        p1_map = {p.percentile: p.value for p in p1_proc}
        p2_map = {p.percentile: p.value for p in p2_proc}
        grid = sorted(set(p1_map.keys()) | set(p2_map.keys()))
        pooled: list[Percentile] = []
        for pct in grid:
            v1 = p1_map.get(pct, p2_map[pct])
            v2 = p2_map.get(pct, p1_map[pct])
            avg_val = clamp_to_bounds((float(v1) + float(v2)) / 2.0, question)
            pooled.append(Percentile(percentile=pct, value=avg_val))

        pooled = enforce_monotonic_percentiles(pooled)

        try:
            dist = NumericDistribution.from_question(pooled, question)
        except Exception:
            clamped = [Percentile(percentile=p.percentile, value=clamp_to_bounds(p.value, question)) for p in pooled]
            clamped = enforce_monotonic_percentiles(clamped)
            dist = NumericDistribution.from_question(clamped, question)

        reasoning = clean_indents(f"""
        Research:
        {research}

        Numeric:
        - normalize pct, clamp bounds, enforce monotonic, average aligned grid
        """).strip()
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    def _numeric_fallback(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        low = getattr(question, "minimum", 0.0)
        high = getattr(question, "maximum", None)
        if high is None or high == float("inf"):
            high = float(low) + max(1.0, abs(float(low)) * 2.0)
        low = float(low)
        high = float(high)
        mid = (low + high) / 2.0
        default_percentiles = [
            Percentile(percentile=0.1, value=low),
            Percentile(percentile=0.5, value=mid),
            Percentile(percentile=0.9, value=high),
        ]
        default_percentiles = enforce_monotonic_percentiles(default_percentiles)
        dist = NumericDistribution.from_question(default_percentiles, question)
        reasoning = clean_indents(f"""
        Research:
        {research}

        Fallback: 10/50/90 from bounds.
        """).strip()
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    async def _run_model_forecast(
        self, model_key: str, question: MetaculusQuestion, research: str
    ) -> Union[float, PredictedOptionList, list[Percentile]]:
        prompt = clean_indents(f"""
            Role: High-signal quantitative forecaster.
            Today (UTC): {now_utc().strftime('%Y-%m-%d')}

            Question: {question.question_text}
            Resolution Criteria: {getattr(question, 'resolution_criteria', 'Not specified')}
            Background: {getattr(question, 'background_info', '')}

            Research:
            {research}

            Requirements:
            1) Explicit base rate / reference class.
            2) Top 2-5 update signals.
            3) Final forecast consistent with evidence.
            4) Avoid unjustified certainty; extreme probs require strong evidence.

            Output must match schema for this question type.
        """).strip()

        llm = self.get_llm(model_key, "llm")

        async def _invoke():
            return await llm.invoke(prompt)

        model_output = await with_timeout(_invoke(), self._llm_timeout, f"llm:{model_key}")
        parser = self.get_llm("parser", "llm")

        if isinstance(question, BinaryQuestion):
            pred = await structure_output(model_output, BinaryPrediction, model=parser)
            return clamp_prob(pred.prediction_in_decimal)

        if isinstance(question, MultipleChoiceQuestion):
            pol = await structure_output(model_output, PredictedOptionList, model=parser)
            probs = {o.option_name: max(0.0, float(o.probability)) for o in pol.predicted_options}
            for opt in question.options:
                probs.setdefault(opt, 0.0)
            s = sum(probs.values())
            if s <= 0:
                uniform = 1.0 / max(1, len(question.options))
                probs = {opt: uniform for opt in question.options}
            else:
                probs = {k: v / s for k, v in probs.items()}
            final = [PredictedOption(option_name=opt, probability=probs.get(opt, 0.0)) for opt in question.options]
            return PredictedOptionList(predicted_options=final)

        return await structure_output(model_output, list[Percentile], model=parser)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5.2",
            "parser": "openrouter/openai/gpt-4.1-mini",
            "forecaster_1": "openrouter/openai/gpt-5.2",
            "forecaster_2": "openrouter/openai/gpt-5.1",
            "summarizer": "openrouter/openai/gpt-4.1-mini",
        }

    def should_skip_due_to_deadline(self, question: MetaculusQuestion) -> bool:
        close_dt = _extract_close_dt(question)
        if not close_dt:
            return False
        remaining = (close_dt - now_utc()).total_seconds()
        return remaining <= self._deadline_buffer_seconds


async def run_once(mode: str) -> None:
    bot = Yrambot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
    )
    client = MetaculusClient()

    if mode == "tournament":
        reports1 = await bot.forecast_on_tournament(client.CURRENT_AI_COMPETITION_ID, return_exceptions=True)
        reports2 = await bot.forecast_on_tournament(client.CURRENT_MINIBENCH_ID, return_exceptions=True)
        forecast_reports = reports1 + reports2
    elif mode == "test_questions":
        example_questions = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
        ]
        bot.skip_previously_forecasted_questions = False
        questions = [client.get_question_by_url(url.strip()) for url in example_questions]
        questions = [q for q in questions if q and not bot.should_skip_due_to_deadline(q)]
        forecast_reports = await bot.forecast_questions(questions, return_exceptions=True)
    else:
        raise NotImplementedError("Only 'tournament' and 'test_questions' modes are configured.")

    bot.log_report_summary(forecast_reports)


async def main(mode: str, loop_mode: bool, interval_seconds: int) -> None:
    if not loop_mode:
        await run_once(mode)
        return
    while True:
        start = now_utc()
        try:
            await run_once(mode)
        except Exception as e:
            logger.exception("Run failed: %s", e)
        elapsed = (now_utc() - start).total_seconds()
        sleep_for = max(5.0, float(interval_seconds) - elapsed)
        await asyncio.sleep(sleep_for)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Run Yrambot forecasting system")
    parser.add_argument("--mode", type=str, choices=["tournament", "test_questions"], default="tournament")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=180)
    args = parser.parse_args()
    asyncio.run(main(args.mode, args.loop, args.interval_seconds))
