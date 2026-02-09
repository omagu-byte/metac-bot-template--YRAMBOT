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
from typing import Union, Optional, Iterable, Any, cast

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
    fixed: list[Percentile] = []
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


def _time_to_close_days(question: MetaculusQuestion) -> Optional[float]:
    close_dt = _extract_close_dt(question)
    if not close_dt:
        return None
    return (close_dt - now_utc()).total_seconds() / 86400.0


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
        return f"--- EXTERNAL RESEARCH ({today_str}) ---\nNo relevant results found."
    lines = []
    for i, it in enumerate(items, start=1):
        date_part = f" ({it.published_date})" if it.published_date else ""
        lines.append(f"[{i}] {it.title}{date_part}\nURL: {it.url}\n{it.snippet}")
    return f"--- EXTERNAL RESEARCH ({today_str}) ---\n" + "\n\n".join(lines)


class ResearchBudget:
    def __init__(self):
        self.max_calls_per_run = int(os.getenv("TAVILY_MAX_CALLS_PER_RUN", "200"))
        self.max_calls_per_question = int(os.getenv("TAVILY_MAX_CALLS_PER_QUESTION", "1"))
        self.calls_made_run = 0
        self.calls_made_question: dict[str, int] = {}

    def can_call(self, qid: str) -> bool:
        if self.calls_made_run >= self.max_calls_per_run:
            return False
        if self.calls_made_question.get(qid, 0) >= self.max_calls_per_question:
            return False
        return True

    def record(self, qid: str) -> None:
        self.calls_made_run += 1
        self.calls_made_question[qid] = self.calls_made_question.get(qid, 0) + 1


class Yrambot(ForecastBot):
    _max_concurrent_questions = int(os.getenv("MAX_CONCURRENT_QUESTIONS", "2"))
    _structure_output_validation_samples = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._question_limiter = asyncio.Semaphore(self._max_concurrent_questions)

        self._llm_concurrency = int(os.getenv("MAX_CONCURRENT_LLM", "3"))
        self._llm_limiter = asyncio.Semaphore(max(1, self._llm_concurrency))

        self._self_consistency_n = int(os.getenv("SELF_CONSISTENCY_N", "3"))
        self._llm_retries = int(os.getenv("LLM_RETRIES", "1"))

        self._mc_temperature = float(os.getenv("MC_TEMPERATURE", "1.05"))
        self._disagreement_shrink_k = float(os.getenv("DISAGREE_SHRINK_K", "10.0"))

        self._deadline_buffer_seconds = int(os.getenv("DEADLINE_BUFFER_SECONDS", "120"))
        self._research_timeout = float(os.getenv("RESEARCH_TIMEOUT_SECONDS", "35"))
        self._llm_timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
        self._tavily_retries = int(os.getenv("TAVILY_RETRIES", "1"))

        self._research_cache_ttl_seconds = int(os.getenv("RESEARCH_CACHE_TTL_SECONDS", "3600"))
        self._research_cache: dict[str, tuple[float, str]] = {}

        self._budget = ResearchBudget()
        self._tavily_client: Optional[TavilyClient] = None

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5.2",
            "forecaster": "openrouter/openai/gpt-5.2",
            "parser": "openrouter/openai/gpt-4.1-mini",
            "summarizer": "openrouter/openai/gpt-4.1-mini",
        }

    def _cache_get(self, key: str) -> Optional[str]:
        item = self._research_cache.get(key)
        if not item:
            return None
        ts, value = item
        if (time.time() - ts) <= self._research_cache_ttl_seconds:
            return value
        self._research_cache.pop(key, None)
        return None

    def _cache_set(self, key: str, value: str) -> None:
        self._research_cache[key] = (time.time(), value)

    def _get_tavily(self) -> Optional[TavilyClient]:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return None
        if self._tavily_client is None:
            self._tavily_client = TavilyClient(api_key=api_key)
        return self._tavily_client

    async def _tavily_search_once(self, query: str) -> list[ResearchItem]:
        tavily = self._get_tavily()
        if not tavily:
            return []

        def _do_search():
            return tavily.search(query=query, search_depth="advanced", max_results=6)

        resp = await asyncio.to_thread(_do_search)
        res = resp.get("results", []) or []
        items: list[ResearchItem] = []
        for r in res:
            title = str(r.get("title", "")).strip()
            url = str(r.get("url", "")).strip()
            content = str(r.get("content", "")).strip()
            snippet = content[:400] + ("…" if len(content) > 400 else "")
            published_date = r.get("published_date") or r.get("publishedDate") or None
            if title and (snippet or url):
                items.append(
                    ResearchItem(
                        title=title,
                        url=url,
                        snippet=snippet,
                        published_date=cast(Optional[str], published_date),
                    )
                )
        return items

    async def _summarize_research_always(self, question: MetaculusQuestion, raw_research: str) -> str:
        summarizer = self.get_llm("summarizer", "llm")

        close_dt = _extract_close_dt(question)
        close_str = close_dt.isoformat() if close_dt else "Unknown"
        ttc_days = _time_to_close_days(question)
        ttc_str = f"{ttc_days:.1f} days" if ttc_days is not None else "Unknown"

        prompt = clean_indents(f"""
        You are Yrambot's research editor. Produce a concise, high-signal forecasting brief.

        Constraints:
        - 6–12 bullets max
        - include dates and numbers (if present)
        - prioritize reputable/primary sources when available
        - explicitly extract:
          (1) base rates / reference class
          (2) strongest update signals
          (3) what would change the forecast before close

        Question:
        {question.question_text}

        Close time (UTC): {close_str}
        Time remaining: {ttc_str}

        Research:
        {raw_research}
        """).strip()

        try:
            async with self._llm_limiter:
                summary = await with_timeout(summarizer.invoke(prompt), self._llm_timeout, "llm:summarizer")
            today_str = now_utc().strftime("%Y-%m-%d")
            return clean_indents(f"""
            --- RESEARCH ({today_str}) ---
            Yrambot SUMMARY (always provided):
            {summary}

            SOURCES:
            {raw_research}
            """).strip()
        except Exception as e:
            logger.warning("Yrambot summarizer failed Q%s: %s", getattr(question, "id", "?"), e)
            today_str = now_utc().strftime("%Y-%m-%d")
            return clean_indents(f"""
            --- RESEARCH ({today_str}) ---
            Yrambot SUMMARY: unavailable (summarizer error)

            SOURCES:
            {raw_research}
            """).strip()

    def _evidence_thin_factor(self, research: str) -> float:
        if "No relevant results found" in research:
            return 0.6
        if "missing TAVILY_API_KEY" in research:
            return 1.0
        if "summarizer error" in research:
            return 0.8
        lines = [ln.strip() for ln in research.splitlines() if ln.strip()]
        if len(lines) < 10:
            return 0.4
        return 0.0

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._question_limiter:
            qid = str(getattr(question, "id", "")) or "unknown"
            cache_key = f"yrambot:research:{qid}"
            cached = self._cache_get(cache_key)
            if cached:
                return cached

            today_str = now_utc().strftime("%Y-%m-%d")
            query = build_tavily_query(question)

            meta_context = clean_indents(f"""
            --- METACULUS CONTEXT ({today_str}) ---
            Question: {question.question_text}
            Resolution Criteria: {getattr(question, "resolution_criteria", "Not specified")}
            Background: {getattr(question, "background_info", "")}
            """).strip()

            items: list[ResearchItem] = []
            tavily = self._get_tavily()
            if not tavily:
                raw = meta_context + "\n\n" + f"--- EXTERNAL RESEARCH ({today_str}) ---\nResearch unavailable (missing TAVILY_API_KEY)."
                summarized = await self._summarize_research_always(question, raw)
                self._cache_set(cache_key, summarized)
                return summarized

            if self._budget.can_call(qid):
                last_err: Optional[Exception] = None
                for i in range(self._tavily_retries + 1):
                    try:
                        self._budget.record(qid)
                        items = await with_timeout(self._tavily_search_once(query), self._research_timeout, "research:tavily")
                        logger.info("Yrambot Tavily Q%s query=%r results=%d", qid, query, len(items))
                        break
                    except Exception as e:
                        last_err = e
                        await asyncio.sleep(0.6 * (2 ** i))
                if not items and last_err:
                    logger.warning("Yrambot Tavily failed Q%s: %s", qid, last_err)

            external_block = format_research(today_str, items)
            raw_research = meta_context + "\n\n" + external_block

            summarized = await self._summarize_research_always(question, raw_research)
            self._cache_set(cache_key, summarized)
            return summarized

    async def _one_forecast_sample(
        self, question: MetaculusQuestion, research: str
    ) -> Union[float, PredictedOptionList, list[Percentile]]:
        close_dt = _extract_close_dt(question)
        close_str = close_dt.isoformat() if close_dt else "Unknown"
        ttc_days = _time_to_close_days(question)
        ttc_str = f"{ttc_days:.1f} days" if ttc_days is not None else "Unknown"

        prompt = clean_indents(f"""
            You are Yrambot, a high-signal quantitative forecaster.
            Today (UTC): {now_utc().strftime('%Y-%m-%d')}

            Question: {question.question_text}
            Resolution Criteria: {getattr(question, 'resolution_criteria', 'Not specified')}
            Background: {getattr(question, 'background_info', '')}

            Close time (UTC): {close_str}
            Time remaining: {ttc_str}

            Research (summary first, then sources):
            {research}

            Requirements (scoring-optimized):
            1) Start with an explicit base rate / reference class.
            2) Apply 2–5 evidence-based updates (with dates/numbers when possible).
            3) Avoid unjustified certainty. Extreme probabilities require strong evidence.
            4) Produce a calibrated forecast; be conservative under weak evidence.

            Output must match schema for this question type.
        """).strip()

        llm = self.get_llm("forecaster", "llm")
        parser = self.get_llm("parser", "llm")
        summarizer = self.get_llm("summarizer", "llm")

        async with self._llm_limiter:
            model_output = await with_timeout(llm.invoke(prompt), self._llm_timeout, "llm:forecaster")

        for attempt in range(self._llm_retries + 1):
            try:
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

            except Exception as e:
                if attempt >= self._llm_retries:
                    raise e
                repair_prompt = clean_indents(f"""
                Rewrite the following output to strictly match the required JSON schema for this question type.
                Output only the schema. Do not add extra keys.

                Output:
                {model_output}
                """).strip()
                async with self._llm_limiter:
                    model_output = await with_timeout(summarizer.invoke(repair_prompt), self._llm_timeout, "llm:repair")

        raise RuntimeError("Yrambot parse failed after retries")

    async def _run_model_forecast_samples(
        self, question: MetaculusQuestion, research: str
    ) -> list[Union[float, PredictedOptionList, list[Percentile]]]:
        n = max(1, int(self._self_consistency_n))
        samples: list[Union[float, PredictedOptionList, list[Percentile]]] = []
        for _ in range(n):
            samples.append(await self._one_forecast_sample(question, research))
        return samples

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        samples = await self._run_model_forecast_samples(question, research)
        ps = [float(x) for x in samples if isinstance(x, (int, float))]

        pooled = pool_binary_logit(ps) if ps else 0.5
        disagreement = (max(ps) - min(ps)) if len(ps) > 1 else (0.25 if not ps else 0.0)

        thin = self._evidence_thin_factor(research)
        effective_disagreement = disagreement + thin * 0.20

        ttc = _time_to_close_days(question)
        horizon_boost = 0.0
        if ttc is not None and ttc > 180:
            horizon_boost = 0.15
        effective_disagreement += horizon_boost

        final_p = conservative_shrink_to_50(pooled, effective_disagreement, self._disagreement_shrink_k)

        reasoning = clean_indents(f"""
        Yrambot (gpt-5.2, self-consistency n={len(ps)}):

        Research:
        {research}

        Samples:
        {", ".join(f"{p:.1%}" for p in ps) if ps else "None"}

        Aggregation:
        - log-odds pooled={pooled:.1%}
        - within-sample range={disagreement:.3f}
        - thin_evidence={thin:.2f}, horizon_boost={horizon_boost:.2f}
        - shrink-to-50 k={self._disagreement_shrink_k:.2f} -> {final_p:.1%}
        """).strip()

        return ReasonedPrediction(prediction_value=final_p, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        samples = await self._run_model_forecast_samples(question, research)
        pols = [s for s in samples if isinstance(s, PredictedOptionList)]

        avg = {opt: 0.0 for opt in question.options}
        if not pols:
            uniform = 1.0 / max(1, len(question.options))
            avg = {opt: uniform for opt in question.options}
        else:
            for pol in pols:
                m = {o.option_name: float(o.probability) for o in pol.predicted_options}
                for opt in question.options:
                    avg[opt] += max(0.0, float(m.get(opt, 0.0)))
            for opt in question.options:
                avg[opt] /= len(pols)
            s = sum(avg.values())
            if s > 0:
                avg = {k: v / s for k, v in avg.items()}
            else:
                uniform = 1.0 / max(1, len(question.options))
                avg = {opt: uniform for opt in question.options}

        thin = self._evidence_thin_factor(research)
        temp = self._mc_temperature + thin * 0.15
        final_probs = temperature_transform(avg, temperature=temp)

        final_options = [PredictedOption(option_name=opt, probability=final_probs.get(opt, 0.0)) for opt in question.options]
        prediction_list = PredictedOptionList(predicted_options=final_options)

        reasoning = clean_indents(f"""
        Yrambot (gpt-5.2, self-consistency n={len(pols)}):

        Research:
        {research}

        Aggregation:
        - avg across samples
        - temperature smoothing T={temp:.2f}
        """).strip()

        return ReasonedPrediction(prediction_value=prediction_list, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        samples = await self._run_model_forecast_samples(question, research)
        raw_lists = [s for s in samples if isinstance(s, list)]

        if not raw_lists:
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
            return enforce_monotonic_percentiles(out)

        proc: list[list[Percentile]] = []
        for lst in raw_lists:
            try:
                p = process(cast(list[Percentile], lst))
                if len(p) >= 2:
                    proc.append(p)
            except Exception:
                continue

        if not proc:
            return self._numeric_fallback(question, research)

        grid = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

        def interp(ps: list[Percentile], q: float) -> float:
            xs = [p.percentile for p in ps]
            ys = [p.value for p in ps]
            if q <= xs[0]:
                return ys[0]
            if q >= xs[-1]:
                return ys[-1]
            for i in range(1, len(xs)):
                if q <= xs[i]:
                    x0, x1 = xs[i - 1], xs[i]
                    y0, y1 = ys[i - 1], ys[i]
                    t = 0.0 if x1 == x0 else (q - x0) / (x1 - x0)
                    return y0 + t * (y1 - y0)
            return ys[-1]

        pooled: list[Percentile] = []
        for q in grid:
            vals = [interp(p, q) for p in proc]
            avg_val = clamp_to_bounds(sum(vals) / len(vals), question)
            pooled.append(Percentile(percentile=q, value=avg_val))

        pooled = enforce_monotonic_percentiles(pooled)

        thin = self._evidence_thin_factor(research)
        if thin > 0:
            widen = 1.0 + 0.25 * thin
            p10 = next((p.value for p in pooled if abs(p.percentile - 0.1) < 1e-9), pooled[1].value)
            p90 = next((p.value for p in pooled if abs(p.percentile - 0.9) < 1e-9), pooled[-2].value)
            mid = next((p.value for p in pooled if abs(p.percentile - 0.5) < 1e-9), pooled[len(pooled) // 2].value)
            lo = clamp_to_bounds(mid - widen * (mid - p10), question)
            hi = clamp_to_bounds(mid + widen * (p90 - mid), question)
            for i, pp in enumerate(pooled):
                if abs(pp.percentile - 0.1) < 1e-9:
                    pooled[i] = Percentile(percentile=pp.percentile, value=lo)
                if abs(pp.percentile - 0.9) < 1e-9:
                    pooled[i] = Percentile(percentile=pp.percentile, value=hi)
            pooled = enforce_monotonic_percentiles(pooled)

        dist = NumericDistribution.from_question(pooled, question)

        reasoning = clean_indents(f"""
        Yrambot (gpt-5.2, self-consistency n={len(proc)}):

        Research:
        {research}

        Numeric:
        - interpolate to fixed grid, average, enforce monotonic
        - thin_evidence={thin:.2f}
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
        Yrambot fallback numeric:

        Research:
        {research}

        Fallback: 10/50/90 from bounds.
        """).strip()
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

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
        reports3 = await bot.forecast_on_tournament("market-pulse-26q1", return_exceptions=True)
        forecast_reports = reports1 + reports2 + reports3
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
            logger.exception("Yrambot run failed: %s", e)
        elapsed = (now_utc() - start).total_seconds()
        sleep_for = max(5.0, float(interval_seconds) - elapsed)
        await asyncio.sleep(sleep_for)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - Yrambot - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Yrambot forecasting system (single-model, Tavily budgeted)")
    parser.add_argument("--mode", type=str, choices=["tournament", "test_questions"], default="tournament")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=180)
    args = parser.parse_args()
    asyncio.run(main(args.mode, args.loop, args.interval_seconds))
