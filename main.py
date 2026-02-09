import argparse
import asyncio
import json
import logging
import os
import random
import re
import time
from datetime import datetime
from typing import Optional, Any, Dict, Tuple, List

import numpy as np
from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
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

from tavily import TavilyClient

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - Yrambot - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Yrambot")

RESEARCH_TIMEOUT_S = float(os.getenv("RESEARCH_TIMEOUT_S", "25"))
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "70"))

MAX_CONCURRENT_QUESTIONS = int(os.getenv("MAX_CONCURRENT_QUESTIONS", "1"))
PUBLISH_SLEEP_S = float(os.getenv("PUBLISH_SLEEP_S", "3.0"))
TOURNAMENT_SLEEP_S = float(os.getenv("TOURNAMENT_SLEEP_S", "8.0"))
RETRY_MAX = int(os.getenv("RETRY_MAX", "6"))
RETRY_BASE_S = float(os.getenv("RETRY_BASE_S", "2.0"))
RETRY_MAX_S = float(os.getenv("RETRY_MAX_S", "60.0"))

EXTREMIZE_THRESHOLD = float(os.getenv("EXTREMIZE_THRESHOLD", "0.60"))
EXTREMIZE_ALPHA = float(os.getenv("EXTREMIZE_ALPHA", "1.35"))
EXTREMIZE_DISPERSION_STD_MAX = float(os.getenv("EXTREMIZE_DISPERSION_STD_MAX", "0.06"))

CROWD_BLEND_WEAK = float(os.getenv("CROWD_BLEND_WEAK", "0.45"))
CROWD_BLEND_MIXED = float(os.getenv("CROWD_BLEND_MIXED", "0.65"))
CROWD_BLEND_STRONG = float(os.getenv("CROWD_BLEND_STRONG", "0.80"))

MIN_P = float(os.getenv("MIN_P", "0.01"))
MAX_P = float(os.getenv("MAX_P", "0.99"))

REQUIRE_RESEARCH = os.getenv("REQUIRE_RESEARCH", "true").lower() in ("1", "true", "yes")

CALIBRATION_LOG_FILE = "forecasting_calibration_log.jsonl"

TAVILY_MAX_CALLS_PER_QUESTION = int(os.getenv("TAVILY_MAX_CALLS_PER_QUESTION", "2"))
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "6"))


def extract_question_id(question: MetaculusQuestion) -> str:
    for attr in ("id", "question_id", "questionId"):
        try:
            qid = getattr(question, attr, None)
            if isinstance(qid, (int, str)) and str(qid).isdigit():
                return str(qid)
        except Exception:
            pass
    for attr in ("url", "page_url", "question_url", "link"):
        try:
            url = str(getattr(question, attr, "") or "")
            m = re.search(r"/questions/(\d+)(?:/|$)", url)
            if m:
                return m.group(1)
        except Exception:
            pass
    try:
        s = str(question)
        m = re.search(r"/questions/(\d+)(?:/|$)", s)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "unknown"


def safe_community_prediction(question: MetaculusQuestion) -> Optional[float]:
    try:
        pred = getattr(question, "community_prediction", None)
        if pred is not None and isinstance(pred, (int, float)):
            return float(pred)
        pred = getattr(question, "prediction", None)
        if pred is not None and isinstance(pred, (int, float)):
            return float(pred)
    except Exception as e:
        logger.warning(f"Failed to get community prediction for Q{extract_question_id(question)}: {e}")
    return None


def clamp01(p: float) -> float:
    return float(max(MIN_P, min(MAX_P, float(p))))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def logit(p: float) -> float:
    p = clamp01(p)
    return float(np.log(p / (1.0 - p)))


def extremize_probability(p: float, alpha: float) -> float:
    return clamp01(sigmoid(alpha * logit(p)))


def should_extremize(p: float, threshold: float = EXTREMIZE_THRESHOLD) -> bool:
    return (p >= threshold) or (p <= (1.0 - threshold))


def is_meaningful_research_text(txt: str) -> bool:
    if not txt:
        return False
    low = txt.lower()
    if "failed:" in low or "error:" in low or "timeout" in low:
        return False
    return len(txt.strip()) > 160


def is_research_sufficient(research_by_source: Dict[str, str]) -> bool:
    return any(is_meaningful_research_text(v) for v in (research_by_source or {}).values())


def interpolate_missing_percentiles(reported: list[Percentile], target_percentiles: list[float]) -> list[Percentile]:
    if not reported:
        return [Percentile(percentile=p, value=0.0) for p in target_percentiles]
    sorted_rep = sorted(reported, key=lambda x: x.percentile)
    xs = [float(p.percentile) for p in sorted_rep]
    ys = [float(p.value) for p in sorted_rep]
    out: List[Percentile] = []
    for tp in target_percentiles:
        if tp in xs:
            val = ys[xs.index(tp)]
        else:
            from bisect import bisect_left
            i = bisect_left(xs, tp)
            if i == 0:
                val = ys[0]
            elif i == len(xs):
                val = ys[-1]
            else:
                x0, x1 = xs[i - 1], xs[i]
                y0, y1 = ys[i - 1], ys[i]
                val = y0 + (y1 - y0) * (tp - x0) / (x1 - x0) if x1 != x0 else y0
        out.append(Percentile(percentile=float(tp), value=float(val)))
    return out


def enforce_numeric_constraints(percentiles: list[Percentile], question: NumericQuestion) -> list[Percentile]:
    lower = -np.inf if getattr(question, "open_lower_bound", False) else getattr(question, "lower_bound", None)
    upper = np.inf if getattr(question, "open_upper_bound", False) else getattr(question, "upper_bound", None)
    if lower is None:
        lower = getattr(question, "nominal_lower_bound", None)
    if upper is None:
        upper = getattr(question, "nominal_upper_bound", None)
    if lower is None:
        lower = -np.inf
    if upper is None:
        upper = np.inf
    bounded = []
    for p in percentiles:
        v = float(p.value)
        v = max(lower, min(upper, v))
        bounded.append(Percentile(percentile=float(p.percentile), value=float(v)))
    srt = sorted(bounded, key=lambda x: x.percentile)
    vals = [p.value for p in srt]
    for i in range(1, len(vals)):
        if vals[i] < vals[i - 1]:
            vals[i] = vals[i - 1]
    return [Percentile(percentile=srt[i].percentile, value=float(vals[i])) for i in range(len(vals))]


def derive_numeric_fallback_bounds(question: NumericQuestion, anchor: Optional[float]) -> Tuple[float, float]:
    lb = getattr(question, "lower_bound", None)
    ub = getattr(question, "upper_bound", None)
    if getattr(question, "open_lower_bound", False):
        lb = None
    if getattr(question, "open_upper_bound", False):
        ub = None
    if lb is None:
        lb = getattr(question, "nominal_lower_bound", None)
    if ub is None:
        ub = getattr(question, "nominal_upper_bound", None)
    if lb is not None and ub is not None and float(ub) > float(lb):
        return float(lb), float(ub)
    if isinstance(anchor, (int, float)):
        a = float(anchor)
        if a > 0:
            return a * 0.25, a * 3.0
        return a - 1.0, a + 1.0
    return -1e9, 1e9


def log_forecast_for_calibration(
    question: MetaculusQuestion,
    prediction_value: Any,
    reasoning: str,
    model_ids: list[str],
    research_used: bool,
    searchers_used: list[str],
):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question_id": extract_question_id(question),
        "question_type": question.__class__.__name__,
        "question_text": getattr(question, "question_text", ""),
        "resolution_date": getattr(question, "resolution_date", None),
        "community_prediction": safe_community_prediction(question),
        "prediction_value": prediction_value,
        "models_used": model_ids,
        "research_used": research_used,
        "searchers_used": searchers_used,
        "reasoning_snippet": reasoning[:1500],
    }
    try:
        with open(CALIBRATION_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log calibration data: {e}")


async def with_timeout(coro, seconds: float, label: str) -> str:
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        return f"{label} timeout after {seconds}s"
    except Exception as e:
        return f"{label} error: {e}"


def backoff_sleep(attempt: int) -> None:
    base = min(RETRY_MAX_S, RETRY_BASE_S * (2 ** attempt))
    jitter = random.uniform(0.0, base * 0.25)
    time.sleep(base + jitter)


def build_reasoning_only(
    question: MetaculusQuestion,
    forecast_text: str,
    base_rate_text: str,
    how_text: str,
    searchers_used: List[str],
    models_used: List[str],
) -> str:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    searchers = ", ".join(searchers_used) if searchers_used else "None"
    models = ", ".join(models_used) if models_used else "Unknown"
    return clean_indents(f"""
    Date (UTC): {today}
    Forecast: {forecast_text}
    Base rate / anchor: {base_rate_text}

    How the forecast was arrived at:
    {how_text}

    Search: {searchers}
    Model: {models}
    """).strip()


_ARITH_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*$")


def _safe_eval(expr: str) -> Optional[float]:
    m = _ARITH_RE.match(expr.strip())
    if not m:
        return None
    a, op, b = float(m[1]), m[2], float(m[3])
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if abs(b) < 1e-12:
            return None
        return a / b
    return None


def sanitize_numeric_json(text: str) -> str:
    def repl(m: re.Match) -> str:
        raw = m.group(2).strip()
        v = _safe_eval(raw)
        return m.group(1) + (str(v) if v is not None else raw)
    return re.sub(r'("percentile"\s*:\s*)([^,\]\}\n]+)', repl, text)


class Yrambot(ForecastBot):
    _max_concurrent_questions = MAX_CONCURRENT_QUESTIONS
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5.2",
            "parser": "openrouter/openai/gpt-4.1-mini",
            "researcher": "openrouter/openai/gpt-5.2",
            "summarizer": "openrouter/openai/gpt-4.1-mini",
            "llm_web_search": "openrouter/openai/gpt-5.2",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
        self._research_meta: Dict[str, Dict[str, Any]] = {}
        self._tavily_calls: Dict[str, int] = {}

    def _metaculus_context_block(self, question: MetaculusQuestion) -> str:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        qtxt = (getattr(question, "question_text", "") or "").strip()
        rc = (getattr(question, "resolution_criteria", "") or "").strip()
        bg = (getattr(question, "background_info", "") or "").strip()
        url = (getattr(question, "url", "") or getattr(question, "page_url", "") or "").strip()
        return clean_indents(f"""
        --- SOURCE METACULUS ({today}) ---
        Question: {qtxt}
        URL: {url or "N/A"}
        Resolution criteria: {rc or "N/A"}
        Background: {bg or "N/A"}
        """).strip()

    def call_tavily(self, query: str) -> str:
        if not self.tavily_client or not getattr(self.tavily_client, "api_key", None):
            return ""
        try:
            response = self.tavily_client.search(query=query, search_depth="advanced", max_results=TAVILY_MAX_RESULTS)
            results = response.get("results", []) or []
            lines: List[str] = []
            for c in results[:TAVILY_MAX_RESULTS]:
                title = (c.get("title") or "").strip()
                content = (c.get("content") or "").strip()
                url = (c.get("url") or "").strip()
                published = (c.get("published_date") or c.get("publishedDate") or "").strip()
                if title or content:
                    snippet = content[:520].strip()
                    lines.append(
                        f"- Title: {title or 'N/A'}\n"
                        f"  Published: {published or 'N/A'}\n"
                        f"  Snippet: {snippet or 'N/A'}\n"
                        f"  Source: {url or 'N/A'}"
                    )
            return "\n".join(lines).strip()
        except Exception as e:
            return f"Tavily failed: {e}"

    async def _expand_queries_with_llm(self, question_text: str) -> List[str]:
        researcher_llm = self.get_llm("researcher", "llm")
        prompt = clean_indents(f"""
        Generate 6 concise web search queries for this forecasting question.
        Mix: entities, resolution-criteria terms, and "latest update" phrasing.
        Output JSON ONLY: {{"queries":["...","..."]}}

        Question: {question_text}
        """).strip()
        raw = await with_timeout(researcher_llm.invoke(prompt), 35, "query_expand")
        try:
            data = json.loads(raw)
            qs = [str(x).strip() for x in data.get("queries", []) if str(x).strip()]
            merged = [question_text.strip()] + qs
            seen = set()
            out = []
            for q in merged:
                k = q.lower()
                if k not in seen:
                    seen.add(k)
                    out.append(q)
            return out[:6]
        except Exception:
            return [question_text.strip()]

    async def _llm_web_search_support(self, question_text: str, tavily_text: str) -> str:
        llm = self.get_llm("llm_web_search", "llm")
        prompt = clean_indents(f"""
        You are a web-research assistant for Yrambot. You MUST NOT invent sources.
        Your job is to improve Tavily research quality by proposing:
        - 3 high-impact follow-up queries
        - 3 key entities/keywords to include
        - 3 "must-check" source types (e.g., gov/central bank filings, earnings calls, agency reports)
        Output JSON ONLY:
        {{
          "followup_queries": ["...","...","..."],
          "keywords": ["...","...","..."],
          "source_types": ["...","...","..."],
          "quality_gap": "one sentence"
        }}

        Forecasting question:
        {question_text}

        Current Tavily results (may be incomplete):
        {tavily_text}
        """).strip()
        raw = await with_timeout(llm.invoke(prompt), 35, "llm_web_search_support")
        try:
            data = json.loads(raw)
            qs = [str(x).strip() for x in data.get("followup_queries", []) if str(x).strip()]
            if qs:
                return json.dumps(data)
        except Exception:
            pass
        return ""

    async def _run_research_impl(self, question: MetaculusQuestion) -> Tuple[str, Dict[str, str], List[str]]:
        async with self._concurrency_limiter:
            qtxt = (getattr(question, "question_text", "") or "").strip()
            if not qtxt:
                raise RuntimeError("Missing question_text; refusing to forecast.")
            qid = extract_question_id(question)
            self._tavily_calls.setdefault(qid, 0)

            metaculus_block = self._metaculus_context_block(question)

            expanded_queries = await self._expand_queries_with_llm(qtxt)

            tav_parts: List[str] = []

            if not self.tavily_client:
                research_by_source = {"metaculus": metaculus_block, "tavily": ""}
                if REQUIRE_RESEARCH and not is_research_sufficient(research_by_source):
                    raise RuntimeError(f"Insufficient research for Q{qid}; refusing to forecast.")
                searchers_used = ["metaculus"]
                synthesized = await self._synthesize_research(question, research_by_source, searchers_used)
                return synthesized, research_by_source, searchers_used

            query0 = expanded_queries[0] if expanded_queries else qtxt
            if self._tavily_calls[qid] < TAVILY_MAX_CALLS_PER_QUESTION:
                tav0 = await with_timeout(asyncio.to_thread(self.call_tavily, query0), RESEARCH_TIMEOUT_S, "tavily_0")
                self._tavily_calls[qid] += 1
                if is_meaningful_research_text(tav0):
                    tav_parts.append(f"## Query: {query0}\n{tav0}")

            tav_all = "\n\n".join(tav_parts).strip()

            if not is_meaningful_research_text(tav_all) and self._tavily_calls[qid] < TAVILY_MAX_CALLS_PER_QUESTION:
                support_json = await self._llm_web_search_support(qtxt, tav_all or "(empty)")
                followups: List[str] = []
                try:
                    data = json.loads(support_json) if support_json else {}
                    followups = [str(x).strip() for x in data.get("followup_queries", []) if str(x).strip()]
                except Exception:
                    followups = []

                query1 = followups[0] if followups else (expanded_queries[1] if len(expanded_queries) > 1 else qtxt)
                tav1 = await with_timeout(asyncio.to_thread(self.call_tavily, query1), RESEARCH_TIMEOUT_S, "tavily_1")
                self._tavily_calls[qid] += 1
                if is_meaningful_research_text(tav1):
                    tav_parts.append(f"## Query: {query1}\n{tav1}")
                tav_all = "\n\n".join(tav_parts).strip()

            research_by_source = {
                "metaculus": metaculus_block,
                "tavily": tav_all,
            }

            searchers_used = ["metaculus"] + (["tavily"] if is_meaningful_research_text(tav_all) else [])

            if REQUIRE_RESEARCH and not is_research_sufficient(research_by_source):
                raise RuntimeError(f"Insufficient research for Q{qid}; refusing to forecast.")

            synthesized = await self._synthesize_research(question, research_by_source, searchers_used)
            if REQUIRE_RESEARCH and not is_meaningful_research_text(synthesized):
                raise RuntimeError(f"Insufficient synthesized research for Q{qid}; refusing to forecast.")

            return synthesized, research_by_source, searchers_used

    async def _synthesize_research(
        self,
        question: MetaculusQuestion,
        research_by_source: Dict[str, str],
        searchers_used: List[str],
    ) -> str:
        researcher_llm = self.get_llm("researcher", "llm")
        qtxt = (getattr(question, "question_text", "") or "").strip()

        met = (research_by_source.get("metaculus") or "").strip()
        tav = (research_by_source.get("tavily") or "").strip()

        raw_block = clean_indents(f"""
        {met}

        --- SOURCE TAVILY ---
        {tav if tav else "N/A"}
        """).strip()

        prompt = clean_indents(f"""
        Summarize evidence for forecasting (Yrambot).
        Requirements:
        - Prefer RECENT items; include dates and numbers where available.
        - Use only information present in the provided sources block.
        - End each bullet with [METACULUS] or [TAVILY] (or both).
        - Include:
          - Base rate / reference class
          - Key updates (2-6)
          - Uncertainties
          - Signposts (what would change the forecast)

        Question:
        {qtxt}

        Sources block:
        {raw_block}

        Output plain text (<=2400 chars).
        """).strip()

        synthesized = await with_timeout(researcher_llm.invoke(prompt), LLM_TIMEOUT_S, "research_synthesis")
        return (synthesized or "").strip()

    async def run_research(self, question: MetaculusQuestion) -> str:
        synthesized, research_by_source, searchers_used = await self._run_research_impl(question)
        qid = extract_question_id(question)
        self._research_meta[qid] = {
            "synthesized": synthesized,
            "raw": research_by_source,
            "searchers_used": searchers_used,
        }
        logger.info(
            f"Q{qid} class={type(question).__name__} "
            f"bounds=({getattr(question,'lower_bound',None)},{getattr(question,'upper_bound',None)}) "
            f"open_bounds=({getattr(question,'open_lower_bound',None)},{getattr(question,'open_upper_bound',None)}) "
            f"searchers_used={searchers_used}"
        )
        return synthesized

    async def _single_forecast(self, question, research: str, model_override: str = None):
        qid = extract_question_id(question)
        if REQUIRE_RESEARCH and (not research or len(research.strip()) < 120):
            raise RuntimeError(f"Missing/insufficient research for Q{qid}; refusing to forecast.")

        llm = GeneralLlm(model=model_override) if model_override else self.get_llm("default", "llm")
        parser_llm = self.get_llm("parser", "llm")
        summarizer_llm = self.get_llm("summarizer", "llm")
        today_utc = datetime.utcnow().strftime("%Y-%m-%d")

        base_rate = safe_community_prediction(question)
        base_rate_str = (
            f"Community anchor: {base_rate:.4f} (decimal)"
            if isinstance(base_rate, (int, float)) and isinstance(question, BinaryQuestion)
            else (f"Community anchor (scalar): {base_rate:,.6g}" if isinstance(base_rate, (int, float)) else "No reliable community anchor.")
        )

        gj = clean_indents("""
        Rules:
        - Outside view first (base rate), then inside view updates from evidence.
        - Weight evidence by recency + credibility; discount stale/noisy.
        - Avoid unjustified certainty; extremes need strong evidence.
        """).strip()

        if isinstance(question, BinaryQuestion):
            prompt = clean_indents(f"""
            {gj}

            Question: {question.question_text}
            Resolution criteria: {question.resolution_criteria}

            {base_rate_str}

            Research summary:
            {research}

            Today (UTC): {today_utc}

            Output JSON ONLY: {{"prediction_in_decimal": number}}
            """).strip()
            raw = await with_timeout(llm.invoke(prompt), LLM_TIMEOUT_S, "binary_llm")
            pred: BinaryPrediction = await structure_output(raw, BinaryPrediction, model=parser_llm)
            return clamp01(float(pred.prediction_in_decimal)), str(raw)

        if isinstance(question, MultipleChoiceQuestion):
            prompt = clean_indents(f"""
            {gj}

            Question: {question.question_text}
            Options (exact): {question.options}
            Resolution criteria: {question.resolution_criteria}

            {base_rate_str}

            Research summary:
            {research}

            Today (UTC): {today_utc}

            Output JSON ONLY:
            {{
              "predicted_options": [
                {{"option_name":"<exact option>","probability": number}},
                ...
              ]
            }}
            Constraints: include all options once; probabilities sum to 1.
            """).strip()
            raw = await with_timeout(llm.invoke(prompt), LLM_TIMEOUT_S, "mc_llm")
            result = await structure_output(
                raw,
                PredictedOptionList,
                model=parser_llm,
                additional_instructions=f"Options must be exactly: {question.options}"
            )
            probs = {o.option_name: max(0.0, float(o.probability)) for o in result.predicted_options}
            for opt in question.options:
                probs.setdefault(opt, 0.0)
            s = sum(probs.values()) or 1.0
            probs = {k: v / s for k, v in probs.items()}
            final = [PredictedOption(option_name=opt, probability=float(probs[opt])) for opt in question.options]
            return PredictedOptionList(predicted_options=final), str(raw)

        if isinstance(question, NumericQuestion):
            lower_ref = getattr(question, "lower_bound", None) or getattr(question, "nominal_lower_bound", None)
            upper_ref = getattr(question, "upper_bound", None) or getattr(question, "nominal_upper_bound", None)

            prompt = clean_indents(f"""
            {gj}

            Question: {question.question_text}
            Units: {question.unit_of_measure or 'Infer from context'}
            Resolution criteria: {question.resolution_criteria}
            Bounds (may be open): lower={lower_ref}, upper={upper_ref}

            {base_rate_str}

            Research summary:
            {research}

            Today (UTC): {today_utc}

            NUMERIC OUTPUT RULES:
            - Output MUST be valid JSON ONLY (no backticks, no commentary).
            - Use ONLY numeric literals (no arithmetic like 0.5-0.25).
            - Percentile in [0,1], strictly increasing.
            - Include these percentiles exactly: 0.1, 0.2, 0.4, 0.6, 0.8, 0.9.

            Output JSON ONLY:
            [
              {{"percentile":0.1,"value":...}},
              {{"percentile":0.2,"value":...}},
              {{"percentile":0.4,"value":...}},
              {{"percentile":0.6,"value":...}},
              {{"percentile":0.8,"value":...}},
              {{"percentile":0.9,"value":...}}
            ]
            """).strip()

            raw = await with_timeout(llm.invoke(prompt), LLM_TIMEOUT_S, "num_llm")

            try:
                percentile_list: list[Percentile] = await structure_output(raw, list[Percentile], model=parser_llm)
            except Exception:
                fixed = sanitize_numeric_json(str(raw))
                repair_prompt = clean_indents(f"""
                Convert the following into valid JSON ONLY for type list[Percentile].
                Rules:
                - Only numeric literals
                - percentile in [0,1]
                - strictly increasing percentiles
                - no arithmetic expressions
                - output only the JSON list

                Text:
                {fixed}
                """).strip()
                repaired = await with_timeout(summarizer_llm.invoke(repair_prompt), LLM_TIMEOUT_S, "num_repair")
                percentile_list = await structure_output(repaired, list[Percentile], model=parser_llm)

            target_ps = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
            interpolated = interpolate_missing_percentiles(percentile_list, target_ps)
            validated = enforce_numeric_constraints(interpolated, question)

            vals = [p.value for p in validated]
            if len(set([round(v, 12) for v in vals])) == 1:
                lb, ub = derive_numeric_fallback_bounds(question, base_rate)
                mid = float(vals[0])
                width = (ub - lb) * 0.08 if np.isfinite(ub - lb) else max(1.0, abs(mid) * 0.25)
                widened = [
                    Percentile(0.1, mid - 1.2 * width),
                    Percentile(0.2, mid - 0.7 * width),
                    Percentile(0.4, mid - 0.2 * width),
                    Percentile(0.6, mid + 0.2 * width),
                    Percentile(0.8, mid + 0.7 * width),
                    Percentile(0.9, mid + 1.2 * width),
                ]
                validated = enforce_numeric_constraints(widened, question)

            return NumericDistribution.from_question(validated, question), str(raw)

        raise ValueError(f"Unsupported question type: {type(question)}")

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        model = "openrouter/openai/gpt-5.2"
        qid = extract_question_id(question)
        base = safe_community_prediction(question)
        base_rate_text = f"{base:.1%}" if isinstance(base, (int, float)) else "None"

        try:
            p, _ = await self._single_forecast(question, research, model_override=model)
            raw_ps = [float(p)]
        except Exception as e:
            fallback = clamp01(float(base) if isinstance(base, (int, float)) else 0.5)
            raw_ps = [fallback]
            logger.warning(f"Binary fallback Q{qid}: {e}")

        p_agg = float(raw_ps[0])
        p_std = 0.0
        agree = True

        p_final = p_agg
        if agree and should_extremize(p_agg, EXTREMIZE_THRESHOLD) and p_std <= EXTREMIZE_DISPERSION_STD_MAX:
            p_final = extremize_probability(p_agg, alpha=EXTREMIZE_ALPHA)

        if isinstance(base, (int, float)):
            w = CROWD_BLEND_MIXED
            p_final = clamp01(w * p_final + (1.0 - w) * float(base))

        meta = self._research_meta.get(qid, {})
        searchers_used = meta.get("searchers_used", []) if isinstance(meta.get("searchers_used", None), list) else []

        how = clean_indents(f"""
        - Used synthesized research summary (dates + key signals) to update from base rate.
        - Single-model forecast (gpt-5.2) with calibration guardrails:
          - optional extremization when sufficiently away from 50%
          - optional blend with Metaculus community anchor when available.
        """).strip()

        reasoning = build_reasoning_only(
            question,
            forecast_text=f"{p_final:.1%}",
            base_rate_text=base_rate_text,
            how_text=how,
            searchers_used=searchers_used,
            models_used=[model],
        )
        log_forecast_for_calibration(question, p_final, reasoning, [model], True, searchers_used)
        time.sleep(PUBLISH_SLEEP_S)
        return ReasonedPrediction(prediction_value=p_final, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        model = "openrouter/openai/gpt-5.2"
        qid = extract_question_id(question)

        try:
            pred, _ = await self._single_forecast(question, research, model_override=model)
            out = pred
        except Exception as e:
            n = len(question.options)
            out = PredictedOptionList(predicted_options=[PredictedOption(option_name=opt, probability=1.0 / n) for opt in question.options])
            logger.warning(f"MC fallback Q{qid}: {e}")

        meta = self._research_meta.get(qid, {})
        searchers_used = meta.get("searchers_used", []) if isinstance(meta.get("searchers_used", None), list) else []

        how = clean_indents(f"""
        - Used synthesized research summary to allocate probability mass across options.
        - Enforced normalization (sum to 1) and coverage (all options included once).
        - Single-model forecast (gpt-5.2).
        """).strip()

        reasoning = build_reasoning_only(
            question,
            forecast_text=", ".join([f"{x.option_name}: {x.probability:.1%}" for x in out.predicted_options]),
            base_rate_text="Metaculus community chart (qualitative anchor)",
            how_text=how,
            searchers_used=searchers_used,
            models_used=[model],
        )
        log_forecast_for_calibration(question, [x.probability for x in out.predicted_options], reasoning, [model], True, searchers_used)
        time.sleep(PUBLISH_SLEEP_S)
        return ReasonedPrediction(prediction_value=out, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        model = "openrouter/openai/gpt-5.2"
        qid = extract_question_id(question)
        base = safe_community_prediction(question)
        base_rate_text = f"{base:,.4g}" if isinstance(base, (int, float)) else "None"

        try:
            dist, _ = await self._single_forecast(question, research, model_override=model)
        except Exception as e:
            lb, ub = derive_numeric_fallback_bounds(question, base)
            center = float(base) if isinstance(base, (int, float)) else (lb + ub) / 2.0
            width = (ub - lb) * 0.30
            target_ps = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
            vals = [
                center - 0.9 * width,
                center - 0.5 * width,
                center - 0.15 * width,
                center + 0.15 * width,
                center + 0.5 * width,
                center + 0.9 * width,
            ]
            vals = [max(lb, min(ub, v)) for v in vals]
            validated = enforce_numeric_constraints([Percentile(p, v) for p, v in zip(target_ps, vals)], question)
            dist = NumericDistribution.from_question(validated, question)
            logger.warning(f"Numeric fallback Q{qid}: {e}")

        declared = list(getattr(dist, "declared_percentiles", [])) or []
        target_ps = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        interpolated = interpolate_missing_percentiles(declared, target_ps)
        validated = enforce_numeric_constraints(interpolated, question)
        dist = NumericDistribution.from_question(validated, question)

        meta = self._research_meta.get(qid, {})
        searchers_used = meta.get("searchers_used", []) if isinstance(meta.get("searchers_used", None), list) else []

        how = clean_indents(f"""
        - Produced a percentile forecast from synthesized research, anchored by any available community scalar.
        - Parser-safe numeric output with repair step (eliminates arithmetic in JSON).
        - Enforced bounds and monotonic percentiles.
        """).strip()

        reasoning = build_reasoning_only(
            question,
            forecast_text=", ".join([f"p{int(p.percentile*100)}={p.value:,.6g}" for p in validated]),
            base_rate_text=base_rate_text,
            how_text=how,
            searchers_used=searchers_used,
            models_used=[model],
        )
        log_forecast_for_calibration(question, [p.value for p in validated], reasoning, [model], True, searchers_used)
        time.sleep(PUBLISH_SLEEP_S)
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Yrambot.")
    parser.add_argument(
        "--tournament-ids",
        nargs="+",
        type=str,
        default=[
            "32916",
            "minibench",
            "market-pulse-26q1",
            MetaculusApi.CURRENT_MINIBENCH_ID,
        ],
    )
    args = parser.parse_args()

    bot = Yrambot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
    )

    try:
        all_reports = []
        for tid in args.tournament_ids:
            logger.info(f"Forecasting on tournament: {tid}")
            for attempt in range(RETRY_MAX):
                try:
                    reports = asyncio.run(bot.forecast_on_tournament(tid, return_exceptions=True))
                    all_reports.extend(reports)
                    break
                except Exception as e:
                    msg = str(e).lower()
                    if "too many requests" in msg or "cloudflare" in msg or "1015" in msg or "429" in msg:
                        logger.error(f"Rate-limited on tournament {tid} (attempt {attempt+1}/{RETRY_MAX}): {e}")
                        backoff_sleep(attempt)
                        continue
                    raise
            time.sleep(TOURNAMENT_SLEEP_S)

        bot.log_report_summary(all_reports)
        logger.info(f"Run completed. Calibration logs saved to {CALIBRATION_LOG_FILE}")
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
