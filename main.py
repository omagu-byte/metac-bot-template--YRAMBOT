import argparse
import asyncio
import json
import logging
import math
import os
import random
import re
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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

import dotenv
dotenv.load_dotenv()

_GPT_MODEL    = "openrouter/openai/gpt-5.4"
_SONNET_MODEL = "openrouter/anthropic/claude-sonnet-4-6"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - Yrambot - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Yrambot")

TAVILY_API_KEY             = os.getenv("TAVILY_API_KEY")
RESEARCH_TIMEOUT_S         = float(os.getenv("RESEARCH_TIMEOUT_S", "25"))
LLM_TIMEOUT_S              = float(os.getenv("LLM_TIMEOUT_S", "70"))
MAX_CONCURRENT_QUESTIONS   = int(os.getenv("MAX_CONCURRENT_QUESTIONS", "1"))
PUBLISH_SLEEP_S            = float(os.getenv("PUBLISH_SLEEP_S", "3.0"))
TOURNAMENT_SLEEP_S         = float(os.getenv("TOURNAMENT_SLEEP_S", "8.0"))
RETRY_MAX                  = int(os.getenv("RETRY_MAX", "6"))
RETRY_BASE_S               = float(os.getenv("RETRY_BASE_S", "2.0"))
RETRY_MAX_S                = float(os.getenv("RETRY_MAX_S", "60.0"))
EXTREMIZE_ENABLED          = os.getenv("EXTREMIZE_ENABLED", "true").lower() in ("1", "true", "yes", "y")
EXTREMIZE_FACTOR           = float(os.getenv("EXTREMIZE_FACTOR", "1.45"))
EXTREMIZE_FLOOR            = float(os.getenv("EXTREMIZE_FLOOR", "0.02"))
EXTREMIZE_CEIL             = float(os.getenv("EXTREMIZE_CEIL", "0.98"))
MINIBENCH_EXTREMIZE_FACTOR = float(os.getenv("MINIBENCH_EXTREMIZE_FACTOR", "1.65"))
CROWD_BLEND_MIXED          = float(os.getenv("CROWD_BLEND_MIXED", "0.65"))
MIN_P                      = float(os.getenv("MIN_P", "0.01"))
MAX_P                      = float(os.getenv("MAX_P", "0.99"))
REQUIRE_RESEARCH           = os.getenv("REQUIRE_RESEARCH", "true").lower() in ("1", "true", "yes")
CALIBRATION_LOG_FILE       = "forecasting_calibration_log.jsonl"
TAVILY_MAX_RESULTS         = int(os.getenv("TAVILY_MAX_RESULTS", "6"))

MINIBENCH_IDS = {
    "minibench",
    str(getattr(MetaculusApi, "CURRENT_MINIBENCH_ID", "")),
}

DOMAINS    = ["geopolitics", "economics", "technology", "science",
              "public_health", "environment", "sports", "finance", "social", "other"]
GEO_SCOPES = ["global", "regional", "national", "local"]


@dataclass
class QuestionProfile:
    domain:                str   = "other"
    geo_scope:             str   = "global"
    geography:             str   = ""
    time_horizon_days:     int   = 365
    is_quantitative:       bool  = False
    confidence_in_profile: float = 0.0


class QuestionAnalyser:
    def __init__(self, llm: GeneralLlm):
        self._llm = llm

    async def classify(self, question: MetaculusQuestion) -> QuestionProfile:
        prompt = (
            "You are a forecasting assistant. Classify the question below and return a single "
            "JSON object — no markdown fences, no extra keys.\n\n"
            "Schema:\n"
            "{\n"
            f'  "domain": "<one of: {", ".join(DOMAINS)}>,"\n'
            f'  "geo_scope": "<one of: {", ".join(GEO_SCOPES)}>",\n'
            '  "geography": "<country or region name, or empty string if global>",\n'
            '  "time_horizon_days": <integer — estimated days until resolution>,\n'
            '  "is_quantitative": <true if the answer is a number or date, else false>,\n'
            '  "confidence_in_profile": <float 0.0 to 1.0>\n'
            "}\n\n"
            f"Question: {question.question_text}\n"
            f"Resolution criteria: {question.resolution_criteria}\n"
            f"Fine print: {getattr(question, 'fine_print', '') or 'None'}"
        )
        try:
            raw = await self._llm.invoke(prompt)
            raw = raw.strip()
            s, e = raw.find("{"), raw.rfind("}")
            if s != -1 and e != -1:
                raw = raw[s:e + 1]
            data = json.loads(raw)
            return QuestionProfile(
                domain=data.get("domain", "other"),
                geo_scope=data.get("geo_scope", "global"),
                geography=data.get("geography", ""),
                time_horizon_days=int(data.get("time_horizon_days", 365)),
                is_quantitative=bool(data.get("is_quantitative", False)),
                confidence_in_profile=float(data.get("confidence_in_profile", 0.5)),
            )
        except Exception as exc:
            logger.warning(f"[Analyser] classify failed: {exc}")
            return QuestionProfile()


class ModellingStrategy:
    @staticmethod
    def select(profile: QuestionProfile, question_type: str = "binary") -> str:
        if question_type == "numeric":
            if profile.domain in ("economics", "finance"):
                return "trend"
            if profile.time_horizon_days < 90:
                return "trend"
            return "base_rate"
        if question_type == "multiple_choice":
            if profile.domain in ("geopolitics", "social"):
                return "analogical"
            if profile.time_horizon_days < 60:
                return "market_signal"
            return "base_rate"
        if profile.domain in ("economics", "finance") and profile.is_quantitative:
            return "trend"
        if profile.domain in ("geopolitics", "social"):
            return "analogical"
        if profile.time_horizon_days < 60:
            return "market_signal"
        return "base_rate"

    @staticmethod
    def get_prompt_block(strategy: str, profile: QuestionProfile) -> str:
        geo_ctx = f" with geographic focus on {profile.geography}" if profile.geography else ""
        strategies = {
            "trend": (
                f"Analytical framework — Trend Extrapolation{geo_ctx}:\n"
                "1. Identify the primary measurable variable driving this outcome.\n"
                "2. Establish its recent trajectory using the most recent 1 to 3 data points.\n"
                "3. Project that trajectory forward to the resolution date.\n"
                "4. Apply a mean-reversion discount — persistent trends are the exception, not the rule.\n"
                "5. State a realistic uncertainty band that reflects data limitations and volatility."
            ),
            "analogical": (
                f"Analytical framework — Historical Analogy{geo_ctx}:\n"
                "1. Identify 2 to 3 past situations that are structurally similar to this one.\n"
                "2. State the base rate: how often did the focal outcome occur across those cases?\n"
                "3. List key similarities and explain how they support your probability estimate.\n"
                "4. List key differences and explain how they require you to adjust from the base rate.\n"
                "5. Weight analogies by structural similarity, not surface-level resemblance."
            ),
            "market_signal": (
                f"Analytical framework — Market Signal Anchoring{geo_ctx}:\n"
                "1. Identify any current prediction-market probabilities from Metaculus, Polymarket, or Metaforecast.\n"
                "2. Treat the prevailing market signal as a strong prior.\n"
                "3. Adjust away from that signal only when you hold concrete information the market has not yet priced in.\n"
                "4. For short time horizons, weight status-quo inertia heavily — things rarely change fast."
            ),
            "base_rate": (
                f"Analytical framework — Base Rate Anchoring{geo_ctx}:\n"
                "1. Define a precise reference class: what category of events does this question belong to?\n"
                "2. State how often the focal outcome has occurred historically within that class.\n"
                "3. Anchor your initial estimate to that base rate before considering case-specific evidence.\n"
                "4. Apply inside-view adjustments only where this case has clearly distinguishing features.\n"
                "5. Limit total adjustment from the base rate to plus or minus 20 percentage points unless evidence is overwhelming."
            ),
        }
        return strategies.get(strategy, strategies["base_rate"])


class BaseSource(ABC):
    name: str = "unnamed_source"

    @abstractmethod
    async def fetch(self, query: str) -> str: ...

    def is_available(self) -> bool:
        return True


class SourceRegistry:
    def __init__(self):
        self._sources: List[BaseSource] = []

    def register(self, source: BaseSource) -> None:
        self._sources.append(source)
        logger.info(f"[SourceRegistry] Registered source: {source.name}")

    def available_sources(self) -> List[BaseSource]:
        return [s for s in self._sources if s.is_available()]

    async def fetch_all(self, query: str) -> List[str]:
        sources = self.available_sources()
        results = await asyncio.gather(*[s.fetch(query) for s in sources], return_exceptions=True)
        blocks: List[str] = []
        for src, res in zip(sources, results):
            if isinstance(res, Exception):
                blocks.append(f"[{src.name}] error: {res}")
            elif isinstance(res, str) and res.strip():
                blocks.append(f"[{src.name}]\n{res}")
        return blocks


class TavilySource(BaseSource):
    name = "tavily_web"

    def __init__(self, api_key: str, max_results: int = TAVILY_MAX_RESULTS):
        self._api_key     = api_key
        self._max_results = max_results
        self._client      = TavilyClient(api_key=api_key) if api_key else None

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def fetch(self, query: str) -> str:
        if not self._client:
            return ""
        try:
            resp    = await asyncio.to_thread(
                self._client.search,
                query=query, search_depth="advanced", max_results=self._max_results
            )
            results = resp.get("results", []) or []
            lines: List[str] = []
            for c in results[:self._max_results]:
                title     = (c.get("title") or "").strip()
                content   = (c.get("content") or "").strip()
                url       = (c.get("url") or "").strip()
                published = (c.get("published_date") or c.get("publishedDate") or "").strip()
                if title or content:
                    lines.append(
                        f"- {title or 'N/A'}\n"
                        f"  Published: {published or 'N/A'}\n"
                        f"  Snippet: {content[:520].strip() or 'N/A'}\n"
                        f"  Source: {url or 'N/A'}"
                    )
            return "\n".join(lines).strip()
        except Exception as exc:
            return f"Tavily error: {exc}"


class GptWebSearchSource(BaseSource):
    name = "gpt_knowledge_search"

    def __init__(self, llm: GeneralLlm):
        self._llm = llm

    def is_available(self) -> bool:
        return True

    async def fetch(self, query: str) -> str:
        prompt = (
            "You are a research assistant. Your task is to produce a concise, factual research brief "
            "based on your training knowledge.\n\n"
            "Requirements:\n"
            "- Include specific statistics, named institutions, and dates where known.\n"
            "- Prefix any uncertain claim with 'Uncertain:' at the start of that bullet point.\n"
            "- Do not fabricate URLs, paper titles, or publication details.\n"
            "- Focus exclusively on information relevant to forecasting the query outcome.\n"
            "- Keep the total response under 600 words.\n\n"
            f"Research query: {query}"
        )
        try:
            raw = await with_timeout(self._llm.invoke(prompt), LLM_TIMEOUT_S, "gpt_knowledge_search")
            return raw.strip() if raw and len(raw.strip()) > 60 else ""
        except Exception as exc:
            return f"GPT knowledge search error: {exc}"


@dataclass
class ValidationRecord:
    question_url:           str
    question_text:          str
    domain:                 str
    geo_scope:              str
    strategy:               str
    prediction_value:       str
    confidence_score:       float
    flagged_low_confidence: bool
    ts: float = field(default_factory=time.time)


class ForecastValidator:
    LOW_CONFIDENCE_THRESHOLD = 0.35

    def __init__(self, db_path: str = "yrambot_validation.db"):
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS forecast_ledger ("
                "question_url TEXT, question_text TEXT, domain TEXT, "
                "geo_scope TEXT, strategy TEXT, prediction_value TEXT, "
                "confidence_score REAL, flagged INTEGER, ts REAL)"
            )
            conn.commit()

    def compute_confidence(self, prediction_value: Any, profile: QuestionProfile,
                           research_length: int) -> float:
        classifier_score = profile.confidence_in_profile
        evidence_score   = min(1.0, research_length / 3000)
        signal_score     = abs(prediction_value - 0.5) * 2 if isinstance(prediction_value, float) else 0.5
        return round(
            min(1.0, max(0.0, 0.4 * classifier_score + 0.35 * evidence_score + 0.25 * signal_score)), 3
        )

    def validate(self, question: MetaculusQuestion, profile: QuestionProfile,
                 strategy: str, prediction_value: Any, research: str) -> ValidationRecord:
        confidence = self.compute_confidence(prediction_value, profile, len(research))
        flagged    = confidence < self.LOW_CONFIDENCE_THRESHOLD
        record = ValidationRecord(
            question_url=getattr(question, "page_url", ""),
            question_text=question.question_text[:300],
            domain=profile.domain,
            geo_scope=profile.geo_scope,
            strategy=strategy,
            prediction_value=str(prediction_value)[:200],
            confidence_score=confidence,
            flagged_low_confidence=flagged,
        )
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT INTO forecast_ledger "
                    "(question_url, question_text, domain, geo_scope, strategy, "
                    "prediction_value, confidence_score, flagged, ts) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (record.question_url, record.question_text, record.domain,
                     record.geo_scope, record.strategy, record.prediction_value,
                     record.confidence_score, int(record.flagged_low_confidence), record.ts)
                )
                conn.commit()
        except Exception as exc:
            logger.warning(f"[Validator] persist failed: {exc}")
        level = logging.WARNING if flagged else logging.INFO
        logger.log(level, f"[Validator] confidence={confidence:.2f} flagged={flagged} "
                          f"domain={profile.domain} strategy={strategy}")
        return record

    def summary(self) -> Dict[str, Any]:
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT domain, COUNT(*) as n, AVG(confidence_score) as avg_conf, "
                    "SUM(flagged) as n_flagged "
                    "FROM forecast_ledger GROUP BY domain ORDER BY n DESC"
                ).fetchall()
            return {
                "by_domain": [
                    {"domain": r[0], "n": r[1], "avg_confidence": round(r[2], 3), "n_flagged": r[3]}
                    for r in rows
                ]
            }
        except Exception:
            return {}


@dataclass
class ClientSpecialisation:
    domain_focus:       List[str] = field(default_factory=list)
    trusted_domains:    List[str] = field(default_factory=list)
    excluded_domains:   List[str] = field(default_factory=list)
    extra_context:      str       = ""
    calibration_target: float     = 0.15


class ResearchCache:
    def __init__(self, db_path: str = "yrambot_cache.db"):
        self._db_path = db_path
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS research_cache "
                "(url TEXT PRIMARY KEY, content TEXT NOT NULL, ts REAL NOT NULL)"
            )
            conn.commit()

    def _get_sync(self, url: str) -> Optional[str]:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT content FROM research_cache WHERE url=?", (url,)
            ).fetchone()
        return row[0] if row else None

    def _set_sync(self, url: str, content: str) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO research_cache (url, content, ts) VALUES (?,?,?)",
                (url, content, time.time())
            )
            conn.commit()

    async def get(self, url: str) -> Optional[str]:
        return await asyncio.to_thread(self._get_sync, url)

    async def set(self, url: str, content: str) -> None:
        await asyncio.to_thread(self._set_sync, url, content)


@dataclass
class ExtremizationConfig:
    enabled: bool  = True
    factor:  float = EXTREMIZE_FACTOR
    floor:   float = EXTREMIZE_FLOOR
    ceil:    float = EXTREMIZE_CEIL


def _logit(p: float) -> float:
    p = min(1.0 - 1e-12, max(1e-12, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def extremize_probability(p: float, cfg: ExtremizationConfig) -> float:
    if not cfg.enabled:
        return max(cfg.floor, min(cfg.ceil, p))
    return max(cfg.floor, min(cfg.ceil, _sigmoid(_logit(p) * cfg.factor)))


def clamp01(p: float) -> float:
    return float(max(MIN_P, min(MAX_P, float(p))))


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
        m = re.search(r"/questions/(\d+)(?:/|$)", str(question))
        if m:
            return m.group(1)
    except Exception:
        pass
    return "unknown"


def safe_community_prediction(question: MetaculusQuestion) -> Optional[float]:
    try:
        for attr in ("community_prediction", "prediction"):
            pred = getattr(question, attr, None)
            if pred is not None and isinstance(pred, (int, float)):
                return float(pred)
    except Exception as e:
        logger.warning(f"Community prediction unavailable for Q{extract_question_id(question)}: {e}")
    return None


def is_meaningful_research_text(txt: str) -> bool:
    if not txt:
        return False
    low = txt.lower()
    if "failed:" in low or "error:" in low or "timeout" in low:
        return False
    return len(txt.strip()) > 160


def interpolate_missing_percentiles(reported: List[Percentile],
                                    target_percentiles: List[float]) -> List[Percentile]:
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


def enforce_numeric_constraints(percentiles: List[Percentile],
                                question: NumericQuestion) -> List[Percentile]:
    lower = -np.inf if getattr(question, "open_lower_bound", False) else getattr(question, "lower_bound", None)
    upper =  np.inf if getattr(question, "open_upper_bound", False) else getattr(question, "upper_bound", None)
    if lower is None:
        lower = getattr(question, "nominal_lower_bound", None)
    if upper is None:
        upper = getattr(question, "nominal_upper_bound", None)
    if lower is None:
        lower = -np.inf
    if upper is None:
        upper = np.inf
    bounded = [
        Percentile(float(p.percentile), float(max(lower, min(upper, p.value))))
        for p in percentiles
    ]
    srt  = sorted(bounded, key=lambda x: x.percentile)
    vals = [p.value for p in srt]
    for i in range(1, len(vals)):
        if vals[i] < vals[i - 1]:
            vals[i] = vals[i - 1]
    return [Percentile(srt[i].percentile, float(vals[i])) for i in range(len(vals))]


def derive_numeric_fallback_bounds(question: NumericQuestion,
                                   anchor: Optional[float]) -> Tuple[float, float]:
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


def log_forecast_for_calibration(question, prediction_value, reasoning,
                                  model_ids, research_used, searchers_used):
    entry = {
        "timestamp":            datetime.utcnow().isoformat(),
        "question_id":          extract_question_id(question),
        "question_type":        question.__class__.__name__,
        "question_text":        getattr(question, "question_text", ""),
        "resolution_date":      getattr(question, "resolution_date", None),
        "community_prediction": safe_community_prediction(question),
        "prediction_value":     prediction_value,
        "models_used":          model_ids,
        "research_used":        research_used,
        "searchers_used":       searchers_used,
        "reasoning_snippet":    reasoning[:1500],
    }
    try:
        with open(CALIBRATION_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.warning(f"Calibration log write failed: {exc}")


async def with_timeout(coro, seconds: float, label: str) -> str:
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        return f"{label} timed out after {seconds}s"
    except Exception as e:
        return f"{label} error: {e}"


def backoff_sleep(attempt: int) -> None:
    base   = min(RETRY_MAX_S, RETRY_BASE_S * (2 ** attempt))
    jitter = random.uniform(0.0, base * 0.25)
    time.sleep(base + jitter)


def build_reasoning_block(question, forecast_text: str, base_rate_text: str,
                           methodology_text: str, strategy: str, profile: QuestionProfile,
                           searchers_used: List[str], models_used: List[str],
                           minibench: bool, ext_factor: float) -> str:
    today         = datetime.utcnow().strftime("%Y-%m-%d")
    searchers     = ", ".join(searchers_used) if searchers_used else "None"
    models        = ", ".join(models_used) if models_used else "Unknown"
    minibench_tag = f" [minibench — extremization factor {ext_factor:.2f}]" if minibench else ""
    return clean_indents(f"""
    Date (UTC): {today}
    Forecast: {forecast_text}
    Anchor / base rate: {base_rate_text}
    Domain: {profile.domain} | Geography: {profile.geography or 'global'} | Strategy: {strategy}{minibench_tag}

    Methodology:
    {methodology_text}

    Research sources: {searchers}
    Models: {models}
    """).strip()


_ARITH_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*$")


def _safe_eval(expr: str) -> Optional[float]:
    m = _ARITH_RE.match(expr.strip())
    if not m:
        return None
    a, op, b = float(m[1]), m[2], float(m[3])
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/": return (a / b) if abs(b) > 1e-12 else None
    return None


def sanitize_numeric_json(text: str) -> str:
    def repl(m: re.Match) -> str:
        raw = m.group(2).strip()
        v   = _safe_eval(raw)
        return m.group(1) + (str(v) if v is not None else raw)
    return re.sub(r'("percentile"\s*:\s*)([^,\]\}\n]+)', repl, text)


class Yrambot(ForecastBot):
    _max_concurrent_questions            = MAX_CONCURRENT_QUESTIONS
    _concurrency_limiter                 = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2
    _min_seconds_between_search_calls    = 1.2
    _min_seconds_between_llm_calls       = 0.35
    _last_search_call_ts                 = 0.0
    _last_llm_call_ts                    = 0.0

    def __init__(self, *args, client_spec: Optional[ClientSpecialisation] = None, **kwargs):
        llms = kwargs.pop("llms", None)
        if llms is None:
            gpt_llm    = GeneralLlm(model=_GPT_MODEL,    temperature=0.15, timeout=90, allowed_tries=3)
            sonnet_llm = GeneralLlm(model=_SONNET_MODEL, temperature=0.15, timeout=60, allowed_tries=3)
            llms = {
                "default":    gpt_llm,
                "researcher": gpt_llm,
                "parser":     sonnet_llm,
                "summarizer": sonnet_llm,
            }
        super().__init__(*args, llms=llms, **kwargs)

        self._client_spec        = client_spec or ClientSpecialisation()
        self._research_cache     = ResearchCache()
        self._validator          = ForecastValidator()
        self._analyser           = QuestionAnalyser(self.get_llm("researcher", "llm"))
        self._research_meta:     Dict[str, Dict[str, Any]] = {}
        self._active_tournament: Optional[str] = None

        gpt_search_llm = GeneralLlm(model=_GPT_MODEL, temperature=0.1, timeout=60, allowed_tries=2)
        self._sources  = SourceRegistry()
        tavily_src     = TavilySource(api_key=TAVILY_API_KEY or "", max_results=TAVILY_MAX_RESULTS)
        self._sources.register(tavily_src)
        if not tavily_src.is_available():
            self._sources.register(GptWebSearchSource(llm=gpt_search_llm))

        self._ext_cfg = ExtremizationConfig(
            enabled=EXTREMIZE_ENABLED,
            factor=EXTREMIZE_FACTOR,
            floor=EXTREMIZE_FLOOR,
            ceil=EXTREMIZE_CEIL,
        )
        self._ext_cfg_minibench = ExtremizationConfig(
            enabled=EXTREMIZE_ENABLED,
            factor=MINIBENCH_EXTREMIZE_FACTOR,
            floor=EXTREMIZE_FLOOR,
            ceil=EXTREMIZE_CEIL,
        )

    def register_source(self, source: BaseSource) -> None:
        self._sources.register(source)

    def _is_minibench(self) -> bool:
        return (self._active_tournament or "").lower() in MINIBENCH_IDS

    def _ext(self) -> ExtremizationConfig:
        return self._ext_cfg_minibench if self._is_minibench() else self._ext_cfg

    def _extremize(self, p: float) -> float:
        return extremize_probability(p, self._ext())

    async def _throttle_search(self) -> None:
        now  = time.time()
        wait = (self._last_search_call_ts + self._min_seconds_between_search_calls) - now
        if wait > 0:
            await asyncio.sleep(wait + random.random() * 0.15)
        self._last_search_call_ts = time.time()

    async def _throttle_llm(self) -> None:
        now  = time.time()
        wait = (self._last_llm_call_ts + self._min_seconds_between_llm_calls) - now
        if wait > 0:
            await asyncio.sleep(wait + random.random() * 0.10)
        self._last_llm_call_ts = time.time()

    async def _llm_invoke(self, model_key: str, prompt: str) -> str:
        await self._throttle_llm()
        return await self.get_llm(model_key, "llm").invoke(prompt)

    def _metaculus_context_block(self, question: MetaculusQuestion) -> str:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        qtxt  = (getattr(question, "question_text", "") or "").strip()
        rc    = (getattr(question, "resolution_criteria", "") or "").strip()
        bg    = (getattr(question, "background_info", "") or "").strip()
        url   = (getattr(question, "url", "") or getattr(question, "page_url", "") or "").strip()
        return (
            f"[Metaculus Context — {today}]\n"
            f"Question: {qtxt}\n"
            f"URL: {url or 'N/A'}\n"
            f"Resolution criteria: {rc or 'N/A'}\n"
            f"Background: {bg or 'N/A'}"
        )

    async def _plan_queries(self, question: MetaculusQuestion,
                            profile: QuestionProfile, question_type: str) -> List[str]:
        type_hints = {
            "binary":          "outcome likelihood, historical precedent, expert consensus, and causal drivers",
            "multiple_choice": "relative likelihood of each option, comparative base rates, and discriminating evidence",
            "numeric":         "quantitative benchmarks, data trends, historical value ranges, and unit-specific context",
        }
        hint     = type_hints.get(question_type, "base rates and key causal developments")
        geo_note = f" Geographic scope: {profile.geography}." if profile.geography else ""

        prompt = (
            "You are a research planning assistant for a structured forecasting system.\n\n"
            f"Domain: {profile.domain}.{geo_note}\n"
            f"Question type: {question_type}. Priority focus: {hint}.\n\n"
            "Generate 5 to 7 precise, non-overlapping web-search queries that together cover:\n"
            "- Historical base rates and reference class frequencies\n"
            "- Recent developments and primary causal drivers\n"
            "- Expert opinion, official data, or institutional forecasts\n"
            "- Prediction market signals (Metaculus, Polymarket, Metaforecast)\n"
            "- Domain-specific data sources (e.g. economic indicators, policy documents, academic studies)\n\n"
            "Return only a JSON array of strings. No explanation, no markdown, no preamble.\n\n"
            f"Question: {question.question_text}\n"
            f"Resolution criteria: {question.resolution_criteria}"
        )
        try:
            raw = await self._llm_invoke("researcher", prompt)
            raw = raw.strip()
            s, e = raw.find("["), raw.rfind("]")
            if s != -1 and e != -1:
                raw = raw[s:e + 1]
            queries = json.loads(raw)
            if isinstance(queries, list):
                return [q.strip() for q in queries if isinstance(q, str) and q.strip()][:7]
        except Exception:
            pass
        return [
            f"{question.question_text} recent developments",
            f"{question.question_text} historical base rate reference class",
            f"Metaculus Polymarket forecast {question.question_text}",
        ]

    async def _multi_source_research_bundle(self, question: MetaculusQuestion,
                                            profile: QuestionProfile,
                                            question_type: str) -> str:
        llm_queries = await self._plan_queries(question, profile, question_type)
        market_queries = [
            f"Metaculus community probability {question.question_text}",
            f"prediction market current odds {question.question_text}",
        ]
        seen: set = set()
        all_queries: List[str] = []
        for q in llm_queries + market_queries:
            q2 = q.strip()
            if q2 and q2 not in seen:
                seen.add(q2)
                all_queries.append(q2)

        blocks: List[str] = []
        for q in all_queries:
            await self._throttle_search()
            source_results = await self._sources.fetch_all(q)
            blocks.extend(source_results)
        return "\n\n".join(b for b in blocks if b.strip()).strip()

    async def _synthesize_research(self, question: MetaculusQuestion,
                                   metaculus_block: str, source_bundle: str,
                                   profile: QuestionProfile, question_type: str) -> str:
        type_focus = {
            "binary":          "Emphasize probability-relevant evidence: base rates, causal mechanisms, and forecast signposts.",
            "multiple_choice": "Emphasize comparative evidence across the option space and any clearly discriminating signals.",
            "numeric":         "Emphasize quantitative benchmarks, trend trajectories, plausible value ranges, and distribution shape.",
        }
        focus = type_focus.get(question_type, "Emphasize base rates and key causal drivers.")

        prompt = (
            f"You are a research synthesis assistant supporting a {profile.domain} forecasting question.\n"
            f"Geographic scope: {profile.geography or 'global'}. Question type: {question_type}.\n"
            f"{focus}\n\n"
            "Synthesize the evidence below into a structured research brief. Requirements:\n"
            "- Prioritize the most recent information; include specific dates and numerical figures.\n"
            "- Use only information present in the provided sources — do not introduce external knowledge.\n"
            "- Label each bullet point [METACULUS] or [WEB] to indicate its origin.\n"
            "- Structure your response as four sections:\n"
            "  1. Base rate / reference class\n"
            "  2. Key updates (3 to 6 items)\n"
            "  3. Primary uncertainties\n"
            "  4. Signposts that would materially shift the forecast\n"
            "- Maximum 2400 characters total.\n\n"
            f"Question: {question.question_text}\n\n"
            "Sources:\n"
            f"{metaculus_block}\n\n"
            f"[Web Research]\n{source_bundle if source_bundle else 'No web research results available.'}"
        )
        synthesized = await with_timeout(
            self.get_llm("summarizer", "llm").invoke(prompt),
            LLM_TIMEOUT_S, "research_synthesis"
        )
        return (synthesized or "").strip()

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            url = getattr(question, "page_url", "") or getattr(question, "url", "") or ""
            if url:
                cached = await self._research_cache.get(url)
                if cached:
                    return cached

            if isinstance(question, NumericQuestion):
                q_type = "numeric"
            elif isinstance(question, MultipleChoiceQuestion):
                q_type = "multiple_choice"
            else:
                q_type = "binary"

            profile  = await self._analyser.classify(question)
            strategy = ModellingStrategy.select(profile, q_type)
            qid      = extract_question_id(question)

            logger.info(
                f"[Yrambot] Q{qid} type={q_type} domain={profile.domain} "
                f"geo={profile.geography or 'global'} strategy={strategy} "
                f"horizon={profile.time_horizon_days}d"
            )

            metaculus_block = self._metaculus_context_block(question)
            source_bundle   = await self._multi_source_research_bundle(question, profile, q_type)
            synthesized     = await self._synthesize_research(
                question, metaculus_block, source_bundle, profile, q_type
            )

            if REQUIRE_RESEARCH and not is_meaningful_research_text(synthesized):
                raise RuntimeError(f"Insufficient synthesized research for Q{qid}.")

            final = (
                f"{metaculus_block}\n\n"
                f"[Research Summary]\n{synthesized}\n\n"
                f"[Raw Web Research]\n{source_bundle}"
            ) if source_bundle else f"{metaculus_block}\n\n[Research Summary]\n{synthesized}"

            if url:
                await self._research_cache.set(url, final)

            searchers_used = ["metaculus"]
            if source_bundle:
                available      = [s.name for s in self._sources.available_sources()]
                searchers_used += available

            self._research_meta[qid] = {
                "synthesized":    synthesized,
                "searchers_used": searchers_used,
                "profile":        profile,
                "strategy":       strategy,
                "question_type":  q_type,
            }
            return final

    async def _get_profile_and_strategy(self, question: MetaculusQuestion,
                                        question_type: str = "binary") -> Tuple[QuestionProfile, str]:
        qid  = extract_question_id(question)
        meta = self._research_meta.get(qid, {})
        if meta.get("profile") and meta.get("strategy"):
            return meta["profile"], meta["strategy"]
        profile  = await self._analyser.classify(question)
        strategy = ModellingStrategy.select(profile, question_type)
        return profile, strategy

    async def _forecast_binary_core(self, question: BinaryQuestion,
                                    research: str, profile: QuestionProfile,
                                    strategy: str) -> Tuple[float, str]:
        qid       = extract_question_id(question)
        base_rate = safe_community_prediction(question)
        base_str  = (
            f"Metaculus community probability: {base_rate:.4f}"
            if isinstance(base_rate, (int, float))
            else "No community probability currently available."
        )
        today_utc   = datetime.utcnow().strftime("%Y-%m-%d")
        client_ctx  = self._client_spec.extra_context or "None provided."
        fine_print  = getattr(question, "fine_print", "") or "None."

        prompt = (
            "You are an expert superforecaster. Your task is to produce a single calibrated probability estimate "
            "for the question below.\n\n"
            "Forecasting protocol — follow each step before stating a number:\n"
            "1. Outside view first: identify the reference class and its historical base rate.\n"
            "2. Inside view: list the specific evidence from the research that shifts the probability up or down.\n"
            "3. Bias audit: check for availability bias, anchoring, and overconfidence.\n"
            "4. Steel-man the opposing view: what is the strongest argument against your current lean?\n"
            "5. Synthesize: produce a final probability that reflects genuine uncertainty. "
            "Avoid round numbers unless truly warranted.\n\n"
            f"{ModellingStrategy.get_prompt_block(strategy, profile)}\n\n"
            f"Question: {question.question_text}\n"
            f"Resolution criteria: {question.resolution_criteria}\n"
            f"Fine print: {fine_print}\n"
            f"Today (UTC): {today_utc}\n"
            f"{base_str}\n\n"
            f"Research summary:\n{research}\n\n"
            f"Client context:\n{client_ctx}\n\n"
            "End your response with this exact line — no other format accepted:\n"
            "FINAL PROBABILITY: <integer between 0 and 100>%"
        )

        if REQUIRE_RESEARCH and (not research or len(research.strip()) < 120):
            raise RuntimeError(f"Insufficient research for binary forecast Q{qid}.")

        raw  = await with_timeout(
            self.get_llm("default", "llm").invoke(prompt), LLM_TIMEOUT_S, "binary_llm"
        )
        pred: BinaryPrediction = await structure_output(
            raw, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        return clamp01(float(pred.prediction_in_decimal)), str(raw)

    async def _forecast_mc_core(self, question: MultipleChoiceQuestion,
                                 research: str, profile: QuestionProfile,
                                 strategy: str) -> Tuple[PredictedOptionList, str]:
        qid        = extract_question_id(question)
        today_utc  = datetime.utcnow().strftime("%Y-%m-%d")
        client_ctx = self._client_spec.extra_context or "None provided."
        options_list = "\n".join(f"  - {opt}" for opt in question.options)

        prompt = (
            "You are an expert superforecaster. Your task is to assign well-calibrated probabilities "
            "to each of the listed options for the question below.\n\n"
            "Forecasting protocol:\n"
            "1. For each option, identify its reference class and historical base rate.\n"
            "2. Apply evidence from the research to update each option's probability independently.\n"
            "3. Ensure the distribution reflects genuine uncertainty — avoid concentrating mass "
            "on one option without strong, specific evidence.\n"
            "4. All probabilities must sum to exactly 1.0. Every option must appear.\n\n"
            f"{ModellingStrategy.get_prompt_block(strategy, profile)}\n\n"
            f"Question: {question.question_text}\n"
            f"Resolution criteria: {question.resolution_criteria}\n"
            f"Today (UTC): {today_utc}\n\n"
            f"Options (use these exact names — do not paraphrase):\n{options_list}\n\n"
            f"Research summary:\n{research}\n\n"
            f"Client context:\n{client_ctx}\n\n"
            "Return a JSON object only — no markdown, no commentary, no preamble:\n"
            '{"predicted_options": [{"option_name": "<exact name>", "probability": <decimal 0 to 1>}, ...]}\n'
            "Every option must appear exactly once. Probabilities must sum to 1."
        )

        if REQUIRE_RESEARCH and (not research or len(research.strip()) < 120):
            raise RuntimeError(f"Insufficient research for multiple-choice forecast Q{qid}.")

        raw    = await with_timeout(
            self.get_llm("default", "llm").invoke(prompt), LLM_TIMEOUT_S, "mc_llm"
        )
        result = await structure_output(
            raw, PredictedOptionList, model=self.get_llm("parser", "llm"),
            additional_instructions=f"Option names must match exactly: {question.options}"
        )
        probs = {o.option_name: max(0.0, float(o.probability)) for o in result.predicted_options}
        for opt in question.options:
            probs.setdefault(opt, 0.0)
        total = sum(probs.values()) or 1.0
        probs = {k: v / total for k, v in probs.items()}
        final = [PredictedOption(option_name=opt, probability=float(probs[opt])) for opt in question.options]
        return PredictedOptionList(predicted_options=final), str(raw)

    async def _forecast_numeric_core(self, question: NumericQuestion,
                                      research: str, profile: QuestionProfile,
                                      strategy: str) -> Tuple[List[Percentile], str]:
        qid       = extract_question_id(question)
        today_utc = datetime.utcnow().strftime("%Y-%m-%d")
        base_rate = safe_community_prediction(question)
        base_str  = (
            f"Metaculus community scalar estimate: {base_rate:,.6g}"
            if isinstance(base_rate, (int, float))
            else "No community scalar estimate available."
        )
        lower_ref = getattr(question, "lower_bound", None) or getattr(question, "nominal_lower_bound", None)
        upper_ref = getattr(question, "upper_bound", None) or getattr(question, "nominal_upper_bound", None)
        open_l    = getattr(question, "open_lower_bound", False)
        open_u    = getattr(question, "open_upper_bound", False)
        bounds_note = (
            f"Lower bound: {lower_ref} ({'open-ended' if open_l else 'closed'}). "
            f"Upper bound: {upper_ref} ({'open-ended' if open_u else 'closed'})."
        )
        client_ctx = self._client_spec.extra_context or "None provided."

        prompt = (
            "You are an expert superforecaster specializing in quantitative estimation. "
            "Your task is to produce a well-calibrated probability distribution over possible outcomes.\n\n"
            "Forecasting protocol:\n"
            "1. Anchor to a historical reference distribution or documented base rate.\n"
            "2. Apply inside-view updates based on the specific evidence in the research summary.\n"
            "3. Explicitly consider tail scenarios — do not produce overconfident narrow intervals.\n"
            "4. Verify that all percentile values are strictly increasing before outputting.\n\n"
            f"{ModellingStrategy.get_prompt_block(strategy, profile)}\n\n"
            f"Question: {question.question_text}\n"
            f"Units: {question.unit_of_measure or 'infer from context'}\n"
            f"Resolution criteria: {question.resolution_criteria}\n"
            f"Today (UTC): {today_utc}\n"
            f"{bounds_note}\n"
            f"{base_str}\n\n"
            f"Research summary:\n{research}\n\n"
            f"Client context:\n{client_ctx}\n\n"
            "Output format — strict rules:\n"
            "- Return a JSON array only. No markdown fences, no commentary, no preamble.\n"
            "- Use numeric literals only. Do not write arithmetic expressions (write 1500, not 1000+500).\n"
            "- Percentile values must be in the range [0, 1] and strictly increasing.\n"
            "- Include exactly these six percentiles: 0.1, 0.2, 0.4, 0.6, 0.8, 0.9.\n\n"
            "Required format:\n"
            '[{"percentile":0.1,"value":<number>},{"percentile":0.2,"value":<number>},'
            '{"percentile":0.4,"value":<number>},{"percentile":0.6,"value":<number>},'
            '{"percentile":0.8,"value":<number>},{"percentile":0.9,"value":<number>}]'
        )

        if REQUIRE_RESEARCH and (not research or len(research.strip()) < 120):
            raise RuntimeError(f"Insufficient research for numeric forecast Q{qid}.")

        raw = await with_timeout(
            self.get_llm("default", "llm").invoke(prompt), LLM_TIMEOUT_S, "num_llm"
        )

        try:
            percentile_list: List[Percentile] = await structure_output(
                raw, list[Percentile], model=self.get_llm("parser", "llm")
            )
        except Exception:
            fixed = sanitize_numeric_json(str(raw))
            repair_prompt = (
                "Convert the text below into a valid JSON array of Percentile objects.\n\n"
                "Strict rules:\n"
                "- Numeric literals only — no arithmetic expressions whatsoever.\n"
                "- Percentile field values must be in [0, 1] and strictly increasing.\n"
                "- Output the JSON array only — no explanation, no markdown, no preamble.\n\n"
                f"Input text:\n{fixed}"
            )
            repaired = await with_timeout(
                self.get_llm("summarizer", "llm").invoke(repair_prompt),
                LLM_TIMEOUT_S, "num_repair"
            )
            percentile_list = await structure_output(
                repaired, list[Percentile], model=self.get_llm("parser", "llm")
            )

        target_ps    = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        interpolated = interpolate_missing_percentiles(percentile_list, target_ps)
        validated    = enforce_numeric_constraints(interpolated, question)

        vals = [p.value for p in validated]
        if len(set(round(v, 12) for v in vals)) == 1:
            lb, ub = derive_numeric_fallback_bounds(question, base_rate)
            mid    = float(vals[0])
            width  = (ub - lb) * 0.08 if np.isfinite(ub - lb) else max(1.0, abs(mid) * 0.25)
            widened = [
                Percentile(0.1, mid - 1.2 * width),
                Percentile(0.2, mid - 0.7 * width),
                Percentile(0.4, mid - 0.2 * width),
                Percentile(0.6, mid + 0.2 * width),
                Percentile(0.8, mid + 0.7 * width),
                Percentile(0.9, mid + 1.2 * width),
            ]
            validated = enforce_numeric_constraints(widened, question)

        return validated, str(raw)

    async def _run_forecast_on_binary(self, question: BinaryQuestion,
                                       research: str) -> ReasonedPrediction[float]:
        profile, strategy = await self._get_profile_and_strategy(question, "binary")
        qid               = extract_question_id(question)
        base              = safe_community_prediction(question)
        base_rate_text    = f"{base:.1%}" if isinstance(base, (int, float)) else "None"

        try:
            raw_p, _ = await self._forecast_binary_core(question, research, profile, strategy)
        except Exception as e:
            raw_p = clamp01(float(base) if isinstance(base, (int, float)) else 0.5)
            logger.warning(f"Binary fallback Q{qid}: {e}")

        p_final = self._extremize(raw_p)

        if isinstance(base, (int, float)):
            p_final = clamp01(CROWD_BLEND_MIXED * p_final + (1.0 - CROWD_BLEND_MIXED) * float(base))

        meta           = self._research_meta.get(qid, {})
        searchers_used = meta.get("searchers_used", [])
        models_used    = [_GPT_MODEL, _SONNET_MODEL]

        self._validator.validate(question, profile, strategy, p_final, research)

        blend_note = (
            f"Community anchor blended at weight {CROWD_BLEND_MIXED}."
            if isinstance(base, (int, float))
            else "No community anchor available for blending."
        )
        methodology = (
            f"Superforecasting protocol with {strategy} analytical framework.\n"
            f"Logit-space extremization applied (factor={self._ext().factor:.2f}).\n"
            f"{'Minibench aggressive pass active.' + chr(10) if self._is_minibench() else ''}"
            f"{blend_note}"
        )

        reasoning = build_reasoning_block(
            question, forecast_text=f"{p_final:.1%}", base_rate_text=base_rate_text,
            methodology_text=methodology, strategy=strategy, profile=profile,
            searchers_used=searchers_used, models_used=models_used,
            minibench=self._is_minibench(), ext_factor=self._ext().factor
        )
        log_forecast_for_calibration(question, p_final, reasoning, models_used, True, searchers_used)
        time.sleep(PUBLISH_SLEEP_S)
        return ReasonedPrediction(prediction_value=p_final, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion,
                                                research: str) -> ReasonedPrediction[PredictedOptionList]:
        profile, strategy = await self._get_profile_and_strategy(question, "multiple_choice")
        qid               = extract_question_id(question)

        try:
            out, _ = await self._forecast_mc_core(question, research, profile, strategy)
        except Exception as e:
            n   = len(question.options)
            out = PredictedOptionList(predicted_options=[
                PredictedOption(option_name=opt, probability=1.0 / n) for opt in question.options
            ])
            logger.warning(f"Multiple-choice fallback Q{qid}: {e}")

        if self._is_minibench():
            raw_probs  = {o.option_name: o.probability for o in out.predicted_options}
            extremized = {k: extremize_probability(v, self._ext_cfg_minibench)
                          for k, v in raw_probs.items()}
            total = sum(extremized.values()) or 1.0
            out   = PredictedOptionList(predicted_options=[
                PredictedOption(option_name=opt, probability=extremized[opt] / total)
                for opt in question.options
            ])

        meta           = self._research_meta.get(qid, {})
        searchers_used = meta.get("searchers_used", [])
        models_used    = [_GPT_MODEL, _SONNET_MODEL]

        self._validator.validate(
            question, profile, strategy,
            [o.probability for o in out.predicted_options], research
        )

        methodology = (
            f"Superforecasting protocol with {strategy} analytical framework.\n"
            f"Probability mass assigned across all {len(question.options)} options and renormalized to sum to 1.\n"
            f"{'Per-option logit extremization applied (minibench mode).' if self._is_minibench() else 'Standard probability distribution without additional extremization.'}"
        )

        reasoning = build_reasoning_block(
            question,
            forecast_text=", ".join([f"{x.option_name}: {x.probability:.1%}" for x in out.predicted_options]),
            base_rate_text="Metaculus community distribution (qualitative anchor)",
            methodology_text=methodology, strategy=strategy, profile=profile,
            searchers_used=searchers_used, models_used=models_used,
            minibench=self._is_minibench(), ext_factor=self._ext().factor
        )
        log_forecast_for_calibration(
            question, [x.probability for x in out.predicted_options],
            reasoning, models_used, True, searchers_used
        )
        time.sleep(PUBLISH_SLEEP_S)
        return ReasonedPrediction(prediction_value=out, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion,
                                        research: str) -> ReasonedPrediction[NumericDistribution]:
        profile, strategy = await self._get_profile_and_strategy(question, "numeric")
        qid               = extract_question_id(question)
        base              = safe_community_prediction(question)
        base_rate_text    = f"{base:,.4g}" if isinstance(base, (int, float)) else "None"

        try:
            validated, _ = await self._forecast_numeric_core(question, research, profile, strategy)
        except Exception as e:
            lb, ub = derive_numeric_fallback_bounds(question, base)
            center = float(base) if isinstance(base, (int, float)) else (lb + ub) / 2.0
            width  = (ub - lb) * 0.30
            target_ps = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
            vals = [
                center - 0.9 * width, center - 0.5 * width, center - 0.15 * width,
                center + 0.15 * width, center + 0.5 * width, center + 0.9 * width,
            ]
            vals      = [max(lb, min(ub, v)) for v in vals]
            validated = enforce_numeric_constraints(
                [Percentile(p, v) for p, v in zip(target_ps, vals)], question
            )
            logger.warning(f"Numeric fallback Q{qid}: {e}")

        dist = NumericDistribution.from_question(validated, question)

        meta           = self._research_meta.get(qid, {})
        searchers_used = meta.get("searchers_used", [])
        models_used    = [_GPT_MODEL, _SONNET_MODEL]

        self._validator.validate(question, profile, strategy, dist, research)

        methodology = (
            f"Superforecasting protocol with {strategy} analytical framework.\n"
            f"Six-percentile distribution (p10 through p90) with bound enforcement and strict monotonicity guarantee.\n"
            f"{'Minibench aggressive extremization pass active.' if self._is_minibench() else 'Standard extremization configuration.'}"
        )

        reasoning = build_reasoning_block(
            question,
            forecast_text=", ".join([f"p{int(p.percentile * 100)}={p.value:,.6g}" for p in validated]),
            base_rate_text=base_rate_text, methodology_text=methodology,
            strategy=strategy, profile=profile,
            searchers_used=searchers_used, models_used=models_used,
            minibench=self._is_minibench(), ext_factor=self._ext().factor
        )
        log_forecast_for_calibration(
            question, [p.value for p in validated],
            reasoning, models_used, True, searchers_used
        )
        time.sleep(PUBLISH_SLEEP_S)
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    async def forecast_on_tournament(self, tournament_id, *args, **kwargs):
        self._active_tournament = str(tournament_id)
        if self._is_minibench():
            logger.info(
                f"[Yrambot] Minibench tournament detected ({tournament_id}) — "
                f"extremization factor set to {MINIBENCH_EXTREMIZE_FACTOR}"
            )
        reports = await super().forecast_on_tournament(tournament_id, *args, **kwargs)
        summary = self._validator.summary()
        if summary:
            logger.info(f"[Yrambot] Validation summary:\n{json.dumps(summary, indent=2)}")
        return reports

    async def forecast_questions(self, *args, **kwargs):
        reports = await super().forecast_questions(*args, **kwargs)
        summary = self._validator.summary()
        if summary:
            logger.info(f"[Yrambot] Validation summary:\n{json.dumps(summary, indent=2)}")
        return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Yrambot.")
    parser.add_argument(
        "--tournament-ids", nargs="+", type=str,
        default=[
            "32916",
            "minibench",
            "market-pulse-26q1",
            MetaculusApi.CURRENT_MINIBENCH_ID,
        ],
    )
    args = parser.parse_args()

    spec = ClientSpecialisation(
        domain_focus=[],
        trusted_domains=[],
        excluded_domains=[],
        extra_context="",
        calibration_target=0.15,
    )

    bot = Yrambot(
        client_spec=spec,
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
                    if any(x in msg for x in ("too many requests", "cloudflare", "1015", "429")):
                        logger.error(
                            f"Rate-limited on tournament {tid} "
                            f"(attempt {attempt + 1}/{RETRY_MAX}): {e}"
                        )
                        backoff_sleep(attempt)
                        continue
                    raise
            time.sleep(TOURNAMENT_SLEEP_S)

        bot.log_report_summary(all_reports)
        logger.info(f"Run completed. Calibration log written to: {CALIBRATION_LOG_FILE}")
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
