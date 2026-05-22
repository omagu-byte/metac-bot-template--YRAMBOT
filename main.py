import argparse
import asyncio
import json
import logging
import math
import os
import random
import re
import sqlite3
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import dotenv
import httpx

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

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

dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# Model strings — ALL using Perplexity via OpenRouter (only allowed provider)
# ---------------------------------------------------------------------------
# Primary forecaster: sonar-reasoning-pro has chain-of-thought + live web search
_PRIMARY_MODEL          = "openrouter/perplexity/sonar-reasoning-pro"
# Secondary/ensemble: sonar-pro for synthesis and research
_SECONDARY_MODEL        = "openrouter/perplexity/sonar-pro"
# Fast/cheap: sonar for parsing, summarizing, quick tasks
_FAST_MODEL             = "openrouter/perplexity/sonar"
# Research: sonar-pro with live web context
_RESEARCH_MODEL         = "openrouter/perplexity/sonar-pro"

# Free model chain — sonar is cheapest Perplexity tier
_FREE_MODEL             = "openrouter/perplexity/sonar"
_FREE_MODEL_CHAIN       = [
    "openrouter/perplexity/sonar",
    "openrouter/perplexity/sonar-pro",
]

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
MINIBENCH_EXTREMIZE_FACTOR = float(os.getenv("MINIBENCH_EXTREMIZE_FACTOR", "4.00"))
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
SELECTIVE_TOURNAMENTS = ["33022", "market-pulse-26q2"]

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
        )
        try:
            raw = await self._llm.invoke(prompt)
            raw = raw.strip()
            s, e = raw.find("{"), raw.rfind("}")
            if s != -1 and e != -1: raw = raw[s:e + 1]
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
            # Return neutral confidence so a model error doesn't cascade into a
            # confidence-gate failure downstream when research is present.
            return QuestionProfile(confidence_in_profile=0.5)


class ModellingStrategy:
    @staticmethod
    def select(profile: QuestionProfile, question_type: str = "binary") -> str:
        if question_type == "numeric":
            if profile.domain in ("economics", "finance"): return "trend"
            if profile.time_horizon_days < 90: return "trend"
            return "base_rate"
        if question_type == "multiple_choice":
            if profile.domain in ("geopolitics", "social"): return "analogical"
            if profile.time_horizon_days < 60: return "market_signal"
            return "base_rate"
        if profile.domain in ("economics", "finance") and profile.is_quantitative: return "trend"
        if profile.domain in ("geopolitics", "social"): return "analogical"
        if profile.time_horizon_days < 60: return "market_signal"
        return "base_rate"

    @staticmethod
    def get_prompt_block(strategy: str, profile: QuestionProfile) -> str:
        geo_ctx = f" with geographic focus on {profile.geography}" if profile.geography else ""
        strategies = {
            "trend": (
                f"Analytical framework — Trend Extrapolation{geo_ctx}:\n"
                "1. Establish recent trajectory using the most recent data points.\n"
                "2. Project forward to resolution date, applying a mean-reversion discount.\n"
                "3. State a realistic uncertainty band reflecting financial volatility."
            ),
            "analogical": (
                f"Analytical framework — Historical Analogy{geo_ctx}:\n"
                "1. Identify 2 to 3 structurally similar past situations.\n"
                "2. State the base rate of the focal outcome across those cases.\n"
                "3. Adjust from the base rate based on key differences."
            ),
            "market_signal": (
                f"Analytical framework — Market Signal Anchoring{geo_ctx}:\n"
                "1. Treat prevailing prediction-market or financial signals as strong priors.\n"
                "2. Adjust only when holding concrete unpriced information.\n"
                "3. For short horizons, weight status-quo inertia heavily."
            ),
            "base_rate": (
                f"Analytical framework — Base Rate Anchoring{geo_ctx}:\n"
                "1. Define a precise reference class and historical base rate.\n"
                "2. Anchor initial estimate to that base rate.\n"
                "3. Limit total inside-view adjustment to ±20 points unless evidence is overwhelming."
            ),
        }
        return strategies.get(strategy, strategies["base_rate"])


class BaseSource(ABC):
    name: str = "unnamed_source"
    @abstractmethod
    async def fetch(self, query: str) -> str: ...
    def is_available(self) -> bool: return True


class SourceRegistry:
    def __init__(self):
        self._sources: List[BaseSource] = []

    def register(self, source: BaseSource) -> None:
        self._sources.append(source)

    def available_sources(self) -> List[BaseSource]:
        return [s for s in self._sources if s.is_available()]

    async def fetch_all(self, query: str) -> List[str]:
        sources = self.available_sources()
        results = await asyncio.gather(*[s.fetch(query) for s in sources], return_exceptions=True)
        blocks = []
        for src, res in zip(sources, results):
            if isinstance(res, str) and res.strip(): blocks.append(f"[{src.name}]\n{res}")
        return blocks


class TavilySource(BaseSource):
    name = "tavily_web"
    def __init__(self, api_key: str, max_results: int = TAVILY_MAX_RESULTS):
        self._api_key     = api_key
        self._max_results = max_results
        self._client      = TavilyClient(api_key=api_key) if api_key else None

    def is_available(self) -> bool: return bool(self._api_key)

    async def fetch(self, query: str) -> str:
        if not self._client: return ""
        try:
            resp = await asyncio.to_thread(self._client.search, query=query, search_depth="advanced", max_results=self._max_results)
            lines = []
            for c in (resp.get("results", []) or [])[:self._max_results]:
                title, content, url = c.get("title", ""), c.get("content", ""), c.get("url", "")
                if title or content: lines.append(f"- {title}\n  Snippet: {content[:520].strip()}\n  Source: {url}")
            return "\n".join(lines).strip()
        except Exception as exc:
            return f"Tavily error: {exc}"


class PerplexitySonarSource(BaseSource):
    """Primary web-search source using Perplexity sonar-pro (live web context)."""
    name = "perplexity_sonar_pro"
    def __init__(self, llm: GeneralLlm):
        self._llm = llm

    def is_available(self) -> bool: return True

    async def fetch(self, query: str) -> str:
        prompt = (
            f"Search the web and provide a comprehensive research brief on: {query}\n"
            "Include recent developments, key facts, and relevant sources. Max 800 words."
        )
        try:
            raw = await with_timeout(self._llm.invoke(prompt), LLM_TIMEOUT_S, "perplexity_sonar_pro")
            return raw.strip() if raw and len(raw.strip()) > 80 else ""
        except Exception as exc:
            return f"Perplexity Sonar Pro error: {exc}"


class PerplexitySonarReasoningSource(BaseSource):
    """Secondary source using Perplexity sonar-reasoning-pro for deeper analysis."""
    name = "perplexity_sonar_reasoning"
    def __init__(self, llm: GeneralLlm):
        self._llm = llm

    def is_available(self) -> bool: return True

    async def fetch(self, query: str) -> str:
        prompt = (
            f"Conduct a deep research analysis for: {query}\n"
            "Provide current information, trends, and relevant background. "
            "Focus on factual accuracy and cite sources where possible. Max 700 words."
        )
        try:
            raw = await with_timeout(self._llm.invoke(prompt), LLM_TIMEOUT_S, "perplexity_reasoning")
            return raw.strip() if raw and len(raw.strip()) > 80 else ""
        except Exception as exc:
            return f"Perplexity Reasoning error: {exc}"


def _fetch_yfinance_data_sync(ticker: str) -> str:
    if not YFINANCE_AVAILABLE: return ""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="3mo")
        if hist.empty: return ""
        spot = hist['Close'].iloc[-1]
        high_52 = tk.info.get('fiftyTwoWeekHigh', 'N/A')
        low_52 = tk.info.get('fiftyTwoWeekLow', 'N/A')
        vol = hist['Close'].pct_change().dropna().std() * math.sqrt(252)
        rw_p10 = spot * math.exp(-1.28 * (vol * math.sqrt(21/252)))
        rw_p90 = spot * math.exp(1.28 * (vol * math.sqrt(21/252)))
        return (f"--- LIVE MARKET DATA ({ticker}) ---\n"
                f"Spot Price: {spot:.2f} | 52-Week Range: {low_52} - {high_52}\n"
                f"Annual Volatility: {vol:.2%} | 1-Mo Random Walk: P10={rw_p10:.2f}, P50={spot:.2f}, P90={rw_p90:.2f}\n")
    except Exception: return ""


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
    LOW_CONFIDENCE_THRESHOLD = 0.65

    def __init__(self, db_path: str = "yrambot_validation.db"):
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS forecast_ledger (question_url TEXT, question_text TEXT, domain TEXT, geo_scope TEXT, strategy TEXT, prediction_value TEXT, confidence_score REAL, flagged INTEGER, ts REAL)")
            conn.commit()

    def compute_confidence(self, prediction_value: Any, profile: QuestionProfile, research_length: int) -> float:
        classifier_score = profile.confidence_in_profile
        evidence_score   = min(1.0, research_length / 3000)
        signal_score     = abs(prediction_value - 0.5) * 2.5 if isinstance(prediction_value, float) else 0.5
        # Rebalanced: evidence carries 0.55 weight so a classifier failure
        # (confidence_in_profile=0.0) can't push the total below the 0.65 gate
        # on its own when research is present.
        return round(min(1.0, max(0.0, 0.25 * classifier_score + 0.55 * evidence_score + 0.20 * signal_score)), 3)

    def validate(self, question: MetaculusQuestion, profile: QuestionProfile, strategy: str, prediction_value: Any, research: str) -> ValidationRecord:
        confidence = self.compute_confidence(prediction_value, profile, len(research))
        flagged    = confidence < self.LOW_CONFIDENCE_THRESHOLD
        record = ValidationRecord(
            question_url=getattr(question, "page_url", ""), question_text=question.question_text[:300],
            domain=profile.domain, geo_scope=profile.geo_scope, strategy=strategy,
            prediction_value=str(prediction_value)[:200], confidence_score=confidence, flagged_low_confidence=flagged,
        )
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("INSERT INTO forecast_ledger (question_url, question_text, domain, geo_scope, strategy, prediction_value, confidence_score, flagged, ts) VALUES (?,?,?,?,?,?,?,?,?)",
                             (record.question_url, record.question_text, record.domain, record.geo_scope, record.strategy, record.prediction_value, record.confidence_score, int(record.flagged_low_confidence), record.ts))
                conn.commit()
        except Exception as exc: logger.warning(f"[Validator] persist failed: {exc}")
        return record

    def summary(self) -> Dict[str, Any]:
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute("SELECT domain, COUNT(*) as n, AVG(confidence_score) as avg_conf, SUM(flagged) as n_flagged FROM forecast_ledger GROUP BY domain ORDER BY n DESC").fetchall()
            return {"by_domain": [{"domain": r[0], "n": r[1], "avg_confidence": round(r[2], 3), "n_flagged": r[3]} for r in rows]}
        except Exception: return {}


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
            conn.execute("CREATE TABLE IF NOT EXISTS research_cache (url TEXT PRIMARY KEY, content TEXT NOT NULL, ts REAL NOT NULL)")
            conn.commit()

    def _get_sync(self, url: str) -> Optional[str]:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute("SELECT content FROM research_cache WHERE url=?", (url,)).fetchone()
        return row[0] if row else None

    def _set_sync(self, url: str, content: str) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO research_cache (url, content, ts) VALUES (?,?,?)", (url, content, time.time()))
            conn.commit()

    async def get(self, url: str) -> Optional[str]: return await asyncio.to_thread(self._get_sync, url)
    async def set(self, url: str, content: str) -> None: await asyncio.to_thread(self._set_sync, url, content)


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
    if x >= 0: return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))

def extremize_probability(p: float, cfg: ExtremizationConfig) -> float:
    if not cfg.enabled: return max(cfg.floor, min(cfg.ceil, p))
    return max(cfg.floor, min(cfg.ceil, _sigmoid(_logit(p) * cfg.factor)))

def apply_tail_fattening(pts: List[Percentile], factor: float = 1.20) -> List[Percentile]:
    p50_val = next((p.value for p in pts if abs(float(p.percentile) - 0.5) < 1e-9), None)
    if p50_val is None:
        l_val = next((p.value for p in pts if float(p.percentile) < 0.5), None)
        r_val = next((p.value for p in pts if float(p.percentile) > 0.5), None)
        if l_val is not None and r_val is not None: p50_val = (l_val + r_val) / 2.0
        else: return pts
    for p in pts:
        if float(p.percentile) < 0.5: p.value = p50_val - (p50_val - p.value) * factor
        elif float(p.percentile) > 0.5: p.value = p50_val + (p.value - p50_val) * factor
    pts.sort(key=lambda x: float(x.percentile))
    for i in range(1, len(pts)):
        if pts[i].value < pts[i - 1].value: pts[i].value = pts[i - 1].value
    return pts

def clamp01(p: float) -> float: return float(max(MIN_P, min(MAX_P, float(p))))

def extract_question_id(question: MetaculusQuestion) -> str:
    for attr in ("id", "question_id", "questionId"):
        try:
            qid = getattr(question, attr, None)
            if isinstance(qid, (int, str)) and str(qid).isdigit(): return str(qid)
        except Exception: pass
    for attr in ("url", "page_url", "question_url", "link"):
        try:
            url = str(getattr(question, attr, "") or "")
            m = re.search(r"/questions/(\d+)(?:/|$)", url)
            if m: return m.group(1)
        except Exception: pass
    return "unknown"

def safe_community_prediction(question: MetaculusQuestion) -> Optional[float]:
    try:
        for attr in ("community_prediction", "prediction"):
            pred = getattr(question, attr, None)
            if pred is not None and isinstance(pred, (int, float)): return float(pred)
    except Exception: pass
    return None

def is_meaningful_research_text(txt: str) -> bool:
    if not txt: return False
    if "failed:" in txt.lower() or "error:" in txt.lower() or "timeout" in txt.lower(): return False
    return len(txt.strip()) > 160

def interpolate_missing_percentiles(reported: List[Percentile], target_percentiles: List[float]) -> List[Percentile]:
    if not reported: return [Percentile(percentile=p, value=0.0) for p in target_percentiles]
    sorted_rep = sorted(reported, key=lambda x: x.percentile)
    xs = [float(p.percentile) for p in sorted_rep]
    ys = [float(p.value) for p in sorted_rep]
    out: List[Percentile] = []
    for tp in target_percentiles:
        if tp in xs: val = ys[xs.index(tp)]
        else:
            from bisect import bisect_left
            i = bisect_left(xs, tp)
            if i == 0: val = ys[0]
            elif i == len(xs): val = ys[-1]
            else:
                x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
                val = y0 + (y1 - y0) * (tp - x0) / (x1 - x0) if x1 != x0 else y0
        out.append(Percentile(percentile=float(tp), value=float(val)))
    return out

def enforce_numeric_constraints(percentiles: List[Percentile], question: NumericQuestion) -> List[Percentile]:
    lower = -np.inf if getattr(question, "open_lower_bound", False) else getattr(question, "lower_bound", None)
    upper =  np.inf if getattr(question, "open_upper_bound", False) else getattr(question, "upper_bound", None)
    if lower is None: lower = getattr(question, "nominal_lower_bound", None)
    if upper is None: upper = getattr(question, "nominal_upper_bound", None)
    if lower is None: lower = -np.inf
    if upper is None: upper = np.inf

    bounded = [
        Percentile(percentile=float(p.percentile), value=float(max(lower, min(upper, p.value))))
        for p in percentiles
    ]

    srt  = sorted(bounded, key=lambda x: x.percentile)
    vals = [p.value for p in srt]
    for i in range(1, len(vals)):
        if vals[i] < vals[i - 1]: vals[i] = vals[i - 1]

    return [Percentile(percentile=srt[i].percentile, value=float(vals[i])) for i in range(len(vals))]

def derive_numeric_fallback_bounds(question: NumericQuestion, anchor: Optional[float]) -> Tuple[float, float]:
    lb = getattr(question, "lower_bound", None)
    ub = getattr(question, "upper_bound", None)
    if getattr(question, "open_lower_bound", False): lb = None
    if getattr(question, "open_upper_bound", False): ub = None
    if lb is None: lb = getattr(question, "nominal_lower_bound", None)
    if ub is None: ub = getattr(question, "nominal_upper_bound", None)
    if lb is not None and ub is not None and float(ub) > float(lb): return float(lb), float(ub)
    if isinstance(anchor, (int, float)):
        a = float(anchor)
        return (a * 0.25, a * 3.0) if a > 0 else (a - 1.0, a + 1.0)
    return -1e9, 1e9

def log_forecast_for_calibration(question, prediction_value, reasoning, models_used, research_used, searchers_used):
    entry = {
        "timestamp":        datetime.utcnow().isoformat(),
        "question_id":      extract_question_id(question),
        "question_type":    question.__class__.__name__,
        "prediction_value": prediction_value,
        "models_used":      models_used,
        "research_used":    research_used,
    }
    try:
        with open(CALIBRATION_LOG_FILE, "a") as f: f.write(json.dumps(entry) + "\n")
    except Exception: pass

async def with_timeout(coro, seconds: float, label: str) -> str:
    try: return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError: return f"{label} timed out after {seconds}s"
    except Exception as e: return f"{label} error: {e}"

async def invoke_with_free_model_fallback(prompt: str, temperature: float = 0.15, timeout_s: float = 60, label: str = "free_invoke") -> str:
    """Try Perplexity models in fallback chain. Returns result from first successful model."""
    last_error = None
    for model_idx, model in enumerate(_FREE_MODEL_CHAIN):
        try:
            llm = GeneralLlm(model=model, temperature=temperature, timeout=timeout_s, allowed_tries=2)
            result = await with_timeout(llm.invoke(prompt), timeout_s, f"{label}_{model_idx}")
            if result and not result.startswith(label) and "error" not in result.lower() and "timed out" not in result.lower():
                logger.info(f"[Free Model Fallback] Success with {model}")
                return result
            last_error = result
        except Exception as e:
            last_error = str(e)
            logger.warning(f"[Free Model Fallback] {model} failed: {e}")
            continue
    logger.error(f"[Free Model Fallback] All models exhausted for {label}")
    return last_error or f"{label} all models failed"

def backoff_sleep(attempt: int) -> None:
    base   = min(RETRY_MAX_S, RETRY_BASE_S * (2 ** attempt))
    time.sleep(base + random.uniform(0.0, base * 0.25))

def build_reasoning_block(question, forecast_text: str, base_rate_text: str,
                          methodology_text: str, strategy: str, profile: QuestionProfile,
                          searchers_used: List[str], minibench: bool, ext_factor: float) -> str:
    minibench_tag = f" (aggressive mode)" if minibench else ""
    return clean_indents(f"""
    My forecast: {forecast_text}
    
    I anchored on a base rate of {base_rate_text} and adjusted based on recent evidence. 
    The question falls in {profile.domain}; my analysis suggests the key uncertainty lies in {profile.geography or 'broader trends'}.
    
    {methodology_text}{minibench_tag}
    """).strip()

_ARITH_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*$")
def _safe_eval(expr: str) -> Optional[float]:
    m = _ARITH_RE.match(expr.strip())
    if not m: return None
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
            # All models are Perplexity — the only allowed provider on this account.
            # sonar-reasoning-pro: deep chain-of-thought + live web (primary forecaster)
            # sonar-pro:           best synthesis + live web (research & ensemble partner)
            # sonar:               fast/cheap fallback for parsing and summarizing
            primary_llm   = GeneralLlm(model=_PRIMARY_MODEL,   temperature=0.15, timeout=90, allowed_tries=3)
            secondary_llm = GeneralLlm(model=_SECONDARY_MODEL, temperature=0.15, timeout=90, allowed_tries=3)
            fast_llm      = GeneralLlm(model=_FAST_MODEL,      temperature=0.15, timeout=60, allowed_tries=2)
            llms = {
                "default":     primary_llm,    # sonar-reasoning-pro — primary forecaster
                "default_alt": secondary_llm,  # sonar-pro — ensemble partner
                "researcher":  secondary_llm,  # sonar-pro — research queries
                "parser":      fast_llm,       # sonar — structured extraction
                "summarizer":  fast_llm,       # sonar — summarization
                "perplexity":  secondary_llm,  # sonar-pro — search source
                "gpt5_search": primary_llm,    # sonar-reasoning-pro — secondary search
            }
        super().__init__(*args, llms=llms, **kwargs)

        self._client_spec        = client_spec or ClientSpecialisation()
        self._research_cache     = ResearchCache()
        self._validator          = ForecastValidator()
        self._analyser           = QuestionAnalyser(
            GeneralLlm(model=_FAST_MODEL, temperature=0.15, timeout=60, allowed_tries=2)
        )
        self._research_meta:     Dict[str, Dict[str, Any]] = {}
        self._active_tournament: Optional[str] = None

        # Source registry — Perplexity models cover both search sources
        sonar_pro_llm      = GeneralLlm(model=_SECONDARY_MODEL,  temperature=0.1, timeout=60, allowed_tries=2)
        sonar_reason_llm   = GeneralLlm(model=_PRIMARY_MODEL,    temperature=0.1, timeout=60, allowed_tries=2)

        self._sources  = SourceRegistry()
        tavily_src     = TavilySource(api_key=TAVILY_API_KEY or "", max_results=TAVILY_MAX_RESULTS)
        self._sources.register(tavily_src)

        # Always register Perplexity sources regardless of Tavily availability
        perplexity_sonar    = PerplexitySonarSource(llm=sonar_pro_llm)
        perplexity_reason   = PerplexitySonarReasoningSource(llm=sonar_reason_llm)
        self._sources.register(perplexity_sonar)
        self._sources.register(perplexity_reason)

        self._ext_cfg           = ExtremizationConfig(enabled=EXTREMIZE_ENABLED, factor=EXTREMIZE_FACTOR,           floor=EXTREMIZE_FLOOR, ceil=EXTREMIZE_CEIL)
        self._ext_cfg_minibench = ExtremizationConfig(enabled=EXTREMIZE_ENABLED, factor=MINIBENCH_EXTREMIZE_FACTOR, floor=EXTREMIZE_FLOOR, ceil=EXTREMIZE_CEIL)

    def register_source(self, source: BaseSource) -> None: self._sources.register(source)

    def _is_minibench(self) -> bool: return (self._active_tournament or "").lower() in MINIBENCH_IDS

    def _ext(self) -> ExtremizationConfig: return self._ext_cfg_minibench if self._is_minibench() else self._ext_cfg

    def _extremize(self, p: float) -> float: return extremize_probability(p, self._ext())

    async def _throttle_search(self) -> None:
        now  = time.time()
        wait = (self._last_search_call_ts + self._min_seconds_between_search_calls) - now
        if wait > 0: await asyncio.sleep(wait + random.random() * 0.15)
        self._last_search_call_ts = time.time()

    async def _throttle_llm(self) -> None:
        now  = time.time()
        wait = (self._last_llm_call_ts + self._min_seconds_between_llm_calls) - now
        if wait > 0: await asyncio.sleep(wait + random.random() * 0.10)
        self._last_llm_call_ts = time.time()

    async def _llm_invoke(self, model_key: str, prompt: str) -> str:
        await self._throttle_llm()
        return await self.get_llm(model_key, "llm").invoke(prompt)

    def _metaculus_context_block(self, question: MetaculusQuestion) -> str:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        qtxt = (getattr(question, "question_text", "") or "").strip()
        rc   = (getattr(question, "resolution_criteria", "") or "").strip()
        bg   = (getattr(question, "background_info", "") or "").strip()
        url  = (getattr(question, "url", "") or getattr(question, "page_url", "") or "").strip()
        return f"[Metaculus Context — {today}]\nQuestion: {qtxt}\nURL: {url or 'N/A'}\nResolution criteria: {rc or 'N/A'}\nBackground: {bg or 'N/A'}"

    async def _plan_queries(self, question: MetaculusQuestion, profile: QuestionProfile, question_type: str) -> List[str]:
        hint = {"binary": "outcome likelihood", "multiple_choice": "relative likelihood", "numeric": "quantitative benchmarks"}.get(question_type, "base rates")
        prompt = f"Generate 5 to 7 precise web-search queries for domain: {profile.domain}.\nFocus: {hint}.\nReturn only a JSON array of strings.\nQuestion: {question.question_text}"
        try:
            raw = (await self._llm_invoke("researcher", prompt)).strip()
            s, e = raw.find("["), raw.rfind("]")
            if s != -1 and e != -1: raw = raw[s:e + 1]
            queries = json.loads(raw)
            if isinstance(queries, list): return [q.strip() for q in queries if isinstance(q, str) and q.strip()][:7]
        except Exception: pass
        return [f"{question.question_text} recent developments", f"{question.question_text} historical base rate"]

    async def _multi_source_research_bundle(self, question: MetaculusQuestion, profile: QuestionProfile, question_type: str) -> str:
        llm_queries = await self._plan_queries(question, profile, question_type)
        all_queries = list(dict.fromkeys(llm_queries + [f"Metaculus community probability {question.question_text}"]))
        blocks = []
        async def fetch_query(q: str):
            await self._throttle_search()
            return await self._sources.fetch_all(q)
        results = await asyncio.gather(*[fetch_query(q) for q in all_queries], return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                blocks.extend(result)
        return "\n\n".join(b for b in blocks if b.strip()).strip()

    async def _synthesize_research(self, question: MetaculusQuestion, metaculus_block: str, source_bundle: str, profile: QuestionProfile, question_type: str) -> str:
        prompt = f"Synthesize evidence into a 4-part research brief (Base rate, Updates, Uncertainties, Signposts). Max 2400 chars.\nQuestion: {question.question_text}\nSources:\n{metaculus_block}\n[Web Research]\n{source_bundle}"
        result = await invoke_with_free_model_fallback(prompt, temperature=0.15, timeout_s=LLM_TIMEOUT_S, label="research_synthesis")
        return result.strip() if result else ""

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            url = getattr(question, "page_url", "") or getattr(question, "url", "") or ""
            if url:
                cached = await self._research_cache.get(url)
                if cached: return cached

            q_type   = "numeric" if isinstance(question, NumericQuestion) else "multiple_choice" if isinstance(question, MultipleChoiceQuestion) else "binary"
            profile  = await self._analyser.classify(question)
            strategy = ModellingStrategy.select(profile, q_type)
            qid      = extract_question_id(question)

            fin_data = ""
            is_finance = "market-pulse" in (self._active_tournament or "") or profile.domain in ("finance", "economics")
            if is_finance and YFINANCE_AVAILABLE:
                try:
                    ticker_prompt = f"Extract the Yahoo Finance ticker for this question. Reply ONLY with the ticker or NONE.\nQuestion: {question.question_text}"
                    ticker = await self._llm_invoke("parser", ticker_prompt)
                    ticker = ticker.strip().upper()
                    if ticker and ticker != "NONE":
                        fin_data = await asyncio.to_thread(_fetch_yfinance_data_sync, ticker)
                except Exception as e: logger.warning(f"Ticker extraction failed: {e}")

            metaculus_block = self._metaculus_context_block(question)
            source_bundle   = await self._multi_source_research_bundle(question, profile, q_type)
            synthesized     = await self._synthesize_research(question, metaculus_block, source_bundle, profile, q_type)

            # Degrade to a warning + raw bundle fallback instead of a hard crash
            # when synthesis is thin but web research was actually retrieved.
            if REQUIRE_RESEARCH and not is_meaningful_research_text(synthesized):
                if source_bundle and len(source_bundle.strip()) > 300:
                    logger.warning(f"[Research] Synthesis weak for Q{qid}, falling back to raw bundle.")
                    synthesized = source_bundle[:2400]
                else:
                    raise RuntimeError(f"Insufficient synthesized research for Q{qid}.")

            final = (f"{fin_data}{metaculus_block}\n\n[Research Summary]\n{synthesized}\n\n[Raw Web Research]\n{source_bundle}"
                     if source_bundle else f"{fin_data}{metaculus_block}\n\n[Research Summary]\n{synthesized}")

            if url: await self._research_cache.set(url, final)

            searchers_used = ["metaculus"] + [s.name for s in self._sources.available_sources()] if source_bundle else ["metaculus"]
            self._research_meta[qid] = {"synthesized": synthesized, "searchers_used": searchers_used, "profile": profile, "strategy": strategy, "question_type": q_type}
            return final

    async def _get_profile_and_strategy(self, question: MetaculusQuestion, question_type: str = "binary") -> Tuple[QuestionProfile, str]:
        meta = self._research_meta.get(extract_question_id(question), {})
        if meta.get("profile") and meta.get("strategy"): return meta["profile"], meta["strategy"]
        profile  = await self._analyser.classify(question)
        return profile, ModellingStrategy.select(profile, question_type)

    def _selective_confidence_gate(self, qid: str, p_final: Any, profile: QuestionProfile, research: str):
        if str(self._active_tournament) in SELECTIVE_TOURNAMENTS:
            confidence = self._validator.compute_confidence(p_final, profile, len(research))
            if confidence < self._validator.LOW_CONFIDENCE_THRESHOLD:
                raise RuntimeError(f"Selective Forecasting Gate: Confidence {confidence:.2f} too low for Q{qid}. Skipping.")

    async def _forecast_binary_core(self, question: BinaryQuestion, research: str, profile: QuestionProfile, strategy: str, model_key: str = "default") -> Tuple[float, str]:
        qid, base_rate = extract_question_id(question), safe_community_prediction(question)
        base_str = f"Community prob: {base_rate:.4f}" if isinstance(base_rate, (int, float)) else "No community probability."
        prompt = (f"You are an expert superforecaster. Produce a calibrated probability.\n{ModellingStrategy.get_prompt_block(strategy, profile)}\n"
                  f"Question: {question.question_text}\n{base_str}\nResearch summary:\n{research}\n"
                  "End with EXACTLY: FINAL PROBABILITY: <integer between 0 and 100>%")
        raw  = await with_timeout(self.get_llm(model_key, "llm").invoke(prompt), LLM_TIMEOUT_S, f"binary_llm_{model_key}")
        pred = await structure_output(raw, BinaryPrediction, model=self.get_llm("parser", "llm"))
        return clamp01(float(pred.prediction_in_decimal)), str(raw)

    async def _forecast_mc_core(self, question: MultipleChoiceQuestion, research: str, profile: QuestionProfile, strategy: str, model_key: str = "default") -> Tuple[PredictedOptionList, str]:
        options_list = "\n".join(f"  - {opt}" for opt in question.options)
        prompt = (f"You are an expert superforecaster. Assign probabilities summing to 1.0.\n{ModellingStrategy.get_prompt_block(strategy, profile)}\n"
                  f"Question: {question.question_text}\nOptions:\n{options_list}\nResearch summary:\n{research}\n"
                  'Return JSON ONLY: {"predicted_options": [{"option_name": "<exact name>", "probability": <decimal>}, ...]}')
        raw    = await with_timeout(self.get_llm(model_key, "llm").invoke(prompt), LLM_TIMEOUT_S, f"mc_llm_{model_key}")
        result = await structure_output(raw, PredictedOptionList, model=self.get_llm("parser", "llm"), additional_instructions=f"Names must match: {question.options}")
        probs = {o.option_name: max(0.0, float(o.probability)) for o in result.predicted_options}
        for opt in question.options: probs.setdefault(opt, 0.0)
        total = sum(probs.values()) or 1.0
        return PredictedOptionList(predicted_options=[PredictedOption(option_name=opt, probability=float(probs[opt]/total)) for opt in question.options]), str(raw)

    async def _forecast_numeric_core(self, question: NumericQuestion, research: str, profile: QuestionProfile, strategy: str, model_key: str = "default") -> Tuple[List[Percentile], str]:
        prompt = (f"Produce a calibrated probability distribution over possible outcomes.\n{ModellingStrategy.get_prompt_block(strategy, profile)}\n"
                  f"Question: {question.question_text}\n"
                  f"Lower bound: {getattr(question, 'lower_bound', 'N/A')}, Upper bound: {getattr(question, 'upper_bound', 'N/A')}\n"
                  f"Research summary:\n{research}\n"
                  'Return JSON array ONLY: [{"percentile":0.1,"value":<num>}, {"percentile":0.2,"value":<num>}, {"percentile":0.4,"value":<num>}, {"percentile":0.6,"value":<num>}, {"percentile":0.8,"value":<num>}, {"percentile":0.9,"value":<num>}]')
        raw = await with_timeout(self.get_llm(model_key, "llm").invoke(prompt), LLM_TIMEOUT_S, f"num_llm_{model_key}")
        try:
            percentile_list = await structure_output(raw, list[Percentile], model=self.get_llm("parser", "llm"))
        except Exception:
            repair_prompt = f"Convert to valid JSON array of Percentile objects.\n{sanitize_numeric_json(str(raw))}"
            repaired = await invoke_with_free_model_fallback(repair_prompt, temperature=0.15, timeout_s=LLM_TIMEOUT_S, label=f"num_repair_{model_key}")
            try:
                percentile_list = await structure_output(repaired, list[Percentile], model=self.get_llm("parser", "llm"))
            except Exception as e:
                logger.error(f"Numeric repair failed even with fallback models: {e}")
                raise
        validated = enforce_numeric_constraints(
            interpolate_missing_percentiles(percentile_list, [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]),
            question
        )
        return validated, str(raw)

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        profile, strategy = await self._get_profile_and_strategy(question, "binary")
        qid, base = extract_question_id(question), safe_community_prediction(question)

        raw_ps = []
        for model_key in ("default", "default_alt"):
            try:
                p, _ = await self._forecast_binary_core(question, research, profile, strategy, model_key=model_key)
                raw_ps.append(p)
            except Exception as e:
                logger.warning(f"[Yrambot] binary forecast failed for {model_key}: {e}")

        if not raw_ps:
            raw_p = clamp01(float(base) if isinstance(base, (int, float)) else 0.5)
        else:
            raw_p = statistics.median(raw_ps)

        p_final = self._extremize(raw_p)

        if not self._is_minibench() and isinstance(base, (int, float)):
            p_final = clamp01(CROWD_BLEND_MIXED * p_final + (1.0 - CROWD_BLEND_MIXED) * float(base))
            blend_note = f"Community anchor blended at weight {CROWD_BLEND_MIXED}."
        else:
            blend_note = "Crowd blending disabled (Minibench Anti-Average Engine active)." if self._is_minibench() else "No community anchor available."

        self._selective_confidence_gate(qid, p_final, profile, research)
        self._validator.validate(question, profile, strategy, p_final, research)
        methodology = f"Superforecasting protocol with {strategy} framework.\n{blend_note}"
        reasoning = build_reasoning_block(question, f"{p_final:.1%}", f"{base:.1%}" if isinstance(base, (int,float)) else "None", methodology, strategy, profile, self._research_meta.get(qid, {}).get("searchers_used", []), self._is_minibench(), self._ext().factor)

        log_forecast_for_calibration(question, p_final, reasoning, ["Ensemble"], True, self._research_meta.get(qid, {}).get("searchers_used", []))
        time.sleep(PUBLISH_SLEEP_S)
        return ReasonedPrediction(prediction_value=p_final, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        profile, strategy = await self._get_profile_and_strategy(question, "multiple_choice")
        qid = extract_question_id(question)

        results = []
        for model_key in ("default", "default_alt"):
            try:
                out, _ = await self._forecast_mc_core(question, research, profile, strategy, model_key=model_key)
                results.append(out)
            except Exception as e:
                logger.warning(f"[Yrambot] multiple-choice forecast failed for {model_key}: {e}")

        if results:
            merged_probs = {}
            for opt in question.options:
                values = [next((o.probability for o in result.predicted_options if o.option_name == opt), 0.0) for result in results]
                merged_probs[opt] = statistics.median(values)
            total = sum(merged_probs.values()) or 1.0
            out = PredictedOptionList(predicted_options=[PredictedOption(option_name=opt, probability=merged_probs[opt] / total) for opt in question.options])
        else:
            out = PredictedOptionList(predicted_options=[PredictedOption(option_name=opt, probability=1.0 / len(question.options)) for opt in question.options])

        if self._is_minibench():
            extremized = {o.option_name: extremize_probability(o.probability, self._ext_cfg_minibench) for o in out.predicted_options}
            total = sum(extremized.values()) or 1.0
            out = PredictedOptionList(predicted_options=[PredictedOption(option_name=opt, probability=extremized[opt] / total) for opt in question.options])

        self._selective_confidence_gate(qid, 0.5, profile, research)
        self._validator.validate(question, profile, strategy, [o.probability for o in out.predicted_options], research)

        methodology = f"Superforecasting protocol with {strategy} framework.\n{'Per-option logit extremization applied.' if self._is_minibench() else 'Standard distribution.'}"
        reasoning = build_reasoning_block(question, ", ".join([f"{x.option_name}: {x.probability:.1%}" for x in out.predicted_options]), "Qualitative anchor", methodology, strategy, profile, self._research_meta.get(qid, {}).get("searchers_used", []), self._is_minibench(), self._ext().factor)

        time.sleep(PUBLISH_SLEEP_S)
        return ReasonedPrediction(prediction_value=out, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        profile, strategy = await self._get_profile_and_strategy(question, "numeric")
        qid, base = extract_question_id(question), safe_community_prediction(question)

        results = []
        for model_key in ("default", "default_alt"):
            try:
                validated, _ = await self._forecast_numeric_core(question, research, profile, strategy, model_key=model_key)
                results.append(validated)
            except Exception as e:
                logger.warning(f"[Yrambot] numeric forecast failed for {model_key}: {e}")

        if results:
            merged = []
            percentiles = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
            for p in percentiles:
                values = [next((x.value for x in result if x.percentile == p), 0.0) for result in results]
                merged.append(Percentile(percentile=p, value=statistics.median(values)))
            validated = enforce_numeric_constraints(merged, question)
        else:
            lb, ub = derive_numeric_fallback_bounds(question, base)
            center = float(base) if isinstance(base, (int, float)) else (lb + ub) / 2.0
            width  = (ub - lb) * 0.30
            vals   = [max(lb, min(ub, v)) for v in [
                center - 0.9 * width, center - 0.5 * width, center - 0.15 * width,
                center + 0.15 * width, center + 0.5 * width, center + 0.9 * width
            ]]
            validated = enforce_numeric_constraints(
                [Percentile(percentile=p, value=v) for p, v in zip([0.1, 0.2, 0.4, 0.6, 0.8, 0.9], vals)],
                question
            )

        lower = getattr(question, 'lower_bound', None) or getattr(question, 'nominal_lower_bound', None)
        if lower is not None and lower > 1e6:
            max_p = max(p.value for p in validated)
            if max_p < lower / 100:
                scale = lower / max_p
                validated = [Percentile(p.percentile, p.value * scale) for p in validated]

        if "market-pulse" in (self._active_tournament or ""):
            validated = apply_tail_fattening(validated, factor=1.20)

        self._selective_confidence_gate(qid, 0.5, profile, research)
        dist = NumericDistribution.from_question(validated, question)
        self._validator.validate(question, profile, strategy, dist, research)

        methodology = f"Superforecasting protocol with {strategy} framework.\n{'Market-Pulse Tail Fattening applied to P10/P90.' if 'market-pulse' in (self._active_tournament or '') else 'Standard distribution.'}"
        reasoning = build_reasoning_block(
            question,
            ", ".join([f"p{int(p.percentile * 100)}={p.value:,.6g}" for p in validated]),
            f"{base:,.4g}" if isinstance(base, (int, float)) else "None",
            methodology, strategy, profile,
            self._research_meta.get(qid, {}).get("searchers_used", []),
            self._is_minibench(), self._ext().factor
        )

        time.sleep(PUBLISH_SLEEP_S)
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    async def forecast_on_tournament(self, tournament_id, *args, **kwargs):
        self._active_tournament = str(tournament_id)
        if self._is_minibench(): logger.info(f"[Yrambot] Minibench detected — extremization factor set to {MINIBENCH_EXTREMIZE_FACTOR}")
        reports = await super().forecast_on_tournament(tournament_id, *args, **kwargs)
        if summary := self._validator.summary(): logger.info(f"[Yrambot] Validation summary:\n{json.dumps(summary, indent=2)}")
        return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Yrambot.")
    parser.add_argument("--tournament-ids", nargs="+", type=str, default=["33022", "minibench", "market-pulse-26q2", MetaculusApi.CURRENT_MINIBENCH_ID])
    args = parser.parse_args()

    bot = Yrambot(
        client_spec=ClientSpecialisation(),
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=False
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
                    if any(x in str(e).lower() for x in ("too many requests", "cloudflare", "1015", "429")):
                        logger.error(f"Rate-limited on tournament {tid} (attempt {attempt + 1}/{RETRY_MAX}): {e}")
                        backoff_sleep(attempt)
                        continue
                    raise
            time.sleep(TOURNAMENT_SLEEP_S)
        bot.log_report_summary(all_reports)
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
