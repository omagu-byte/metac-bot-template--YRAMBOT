import argparse
import asyncio
import json
import logging
import os
import random
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import dotenv
import numpy as np

from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    ConditionalPrediction,
    ConditionalQuestion,
    DatePercentile,
    DateQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictionAffirmed,
    PredictionTypes,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


def _get_option_name(option: Any) -> str:
    if hasattr(option, "option_name"):
        return option.option_name
    if isinstance(option, dict):
        return option.get("option", option.get("option_name", ""))
    return ""


def _get_option_probability(option: Any) -> float:
    if hasattr(option, "probability"):
        return float(option.probability)
    if isinstance(option, dict):
        return float(option.get("probability", 0.0))
    return 0.0


TAVILY_API_KEY  = os.getenv("TAVILY_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

VULTR_API_BASE = os.getenv("VULTR_INFERENCE_API_BASE", "https://api.vultrinference.com/v1")
VULTR_API_KEY  = os.getenv("VULTR_SERVERLESS_INFERENCE_API_KEY", "")

# Vultr model IDs (openai/ prefix routes via LiteLLM OpenAI-compatible provider)
_MODEL_PRIMARY   = os.getenv("VULTR_MODEL_PRIMARY",   "openai/deepseek-r1-distill-llama-70b")
_MODEL_SECONDARY = os.getenv("VULTR_MODEL_SECONDARY", "openai/llama-3.3-70b-instruct-fp8")
_MODEL_PARSER    = os.getenv("VULTR_MODEL_PARSER",    "openai/qwen2.5-32b-instruct")


def _make_vultr_llm(model: str, *, temperature: float = 0.10, timeout: int = 90,
                    allowed_tries: int = 3) -> GeneralLlm:
    return GeneralLlm(
        model=model,
        temperature=temperature,
        timeout=timeout,
        allowed_tries=allowed_tries,
        api_key=VULTR_API_KEY,
        base_url=VULTR_API_BASE,
    )

DOMAINS = [
    "geopolitics", "economics", "technology", "science",
    "public_health", "environment", "sports", "finance", "social", "other",
]
GEO_SCOPES = ["global", "regional", "national", "local"]


@dataclass
class QuestionProfile:
    domain: str                  = "other"
    geo_scope: str               = "global"
    geography: str               = ""
    time_horizon_days: int       = 365
    is_quantitative: bool        = False
    confidence_in_profile: float = 0.0


class QuestionAnalyser:
    def __init__(self, llm: GeneralLlm):
        self._llm = llm

    async def classify(self, question: MetaculusQuestion) -> QuestionProfile:
        prompt = clean_indents(
            f"""
            Classify the following forecasting question. Reply ONLY with a JSON
            object matching this exact schema (no markdown, no extra keys):

            {{
              "domain": "<one of: {', '.join(DOMAINS)}>",
              "geo_scope": "<one of: {', '.join(GEO_SCOPES)}>",
              "geography": "<country/region name, or empty string if global>",
              "time_horizon_days": <integer, estimated days until resolution>,
              "is_quantitative": <true if the answer is a number or date, false otherwise>,
              "confidence_in_profile": <float 0.0-1.0>
            }}

            Question: {question.question_text}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            """
        )
        try:
            raw = await self._llm.invoke(prompt)
            raw = raw.strip()
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1:
                raw = raw[start : end + 1]
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
            logger.warning(f"[Analyser] Failed to classify question: {exc}")
            return QuestionProfile()


class ModellingStrategy:
    @staticmethod
    def select(profile: QuestionProfile) -> str:
        if profile.domain in ("economics", "finance") and profile.is_quantitative:
            return "trend"
        if profile.domain in ("geopolitics", "social"):
            return "analogical"
        if profile.time_horizon_days < 60:
            return "market_signal"
        return "base_rate"

    @staticmethod
    def get_prompt_block(strategy: str, profile: QuestionProfile) -> str:
        geo_ctx = f" focusing on {profile.geography}" if profile.geography else ""

        if strategy == "trend":
            return clean_indents(
                f"""
                ## Strategy: Trend Extrapolation{geo_ctx}
                1. Identify the key measurable variable.
                2. Find its recent trajectory (last 1-3 data points).
                3. Project forward to resolution date.
                4. Apply mean-reversion: trends rarely persist at full strength.
                5. Bound estimate with a realistic uncertainty range.
                """
            ).strip()

        if strategy == "analogical":
            return clean_indents(
                f"""
                ## Strategy: Analogical Reasoning{geo_ctx}
                1. Identify 2-3 structurally similar historical situations.
                2. How did those resolve? What was the base rate?
                3. Key SIMILARITIES – how they support your estimate.
                4. Key DIFFERENCES – how they require adjustment.
                5. Weight analogies by structural similarity, not surface resemblance.
                """
            ).strip()

        if strategy == "market_signal":
            return clean_indents(
                f"""
                ## Strategy: Market Signal{geo_ctx}
                1. Check prediction markets (Metaculus, Polymarket, Metaforecast).
                2. If a signal exists, treat it as a strong prior.
                3. Adjust only if you have concrete information it hasn't priced in.
                4. Short horizons: weight inertia very heavily.
                """
            ).strip()

        return clean_indents(
            f"""
            ## Strategy: Base Rate{geo_ctx}
            1. Define the reference class for this type of event.
            2. Historical frequency of the outcome in that class.
            3. Anchor to that base rate.
            4. Apply inside-view adjustments only for clear distinguishing features.
            5. Limit total adjustment from base rate to ±20 pp unless evidence is overwhelming.
            """
        ).strip()


class BaseSource(ABC):
    name: str = "unnamed_source"

    @abstractmethod
    async def fetch(self, query: str) -> str:
        ...

    def is_available(self) -> bool:
        return True


class SourceRegistry:
    def __init__(self):
        self._sources: list[BaseSource] = []

    def register(self, source: BaseSource) -> None:
        self._sources.append(source)
        logger.info(f"[SourceRegistry] Registered source: {source.name}")

    def available_sources(self) -> list[BaseSource]:
        return [s for s in self._sources if s.is_available()]

    async def fetch_all(self, query: str) -> list[str]:
        sources = self.available_sources()
        tasks   = [s.fetch(query) for s in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        blocks: list[str] = []
        for src, res in zip(sources, results):
            if isinstance(res, Exception):
                blocks.append(f"[{src.name}] Query failed: {res}")
            elif isinstance(res, str) and res.strip():
                blocks.append(f"[{src.name}]\n{res}")
        return blocks


def _format_tavily_results(query: str, results: dict[str, Any], max_results: int = 6) -> str:
    items = results.get("results", []) or []
    lines = [f"Query: {query}"]
    for r in items[:max_results]:
        title   = (r.get("title")       or "").strip()
        url     = (r.get("url")         or "").strip()
        snippet = (r.get("content")     or "").strip()
        raw     = (r.get("raw_content") or "").strip()
        if title or url or snippet:
            lines.append(f"- {title}")
            if url:
                lines.append(f"  URL: {url}")
            if snippet:
                lines.append(f"  Notes: {snippet}")
            if raw and raw != snippet:
                lines.append(f"  Full text (truncated): {raw[:1500]}")
    return "\n".join(lines).strip()


class TavilySearcher:
    def __init__(self, api_key: str, max_results: int = 6, search_depth: str = "advanced",
                 include_answer: bool = False, include_raw_content: bool = True,
                 include_images: bool = False, include_domains: list[str] | None = None,
                 exclude_domains: list[str] | None = None, timeout_s: int = 30):
        self.api_key             = api_key
        self.max_results         = max_results
        self.search_depth        = search_depth
        self.include_answer      = include_answer
        self.include_raw_content = include_raw_content
        self.include_images      = include_images
        self.include_domains     = include_domains
        self.exclude_domains     = exclude_domains
        self.timeout_s           = timeout_s

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req  = Request(url, data=data,
                       headers={"Content-Type": "application/json", "Accept": "application/json"},
                       method="POST")
        with urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    async def search(self, query: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "api_key": self.api_key, "query": query,
            "max_results": self.max_results, "search_depth": self.search_depth,
            "include_answer": self.include_answer, "include_raw_content": self.include_raw_content,
            "include_images": self.include_images,
        }
        if self.include_domains: payload["include_domains"] = self.include_domains
        if self.exclude_domains: payload["exclude_domains"] = self.exclude_domains
        return await asyncio.to_thread(self._post_json, "https://api.tavily.com/search", payload)


class TavilySource(BaseSource):
    name = "tavily_web"

    def __init__(self, api_key: str, include_domains: list[str] | None = None,
                 exclude_domains: list[str] | None = None):
        self._api_key  = api_key
        self._searcher = TavilySearcher(api_key=api_key, include_domains=include_domains,
                                        exclude_domains=exclude_domains) if api_key else None

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def fetch(self, query: str) -> str:
        if not self._searcher:
            return ""
        try:
            results = await self._searcher.search(query)
            return _format_tavily_results(query, results, self._searcher.max_results)
        except Exception as exc:
            return f"Query: {query}\n- Tavily failed: {type(exc).__name__}"


class ExaSource(BaseSource):
    name    = "exa_neural"
    _API_URL = "https://api.exa.ai/search"

    def __init__(self, api_key: str, num_results: int = 5, use_autoprompt: bool = True,
                 timeout_s: int = 30):
        self._api_key     = api_key
        self._num_results = num_results
        self._autoprompt  = use_autoprompt
        self._timeout_s   = timeout_s

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req  = Request(self._API_URL, data=data,
                       headers={"Content-Type": "application/json", "Accept": "application/json",
                                "x-api-key": self._api_key}, method="POST")
        with urlopen(req, timeout=self._timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    async def fetch(self, query: str) -> str:
        if not self._api_key:
            return ""
        try:
            payload = {"query": query, "numResults": self._num_results,
                       "useAutoprompt": self._autoprompt,
                       "contents": {"text": {"maxCharacters": 1500}}}
            raw     = await asyncio.to_thread(self._post_json, payload)
            results = raw.get("results", [])
            lines   = [f"Query: {query}"]
            for r in results:
                title   = (r.get("title") or "").strip()
                url     = (r.get("url")   or "").strip()
                excerpt = (r.get("text")  or "").strip()
                score   = r.get("score", 0.0)
                lines.append(f"- {title}  [score={score:.3f}]")
                if url:     lines.append(f"  URL: {url}")
                if excerpt: lines.append(f"  Excerpt: {excerpt[:1200]}")
            return "\n".join(lines).strip()
        except Exception as exc:
            return f"Query: {query}\n- Exa failed: {type(exc).__name__}: {exc}"


def _format_serpapi_results(query: str, results: dict[str, Any], max_results: int = 6) -> str:
    lines = [f"Query: {query}"]

    answer_box = results.get("answer_box") or {}
    if isinstance(answer_box, dict):
        answer = (answer_box.get("answer") or answer_box.get("snippet") or "").strip()
        if answer:
            lines.append(f"- Answer box: {answer}")

    knowledge_graph = results.get("knowledge_graph") or {}
    if isinstance(knowledge_graph, dict):
        kg_title = (knowledge_graph.get("title") or "").strip()
        kg_desc  = (knowledge_graph.get("description") or "").strip()
        if kg_title or kg_desc:
            lines.append(f"- Knowledge graph: {kg_title}")
            if kg_desc:
                lines.append(f"  Notes: {kg_desc[:1200]}")

    organic = results.get("organic_results") or []
    for r in organic[:max_results]:
        if not isinstance(r, dict):
            continue
        title   = (r.get("title")   or "").strip()
        url     = (r.get("link")    or "").strip()
        snippet = (r.get("snippet") or "").strip()
        date    = (r.get("date")    or "").strip()
        if title or url or snippet:
            lines.append(f"- {title}")
            if url:
                lines.append(f"  URL: {url}")
            if date:
                lines.append(f"  Date: {date}")
            if snippet:
                lines.append(f"  Notes: {snippet[:1200]}")

    return "\n".join(lines).strip()


class SerpApiSource(BaseSource):
    name     = "serpapi_google"
    _API_URL = "https://serpapi.com/search.json"

    def __init__(self, api_key: str, num_results: int = 6, timeout_s: int = 30):
        self._api_key     = api_key
        self._num_results = num_results
        self._timeout_s   = timeout_s

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _get_json(self, query: str) -> dict[str, Any]:
        params = urlencode({
            "engine":  "google",
            "q":       query,
            "api_key": self._api_key,
            "num":     self._num_results,
        })
        req = Request(f"{self._API_URL}?{params}",
                      headers={"Accept": "application/json"}, method="GET")
        with urlopen(req, timeout=self._timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    async def fetch(self, query: str) -> str:
        if not self._api_key:
            return ""
        try:
            raw = await asyncio.to_thread(self._get_json, query)
            if raw.get("error"):
                return f"Query: {query}\n- SerpAPI failed: {raw['error']}"
            return _format_serpapi_results(query, raw, self._num_results)
        except Exception as exc:
            return f"Query: {query}\n- SerpAPI failed: {type(exc).__name__}: {exc}"


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
                """CREATE TABLE IF NOT EXISTS forecast_ledger (
                    question_url TEXT, question_text TEXT, domain TEXT, geo_scope TEXT,
                    strategy TEXT, prediction_value TEXT, confidence_score REAL,
                    flagged INTEGER, ts REAL)""")
            conn.commit()

    def compute_confidence(self, prediction_value: Any, profile: QuestionProfile,
                           research_length: int) -> float:
        classifier_score = profile.confidence_in_profile
        evidence_score   = min(1.0, research_length / 3000)
        signal_score     = abs(prediction_value - 0.5) * 2 if isinstance(prediction_value, float) else 0.5
        score = 0.40 * classifier_score + 0.35 * evidence_score + 0.25 * signal_score
        return round(min(1.0, max(0.0, score)), 3)

    def validate(self, question: MetaculusQuestion, profile: QuestionProfile, strategy: str,
                 prediction_value: Any, research: str) -> ValidationRecord:
        confidence = self.compute_confidence(prediction_value, profile, len(research))
        flagged    = confidence < self.LOW_CONFIDENCE_THRESHOLD
        record = ValidationRecord(
            question_url=question.page_url, question_text=question.question_text[:300],
            domain=profile.domain, geo_scope=profile.geo_scope, strategy=strategy,
            prediction_value=str(prediction_value)[:200],
            confidence_score=confidence, flagged_low_confidence=flagged)
        self._persist(record)
        level = logging.WARNING if flagged else logging.INFO
        logger.log(level, f"[Validator] confidence={confidence:.2f} flagged={flagged} "
                          f"domain={profile.domain} strategy={strategy} | {question.page_url}")
        return record

    def _persist(self, record: ValidationRecord) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT INTO forecast_ledger (question_url, question_text, domain, geo_scope, "
                    "strategy, prediction_value, confidence_score, flagged, ts) VALUES (?,?,?,?,?,?,?,?,?)",
                    (record.question_url, record.question_text, record.domain, record.geo_scope,
                     record.strategy, record.prediction_value, record.confidence_score,
                     int(record.flagged_low_confidence), record.ts))
                conn.commit()
        except Exception as exc:
            logger.warning(f"[Validator] Persist failed: {exc}")

    def summary(self) -> dict[str, Any]:
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT domain, COUNT(*) as n, AVG(confidence_score) as avg_conf, "
                    "SUM(flagged) as n_flagged FROM forecast_ledger GROUP BY domain ORDER BY n DESC"
                ).fetchall()
            return {"by_domain": [{"domain": r[0], "n": r[1],
                                   "avg_confidence": round(r[2], 3), "n_flagged": r[3]} for r in rows]}
        except Exception:
            return {}


@dataclass
class ClientSpecialisation:
    """Optional configuration block for client-specific tuning. Inject at Yrambot construction time."""
    domain_focus:       list[str] = field(default_factory=list)
    trusted_domains:    list[str] = field(default_factory=list)
    excluded_domains:   list[str] = field(default_factory=list)
    extra_context:      str       = ""
    calibration_target: float     = 0.15


class ResearchCache:
    def __init__(self, db_path: str = "yrambot_cache.db"):
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS research_cache "
                "(url TEXT PRIMARY KEY, content TEXT NOT NULL, ts REAL NOT NULL)")
            conn.commit()

    def _get_sync(self, url: str) -> str | None:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute("SELECT content FROM research_cache WHERE url = ?", (url,)).fetchone()
        return row[0] if row else None

    def _set_sync(self, url: str, content: str) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO research_cache (url, content, ts) VALUES (?, ?, ?)",
                         (url, content, time.time()))
            conn.commit()

    async def get(self, url: str) -> str | None:
        return await asyncio.to_thread(self._get_sync, url)

    async def set(self, url: str, content: str) -> None:
        await asyncio.to_thread(self._set_sync, url, content)


class Yrambot(ForecastBot):
    """
    Yrambot – superforecaster bot with multi-API research (Tavily + Exa + SerpAPI).

    Forecasting LLMs: Vultr Serverless Inference (primary for 2 of 3 committee
    votes, secondary model for the third). All forecast prompts require grounding
    in the research block over the model's own priors (see _research_grounding_instruction).
    """

    _max_concurrent_questions            = 3
    _concurrency_limiter                 = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2
    _min_seconds_between_search_calls    = 1.2
    _min_seconds_between_llm_calls       = 0.50
    _last_search_call_ts                 = 0.0
    _last_llm_call_ts                    = 0.0

    def __init__(self, *args, client_spec: ClientSpecialisation | None = None, **kwargs):
        llms = kwargs.pop("llms", None)
        if llms is None:
            llms = {
                "default":    _make_vultr_llm(_MODEL_PRIMARY,   timeout=120),
                "summarizer": _make_vultr_llm(_MODEL_PRIMARY,   timeout=120),
                "researcher": _make_vultr_llm(_MODEL_PRIMARY,   timeout=90),
                "parser":     _make_vultr_llm(_MODEL_PARSER,    timeout=60),
            }
        super().__init__(*args, llms=llms, **kwargs)

        self._client_spec    = client_spec or ClientSpecialisation()
        self._research_cache = ResearchCache()
        self._validator      = ForecastValidator()
        self._analyser       = QuestionAnalyser(self.get_llm("researcher", "llm"))

        self._use_committee_voting  = True
        self._active_tournament_id: str = "minibench"

        self._sources  = SourceRegistry()
        tavily_key     = os.getenv("TAVILY_API_KEY", "").strip()
        self._sources.register(TavilySource(
            api_key=tavily_key,
            include_domains=self._client_spec.trusted_domains or None,
            exclude_domains=self._client_spec.excluded_domains or None,
        ))
        exa_key = os.getenv("EXA_API_KEY", "").strip()
        self._sources.register(ExaSource(api_key=exa_key))
        serpapi_key = (SERPAPI_API_KEY or "").strip()
        self._sources.register(SerpApiSource(api_key=serpapi_key))

    def register_source(self, source: BaseSource) -> None:
        self._sources.register(source)

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

    @staticmethod
    def _superforecasting_preamble() -> str:
        return clean_indents(
            """
            ## Superforecasting Protocol
            1. **Outside view first** – anchor to historical base rate for this class of event.
            2. **Inside view** – identify 2-3 causal drivers for and against the outcome.
            3. **Time horizon** – longer horizons regress to base rate; short horizons weight inertia.
            4. **Bias check** – flag availability bias, anchoring, overconfidence.
            5. **Disconfirmation** – what most strongly argues against your lean?
            6. **Synthesise** – blend views; adjust less than feels natural.
            7. **Calibration** – 50% = genuine uncertainty; 5%/95% only with overwhelming evidence.
            """
        ).strip()

    @staticmethod
    def _anti_hedging_instruction() -> str:
        return clean_indents(
            """
            ## Conviction requirement
            Do not hedge toward 50% out of caution or politeness.
            If your reasoning points clearly in one direction, commit to it.
            Answers like 48%–52% are only appropriate when evidence is genuinely
            balanced on both sides. Express your actual conviction based on the
            evidence — a well-reasoned 75% is better than a timid 53%.
            """
        ).strip()

    @staticmethod
    def _research_grounding_instruction(today_str: str) -> str:
        return clean_indents(
            f"""
            ## Research grounding requirement
            Your forecast must be driven primarily by the RESEARCH block below,
            not by your own prior beliefs about the topic. Concretely:
            1. Before reasoning, list the 2-4 most decision-relevant facts found
               in the research (status quo state, recent developments, any
               market/forecaster signal, relevant base rate data), and note the
               date of each fact where the research provides one.
            2. If the research contains a clear quantitative signal (e.g. a
               prediction market price, a community prediction, a poll, a
               survey), treat it as your primary anchor and only deviate from
               it if you can name a specific piece of research evidence that
               the signal hasn't priced in.
            3. If the research is thin or stale relative to the question's
               time horizon, say so explicitly and fall back to the base-rate
               strategy below rather than guessing.
            4. Do not introduce outside facts that are not present in the
               research and are not common, stable knowledge.

            ## Temporal anchoring requirement
            Today's actual date is {today_str}. Your training data has an earlier
            cutoff — do not assume the world is in the state your training data implies.
            1. Treat every dated fact in the RESEARCH block as more current and more
               authoritative than anything you remember from training.
            2. Do not silently default to a pre-cutoff baseline.
            3. Compute all durations relative to {today_str}.
            """
        ).strip()

    def call_tavily(self, query: str) -> str:
        if not TAVILY_API_KEY or not TavilyClient:
            return ""
        try:
            tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
            response = tavily_client.search(query=query, search_depth="advanced")
            return "\n".join([f"- {c['content']}" for c in response.get('results', [])])
        except Exception as e:
            return f"Tavily failed: {e}"

    async def _plan_queries(self, question: MetaculusQuestion, profile: QuestionProfile) -> list[str]:
        geo_hint  = f" (geography: {profile.geography})" if profile.geography else ""
        today_str = datetime.now().strftime("%Y-%m-%d")
        prompt = clean_indents(
            f"""
            Build a research plan for a {profile.domain} forecasting question{geo_hint}.
            Today's actual date is {today_str}.
            Return 4 to 6 web-search queries covering: base rates, key drivers,
            recent developments, timelines, expert opinion, prediction market signals.
            Output ONLY a JSON array of strings.

            Question: {question.question_text}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            """
        )
        try:
            raw = await self._llm_invoke("researcher", prompt)
            raw = raw.strip()
            s, e = raw.find("["), raw.rfind("]")
            if s != -1 and e != -1:
                raw = raw[s : e + 1]
            queries = json.loads(raw)
            if isinstance(queries, list):
                return [q.strip() for q in queries if isinstance(q, str) and q.strip()][:6]
        except Exception:
            pass
        return [
            f"{question.question_text} latest updates{geo_hint}",
            f"{question.question_text} base rate historical frequency",
            f"{question.question_text} prediction market probability",
        ]

    async def _multi_source_research_bundle(self, question: MetaculusQuestion,
                                            profile: QuestionProfile) -> str:
        llm_queries    = await self._plan_queries(question, profile)
        market_queries = [f"metaforecast {question.question_text}",
                          f"prediction market odds {question.question_text}"]
        seen: set[str] = set()
        all_queries: list[str] = []
        for q in llm_queries + market_queries:
            q2 = q.strip()
            if q2 and q2 not in seen:
                seen.add(q2); all_queries.append(q2)

        await self._throttle_search()
        results = await asyncio.gather(*[self._sources.fetch_all(q) for q in all_queries],
                                       return_exceptions=True)
        blocks: list[str] = []
        for q, res in zip(all_queries, results):
            if isinstance(res, Exception):
                blocks.append(f"[research] Query '{q}' failed: {type(res).__name__}: {res}")
                continue
            blocks.extend(res)
        return "\n\n".join(b for b in blocks if b.strip()).strip()

    def _format_metaculus_research(self, question: MetaculusQuestion) -> str:
        lines: list[str] = ["[Metaculus]"]
        if question.page_url:           lines.append(f"Question URL: {question.page_url}")
        if question.background_info:    lines.append(f"Background:\n{question.background_info.strip()}")
        if question.fine_print:         lines.append(f"Fine print:\n{question.fine_print.strip()}")
        if question.num_forecasters is not None: lines.append(f"Num forecasters: {question.num_forecasters}")
        if question.num_predictions is not None: lines.append(f"Num predictions: {question.num_predictions}")
        if question.close_time is not None:      lines.append(f"Close time: {question.close_time.isoformat()}")
        if question.published_time is not None:  lines.append(f"Published time: {question.published_time.isoformat()}")
        if question.open_time is not None:       lines.append(f"Open time: {question.open_time.isoformat()}")
        if question.cp_reveal_time is not None:  lines.append(f"Community prediction reveal time: {question.cp_reveal_time.isoformat()}")

        community_prediction = getattr(question, "community_prediction_at_access_time", None)
        if community_prediction is None:
            try:
                aggregations = (question.api_json.get("question", {}).get("aggregations", {})
                                if isinstance(question.api_json, dict) else {})
                community_prediction = (
                    aggregations.get("recency_weighted", {}).get("latest", {}).get("centers")
                    or aggregations.get("unweighted", {}).get("latest", {}).get("centers"))
                if isinstance(community_prediction, list) and len(community_prediction) == 1:
                    community_prediction = community_prediction[0]
                else:
                    community_prediction = None
            except Exception:
                community_prediction = None
        if community_prediction is not None:
            lines.append(f"Community prediction: {community_prediction}")
        return "\n".join(line for line in lines if line is not None and str(line).strip()).strip()

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            cached = await self._research_cache.get(question.page_url)
            if cached:
                return cached

            profile  = await self._analyser.classify(question)
            strategy = ModellingStrategy.select(profile)
            logger.info(f"[Yrambot] '{question.question_text[:60]}…' → "
                        f"domain={profile.domain} geo={profile.geography or 'global'} strategy={strategy}")

            base = clean_indents(
                f"""
                Question: {question.question_text}
                Resolution criteria: {question.resolution_criteria}
                Fine print: {question.fine_print}
                """
            ).strip()

            source_bundle   = await self._multi_source_research_bundle(question, profile)
            metaculus_block = self._format_metaculus_research(question)
            research_raw    = (
                f"{base}\n\n--- MULTI-SOURCE RESEARCH (Metaculus / Tavily / Exa / SerpAPI) ---\n"
                f"{metaculus_block}\n\n{source_bundle}"
                if source_bundle else f"{base}\n\n--- Metaculus research ---\n{metaculus_block}"
            )

            today_str = datetime.now().strftime("%Y-%m-%d")
            summarize_prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster working on a {profile.domain} question
                (geography: {profile.geography or 'global'}).
                Today's actual date is {today_str}. Research dated at or near {today_str} is
                the current ground truth. Preserve explicit dates in your summary.
                Summarise the most relevant evidence. Be concise but information-dense.
                Cover: status quo, key drivers, base rates, timelines, market probabilities.

                {research_raw}
                """
            )
            try:
                summary = await self._llm_invoke("summarizer", summarize_prompt)
                final = clean_indents(
                    f"""
                    {base}

                    --- RESEARCH SUMMARY ---
                    {summary}

                    --- RAW RESEARCH ---
                    {source_bundle}
                    """
                ).strip() if source_bundle else f"{base}\n\n--- RESEARCH SUMMARY ---\n{summary}"
            except Exception:
                final = research_raw

            await self._research_cache.set(question.page_url, final)
            return final

    async def _get_profile_and_strategy(self, question: MetaculusQuestion) -> tuple[QuestionProfile, str]:
        profile  = await self._analyser.classify(question)
        strategy = ModellingStrategy.select(profile)
        return profile, strategy

    def _client_context_block(self) -> str:
        if self._client_spec.extra_context:
            return f"\n## Client Context\n{self._client_spec.extra_context}\n"
        return ""

    async def _run_forecast_on_binary(self, question: BinaryQuestion,
                                      research: str) -> ReasonedPrediction[float]:
        profile, strategy = await self._get_profile_and_strategy(question)

        if self._use_committee_voting:
            forecasts, reasonings = [], []
            for i in range(3):
                try:
                    pred, reason = await self._single_forecast(question, research, use_claude=(i == 2))
                    if pred is not None and isinstance(pred, float):
                        forecasts.append(pred); reasonings.append(reason)
                except Exception as e:
                    logger.warning(f"[Yrambot] Committee member {i} failed: {e}")
            if not forecasts:
                logger.warning(f"[Yrambot] All committee members failed for {question.page_url}")
                return ReasonedPrediction(prediction_value=0.5, reasoning="Committee failed; returning 50% prior.")
            median_pred = float(np.median(forecasts))
            logger.info(f"[Yrambot] Committee binary: votes={forecasts} → median={median_pred:.3f}")
            result = ReasonedPrediction(prediction_value=median_pred, reasoning=" | ".join(reasonings))
        else:
            today_str = datetime.now().strftime("%Y-%m-%d")
            prompt = clean_indents(
                f"""
                You are Yrambot, a professional superforecaster.
                {self._client_context_block()}
                {self._superforecasting_preamble()}
                {self._anti_hedging_instruction()}
                {self._research_grounding_instruction(today_str)}
                {ModellingStrategy.get_prompt_block(strategy, profile)}
                ---
                Question: {question.question_text}
                Background: {question.background_info}
                Resolution criteria (not yet satisfied): {question.resolution_criteria}
                {question.fine_print}
                Research: {research}
                Today is {today_str}.
                Write exactly 3 paragraphs in first person as Yrambot, summarizing the key logic
                that informed this forecast. Do not mention any models, search sources, or research methods.
                {self._get_conditional_disclaimer_if_necessary(question)}
                End with: "Probability: ZZ%" (0-100)
                """
            )
            try:
                reasoning = await self._llm_invoke("default", prompt)
            except Exception as exc:
                logger.warning(f"[Yrambot] LLM failed for {question.page_url}: {exc}. Returning 50% prior.")
                return ReasonedPrediction(prediction_value=0.5, reasoning="LLM failed; returning uninformative prior.")
            logger.info(f"[Yrambot] Reasoning for {question.page_url}: {reasoning}")
            binary_prediction: BinaryPrediction = await structure_output(
                reasoning, BinaryPrediction, model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples)
            raw_p  = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))
            logger.info(f"[Yrambot] Forecast: p={raw_p:.3f} (tournament={self._active_tournament_id or 'unknown'})")
            result = ReasonedPrediction(prediction_value=raw_p, reasoning=reasoning)

        self._validator.validate(question, profile, strategy, result.prediction_value, research)
        return result

    async def _single_forecast(
        self,
        question: BinaryQuestion | MultipleChoiceQuestion | NumericQuestion,
        research: str,
        use_claude: bool = False,
    ) -> tuple[Any, str]:
        """Single committee vote. use_claude=True routes through _MODEL_SECONDARY."""
        if use_claude:
            original_default = self._llms.get("default")
            original_parser  = self._llms.get("parser")
            self._llms["default"] = _make_vultr_llm(_MODEL_SECONDARY, timeout=90)
            self._llms["parser"]  = _make_vultr_llm(_MODEL_PARSER,    timeout=60)

        today_str = datetime.now().strftime("%Y-%m-%d")
        reasoning = ""
        try:
            if isinstance(question, BinaryQuestion):
                prompt = clean_indents(f"""
                You are a professional forecaster.
                {self._research_grounding_instruction(today_str)}
                Question: {question.question_text}
                Background: {question.background_info}
                Resolution: {question.resolution_criteria}
                Fine print: {question.fine_print}
                Research: {research}
                Today: {today_str}
                Write analysis, then end with: "Probability: ZZ%"
                """)
                reasoning = await self._llm_invoke("default", prompt)
                pred: BinaryPrediction = await structure_output(reasoning, BinaryPrediction,
                                                                model=self.get_llm("parser", "llm"))
                result = max(0.01, min(0.99, pred.prediction_in_decimal))

            elif isinstance(question, MultipleChoiceQuestion):
                prompt = clean_indents(f"""
                {self._research_grounding_instruction(today_str)}
                Question: {question.question_text}
                Options: {question.options}
                Background: {question.background_info}
                Resolution: {question.resolution_criteria}
                Research: {research}
                Today: {today_str}
                Write analysis, then list probabilities for each option in order.
                """)
                reasoning = await self._llm_invoke("default", prompt)
                result = await structure_output(reasoning, PredictedOptionList,
                                                model=self.get_llm("parser", "llm"),
                                                additional_instructions=f"Options must be exactly: {question.options}")

            elif isinstance(question, NumericQuestion):
                lower_msg = f"Lower bound: {'open' if question.open_lower_bound else 'closed'} at {question.lower_bound or question.nominal_lower_bound}"
                upper_msg = f"Upper bound: {'open' if question.open_upper_bound else 'closed'} at {question.upper_bound or question.nominal_upper_bound}"
                prompt = clean_indents(f"""
                {self._research_grounding_instruction(today_str)}
                Question: {question.question_text}
                Units: {question.unit_of_measure or 'Infer from context'}
                Background: {question.background_info}
                Resolution: {question.resolution_criteria}
                {lower_msg}
                {upper_msg}
                Research: {research}
                Today: {today_str}
                Write analysis, then provide percentiles: 10, 20, 40, 60, 80, 90.
                """)
                reasoning = await self._llm_invoke("default", prompt)
                percentile_list: list[Percentile] = await structure_output(
                    reasoning, list[Percentile], model=self.get_llm("parser", "llm"))
                result = NumericDistribution.from_question(percentile_list, question)
            else:
                result = None
        finally:
            if use_claude:
                self._llms["default"] = original_default or _make_vultr_llm(_MODEL_PRIMARY, timeout=120)
                self._llms["parser"]  = original_parser  or _make_vultr_llm(_MODEL_PARSER,    timeout=60)

        return result, reasoning

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion,
                                               research: str) -> ReasonedPrediction[PredictedOptionList]:
        profile, strategy = await self._get_profile_and_strategy(question)

        if self._use_committee_voting:
            forecasts, reasonings = [], []
            for i in range(3):
                try:
                    pred, reason = await self._single_forecast(question, research, use_claude=(i == 2))
                    if pred is not None and isinstance(pred, PredictedOptionList):
                        forecasts.append(pred); reasonings.append(reason)
                except Exception as e:
                    logger.warning(f"[Yrambot] Committee member {i} failed: {e}")
            if not forecasts:
                logger.warning(f"[Yrambot] All committee members failed for {question.page_url}")
                return ReasonedPrediction(prediction_value=PredictedOptionList([]), reasoning="Committee failed")
            all_probs = np.array([
                [_get_option_probability(opt) for opt in forecast.predicted_options]
                for forecast in forecasts
            ])
            median_probs = np.median(all_probs, axis=0)
            median_probs = median_probs / median_probs.sum() if median_probs.sum() > 0 else np.full_like(median_probs, 1.0 / len(median_probs))
            options = forecasts[0].predicted_options
            median_forecast = PredictedOptionList([
                {"option_name": _get_option_name(opt), "probability": float(p)}
                for opt, p in zip(options, median_probs)
            ])
            logger.info(f"[Yrambot] Committee multiple_choice: median_probs={list(median_probs)}")
            result = ReasonedPrediction(prediction_value=median_forecast, reasoning=" | ".join(reasonings))
        else:
            today_str = datetime.now().strftime("%Y-%m-%d")
            prompt = clean_indents(
                f"""
                You are Yrambot, a professional superforecaster aiming for accurate forecasts and high scores on metaculus.
                {self._client_context_block()}
                {self._superforecasting_preamble()}
                {self._anti_hedging_instruction()}
                {self._research_grounding_instruction(today_str)}
                {ModellingStrategy.get_prompt_block(strategy, profile)}
                ---
                Question: {question.question_text}
                Options: {question.options}
                Background: {question.background_info}
                Resolution criteria: {question.resolution_criteria}
                {question.fine_print}
                Research: {research}
                Today is {today_str}.
                Write exactly 3 paragraphs in first person as Yrambot, summarizing the key logic
                that informed this forecast. Do not mention any models, search sources, or research methods.
                {self._get_conditional_disclaimer_if_necessary(question)}
                Avoid 0% unless logically impossible.
                End with probabilities in this exact order {question.options}:
                Option_A: Probability_A ...
                """
            )
            reasoning = await self._llm_invoke("default", prompt)
            logger.info(f"[Yrambot] Reasoning for {question.page_url}: {reasoning}")
            predicted_option_list: PredictedOptionList = await structure_output(
                text_to_structure=reasoning, output_type=PredictedOptionList,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
                additional_instructions=f"Option names must match one of: {question.options}. Do not drop any option.")
            result = ReasonedPrediction(prediction_value=predicted_option_list, reasoning=reasoning)

        self._validator.validate(question, profile, strategy, result.prediction_value, research)
        return result

    async def _run_forecast_on_numeric(self, question: NumericQuestion,
                                       research: str) -> ReasonedPrediction[NumericDistribution]:
        profile, strategy = await self._get_profile_and_strategy(question)
        upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)

        if self._use_committee_voting:
            forecasts, reasonings = [], []
            for i in range(3):
                try:
                    pred, reason = await self._single_forecast(question, research, use_claude=(i == 2))
                    if pred is not None and isinstance(pred, NumericDistribution):
                        forecasts.append(pred); reasonings.append(reason)
                except Exception as e:
                    logger.warning(f"[Yrambot] Committee member {i} failed: {e}")
            if not forecasts:
                logger.warning(f"[Yrambot] All committee members failed for {question.page_url}")
                return ReasonedPrediction(prediction_value=NumericDistribution([]), reasoning="Committee failed")
            target_percentiles = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
            aggregated = []
            for p in target_percentiles:
                values = []
                for f in forecasts:
                    for item in f.declared_percentiles:
                        if abs(item.percentile - p) < 0.01:
                            values.append(item.value); break
                    else:
                        values.append(0.0)
                aggregated.append(Percentile(percentile=p, value=float(np.median(values)) if values else 0.0))
            distribution = NumericDistribution.from_question(aggregated, question)
            logger.info(f"[Yrambot] Committee numeric: aggregated percentiles from {len(forecasts)} votes")
            result = ReasonedPrediction(prediction_value=distribution, reasoning=" | ".join(reasonings))
        else:
            today_str = datetime.now().strftime("%Y-%m-%d")
            prompt = clean_indents(
                f"""
                You are Yrambot, a professional superforecaster.
                {self._client_context_block()}
                {self._superforecasting_preamble()}
                {self._research_grounding_instruction(today_str)}
                {ModellingStrategy.get_prompt_block(strategy, profile)}
                ---
                Question: {question.question_text}
                Background: {question.background_info}
                {question.resolution_criteria}
                {question.fine_print}
                Units: {question.unit_of_measure if question.unit_of_measure else "Not stated (infer)"}
                Research: {research}
                Today is {today_str}.
                {lower_msg}
                {upper_msg}
                Formatting: no scientific notation; percentiles strictly increasing.
                Write exactly 3 paragraphs in first person as Yrambot, summarizing the key logic
                that informed this forecast. Do not mention any models, search sources, or research methods.
                {self._get_conditional_disclaimer_if_necessary(question)}
                End with:
                Percentile 10: XX  Percentile 20: XX  Percentile 40: XX
                Percentile 60: XX  Percentile 80: XX  Percentile 90: XX
                """
            )
            reasoning = await self._llm_invoke("default", prompt)
            logger.info(f"[Yrambot] Reasoning for {question.page_url}: {reasoning}")
            percentile_list: list[Percentile] = await structure_output(
                reasoning, list[Percentile], model=self.get_llm("parser", "llm"),
                additional_instructions=(
                    f'Parse a numeric percentile forecast for: "{question.question_text}"\n'
                    f"Units: {question.unit_of_measure}. Convert units if needed."),
                num_validation_samples=self._structure_output_validation_samples)
            result = ReasonedPrediction(
                prediction_value=NumericDistribution.from_question(percentile_list, question),
                reasoning=reasoning)

        self._validator.validate(question, profile, strategy, result.prediction_value, research)
        return result

    async def _run_forecast_on_date(self, question: DateQuestion,
                                    research: str) -> ReasonedPrediction[NumericDistribution]:
        profile, strategy = await self._get_profile_and_strategy(question)
        upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)
        today_str = datetime.now().strftime("%Y-%m-%d")
        prompt = clean_indents(
            f"""
            You are Yrambot, a professional superforecaster.
            {self._client_context_block()}
            {self._superforecasting_preamble()}
            {self._research_grounding_instruction(today_str)}
            {ModellingStrategy.get_prompt_block(strategy, profile)}
            ---
            Question: {question.question_text}
            Background: {question.background_info}
            {question.resolution_criteria}
            {question.fine_print}
            Research: {research}
            Today is {today_str}.
            {lower_msg}
            {upper_msg}
            Formatting: dates as YYYY-MM-DD; percentiles chronological and strictly increasing.
            Write exactly 3 paragraphs in first person as Yrambot, summarizing the key logic
            that informed this forecast. Do not mention any models, search sources, or research methods.
            {self._get_conditional_disclaimer_if_necessary(question)}
            End with:
            Percentile 10: YYYY-MM-DD  Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD  Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD  Percentile 90: YYYY-MM-DD
            """
        )
        result = await self._date_prompt_to_forecast(question, prompt)
        self._validator.validate(question, profile, strategy, result.prediction_value, research)
        return result

    async def _date_prompt_to_forecast(self, question: DateQuestion,
                                       prompt: str) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self._llm_invoke("default", prompt)
        logger.info(f"[Yrambot] Reasoning for {question.page_url}: {reasoning}")
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning, list[DatePercentile], model=self.get_llm("parser", "llm"),
            additional_instructions=(
                f'Parse a date percentile forecast for: "{question.question_text}"\n'
                "Assume midnight UTC if no time given."),
            num_validation_samples=self._structure_output_validation_samples)
        percentile_list = [Percentile(percentile=p.percentile, value=p.value.timestamp())
                           for p in date_percentile_list]
        return ReasonedPrediction(
            prediction_value=NumericDistribution.from_question(percentile_list, question),
            reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            upper = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
            lower = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
            unit  = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper = question.upper_bound.date().isoformat()
            lower = question.lower_bound.date().isoformat()
            unit  = ""
        else:
            raise ValueError()
        upper_msg = (f"The question creator thinks the value is likely not higher than {upper} {unit}."
                     if question.open_upper_bound else
                     f"The outcome cannot be higher than {upper} {unit}.")
        lower_msg = (f"The question creator thinks the value is likely not lower than {lower} {unit}."
                     if question.open_lower_bound else
                     f"The outcome cannot be lower than {lower} {unit}.")
        return upper_msg, lower_msg

    async def _run_forecast_on_conditional(self, question: ConditionalQuestion,
                                           research: str) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(question.parent,       research,      "parent")
        child_info,  full_research = await self._get_question_prediction_info(question.child,        full_research, "child")
        yes_info,    full_research = await self._get_question_prediction_info(question.question_yes, full_research, "yes")
        no_info,     full_research = await self._get_question_prediction_info(question.question_no,  full_research, "no")
        full_reasoning = clean_indents(
            f"## Parent Reasoning\n{parent_info.reasoning}\n"
            f"## Child Reasoning\n{child_info.reasoning}\n"
            f"## Yes Reasoning\n{yes_info.reasoning}\n"
            f"## No Reasoning\n{no_info.reasoning}").strip()
        return ReasonedPrediction(
            reasoning=full_reasoning,
            prediction_value=ConditionalPrediction(
                parent=parent_info.prediction_value, child=child_info.prediction_value,
                prediction_yes=yes_info.prediction_value, prediction_no=no_info.prediction_value))

    async def _get_question_prediction_info(self, question: MetaculusQuestion, research: str,
                                            question_type: str):
        from forecasting_tools.data_models.data_organizer import DataOrganizer
        previous_forecasts = question.previous_forecasts
        if (question_type in ["parent", "child"] and previous_forecasts
                and question_type not in self.force_reforecast_in_conditional):
            pf = previous_forecasts[-1]
            if pf.timestamp_end is None or pf.timestamp_end > datetime.now(timezone.utc):
                return (ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Reaffirmed at {DataOrganizer.get_readable_prediction(pf)}."), research)
        info = await self._make_prediction(question, research)
        return info, self._add_reasoning_to_research(research, info, question_type)

    def _add_reasoning_to_research(self, research: str, reasoning, question_type: str) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer
        qt = question_type.title()
        return clean_indents(
            f"{research}\n---\n## {qt} Question Information\n"
            f"Previously forecasted to: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}\n"
            f"Reasoning:\n```\n{reasoning.reasoning}\n```\n"
            f"Do NOT re-forecast the {qt} question.").strip()

    def _get_conditional_disclaimer_if_necessary(self, question: MetaculusQuestion) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return "Forecast ONLY the CHILD question given the parent's resolution. Do not re-forecast the parent."

    async def forecast_on_tournament(self, *args, **kwargs):
        reports = await super().forecast_on_tournament(*args, **kwargs)
        if isinstance(reports, list):
            summary = self._validator.summary()
            if summary:
                logger.info(f"[Yrambot] Validation summary:\n{json.dumps(summary, indent=2)}")
        return reports

    async def forecast_questions(self, *args, **kwargs):
        reports = await super().forecast_questions(*args, **kwargs)
        if isinstance(reports, list):
            summary = self._validator.summary()
            if summary:
                logger.info(f"[Yrambot] Validation summary:\n{json.dumps(summary, indent=2)}")
        return reports


async def _run_tournament(bot: Yrambot, tournament_id: str | int) -> list[Any]:
    bot._active_tournament_id = str(tournament_id)
    logger.info(f"[Yrambot] Starting tournament={tournament_id}")
    return await bot.forecast_on_tournament(tournament_id, return_exceptions=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").propagate = False

    parser = argparse.ArgumentParser(description="Run Yrambot")
    parser.add_argument("--mode", type=str,
                        choices=["tournament", "metaculus_cup", "test_questions"],
                        default="tournament")
    parser.add_argument("--tournament-ids", nargs="+", type=str, default=None,
                        help="Tournament IDs to forecast on (for tournament mode)")
    parser.add_argument("--use-committee", action=argparse.BooleanOptionalAction, default=True,
                        help="Use committee voting (primary x2 + secondary) with median aggregation")
    args    = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    spec = ClientSpecialisation(domain_focus=[], trusted_domains=[], excluded_domains=[],
                                extra_context="", calibration_target=0.15)

    bot = Yrambot(
        client_spec=spec,
        research_reports_per_question=1,
        predictions_per_research_report=1 if args.use_committee else 3,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=False,
    )
    bot._use_committee_voting = args.use_committee
    client = MetaculusClient()

    if run_mode == "tournament":
        if args.tournament_ids is None:
            args.tournament_ids = ["market-pulse-26q3", "33022", client.CURRENT_MINIBENCH_ID]
        all_reports = []
        for tid in args.tournament_ids:
            logger.info(f"Forecasting on tournament: {tid}")
            reports = asyncio.run(_run_tournament(bot, tid))
            all_reports.extend(reports if isinstance(reports, list) else [reports])

    elif run_mode == "metaculus_cup":
        bot.skip_previously_forecasted_questions = False
        bot._active_tournament_id = str(client.CURRENT_METACULUS_CUP_ID)
        all_reports = asyncio.run(
            bot.forecast_on_tournament(client.CURRENT_METACULUS_CUP_ID, return_exceptions=True))

    elif run_mode == "test_questions":
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
        ]
        bot.skip_previously_forecasted_questions = True
        bot._active_tournament_id = "test"
        questions   = [client.get_question_by_url(u) for u in EXAMPLE_QUESTIONS]
        all_reports = asyncio.run(bot.forecast_questions(questions, return_exceptions=True))

    try:
        bot.log_report_summary(all_reports)
        logger.info("Run completed successfully.")
    except Exception as e:
        logger.error(f"Failed to log report summary: {e}")
        logger.error(f"Total reports: {len(all_reports) if 'all_reports' in locals() else 0}")
