import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import List, Optional

from pydantic import model_validator

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

# pip install tavily-python exa-py
try:
    from tavily import AsyncTavilyClient
    TAVILY_AVAILABLE = bool(os.environ.get("TAVILY_API_KEY"))
except ImportError:
    TAVILY_AVAILABLE = False

try:
    from exa_py import Exa
    EXA_AVAILABLE = bool(os.environ.get("EXA_API_KEY"))
except ImportError:
    EXA_AVAILABLE = False


def median(lst: List[float]) -> float:
    if not lst:
        raise ValueError("median() arg is an empty sequence")
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2.0
    else:
        return float(sorted_lst[mid])


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Yrambot")


# ------------------------------------------------------------------
# Stand-alone research helpers (called before LLM synthesis)
# ------------------------------------------------------------------

async def _research_tavily(query: str) -> Optional[str]:
    """Tavily: LLM-optimized search, returns scored snippets + AI answer."""
    if not TAVILY_AVAILABLE:
        return None
    try:
        client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        resp = await client.search(
            query=query,
            search_depth="advanced",   # deeper crawl, worth the cost for forecasting
            max_results=6,
            include_answer=True,       # ask Tavily to synthesize an answer too
        )
        lines = []
        if resp.get("answer"):
            lines.append(f"Tavily answer: {resp['answer']}\n")
        for r in resp.get("results", []):
            score = r.get("score", 0)
            if score >= 0.4:           # filter low-relevance noise
                lines.append(f"[{score:.2f}] {r['title']}\n{r['url']}\n{r.get('content','')}\n")
        return "\n".join(lines) if lines else None
    except Exception as e:
        logger.warning(f"Tavily failed: {e}")
        return None


async def _research_exa(query: str) -> Optional[str]:
    """Exa: neural/semantic search — strong on multi-hop research queries."""
    if not EXA_AVAILABLE:
        return None
    try:
        # exa_py is sync; run in thread to avoid blocking the event loop
        exa = Exa(api_key=os.environ["EXA_API_KEY"])
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: exa.search_and_contents(
                query,
                num_results=5,
                use_autoprompt=True,   # Exa rewrites query for better recall
                text={"max_characters": 1500},
            )
        )
        lines = []
        for r in resp.results:
            lines.append(f"{r.title}\n{r.url}\n{r.text}\n")
        return "\n".join(lines) if lines else None
    except Exception as e:
        logger.warning(f"Exa failed: {e}")
        return None


async def _research_sonar(llm, query: str, today_str: str) -> Optional[str]:
    """Perplexity Sonar-Pro fallback — LLM with built-in live search."""
    try:
        prompt = clean_indents(f"""
            You are a research assistant with live web-search capability.
            Today is {today_str}.

            {query}

            Be factual. Cite sources where possible.
            Output a concise research summary for a professional forecaster.
        """)
        return await llm.invoke(prompt)
    except Exception as e:
        logger.warning(f"Sonar-Pro fallback failed: {e}")
        return None


class Yrambot(ForecastBot):
    """
    Conservative hybrid forecaster.
    Research chain: Tavily → Exa → xiaomi/mimo-v2.5 (first success wins).
    Committee: GPT-5.1, GPT-5.4, xiaomi/mimo-v2.5o.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default":    "openrouter/openai/gpt-5.5",
            "parser":     "openrouter/openai/gpt-4.1-mini",
            "researcher": "openrouter/xiaomi/mimo-v2.5",   # Sonar as last-resort researcher
            "summarizer": "openrouter/openai/gpt-4.1-mini",
        }

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            today_str = datetime.now().strftime("%Y-%m-%d")

            # Build a rich query string from question fields
            query = clean_indents(f"""
                Forecasting question (resolve by analyzing current events):
                {question.question_text}

                Background: {question.background_info or 'None'}
                Resolution criteria: {question.resolution_criteria or 'Standard'}
                Fine print: {question.fine_print or 'None'}
                Today: {today_str}

                Find: current status, recent news, scheduled events before resolution,
                base rates, and any data that affects the probability of this outcome.
            """)

            raw_search: Optional[str] = None
            source_label = "unknown"

            # ── Tier 1: Tavily ──────────────────────────────────────────
            if TAVILY_AVAILABLE:
                logger.info("Research tier 1: Tavily")
                raw_search = await _research_tavily(query)
                if raw_search:
                    source_label = "Tavily (advanced)"

            # ── Tier 2: Exa ─────────────────────────────────────────────
            if raw_search is None and EXA_AVAILABLE:
                logger.info("Research tier 2: Exa")
                raw_search = await _research_exa(query)
                if raw_search:
                    source_label = "Exa (neural search)"

            # ── Tier 3: Perplexity Sonar-Pro ────────────────────────────
            if raw_search is None:
                logger.info("Research tier 3: Perplexity Sonar-Pro (LLM fallback)")
                raw_search = await _research_sonar(
                    self.get_llm("researcher", "llm"), query, today_str
                )
                if raw_search:
                    source_label = "Perplexity Sonar-Pro"

            # ── Total failure ────────────────────────────────────────────
            if raw_search is None:
                raw_search = (
                    "[All research providers failed. "
                    "Proceeding with model prior knowledge only.]"
                )
                source_label = "none"
                logger.error("All research tiers failed.")

            # ── Synthesize with GPT summarizer ───────────────────────────
            synthesis_prompt = clean_indents(f"""
                You are a professional forecasting researcher.
                Today is {today_str}.

                Below are raw search results for this forecasting question:
                Question: {question.question_text}

                --- RAW SEARCH RESULTS ---
                {raw_search}
                --- END ---

                Synthesize into a concise research brief covering:
                1. Current status of the topic (as of {today_str})
                2. Recent developments that shift the probability
                3. Scheduled events before resolution that matter
                4. Relevant base rates or historical analogues
                5. Key uncertainties

                Be factual. Flag any contradictions. Keep it under 400 words.
            """)

            try:
                synthesis = await self.get_llm("summarizer", "llm").invoke(synthesis_prompt)
            except Exception as e:
                synthesis = raw_search   # fallback: just use raw results
                logger.warning(f"Synthesis step failed, using raw results: {e}")

            return (
                f"--- RESEARCH (source: {source_label}, as of {today_str}) ---\n"
                f"{synthesis}\n"
            )

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(f"""
            You are a professional forecaster known for conservative, well-calibrated predictions.
            Today is {datetime.now().strftime('%Y-%m-%d')}.

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            Research: {research}

            Consider:
            (a) Time until resolution
            (b) Status quo bias — the world changes slowly
            (c) Base rates
            (d) Any conflicts or uncertainty in the research

            Be humble. Avoid overconfidence.

            The last thing you write is your final answer as: "Probability: ZZ%", 0–100
        """)
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        binary_pred: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        decimal_pred = max(0.01, min(0.99, binary_pred.prediction_in_decimal))
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(f"""
            You are a professional forecaster.

            Question: {question.question_text}
            Options: {question.options}
            Background: {question.background_info}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            Research: {research}
            Today: {datetime.now().strftime('%Y-%m-%d')}

            Before answering:
            (a) Time until resolution
            (b) Status quo outcome
            (c) Unexpected scenario

            Remember: leave moderate probability on most options.

            The last thing you write is your final probabilities as:
            Option_A: XX%
            Option_B: YY%
            ...
        """)
        parsing_instructions = f"Valid options: {question.options}"
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        pred: PredictedOptionList = await structure_output(
            reasoning,
            PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )
        return ReasonedPrediction(prediction_value=pred, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        lower_msg = (
            f"The outcome cannot be lower than {question.lower_bound}."
            if not question.open_lower_bound
            else f"The question creator thinks it's unlikely to be below {question.lower_bound}."
        )
        upper_msg = (
            f"The outcome cannot be higher than {question.upper_bound}."
            if not question.open_upper_bound
            else f"The question creator thinks it's unlikely to be above {question.upper_bound}."
        )
        prompt = clean_indents(f"""
            You are a professional forecaster.

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            Units: {question.unit_of_measure or 'Inferred'}
            Research: {research}
            Today: {datetime.now().strftime('%Y-%m-%d')}

            {lower_msg}
            {upper_msg}

            Formatting:
            - Never use scientific notation
            - Start with smaller number, increase

            Before answering:
            (a) Time until resolution
            (b) Outcome if nothing changed
            (c) Outcome if trend continued
            (d) Expert/market expectations
            (e) Low-outcome scenario
            (f) High-outcome scenario

            The last thing you write is:
            Percentile 10: X
            Percentile 20: X
            Percentile 40: X
            Percentile 60: X
            Percentile 80: X
            Percentile 90: X
        """)
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        dist = NumericDistribution.from_question(percentile_list, question)
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        models = [
            "openrouter/openai/gpt-5.1",
            "openrouter/openai/gpt-5.4",
            "openrouter/xiaomi/mimo-v2.5",
        ]
        predictions = []
        reasonings = []

        for model in models:
            original_default = self._llms.get("default")
            original_parser  = self._llms.get("parser")
            self._llms["default"] = GeneralLlm(model=model)
            self._llms["parser"]  = GeneralLlm(model="openrouter/openai/gpt-4.1-mini")

            try:
                if isinstance(question, BinaryQuestion):
                    pred = await self._run_forecast_on_binary(question, research)
                elif isinstance(question, MultipleChoiceQuestion):
                    pred = await self._run_forecast_on_multiple_choice(question, research)
                elif isinstance(question, NumericQuestion):
                    pred = await self._run_forecast_on_numeric(question, research)
                else:
                    raise ValueError(f"Unsupported question type: {type(question)}")
                predictions.append(pred.prediction_value)
                reasonings.append(pred.reasoning)
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
            finally:
                self._llms["default"] = original_default
                self._llms["parser"]  = original_parser

        if not predictions:
            raise ValueError("All committee models failed — no predictions generated.")

        if isinstance(question, BinaryQuestion):
            final_pred = ReasonedPrediction(
                prediction_value=median([p for p in predictions]),
                reasoning=" | ".join(reasonings),
            )
        elif isinstance(question, MultipleChoiceQuestion):
            options = question.options
            avg_probs = {}
            for opt in options:
                option_probs = []
                for p in predictions:
                    pred_dict = {po.option_name: po.probability for po in p.predicted_options}
                    option_probs.append(pred_dict.get(opt, 0.0))
                avg_probs[opt] = median(option_probs)
            total = sum(avg_probs.values())
            if total > 0:
                avg_probs = {k: v / total for k, v in avg_probs.items()}
            final_pred = ReasonedPrediction(
                prediction_value=PredictedOptionList(predicted_options=[
                    PredictedOption(option_name=opt, probability=prob)
                    for opt, prob in avg_probs.items()
                ]),
                reasoning=" | ".join(reasonings),
            )
        elif isinstance(question, NumericQuestion):
            target_pts = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
            median_percentiles = []
            for pt in target_pts:
                vals = []
                for p in predictions:
                    for item in p.declared_percentiles:
                        if abs(item.percentile - pt) < 0.01:
                            vals.append(item.value)
                            break
                median_percentiles.append(
                    Percentile(percentile=pt, value=median(vals) if vals else 0.0)
                )
            final_pred = ReasonedPrediction(
                prediction_value=NumericDistribution.from_question(median_percentiles, question),
                reasoning=" | ".join(reasonings),
            )
        else:
            final_pred = ReasonedPrediction(
                prediction_value=predictions[0], reasoning=" | ".join(reasonings)
            )

        return final_pred


# ------------------------------------------------------------------
# MONKEY-PATCH: fix PredictedOptionList validator
# ------------------------------------------------------------------
@model_validator(mode='after')
def _fixed_normalize_probabilities(self: PredictedOptionList):
    if not self.predicted_options:
        return self
    sum_ = sum(p.probability for p in self.predicted_options)
    if sum_ <= 0:
        logger.warning(f"PredictedOptionList sum is {sum_}; cannot normalize.")
        return self
    if abs(sum_ - 1.0) > 0.001:
        logger.info(f"Normalizing probabilities from sum={sum_}.")
        for option in self.predicted_options:
            option.probability = option.probability / sum_
    for option in self.predicted_options:
        if option.probability < 0:
            option.probability = 0.0
    return self

PredictedOptionList.__pydantic_post_validate__ = _fixed_normalize_probabilities
logger.info("Monkey-patched 'PredictedOptionList' validator successfully.")
# ------------------------------------------------------------------


if __name__ == "__main__":
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").propagate = False

    parser = argparse.ArgumentParser(description="Run Yrambot")
    parser.add_argument(
        "--tournament-ids", nargs="+", type=str,
        default=["33022", MetaculusApi.CURRENT_MINIBENCH_ID],
    )
    args = parser.parse_args()

    bot = Yrambot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=False,
    )

    all_reports = []
    for tid in args.tournament_ids:
        logger.info(f"Forecasting on tournament: {tid}")
        reports = asyncio.run(bot.forecast_on_tournament(tid, return_exceptions=True))
        all_reports.extend(reports)

    bot.log_report_summary(all_reports)
    logger.info("✅ Yrambot run completed.")
