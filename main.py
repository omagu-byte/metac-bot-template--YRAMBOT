import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import List

# Tavily integration
from tavily import TavilyClient

# --- FIX 2: Import pydantic validator for monkey-patching ---
from pydantic import model_validator
# -------------------------------------------------------------

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

# -----------------------------
# Helper: Pure-Python median
# -----------------------------
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

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Yrambot")

# Initialize Tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
if not os.getenv("TAVILY_API_KEY"):
    raise EnvironmentError("TAVILY_API_KEY environment variable not set.")

class Yrambot(ForecastBot):
    """
    BOLD superforecaster hybrid using GPT-5, Claude Sonnet 4.5, and REAL-TIME Tavily research.
    Leverages base rates, trend momentum, and statistical confidence — not conservatism.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5",
            "parser": "openrouter/openai/gpt-4.1-mini",
            "researcher_gpt": "openrouter/openai/gpt-5",
            "researcher_claude": "openrouter/anthropic/claude-sonnet-4.5",
            "summarizer": "openrouter/openai/gpt-4.1-mini",
        }

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            today_str = datetime.now().strftime("%Y-%m-%d")
            
            # --- TAVILY RESEARCH ---
            query = f"{question.question_text} {question.background_info or ''}".strip()
            try:
                tavily_response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: tavily_client.search(
                        query=query,
                        search_depth="advanced",
                        include_answer=True,
                        max_results=6,
                        include_raw_content=False,
                        include_domains=[],
                        exclude_domains=[],
                    )
                )
                tavily_summary = (
                    f"Answer: {tavily_response.get('answer', 'No direct answer.')}\n"
                    + "\n".join(
                        f"[{i+1}] {r['title']}: {r['content']}" 
                        for i, r in enumerate(tavily_response.get('results', []))
                    )
                )
            except Exception as e:
                logger.error(f"Tavily research failed: {e}")
                tavily_summary = f"[Tavily research error: {str(e)}]"

            # --- LLM RESEARCH (optional supplement) ---
            gpt_prompt = clean_indents(f"""
                You are a world-class superforecaster with knowledge up to June 2024, but you know today is {today_str}.
                Analyze the following forecasting question with strategic boldness and statistical rigor.

                Question: {question.question_text}
                Background: {question.background_info or 'None provided'}
                Resolution criteria: {question.resolution_criteria or 'Standard'}
                Fine print: {question.fine_print or 'None'}

                Focus on:
                - Base rates and historical analogs
                - Trend momentum (is the trajectory accelerating or stalling?)
                - Key decision-makers and upcoming deadlines
                - Asymmetric risks (what would make this outcome much more or less likely?)
                - Don’t hedge—be boldly calibrated.

                Provide a sharp, evidence-based summary. Avoid fluff.
            """)

            claude_prompt = clean_indents(f"""
                You are Claude Sonnet 4.5. Today is {today_str}. Be precise, statistical, and unafraid of confident inference.

                Question: {question.question_text}
                Context: {question.background_info or 'Not specified'}
                Resolution rules: {question.resolution_criteria or 'Default'}

                Apply:
                - Reference class forecasting
                - Regression to the mean vs. disruption potential
                - Known inflection points before resolution

                Output only high-signal insights. No filler.
            """)

            try:
                gpt_response = await self.get_llm("researcher_gpt", "llm").invoke(gpt_prompt)
            except Exception as e:
                gpt_response = f"[GPT-5 research failed: {str(e)}]"

            try:
                claude_response = await self.get_llm("researcher_claude", "llm").invoke(claude_prompt)
            except Exception as e:
                claude_response = f"[Claude Sonnet research failed: {str(e)}]"

            return (
                f"--- TAVILY REAL-TIME RESEARCH (as of {today_str}) ---\n{tavily_summary}\n\n"
                f"--- GPT-5 FORECASTER ANALYSIS ---\n{gpt_response}\n\n"
                f"--- CLAUDE SONNET STRATEGIC REVIEW ---\n{claude_response}\n"
            )

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(f"""
            You are a top-tier superforecaster known for bold, well-calibrated, and statistically grounded predictions.
            Today is {datetime.now().strftime('%Y-%m-%d')}.

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            Research: {research}

            Apply:
            (a) Reference class: What’s the base rate?
            (b) Trend vector: Is momentum increasing or decaying?
            (c) Key thresholds: What would push this over the edge?
            (d) Time until resolution: How much can realistically change?

            Be confident. Avoid false modesty. If evidence points strongly, say so.

            Final line: "Probability: ZZ%" (0–100)
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
            You are a superforecaster using base rates, momentum, and scenario analysis.

            Question: {question.question_text}
            Options: {question.options}
            Background: {question.background_info}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            Research: {research}
            Today: {datetime.now().strftime('%Y-%m-%d')}

            Assign probabilities based on:
            - Likelihood of status quo vs. disruption
            - Historical frequencies of similar outcomes
            - Upcoming catalysts

            Be decisive. Don’t spread probability thinly unless truly ambiguous.

            Final output format:
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
        lower_msg = f"The outcome cannot be lower than {question.lower_bound}." if not question.open_lower_bound else f"Unlikely below {question.lower_bound}."
        upper_msg = f"The outcome cannot be higher than {question.upper_bound}." if not question.open_upper_bound else f"Unlikely above {question.upper_bound}."

        prompt = clean_indents(f"""
            You are a quantitative superforecaster. Think in distributions, not point estimates.

            Question: {question.question_text}
            Units: {question.unit_of_measure or 'Inferred'}
            Research: {research}
            Today: {datetime.now().strftime('%Y-%m-%d')}

            {lower_msg}
            {upper_msg}

            Construct a distribution using:
            - Base rate from historical analogs
            - Current trend slope
            - Volatility and uncertainty bandwidth
            - Known ceiling/floor effects

            Output format:
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

    # -----------------------------
    # Override with committee + median (unchanged logic)
    # -----------------------------
    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        models = [
            "openrouter/openai/gpt-5",
            "openrouter/openai/gpt-4o",
            "openrouter/anthropic/claude-sonnet-4.5"
        ]
        predictions = []
        reasonings = []

        for model in models:
            original_default = self._llms.get("default")
            original_parser = self._llms.get("parser")
            self._llms["default"] = GeneralLlm(model=model)
            self._llms["parser"] = GeneralLlm(model="openrouter/openai/gpt-4.1-mini")

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
            finally:
                self._llms["default"] = original_default
                self._llms["parser"] = original_parser

        # Median aggregation logic (unchanged)
        if isinstance(question, BinaryQuestion):
            median_val = median([p for p in predictions])
            final_pred = ReasonedPrediction(prediction_value=median_val, reasoning=" | ".join(reasonings))
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
            predicted_options_list = [
                PredictedOption(option_name=opt, probability=prob)
                for opt, prob in avg_probs.items()
            ]
            final_pred = ReasonedPrediction(
                prediction_value=PredictedOptionList(predicted_options=predicted_options_list),
                reasoning=" | ".join(reasonings)
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
                median_val = median(vals) if vals else 0.0
                median_percentiles.append(Percentile(percentile=pt, value=median_val))
            final_dist = NumericDistribution.from_question(median_percentiles, question)
            final_pred = ReasonedPrediction(prediction_value=final_dist, reasoning=" | ".join(reasonings))
        else:
            if predictions:
                final_pred = ReasonedPrediction(prediction_value=predictions[0], reasoning=" | ".join(reasonings))
            else:
                raise ValueError("No predictions generated.")
        return final_pred

# ------------------------------------------------------------------
# MONKEY-PATCH: Fix PredictedOptionList validator
# ------------------------------------------------------------------
@model_validator(mode='after')
def _fixed_normalize_probabilities(self: PredictedOptionList):
    if not self.predicted_options:
        return self
    sum_ = sum(p.probability for p in self.predicted_options)
    if sum_ <= 0:
        logger.warning(f"PredictedOptionList sum is {sum_}. Cannot normalize. Raw: {self.predicted_options}")
        return self
    if abs(sum_ - 1.0) > 0.001:
        logger.info(f"Normalizing probabilities. Original sum: {sum_}")
        for option in self.predicted_options:
            option.probability = option.probability / sum_
    for option in self.predicted_options:
        if option.probability < 0:
            option.probability = 0.0
    return self

PredictedOptionList.__pydantic_post_validate__ = _fixed_normalize_probabilities
logger.info("Monkey-patched 'PredictedOptionList' validator successfully.")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(description="Run Yrambot (Bold Superforecaster + Tavily)")
    parser.add_argument(
        "--tournament-ids",
        nargs="+",
        type=str,
        default=[
            "32813",
            "32916",                    # added
            "ACX2026",                  # added
            "metaculus-cup-fall-2026",
            "market-pulse-26q1",        # replaced 25q4 → 26q1
            MetaculusApi.CURRENT_MINIBENCH_ID
        ],
    )
    args = parser.parse_args()

    bot = Yrambot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
    )

    all_reports = []
    for tid in args.tournament_ids:
        logger.info(f"Forecasting on tournament: {tid}")
        reports = asyncio.run(bot.forecast_on_tournament(tid, return_exceptions=True))
        all_reports.extend(reports)

    bot.log_report_summary(all_reports)
    logger.info("✅ Yrambot (Bold + Tavily) run completed.")
