import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import List

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
    # --- FIX 3: Import PredictedOption ---
    # This class is needed to correctly construct the PredictedOptionList
    # in your _make_prediction override.
    PredictedOption,
    # ------------------------------------
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


class Yrambot(ForecastBot):
    """
    Conservative hybrid forecaster using only GPT-5 and Claude Sonnet 4.5.
    No external news or web search — relies on model knowledge with strong temporal awareness.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5",
            "parser": "openrouter/openai/gpt-4.1-mini",
            "researcher_gpt": "openrouter/openai/gpt-5",
            "researcher_claude": "openrouter/anthropic/claude-sonnet-4.5",
            # --- FIX 1: Add the 'summarizer' key ---
            # The parent ForecastBot class requires this key to summarize research
            # when 'publish_reports_to_metaculus' is True. This fixes your
            # "Unknown llm requested... for purpose: 'summarizer'" error.
            # We can use the 'parser' model as it's good for this kind of task.
            "summarizer": "openrouter/openai/gpt-4.1-mini",
            # ----------------------------------------
        }

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            today_str = datetime.now().strftime("%Y-%m-%d")
            
            gpt_prompt = clean_indents(f"""
                You are an expert researcher with deep world knowledge up to June 2024, but you understand that today is {today_str}.
                Analyze the following forecasting question with attention to recent developments, trends, and timing.

                Question: {question.question_text}
                Background: {question.background_info or 'None provided'}
                Resolution criteria: {question.resolution_criteria or 'Standard'}
                Fine print: {question.fine_print or 'None'}

                Consider:
                - How much time remains until resolution?
                - Has anything changed recently that affects this outcome?
                - What is the status quo? (World changes slowly.)
                - Are there known upcoming events (elections, product launches, policy deadlines)?
                - If uncertain, say so. Do not hallucinate.

                Provide a concise, factual summary for a professional forecaster.
            """)
            
            claude_prompt = clean_indents(f"""
                You are Claude Sonnet 4.5, a precise and cautious AI with knowledge cutoff in early 2024, but aware that today is {today_str}.
                Your task: analyze the forecasting question below with strong temporal reasoning.

                Question: {question.question_text}
                Context: {question.background_info or 'Not specified'}
                Resolution rules: {question.resolution_criteria or 'Default'}

                Focus on:
                - Recency: Is this question about a near-term or long-term event?
                - Plausibility given current date ({today_str})
                - Base rates and historical analogs
                - Known constraints or scheduled events before resolution

                Be honest about uncertainty. Avoid speculation beyond your knowledge.

                Output only relevant facts and reasoned considerations.
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
                f"--- RESEARCH FROM GPT-5 (as of {today_str}) ---\n{gpt_response}\n\n"
                f"--- RESEARCH FROM CLAUDE SONNET 4.5 (as of {today_str}) ---\n{claude_response}\n"
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
            (d) Model disagreements in research

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
        lower_msg = f"The outcome cannot be lower than {question.lower_bound}." if not question.open_lower_bound else f"The question creator thinks it's unlikely to be below {question.lower_bound}."
        upper_msg = f"The outcome cannot be higher than {question.upper_bound}." if not question.open_upper_bound else f"The question creator thinks it's unlikely to be above {question.upper_bound}."

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

    # -----------------------------
    # Override parent to inject committee + median logic
    # -----------------------------
    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        """
        Override to generate 3 predictions (GPT-5, GPT-4o, Claude) and return median.
        """
        models = [
            "openrouter/openai/gpt-5",
            "openrouter/openai/gpt-4o",
            "openrouter/anthropic/claude-sonnet-4.5"
        ]
        predictions = []
        reasonings = []

        for model in models:
            # Temporarily override default LLM
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
                # Restore originals
                self._llms["default"] = original_default
                self._llms["parser"] = original_parser

        # Median aggregation
        if isinstance(question, BinaryQuestion):
            median_val = median([p for p in predictions])
            final_pred = ReasonedPrediction(prediction_value=median_val, reasoning=" | ".join(reasonings))
        elif isinstance(question, MultipleChoiceQuestion):
            # Average then median probabilities per option
            options = question.options
            avg_probs = {}
            for opt in options:
                # Note: p.predicted_options is a list of PredictedOption objects
                # We need to convert it to a dict for easy lookup
                prob_dict = {po.option_name: po.probability for po in p.predicted_options}
                
                # Get all probabilities for this option from all model predictions
                # Use a default of 0 if a model didn't predict this option
                option_probs = []
                for p in predictions:
                    pred_dict = {po.option_name: po.probability for po in p.predicted_options}
                    option_probs.append(pred_dict.get(opt, 0.0))
                
                avg_probs[opt] = median(option_probs)
            
            total = sum(avg_probs.values())
            if total > 0:
                avg_probs = {k: v / total for k, v in avg_probs.items()}
            
            # --- FIX 3: Correctly construct PredictedOptionList ---
            # The Pydantic model expects a list of `PredictedOption` objects,
            # not a list of tuples. We build that list here.
            predicted_options_list = [
                PredictedOption(option_name=opt, probability=prob)
                for opt, prob in avg_probs.items()
            ]
            final_pred = ReasonedPrediction(
                prediction_value=PredictedOptionList(predicted_options=predicted_options_list),
                reasoning=" | ".join(reasonings)
            )
            # -----------------------------------------------------

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
            # Fallback in case of an issue
            if predictions:
                final_pred = ReasonedPrediction(prediction_value=predictions[0], reasoning=" | ".join(reasonings))
            else:
                raise ValueError("No predictions were generated.")

        return final_pred


# ------------------------------------------------------------------
# --- FIX 2: MONKEY-PATCH TO FIX PYDANTIC VALIDATOR ---
# ------------------------------------------------------------------
# This section dynamically replaces the broken validator in the imported
# PredictedOptionList class with a new one that correctly handles
# empty lists (sum=0) and rounding errors (sum=1.02).
# This avoids needing to edit the library files directly.

# 1. Define the new, correct validator function
@model_validator(mode='after')
def _fixed_normalize_probabilities(self: PredictedOptionList):
    """
    This is the fixed validator that will replace the original.
    """
    # Note: 'self' here is an instance of PredictedOptionList
    if not self.predicted_options:
        # Case 1: Empty list. Fixes "Sum=0" error.
        return self

    sum_ = sum(p.probability for p in self.predicted_options)

    if sum_ <= 0:
        # Case 2: Non-empty list, but sum is zero.
        logger.warning(
            f"PredictedOptionList has non-empty list but sum is {sum_}. "
            f"Probabilities cannot be normalized. Raw options: {self.predicted_options}"
        )
        return self

    # Case 3: Non-empty list with a positive sum.
    # We will *always* normalize to fix rounding errors (like the "Sum=1.02" error).
    if abs(sum_ - 1.0) > 0.001:
        logger.info(
            f"Normalizing probabilities. Original sum was {sum_}. "
            f"This fixes potential errors from sums like 1.02 or 0.99."
        )
        for option in self.predicted_options:
            option.probability = option.probability / sum_
    
    # Ensure no probabilities are negative after normalization (shouldn't happen if sum_ > 0)
    for option in self.predicted_options:
        if option.probability < 0:
            option.probability = 0.0

    return self

# 2. Attach the new validator to the imported class, replacing the old one.
# We re-assign the __pydantic_post_validate__ dunder method,
# which is what `model_validator(mode='after')` creates.
PredictedOptionList.__pydantic_post_validate__ = _fixed_normalize_probabilities

logger.info("Monkey-patched 'PredictedOptionList' validator successfully.")
# --- END OF MONKEY-PATCH ---
# ------------------------------------------------------------------


# -----------------------------
# MAIN — Tournament Mode
# -----------------------------
if __name__ == "__main__":
    # Suppress LiteLLM noise
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(description="Run Yrambot (GPT-5 + Claude hybrid)")
    parser.add_argument(
        "--tournament-ids",
        nargs="+",
        type=str,
        default=["32813", "metaculus-cup-fall-2025", "market-pulse-25q4", MetaculusApi.CURRENT_MINIBENCH_ID],
    )
    args = parser.parse_args()

    bot = Yrambot(
        research_reports_per_question=1,
        predictions_per_research_report=1,  # Handled via _make_prediction override
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
    )

    all_reports = []
    for tid in args.tournament_ids:
        logger.info(f"Forecasting on tournament: {tid}")
        reports = asyncio.run(bot.forecast_on_tournament(tid, return_exceptions=True))
        all_reports.extend(reports)

    bot.log_report_summary(all_reports)
    logger.info("✅ Yrambot run completed.")
