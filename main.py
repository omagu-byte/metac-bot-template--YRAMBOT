import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import List

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


class Yrambot(ForecastBot):
    """
    Conservative hybrid forecaster.
    Researcher: Perplexity Sonar (live web search).
    Committee: GPT-5, GPT-4o, Perplexity Sonar (fallback).
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default":    "openrouter/openai/gpt-5.5",
            "parser":     "openrouter/openai/gpt-4.1-mini",
            # ── Single researcher using Sonar's built-in web search ──
            "researcher": "openrouter/perplexity/sonar-pro",
            "summarizer": "openrouter/openai/gpt-4.1-mini",
        }

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            today_str = datetime.now().strftime("%Y-%m-%d")

            sonar_prompt = clean_indents(f"""
                You are a research assistant with live web-search capability.
                Today is {today_str}. Use your search tools to find the most
                recent, relevant information for the forecasting question below.

                Question: {question.question_text}
                Background: {question.background_info or 'None provided'}
                Resolution criteria: {question.resolution_criteria or 'Standard'}
                Fine print: {question.fine_print or 'None'}

                Focus on:
                - Current status of the topic as of today ({today_str})
                - Recent news, data releases, or events that affect the outcome
                - Scheduled events (elections, product launches, policy deadlines)
                  that fall before the resolution date
                - Base rates and historical analogues
                - Time remaining until resolution and whether the status quo is
                  likely to hold

                Be factual. Cite sources where possible. Do not speculate beyond
                what the evidence supports.

                Output a concise research summary for a professional forecaster.
            """)

            try:
                sonar_response = await self.get_llm("researcher", "llm").invoke(sonar_prompt)
            except Exception as e:
                sonar_response = f"[Perplexity Sonar research failed: {str(e)}]"

            return (
                f"--- RESEARCH FROM PERPLEXITY SONAR (as of {today_str}) ---\n"
                f"{sonar_response}\n"
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
        """
        Committee: GPT-5 (primary), GPT-4o, Perplexity Sonar (fallback).
        Returns median-aggregated prediction.
        """
        models = [
            "openrouter/openai/gpt-5.1",
            "openrouter/openai/gpt-5.4",
            "openrouter/perplexity/sonar-pro",   
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
                median_val = median(vals) if vals else 0.0
                median_percentiles.append(Percentile(percentile=pt, value=median_val))
            final_dist = NumericDistribution.from_question(median_percentiles, question)
            final_pred = ReasonedPrediction(prediction_value=final_dist, reasoning=" | ".join(reasonings))

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


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").propagate = False

    parser = argparse.ArgumentParser(description="Run Yrambot")
    parser.add_argument(
        "--tournament-ids",
        nargs="+",
        type=str,
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
