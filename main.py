import argparse
import asyncio
import logging
import os
import textwrap
import re
import math
from datetime import datetime
from typing import Union

from tavily import TavilyClient

from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
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

# --- LOG-ODDS EXTREMIZING (DETERMINISTIC & STABLE) ---
def extremize(p: float, factor: float = 5.8) -> float:
    """Extremizes probability by scaling log-odds. Deterministic and numerically stable."""
    p = max(1e-10, min(1 - 1e-10, p))
    logit = math.log(p / (1 - p))
    extremized_logit = factor * logit
    result = 1 / (1 + math.exp(-extremized_logit))
    return max(0.001, min(0.999, result))

def build_tavily_query(question: MetaculusQuestion, max_chars: int = 397) -> str:
    q = re.sub(r"http\S+", "", question.question_text.strip())
    q = re.sub(r"\s+", " ", q).strip()
    return textwrap.shorten(q, width=max_chars, placeholder="…")

logger = logging.getLogger(__name__)

class Yrambot(ForecastBot):
    """
    ULTRA-BOLD Superforecaster Bot — Refactored to SpringTemplateBot2026 Style.
    - Uses dual aggressive models (gpt-5.2 + gpt-5.1)
    - Applies log-odds extremizing based on model agreement
    - Scales numeric predictions for financial contexts
    - Research via Tavily
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2  # Match template

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            today_str = datetime.now().strftime("%Y-%m-%d")
            query = build_tavily_query(question)
            try:
                loop = asyncio.get_event_loop()
                tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
                tavily_response = await loop.run_in_executor(
                    None, lambda: tavily_client.search(query=query, search_depth="advanced", max_results=6)
                )
                res = tavily_response.get("results", [])
                if not res:
                    return f"--- REAL-TIME MARKET INTELLIGENCE ({today_str}) ---\nNo relevant results found."
                tavily_summary = "\n".join([f"[{i+1}] {r['title']}: {r['content'][:350]}" for i, r in enumerate(res)])
                research = f"--- REAL-TIME MARKET INTELLIGENCE ({today_str}) ---\n{tavily_summary}"
                logger.info(f"Found Research for URL {question.page_url}:\n{research}")
                return research
            except Exception as e:
                logger.warning(f"Tavily research failed for Q{question.id}: {e}")
                fallback = f"--- REAL-TIME MARKET INTELLIGENCE ({today_str}) ---\nResearch unavailable."
                logger.info(f"Using fallback research for {question.page_url}")
                return fallback

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        p1 = await self._run_model_forecast("forecaster_1", question, research)
        p2 = await self._run_model_forecast("forecaster_2", question, research)

        avg_p = (p1 + p2) / 2
        deviation = abs(p1 - p2)
        bold_factor = 6.2 if deviation < 0.02 else 4.0
        final_p = extremize(avg_p, factor=bold_factor)
        final_p = max(0.01, min(0.99, final_p))

        reasoning = (
            f"### [BOLD FORECAST]\n"
            f"Research:\n{research}\n\n"
            f"Model Consensus: GPT-5.2={p1:.1%}, GPT-5.1={p2:.1%}. "
            f"Deviation={deviation:.1%}, Extremizing Factor={bold_factor}."
        )
        logger.info(f"Forecasted URL {question.page_url} with prediction: {final_p:.1%}")
        return ReasonedPrediction(prediction_value=final_p, reasoning=reasoning)

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        p1 = await self._run_model_forecast("forecaster_1", question, research)
        p2 = await self._run_model_forecast("forecaster_2", question, research)

        combined_options = {}
        for opt in question.options:
            v1 = next((o.probability for o in p1.predicted_options if o.option_name == opt), 0.0)
            v2 = next((o.probability for o in p2.predicted_options if o.option_name == opt), 0.0)
            combined_options[opt] = extremize((v1 + v2) / 2, factor=5.0)

        total = sum(combined_options.values())
        if total == 0:
            # Uniform fallback
            uniform_prob = 1.0 / len(question.options)
            final_options = [PredictedOption(option_name=k, probability=uniform_prob) for k in question.options]
        else:
            final_options = [PredictedOption(option_name=k, probability=v / total) for k, v in combined_options.items()]

        prediction_list = PredictedOptionList(predicted_options=final_options)
        logger.info(f"Forecasted URL {question.page_url} with options: {[opt.option_name for opt in final_options]}")
        reasoning = f"### [BOLD MULTIPLE CHOICE]\nResearch:\n{research}"
        return ReasonedPrediction(prediction_value=prediction_list, reasoning=reasoning)

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        p1 = await self._run_model_forecast("forecaster_1", question, research)
        p2 = await self._run_model_forecast("forecaster_2", question, research)

        if not isinstance(p1, list) or not isinstance(p2, list):
            raise ValueError("LLM did not return a list of percentiles")

        def normalize_percentile_level(level: float) -> float:
            return level / 100.0 if level > 1.0 else level

        def process_percentiles(percentile_list: list[Percentile]) -> list[Percentile]:
            processed = []
            for p in percentile_list:
                norm_pct = normalize_percentile_level(p.percentile)
                scaled_val = self._scale_numeric_prediction(p.value, question)
                processed.append(Percentile(percentile=norm_pct, value=scaled_val))
            return processed

        try:
            p1_proc = process_percentiles(p1)
            p2_proc = process_percentiles(p2)
        except Exception as e:
            logger.error(f"Failed to process percentiles: {e}")
            return self._numeric_fallback(question, research)

        if len(p1_proc) < 2 or len(p2_proc) < 2:
            return self._numeric_fallback(question, research)

        p1_map = {p.percentile: p.value for p in p1_proc}
        p2_map = {p.percentile: p.value for p in p2_proc}
        all_percentiles = sorted(set(p1_map.keys()) | set(p2_map.keys()))

        final_percentiles = []
        for pct in all_percentiles:
            v1 = p1_map.get(pct, p2_map[pct])
            v2 = p2_map.get(pct, p1_map[pct])
            avg_val = (v1 + v2) / 2
            final_percentiles.append(Percentile(percentile=pct, value=avg_val))

        if len(final_percentiles) < 2:
            return self._numeric_fallback(question, research)

        try:
            dist = NumericDistribution.from_question(final_percentiles, question)
        except ValueError as e:
            logger.warning(f"Validation failed, clamping to bounds: {e}")
            low = getattr(question, 'minimum', float('-inf'))
            high = getattr(question, 'maximum', float('inf'))
            clamped = [
                Percentile(percentile=p.percentile, value=max(low, min(high, p.value)))
                for p in final_percentiles
            ]
            dist = NumericDistribution.from_question(clamped, question)

        logger.info(f"Forecasted URL {question.page_url} with percentiles: {dist.declared_percentiles}")
        reasoning = f"### [BOLD NUMERIC]\nResearch:\n{research}"
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    def _scale_numeric_prediction(self, value: float, question: NumericQuestion) -> float:
        """Scale LLM output based on context (e.g., billions for revenue)."""
        text = question.question_text.lower()
        revenue_indicators = ["revenue", "sales", "earnings", "income", "quarterly", "fy202", "financial"]
        company_tickers = ["AAPL", "AMD", "AMZN", "NVDA", "MSFT", "GOOGL", "META", "TSLA", "NFLX", "ADBE"]

        if any(ind in text for ind in revenue_indicators) and any(ticker in text for ticker in company_tickers):
            return value * 1e9
        return value

    def _numeric_fallback(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        low = getattr(question, 'minimum', 0)
        high = getattr(question, 'maximum', max(1, abs(low) * 2))
        default_percentiles = [
            Percentile(percentile=0.1, value=low),
            Percentile(percentile=0.5, value=(low + high) / 2),
            Percentile(percentile=0.9, value=high),
        ]
        dist = NumericDistribution.from_question(default_percentiles, question)
        logger.warning(f"Using numeric fallback for {question.page_url}")
        reasoning = f"### [BOLD NUMERIC - FALLBACK]\nResearch:\n{research}"
        return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

    ##################################### CORE MODEL INFERENCE #####################################

    async def _run_model_forecast(
        self, model_key: str, question: MetaculusQuestion, research: str
    ) -> Union[float, PredictedOptionList, list[Percentile]]:
        prompt = clean_indents(f"""
            Role: Elite Quantitative Superforecaster. Today: {datetime.now().strftime('%Y-%m-%d')}
            Question: {question.question_text}
            Resolution Criteria: {getattr(question, 'resolution_criteria', 'Not specified')}
            Background: {getattr(question, 'background_info', '')}
            Research: {research}

            Mandate:
            1. Base Rate: What is the historical frequency?
            2. Signal Capture: How does current evidence shift this?
            3. Boldness: Do not hedge. Be decisive.
        """)
        llm = self.get_llm(model_key, "llm")
        reasoning = await llm.invoke(prompt)
        parser = self.get_llm("parser", "llm")

        if isinstance(question, BinaryQuestion):
            pred = await structure_output(reasoning, BinaryPrediction, model=parser)
            return pred.prediction_in_decimal
        elif isinstance(question, MultipleChoiceQuestion):
            return await structure_output(reasoning, PredictedOptionList, model=parser)
        else:
            return await structure_output(reasoning, list[Percentile], model=parser)

    ##################################### LLMS CONFIG #####################################

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5.2",
            "parser": "openrouter/openai/gpt-4.1-mini",
            "forecaster_1": "openrouter/openai/gpt-5.2",
            "forecaster_2": "openrouter/openai/gpt-5.1",
            "summarizer": "openrouter/openai/gpt-4.1-mini",
            "researcher": "tavily",  # Not used directly, but defined for clarity
        }


# --- MAIN EXECUTION (MATCHES TEMPLATE STYLE) ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run Yrambot forecasting system")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()

    bot = Yrambot(
        research_reports_per_question=1,
        predictions_per_research_report=1,  # Your logic produces 1 aggregate forecast per research
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
    )

    client = MetaculusClient()
    if args.mode == "tournament":
        reports1 = asyncio.run(bot.forecast_on_tournament(client.CURRENT_AI_COMPETITION_ID, return_exceptions=True))
        reports2 = asyncio.run(bot.forecast_on_tournament(client.CURRENT_MINIBENCH_ID, return_exceptions=True))
        forecast_reports = reports1 + reports2
    elif args.mode == "test_questions":
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
        ]
        bot.skip_previously_forecasted_questions = False
        questions = [client.get_question_by_url(url.strip()) for url in EXAMPLE_QUESTIONS]
        forecast_reports = asyncio.run(bot.forecast_questions(questions, return_exceptions=True))
    else:
        raise NotImplementedError("Only 'tournament' and 'test_questions' modes are configured.")

    bot.log_report_summary(forecast_reports)
