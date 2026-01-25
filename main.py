import argparse
import asyncio
import logging
import os
import textwrap
import re
import random
from datetime import datetime
from typing import List, Union
from tavily import TavilyClient
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

# --- STATISTICAL ENGINE ---
def extremize(p: float, factor: float = 5.8) -> float:
    """Aggressively shifts probabilities away from 0.5."""
    p = max(0.001, min(0.999, p))
    # Avoid log(1) = 0 when p is too close to 0.5
    if abs(p - 0.5) < 1e-5:
        p += random.choice([-0.001, 0.001])
    
    odds = p / (1 - p)
    extremized_odds = odds**factor
    result = extremized_odds / (1 + extremized_odds)
    return max(0.001, min(0.999, result))

def build_tavily_query(question: MetaculusQuestion, max_chars: int = 397) -> str:
    q = re.sub(r"http\S+", "", question.question_text.strip())
    q = re.sub(r"\s+", " ", q).strip()
    return textwrap.shorten(q, width=max_chars, placeholder="…")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Yrambot")

class Yrambot(ForecastBot):
    """
    ULTRA-BOLD Superforecaster. 
    Uses GPT-5.2 and GPT-5.1 with extreme log-odds shifts.
    """
    
    _max_concurrent_questions = 1
    _forecast_semaphore = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5.2",
            "parser": "openrouter/openai/gpt-4.1-mini",
            "forecaster_1": "openrouter/openai/gpt-5.2",
            "forecaster_2": "openrouter/openai/gpt-5.1",
            "summarizer": "openrouter/openai/gpt-4.1-mini",  # Critical fix
        }

    def _scale_numeric_prediction(self, value: float, question: NumericQuestion) -> float:
        """Scale LLM output based on question context (e.g., billions for revenue)."""
        text = question.question_text.lower()
        # Detect if this is a company revenue question (common in Market Pulse)
        revenue_indicators = ["revenue", "sales", "earnings", "income", "quarterly", "fy202", "financial"]
        company_tickers = ["AAPL", "AMD", "AMZN", "NVDA", "MSFT", "GOOGL", "META", "TSLA", "NFLX", "ADBE"]
        
        if any(ind in text for ind in revenue_indicators) and any(ticker in text for ticker in company_tickers):
            # Assume output is in billions (e.g., "65.35" → 65.35B = 65,350,000,000)
            return value * 1e9
        return value

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._forecast_semaphore:
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
                return f"--- REAL-TIME MARKET INTELLIGENCE ({today_str}) ---\n{tavily_summary}"
            except Exception as e:
                logger.warning(f"Tavily research failed for Q{question.id}: {e}")
                return f"--- REAL-TIME MARKET INTELLIGENCE ({today_str}) ---\nResearch unavailable."

    async def _run_model_forecast(self, model_key: str, question: MetaculusQuestion, research: str) -> Union[float, PredictedOptionList, List[Percentile]]:
        prompt = clean_indents(f"""
            Role: Elite Quantitative Superforecaster. Today: {datetime.now().strftime('%Y-%m-%d')}
            Question: {question.question_text}
            Research: {research}
            
            Mandate: 
            1. Base Rate: Determine historical frequency.
            2. Signal Capture: Use research to adjust the probability. 
            3. Boldness: Do not hedge. If p < 0.5, push toward 1%. If p > 0.5, push toward 99%.
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

    # --- REQUIRED ABSTRACT METHODS ---
    
    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        p1 = await self._run_model_forecast("forecaster_1", question, research)
        p2 = await self._run_model_forecast("forecaster_2", question, research)
        
        avg_p = (p1 + p2) / 2
        deviation = abs(p1 - p2)
        bold_factor = 6.2 if deviation < 0.02 else 4.0
        final_p = extremize(avg_p, factor=bold_factor)
        final_p = max(0.01, min(0.99, final_p))
        
        reasoning = f"### [BOLD FORECAST]\n{research}\n\nConviction: GPT-5.2({p1:.1%}) & GPT-5.1({p2:.1%}). Factor: {bold_factor}."
        return ReasonedPrediction(prediction_value=final_p, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        p1 = await self._run_model_forecast("forecaster_1", question, research)
        p2 = await self._run_model_forecast("forecaster_2", question, research)
        
        combined_options = {}
        for opt in question.options:
            v1 = next((o.probability for o in p1.predicted_options if o.option_name == opt), 0.0)
            v2 = next((o.probability for o in p2.predicted_options if o.option_name == opt), 0.0)
            combined_options[opt] = extremize((v1 + v2) / 2, factor=5.0)
        
        total = sum(combined_options.values())
        final_options = [PredictedOption(option_name=k, probability=v/total) for k, v in combined_options.items()]
        prediction_list = PredictedOptionList(predicted_options=final_options)
        
        # Explicit normalization
        if prediction_list.predicted_options:
            norm_total = sum(opt.probability for opt in prediction_list.predicted_options)
            if norm_total > 0 and abs(norm_total - 1.0) > 0.001:
                for opt in prediction_list.predicted_options:
                    opt.probability /= norm_total
        
        return ReasonedPrediction(prediction_value=prediction_list, reasoning=f"### [BOLD MC]\n{research}")

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        p1 = await self._run_model_forecast("forecaster_1", question, research)
        p2 = await self._run_model_forecast("forecaster_2", question, research)
        
        if not isinstance(p1, list) or not isinstance(p2, list):
            raise ValueError("LLM did not return a list of percentiles")
        
        if len(p1) < 2 or len(p2) < 2:
            # Use safe fallback bounds from question attributes
            low = getattr(question, 'minimum', 0)
            high = getattr(question, 'maximum', max(1, abs(low) * 2))
            logger.warning(f"Insufficient percentiles ({len(p1)}/{len(p2)}). Using bounds [{low}, {high}].")
            default_percentiles = [
                Percentile(percentile=10, value=low),
                Percentile(percentile=50, value=(low + high) / 2),
                Percentile(percentile=90, value=high),
            ]
            dist = NumericDistribution.from_question(default_percentiles, question)
            return ReasonedPrediction(prediction_value=dist, reasoning=f"### [BOLD NUMERIC - DEFAULTS]\n{research}")
        
        # Align by percentile key
        p1_map = {p.percentile: p.value for p in p1}
        p2_map = {p.percentile: p.value for p in p2}
        all_percentiles = sorted(set(p1_map.keys()) | set(p2_map.keys()))
        
        final_percentiles = []
        for pct in all_percentiles:
            v1 = p1_map.get(pct, p2_map[pct])
            v2 = p2_map.get(pct, p1_map[pct])
            avg_val = (v1 + v2) / 2
            scaled_val = self._scale_numeric_prediction(avg_val, question)
            final_percentiles.append(Percentile(percentile=pct, value=scaled_val))
        
        # Final safety: ensure at least 2 percentiles
        if len(final_percentiles) < 2:
            low = getattr(question, 'minimum', 0)
            high = getattr(question, 'maximum', 100)
            final_percentiles = [
                Percentile(percentile=10, value=low),
                Percentile(percentile=90, value=high),
            ]
        
        # Attempt to create distribution, with clamping if needed
        try:
            dist = NumericDistribution.from_question(final_percentiles, question)
        except ValueError as e:
            logger.warning(f"Validation failed, clamping to bounds: {e}")
            low = getattr(question, 'minimum', float('-inf'))
            high = getattr(question, 'maximum', float('inf'))
            clamped_percentiles = [
                Percentile(percentile=p.percentile, value=max(low, min(high, p.value)))
                for p in final_percentiles
            ]
            dist = NumericDistribution.from_question(clamped_percentiles, question)
        
        return ReasonedPrediction(prediction_value=dist, reasoning=f"### [BOLD NUMERIC]\n{research}")

# --- MAIN ---
if __name__ == "__main__":
    bot = Yrambot(publish_reports_to_metaculus=True, skip_previously_forecasted_questions=True)
    async def run():
        tournaments = ["32916", "market-pulse-26q1", MetaculusApi.CURRENT_MINIBENCH_ID]
        for tid in tournaments:
            logger.info(f"Forecasting on tournament: {tid}")
            await bot.forecast_on_tournament(tid, return_exceptions=True)
    asyncio.run(run())
