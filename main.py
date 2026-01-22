import argparse
import asyncio
import logging
import os
import textwrap
import re
import math
from datetime import datetime
from typing import List, Union
from tavily import TavilyClient
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

def extremize(p: float, factor: float = 5.8) -> float:
    p = max(0.005, min(0.995, p))
    odds = p / (1 - p)
    extremized_odds = odds**factor
    return extremized_odds / (1 + extremized_odds)

def build_tavily_query(question: MetaculusQuestion, max_chars: int = 397) -> str:
    q = re.sub(r"http\S+", "", question.question_text.strip())
    q = re.sub(r"\s+", " ", q).strip()
    return textwrap.shorten(q, width=max_chars, placeholder="…")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Yrambot")

class Yrambot(ForecastBot):
    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5.2",
            "parser": "openrouter/openai/gpt-4.1-mini",
            "forecaster_1": "openrouter/openai/gpt-5.2",
            "forecaster_2": "openrouter/openai/gpt-5.1",
        }

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
                tavily_summary = "\n".join([f"[{i+1}] {r['title']}: {r['content'][:350]}" for i, r in enumerate(res)])
            except Exception as e:
                tavily_summary = f"Research failed: {e}"
            return f"--- REAL-TIME MARKET INTELLIGENCE ({today_str}) ---\n{tavily_summary}"

    async def _run_model_forecast(self, model_key: str, question: MetaculusQuestion, research: str) -> Union[float, PredictedOptionList, List[Percentile]]:
        prompt = clean_indents(f"""
            Role: Elite Quantitative Superforecaster. Today: {datetime.now().strftime('%Y-%m-%d')}
            Question: {question.question_text}
            Background: {question.background_info}
            Research: {research}
            Mandate: 
            1. Establish the Base Rate: What is the historical frequency?
            2. Adjust for Signal: Use research to refine the probability.
            3. Be Bold: If the evidence points to a 'No' (below 50%), push toward 1%-15%. If it points to a 'Yes' (above 50%), push toward 87%-99%.
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

    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        p1 = await self._run_model_forecast("forecaster_1", question, research)
        p2 = await self._run_model_forecast("forecaster_2", question, research)
        comment_header = f"### [YRAMBOT BOLD FORECAST]\n\n{research}\n\n### STATISTICAL SYNTHESIS\n"
        if isinstance(question, BinaryQuestion):
            avg_p = (p1 + p2) / 2
            deviation = abs(p1 - p2)
            bold_factor = 5.8 if deviation < 0.05 else 3.8
            final_p = extremize(avg_p, factor=bold_factor)
            final_p = max(0.01, min(0.99, final_p))
            reasoning = comment_header + f"GPT-5.2: {p1:.1%} | GPT-5.1: {p2:.1%}\nBoldness Factor: {bold_factor}. Final: {final_p:.1%}"
            return ReasonedPrediction(prediction_value=final_p, reasoning=reasoning)
        elif isinstance(question, MultipleChoiceQuestion):
            combined_options = {}
            for opt in question.options:
                v1 = next((o.probability for o in p1.predicted_options if o.option_name == opt), 0.0)
                v2 = next((o.probability for o in p2.predicted_options if o.option_name == opt), 0.0)
                combined_options[opt] = extremize((v1 + v2) / 2, factor=4.5)
            total = sum(combined_options.values())
            final_options = [PredictedOption(option_name=k, probability=v/total) for k, v in combined_options.items()]
            return ReasonedPrediction(prediction_value=PredictedOptionList(predicted_options=final_options), reasoning=comment_header)
        elif isinstance(question, NumericQuestion):
            final_percentiles = []
            for i in range(len(p1)):
                avg_val = (p1[i].value + p2[i].value) / 2
                final_percentiles.append(Percentile(percentile=p1[i].percentile, value=avg_val))
            dist = NumericDistribution.from_question(final_percentiles, question)
            return ReasonedPrediction(prediction_value=dist, reasoning=comment_header)

@model_validator(mode='after')
def _fixed_normalize_probabilities(self: PredictedOptionList):
    if not self.predicted_options: return self
    sum_ = sum(p.probability for p in self.predicted_options)
    if sum_ > 0 and abs(sum_ - 1.0) > 0.001:
        for option in self.predicted_options:
            option.probability /= sum_
    return self

PredictedOptionList.__pydantic_post_validate__ = _fixed_normalize_probabilities

if __name__ == "__main__":
    bot = Yrambot(publish_reports_to_metaculus=True, skip_previously_forecasted_questions=True)
    async def run():
        tournaments = ["32916", "market-pulse-26q1", MetaculusApi.CURRENT_MINIBENCH_ID]
        for tid in tournaments:
            await bot.forecast_on_tournament(tid, return_exceptions=True)
    asyncio.run(run())
