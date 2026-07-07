#!/usr/bin/env python3
"""
Diagnostic script to help troubleshoot Vultr inference and research API issues.

Run this script to check:
1. API key availability
2. Vultr model endpoint availability
3. Research source configuration

Usage:
    python diagnose_api_issues.py
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime

import dotenv
from forecasting_tools import GeneralLlm

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - Diagnostics - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Diagnostics")

VULTR_API_BASE = os.getenv("VULTR_INFERENCE_API_BASE", "https://api.vultrinference.com/v1")
VULTR_API_KEY  = os.getenv("VULTR_SERVERLESS_INFERENCE_API_KEY", "")

_VULTR_MODELS = [
    os.getenv("VULTR_MODEL_PRIMARY",   "openai/deepseek-r1-distill-llama-70b"),
    os.getenv("VULTR_MODEL_SECONDARY", "openai/llama-3.3-70b-instruct-fp8"),
    os.getenv("VULTR_MODEL_PARSER",    "openai/qwen2.5-32b-instruct"),
]

TAVILY_API_KEY  = os.getenv("TAVILY_API_KEY")
EXA_API_KEY     = os.getenv("EXA_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")


async def test_model_availability(model: str, timeout: int = 60) -> dict:
    """Test if a specific Vultr model is available."""
    logger.info(f"Testing model: {model}")

    result = {
        "model": model,
        "status": "untested",
        "response": None,
        "error": None,
    }

    try:
        llm = GeneralLlm(
            model=model,
            temperature=0.15,
            timeout=timeout,
            allowed_tries=1,
            api_key=VULTR_API_KEY,
            base_url=VULTR_API_BASE,
        )
        response = await asyncio.wait_for(llm.invoke("Say 'OK'"), timeout=timeout)

        result["status"] = "success" if response and "ok" in response.lower() else "unexpected_response"
        result["response"] = response[:150] if response else None

        if result["status"] == "success":
            logger.info(f"✓ Model {model} is available")
        else:
            logger.warning(f"⚠ Model {model} returned unexpected response: {response[:100]}")

    except asyncio.TimeoutError:
        result["status"] = "timeout"
        result["error"] = "Request timed out"
        logger.error(f"✗ Model {model} timed out after {timeout}s")

    except Exception as e:
        error_type = type(e).__name__
        result["status"] = "error"
        result["error"] = f"{error_type}: {str(e)[:200]}"

        if "notfound" in str(e).lower() or error_type == "NotFoundError":
            logger.error(f"✗ Model {model} NOT FOUND on Vultr")
        else:
            logger.error(f"✗ Model {model} error: {error_type}: {str(e)[:100]}")

    return result


async def diagnose_all() -> dict:
    """Run all diagnostic checks."""
    logger.info("=" * 80)
    logger.info("API Health Diagnostic Report")
    logger.info("=" * 80)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "environment": {},
        "models": {},
        "recommendations": [],
    }

    logger.info("\n1. Checking API Keys...")
    logger.info("-" * 40)

    env_checks = {
        "VULTR_SERVERLESS_INFERENCE_API_KEY": bool(VULTR_API_KEY),
        "TAVILY_API_KEY":  bool(TAVILY_API_KEY),
        "EXA_API_KEY":     bool(EXA_API_KEY),
        "SERPAPI_API_KEY": bool(SERPAPI_API_KEY),
        "METACULUS_TOKEN": bool(METACULUS_TOKEN),
    }

    for key, present in env_checks.items():
        status = "✓ Present" if present else "✗ Missing"
        logger.info(f"  {key}: {status}")
        report["environment"][key] = present

    if not VULTR_API_KEY:
        report["recommendations"].append(
            "VULTR_SERVERLESS_INFERENCE_API_KEY is missing. "
            "Get one from the Vultr Console → Serverless Inference."
        )

    if not TAVILY_API_KEY and not EXA_API_KEY and not SERPAPI_API_KEY:
        report["recommendations"].append(
            "No research API keys set (TAVILY_API_KEY, EXA_API_KEY, SERPAPI_API_KEY). "
            "Research quality will be limited without at least one search provider."
        )

    logger.info("\n2. Checking Vultr Model Availability...")
    logger.info("-" * 40)

    all_models = list(dict.fromkeys(_VULTR_MODELS))

    for model in all_models:
        test_result = await test_model_availability(model)
        report["models"][model] = test_result

        if test_result["status"] == "error" and "not found" in test_result.get("error", "").lower():
            report["recommendations"].append(
                f"Model '{model}' not found on Vultr. "
                "Check available models: GET https://api.vultrinference.com/v1/models"
            )

    logger.info("\n3. Summary & Recommendations")
    logger.info("-" * 40)

    failed_models = [
        m for m, r in report["models"].items()
        if r["status"] in ("error", "timeout")
    ]

    if failed_models:
        logger.warning(f"\n⚠ {len(failed_models)} model(s) failed:")
        for model in failed_models:
            status = report["models"][model]
            logger.warning(f"  - {model}: {status['status']}")
            if status.get("error"):
                logger.warning(f"    Error: {status['error'][:100]}")

    working_models = [
        m for m, r in report["models"].items()
        if r["status"] == "success"
    ]

    if working_models:
        logger.info(f"\n✓ {len(working_models)} model(s) available:")
        for model in working_models:
            logger.info(f"  - {model}")

    if report["recommendations"]:
        logger.warning("\n💡 Recommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            logger.warning(f"  {i}. {rec}")

    logger.info("\n4. Full Report")
    logger.info("-" * 40)
    logger.info(json.dumps(report, indent=2))

    report_file = "api_diagnostics.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport saved to {report_file}")

    return report


async def main():
    try:
        report = await diagnose_all()

        if not report["environment"]["VULTR_SERVERLESS_INFERENCE_API_KEY"]:
            logger.error("\nCritical: VULTR_SERVERLESS_INFERENCE_API_KEY not set. Cannot proceed.")
            sys.exit(1)

        failed_count = sum(
            1 for r in report["models"].values()
            if r["status"] in ("error", "timeout")
        )

        if failed_count > 0:
            logger.warning(f"\n⚠ {failed_count} models failed. See recommendations above.")
            sys.exit(1)

        logger.info("\n✓ Diagnostics complete - All systems operational")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Diagnostic failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
