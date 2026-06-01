#!/usr/bin/env python3
"""
Diagnostic script to help troubleshoot NotFoundError and other API issues.

Run this script to check:
1. API key availability
2. Model endpoint availability on OpenRouter
3. API connectivity
4. Research source availability

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

# Load environment variables
dotenv.load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - Diagnostics - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Diagnostics")

# Model configuration (from main.py)
_FREE_MODEL_CHAIN = [
    "openrouter/perplexity/sonar",
    "openrouter/perplexity/sonar-pro",
]

_PRIMARY_MODELS = [
    "openrouter/perplexity/sonar-reasoning-pro",
    "openrouter/perplexity/sonar-pro",
    "openrouter/perplexity/sonar",
]

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")


async def test_model_availability(model: str, timeout: int = 30) -> dict:
    """Test if a specific model is available on OpenRouter."""
    logger.info(f"Testing model: {model}")
    
    result = {
        "model": model,
        "status": "untested",
        "response": None,
        "error": None,
    }
    
    try:
        llm = GeneralLlm(model=model, temperature=0.15, timeout=timeout, allowed_tries=1)
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
        
        # Special handling for NotFoundError
        if "notfound" in str(e).lower() or error_type == "NotFoundError":
            logger.error(f"✗ Model {model} NOT FOUND on OpenRouter")
            logger.error(f"  This suggests the model endpoint does not exist or has been removed.")
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
    
    # Check API keys
    logger.info("\n1. Checking API Keys...")
    logger.info("-" * 40)
    
    env_checks = {
        "OPENROUTER_API_KEY": bool(OPENROUTER_API_KEY),
        "TAVILY_API_KEY": bool(TAVILY_API_KEY),
        "METACULUS_TOKEN": bool(METACULUS_TOKEN),
    }
    
    for key, present in env_checks.items():
        status = "✓ Present" if present else "✗ Missing"
        logger.info(f"  {key}: {status}")
        report["environment"][key] = present
    
    if not OPENROUTER_API_KEY:
        report["recommendations"].append(
            "OPENROUTER_API_KEY is missing. This is required for model access. "
            "Get one at https://openrouter.ai/"
        )
    
    # Check model availability
    logger.info("\n2. Checking Model Availability...")
    logger.info("-" * 40)
    
    all_models = list(set(_PRIMARY_MODELS + _FREE_MODEL_CHAIN))
    
    for model in all_models:
        test_result = await test_model_availability(model)
        report["models"][model] = test_result
        
        if test_result["status"] == "error" and "not found" in test_result.get("error", "").lower():
            report["recommendations"].append(
                f"Model '{model}' not found on OpenRouter. "
                "This causes NotFoundError exceptions. "
                "Please update the model name or use an available model."
            )
    
    # Check for NotFoundError patterns
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
    
    # Provide recommendations
    if report["recommendations"]:
        logger.warning(f"\n💡 Recommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            logger.warning(f"  {i}. {rec}")
    
    # Export report
    logger.info("\n4. Full Report")
    logger.info("-" * 40)
    logger.info(json.dumps(report, indent=2))
    
    # Write report to file
    report_file = "api_diagnostics.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport saved to {report_file}")
    
    return report


async def main():
    """Main entry point."""
    try:
        report = await diagnose_all()
        
        # Exit with error if critical issues found
        if not report["environment"]["OPENROUTER_API_KEY"]:
            logger.error("\nCritical: OPENROUTER_API_KEY not set. Cannot proceed.")
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
