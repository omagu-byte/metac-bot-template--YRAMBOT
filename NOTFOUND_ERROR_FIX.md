# NotFoundError While Summarizing Research - Solution Guide

## Problem Summary

The bot was raising a `NotFoundError` exception while attempting to summarize research. This typically occurs when:

1. **Model endpoints are not found** on OpenRouter
2. **API keys are invalid or missing**
3. **Model names or paths have changed**
4. **Network connectivity issues**

## Root Cause Analysis

The error originates from the `invoke_with_free_model_fallback()` function which tries to use Perplexity models through OpenRouter to synthesize research. If all fallback models fail with NotFoundError, the research summarization fails entirely.

### Before (Problematic Code)
```python
async def invoke_with_free_model_fallback(...):
    # Limited error handling
    # All NotFoundError exceptions treated the same
    # Minimal logging for debugging
```

### After (Enhanced Code)
```python
async def invoke_with_free_model_fallback(...):
    # Specific handling for NotFoundError
    # Distinguishes model-not-found from other errors
    # Detailed logging for troubleshooting
    # Informative error messages
```

## Changes Made

### 1. **Enhanced Error Handling** (main.py)

The `invoke_with_free_model_fallback()` function now:
- Catches `NotFoundError` exceptions specifically
- Logs which model endpoints are unavailable
- Returns informative error messages
- Helps identify which models need updating

### 2. **Improved Research Synthesis** (main.py)

The `_synthesize_research()` method now:
- Detects API errors in responses
- Falls back to raw web research when synthesis fails
- Catches exceptions with detailed logging
- Prevents hard crashes from API errors

### 3. **Better Fallback Logic** (main.py)

The `run_research()` method now:
- Distinguishes between API errors and thin synthesis
- Uses raw web bundle when synthesis fails due to API errors
- Respects the REQUIRE_RESEARCH configuration
- Provides clear error messages

### 4. **Diagnostic Tools**

- **`diagnose_api_health()`** - Async function in main.py to test API connectivity
- **`diagnose_api_issues.py`** - Standalone diagnostic script

## How to Troubleshoot

### Step 1: Run the Diagnostic Script

```bash
python diagnose_api_issues.py
```

This will:
- Check if OPENROUTER_API_KEY is set
- Test each model in the fallback chain
- Identify which models are unavailable
- Generate a diagnostic report

### Step 2: Review the Diagnostic Report

The script generates `api_diagnostics.json` with detailed information:
- API key status
- Model availability for each endpoint
- Specific error messages
- Recommendations

### Step 3: Apply Recommendations

Based on the diagnostic output:

**If models are not found:**
- Update the model names in main.py
- Check OpenRouter's current model list
- Use available models that provide similar functionality

**If API key is missing:**
```bash
# Add to .env file
OPENROUTER_API_KEY=your_key_here
```

**If all models fail:**
- Check your internet connection
- Verify OpenRouter is accessible
- Check API rate limits

## Configuration Variables

### Model Fallback Chain (main.py)
```python
_FREE_MODEL_CHAIN = [
    "openrouter/perplexity/sonar",
    "openrouter/perplexity/sonar-pro",
]
```

You can add more models or update these if they're unavailable:
```python
_FREE_MODEL_CHAIN = [
    "openrouter/perplexity/sonar",
    "openrouter/perplexity/sonar-pro",
    "openrouter/your-new-model/endpoint",  # Add new fallback
]
```

### Research Requirements
```bash
# .env file
REQUIRE_RESEARCH=true  # Enforce research availability (default: true)
```

When `REQUIRE_RESEARCH=true`:
- Bot won't forecast if research is completely unavailable
- Helps ensure high-quality forecasts

When `REQUIRE_RESEARCH=false`:
- Bot attempts to forecast even without research
- Provides warnings in logs

## Logging Output

### Before and After Examples

**Before (Unhelpful):**
```
ERROR: NotFoundError while summarizing research
```

**After (Helpful):**
```
ERROR [Free Model Fallback] Model not found on OpenRouter: openrouter/perplexity/sonar-reasoning-pro.
This may indicate the model endpoint does not exist or has been removed.

ERROR [Free Model Fallback] All models exhausted for research_synthesis. 
Last error: HTTP 404: Model endpoint not found.

WARNING [Research] Using raw web bundle as fallback due to API error.
```

## Common Issues and Solutions

### Issue: "Model not found" for sonar-reasoning-pro

**Solution:** This model may no longer be available on OpenRouter. 
Update `_FREE_MODEL_CHAIN` to use available models:

```python
_FREE_MODEL_CHAIN = [
    "openrouter/perplexity/sonar",
    "openrouter/perplexity/sonar-pro",
]
```

### Issue: "OPENROUTER_API_KEY is missing"

**Solution:** Set your API key in the .env file:

```bash
cp .env.template .env
# Edit .env and add your OPENROUTER_API_KEY
```

### Issue: All models timeout

**Solution:** Increase timeout values:

```python
# main.py
RESEARCH_TIMEOUT_S = 60.0  # Increase from 25s
LLM_TIMEOUT_S = 120.0      # Increase from 70s
```

### Issue: Research synthesis consistently fails but forecasting continues

**Solution:** This is expected behavior with the new error handling:
- Raw web research is used as fallback
- Bot still produces forecasts
- Check logs for error details

## Testing the Fix

### Unit Test
```python
import asyncio
from main import diagnose_api_health

# Run diagnostics
result = asyncio.run(diagnose_api_health())
print(result)
```

### Integration Test
```bash
# Test a single tournament
python main.py --mode run --tournament-ids minibench
```

## Performance Impact

- **Minimal:** Diagnostic function adds ~10-30 seconds to startup (optional)
- **Fallback chain:** Usually resolves on second model attempt
- **Error handling:** Graceful degradation, no performance penalty

## Future Improvements

1. **Model Registry:** Maintain list of verified working models
2. **Automatic Retry:** Retry with different parameters before failing
3. **Health Checks:** Periodic model availability monitoring
4. **Cached Research:** Store successful research synthesis for reuse

## Support

If issues persist after running diagnostics:

1. Check the `api_diagnostics.json` report
2. Review logs for specific error messages
3. Verify API keys and network connectivity
4. Check OpenRouter's status page
5. Contact support with the diagnostic report

## References

- OpenRouter Documentation: https://openrouter.ai/docs
- Perplexity Models: https://openrouter.ai/docs/models/perplexity
- Forecasting Tools: https://github.com/Metaculus/forecasting-tools
- Metaculus Discord: https://discord.com/invite/NJgCC2nDfh
