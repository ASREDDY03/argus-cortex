"""
Retry utilities for Anthropic API calls.

Two retry layers:
  1. API-level retries   — handles rate limits, connection errors, timeouts
  2. JSON-level retries  — handles malformed LLM output (re-prompts for clean JSON)
"""
import time
import json
import logging
from functools import wraps
from typing import Callable
import anthropic

logger = logging.getLogger(__name__)

# Retry config
MAX_API_ATTEMPTS = 3
BACKOFF_SECONDS = [1, 2, 4]     # wait 1s, then 2s, then 4s

MAX_JSON_ATTEMPTS = 2            # re-prompt once if JSON is malformed

# Errors worth retrying (transient)
_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)

# Errors NOT worth retrying (permanent)
_FATAL = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
)


def retry_api(func: Callable) -> Callable:
    """
    Decorator — retries Anthropic API calls with exponential backoff.
    Apply to any function that calls client.messages.create().

    Fatal errors (bad API key, 404) are raised immediately.
    Transient errors (rate limit, connection) are retried up to MAX_API_ATTEMPTS times.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        last_error = None
        for attempt, wait in enumerate(BACKOFF_SECONDS[:MAX_API_ATTEMPTS], start=1):
            try:
                return func(*args, **kwargs)
            except _FATAL:
                raise
            except _RETRYABLE as e:
                last_error = e
                if attempt < MAX_API_ATTEMPTS:
                    logger.warning(
                        f"[retry] {func.__name__} attempt {attempt}/{MAX_API_ATTEMPTS} "
                        f"failed ({type(e).__name__}). Retrying in {wait}s..."
                    )
                    time.sleep(wait)
            except anthropic.APIError as e:
                # Catch-all for other API errors — retry once
                last_error = e
                if attempt < MAX_API_ATTEMPTS:
                    logger.warning(f"[retry] API error on {func.__name__}: {e}. Retrying...")
                    time.sleep(wait)
        raise last_error
    return wrapper


def parse_json_with_retry(
    raw: str,
    kind: str,           # "array" or "object"
    reprompt_fn: Callable[[], str],
) -> list | dict:
    """
    Try to parse JSON from raw LLM output.
    If parsing fails, call reprompt_fn() to get a cleaner response and try once more.

    kind="array"  → extract [...] from raw
    kind="object" → extract {...} from raw
    """
    def _extract(text: str) -> str:
        if kind == "array":
            s, e = text.find("["), text.rfind("]") + 1
        else:
            s, e = text.find("{"), text.rfind("}") + 1
        if s == -1 or e == 0:
            raise ValueError(f"No JSON {kind} found in response")
        return text[s:e]

    for attempt in range(1, MAX_JSON_ATTEMPTS + 1):
        try:
            chunk = _extract(raw if attempt == 1 else reprompt_fn())
            return json.loads(chunk)
        except (ValueError, json.JSONDecodeError) as e:
            if attempt == MAX_JSON_ATTEMPTS:
                logger.error(f"[retry] JSON parse failed after {attempt} attempts: {e}")
                return [] if kind == "array" else {}
            logger.warning(f"[retry] JSON parse attempt {attempt} failed. Re-prompting...")
