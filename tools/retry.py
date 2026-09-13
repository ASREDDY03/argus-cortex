"""
Retry utilities for Anthropic API calls.

Handles rate limits, connection errors, and timeouts with exponential backoff.
JSON parsing is no longer needed — all agents use tool_use for structured output.
"""
import time
import logging
from functools import wraps
from typing import Callable
import anthropic

logger = logging.getLogger(__name__)

# Retry config
MAX_API_ATTEMPTS = 3
BACKOFF_SECONDS = [1, 2, 4]     # wait 1s, then 2s, then 4s

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
