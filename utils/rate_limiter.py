"""Rate limiting utilities for LLM API calls.

This module provides rate limiting functionality to prevent excessive API usage
and handle provider rate limits gracefully.
"""

import os
import time
from collections import deque
from threading import Lock
from typing import Optional

from .logging import get_logger

logger = get_logger(__name__)


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""

    pass


class RateLimiter:
    """Rate limiter using a sliding window algorithm.

    This limiter tracks API calls within a time window and blocks calls
    that would exceed the configured rate limit.
    """

    def __init__(
        self,
        max_calls: int = 60,
        time_window: float = 60.0,
        min_interval: Optional[float] = None,
    ):
        """Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in time_window seconds
            time_window: Time window in seconds (default: 60 seconds)
            min_interval: Minimum interval between calls in seconds (optional)
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.min_interval = min_interval
        self.call_times: deque = deque()
        self.last_call_time: Optional[float] = None
        self.lock = Lock()
        logger.debug(
            f"RateLimiter initialized: max_calls={max_calls}, "
            f"time_window={time_window}s, min_interval={min_interval}s"
        )

    def acquire(self, wait: bool = True, timeout: Optional[float] = None) -> bool:
        """Acquire permission to make an API call.

        Args:
            wait: If True, wait until rate limit allows the call
            timeout: Maximum time to wait in seconds (None = wait indefinitely)

        Returns:
            True if permission granted, False if timeout exceeded

        Raises:
            RateLimitError: If rate limit exceeded and wait=False
        """
        with self.lock:
            current_time = time.time()

            # Check minimum interval between calls
            if self.min_interval and self.last_call_time is not None:
                elapsed = current_time - self.last_call_time
                if elapsed < self.min_interval:
                    wait_time = self.min_interval - elapsed
                    if not wait:
                        raise RateLimitError(
                            f"Rate limit: minimum interval {self.min_interval}s not met. "
                            f"Wait {wait_time:.2f}s before next call."
                        )
                    if timeout is None or wait_time <= timeout:
                        time.sleep(wait_time)
                        current_time = time.time()
                    else:
                        return False

            # Remove calls outside the time window
            cutoff_time = current_time - self.time_window
            while self.call_times and self.call_times[0] < cutoff_time:
                self.call_times.popleft()

            # Check if we're at the limit
            if len(self.call_times) >= self.max_calls:
                if not wait:
                    raise RateLimitError(
                        f"Rate limit exceeded: {self.max_calls} calls per {self.time_window}s. "
                        f"Wait {self.time_window - (current_time - self.call_times[0]):.2f}s"
                    )

                # Calculate wait time until oldest call expires
                oldest_call = self.call_times[0]
                wait_time = self.time_window - (current_time - oldest_call) + 0.1  # Small buffer

                if timeout is not None and wait_time > timeout:
                    logger.warning(
                        f"Rate limit wait time {wait_time:.2f}s exceeds timeout {timeout}s"
                    )
                    return False

                logger.info(f"Rate limit reached. Waiting {wait_time:.2f}s before next call...")
                time.sleep(wait_time)
                current_time = time.time()

                # Re-check after waiting
                cutoff_time = current_time - self.time_window
                while self.call_times and self.call_times[0] < cutoff_time:
                    self.call_times.popleft()

            # Record this call
            self.call_times.append(current_time)
            self.last_call_time = current_time
            return True

    def reset(self) -> None:
        """Reset rate limiter state."""
        with self.lock:
            self.call_times.clear()
            self.last_call_time = None
            logger.debug("Rate limiter reset")

    def get_stats(self) -> dict:
        """Get current rate limiter statistics.

        Returns:
            Dictionary with current stats
        """
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - self.time_window
            active_calls = sum(1 for t in self.call_times if t >= cutoff_time)

            return {
                "active_calls": active_calls,
                "max_calls": self.max_calls,
                "time_window": self.time_window,
                "remaining_calls": max(0, self.max_calls - active_calls),
                "last_call_time": self.last_call_time,
            }


def get_rate_limiter(
    provider: Optional[str] = None,
    max_calls: Optional[int] = None,
    time_window: Optional[float] = None,
    min_interval: Optional[float] = None,
) -> RateLimiter:
    """Get a rate limiter with provider-specific defaults.

    Rate limits can be configured via environment variables:
    - AGENTPREP_LLM_RATE_LIMIT_MAX_CALLS: Maximum calls per window
    - AGENTPREP_LLM_RATE_LIMIT_TIME_WINDOW: Time window in seconds
    - AGENTPREP_LLM_RATE_LIMIT_MIN_INTERVAL: Minimum interval between calls

    Provider-specific defaults:
    - OpenAI: 60 calls/minute, 1s min interval
    - Anthropic: 50 calls/minute, 1.2s min interval
    - Gemini: 60 calls/minute, 1s min interval

    Args:
        provider: LLM provider name (for provider-specific defaults)
        max_calls: Override max calls (or from env var)
        time_window: Override time window (or from env var)
        min_interval: Override min interval (or from env var)

    Returns:
        Configured RateLimiter instance
    """
    # Get defaults from environment or use provider-specific defaults
    env_max_calls = os.getenv("AGENTPREP_LLM_RATE_LIMIT_MAX_CALLS")
    env_time_window = os.getenv("AGENTPREP_LLM_RATE_LIMIT_TIME_WINDOW")
    env_min_interval = os.getenv("AGENTPREP_LLM_RATE_LIMIT_MIN_INTERVAL")

    # Provider-specific defaults
    provider_defaults = {
        "openai": {"max_calls": 60, "time_window": 60.0, "min_interval": 1.0},
        "anthropic": {"max_calls": 50, "time_window": 60.0, "min_interval": 1.2},
        "gemini": {"max_calls": 60, "time_window": 60.0, "min_interval": 1.0},
    }

    # Determine values (env var > parameter > provider default > global default)
    if max_calls is None:
        max_calls = (
            int(env_max_calls)
            if env_max_calls
            else provider_defaults.get(provider or "", {}).get("max_calls", 60)
        )

    if time_window is None:
        time_window = (
            float(env_time_window)
            if env_time_window
            else provider_defaults.get(provider or "", {}).get("time_window", 60.0)
        )

    if min_interval is None:
        min_interval = (
            float(env_min_interval)
            if env_min_interval
            else provider_defaults.get(provider or "", {}).get("min_interval", 1.0)
        )

    return RateLimiter(
        max_calls=max_calls,
        time_window=time_window,
        min_interval=min_interval,
    )
