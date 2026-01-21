"""LLM client initialization utilities.

This module provides functions to initialize LLM clients from environment variables.
Supports multiple providers (OpenAI, Anthropic, Gemini) with graceful fallback.
"""

import os
import time
from typing import Any, Optional

from .logging import get_logger
from .rate_limiter import RateLimitError, get_rate_limiter

logger = get_logger(__name__)


class LLMClientWrapper:
    """Wrapper to provide unified interface for different LLM providers.

    Includes rate limiting to prevent excessive API usage and handle
    provider rate limits gracefully.
    """

    def __init__(self, client: Any, provider: str):
        """Initialize wrapper with LLM client and provider name.

        Args:
            client: The actual LLM client instance
            provider: Provider name ('openai', 'anthropic', 'gemini')
        """
        self.client = client
        self.provider = provider
        # Initialize rate limiter with provider-specific defaults
        self.rate_limiter = get_rate_limiter(provider=provider)
        logger.debug(f"LLMClientWrapper initialized for provider: {provider}")

    def complete(self, prompt: str, **kwargs) -> str:
        """Call LLM with prompt and return response text.

        Includes rate limiting to prevent excessive API usage.

        Args:
            prompt: Input prompt string
            **kwargs: Additional arguments for LLM call

        Returns:
            LLM response text

        Raises:
            ValueError: If provider is unknown
            RuntimeError: If LLM API call fails
            ConnectionError: If network connection fails
            TimeoutError: If request times out
            RateLimitError: If rate limit is exceeded and cannot wait
        """
        # Acquire rate limit permission (wait if necessary)
        max_wait_time = kwargs.get("rate_limit_timeout", 300.0)  # Default 5 minutes
        try:
            if not self.rate_limiter.acquire(wait=True, timeout=max_wait_time):
                raise RateLimitError(
                    f"Rate limit timeout exceeded after {max_wait_time}s. "
                    "Too many LLM calls in a short period."
                )
        except RateLimitError as e:
            logger.error(f"Rate limit error: {e}")
            raise RuntimeError(f"LLM rate limit exceeded: {e}") from e

        # Make the actual API call
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=kwargs.get("model", "gpt-4"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", 0.7),
                )
                return response.choices[0].message.content

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=kwargs.get("model", "claude-3-opus-20240229"),
                    max_tokens=kwargs.get("max_tokens", 4096),
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text

            elif self.provider == "gemini":
                import google.generativeai as genai

                model = genai.GenerativeModel(
                    kwargs.get("model", "gemini-pro")
                )
                response = model.generate_content(prompt)
                return response.text

            else:
                raise ValueError(f"Unknown provider: {self.provider}")
        except (ConnectionError, TimeoutError) as e:
            raise RuntimeError(f"LLM API connection failed for provider {self.provider}: {e}") from e
        except (KeyError, AttributeError, IndexError) as e:
            raise RuntimeError(f"LLM API returned unexpected response format for provider {self.provider}: {e}") from e
        except Exception as e:
            # Check for rate limit errors from the API provider
            error_str = str(e).lower()
            if any(
                keyword in error_str
                for keyword in ["rate limit", "rate_limit", "429", "too many requests", "quota"]
            ):
                # Provider rate limit hit - wait and potentially retry
                wait_time = 60.0  # Default wait time
                logger.warning(
                    f"Provider rate limit hit for {self.provider}. "
                    f"Waiting {wait_time}s before retry..."
                )
                time.sleep(wait_time)
                # Reset our rate limiter to allow retry
                self.rate_limiter.reset()
                raise RuntimeError(
                    f"LLM API rate limit exceeded for provider {self.provider}. "
                    f"Please wait before retrying: {e}"
                ) from e

            # Catch API-specific errors (e.g., openai.APIError, anthropic.APIError)
            if "API" in str(type(e).__name__) or "Error" in str(type(e).__name__):
                raise RuntimeError(f"LLM API error for provider {self.provider}: {e}") from e
            raise RuntimeError(f"Unexpected error calling LLM API for provider {self.provider}: {e}") from e


from typing import Any, Optional

def _init_openai() -> Optional[LLMClientWrapper]:
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return None

    try:
        import openai
        logger.info("Initializing OpenAI client from OPENAI_API_KEY")
        client = openai.OpenAI(api_key=openai_key)
        return LLMClientWrapper(client, "openai")
    except ImportError:
        logger.warning(
            "OPENAI_API_KEY found but openai package not installed. "
            "Install with: pip install openai"
        )
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
    return None


def _init_anthropic() -> Optional[LLMClientWrapper]:
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        return None

    try:
        import anthropic  # type: ignore
        logger.info("Initializing Anthropic client from ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=anthropic_key)
        return LLMClientWrapper(client, "anthropic")
    except ImportError:
        logger.warning(
            "ANTHROPIC_API_KEY found but anthropic package not installed. "
            "Install with: pip install anthropic"
        )
    except Exception as e:
        logger.error(f"Failed to initialize Anthropic client: {e}")
    return None


def _init_gemini() -> Optional[LLMClientWrapper]:
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return None

    try:
        import google.generativeai as genai  # type: ignore
        logger.info("Initializing Gemini client from GEMINI_API_KEY")
        genai.configure(api_key=gemini_key)
        return LLMClientWrapper(genai, "gemini")
    except ImportError:
        logger.warning(
            "GEMINI_API_KEY found but google-generativeai package not installed. "
            "Install with: pip install google-generativeai"
        )
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
    return None


def get_llm_client(preferred_provider: Optional[str] = None) -> Optional[Any]:
    """Get LLM client from environment variables.

    If ``preferred_provider`` is provided, only that provider is attempted.
    Otherwise, providers are tried in the following order:

    1. OpenAI (OPENAI_API_KEY)
    2. Anthropic (ANTHROPIC_API_KEY)
    3. Gemini (GEMINI_API_KEY)
    """
    provider = preferred_provider.lower() if preferred_provider else None

    if provider == "openai":
        client = _init_openai()
        if client is None:
            logger.error(
                "Requested provider 'openai' but OPENAI_API_KEY is missing or client init failed."
            )
        return client

    if provider == "anthropic":
        client = _init_anthropic()
        if client is None:
            logger.error(
                "Requested provider 'anthropic' but ANTHROPIC_API_KEY is missing or client init failed."
            )
        return client

    if provider == "gemini":
        client = _init_gemini()
        if client is None:
            logger.error(
                "Requested provider 'gemini' but GEMINI_API_KEY is missing or client init failed."
            )
        return client

    # No preferred provider: fall back to default order
    for initializer in (_init_openai, _init_anthropic, _init_gemini):
        client = initializer()
        if client is not None:
            return client

    logger.debug(
        "No LLM API key found in environment variables. "
        "Agents will run in stub mode (no LLM proposals). "
        "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY to enable LLM features."
    )
    return None
