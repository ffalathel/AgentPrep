"""LLM client initialization utilities.

This module provides functions to initialize LLM clients from environment variables.
Supports multiple providers (OpenAI, Anthropic, Gemini) with graceful fallback.
"""

import os
from typing import Any, Optional

from .logging import get_logger

logger = get_logger(__name__)


class LLMClientWrapper:
    """Wrapper to provide unified interface for different LLM providers."""

    def __init__(self, client: Any, provider: str):
        """Initialize wrapper with LLM client and provider name.

        Args:
            client: The actual LLM client instance
            provider: Provider name ('openai', 'anthropic', 'gemini')
        """
        self.client = client
        self.provider = provider

    def complete(self, prompt: str, **kwargs) -> str:
        """Call LLM with prompt and return response text.

        Args:
            prompt: Input prompt string
            **kwargs: Additional arguments for LLM call

        Returns:
            LLM response text
        """
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


def get_llm_client() -> Optional[Any]:
    """Get LLM client from environment variables.

    Checks for API keys in the following order:
    1. OPENAI_API_KEY (OpenAI GPT models)
    2. ANTHROPIC_API_KEY (Anthropic Claude models)
    3. GEMINI_API_KEY (Google Gemini models)

    Returns:
        LLM client instance if API key found, None otherwise

    Note:
        If no API key is found, returns None. Agents will work in stub mode
        (returning empty proposals) which is fine for testing.
    """
    # Try OpenAI first
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
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

    # Try Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
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

    # Try Gemini
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai  # type: ignore

            logger.info("Initializing Gemini client from GEMINI_API_KEY")
            genai.configure(api_key=gemini_key)
            # Return wrapper with genai module as client
            return LLMClientWrapper(genai, "gemini")
        except ImportError:
            logger.warning(
                "GEMINI_API_KEY found but google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            return None

    # No API key found
    logger.debug(
        "No LLM API key found in environment variables. "
        "Agents will run in stub mode (no LLM proposals). "
        "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY to enable LLM features."
    )
    return None
