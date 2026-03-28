# ============================================================
# LLM configuration — API key, model, rate limits, provider
# Stage 1: Offline Teacher (llm_teacher)
#
# Supported providers:
#   "anthropic" — Claude models via Anthropic SDK
#   "gemini"    — Gemini models via Google GenAI SDK
# ============================================================

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

logger = logging.getLogger(__name__)

LLMProvider = Literal["anthropic", "gemini"]

_VALID_PROVIDERS: set[str] = {"anthropic", "gemini"}

_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_PROMPT_VERSION = "v1"

# Provider → model_id prefix whitelist (sanity check, not exhaustive)
_PROVIDER_MODEL_PREFIXES: dict[str, list[str]] = {
    "anthropic": ["claude-"],
    "gemini": ["gemini-", "models/gemini-"],
}


def resolve_provider_api_key(provider: str) -> str:
    """Resolve the API key for the given provider from environment variables.

    anthropic → ANTHROPIC_API_KEY
    gemini    → GEMINI_API_KEY, fallback GOOGLE_API_KEY
    """
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY", "")
    elif provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            key = os.environ.get("GOOGLE_API_KEY", "")
        return key
    return ""


def validate_provider(provider: str) -> None:
    """Raise ValueError if provider is not supported."""
    if provider not in _VALID_PROVIDERS:
        raise ValueError(
            f"Unknown llm_provider '{provider}'. "
            f"Supported: {sorted(_VALID_PROVIDERS)}"
        )


def validate_provider_model(provider: str, model_id: str) -> None:
    """Warn if model_id doesn't match provider's expected prefixes."""
    prefixes = _PROVIDER_MODEL_PREFIXES.get(provider, [])
    if prefixes and not any(model_id.startswith(p) for p in prefixes):
        logger.warning(
            "model_id '%s' does not match expected prefixes for provider '%s': %s. "
            "Proceeding anyway — ensure this is intentional.",
            model_id, provider, prefixes,
        )


def check_provider_sdk(provider: str) -> tuple[bool, str]:
    """Check if the SDK for the given provider is importable.

    Returns (ok, message).
    """
    if provider == "anthropic":
        try:
            import anthropic  # noqa: F401
            return True, "anthropic SDK available"
        except ImportError:
            return False, "anthropic SDK not installed: pip install anthropic"
    elif provider == "gemini":
        try:
            import google.generativeai  # noqa: F401
            return True, "google-generativeai SDK available"
        except ImportError:
            return False, "google-generativeai SDK not installed: pip install google-generativeai"
    return False, f"Unknown provider: {provider}"


def preflight_check_key(provider: str) -> tuple[bool, str]:
    """Verify that the API key is set (non-empty) for the provider.

    Returns (ok, message).  Does NOT make an API call.
    """
    key = resolve_provider_api_key(provider)
    if not key or key == "NOT_SET":
        env_vars = {
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY or GOOGLE_API_KEY",
        }
        return False, f"API key not set for provider '{provider}'. Set {env_vars.get(provider, '???')}"
    return True, f"API key present for '{provider}' (length={len(key)})"


def verify_api_key(provider: str, api_key: str) -> tuple[bool, str]:
    """Make a minimal API call to verify the key is valid.

    Anthropic: messages.create with max_tokens=1, ~$0.0001 per call.
    Gemini: models.list (free, no generation cost).

    Returns (ok, message).
    """
    if not api_key:
        return False, "API key is empty"

    if provider == "anthropic":
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            # Minimal call: 1 token output, cheapest model
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True, "Anthropic API key verified (minimal call, ~$0.0001)"
        except anthropic.AuthenticationError as exc:
            return False, f"Anthropic authentication failed: {exc}"
        except Exception as exc:
            return False, f"Anthropic verification error: {exc}"

    elif provider == "gemini":
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            # Free call: list models (no generation cost)
            models = list(genai.list_models())
            if models:
                return True, "Gemini API key verified (models.list, free)"
            return False, "Gemini key accepted but no models returned"
        except Exception as exc:
            return False, f"Gemini verification error: {exc}"

    return False, f"Unknown provider: {provider}"


@dataclass
class LLMConfig:
    """Configuration for LLM teacher/planner calls."""

    llm_provider: str = "anthropic"
    api_key: str = field(default="", repr=False)
    model_id: str = _DEFAULT_MODEL
    prompt_version: str = _DEFAULT_PROMPT_VERSION
    temperature: float = 0.0
    max_tokens: int = 64
    cache_enabled: bool = True
    # Rate limit in calls/minute (0 = unlimited)
    max_calls_per_minute: int = 60
    # Trace output directory (relative to project root)
    trace_dir: str = "data/llm_traces"
    # SQLite cache path (relative to project root)
    cache_db: str = "data/llm_cache.db"

    def __post_init__(self):
        validate_provider(self.llm_provider)
        if not self.api_key:
            self.api_key = resolve_provider_api_key(self.llm_provider)


def from_env(
    model_id: str = _DEFAULT_MODEL,
    prompt_version: str = _DEFAULT_PROMPT_VERSION,
    temperature: float = 0.0,
    max_tokens: int = 64,
    cache_enabled: bool = True,
    llm_provider: str = "anthropic",
) -> LLMConfig:
    """Create LLMConfig populated from environment variables."""
    validate_provider(llm_provider)
    return LLMConfig(
        llm_provider=llm_provider,
        api_key=resolve_provider_api_key(llm_provider),
        model_id=model_id,
        prompt_version=prompt_version,
        temperature=temperature,
        max_tokens=max_tokens,
        cache_enabled=cache_enabled,
    )
