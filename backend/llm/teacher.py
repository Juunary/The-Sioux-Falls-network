# ============================================================
# LLM Offline Teacher — factory closure implementing PolicyFnDRT
#
# make_llm_teacher_policy_fn(ctx, cfg) -> PolicyFnDRT
#   closure that:
#     1. Checks cache (obs_hash, mask_hash, env_version, model_id,
#        prompt_version, "llm_teacher", llm_provider)
#     2. On miss: calls LLM API (Anthropic or Gemini), parses action,
#        stores to cache
#     3. Writes JSONL trace entry (includes llm_provider)
#     4. Accumulates stats into ctx
#
# Requires: pip install -r backend/requirements-llm.txt
# ============================================================

from __future__ import annotations

import json
import logging
import pathlib
import re as _re
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

_POLICY_TYPE = "llm_teacher"


# ============================================================
# Provider-specific client helpers
# ============================================================

def _create_anthropic_client(api_key: str):
    """Create and return an Anthropic client."""
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def _call_anthropic(client, model_id: str, prompt_text: str,
                     temperature: float, max_tokens: int) -> tuple[str, int, int]:
    """Call Anthropic messages API.

    Returns (raw_text, input_tokens, output_tokens).
    """
    response = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt_text}],
    )
    raw = response.content[0].text if response.content else ""
    inp = response.usage.input_tokens if response.usage else 0
    out = response.usage.output_tokens if response.usage else 0
    return raw, inp, out


def _create_gemini_client(api_key: str):
    """Configure google.generativeai and return the module (used as client)."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    return genai


def _call_gemini(genai_module, model_id: str, prompt_text: str,
                  temperature: float, max_tokens: int) -> tuple[str, int, int]:
    """Call Gemini generateContent API.

    Returns (raw_text, input_tokens, output_tokens).
    """
    model = genai_module.GenerativeModel(model_id)
    generation_config = genai_module.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    response = model.generate_content(prompt_text, generation_config=generation_config)
    # response.text raises ValueError if no valid Part (e.g. safety-blocked)
    try:
        raw = response.text or ""
    except (ValueError, AttributeError):
        raw = ""
    # Token usage from usage_metadata
    inp = 0
    out = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        inp = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
        out = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
    return raw, inp, out


# ============================================================
# Factory
# ============================================================

def make_llm_teacher_policy_fn(
    ctx: "PolicyContext",
    cfg: Optional["LLMConfig"] = None,
) -> Callable[[np.ndarray, np.ndarray], int]:
    """Create a PolicyFnDRT that calls an LLM for each decision.

    Args:
        ctx: PolicyContext with env_version, model_id, prompt_version,
             llm_provider.  Counters are accumulated in-place.
        cfg: LLMConfig.  If None, constructed from environment variables
             using ctx.llm_provider.

    Returns:
        policy_fn(obs, mask) -> action  — compatible with PolicyFnDRT.

    Raises:
        ImportError: if the required SDK is not installed.
        ValueError: if the API key is empty.
    """
    from backend.llm.config import LLMConfig, from_env, validate_provider
    from backend.llm.obs_to_text import obs_to_text
    from backend.llm.text_to_action import text_to_action
    from backend.llm.cache import cache_lookup, cache_store

    provider = ctx.llm_provider
    validate_provider(provider)

    # Verify SDK importable
    if provider == "anthropic":
        try:
            import anthropic as _anthropic  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "LLM teacher (anthropic) requires the anthropic SDK.  "
                "Run: pip install -r backend/requirements-llm.txt"
            ) from exc
    elif provider == "gemini":
        try:
            import google.generativeai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "LLM teacher (gemini) requires the google-generativeai SDK.  "
                "Run: pip install google-generativeai"
            ) from exc

    if cfg is None:
        cfg = from_env(
            model_id=ctx.model_id,
            prompt_version=ctx.prompt_version,
            llm_provider=provider,
        )

    if not cfg.api_key:
        env_vars = {
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY or GOOGLE_API_KEY",
        }
        raise ValueError(
            f"API key for provider '{provider}' is not set.  "
            f"Set the environment variable {env_vars.get(provider, '???')} "
            f"or pass cfg.api_key."
        )

    # Resolve trace dir and cache db relative to project root
    _root = pathlib.Path(__file__).resolve().parents[2]
    trace_dir = _root / cfg.trace_dir
    trace_dir.mkdir(parents=True, exist_ok=True)
    cache_db = _root / cfg.cache_db

    # One trace file per factory invocation
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    trace_path = trace_dir / f"llm_teacher_{ctx.env_version}_{ts}.jsonl"

    step_counter = [0]

    # Lazy client — created on first non-cache call
    _client: list[Optional[object]] = [None]

    def _get_client():
        if _client[0] is None:
            if provider == "anthropic":
                _client[0] = _create_anthropic_client(cfg.api_key)
            elif provider == "gemini":
                _client[0] = _create_gemini_client(cfg.api_key)
        return _client[0]

    _MAX_RETRIES = 3
    _daily_exhausted: list[bool] = [False]  # circuit-breaker for daily quota

    def _call_llm(prompt_text: str) -> tuple[str, int, int]:
        """Call the configured LLM provider with rate-limit retry.

        Per-minute 429: retries with API-suggested delay (up to _MAX_RETRIES).
        Per-day 429:    raises immediately (no retry — daily quota exhausted).
                        Sets circuit-breaker so subsequent calls skip API.
        Other errors:   raises immediately.

        Returns (raw_text, input_tokens, output_tokens).
        """
        if _daily_exhausted[0]:
            raise RuntimeError(
                f"Daily quota exhausted for {provider}/{ctx.model_id} "
                f"(circuit-breaker active, skipping API call)"
            )

        for attempt in range(_MAX_RETRIES + 1):
            try:
                client = _get_client()
                if provider == "anthropic":
                    return _call_anthropic(
                        client, ctx.model_id, prompt_text,
                        cfg.temperature, cfg.max_tokens,
                    )
                elif provider == "gemini":
                    return _call_gemini(
                        client, ctx.model_id, prompt_text,
                        cfg.temperature, cfg.max_tokens,
                    )
                raise ValueError(f"Unsupported provider: {provider}")
            except Exception as exc:
                exc_str = str(exc)
                is_rate_limit = (
                    "429" in exc_str
                    or "ResourceExhausted" in type(exc).__name__
                )
                is_daily = "PerDay" in exc_str
                if is_rate_limit and not is_daily and attempt < _MAX_RETRIES:
                    m = _re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", exc_str)
                    wait = int(m.group(1)) + 2 if m else 62
                    logger.warning(
                        "Rate limited (per-minute, %s), waiting %ds "
                        "(attempt %d/%d)...",
                        provider, wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                if is_daily:
                    _daily_exhausted[0] = True
                    logger.error(
                        "Daily quota exhausted for %s/%s — "
                        "circuit-breaker activated, skipping further API calls.",
                        provider, ctx.model_id,
                    )
                raise

    def policy_fn(obs: np.ndarray, mask: np.ndarray) -> int:
        step = step_counter[0]
        step_counter[0] += 1
        t_step = time.time()

        # ---- Cache lookup ----
        if cfg.cache_enabled:
            cached = cache_lookup(
                obs, mask,
                env_version=ctx.env_version,
                model_id=ctx.model_id,
                prompt_version=ctx.prompt_version,
                policy_type=_POLICY_TYPE,
                llm_provider=provider,
                db_path=cache_db,
            )
        else:
            cached = None

        if cached is not None:
            action, reasoning = cached
            ctx.cache_hits += 1
            step_ms = (time.time() - t_step) * 1000
            ctx.step_latencies_ms.append(step_ms)
            _write_trace(
                trace_path, ctx, step, obs, mask,
                prompt_text="(cached)",
                raw_response="(cached)",
                parsed_action=action,
                parse_success=True,
                mask_compliant=True,
                reasoning=reasoning,
                latency_ms=0,
                input_tokens=0,
                output_tokens=0,
                cache_hit=True,
            )
            return action

        # ---- Build prompt ----
        prompt_text = obs_to_text(obs, mask, ctx.env_version, ctx.prompt_version)

        # ---- Call LLM API ----
        raw_response = ""
        reasoning = ""
        input_tokens = 0
        output_tokens = 0
        t0 = time.time()

        try:
            raw_response, input_tokens, output_tokens = _call_llm(prompt_text)
        except Exception as exc:
            logger.error("LLM API call failed (%s): %s", provider, exc)
            # Fallback: first valid action
            action = 0
            for i, m in enumerate(mask):
                if m:
                    action = i
                    break
            ctx.llm_calls += 1
            ctx.invalid_actions += 1
            step_ms = (time.time() - t_step) * 1000
            ctx.step_latencies_ms.append(step_ms)
            _write_trace(
                trace_path, ctx, step, obs, mask,
                prompt_text=prompt_text,
                raw_response=f"ERROR: {exc}",
                parsed_action=action,
                parse_success=False,
                mask_compliant=True,
                reasoning="",
                latency_ms=int((time.time() - t0) * 1000),
                input_tokens=0,
                output_tokens=0,
                cache_hit=False,
            )
            return action

        latency_ms = int((time.time() - t0) * 1000)
        ctx.llm_calls += 1
        ctx.input_tokens += input_tokens
        ctx.output_tokens += output_tokens

        # ---- Parse action ----
        action, parse_success, mask_compliant = text_to_action(
            raw_response, mask, ctx.env_version
        )
        if not parse_success or not mask_compliant:
            ctx.invalid_actions += 1

        # ---- Store to cache ----
        if cfg.cache_enabled:
            cache_store(
                obs, mask,
                env_version=ctx.env_version,
                model_id=ctx.model_id,
                prompt_version=ctx.prompt_version,
                policy_type=_POLICY_TYPE,
                llm_provider=provider,
                action=action,
                reasoning=reasoning,
                db_path=cache_db,
            )

        # ---- Write trace ----
        step_ms = (time.time() - t_step) * 1000
        ctx.step_latencies_ms.append(step_ms)
        _write_trace(
            trace_path, ctx, step, obs, mask,
            prompt_text=prompt_text,
            raw_response=raw_response,
            parsed_action=action,
            parse_success=parse_success,
            mask_compliant=mask_compliant,
            reasoning=reasoning,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit=False,
        )

        return action

    return policy_fn


def _write_trace(
    trace_path: pathlib.Path,
    ctx: "PolicyContext",
    step: int,
    obs: np.ndarray,
    mask: np.ndarray,
    *,
    prompt_text: str,
    raw_response: str,
    parsed_action: int,
    parse_success: bool,
    mask_compliant: bool,
    reasoning: str,
    latency_ms: int,
    input_tokens: int,
    output_tokens: int,
    cache_hit: bool,
) -> None:
    entry = {
        "trace_id": str(uuid.uuid4()),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "policy_type": _POLICY_TYPE,
        "llm_provider": ctx.llm_provider,
        "model_id": ctx.model_id,
        "prompt_version": ctx.prompt_version,
        "env_version": ctx.env_version,
        "scenario_id": ctx.scenario_id,
        "episode_id": ctx.episode_id,
        "step": step,
        "obs": obs.tolist(),
        "mask": mask.tolist(),
        "prompt_text": prompt_text,
        "raw_response": raw_response,
        "parsed_action": parsed_action,
        "parse_success": parse_success,
        "mask_compliant": mask_compliant,
        "reasoning": reasoning,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_hit": cache_hit,
    }
    with open(trace_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
