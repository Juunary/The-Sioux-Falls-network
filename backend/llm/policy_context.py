# ============================================================
# PolicyContext — shared state threaded through factory closure
# Mutable per-episode counters; callers must call reset_episode_stats()
# before each new episode.
#
# compute_llm_metrics() produces v13-canonical LLM metrics dict
# with all 11 DB columns + 5 legacy raw counters.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Simple pricing per 1M tokens (USD) — matched by model_id prefix
_MODEL_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # prefix: (input_price_per_1M, output_price_per_1M)
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-4": (0.80, 4.0),
    "claude-opus-4": (15.0, 75.0),
    # Gemini models (as of 2025-Q1)
    "gemini-2.5-flash-lite": (0.05, 0.30),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.0),
}


@dataclass
class PolicyContext:
    """Context captured by make_llm_teacher_policy_fn factory closure.

    Immutable fields (env_version, model_id, prompt_version, llm_provider)
    are set once at construction.  Mutable fields (scenario_id, episode_id
    and all counters) are updated by the evaluator before / during each
    episode.
    """

    env_version: str       # e.g. "drt_env_v1"
    model_id: str          # e.g. "claude-sonnet-4-6"
    prompt_version: str    # e.g. "v1"
    llm_provider: str = "anthropic"  # "anthropic" | "gemini"

    # Set per-episode by the evaluator
    scenario_id: str = ""
    episode_id: str = ""

    # Per-episode accumulators — reset via reset_episode_stats()
    llm_calls: int = field(default=0, compare=False)
    cache_hits: int = field(default=0, compare=False)
    input_tokens: int = field(default=0, compare=False)
    output_tokens: int = field(default=0, compare=False)
    invalid_actions: int = field(default=0, compare=False)

    # Per-step latency tracking (ms) — populated by teacher closure
    step_latencies_ms: list = field(default_factory=list, compare=False, repr=False)

    def reset_episode_stats(self) -> None:
        """Reset per-episode counters.  Call before each new episode."""
        self.llm_calls = 0
        self.cache_hits = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.invalid_actions = 0
        self.step_latencies_ms = []

    def to_stats_dict(self) -> dict:
        """Return raw per-episode LLM stats as a plain dict (legacy 5 keys)."""
        return {
            "llm_calls": self.llm_calls,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "invalid_actions": self.invalid_actions,
        }

    def compute_llm_metrics(self) -> dict:
        """Compute v13-canonical LLM metrics from raw counters.

        Returns dict with:
          - 5 legacy raw columns (llm_calls, cache_hits, input_tokens,
            output_tokens, invalid_actions)
          - 8 v13 canonical columns (invalid_action_rate,
            api_cost_per_episode, api_cost_per_decision,
            episode_latency_p50, episode_latency_p95,
            teacher_agreement_rate, planner_override_rate,
            tool_call_success_rate)
        """
        d = self.to_stats_dict()

        # Total decisions = all policy_fn invocations
        total_decisions = self.llm_calls + self.cache_hits

        # invalid_action_rate = invalid / total_decisions
        d["invalid_action_rate"] = (
            round(self.invalid_actions / total_decisions, 4)
            if total_decisions > 0 else None
        )

        # Latency percentiles from per-step wall-clock measurements
        if self.step_latencies_ms:
            d["episode_latency_p50"] = round(
                float(np.percentile(self.step_latencies_ms, 50)), 2
            )
            d["episode_latency_p95"] = round(
                float(np.percentile(self.step_latencies_ms, 95)), 2
            )
        else:
            d["episode_latency_p50"] = None
            d["episode_latency_p95"] = None

        # Cost estimation from token counts × model pricing
        cost = self._estimate_cost()
        d["api_cost_per_episode"] = round(cost, 6) if cost is not None else None
        d["api_cost_per_decision"] = (
            round(cost / total_decisions, 8)
            if cost is not None and total_decisions > 0
            else None
        )

        # NULL for Stage 1 — populated in Stage 2/3
        d["teacher_agreement_rate"] = None
        d["planner_override_rate"] = None
        d["tool_call_success_rate"] = None

        return d

    def _estimate_cost(self) -> Optional[float]:
        """Estimate USD cost from token counts and model_id prefix."""
        for prefix, (inp_per_m, out_per_m) in _MODEL_PRICING_USD_PER_MTOK.items():
            if self.model_id.startswith(prefix):
                return (
                    self.input_tokens * inp_per_m / 1_000_000
                    + self.output_tokens * out_per_m / 1_000_000
                )
        return None
