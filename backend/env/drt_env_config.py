# ============================================================
# DRT environment configuration — frozen dataclass presets
#
# Stage 0.5: DRT_V1 (L0) only.
# L1/L2/L3 presets and hybrid variants are out of scope here.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DRTEnvConfig:
    """Immutable description of a DRT environment variant.

    Passed to RotatingDRTEpisodeEnv and evaluate_drt_policy_on_manifest
    to enable comparison_group compatibility checks across manifest entries.
    Does NOT replace backend/env/drt_config.py — DRTEnv still reads its
    constants from there.  This file is additive metadata only.
    """

    env_version: str          # e.g. "drt_env_v1"
    comparison_group: str     # e.g. "L0" — manifest compatibility key
    max_num_vehicles: int
    veh_capacity: int
    max_num_request: int      # request slots → action_dim = max_num_request + 1
    max_wait_time: int
    max_invehicle_time: int
    num_nodes: int
    max_episode_time: int
    obs_dim: int


# ---- Stage 0.5 preset ----------------------------------------

DRT_V1 = DRTEnvConfig(
    env_version="drt_env_v1",
    comparison_group="L0",
    max_num_vehicles=2,
    veh_capacity=5,
    max_num_request=8,
    max_wait_time=10,
    max_invehicle_time=10,
    num_nodes=24,
    max_episode_time=200,
    obs_dim=157,
)

# Lookup by env_version string
ENV_CONFIGS: dict[str, DRTEnvConfig] = {
    c.env_version: c for c in [DRT_V1]
}
