# ============================================================
# obs_to_text — decode DRT obs vector into an LLM-readable prompt
#
# obs_to_text(obs, mask, env_version) -> str
#
# Supports:
#   drt_env_v1          (157-dim, 8 slots)
#   drt_env_v1_hybrid   (165-dim, 8 slots + 8 plan bits — base decoded only)
#   drt_env_v1_scaled   (261-dim, 16 slots) — structure identical, n_slots=16
#   drt_env_v1_scaled_hybrid (277-dim, 16 slots + 16 plan bits)
#   drt_env_v2          (469-dim, 32 slots)
#   drt_env_v2_hybrid   (501-dim, 32 slots + 32 plan bits)
#
# In production/eval paths env_version MUST be passed explicitly.
# Auto-detect (env_version=None) is a debug fallback only.
# ============================================================

from __future__ import annotations

import numpy as np

# ---- env_version → (n_slots, base_dim, action_dim) ----
_ENV_SPECS: dict[str, tuple[int, int, int]] = {
    "drt_env_v1":                  (8,  53,  9),
    "drt_env_v1_hybrid":           (8,  53,  9),   # obs[157:165] plan bits ignored
    "drt_env_v1_scaled":           (16, 53, 17),
    "drt_env_v1_scaled_hybrid":    (16, 53, 17),
    "drt_env_v2":                  (32, 53, 33),
    "drt_env_v2_hybrid":           (32, 53, 33),
}

# Fallback: obs_dim → env_version (debug use only)
_DIM_TO_ENV: dict[int, str] = {
    157: "drt_env_v1",
    165: "drt_env_v1_hybrid",
    261: "drt_env_v1_scaled",
    277: "drt_env_v1_scaled_hybrid",
    469: "drt_env_v2",
    501: "drt_env_v2_hybrid",
}

_STATUS_NAMES = ("PENDING", "ACCEPTED", "PICKEDUP")
_VEHICLE_STATUS = ("IDLE", "PICKUP", "DROPOFF")

# DRT V1 normalisation constants (from drt_config.py)
_NUM_NODES = 24
_VEH_CAPACITY = 5
_MAX_WAIT = 10
_MAX_EPISODE_TIME = 200


def _resolve_env_version(obs: np.ndarray, env_version: str | None) -> str:
    if env_version is not None:
        if env_version not in _ENV_SPECS:
            raise ValueError(f"obs_to_text: unknown env_version '{env_version}'")
        return env_version
    # Debug fallback: infer from obs dim
    dim = len(obs)
    if dim not in _DIM_TO_ENV:
        raise ValueError(
            f"obs_to_text: ambiguous or unknown obs dim {dim}. "
            "Pass env_version explicitly."
        )
    return _DIM_TO_ENV[dim]


def _decode_base(obs: np.ndarray, n_slots: int) -> dict:
    """Decode base obs fields (first 53 dims) and request slots.

    Returns a plain dict with decoded values.
    """
    # Self node: argmax over [0:24], 1-based
    self_node_idx = int(np.argmax(obs[0:24]))
    self_node = self_node_idx + 1 if obs[self_node_idx] > 0.5 else 0

    # Fleet distribution [24:48]: identify max-weight node
    fleet_dist = obs[24:48]
    fleet_text_parts: list[str] = []
    for i, v in enumerate(fleet_dist):
        if v > 0.01:
            fleet_text_parts.append(f"node {i + 1} ({v * 100:.0f}%)")

    # Capacity [48]
    cap_ratio = float(obs[48])
    remaining_cap = round(cap_ratio * _VEH_CAPACITY)

    # Status [49:52]
    status_idx = int(np.argmax(obs[49:52]))
    status_str = _VEHICLE_STATUS[status_idx] if obs[49 + status_idx] > 0.5 else "IDLE"

    # Time [52]
    time_norm = float(obs[52])
    curr_time = round(time_norm * _MAX_EPISODE_TIME)

    # Request slots
    slots: list[dict] = []
    for i in range(n_slots):
        base = 53 + i * 13
        slot: dict = {"idx": i, "valid": False}

        is_valid = obs[base + 3] > 0.5
        if not is_valid:
            slot["valid"] = False
            slots.append(slot)
            continue

        slot["valid"] = True

        # status
        st_vec = obs[base:base + 3]
        st_idx = int(np.argmax(st_vec))
        slot["status"] = _STATUS_NAMES[st_idx] if st_vec[st_idx] > 0.5 else "PENDING"

        # nodes (1-based, rounded)
        slot["from_node"] = round(float(obs[base + 4]) * _NUM_NODES)
        slot["to_node"]   = round(float(obs[base + 5]) * _NUM_NODES)
        slot["from_node"] = max(1, slot["from_node"])
        slot["to_node"]   = max(1, slot["to_node"])

        # timing
        slot["wait_ratio"]          = round(float(obs[base + 6]), 2)
        slot["arrival_due_ratio"]   = round(float(obs[base + 7]), 2)
        slot["need_dropoff_by_me"]  = obs[base + 8] > 0.5
        slot["dur_to_target_ratio"] = round(float(obs[base + 9]), 2)
        slot["travel_time_ratio"]   = round(float(obs[base + 10]), 2)
        slot["num_passengers"]      = round(float(obs[base + 11]) * _VEH_CAPACITY)

        slots.append(slot)

    return {
        "self_node": self_node,
        "fleet_text_parts": fleet_text_parts,
        "remaining_cap": remaining_cap,
        "status_str": status_str,
        "curr_time": curr_time,
        "slots": slots,
    }


def _format_prompt_v1(
    decoded: dict,
    mask: np.ndarray,
    n_slots: int,
    action_dim: int,
) -> str:
    """Format decoded obs into v1 prompt template."""
    lines: list[str] = []

    # Header
    fleet = ", ".join(decoded["fleet_text_parts"]) or "none"
    lines.append(
        f"Vehicle at node {decoded['self_node']}, "
        f"capacity {decoded['remaining_cap']}/{_VEH_CAPACITY}, "
        f"status {decoded['status_str']}, "
        f"time {decoded['curr_time']}/{_MAX_EPISODE_TIME}."
    )
    lines.append(f"Other fleet: {fleet}.")

    # Request slots
    lines.append("Requests:")
    valid_slot_count = sum(1 for s in decoded["slots"] if s["valid"])

    for s in decoded["slots"]:
        i = s["idx"]
        if not s["valid"]:
            continue
        st = s["status"]
        from_n = s["from_node"]
        to_n = s["to_node"]
        wait_frac = s["wait_ratio"]
        due_frac = s["arrival_due_ratio"]
        dist_frac = s["dur_to_target_ratio"]
        by_me = s["need_dropoff_by_me"]

        # Action label
        if st == "PICKEDUP" and by_me:
            action_label = "DROPOFF"
        elif st in ("PENDING", "ACCEPTED"):
            action_label = "PICKUP"
        else:
            action_label = "SERVE"

        action_valid = bool(mask[i]) if i < len(mask) else False
        valid_tag = f"[valid: {action_label}]" if action_valid else "[blocked]"

        lines.append(
            f"  [Slot {i}] {st}: node {from_n}->{to_n}, "
            f"wait {wait_frac:.2f}x/{_MAX_WAIT}t, "
            f"dist {dist_frac:.2f} (norm), "
            f"due {due_frac:.2f}. {valid_tag}"
        )

    if valid_slot_count == 0:
        lines.append("  (no active requests)")

    # Valid action summary
    reject_idx = action_dim - 1
    valid_actions: list[str] = []
    for i in range(action_dim):
        if i < len(mask) and mask[i]:
            if i == reject_idx:
                valid_actions.append(f"{i}: REJECT")
            else:
                valid_actions.append(f"{i}: serve slot {i}")
    lines.append(f"Valid actions: {{{', '.join(valid_actions)}}}")
    lines.append("Reply with only the action number (e.g., \"0\").")

    return "\n".join(lines)


def obs_to_text(
    obs: np.ndarray,
    mask: np.ndarray,
    env_version: str | None = None,
    prompt_version: str = "v1",
) -> str:
    """Decode a DRT observation vector into a human-readable LLM prompt.

    Args:
        obs:           Flat float32 obs array (e.g. 157-dim for drt_env_v1).
        mask:          Bool action mask (length = action_dim).
        env_version:   Explicit env_version string.  Pass None only for debug.
        prompt_version: Prompt template version ('v1').

    Returns:
        Formatted prompt string ready to send to an LLM.
    """
    env_version = _resolve_env_version(obs, env_version)
    n_slots, _, action_dim = _ENV_SPECS[env_version]

    # Hybrid variants: decode only base portion of obs
    base_obs = obs[:53 + n_slots * 13]
    decoded = _decode_base(base_obs, n_slots)

    if prompt_version == "v1":
        return _format_prompt_v1(decoded, mask, n_slots, action_dim)

    raise ValueError(f"obs_to_text: unknown prompt_version '{prompt_version}'")
