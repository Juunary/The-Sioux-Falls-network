# ============================================================
# DRT greedy baseline policy
#
# Policy: obs-only, no environment coupling beyond (obs, mask).
# Priority:
#   1. Dropoff (need_dropoff_by_me == 1) — nearest first
#   2. Pickup  (PENDING/ACCEPTED)        — nearest first
#   3. REJECT  (action 8)               — fallback
#
# Observation slot layout (base = 53 + i * 13):
#   base+3  is_valid
#   base+8  need_dropoff_by_me
#   base+9  dur_to_target / max_duration
# ============================================================

from __future__ import annotations

import numpy as np

BASELINE_NAME = "drt_greedy_v1"

_SLOT_BASE = 53
_SLOT_DIM = 13
_IS_VALID = 3
_NEED_DROPOFF = 8
_DUR_TO_TARGET = 9
_REJECT_ACTION = 8
_NUM_SLOTS = 8


def drt_greedy_policy(obs: np.ndarray, mask: np.ndarray) -> int:
    """
    Greedy DRT policy derived entirely from the flat 157-dim observation.

    Args:
        obs:  float32 array of shape (157,), values in [0, 1]
        mask: bool array of shape (9,); mask[8] (REJECT) is always True

    Returns:
        action index in [0, 8]
    """
    dropoff_candidates: list[tuple[float, int]] = []  # (dur, slot_idx)
    pickup_candidates: list[tuple[float, int]] = []

    for i in range(_NUM_SLOTS):
        if not mask[i]:
            continue
        base = _SLOT_BASE + i * _SLOT_DIM
        if not obs[base + _IS_VALID]:
            continue

        dur = float(obs[base + _DUR_TO_TARGET])

        if obs[base + _NEED_DROPOFF] >= 0.5:
            dropoff_candidates.append((dur, i))
        else:
            pickup_candidates.append((dur, i))

    if dropoff_candidates:
        return min(dropoff_candidates)[1]  # lowest dur first, then lowest index

    if pickup_candidates:
        return min(pickup_candidates)[1]

    return _REJECT_ACTION
