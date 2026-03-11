# ============================================================
# Demand-aware greedy baseline — demand_aware_greedy_v1
#
# Algorithm (from plan Section 8):
#   scores[node] = waiting_per_node[node] - fleet_at_chargers_penalty
#   Select argmax(scores) among reachable nodes, excluding current_node.
#   Tie-break: smallest node index (deterministic).
#
# Observation schema indices (96-dim flat Box, plan Section 4):
#   0:24   self_node one-hot
#   24     self_battery
#   25     self_passengers
#   26:30  self_state one-hot
#   30     self_speed
#   31:55  onboard_dest_dist [24]
#   55:79  waiting_per_node [24]
#   79:87  fleet_at_chargers [8]
#   87:95  charger_distances [8]
#   95     time_normalized
# ============================================================

from __future__ import annotations

import numpy as np

from backend.env.network import CHARGING_STATION_IDS

BASELINE_VERSION = "1.0"
BASELINE_NAME = "demand_aware_greedy_v1"

# Fleet concentration penalty weight (from plan)
_FLEET_PENALTY_WEIGHT = 0.3


def demand_aware_greedy(
    obs: np.ndarray,
    bus_current_node: int,
    action_mask: np.ndarray,
) -> int:
    """
    demand_aware_greedy_v1 policy.

    Args:
        obs:              96-dim observation vector (float32, all in [0,1])
        bus_current_node: current node of the decision bus (for exclusion)
        action_mask:      bool[24] — True = valid action

    Returns:
        action: int in [0, 23], the selected target node
    """
    waiting = obs[55:79].astype(np.float64)   # waiting_per_node [24]
    fleet = obs[79:87].astype(np.float64)     # fleet_at_chargers [8]

    scores = waiting.copy()

    # Apply fleet concentration penalty at charging station nodes
    for i, cs_node in enumerate(CHARGING_STATION_IDS):
        scores[cs_node] -= fleet[i] * _FLEET_PENALTY_WEIGHT

    # Mask invalid actions
    scores[bus_current_node] = -np.inf
    scores[~action_mask] = -np.inf

    # Argmax with tie-break: np.argmax returns lowest index on tie
    action = int(np.argmax(scores))
    return action
