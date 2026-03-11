# ============================================================
# Observation builder — 96-dimensional flat Box
#
# Schema (96 values, all in [0, 1]):
#
# [ Self state — 31 values ]
#   self_node:        float[24]  one-hot (current node)
#   self_battery:     float      battery / 100
#   self_passengers:  float      len(passenger_ids) / capacity
#   self_state:       float[4]   one-hot (moving/charging/idle/rerouting)
#   self_speed:       float      (speed - 80) / 40
#
# [ On-board passenger destination distribution — 24 values ]
#   onboard_dest_dist: float[24] count at each dest node / capacity
#
# [ Network waiting passenger distribution — 24 values ]
#   waiting_per_node:  float[24] waiting count / 5, clipped to 1.0
#
# [ Fleet at chargers — 8 values ]
#   fleet_at_chargers: float[8]  buses at each CS node / bus_count
#
# [ Distance to chargers — 8 values ]
#   charger_distances: float[8]  dijkstra dist / MAX_DIST
#
# [ Time — 1 value ]
#   time_normalized:   float     sim_time / episode_length
#
# Total: 24+1+1+4+1 + 24 + 24 + 8 + 8 + 1 = 96
# ============================================================

from __future__ import annotations

import math

import numpy as np

from backend.env.network import CHARGING_STATION_IDS, NODES
from backend.env.pathfinding import MAX_DIST, dijkstra

_STATE_ORDER = ("moving", "charging", "idle", "rerouting")
_STATE_INDEX = {s: i for i, s in enumerate(_STATE_ORDER)}

NUM_NODES = 24
NUM_CHARGERS = 8  # len(CHARGING_STATION_IDS)
OBS_DIM = 96


def build_observation(
    bus,           # BusState
    all_buses: list,   # list[BusState]
    passengers: list,  # list[Passenger]
    sim_time: float,
    episode_length: float,
) -> np.ndarray:
    """
    Build the 96-dim flat observation vector for the given bus.
    All values are in [0, 1].
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    offset = 0

    # ---- self_node: one-hot [24] ----
    obs[offset + bus.current_node] = 1.0
    offset += NUM_NODES  # 24

    # ---- self_battery: float ----
    obs[offset] = bus.battery / 100.0
    offset += 1  # 25

    # ---- self_passengers: float ----
    obs[offset] = len(bus.passenger_ids) / max(1, bus.capacity)
    offset += 1  # 26

    # ---- self_state: one-hot [4] (moving/charging/idle/rerouting) ----
    state_idx = _STATE_INDEX.get(bus.state, 2)  # default idle
    obs[offset + state_idx] = 1.0
    offset += 4  # 30

    # ---- self_speed: float ----
    obs[offset] = (bus.speed - 80.0) / 40.0
    offset += 1  # 31

    # ---- onboard_dest_dist: float[24] ----
    # Count passengers on this bus grouped by destination
    on_board = {p.id for p in passengers if p.bus_id == bus.id and p.state == "riding"}
    for p in passengers:
        if p.id in on_board:
            obs[offset + p.destination_node] += 1.0
    if bus.capacity > 0:
        obs[offset:offset + NUM_NODES] /= bus.capacity
    offset += NUM_NODES  # 55

    # ---- waiting_per_node: float[24] ----
    waiting_counts = np.zeros(NUM_NODES, dtype=np.float32)
    for p in passengers:
        if p.state == "waiting":
            waiting_counts[p.current_node] += 1.0
    obs[offset:offset + NUM_NODES] = np.clip(waiting_counts / 5.0, 0.0, 1.0)
    offset += NUM_NODES  # 79

    # ---- fleet_at_chargers: float[8] ----
    bus_count = len(all_buses)
    for i, cs_id in enumerate(CHARGING_STATION_IDS):
        count = sum(1 for b in all_buses if b.current_node == cs_id and b.current_edge is None)
        obs[offset + i] = count / max(1, bus_count)
    offset += NUM_CHARGERS  # 87

    # ---- charger_distances: float[8] ----
    dist_map, _ = dijkstra(bus.current_node)
    for i, cs_id in enumerate(CHARGING_STATION_IDS):
        d = dist_map.get(cs_id, math.inf)
        obs[offset + i] = 0.0 if math.isinf(d) else min(1.0, d / MAX_DIST)
    offset += NUM_CHARGERS  # 95

    # ---- time_normalized: float ----
    obs[offset] = min(1.0, sim_time / max(1.0, episode_length))
    offset += 1  # 96

    assert offset == OBS_DIM, f"obs dimension mismatch: {offset} != {OBS_DIM}"
    return obs
