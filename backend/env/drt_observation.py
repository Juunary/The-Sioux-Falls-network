# ============================================================
# DRT observation builder — 157-dim flat Box[0, 1]
#
# Layout (all values clipped to [0, 1]):
#   [0   :24 ]  self_curr_node one-hot            (1-based → idx node_id-1)
#   [24  :48 ]  fleet_curr_node distribution      (other vehicles)
#   [48  ]      self_capacity_ratio               (remaining / VEH_CAPACITY)
#   [49  :52 ]  self_status one-hot               (IDLE / PICKUP / DROPOFF)
#   [52  ]      time_normalized                   (curr_time / MAX_EPISODE_TIME)
#
#   For each request slot i in 0..7  (base = 53 + i*13):
#   [base  :base+3]  status one-hot  (PENDING/ACCEPTED/PICKEDUP; all-zero if padding)
#   [base+3]         is_valid        (1 if real request, 0 if padding)
#   [base+4]         from_node / NUM_NODES  (1-based → normalized)
#   [base+5]         to_node / NUM_NODES
#   [base+6]         waiting_time / MAX_WAIT_TIME
#   [base+7]         arrival_due_left / (max_duration + ARRIVAL_TOL)
#   [base+8]         need_dropoff_by_me   (1 if vehicle already picked up this request)
#   [base+9]         dur_to_target / max_duration
#   [base+10]        travel_time / max_duration
#   [base+11]        num_passengers / VEH_CAPACITY
#   [base+12]        (reserved = 0)
#
# Total: 53 + 8*13 = 157  →  OBS_DIM_DRT = 157
# ============================================================

from __future__ import annotations

import numpy as np

from backend.env.drt_config import (
    MAX_EPISODE_TIME,
    MAX_NUM_REQUEST,
    MAX_WAIT_TIME,
    NUM_NODES,
    OBS_DIM_DRT,
    VEH_CAPACITY,
)
from backend.env.drt_request import Request, RequestStatus
from backend.env.drt_vehicle import Vehicle, VehicleStatus

# Number of status dims encoded in obs (only PENDING/ACCEPTED/PICKEDUP)
_OBS_STATUS_CLASSES = 3
_SLOT_DIM = 13
_ARRIVAL_TOL = Request.ARRIVAL_TOLERANCE_TIME


def build_drt_observation(
    pending_vehicle: Vehicle,
    all_vehicles: list[Vehicle],
    active_request_list: list[Request],
    curr_time: int,
    network,
) -> np.ndarray:
    """
    Build a 157-dim flat observation vector centred on pending_vehicle.
    All values are in [0, 1].
    """
    obs = np.zeros(OBS_DIM_DRT, dtype=np.float32)
    max_dur = network.max_duration if network.max_duration > 0 else 1.0

    # ---- self_curr_node one-hot [0:24] ----
    node = pending_vehicle.curr_node
    if 1 <= node <= NUM_NODES:
        obs[node - 1] = 1.0   # 1-based → 0-based index

    # ---- fleet_curr_node distribution [24:48] ----
    other_vehicles = [v for v in all_vehicles if v.id != pending_vehicle.id]
    if other_vehicles:
        for v in other_vehicles:
            n = v.curr_node
            if 1 <= n <= NUM_NODES:
                obs[24 + (n - 1)] += 1.0
        obs[24:48] /= len(other_vehicles)

    # ---- self_capacity_ratio [48] ----
    remaining = VEH_CAPACITY - pending_vehicle.num_passengers
    obs[48] = float(remaining) / VEH_CAPACITY

    # ---- self_status one-hot [49:52] ----
    status_idx = int(pending_vehicle.status) - 1   # IDLE=0, PICKUP=1, DROPOFF=2
    if 0 <= status_idx < _OBS_STATUS_CLASSES:
        obs[49 + status_idx] = 1.0

    # ---- time_normalized [52] ----
    obs[52] = min(float(curr_time) / MAX_EPISODE_TIME, 1.0)

    # ---- request slots [53:157] ----
    for i in range(MAX_NUM_REQUEST):
        base = 53 + i * _SLOT_DIM

        if i >= len(active_request_list):
            # padding — all zeros
            continue

        r = active_request_list[i]

        # status one-hot [base:base+3]
        s = int(r.status)
        if 1 <= s <= _OBS_STATUS_CLASSES:
            obs[base + s - 1] = 1.0

        # is_valid [base+3]
        obs[base + 3] = 1.0

        # from_node / NUM_NODES [base+4]
        obs[base + 4] = float(r.from_node_id) / NUM_NODES

        # to_node / NUM_NODES [base+5]
        obs[base + 5] = float(r.to_node_id) / NUM_NODES

        # waiting_time / MAX_WAIT_TIME [base+6]
        if r.waiting_time >= 0:
            obs[base + 6] = min(float(r.waiting_time) / MAX_WAIT_TIME, 1.0)

        # arrival_due_left [base+7]
        denom = max_dur + _ARRIVAL_TOL
        if r.arrival_due_left >= 0:
            obs[base + 7] = min(float(r.arrival_due_left) / denom, 1.0)

        # need_dropoff_by_me [base+8]
        if r in pending_vehicle.active_request_list:
            obs[base + 8] = 1.0

        # dur_to_target / max_duration [base+9]
        if r.status == RequestStatus.PENDING or r.status == RequestStatus.ACCEPTED:
            target_node = r.from_node_id
        elif r.status == RequestStatus.PICKEDUP and r in pending_vehicle.active_request_list:
            target_node = r.to_node_id
        else:
            target_node = None

        if target_node is not None:
            try:
                dur = network.get_duration(pending_vehicle.curr_node, target_node)
                obs[base + 9] = min(dur / max_dur, 1.0)
            except (KeyError, TypeError):
                pass

        # travel_time / max_duration [base+10]
        if r.travel_time > 0:
            obs[base + 10] = min(r.travel_time / max_dur, 1.0)

        # num_passengers / VEH_CAPACITY [base+11]
        obs[base + 11] = float(r.num_passengers) / VEH_CAPACITY

        # [base+12] reserved = 0 (already zero)

    return obs
