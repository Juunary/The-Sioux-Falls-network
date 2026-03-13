# ============================================================
# Request domain object for the DRT environment.
# Ported from KW_DRT/app/request.py + request_status.py
# get_vector() is kept for reference; obs building is done
# independently in drt_observation.py.
# ============================================================

from __future__ import annotations
from enum import IntEnum


class RequestStatus(IntEnum):
    PENDING = 1
    ACCEPTED = 2
    PICKEDUP = 3
    SERVED = 4
    CANCELLED = 5


REQUEST_STATUS_NUM_CLASSES: int = 5   # full enum size
# Only 3 classes are used in obs (PENDING/ACCEPTED/PICKEDUP)
OBS_REQUEST_STATUS_CLASSES: int = 3


class Request:
    PICKUP_TOLERANCE_TIME: int = 10
    ARRIVAL_TOLERANCE_TIME: int = 20

    def __init__(
        self,
        request_id: int,
        from_node_id: int,
        to_node_id: int,
        request_time: int,
        network,
    ) -> None:
        self.num_passengers: int = 1

        # Immutable
        self.id: int = request_id
        self.network = network
        self.from_node_id: int = from_node_id    # 1-based
        self.to_node_id: int = to_node_id        # 1-based
        self.request_time: int = request_time
        self.travel_time: float = -10_000_000.0
        self.pickup_due: int = -1
        self.arrival_due: int = -1

        # Mutable (updated each tick)
        self.status: RequestStatus = RequestStatus.PENDING
        self.waiting_time: int = -1
        self.in_vehicle_time: int = 0
        self.arrival_due_left: int = -1
        self.assigned_v_id: int = -1
        self.slot_idx: int = -1

        # Logging
        self.detour_time: float = -1.0
        self.pickup_at: int | None = None
        self.dropoff_at: int | None = None

    def __str__(self) -> str:
        return (
            f"<R>(id={self.id} / "
            f"{self.from_node_id} -> {self.to_node_id} / "
            f"status={self.status} / "
            f"veh={self.assigned_v_id} / "
            f"rt={self.request_time} / "
            f"wt={self.waiting_time} / "
            f"ivt={self.in_vehicle_time} / "
            f"odt={self.travel_time} / "
            f"pt={self.pickup_at} / "
            f"dt={self.dropoff_at} / "
            f"p_due={self.pickup_due} / "
            f"a_due={self.arrival_due})"
        )

    def set_travel_time(self, travel_time: float) -> None:
        self.travel_time = travel_time
        self.pickup_due = self.request_time + Request.PICKUP_TOLERANCE_TIME
        self.arrival_due = (
            self.request_time + int(self.travel_time) + Request.ARRIVAL_TOLERANCE_TIME
        )

    # ------------------------------------------------------------------
    # Reference implementation — kept for design documentation.
    # drt_observation.py builds obs independently with explicit indexing.
    # ------------------------------------------------------------------
    def get_vector(self) -> list[float]:
        from backend.env.drt_config import MAX_WAIT_TIME, VEH_CAPACITY
        num_nodes = self.network.num_nodes
        max_dur = self.network.max_duration

        vec_status = [0.0] * OBS_REQUEST_STATUS_CLASSES
        if 1 <= int(self.status) <= OBS_REQUEST_STATUS_CLASSES:
            vec_status[int(self.status) - 1] = 1.0

        vec_from = [0.0] * num_nodes
        if 1 <= self.from_node_id <= num_nodes:
            vec_from[self.from_node_id - 1] = 1.0   # 1-based → 0-based

        vec_to = [0.0] * num_nodes
        if 1 <= self.to_node_id <= num_nodes:
            vec_to[self.to_node_id - 1] = 1.0       # 1-based → 0-based

        vec_passengers = [self.num_passengers / VEH_CAPACITY]
        vec_travel = [self.travel_time / max_dur] if max_dur > 0 else [0.0]
        vec_wait = [self.waiting_time / MAX_WAIT_TIME] if self.waiting_time >= 0 else [0.0]
        denom = max_dur + Request.ARRIVAL_TOLERANCE_TIME
        vec_deadline = [self.arrival_due_left / denom] if denom > 0 and self.arrival_due_left >= 0 else [0.0]

        return vec_status + vec_from + vec_to + vec_passengers + vec_travel + vec_wait + vec_deadline
