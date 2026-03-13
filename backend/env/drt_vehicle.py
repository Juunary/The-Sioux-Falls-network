# ============================================================
# Vehicle domain object for the DRT environment.
# Ported from KW_DRT/app/vehicle.py + vehicle_status.py
# get_vector() is kept for reference; obs building is done
# independently in drt_observation.py.
# ============================================================

from __future__ import annotations
from enum import IntEnum


class VehicleStatus(IntEnum):
    IDLE = 1
    PICKUP = 2
    DROPOFF = 3
    REJECT = 4


VEHICLE_STATUS_NUM_CLASSES: int = 4


class Vehicle:
    def __init__(self, veh_id: int, curr_node: int, network) -> None:
        self.id: int = veh_id          # 0-based index
        self.network = network

        self.status: VehicleStatus = VehicleStatus.IDLE
        self.curr_node: int = curr_node   # 1-based node ID
        self.next_node: int = 0
        self.active_request_list: list = []
        self.target_request = None
        self.target_arrival_time: int = -1
        self.num_passengers: int = 0

        # Logging counters
        self.num_accept: int = 0
        self.num_serve: int = 0
        self.idle_time: int = 0
        self.on_service_driving_time: int = 0

    def __str__(self) -> str:
        return (
            f"[V](id={self.id} / "
            f"{self.curr_node} -> {self.next_node} / "
            f"target_r={self.target_request.id if self.target_request else 'None'} / "
            f"status={self.status} / "
            f"at={self.target_arrival_time} / "
            f"np={self.num_passengers} / "
            f"active_r_num={len(self.active_request_list)})"
        )

    # ------------------------------------------------------------------
    # Reference implementation — kept for design documentation.
    # drt_observation.py builds obs independently with explicit indexing.
    # ------------------------------------------------------------------
    def get_vector(self) -> list[float]:
        num_nodes = self.network.num_nodes
        from backend.env.drt_config import VEH_CAPACITY

        vec_status = [0.0] * VEHICLE_STATUS_NUM_CLASSES
        if 1 <= int(self.status) <= VEHICLE_STATUS_NUM_CLASSES:
            vec_status[int(self.status) - 1] = 1.0

        vec_from = [0.0] * num_nodes
        if 1 <= self.curr_node <= num_nodes:
            vec_from[self.curr_node - 1] = 1.0  # 1-based → 0-based

        vec_to = [0.0] * num_nodes
        if 1 <= self.next_node <= num_nodes:
            vec_to[self.next_node - 1] = 1.0   # 1-based → 0-based

        vec_capa = [(VEH_CAPACITY - self.num_passengers) / VEH_CAPACITY]

        return vec_status + vec_from + vec_to + vec_capa
