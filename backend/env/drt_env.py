# ============================================================
# DRTEnv — Gymnasium environment for the KW_DRT ride-sharing task.
#
# Semantics: semi-MDP (mirrors SiouxFallsEnv pattern).
# One env.step(action) = one idle vehicle's assign/reject decision
# + advance simulation time until the next idle vehicle or done.
#
# Action space:  Discrete(9)
#   0..7 = request slot index (PICKUP if not yet assigned,
#                              DROPOFF if this vehicle already has it)
#   8    = REJECT (always valid)
#
# Observation:   Box(157,) flat float32 in [0, 1]
#   See drt_observation.py for layout.
#
# Use with sb3-contrib MaskablePPO:
#   model = MaskablePPO("MlpPolicy", env, ...)
# ============================================================

from __future__ import annotations

import copy
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from backend.env.drt_config import (
    MAX_EPISODE_TIME,
    MAX_INVEHICLE_TIME,
    MAX_NUM_REQUEST,
    MAX_WAIT_TIME,
    OBS_DIM_DRT,
    POSSIBLE_ACTION,
    VEH_CAPACITY,
)
from backend.env.drt_network import DRTNetwork
from backend.env.drt_observation import build_drt_observation
from backend.env.drt_request import Request, RequestStatus
from backend.env.drt_vehicle import Vehicle, VehicleStatus
from backend.datasets.drt_loader import load_drt_data


# ---- Reward constants ----
_REWARD_PICKUP_COMPLETE = 0.5    # × (1 - wait_time / MAX_WAIT_TIME)
_REWARD_DROPOFF_COMPLETE = 1.0   # × (1 - ivt / MAX_INVEHICLE_TIME)
_PENALTY_CANCEL = -1.0
_PENALTY_REJECT = -0.1
_PENALTY_TIME = -0.005           # per elapsed tick
_MAX_STEPS = 2000                # truncation guard


class DRTEnv(gym.Env):
    """
    Gymnasium environment for the KW_DRT dynamic ride-sharing task.

    observation_space: Box(157,) in [0, 1]
    action_space:      Discrete(9)

    Use with MaskablePPO from sb3-contrib.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        requests_path: str,
        vehicle_positions_path: str,
        od_matrix_path: str,
    ) -> None:
        super().__init__()

        self._requests_path = requests_path
        self._vehicle_positions_path = vehicle_positions_path
        self._od_matrix_path = od_matrix_path

        # Load network once; request list and positions are deep-copied on reset
        self._network, self._base_request_list, self._base_vehicle_positions = (
            load_drt_data(requests_path, vehicle_positions_path, od_matrix_path)
        )

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM_DRT,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(POSSIBLE_ACTION)

        # Episode state (initialised in reset())
        self.curr_time: int = 0
        self.curr_step: int = 0
        self.vehicle_list: list[Vehicle] = []
        self.future_request_list: list[Request] = []
        self.active_request_list: list[Request] = []
        self.done_request_list: list[Request] = []
        self.pending_decision_vehicle: Optional[Vehicle] = None
        self._deferred_idle_vehicles: list[Vehicle] = []

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.curr_time = 0
        self.curr_step = 0

        self.future_request_list = copy.deepcopy(self._base_request_list)
        self.active_request_list = []
        self.done_request_list = []
        self._deferred_idle_vehicles = []

        self.vehicle_list = []
        for idx, pos in enumerate(self._base_vehicle_positions):
            v = Vehicle(idx, int(pos), self._network)
            self.vehicle_list.append(v)

        # Load requests whose time == 0
        self._load_arriving_requests()

        # Advance to first decision frontier
        self._advance_to_next_decision()

        obs = self._build_obs()
        return obs, {"curr_time": self.curr_time}

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.pending_decision_vehicle is not None, "Call reset() before step()"
        assert 0 <= action < POSSIBLE_ACTION, f"Invalid action {action}"

        reward = 0.0

        vehicle = self.pending_decision_vehicle

        # ---- Apply action ----
        if action == POSSIBLE_ACTION - 1:
            # REJECT — VehicleStatus.REJECT is preserved for semantic parity with KW_DRT
            # (original: REJECT → IDLE at start of next tick via handle_time_update).
            # Long-term simplification candidate: set IDLE immediately and remove this enum value.
            vehicle.status = VehicleStatus.REJECT
            vehicle.idle_time += 1
            reward += _PENALTY_REJECT
        else:
            # Assign request slot
            if action < len(self.active_request_list):
                r = self.active_request_list[action]
                reward += self._apply_assignment(vehicle, r)
            else:
                # Slot is padding — treat as reject (should be masked)
                vehicle.status = VehicleStatus.REJECT
                vehicle.idle_time += 1
                reward += _PENALTY_REJECT

        self.curr_step += 1

        # ---- Advance to next decision frontier ----
        elapsed, events = self._advance_to_next_decision()
        reward += self._compute_reward(events, elapsed)

        done = self._is_done()
        truncated = self.curr_step >= _MAX_STEPS

        obs = self._build_obs()
        info = {
            "curr_time": self.curr_time,
            "elapsed_ticks": elapsed,
            "events": events,
            "done_requests": len(self.done_request_list),
        }
        return obs, reward, done, truncated, info

    # ------------------------------------------------------------------
    # action_masks (MaskablePPO protocol)
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """
        Returns bool[9]: True = valid action for pending_decision_vehicle.
        Rules:
          - slot i: PENDING + spare seats  →  True  (PICKUP)
          - slot i: PICKEDUP + assigned to me  →  True  (DROPOFF)
          - slot 8 (REJECT): always True
        """
        mask = np.zeros(POSSIBLE_ACTION, dtype=bool)
        mask[POSSIBLE_ACTION - 1] = True   # REJECT always valid

        if self.pending_decision_vehicle is None:
            return mask

        v = self.pending_decision_vehicle
        for i, r in enumerate(self.active_request_list):
            if i >= MAX_NUM_REQUEST:
                break
            if r.status == RequestStatus.PICKEDUP:
                if r.assigned_v_id == v.id:
                    mask[i] = True
            elif r.status == RequestStatus.PENDING:
                if VEH_CAPACITY - v.num_passengers >= r.num_passengers:
                    mask[i] = True
        return mask

    # ------------------------------------------------------------------
    # Internal: apply assignment decision
    # ------------------------------------------------------------------

    def _apply_assignment(self, v: Vehicle, r: Request) -> float:
        """Apply PICKUP or DROPOFF decision. Returns immediate reward."""
        reward = 0.0

        if r not in v.active_request_list:
            # PICKUP decision
            v.status = VehicleStatus.PICKUP
            v.active_request_list.append(r)
            v.next_node = r.from_node_id
            v.target_request = r
            pickup_duration = int(self._network.get_duration(v.curr_node, v.next_node))
            v.target_arrival_time = self.curr_time + pickup_duration

            r.status = RequestStatus.ACCEPTED
            r.assigned_v_id = v.id

            v.num_accept += 1

            # Immediate pickup reward (scaled by proximity and urgency)
            reward += (
                0.5 * (1.0 - pickup_duration / self._network.max_duration)
                + 0.5 * (1.0 - max(r.waiting_time, 0) / MAX_WAIT_TIME)
            )

            # Immediate completion when already at pickup node
            if v.curr_node == v.next_node:
                self._complete_pickup(v, r)
                v.status = VehicleStatus.IDLE   # reset state set above
                v.next_node = 0
                v.target_request = None
                v.target_arrival_time = -1
                reward += 1.0
        else:
            # DROPOFF decision
            v.status = VehicleStatus.DROPOFF
            v.next_node = r.to_node_id
            v.target_request = r
            dropoff_duration = int(self._network.get_duration(v.curr_node, v.next_node))
            v.target_arrival_time = self.curr_time + dropoff_duration

            in_vehicle_ratio = min(r.in_vehicle_time / MAX_INVEHICLE_TIME, 1.0)
            reward += (
                0.5 * (1.0 - dropoff_duration / self._network.max_duration)
                + 0.5 * (1.0 - in_vehicle_ratio)
            )

            # Immediate completion when already at dropoff node
            if v.curr_node == v.next_node:
                self._complete_dropoff(v, r)
                v.status = VehicleStatus.IDLE   # reset state set above
                v.next_node = 0
                v.target_request = None
                v.target_arrival_time = -1
                reward += 1.0 + 0.5   # instant completion + delivery bonus

        return reward

    # ------------------------------------------------------------------
    # Internal: advance simulation to next decision frontier
    # ------------------------------------------------------------------

    def _advance_to_next_decision(self) -> tuple[int, list[dict]]:
        """
        Advance time until the next idle vehicle needs a decision.
        Returns (elapsed_ticks, list_of_reward_events).
        """
        events: list[dict] = []
        elapsed = 0

        # Serve deferred frontier first (no time advance needed)
        if self._deferred_idle_vehicles:
            self.pending_decision_vehicle = self._deferred_idle_vehicles.pop(0)
            return 0, []

        while not self._is_done() and self.curr_step < _MAX_STEPS:
            self.curr_time += 1
            elapsed += 1
            tick_events = self._handle_time_update()
            events.extend(tick_events)

            # Collect idle vehicles (ascending id order)
            idle = sorted(
                [
                    v for v in self.vehicle_list
                    if v.status == VehicleStatus.IDLE
                ],
                key=lambda v: v.id,
            )

            if idle:
                self.pending_decision_vehicle = idle[0]
                self._deferred_idle_vehicles = idle[1:]
                return elapsed, events

            if self._is_done():
                break

        self.pending_decision_vehicle = None
        return elapsed, events

    # ------------------------------------------------------------------
    # Internal: one tick — mirrors KW_DRT env.handle_time_update()
    # ------------------------------------------------------------------

    def _handle_time_update(self) -> list[dict]:
        """
        Process one time tick. Returns reward-relevant events.
        Mirrors KW_DRT/app/env.py handle_time_update() logic.
        """
        events: list[dict] = []

        # Load new requests arriving at curr_time
        self._load_arriving_requests()

        # ---- Vehicle state transitions ----
        for v in self.vehicle_list:
            # REJECT → IDLE (transition happens at start of next tick)
            if v.status == VehicleStatus.REJECT:
                v.status = VehicleStatus.IDLE

            # PICKUP arrival
            if v.status == VehicleStatus.PICKUP:
                if v.target_arrival_time == self.curr_time:
                    r = v.target_request
                    v.status = VehicleStatus.IDLE
                    v.curr_node = v.next_node
                    v.next_node = 0
                    v.target_request = None
                    v.target_arrival_time = -1

                    if r.status == RequestStatus.CANCELLED:
                        # Pickup failed — request cancelled while en route
                        if r in v.active_request_list:
                            v.active_request_list.remove(r)
                        events.append({"type": "pickup_fail", "request_id": r.id})
                    else:
                        self._complete_pickup(v, r)
                        events.append({"type": "pickup_complete", "request_id": r.id,
                                       "wait_time": r.waiting_time})

            # DROPOFF arrival
            elif v.status == VehicleStatus.DROPOFF:
                if v.target_arrival_time == self.curr_time:
                    r = v.target_request
                    v.status = VehicleStatus.IDLE
                    v.curr_node = v.next_node
                    v.next_node = 0
                    v.target_request = None
                    v.target_arrival_time = -1
                    if r in v.active_request_list:
                        v.active_request_list.remove(r)
                    v.num_passengers -= r.num_passengers

                    r.status = RequestStatus.SERVED
                    r.arrival_due_left = max(r.arrival_due - self.curr_time, 0)
                    r.in_vehicle_time = self.curr_time - r.pickup_at
                    r.dropoff_at = self.curr_time
                    if r in self.active_request_list:
                        self.active_request_list.remove(r)
                    self.done_request_list.append(r)

                    v.num_serve += 1
                    events.append({"type": "dropoff_complete", "request_id": r.id,
                                   "in_vehicle_time": r.in_vehicle_time})

        # ---- Request time updates + cancellation ----
        cancelled: list[Request] = []
        for r in self.active_request_list:
            r.arrival_due_left = max(r.arrival_due - self.curr_time, 0)

            if r.status in (RequestStatus.PENDING, RequestStatus.ACCEPTED):
                r.waiting_time = self.curr_time - r.request_time
                if r.waiting_time >= MAX_WAIT_TIME:
                    r.status = RequestStatus.CANCELLED
                    cancelled.append(r)

            elif r.status == RequestStatus.PICKEDUP:
                if r.pickup_at is not None:
                    r.in_vehicle_time = self.curr_time - r.pickup_at

        for r in cancelled:
            # Free the vehicle that was heading to pick up this request
            if r.assigned_v_id >= 0:
                for v in self.vehicle_list:
                    if (
                        v.id == r.assigned_v_id
                        and v.status == VehicleStatus.PICKUP
                        and v.target_request is r
                    ):
                        v.status = VehicleStatus.IDLE
                        v.next_node = 0
                        v.target_request = None
                        v.target_arrival_time = -1
                        if r in v.active_request_list:
                            v.active_request_list.remove(r)
                        break

            if r in self.active_request_list:
                self.active_request_list.remove(r)
            self.done_request_list.append(r)
            events.append({"type": "cancel", "request_id": r.id})

        # Update slot indices
        for idx, r in enumerate(self.active_request_list):
            r.slot_idx = idx

        return events

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _load_arriving_requests(self) -> None:
        """
        Move requests whose request_time <= curr_time into active list.
        Cap at MAX_NUM_REQUEST: overflow requests stay in future_request_list
        and are promoted when a slot opens (SERVED/CANCELLED removes one).
        Overflow requests do NOT accumulate waiting_time while queued here.
        """
        while (
            self.future_request_list
            and self.future_request_list[0].request_time <= self.curr_time
            and len(self.active_request_list) < MAX_NUM_REQUEST
        ):
            self.active_request_list.append(self.future_request_list.pop(0))

    def _complete_pickup(self, v: Vehicle, r: Request) -> None:
        """Finalise an immediate or deferred pickup."""
        v.num_passengers += r.num_passengers
        r.status = RequestStatus.PICKEDUP
        r.waiting_time = self.curr_time - r.request_time
        r.pickup_at = self.curr_time

    def _complete_dropoff(self, v: Vehicle, r: Request) -> None:
        """Finalise an immediate dropoff (same-node case)."""
        if r in v.active_request_list:
            v.active_request_list.remove(r)
        v.num_passengers -= r.num_passengers
        r.status = RequestStatus.SERVED
        r.arrival_due_left = max(r.arrival_due - self.curr_time, 0)
        r.in_vehicle_time = self.curr_time - (r.pickup_at or self.curr_time)
        r.dropoff_at = self.curr_time
        if r in self.active_request_list:
            self.active_request_list.remove(r)
        self.done_request_list.append(r)
        v.num_serve += 1

    def _compute_reward(self, events: list[dict], elapsed: int) -> float:
        reward = 0.0
        for ev in events:
            if ev["type"] == "pickup_complete":
                wt = ev.get("wait_time", 0)
                reward += _REWARD_PICKUP_COMPLETE * (
                    1.0 - min(wt, MAX_WAIT_TIME) / MAX_WAIT_TIME
                )
            elif ev["type"] == "dropoff_complete":
                ivt = ev.get("in_vehicle_time", 0)
                reward += _REWARD_DROPOFF_COMPLETE * (
                    1.0 - min(ivt, MAX_INVEHICLE_TIME) / MAX_INVEHICLE_TIME
                )
            elif ev["type"] == "cancel":
                reward += _PENALTY_CANCEL
        reward += _PENALTY_TIME * elapsed
        return reward

    def _is_done(self) -> bool:
        return (
            len(self.future_request_list) == 0
            and len(self.active_request_list) == 0
        )

    def _build_obs(self) -> np.ndarray:
        if self.pending_decision_vehicle is None:
            return np.zeros(OBS_DIM_DRT, dtype=np.float32)
        return build_drt_observation(
            pending_vehicle=self.pending_decision_vehicle,
            all_vehicles=self.vehicle_list,
            active_request_list=self.active_request_list,
            curr_time=self.curr_time,
            network=self._network,
        )
