# ============================================================
# SiouxFallsEnv — Gymnasium environment for the Sioux Falls
# PPO bus routing task.
#
# Semantics: event-driven semi-MDP.
# One env.step(action) = one bus's one routing decision + advance
# simulation to the next decision frontier.
#
# Decision frontier: after each tick, collect all buses in
# (idle | rerouting) state that:
#   - Have no remaining route[]
#   - Do NOT need emergency charging (those are auto-routed)
# Process frontier in bus.id ascending order without advancing time.
#
# Tick order (matches TypeScript engine.ts exactly):
#   1. processPassengers()
#   2. stepBus() for each bus in bus.id ascending order
#   3. replaySpawn() — consume passenger_events queue
#
# Battery emergency override does NOT consume a PPO step.
# ============================================================

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import replace
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from backend.api.schemas import ScenarioData
from backend.env.bus import BusState, _decide_next_route, _route_to_nearest_charger
from backend.env.network import (
    ALL_NODE_IDS,
    CHARGING_STATION_IDS,
    EDGES,
    NODES,
)
from backend.env.observation import OBS_DIM, build_observation
from backend.env.passenger import Passenger, process_passengers, replay_spawn
from backend.env.pathfinding import REACHABILITY, dijkstra

SIM_DT = 0.016  # must match TypeScript SIM_DT constant


# ============================================================
# Reward constants (from plan Section 10)
# ============================================================
REWARD_DELIVERED = 2.0
REWARD_BOARDED = 0.3
PENALTY_GONE = -1.5
PENALTY_TIME_STEP = -0.005      # per elapsed sim second
PENALTY_EMPTY_MOVE_COEFF = -0.2  # * (dist / MAX_DIST) per move without passengers
PENALTY_BATTERY_DEPLETION = -3.0


class SiouxFallsEnv(gym.Env):
    """
    Gymnasium environment for the Sioux Falls bus routing task.

    observation_space: Box(96,) all values in [0, 1]
    action_space:      Discrete(24) — target node for pending_decision_bus

    Use with MaskablePPO from sb3-contrib:
        model = MaskablePPO("MlpPolicy", env, ...)
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        scenario: ScenarioData,
        speed_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.speed_multiplier = speed_multiplier

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(ALL_NODE_IDS))

        # Initialized in reset()
        self.buses: list[BusState] = []
        self.passengers: list[Passenger] = []
        self.sim_time: float = 0.0
        self.next_passenger_id: int = 0
        self.replay_queue: list = []    # list[PassengerEvent]
        self.pending_decision_bus: Optional[BusState] = None
        self._deferred_frontier: list[BusState] = []

        # Metrics
        self._arrived_count: int = 0
        self._gone_count: int = 0
        self._boarded_this_step: int = 0
        # Evaluated metrics (populated during episode, read by evaluator)
        self._boarding_wait_times: list[float] = []   # wait secs per passenger that boarded
        self._total_distance_px: float = 0.0          # cumulative edge distance all buses

    # ----------------------------------------------------------
    # reset
    # ----------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        meta = self.scenario.meta

        # Build buses from scenario
        self.buses = [
            BusState(
                id=b.id,
                current_node=b.start_node,
                current_edge=None,
                progress=0.0,
                speed=b.speed,
                battery=b.battery,
                state="idle",
                route=[],
                destination=None,
                charge_time_left=0.0,
                passenger_ids=[],
                capacity=meta.passenger_capacity,
            )
            for b in self.scenario.buses_initial
        ]

        self.passengers = []
        self.sim_time = 0.0
        self.next_passenger_id = 0
        self.replay_queue = list(self.scenario.passenger_events)
        self.pending_decision_bus = None
        self._deferred_frontier = []
        self._arrived_count = 0
        self._gone_count = 0
        self._boarded_this_step = 0
        self._boarding_wait_times = []
        self._total_distance_px = 0.0

        # Advance to first decision
        elapsed, _ = self._advance_to_next_decision()

        obs = self._build_observation()
        return obs, {"elapsed_sim_time": elapsed}

    # ----------------------------------------------------------
    # step
    # ----------------------------------------------------------

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.pending_decision_bus is not None, "Call reset() before step()"

        bus = self.pending_decision_bus
        meta = self.scenario.meta

        # Apply action: route this bus to `action` node
        self._apply_action(bus, action)

        # Sync back into buses list
        for i, b in enumerate(self.buses):
            if b.id == bus.id:
                self.buses[i] = bus
                break

        # Advance to next decision frontier
        elapsed, events = self._advance_to_next_decision()

        reward = self._compute_reward(events, elapsed)

        obs = self._build_observation()
        done = self.sim_time >= meta.episode_length
        truncated = False

        info = {
            "elapsed_sim_time": elapsed,
            "events": events,
            "raw_reward": reward,
            "sim_time": self.sim_time,
            "arrived": self._arrived_count,
            "gone": self._gone_count,
        }

        return obs, reward, done, truncated, info

    # ----------------------------------------------------------
    # Action masking (for MaskablePPO)
    # ----------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """
        Returns bool[24]: True = valid action.
        Rules:
          - current_node is always invalid (no self-loop)
          - unreachable nodes are invalid
        """
        bus = self.pending_decision_bus
        if bus is None:
            return np.ones(len(ALL_NODE_IDS), dtype=bool)

        mask = np.array(REACHABILITY[bus.current_node], dtype=bool)
        mask[bus.current_node] = False  # current node always invalid
        return mask

    # ----------------------------------------------------------
    # Internal: apply action → route bus
    # ----------------------------------------------------------

    def _apply_action(self, bus: BusState, target_node: int) -> None:
        """
        Route bus to target_node using Dijkstra shortest path.
        Enforces route invariant.
        """
        from backend.env.bus import _start_route
        from backend.env.pathfinding import reconstruct_path

        dist_map, prev_map = dijkstra(bus.current_node)
        path = reconstruct_path(bus.current_node, target_node, prev_map)

        if len(path) < 2:
            bus.state = "idle"
            return

        _start_route(bus, path, target_node)

    # ----------------------------------------------------------
    # Internal: advance simulation to next decision frontier
    # ----------------------------------------------------------

    def _advance_to_next_decision(self) -> tuple[float, list[dict]]:
        """
        Run simulation ticks until a bus needs a routing decision.
        Returns (elapsed_sim_time, list_of_reward_events).

        Decision frontier rules:
          - After each tick, collect idle/rerouting buses with empty route
            and sufficient battery (battery >= threshold if at charger, else auto-route)
          - Battery-emergency buses are auto-routed WITHOUT consuming a PPO step
          - If deferred frontier is non-empty from the previous tick, serve next bus first
          - Process frontier in bus.id ascending order
        """
        meta = self.scenario.meta
        events: list[dict] = []
        elapsed = 0.0

        # Serve deferred frontier from previous tick first (no time advance)
        if self._deferred_frontier:
            self.pending_decision_bus = self._deferred_frontier.pop(0)
            return 0.0, []

        while self.sim_time < meta.episode_length:
            # Run one tick
            tick_events = self._tick(SIM_DT)
            self.sim_time += SIM_DT
            elapsed += SIM_DT
            events.extend(tick_events)

            # Auto-route battery-emergency buses (no PPO step)
            for bus in self.buses:
                if bus.state in ("idle", "rerouting") and self._needs_emergency_charge(bus):
                    _route_to_nearest_charger(bus)

            # Collect decision frontier (bus.id ascending)
            frontier = sorted(
                [
                    b for b in self.buses
                    if b.state in ("idle", "rerouting")
                    and not b.route
                    and not self._needs_emergency_charge(b)
                    and b.current_edge is None
                ],
                key=lambda b: b.id,
            )

            if frontier:
                self.pending_decision_bus = frontier[0]
                self._deferred_frontier = frontier[1:]
                return elapsed, events

        # Episode ended
        self.pending_decision_bus = None
        return elapsed, events

    # ----------------------------------------------------------
    # Internal: one tick
    # ----------------------------------------------------------

    def _tick(self, dt: float) -> list[dict]:
        """
        Run one simulation tick. Returns list of reward-relevant events.
        Tick order:
          1. processPassengers
          2. stepBus (bus.id asc)
          3. replaySpawn
        """
        meta = self.scenario.meta
        events: list[dict] = []

        prev_states = {p.id: p.state for p in self.passengers}
        prev_bus_passengers = {b.id: set(b.passenger_ids) for b in self.buses}

        # 1. Process passengers
        updated_buses, updated_pax = process_passengers(
            buses=self.buses,
            passengers=self.passengers,
            routing_mode=meta.routing_mode if hasattr(meta, "routing_mode") else "greedy",
            sim_time=self.sim_time + dt,
            reboard_enabled=meta.reboard_enabled,
        )

        # Detect arrived / gone events
        for i, p in enumerate(updated_pax):
            prev = prev_states.get(p.id)
            if prev == "riding" and p.state == "arrived":
                events.append({"type": "arrived", "passenger_id": p.id})
                self._arrived_count += 1
                if p.dropoff_at is None:
                    updated_pax[i] = replace(p, dropoff_at=self.sim_time + dt)
            elif prev == "waiting" and p.state == "gone":
                events.append({"type": "gone", "passenger_id": p.id})
                self._gone_count += 1

        # Detect boarded events; record boarding wait time (spawn → board)
        pax_by_id = {p.id: p for p in updated_pax}
        pax_index = {p.id: i for i, p in enumerate(updated_pax)}
        for bus in updated_buses:
            prev_pids = prev_bus_passengers.get(bus.id, set())
            new_pids = set(bus.passenger_ids) - prev_pids
            for pid in new_pids:
                events.append({"type": "boarded", "passenger_id": pid, "bus_id": bus.id})
                pax = pax_by_id.get(pid)
                if pax is not None:
                    self._boarding_wait_times.append(
                        max(0.0, (self.sim_time + dt) - pax.waiting_since)
                    )
                    if pax.pickup_at is None:
                        updated_pax[pax_index[pid]] = replace(pax, pickup_at=self.sim_time + dt)

        # 2. Step buses (bus.id ascending)
        stepped_buses = []
        for bus in sorted(updated_buses, key=lambda b: b.id):
            # If this bus is a frontier candidate (needs PPO decision), skip step_bus.
            # Charging-in-place and emergency charge are NOT frontier candidates and
            # are still processed normally via step_bus.
            if self._is_ppo_frontier_candidate(bus, meta):
                stepped_buses.append(bus)
                continue
            from backend.env.bus import step_bus
            new_bus = step_bus(
                bus=bus,
                dt=dt,
                speed_multiplier=self.speed_multiplier,
                battery_drain_rate=meta.battery_drain_rate,
                charging_enabled=meta.charging_enabled,
                low_battery_threshold=meta.low_battery_threshold,
                charge_duration=meta.charge_duration,
                routing_mode="greedy",  # for autonomous non-PPO steps within tick
                all_buses=updated_buses,
                passengers=updated_pax,
            )
            # Detect battery depletion
            if new_bus.battery == 0.0 and bus.battery > 0.0:
                events.append({"type": "battery_depleted", "bus_id": bus.id})
            # Accumulate distance: bus just completed an edge (currentEdge None→None transition
            # via node arrival detected as old edge present, new edge absent)
            if bus.current_edge is not None and new_bus.current_edge is None:
                edge_weight = EDGES[bus.current_edge].weight
                self._total_distance_px += edge_weight
                new_bus.distance_px += edge_weight
            stepped_buses.append(new_bus)

        # 3. Spawn passengers from replay queue
        updated_pax, self.next_passenger_id = replay_spawn(
            passengers=updated_pax,
            next_id=self.next_passenger_id,
            current_sim_time=self.sim_time + dt,
            queue=self.replay_queue,
        )

        self.buses = stepped_buses
        self.passengers = updated_pax
        return events

    # ----------------------------------------------------------
    # Internal: reward computation
    # ----------------------------------------------------------

    def _compute_reward(self, events: list[dict], elapsed: float) -> float:
        """Compute step reward from accumulated events."""
        reward = 0.0

        for ev in events:
            if ev["type"] == "arrived":
                reward += REWARD_DELIVERED
            elif ev["type"] == "gone":
                reward += PENALTY_GONE
            elif ev["type"] == "boarded":
                reward += REWARD_BOARDED
            elif ev["type"] == "battery_depleted":
                reward += PENALTY_BATTERY_DEPLETION

        # Time penalty proportional to elapsed sim time
        reward += PENALTY_TIME_STEP * elapsed

        return reward

    # ----------------------------------------------------------
    # Internal: helpers
    # ----------------------------------------------------------

    def _is_ppo_frontier_candidate(self, bus: BusState, meta) -> bool:
        """
        True if this bus needs a PPO routing decision (should not be auto-routed).
        A bus is a frontier candidate when it is idle/rerouting with an empty route,
        no current edge, and neither emergency-charge nor charge-in-place applies.
        Those two charging transitions are handled autonomously by step_bus.
        """
        if bus.state not in ("idle", "rerouting"):
            return False
        if bus.route or bus.current_edge is not None:
            return False
        if self._needs_emergency_charge(bus):
            return False
        # Charge-in-place: at a charging station with low battery → let step_bus handle it
        if meta.charging_enabled:
            node = NODES[bus.current_node]
            if node.is_charging_station and bus.battery <= meta.low_battery_threshold + 5:
                return False
        return True

    def _needs_emergency_charge(self, bus: BusState) -> bool:
        meta = self.scenario.meta
        if not meta.charging_enabled:
            return False
        node = NODES[bus.current_node]
        return (
            bus.battery < meta.low_battery_threshold
            and not node.is_charging_station
        )

    def _build_observation(self) -> np.ndarray:
        if self.pending_decision_bus is None:
            return np.zeros(OBS_DIM, dtype=np.float32)

        return build_observation(
            bus=self.pending_decision_bus,
            all_buses=self.buses,
            passengers=self.passengers,
            sim_time=self.sim_time,
            episode_length=self.scenario.meta.episode_length,
        )

    def _get_routing_mode(self) -> str:
        meta = self.scenario.meta
        return getattr(meta, "routing_mode", "greedy")
