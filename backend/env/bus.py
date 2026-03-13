# ============================================================
# Bus data class and physics
#
# Mirrors stepBus() / stepMoving() / stepCharging() in engine.ts
# and decideNextRoute() in routing.ts.
#
# Route State Invariant (must be maintained by ALL branches):
#   IF bus.current_edge is not None:
#     route contains only nodes AFTER current_edge.target
#     (route[0] != current_edge target)
#   IF bus.current_edge is None:
#     route[0] is the next node to move toward (if route non-empty)
#     empty route → bus is idle, needs new decision
#
# rerouting is treated identically to idle (MVP).
# ============================================================

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Literal, Optional

from backend.env.network import EDGES, NODES, find_edge
from backend.env.pathfinding import (
    dijkstra,
    insertion_heuristic_tour,
    nearest_charging_station,
    reconstruct_path,
)

BusStateType = Literal["moving", "charging", "idle", "rerouting"]


@dataclass
class BusState:
    id: int
    current_node: int
    current_edge: Optional[int]   # edge id, or None if at a node
    progress: float               # 0.0 → 1.0 along current_edge
    speed: float                  # px/sec
    battery: float                # 0–100 %
    state: BusStateType
    route: list[int]              # nodes to visit after current_edge.target
    destination: Optional[int]
    charge_time_left: float
    passenger_ids: list[int]
    capacity: int
    distance_px: float = 0.0  # cumulative distance this bus has traveled (px)

    def copy(self) -> "BusState":
        b = copy.copy(self)
        b.route = list(self.route)
        b.passenger_ids = list(self.passenger_ids)
        return b


# ============================================================
# Route helpers (enforce invariant)
# ============================================================

def _start_route(bus: BusState, path: list[int], destination: int) -> BusState:
    """
    Apply route invariant: set current_edge to first hop, route to remainder.
    path = [current_node, node1, node2, ..., dest]

    Route invariant: route must NOT contain current_edge.target (= path[1]).
    So: current_edge = edge(path[0]→path[1]), route = path[2:]
    """
    if len(path) < 2:
        bus.state = "idle"
        return bus
    first_hop = path[1]
    edge = find_edge(path[0], first_hop)
    if edge is None:
        bus.state = "idle"
        return bus
    bus.state = "moving"
    bus.current_edge = edge.id
    bus.progress = 0.0
    bus.route = path[2:]   # invariant: first_hop excluded
    bus.destination = destination
    return bus


def _route_to_nearest_charger(bus: BusState) -> BusState:
    """Route bus to nearest charging station. Respects route invariant."""
    result = nearest_charging_station(bus.current_node)
    if result is None or len(result[1]) < 2:
        bus.state = "idle"
        return bus
    charger_id, path = result
    return _start_route(bus, path, charger_id)


# ============================================================
# stepBus — mirrors stepBus() in engine.ts
# ============================================================

def step_bus(
    bus: BusState,
    dt: float,
    speed_multiplier: float,
    battery_drain_rate: float,
    charging_enabled: bool,
    low_battery_threshold: float,
    charge_duration: float,
    routing_mode: str,
    all_buses: list[BusState],
    passengers: list,  # list[Passenger]
) -> BusState:
    """
    Advance one bus by dt seconds. Returns a new BusState (copy).
    Mirrors the switch in stepBus() in engine.ts.
    """
    bus = bus.copy()

    if bus.state == "charging":
        return _step_charging(bus, dt, charge_duration)
    elif bus.state == "moving":
        return _step_moving(bus, dt, speed_multiplier, battery_drain_rate)
    elif bus.state in ("idle", "rerouting"):
        # rerouting is treated identically to idle (MVP)
        return _decide_next_route(
            bus,
            charging_enabled=charging_enabled,
            low_battery_threshold=low_battery_threshold,
            charge_duration=charge_duration,
            routing_mode=routing_mode,
            all_buses=all_buses,
            passengers=passengers,
        )
    return bus


def _step_charging(bus: BusState, dt: float, charge_duration: float) -> BusState:
    """Mirrors stepCharging() in engine.ts."""
    charge_rate = 100.0 / charge_duration
    bus.battery = min(100.0, bus.battery + charge_rate * dt)
    bus.charge_time_left -= dt
    if bus.charge_time_left <= 0:
        bus.battery = 100.0
        bus.charge_time_left = 0.0
        # Transition to idle — will trigger decideNextRoute next tick
        bus.state = "idle"
    return bus


def _step_moving(bus: BusState, dt: float, speed_multiplier: float, battery_drain_rate: float) -> BusState:
    """Mirrors stepMoving() in engine.ts."""
    if bus.current_edge is None:
        bus.state = "idle"
        return bus

    edge = EDGES[bus.current_edge]
    if edge is None:
        bus.state = "idle"
        return bus

    progress_per_second = (bus.speed * speed_multiplier) / edge.weight
    new_progress = bus.progress + progress_per_second * dt
    distance_traveled = progress_per_second * dt * edge.weight
    bus.battery = max(0.0, bus.battery - battery_drain_rate * distance_traveled)

    if new_progress >= 1.0:
        # Arrive at target node — idle for one tick so processPassengers can act
        bus.current_node = edge.target
        bus.current_edge = None
        bus.progress = 0.0
        bus.state = "idle"
    else:
        bus.progress = new_progress

    return bus


# ============================================================
# decideNextRoute — mirrors decideNextRoute() in routing.ts
# ============================================================

def _decide_next_route(
    bus: BusState,
    charging_enabled: bool,
    low_battery_threshold: float,
    charge_duration: float,
    routing_mode: str,
    all_buses: list[BusState],
    passengers: list,
) -> BusState:
    """
    Decision order (mirrors routing.ts decideNextRoute):
    1. Charge in place if at charging station with low battery
    2. Battery emergency → route to nearest charger (no PPO step)
    3. Continue existing route[] if non-empty
    4. Choose new destination based on routing_mode
    """
    node = NODES[bus.current_node]

    # 1. Charge in place
    if (
        charging_enabled
        and bus.state != "charging"
        and node.is_charging_station
        and bus.battery <= low_battery_threshold + 5
    ):
        bus.state = "charging"
        bus.current_edge = None
        bus.progress = 0.0
        bus.route = []
        bus.destination = None
        bus.charge_time_left = charge_duration
        return bus

    # 2. Battery emergency
    if charging_enabled and bus.battery < low_battery_threshold and not node.is_charging_station:
        return _route_to_nearest_charger(bus)

    # 3. Continue existing route
    if bus.route:
        next_node = bus.route[0]
        edge = find_edge(bus.current_node, next_node)
        if edge:
            bus.state = "moving"
            bus.current_edge = edge.id
            bus.progress = 0.0
            bus.route = bus.route[1:]
            return bus

    # 4. New destination by routing mode
    return _new_destination(bus, routing_mode, all_buses, passengers)


def _new_destination(
    bus: BusState,
    routing_mode: str,
    all_buses: list[BusState],
    passengers: list,
) -> BusState:
    """Choose a new destination. Called only when route[] is exhausted."""
    from backend.env.network import ALL_NODE_IDS, ADJACENCY

    if routing_mode == "random":
        return _random_decision(bus)
    elif routing_mode == "shortest":
        return _shortest_decision(bus)
    elif routing_mode == "insertion":
        return _insertion_decision(bus)
    elif routing_mode in ("greedy", "ppo"):
        # ppo falls back to greedy for autonomous (non-env) step calls
        return _greedy_decision(bus, passengers)
    return _random_decision(bus)


def _random_decision(bus: BusState) -> BusState:
    """Pick the lowest-id outgoing edge (deterministic fallback — random not needed in PPO env)."""
    from backend.env.network import ADJACENCY, EDGES as _EDGES

    out_edges = ADJACENCY.get(bus.current_node, [])
    if not out_edges:
        bus.state = "idle"
        return bus

    edge_id = min(out_edges)   # deterministic: smallest edge id
    edge = _EDGES[edge_id]
    bus.state = "moving"
    bus.current_edge = edge_id
    bus.progress = 0.0
    bus.route = []
    bus.destination = edge.target
    return bus


def _shortest_decision(bus: BusState) -> BusState:
    """Pick lowest-id reachable destination and route shortest path (deterministic fallback)."""
    from backend.env.network import ALL_NODE_IDS

    dest = min(n for n in ALL_NODE_IDS if n != bus.current_node)
    dist, prev = dijkstra(bus.current_node)
    path = reconstruct_path(bus.current_node, dest, prev)
    if len(path) < 2:
        bus.state = "idle"
        return bus
    return _start_route(bus, path, dest)


def _insertion_decision(bus: BusState) -> BusState:
    """Pick first 4 waypoints by node id (deterministic fallback — random not needed in PPO env)."""
    from backend.env.network import ALL_NODE_IDS

    NUM_WAYPOINTS = 4
    candidates = sorted(n for n in ALL_NODE_IDS if n != bus.current_node)
    waypoints = candidates[:NUM_WAYPOINTS]
    ordered = insertion_heuristic_tour(bus.current_node, waypoints)

    full_path: list[int] = [bus.current_node]
    cur = bus.current_node
    for wp in ordered:
        dist_map, prev_map = dijkstra(cur)
        seg = reconstruct_path(cur, wp, prev_map)
        if len(seg) > 1:
            full_path.extend(seg[1:])
        cur = wp

    if len(full_path) < 2:
        bus.state = "idle"
        return bus
    return _start_route(bus, full_path, full_path[-1])


def _greedy_decision(bus: BusState, passengers: list) -> BusState:
    """Demand-aware greedy: go to node with most waiting passengers. Mirrors greedyDecision()."""
    from backend.env.network import ALL_NODE_IDS

    waiting_by_node: dict[int, int] = {}
    for p in passengers:
        if p.state == "waiting":
            waiting_by_node[p.current_node] = waiting_by_node.get(p.current_node, 0) + 1

    dist_map, prev_map = dijkstra(bus.current_node)

    best_node = -1
    best_score = -math.inf

    for node_id in ALL_NODE_IDS:
        if node_id == bus.current_node:
            continue
        if math.isinf(dist_map.get(node_id, math.inf)):
            continue
        score = waiting_by_node.get(node_id, 0)
        # Tie-break: smaller node_id wins (deterministic)
        if score > best_score or (score == best_score and node_id < best_node):
            best_score = score
            best_node = node_id

    if best_node == -1:
        return _random_decision(bus)

    dist_map2, prev_map2 = dijkstra(bus.current_node)
    path = reconstruct_path(bus.current_node, best_node, prev_map2)
    if len(path) < 2:
        bus.state = "idle"
        return bus
    return _start_route(bus, path, best_node)
