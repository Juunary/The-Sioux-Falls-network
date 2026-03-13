# ============================================================
# Passenger data class and lifecycle logic
#
# Mirrors processPassengers() in src/sim/engine.ts.
#
# Tick order (must match TypeScript exactly):
#   1. Patience check  — waiting too long → 'gone'
#   2. Alighting       — riders at destination → 'arrived' (benchmark: no reboard)
#   3. Boarding        — waiting at bus node → 'riding' (bus.id asc order)
#
# Spawn:
#   - Happens AFTER processPassengers in every tick (step 3 in engine)
#   - Same-tick boarding after spawn is NOT allowed
#   - Spawn boundary: event.spawn_time ∈ (prev_sim_time, current_sim_time]
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from backend.env.bus import BusState

PassengerState = Literal["waiting", "riding", "arrived", "gone"]


@dataclass
class Passenger:
    id: int
    origin_node: int
    destination_node: int
    current_node: int
    bus_id: int | None
    state: PassengerState
    waiting_since: float
    patience: float
    pickup_at: float | None = None    # sim_time when passenger boarded a bus
    dropoff_at: float | None = None   # sim_time when passenger alighted at destination
    pickup_bus_distance_px: float | None = None   # bus.distance_px at boarding tick (before step_bus)
    dropoff_bus_distance_px: float | None = None  # bus.distance_px at alighting tick (before step_bus)


# ============================================================
# shouldBoard — mirrors shouldBoard() in engine.ts
# ============================================================

def should_board(
    bus: "BusState",
    passenger: Passenger,
    routing_mode: str,
) -> bool:
    """
    Determine if a waiting passenger boards the given bus.

    random mode: always board (Python env uses pre-generated actions — no 80%
    random boarding; simplified to always-board since routing is deterministic).
    Other modes: board only if destination is in bus.route or bus.destination.

    NOTE: In the Python env, buses follow exact routes from the PPO action or
    baseline — boarding is deterministic. The 80% random heuristic only applies
    to the TS free-play UI mode where routes are unpredictable.
    """
    if routing_mode == "random":
        return True  # deterministic env: always board in random mode
    return (
        passenger.destination_node in bus.route
        or passenger.destination_node == bus.destination
    )


# ============================================================
# processPassengers — tick step 1 (mirrors TypeScript)
# ============================================================

def process_passengers(
    buses: list["BusState"],
    passengers: list[Passenger],
    routing_mode: str,
    sim_time: float,
    reboard_enabled: bool,
) -> tuple[list["BusState"], list[Passenger]]:
    """
    Run the passenger lifecycle for one tick.

    Returns updated (buses, passengers).
    Buses are mutated in-place (list is copied before modification).
    Passengers list is returned as a new list.
    """
    from backend.env.bus import BusState  # local import to avoid circular

    # Deep-copy passenger list
    pax: list[Passenger] = list(passengers)

    # 1. Patience check
    pax = [
        replace(p, state="gone") if (p.state == "waiting" and sim_time - p.waiting_since > p.patience) else p
        for p in pax
    ]

    # Build index for fast lookup
    pax_by_id: dict[int, int] = {p.id: i for i, p in enumerate(pax)}

    # Work on bus copies (bus.id ascending — deterministic)
    updated_buses = [b.copy() for b in sorted(buses, key=lambda b: b.id)]

    # 2. Alighting — bus.id ascending order
    for bus in updated_buses:
        if bus.current_edge is not None:
            continue  # bus is between nodes — no alighting

        riders_at_dest = [
            p for p in pax
            if p.state == "riding"
            and p.bus_id == bus.id
            and p.destination_node == bus.current_node
        ]

        for p in riders_at_dest:
            bus.passenger_ids = [pid for pid in bus.passenger_ids if pid != p.id]
            idx = pax_by_id[p.id]
            if reboard_enabled:
                # In benchmark mode reboard_enabled=False, so this branch is inactive
                pax[idx] = replace(
                    p,
                    state="waiting",
                    bus_id=None,
                    origin_node=bus.current_node,
                    current_node=bus.current_node,
                    waiting_since=sim_time,
                    # destination assigned by env; simplified: keep destination for parity
                )
            else:
                pax[idx] = replace(p, state="arrived", bus_id=None,
                                   dropoff_bus_distance_px=bus.distance_px)

    # Rebuild index after alighting changes
    pax_by_id = {p.id: i for i, p in enumerate(pax)}

    # 3. Boarding — bus.id ascending order
    for bus in updated_buses:
        if bus.current_edge is not None:
            continue
        if bus.state == "charging":
            continue

        waiting_here = [
            p for p in pax
            if p.state == "waiting" and p.current_node == bus.current_node
        ]

        for p in waiting_here:
            if len(bus.passenger_ids) >= bus.capacity:
                break
            if not should_board(bus, p, routing_mode):
                continue

            bus.passenger_ids = bus.passenger_ids + [p.id]
            idx = pax_by_id[p.id]
            pax[idx] = replace(p, state="riding", bus_id=bus.id,
                               pickup_bus_distance_px=bus.distance_px)

    return updated_buses, pax


# ============================================================
# Passenger spawning from pre-generated replay queue
# ============================================================

def replay_spawn(
    passengers: list[Passenger],
    next_id: int,
    current_sim_time: float,
    queue: list,  # list[PassengerEvent] — consumed in place
) -> tuple[list[Passenger], int]:
    """
    Consume events from the replay queue with spawn_time ∈ (prev_t, current_sim_time].
    Returns updated (passengers, next_id).

    NOTE: This function does NOT filter by prev_time — caller passes the queue
    already positioned. Events are consumed from the front while spawn_time ≤ current_sim_time.
    """
    new_pax = list(passengers)
    pid = next_id

    while queue and queue[0].spawn_time <= current_sim_time:
        ev = queue.pop(0)
        new_pax.append(Passenger(
            id=pid,
            origin_node=ev.origin,
            destination_node=ev.destination,
            current_node=ev.origin,
            bus_id=None,
            state="waiting",
            waiting_since=ev.spawn_time,
            patience=ev.patience,
        ))
        pid += 1

    return new_pax, pid
