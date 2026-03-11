# ============================================================
# Scenario generator for the Sioux Falls Network Simulator
#
# Determinism guarantee:
#   generate_scenario(seed=N, ...) always produces the same JSON.
#   All stochastic choices (bus start node, battery, speed, every
#   passenger spawn time, OD pair, patience) are derived exclusively
#   from numpy.random.default_rng(seed).
#
# Graph version: v1 (76 directed edges, locked before dataset generation)
# SIM_DT: 0.016 s — must match TypeScript SIM_DT constant exactly
# ============================================================

from __future__ import annotations

import json
import math
import pathlib
from typing import Literal

import numpy as np

from backend.api.schemas import (
    BusInitial,
    GenerateRequest,
    GenerateResponse,
    PassengerEvent,
    ScenarioData,
    ScenarioMeta,
)

# ---- Constants ----

GRAPH_VERSION = "v1"
GENERATOR_VERSION = "1.0"
SIM_DT = 0.016  # seconds — must match src/sim/engine.ts SIM_DT

# 24 nodes (0-indexed)
NUM_NODES = 24

# Bus speed range: 80–120 px/sec (mirrors makeBus() in store.ts)
BUS_SPEED_MIN = 80.0
BUS_SPEED_MAX = 120.0

# Bus battery range: 60–100 %
BUS_BATTERY_MIN = 60.0
BUS_BATTERY_MAX = 100.0

# Patience range: 20–60 s (mirrors maybeSpawnPassengers() in engine.ts)
PATIENCE_MIN = 20.0
PATIENCE_MAX = 60.0

# Difficulty → spawn rate multiplier
DIFFICULTY_SPAWN_MULTIPLIER: dict[str, float] = {
    "easy": 0.5,
    "medium": 1.0,
    "hard": 1.5,
    "stress": 2.5,
}

# Default data directory (relative to project root)
_DEFAULT_DATA_DIR = pathlib.Path(__file__).parent.parent.parent / "data" / "scenarios"


# ============================================================
# Core generator
# ============================================================

def generate_scenario(
    seed: int,
    split: Literal["train", "val", "test"] = "train",
    difficulty: Literal["easy", "medium", "hard", "stress"] = "medium",
    bus_count: int = 6,
    episode_length: float = 600.0,
    spawn_rate: float = 0.8,
    passenger_capacity: int = 6,
    reboard_enabled: bool = False,
    charging_enabled: bool = True,
    low_battery_threshold: float = 20.0,
    charge_duration: float = 8.0,
    battery_drain_rate: float = 0.08,
) -> ScenarioData:
    """
    Generate one reproducible scenario from a single integer seed.

    All random draws use numpy.random.default_rng(seed) exclusively.
    Same seed → byte-identical output regardless of platform.
    """
    rng = np.random.default_rng(seed)

    effective_spawn_rate = spawn_rate * DIFFICULTY_SPAWN_MULTIPLIER[difficulty]

    # ---- Buses ----
    buses: list[BusInitial] = []
    for i in range(bus_count):
        buses.append(BusInitial(
            id=i,
            start_node=int(rng.integers(0, NUM_NODES)),
            battery=float(BUS_BATTERY_MIN + rng.random() * (BUS_BATTERY_MAX - BUS_BATTERY_MIN)),
            speed=float(BUS_SPEED_MIN + rng.random() * (BUS_SPEED_MAX - BUS_SPEED_MIN)),
        ))

    # ---- Passenger events (pre-generated Poisson process) ----
    # We sample all spawn times up front using the inter-arrival time method
    # (exponential with rate = effective_spawn_rate), then assign OD and patience.
    passenger_events: list[PassengerEvent] = []
    event_id = 0
    t = 0.0

    while True:
        # Exponential inter-arrival time: E[X] = 1 / rate
        inter_arrival = float(rng.exponential(1.0 / effective_spawn_rate))
        t += inter_arrival
        if t > episode_length:
            break

        origin = int(rng.integers(0, NUM_NODES))
        # Destination must differ from origin
        dest_candidates = [n for n in range(NUM_NODES) if n != origin]
        destination = int(dest_candidates[rng.integers(0, len(dest_candidates))])
        patience = float(PATIENCE_MIN + rng.random() * (PATIENCE_MAX - PATIENCE_MIN))

        passenger_events.append(PassengerEvent(
            event_id=event_id,
            spawn_time=round(t, 6),   # round to microsecond for compact JSON
            origin=origin,
            destination=destination,
            patience=round(patience, 4),
        ))
        event_id += 1

    # ---- Scenario ID ----
    scenario_id = f"{split}_{seed:06d}"

    meta = ScenarioMeta(
        scenario_id=scenario_id,
        graph_version=GRAPH_VERSION,
        seed=seed,
        split=split,
        difficulty=difficulty,
        episode_length=episode_length,
        sim_dt=SIM_DT,
        bus_count=bus_count,
        spawn_rate=effective_spawn_rate,
        reboard_enabled=reboard_enabled,
        charging_enabled=charging_enabled,
        low_battery_threshold=low_battery_threshold,
        charge_duration=charge_duration,
        battery_drain_rate=battery_drain_rate,
        passenger_capacity=passenger_capacity,
        generator_version=GENERATOR_VERSION,
    )

    return ScenarioData(meta=meta, buses_initial=buses, passenger_events=passenger_events)


# ============================================================
# Batch generation + file I/O
# ============================================================

def generate_batch(
    request: GenerateRequest,
    output_dir: pathlib.Path | None = None,
) -> GenerateResponse:
    """
    Generate `request.count` scenarios starting from `request.seed_start`.
    Each scenario is saved as  <output_dir>/<scenario_id>.json.
    Returns the list of scenario IDs and the output directory.
    """
    if output_dir is None:
        output_dir = _DEFAULT_DATA_DIR / request.split

    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_ids: list[str] = []

    for i in range(request.count):
        seed = request.seed_start + i
        scenario = generate_scenario(
            seed=seed,
            split=request.split,
            difficulty=request.difficulty,
            bus_count=request.bus_count,
            episode_length=request.episode_length,
            spawn_rate=request.spawn_rate,
            passenger_capacity=request.passenger_capacity,
            reboard_enabled=request.reboard_enabled,
            charging_enabled=request.charging_enabled,
            low_battery_threshold=request.low_battery_threshold,
            charge_duration=request.charge_duration,
            battery_drain_rate=request.battery_drain_rate,
        )

        out_path = output_dir / f"{scenario.meta.scenario_id}.json"
        out_path.write_text(
            json.dumps(scenario.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        scenario_ids.append(scenario.meta.scenario_id)

    return GenerateResponse(
        generated=request.count,
        scenario_ids=scenario_ids,
        output_dir=str(output_dir),
    )


def load_scenario(path: pathlib.Path) -> ScenarioData:
    """Load and validate a scenario JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return ScenarioData.model_validate(data)


def list_scenarios(
    data_dir: pathlib.Path | None = None,
    split: str | None = None,
) -> list[pathlib.Path]:
    """Return sorted list of scenario JSON paths, optionally filtered by split."""
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR

    if split:
        paths = sorted((data_dir / split).glob("*.json"))
    else:
        paths = sorted(data_dir.glob("**/*.json"))

    return paths
