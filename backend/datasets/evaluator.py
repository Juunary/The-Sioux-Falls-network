# ============================================================
# Episode evaluator
#
# Runs a policy on one or more scenarios and returns EpisodeMetrics.
# Supports:
#   - demand_aware_greedy_v1 baseline (deterministic, no model loading)
#   - Pluggable policy callable: (obs, mask) → action
#
# Metrics (mirrors EpisodeMetrics in types/network.ts):
#   scenario_id, policy, episode_reward, passengers_spawned,
#   passengers_served, passengers_gone, service_rate, unserved_rate,
#   avg_wait_time_sec, charge_events, total_distance_px,
#   avg_in_vehicle_sec, avg_detour_px
# ============================================================

from __future__ import annotations

import pathlib
import json
import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import numpy as np

from backend.api.schemas import ScenarioData
from backend.baseline.demand_aware import demand_aware_greedy, BASELINE_NAME
from backend.datasets.csv_logger import (
    append_episode_summary,
    write_bus_trace,
    write_request_trace,
)
from backend.datasets.generator import list_scenarios, load_scenario
from backend.env.pathfinding import shortest_distance
from backend.env.sioux_falls_env import SiouxFallsEnv


# ============================================================
# Metrics
# ============================================================

@dataclass
class EpisodeMetrics:
    scenario_id: str
    policy: str
    episode_reward: float
    passengers_spawned: int
    passengers_served: int
    passengers_gone: int
    service_rate: float
    unserved_rate: float
    avg_wait_time_sec: float
    charge_events: int
    total_distance_px: float
    avg_in_vehicle_sec: float = 0.0
    avg_detour_px: float = 0.0
    wall_time_sec: float = 0.0


# ============================================================
# Policy type
# ============================================================

PolicyFn = Callable[[np.ndarray, np.ndarray], int]
# (obs: ndarray[96], action_mask: ndarray[24 bool]) → action: int


def greedy_policy_fn(obs: np.ndarray, mask: np.ndarray) -> int:
    """demand_aware_greedy_v1 policy wrapped as PolicyFn."""
    # Determine current bus node from one-hot encoding (obs[0:24])
    current_node = int(np.argmax(obs[0:24]))
    return demand_aware_greedy(obs, current_node, mask)


# ============================================================
# Single-episode runner
# ============================================================

def run_episode(
    scenario: ScenarioData,
    policy_fn: PolicyFn,
    policy_name: str,
    speed_multiplier: float = 1.0,
    output_csv_dir: Optional[pathlib.Path] = None,
) -> EpisodeMetrics:
    """
    Run a full episode with the given policy and return metrics.
    """
    # Hard wall-clock timeout: prevents infinite loops if env never signals done.
    # 120 s covers any realistic episode up to stress difficulty (600 s sim, many buses).
    MAX_WALL_SECS = 120.0

    t0 = time.perf_counter()

    env = SiouxFallsEnv(scenario, speed_multiplier=speed_multiplier)
    obs, _ = env.reset()

    total_reward = 0.0
    charge_events = 0

    while True:
        if env.pending_decision_bus is None:
            break
        if time.perf_counter() - t0 > MAX_WALL_SECS:
            break

        mask = env.action_masks()
        action = policy_fn(obs, mask)

        obs, reward, done, _, info = env.step(action)
        total_reward += reward

        # Count battery-depletion events (battery hit 0 during episode)
        for ev in info.get("events", []):
            if ev["type"] == "battery_depleted":
                charge_events += 1

        if done:
            break

    passengers_spawned = env.next_passenger_id
    passengers_served = env._arrived_count
    passengers_gone = env._gone_count

    service_rate = passengers_served / max(1, passengers_spawned)
    unserved_rate = passengers_gone / max(1, passengers_spawned)

    avg_wait = (
        float(np.mean(env._boarding_wait_times))
        if env._boarding_wait_times
        else 0.0
    )

    # avg_in_vehicle_sec: mean ride duration for passengers who completed their trip
    served_with_times = [
        p for p in env.passengers
        if p.state == "arrived"
        and p.pickup_at is not None
        and p.dropoff_at is not None
    ]
    in_vehicle_times = [p.dropoff_at - p.pickup_at for p in served_with_times]
    avg_in_vehicle = float(np.mean(in_vehicle_times)) if in_vehicle_times else 0.0

    # avg_detour_px: actual ride distance (bus odometer diff) minus shortest path distance
    # raw values — no clamp; minor negatives from FP/timing boundary are expected
    served_with_dist = [
        p for p in env.passengers
        if p.state == "arrived"
        and p.pickup_bus_distance_px is not None
        and p.dropoff_bus_distance_px is not None
    ]
    detour_values = [
        (p.dropoff_bus_distance_px - p.pickup_bus_distance_px)
        - shortest_distance(p.origin_node, p.destination_node)
        for p in served_with_dist
    ]
    avg_detour = float(np.mean(detour_values)) if detour_values else 0.0

    wall_time = time.perf_counter() - t0

    total_distance_px = sum(bus.distance_px for bus in env.buses)

    metrics = EpisodeMetrics(
        scenario_id=scenario.meta.scenario_id,
        policy=policy_name,
        episode_reward=total_reward,
        passengers_spawned=passengers_spawned,
        passengers_served=passengers_served,
        passengers_gone=passengers_gone,
        service_rate=service_rate,
        unserved_rate=unserved_rate,
        avg_wait_time_sec=avg_wait,
        charge_events=charge_events,
        total_distance_px=total_distance_px,
        avg_in_vehicle_sec=avg_in_vehicle,
        avg_detour_px=avg_detour,
        wall_time_sec=wall_time,
    )

    if output_csv_dir is not None:
        output_csv_dir.mkdir(parents=True, exist_ok=True)
        sid = scenario.meta.scenario_id
        write_request_trace(env.passengers, output_csv_dir / f"{sid}_requests.csv")
        write_bus_trace(env.buses, output_csv_dir / f"{sid}_buses.csv")
        append_episode_summary(metrics, output_csv_dir / "episodes.csv")

    return metrics


# ============================================================
# Batch evaluator
# ============================================================

def evaluate_policy_on_scenarios(
    scenario_paths: list[pathlib.Path],
    policy_fn: PolicyFn,
    policy_name: str,
    output_jsonl: Optional[pathlib.Path] = None,
    output_csv_dir: Optional[pathlib.Path] = None,
) -> list[EpisodeMetrics]:
    """
    Evaluate a policy on a list of scenario files.
    If output_jsonl is given, append each metric record as a JSONL line.
    If output_csv_dir is given, write per-episode request/bus trace CSVs and
    append a row to episodes.csv in that directory.
    """
    results: list[EpisodeMetrics] = []

    for path in scenario_paths:
        scenario = load_scenario(path)
        metrics = run_episode(scenario, policy_fn, policy_name, output_csv_dir=output_csv_dir)
        results.append(metrics)

        if output_jsonl is not None:
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with open(output_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(metrics)) + "\n")

    return results


def evaluate_baseline(
    split: str = "test",
    output_jsonl: Optional[pathlib.Path] = None,
    output_csv_dir: Optional[pathlib.Path] = None,
) -> list[EpisodeMetrics]:
    """
    Convenience: evaluate demand_aware_greedy_v1 on all scenarios in the given split.
    """
    paths = list_scenarios(split=split)
    if not paths:
        raise FileNotFoundError(
            f"No scenarios found for split='{split}'. "
            "Run POST /api/datasets/generate first."
        )
    return evaluate_policy_on_scenarios(
        paths, greedy_policy_fn, BASELINE_NAME, output_jsonl,
        output_csv_dir=output_csv_dir,
    )
