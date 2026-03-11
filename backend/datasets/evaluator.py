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
#   avg_wait_time_sec, charge_events, total_distance_px
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
from backend.datasets.generator import list_scenarios, load_scenario
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

        # Count charge events from events list
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

    wall_time = time.perf_counter() - t0

    return EpisodeMetrics(
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
        total_distance_px=env._total_distance_px,
        wall_time_sec=wall_time,
    )


# ============================================================
# Batch evaluator
# ============================================================

def evaluate_policy_on_scenarios(
    scenario_paths: list[pathlib.Path],
    policy_fn: PolicyFn,
    policy_name: str,
    output_jsonl: Optional[pathlib.Path] = None,
) -> list[EpisodeMetrics]:
    """
    Evaluate a policy on a list of scenario files.
    If output_jsonl is given, append each metric record as a JSONL line.
    """
    results: list[EpisodeMetrics] = []

    for path in scenario_paths:
        scenario = load_scenario(path)
        metrics = run_episode(scenario, policy_fn, policy_name)
        results.append(metrics)

        if output_jsonl is not None:
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with open(output_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(metrics)) + "\n")

    return results


def evaluate_baseline(
    split: str = "test",
    output_jsonl: Optional[pathlib.Path] = None,
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
        paths, greedy_policy_fn, BASELINE_NAME, output_jsonl
    )
