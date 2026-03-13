# ============================================================
# DRT episode evaluator
#
# run_drt_episode(env, policy_fn, ...)  → DRTEpisodeMetrics
#   - works with any policy: greedy baseline OR PPO
#   - writes per-episode CSVs if output_csv_dir is given
#
# evaluate_drt_baseline(...)  — convenience wrapper for greedy policy
# make_ppo_policy_fn(model)   — wrap SB3 MaskablePPO as PolicyFnDRT
#
# CSV outputs (all under output_csv_dir / episode_id /):
#   requests.csv   — per-request trace
#   vehicles.csv   — per-vehicle final state
#   ../episodes.csv — cumulative summary (appended)
# ============================================================

from __future__ import annotations

import csv
import pathlib
import time
from dataclasses import dataclass, fields
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np

from backend.env.drt_request import RequestStatus

if TYPE_CHECKING:
    from backend.env.drt_env import DRTEnv

PolicyFnDRT = Callable[[np.ndarray, np.ndarray], int]

_MAX_STEPS = 5_000   # hard truncation per episode

# ---- CSV fieldnames ------------------------------------------------

_REQUEST_FIELDS = [
    "request_id", "from_node", "to_node", "request_time", "status",
    "assigned_vehicle_id", "waiting_time", "in_vehicle_time",
    "travel_time", "detour_ticks", "pickup_at", "dropoff_at",
]

_VEHICLE_FIELDS = [
    "vehicle_id", "final_node", "num_accept", "num_serve", "idle_time",
]

_EPISODE_FIELDS = [
    "episode_id", "policy", "served_count", "cancelled_count", "total_requests",
    "serve_rate", "cancel_rate", "mean_wait_time", "mean_in_vehicle_time",
    "mean_detour_ticks", "total_reward", "total_steps", "total_ticks",
    "wall_time_sec",
]


# ---- Metrics dataclass ---------------------------------------------

@dataclass
class DRTEpisodeMetrics:
    episode_id: str
    policy: str
    served_count: int
    cancelled_count: int
    total_requests: int
    serve_rate: float           # served / total_requests
    cancel_rate: float          # cancelled / total_requests
    mean_wait_time: float       # mean waiting_time for SERVED (ticks)
    mean_in_vehicle_time: float # mean in_vehicle_time for SERVED (ticks)
    mean_detour_ticks: float    # mean (in_vehicle_time - travel_time) for SERVED where travel_time > 0
    total_reward: float
    total_steps: int
    total_ticks: int
    wall_time_sec: float

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ---- Core runner ---------------------------------------------------

def run_drt_episode(
    env: "DRTEnv",
    policy_fn: PolicyFnDRT,
    policy_name: str,
    episode_id: str = "ep_0",
    output_csv_dir: Optional[pathlib.Path] = None,
) -> DRTEpisodeMetrics:
    """
    Run one DRT episode with the given policy.

    env is reset() internally; the caller should NOT call reset() first.

    Args:
        env:           DRTEnv instance (will be reset inside)
        policy_fn:     (obs, mask) -> action
        policy_name:   label written to metrics / CSVs
        episode_id:    unique label for this run (used in CSV filenames)
        output_csv_dir: if given, writes CSVs under output_csv_dir/episode_id/
                        and appends a row to output_csv_dir/episodes.csv

    Returns:
        DRTEpisodeMetrics
    """
    t0 = time.monotonic()

    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    truncated = False

    while not (done or truncated):
        mask = env.action_masks()
        action = policy_fn(obs, mask)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if steps >= _MAX_STEPS:
            truncated = True

    wall_time = time.monotonic() - t0

    # ---- Aggregate metrics from done_request_list ----
    all_done = env.done_request_list  # all SERVED + CANCELLED
    served = [r for r in all_done if r.status == RequestStatus.SERVED]
    cancelled = [r for r in all_done if r.status == RequestStatus.CANCELLED]

    n_total = len(all_done)
    n_served = len(served)
    n_cancelled = len(cancelled)

    serve_rate = n_served / n_total if n_total > 0 else 0.0
    cancel_rate = n_cancelled / n_total if n_total > 0 else 0.0

    wait_times = [r.waiting_time for r in served if r.waiting_time >= 0]
    ivt_times = [r.in_vehicle_time for r in served]
    detour_vals = [
        r.in_vehicle_time - r.travel_time
        for r in served
        if r.travel_time > 0
    ]

    mean_wait = float(np.mean(wait_times)) if wait_times else 0.0
    mean_ivt = float(np.mean(ivt_times)) if ivt_times else 0.0
    mean_detour = float(np.mean(detour_vals)) if detour_vals else 0.0

    metrics = DRTEpisodeMetrics(
        episode_id=episode_id,
        policy=policy_name,
        served_count=n_served,
        cancelled_count=n_cancelled,
        total_requests=n_total,
        serve_rate=round(serve_rate, 4),
        cancel_rate=round(cancel_rate, 4),
        mean_wait_time=round(mean_wait, 4),
        mean_in_vehicle_time=round(mean_ivt, 4),
        mean_detour_ticks=round(mean_detour, 4),
        total_reward=round(total_reward, 4),
        total_steps=steps,
        total_ticks=env.curr_time,
        wall_time_sec=round(wall_time, 4),
    )

    if output_csv_dir is not None:
        _write_csvs(env, metrics, output_csv_dir)

    return metrics


# ---- CSV writers ---------------------------------------------------

def _write_csvs(
    env: "DRTEnv",
    metrics: DRTEpisodeMetrics,
    output_csv_dir: pathlib.Path,
) -> None:
    ep_dir = output_csv_dir / metrics.episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    _write_request_trace(env.done_request_list, ep_dir / "requests.csv")
    _write_vehicle_trace(env.vehicle_list, ep_dir / "vehicles.csv")
    _append_episode_summary(metrics, output_csv_dir / "episodes.csv")


def _write_request_trace(requests, path: pathlib.Path) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_REQUEST_FIELDS)
        writer.writeheader()
        for r in requests:
            served = r.status == RequestStatus.SERVED
            detour = (
                round(r.in_vehicle_time - r.travel_time, 4)
                if (served and r.travel_time > 0)
                else ""
            )
            writer.writerow({
                "request_id":        r.id,
                "from_node":         r.from_node_id,
                "to_node":           r.to_node_id,
                "request_time":      r.request_time,
                "status":            r.status.name,
                "assigned_vehicle_id": r.assigned_v_id,
                "waiting_time":      r.waiting_time if r.waiting_time >= 0 else "",
                "in_vehicle_time":   r.in_vehicle_time if served else "",
                "travel_time":       round(r.travel_time, 4) if r.travel_time > 0 else "",
                "detour_ticks":      detour,
                "pickup_at":         r.pickup_at if r.pickup_at is not None else "",
                "dropoff_at":        r.dropoff_at if r.dropoff_at is not None else "",
            })


def _write_vehicle_trace(vehicles, path: pathlib.Path) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_VEHICLE_FIELDS)
        writer.writeheader()
        for v in vehicles:
            writer.writerow({
                "vehicle_id":  v.id,
                "final_node":  v.curr_node,
                "num_accept":  v.num_accept,
                "num_serve":   v.num_serve,
                "idle_time":   v.idle_time,
            })


def _append_episode_summary(
    metrics: DRTEpisodeMetrics,
    path: pathlib.Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_EPISODE_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(metrics.to_dict())


# ---- Convenience wrappers ------------------------------------------

def evaluate_drt_baseline(
    requests_path: str,
    vehicle_positions_path: str,
    od_matrix_path: str,
    episode_id: str = "greedy_ep",
    output_csv_dir: Optional[pathlib.Path] = None,
) -> DRTEpisodeMetrics:
    """
    Run one episode with the DRT greedy baseline policy and return metrics.
    No PPO checkpoint required.
    """
    from backend.env.drt_env import DRTEnv
    from backend.baseline.drt_greedy import drt_greedy_policy, BASELINE_NAME

    env = DRTEnv(requests_path, vehicle_positions_path, od_matrix_path)
    return run_drt_episode(
        env,
        policy_fn=drt_greedy_policy,
        policy_name=BASELINE_NAME,
        episode_id=episode_id,
        output_csv_dir=output_csv_dir,
    )


def evaluate_drt_policy_on_scenarios(
    requests_paths: list[str],
    vehicle_positions_path: str,
    od_matrix_path: str,
    policy_fn: PolicyFnDRT,
    policy_name: str,
    output_csv_dir: Optional[pathlib.Path] = None,
) -> list[DRTEpisodeMetrics]:
    """
    Run one DRT episode per entry in requests_paths.

    episode_id is derived from the requests file stem
    (e.g. "requests_8" for "KW_DRT/data/requests_8.csv").

    Args:
        requests_paths:       List of paths to requests CSV files.
        vehicle_positions_path: Path to vehicle positions CSV.
        od_matrix_path:       Path to OD matrix CSV.
        policy_fn:            (obs, mask) -> action  — greedy or PPO wrapper.
        policy_name:          Label written to metrics / CSVs.
        output_csv_dir:       If given, CSVs are written under
                              output_csv_dir/{episode_id}/ per episode,
                              and a row is appended to output_csv_dir/episodes.csv.

    Returns:
        List of DRTEpisodeMetrics, one per requests_path.
    """
    from backend.env.drt_env import DRTEnv

    results: list[DRTEpisodeMetrics] = []
    for req_path in requests_paths:
        episode_id = pathlib.Path(req_path).stem
        env = DRTEnv(req_path, vehicle_positions_path, od_matrix_path)
        metrics = run_drt_episode(
            env,
            policy_fn=policy_fn,
            policy_name=policy_name,
            episode_id=episode_id,
            output_csv_dir=output_csv_dir,
        )
        results.append(metrics)
    return results


def make_ppo_policy_fn(model) -> PolicyFnDRT:
    """
    Wrap a trained SB3 MaskablePPO model as a PolicyFnDRT.

    Usage:
        policy = make_ppo_policy_fn(model)
        metrics = run_drt_episode(env, policy, "maskable_ppo", ...)
    """
    def _policy(obs: np.ndarray, mask: np.ndarray) -> int:
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)
    return _policy
