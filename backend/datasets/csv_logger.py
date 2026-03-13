# ============================================================
# CSV Logger — writes episode results to structured CSV files
#
# Three outputs (all optional, path-driven):
#   write_request_trace()     — per-passenger lifecycle row
#   write_bus_trace()         — per-bus final-state row
#   append_episode_summary()  — one aggregate row appended to episodes.csv
# ============================================================

from __future__ import annotations

import csv
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.env.bus import BusState
    from backend.env.passenger import Passenger
    from backend.datasets.evaluator import EpisodeMetrics


_REQUEST_TRACE_FIELDNAMES = [
    "passenger_id",
    "origin_node",
    "destination_node",
    "state",
    "waiting_since",
    "pickup_at",
    "dropoff_at",
    "wait_time_sec",
    "in_vehicle_sec",
]

_BUS_TRACE_FIELDNAMES = [
    "bus_id",
    "final_node",
    "battery",
    "state",
    "passenger_count",
    "distance_px",
]

_EPISODE_SUMMARY_FIELDNAMES = [
    "scenario_id",
    "policy",
    "episode_reward",
    "passengers_spawned",
    "passengers_served",
    "passengers_gone",
    "service_rate",
    "unserved_rate",
    "avg_wait_time_sec",
    "avg_in_vehicle_sec",
    "charge_events",
    "total_distance_px",
    "wall_time_sec",
]


def write_request_trace(
    passengers: list["Passenger"],
    output_path: pathlib.Path,
) -> None:
    """Write one row per passenger to a CSV file.

    Columns: passenger_id, origin_node, destination_node, state,
             waiting_since, pickup_at, dropoff_at,
             wait_time_sec, in_vehicle_sec
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_REQUEST_TRACE_FIELDNAMES)
        writer.writeheader()
        for p in passengers:
            wait_time = (
                p.pickup_at - p.waiting_since
                if p.pickup_at is not None
                else 0.0
            )
            in_vehicle = (
                p.dropoff_at - p.pickup_at
                if p.pickup_at is not None and p.dropoff_at is not None
                else 0.0
            )
            writer.writerow({
                "passenger_id": p.id,
                "origin_node": p.origin_node,
                "destination_node": p.destination_node,
                "state": p.state,
                "waiting_since": p.waiting_since,
                "pickup_at": p.pickup_at if p.pickup_at is not None else "",
                "dropoff_at": p.dropoff_at if p.dropoff_at is not None else "",
                "wait_time_sec": round(wait_time, 4),
                "in_vehicle_sec": round(in_vehicle, 4),
            })


def write_bus_trace(
    buses: list["BusState"],
    output_path: pathlib.Path,
) -> None:
    """Write one row per bus to a CSV file.

    Columns: bus_id, final_node, battery, state,
             passenger_count, distance_px
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_BUS_TRACE_FIELDNAMES)
        writer.writeheader()
        for bus in buses:
            writer.writerow({
                "bus_id": bus.id,
                "final_node": bus.current_node,
                "battery": round(bus.battery, 2),
                "state": bus.state,
                "passenger_count": len(bus.passenger_ids),
                "distance_px": round(bus.distance_px, 2),
            })


def append_episode_summary(
    metrics: "EpisodeMetrics",
    output_path: pathlib.Path,
) -> None:
    """Append one aggregate row to episodes.csv.

    Creates the file with a header row if it does not already exist;
    otherwise appends without writing the header again.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with open(output_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_EPISODE_SUMMARY_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "scenario_id": metrics.scenario_id,
            "policy": metrics.policy,
            "episode_reward": round(metrics.episode_reward, 4),
            "passengers_spawned": metrics.passengers_spawned,
            "passengers_served": metrics.passengers_served,
            "passengers_gone": metrics.passengers_gone,
            "service_rate": round(metrics.service_rate, 4),
            "unserved_rate": round(metrics.unserved_rate, 4),
            "avg_wait_time_sec": round(metrics.avg_wait_time_sec, 4),
            "avg_in_vehicle_sec": round(metrics.avg_in_vehicle_sec, 4),
            "charge_events": metrics.charge_events,
            "total_distance_px": round(metrics.total_distance_px, 2),
            "wall_time_sec": round(metrics.wall_time_sec, 4),
        })
