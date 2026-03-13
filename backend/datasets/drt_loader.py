# ============================================================
# DRT data loader — reads CSV data files and builds domain
# objects for DRTEnv.
# Ported from KW_DRT/app/env_builder.py
# ============================================================

from __future__ import annotations

import csv
import pathlib

import pandas as pd

from backend.env.drt_config import MAX_NUM_VEHICLES
from backend.env.drt_network import DRTNetwork
from backend.env.drt_request import Request
from backend.env.drt_vehicle import Vehicle


def load_drt_data(
    requests_path: str | pathlib.Path,
    vehicle_positions_path: str | pathlib.Path,
    od_matrix_path: str | pathlib.Path,
) -> tuple[DRTNetwork, list[Request], list[int]]:
    """
    Load all DRT data from CSV files.

    Returns:
        network              — DRTNetwork with OD matrix loaded
        request_list         — sorted by request_time ascending
        vehicle_init_pos     — list of 1-based starting node IDs
    """
    network = DRTNetwork()
    network.set_od_matrix(str(od_matrix_path))

    request_list = _load_requests(str(requests_path), network)
    for r in request_list:
        r.set_travel_time(network.get_duration(r.from_node_id, r.to_node_id))

    vehicle_positions: list[int] = (
        pd.read_csv(str(vehicle_positions_path))["initial_position"]
        .tolist()[:MAX_NUM_VEHICLES]  # mirror KW_DRT initialize_vehicles() range(MAX_NUM_VEHICLES)
    )

    return network, request_list, vehicle_positions


def _load_requests(path: str, network: DRTNetwork) -> list[Request]:
    requests: list[Request] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            req = Request(
                request_id=int(row["User_ID"]),
                from_node_id=int(row["Start_node"]),
                to_node_id=int(row["End_node"]),
                request_time=int(row["Request_time"]),
                network=network,
            )
            requests.append(req)
    return sorted(requests, key=lambda r: r.request_time)
