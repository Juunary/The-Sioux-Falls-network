# ============================================================
# DRT scenario generator — Stage 0.5
#
# Produces diversified request CSV files and a manifest.jsonl
# under KW_DRT/data/scenarios/ from two source files:
#   requests_80.csv  (seeds 0-69: train/val/test splits)
#   requests_8.csv   (test_r8_original: direct copy)
#
# Perturbation methods:
#   A) Request arrival jitter  — ±jitter_range ticks on Request_time
#   B) Request subset sampling — drop_rate fraction dropped;
#                                node_swap_prob per node after drop
#   C) Vehicle position perturbation — perturb_prob per used vehicle
#
# Output layout:
#   KW_DRT/data/scenarios/
#     manifest.jsonl
#     train/scenario_s{seed:02d}.csv        (seeds 0-49)
#     val/scenario_s{seed:02d}.csv          (seeds 50-59)
#     test/scenario_s{seed:02d}.csv         (seeds 60-69)
#     test/requests_8_original.csv
#     vehicle_positions/vp_s{seed:02d}.csv  (one per perturbed scenario)
#
# All manifest paths are relative to the manifest.jsonl directory.
# ============================================================

from __future__ import annotations

import json
import os
import pathlib
import shutil

import numpy as np
import pandas as pd


# ---- Seed offsets (keep perturbations statistically independent) ----
_SEED_JITTER    = 10_000
_SEED_SAMPLE    = 20_000
_SEED_VEHICLE   = 30_000

_MAX_EPISODE_TIME = 200   # clip ceiling for jittered Request_time


# ============================================================
# Internal helpers
# ============================================================

def _build_adjacency(travel_time_path: str | pathlib.Path) -> dict[int, list[int]]:
    """Build directed adjacency list from travel_time.csv (From → [To, ...])."""
    df = pd.read_csv(str(travel_time_path))
    adj: dict[int, list[int]] = {}
    for _, row in df.iterrows():
        src = int(row["From"])
        dst = int(row["To"])
        adj.setdefault(src, [])
        if dst not in adj[src]:
            adj[src].append(dst)
    return adj


def _apply_jitter(
    df: pd.DataFrame,
    seed: int,
    jitter_range: int,
) -> pd.DataFrame:
    """Perturb Request_time by ±jitter_range ticks (clipped to [0, MAX_EPISODE_TIME])."""
    rng = np.random.RandomState(seed + _SEED_JITTER)
    df = df.copy()
    delta = rng.randint(-jitter_range, jitter_range + 1, size=len(df))
    df["Request_time"] = (
        (df["Request_time"].to_numpy(dtype=int) + delta)
        .clip(0, _MAX_EPISODE_TIME)
    )
    # Re-sort by Request_time to maintain the loader's expected ordering
    df = df.sort_values("Request_time").reset_index(drop=True)
    return df


def _apply_sampling_and_swap(
    df: pd.DataFrame,
    seed: int,
    drop_rate: float,
    node_swap_prob: float,
    adjacency: dict[int, list[int]],
) -> pd.DataFrame:
    """Drop a fraction of requests and optionally swap Start/End nodes."""
    rng = np.random.RandomState(seed + _SEED_SAMPLE)
    n = len(df)
    n_keep = max(1, int(round(n * (1.0 - drop_rate))))
    keep_idx = sorted(rng.choice(n, size=n_keep, replace=False).tolist())
    df = df.iloc[keep_idx].copy().reset_index(drop=True)

    for i in range(len(df)):
        start = int(df.at[i, "Start_node"])
        end   = int(df.at[i, "End_node"])
        new_start, new_end = start, end

        if rng.random() < node_swap_prob and start in adjacency:
            candidates = [v for v in adjacency[start] if v != end]
            if candidates:
                new_start = int(rng.choice(candidates))

        if rng.random() < node_swap_prob and end in adjacency:
            candidates = [v for v in adjacency[end] if v != new_start]
            if candidates:
                new_end = int(rng.choice(candidates))

        df.at[i, "Start_node"] = new_start
        df.at[i, "End_node"]   = new_end

    return df


def _apply_vehicle_perturbation(
    veh_df: pd.DataFrame,
    seed: int,
    perturb_prob: float,
    adjacency: dict[int, list[int]],
    max_num_vehicles: int,
) -> pd.DataFrame:
    """Perturb initial_position of the first max_num_vehicles rows."""
    rng = np.random.RandomState(seed + _SEED_VEHICLE)
    df = veh_df.copy()
    for i in range(min(max_num_vehicles, len(df))):
        if rng.random() < perturb_prob:
            node = int(df.at[i, "initial_position"])
            neighbors = adjacency.get(node, [])
            if neighbors:
                df.at[i, "initial_position"] = int(rng.choice(neighbors))
    return df


def _write_requests_csv(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def _write_vehicles_csv(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


# ============================================================
# Public entry point
# ============================================================

def generate_all_scenarios(
    requests_80_path: str | pathlib.Path,
    requests_8_path: str | pathlib.Path,
    vehicle_positions_path: str | pathlib.Path,
    od_matrix_path: str | pathlib.Path,
    travel_time_path: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    jitter_range: int = 2,
    drop_rate: float = 0.1,
    node_swap_prob: float = 0.05,
    perturb_prob: float = 0.5,
    env_version: str = "drt_env_v1",
    comparison_group: str = "L0",
    max_num_vehicles: int = 2,
    n_train: int = 50,
    n_val: int = 10,
    n_test_r80: int = 10,
) -> pathlib.Path:
    """Generate diversified scenarios and write manifest.jsonl.

    Returns:
        Path to the written manifest.jsonl
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adjacency = _build_adjacency(travel_time_path)

    req80_df = pd.read_csv(str(requests_80_path))
    veh_df   = pd.read_csv(str(vehicle_positions_path))

    # Relative paths in manifest are resolved against manifest_dir (= output_dir)
    od_rel = os.path.relpath(
        pathlib.Path(od_matrix_path).resolve(),
        output_dir.resolve(),
    ).replace("\\", "/")

    manifest_entries: list[dict] = []

    # ---- Split definitions -----------------------------------------------
    splits: list[tuple[str, range]] = [
        ("train", range(0,          n_train)),
        ("val",   range(n_train,    n_train + n_val)),
        ("test",  range(n_train + n_val, n_train + n_val + n_test_r80)),
    ]

    for split_name, seed_range in splits:
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        vp_dir = output_dir / "vehicle_positions"
        vp_dir.mkdir(exist_ok=True)

        for seed in seed_range:
            # A) Jitter
            req_df = _apply_jitter(req80_df, seed, jitter_range)
            # B) Sampling + node swap
            req_df = _apply_sampling_and_swap(
                req_df, seed, drop_rate, node_swap_prob, adjacency
            )
            # C) Vehicle perturbation
            vp_df = _apply_vehicle_perturbation(
                veh_df, seed, perturb_prob, adjacency, max_num_vehicles
            )

            req_filename = f"scenario_s{seed:02d}.csv"
            vp_filename  = f"vp_s{seed:02d}.csv"

            req_path = split_dir / req_filename
            vp_path  = vp_dir / vp_filename

            _write_requests_csv(req_df, req_path)
            _write_vehicles_csv(vp_df, vp_path)

            # scenario_id naming: train_s00 … val_s50 … test_s60 …
            scenario_id = f"{split_name}_s{seed:02d}"

            manifest_entries.append({
                "scenario_id":             scenario_id,
                "split":                   split_name,
                "comparison_group":        comparison_group,
                "requests_path":           f"{split_name}/{req_filename}",
                "vehicle_positions_path":  f"vehicle_positions/{vp_filename}",
                "od_matrix_path":          od_rel,
                "source_requests":         pathlib.Path(requests_80_path).name,
                "seed":                    seed,
                "jitter_range":            jitter_range,
                "drop_rate":               drop_rate,
                "node_swap_prob":          node_swap_prob,
                "perturb_prob":            perturb_prob,
                "env_version":             env_version,
                "num_requests":            len(req_df),
            })

    # ---- test_r8_original ------------------------------------------------
    test_dir = output_dir / "test"
    test_dir.mkdir(exist_ok=True)
    r8_dest = test_dir / "requests_8_original.csv"
    shutil.copy2(str(requests_8_path), str(r8_dest))

    # Original vehicle_positions.csv lives one level up from output_dir
    vp_original_rel = os.path.relpath(
        pathlib.Path(vehicle_positions_path).resolve(),
        output_dir.resolve(),
    ).replace("\\", "/")

    req8_df = pd.read_csv(str(requests_8_path))
    manifest_entries.append({
        "scenario_id":             "test_r8_original",
        "split":                   "test",
        "comparison_group":        comparison_group,
        "requests_path":           "test/requests_8_original.csv",
        "vehicle_positions_path":  vp_original_rel,
        "od_matrix_path":          od_rel,
        "source_requests":         pathlib.Path(requests_8_path).name,
        "seed":                    -1,
        "jitter_range":            0,
        "drop_rate":               0.0,
        "node_swap_prob":          0.0,
        "perturb_prob":            0.0,
        "env_version":             env_version,
        "num_requests":            len(req8_df),
    })

    # ---- Write manifest.jsonl -------------------------------------------
    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(
        f"[generator] {len(manifest_entries)} scenarios written to {output_dir}\n"
        f"  train={n_train}  val={n_val}  test={n_test_r80 + 1} (incl. r8_original)\n"
        f"  manifest -> {manifest_path}"
    )
    return manifest_path
