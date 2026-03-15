#!/usr/bin/env python
# ============================================================
# S0.5-5: Greedy baseline on test split (manifest mode)
#
# Usage (run from project root):
#   python scripts/run_s0_5_5.py
#
# Output:
#   data/drt_benchmarks/greedy_test_split/
#     episodes.csv                       -- all 11 episodes appended
#     test_r8_original/requests.csv
#     test_r8_original/vehicles.csv
#     test_s60/requests.csv
#     test_s60/vehicles.csv
#     ...
# ============================================================

from __future__ import annotations

import pathlib
import statistics
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_MANIFEST  = _ROOT / "KW_DRT" / "data" / "scenarios" / "manifest.jsonl"
_OUTPUT    = _ROOT / "data" / "drt_benchmarks" / "greedy_test_split"
_ENV_VER   = "drt_env_v1"
_SPLIT     = "test"


def _mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def _std(vals: list[float]) -> float:
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def main() -> None:
    if not _MANIFEST.exists():
        print(f"[ERROR] manifest.jsonl not found: {_MANIFEST}")
        sys.exit(1)

    if _OUTPUT.exists() and (_OUTPUT / "episodes.csv").exists():
        print(f"[S0.5-5] Output already exists: {_OUTPUT / 'episodes.csv'}")
        print("[S0.5-5] Remove it to re-run. Aborting.")
        sys.exit(1)

    from backend.baseline.drt_greedy import drt_greedy_policy, BASELINE_NAME
    from backend.datasets.drt_evaluator import evaluate_drt_policy_on_manifest
    from backend.experiments.tracker import (
        create_experiment, record_drt_metrics, aggregate_drt_metrics,
    )
    from backend.env.drt_env_config import ENV_CONFIGS

    cfg = ENV_CONFIGS[_ENV_VER]

    print("=" * 60)
    print("[S0.5-5] Greedy baseline on test split")
    print(f"  manifest  : {_MANIFEST}")
    print(f"  split     : {_SPLIT}")
    print(f"  env_ver   : {_ENV_VER}  (comparison_group={cfg.comparison_group})")
    print(f"  policy    : {BASELINE_NAME}")
    print(f"  output    : {_OUTPUT}")
    print("=" * 60)

    # ---- Run evaluation ----
    metrics_list = evaluate_drt_policy_on_manifest(
        manifest_path=str(_MANIFEST),
        split=_SPLIT,
        policy_fn=drt_greedy_policy,
        policy_name=BASELINE_NAME,
        env_version=_ENV_VER,
        output_csv_dir=_OUTPUT,
    )

    # ---- Record in experiments DB ----
    from datetime import datetime, timezone
    exp_id = f"greedy_test_split_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    create_experiment(
        experiment_id=exp_id,
        policy=BASELINE_NAME,
        split=_SPLIT,
        config={
            "manifest_path": str(_MANIFEST),
            "split": _SPLIT,
            "env_version": _ENV_VER,
        },
        domain="drt",
        env_version=_ENV_VER,
        comparison_group=cfg.comparison_group,
    )
    record_drt_metrics(exp_id, metrics_list)
    agg = aggregate_drt_metrics(exp_id)

    # ---- Partition results ----
    r80_metrics = [m for m in metrics_list if m.episode_id != "test_r8_original"]
    r8_metrics  = [m for m in metrics_list if m.episode_id == "test_r8_original"]

    # ---- Report ----
    print()
    print("=" * 60)
    print("[S0.5-5] Results")
    print("=" * 60)

    def _print_group(label: str, mlist) -> None:
        if not mlist:
            print(f"\n  {label}: (empty)")
            return
        sr  = [m.serve_rate       for m in mlist]
        cr  = [m.cancel_rate      for m in mlist]
        rw  = [m.total_reward     for m in mlist]
        st  = [m.total_steps      for m in mlist]
        tk  = [m.total_ticks      for m in mlist]
        wt  = [m.mean_wait_time   for m in mlist]
        req = [m.total_requests   for m in mlist]
        print(f"\n  [{label}]  n={len(mlist)}")
        print(f"    serve_rate    : {_mean(sr):.4f}  (std={_std(sr):.4f}  min={min(sr):.4f}  max={max(sr):.4f})")
        print(f"    cancel_rate   : {_mean(cr):.4f}  (std={_std(cr):.4f})")
        print(f"    total_reward  : {_mean(rw):.4f}  (std={_std(rw):.4f})")
        print(f"    total_steps   : {_mean(st):.1f}  (std={_std(st):.1f})")
        print(f"    total_ticks   : {_mean(tk):.1f}  (std={_std(tk):.1f})")
        print(f"    mean_wait_time: {_mean(wt):.4f}  ticks")
        print(f"    total_requests: {_mean(req):.1f}  (mean per scenario)")

    _print_group("ALL TEST (11 scenarios)", metrics_list)
    _print_group("R80-series  (10 scenarios, test_r8_original excluded)", r80_metrics)

    print(f"\n  [test_r8_original (R8 original, n=1)]")
    if r8_metrics:
        m = r8_metrics[0]
        print(f"    serve_rate    : {m.serve_rate:.4f}")
        print(f"    cancel_rate   : {m.cancel_rate:.4f}")
        print(f"    total_reward  : {m.total_reward:.4f}")
        print(f"    total_steps   : {m.total_steps}")
        print(f"    total_ticks   : {m.total_ticks}")
        print(f"    mean_wait_time: {m.mean_wait_time:.4f}  ticks")
        print(f"    total_requests: {m.total_requests}")
    else:
        print("    (not found in metrics_list)")

    print(f"\n  experiment_id : {exp_id}")
    print(f"  DB aggregate  : {agg}")
    print()
    print("  Output files:")
    print(f"    episodes.csv -> {_OUTPUT / 'episodes.csv'}")
    for ep_dir in sorted(_OUTPUT.iterdir()):
        if ep_dir.is_dir():
            csvs = [f.name for f in ep_dir.iterdir() if f.suffix == ".csv"]
            print(f"    {ep_dir.name}/  {csvs}")

    print()
    print("[S0.5-5] DONE")


if __name__ == "__main__":
    main()
