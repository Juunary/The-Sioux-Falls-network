#!/usr/bin/env python
# ============================================================
# S0.5-4: PPO training on manifest train split + val evaluation
#
# Usage (run from project root):
#   python scripts/run_s0_5_4.py
#
# Phases:
#   1. Train PPO on manifest train split (500K steps, n_envs=2)
#   2. Evaluate trained model on val split (10 scenarios)
#   3. Evaluate trained model on train split (all 50 scenarios)
#   4. Compare train vs val -> overfitting judgment
#
# Output:
#   data/training/drt_manifest_train_500k/
#     ppo_final.zip
#     vecnormalize.pkl
#     metrics.jsonl
#     checkpoints/
#   data/drt_benchmarks/s054_val/episodes.csv
#   data/drt_benchmarks/s054_train/episodes.csv
# ============================================================

from __future__ import annotations

import json
import pathlib
import statistics
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
# Ensure project root is on sys.path so backend imports work in-process
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_MANIFEST = _ROOT / "KW_DRT" / "data" / "scenarios" / "manifest.jsonl"
_JOB_ID   = "drt_manifest_train_500k"
_JOB_DIR  = _ROOT / "data" / "training" / _JOB_ID
_CKPT     = _JOB_DIR / "ppo_final.zip"

_TRAIN_CONFIG = {
    "env_type":        "drt",
    "manifest_path":   str(_MANIFEST),
    "split":           "train",
    "env_version":     "drt_env_v1",
    "total_timesteps": 500_000,
    "n_envs":          2,
    "n_steps":         2048,
    "batch_size":      256,
    "learning_rate":   3e-4,
    "gamma":           0.99,
    "clip_range":      0.2,
    "ent_coef":        0.01,
    "checkpoint_freq": 100_000,
    "verbose":         1,
    "seed":            0,
}


def _mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def _std(vals: list[float]) -> float:
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def _print_split_summary(split: str, metrics_list) -> None:
    serve_rates  = [m.serve_rate       for m in metrics_list]
    rewards      = [m.total_reward     for m in metrics_list]
    cancel_rates = [m.cancel_rate      for m in metrics_list]
    wait_times   = [m.mean_wait_time   for m in metrics_list]

    print(f"\n  --- {split.upper()} split ({len(metrics_list)} scenarios) ---")
    print(f"  serve_rate    : {_mean(serve_rates):.4f}  (std={_std(serve_rates):.4f})")
    print(f"  cancel_rate   : {_mean(cancel_rates):.4f}  (std={_std(cancel_rates):.4f})")
    print(f"  total_reward  : {_mean(rewards):.4f}  (std={_std(rewards):.4f})")
    print(f"  mean_wait_time: {_mean(wait_times):.4f}  ticks (std={_std(wait_times):.4f})")


# ============================================================
# Phase 1 — Training
# ============================================================

def phase_train() -> int:
    if _CKPT.exists():
        print(f"[S0.5-4] Checkpoint already exists: {_CKPT}")
        print("[S0.5-4] Skipping training phase.")
        return 0

    print("=" * 60)
    print("[S0.5-4] Phase 1: Training PPO on manifest train split")
    print(f"         manifest : {_MANIFEST}")
    print(f"         output   : {_JOB_DIR}")
    print("=" * 60)

    result = subprocess.run(
        [
            sys.executable,
            "-m", "backend.trainer.training_worker",
            "--job-id", _JOB_ID,
            "--config", json.dumps(_TRAIN_CONFIG),
        ],
        cwd=str(_ROOT),
    )
    return result.returncode


# ============================================================
# Phase 2+3 — Evaluation
# ============================================================

def phase_evaluate() -> None:
    from sb3_contrib import MaskablePPO
    from backend.datasets.drt_evaluator import (
        evaluate_drt_policy_on_manifest,
        make_ppo_policy_fn,
    )

    print("\n" + "=" * 60)
    print("[S0.5-4] Phase 2+3: Evaluating on val and train splits")
    print("=" * 60)

    model = MaskablePPO.load(str(_CKPT))
    policy_fn = make_ppo_policy_fn(model)

    val_out   = _ROOT / "data" / "drt_benchmarks" / "s054_val"
    train_out = _ROOT / "data" / "drt_benchmarks" / "s054_train"

    # --- val ---
    print("\n[S0.5-4] Evaluating val split (10 scenarios)...")
    val_metrics = evaluate_drt_policy_on_manifest(
        manifest_path=str(_MANIFEST),
        split="val",
        policy_fn=policy_fn,
        policy_name="ppo_manifest_train_500k",
        env_version="drt_env_v1",
        output_csv_dir=val_out,
    )

    # --- train ---
    print("[S0.5-4] Evaluating train split (50 scenarios)...")
    train_metrics = evaluate_drt_policy_on_manifest(
        manifest_path=str(_MANIFEST),
        split="train",
        policy_fn=policy_fn,
        policy_name="ppo_manifest_train_500k",
        env_version="drt_env_v1",
        output_csv_dir=train_out,
    )

    # ============================================================
    # Phase 4 — Comparison report
    # ============================================================

    print("\n" + "=" * 60)
    print("[S0.5-4] Phase 4: Train vs Val comparison")
    print("=" * 60)

    _print_split_summary("train", train_metrics)
    _print_split_summary("val",   val_metrics)

    train_sr = _mean([m.serve_rate   for m in train_metrics])
    val_sr   = _mean([m.serve_rate   for m in val_metrics])
    gap_sr   = train_sr - val_sr

    train_rw = _mean([m.total_reward for m in train_metrics])
    val_rw   = _mean([m.total_reward for m in val_metrics])
    gap_rw   = train_rw - val_rw

    print(f"\n  serve_rate  gap (train - val): {gap_sr:+.4f}")
    print(f"  total_reward gap (train - val): {gap_rw:+.4f}")

    # Conservative overfitting judgment
    print("\n  --- Overfitting judgment ---")
    if gap_sr > 0.15:
        verdict = "OVERFITTING (serve_rate gap > 0.15)"
    elif gap_sr > 0.08:
        verdict = "MILD OVERFIT (serve_rate gap 0.08-0.15) - monitor"
    elif abs(gap_sr) <= 0.08:
        verdict = "GENERALISING (gap <= 0.08) - acceptable"
    else:
        verdict = f"VAL > TRAIN by {-gap_sr:.4f} - unusual, check data"

    print(f"  Verdict: {verdict}")
    print(f"\n  CSV outputs:")
    print(f"    val   -> {val_out / 'episodes.csv'}")
    print(f"    train -> {train_out / 'episodes.csv'}")
    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    if not _MANIFEST.exists():
        print(f"[ERROR] manifest.jsonl not found: {_MANIFEST}")
        sys.exit(1)

    rc = phase_train()
    if rc != 0:
        print(f"[ERROR] Training failed with exit code {rc}")
        sys.exit(rc)

    phase_evaluate()
