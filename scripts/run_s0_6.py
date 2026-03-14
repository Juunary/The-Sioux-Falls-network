#!/usr/bin/env python
# ============================================================
# S0-6: PPO R8 500K full training
#
# Usage (run from project root):
#   python scripts/run_s0_6.py
#
# Output:
#   data/training/drt_r8_500k/
#     ppo_final.zip
#     vecnormalize.pkl
#     metrics.jsonl          (one record per rollout, real ep_rew_mean)
#     checkpoints/ppo_50000.zip  ... ppo_500000.zip
#
# Does NOT touch data/training/drt_smoke_test/.
# ============================================================

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]

_CONFIG = {
    "env_type":                    "drt",
    "drt_requests_path":           str(_ROOT / "KW_DRT" / "data" / "requests_8.csv"),
    "drt_vehicle_positions_path":  str(_ROOT / "KW_DRT" / "data" / "vehicle_positions.csv"),
    "drt_od_matrix_path":          str(_ROOT / "KW_DRT" / "data" / "od_matrix.csv"),
    "total_timesteps":             500_000,
    "n_envs":                      2,
    "n_steps":                     2048,
    "batch_size":                  256,
    "learning_rate":               3e-4,
    "gamma":                       0.99,
    "clip_range":                  0.2,
    "ent_coef":                    0.01,
    "checkpoint_freq":             50_000,
    "verbose":                     1,
    "seed":                        0,
}

if __name__ == "__main__":
    result = subprocess.run(
        [
            sys.executable,
            "-m", "backend.trainer.training_worker",
            "--job-id", "drt_r8_500k",
            "--config", json.dumps(_CONFIG),
        ],
        cwd=str(_ROOT),
    )
    sys.exit(result.returncode)
