# ============================================================
# Training worker — subprocess entry point
#
# Invoked by job_manager.py as a subprocess:
#   python -m backend.trainer.training_worker --job-id JOB --config CONFIG_JSON
#
# Writes metrics to: data/training/<job_id>/metrics.jsonl
# Writes checkpoints to: data/training/<job_id>/checkpoints/
# ============================================================

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from stable_baselines3.common.callbacks import CallbackList

from backend.trainer.callbacks import CheckpointCallback, MetricsCallback
from backend.trainer.ppo_trainer import build_model
from backend.datasets.generator import list_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO training worker")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--config", required=True, help="JSON config string")
    args = parser.parse_args()

    config: dict = json.loads(args.config)
    job_id: str = args.job_id

    # ---- Directories ----
    job_dir = pathlib.Path("data") / "training" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = job_dir / "checkpoints"
    metrics_path = job_dir / "metrics.jsonl"

    # ---- Load scenarios ----
    split = config.get("split", "train")
    scenario_paths = list_scenarios(split=split)
    if not scenario_paths:
        print(f"[worker] ERROR: No scenarios found for split='{split}'", file=sys.stderr)
        sys.exit(1)

    print(f"[worker] job_id={job_id} | {len(scenario_paths)} scenarios | split={split}")

    # ---- Build model ----
    model, vec_env = build_model(
        scenario_paths=scenario_paths,
        n_envs=config.get("n_envs", 4),
        learning_rate=config.get("learning_rate", 3e-4),
        gamma=config.get("gamma", 0.95),
        clip_range=config.get("clip_range", 0.2),
        ent_coef=config.get("ent_coef", 0.01),
        n_steps=config.get("n_steps", 2048),
        batch_size=config.get("batch_size", 256),
        verbose=config.get("verbose", 0),
        seed=config.get("seed", 0),
    )

    # ---- Callbacks ----
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=config.get("checkpoint_freq", 50_000),
            save_dir=checkpoint_dir,
            name_prefix="ppo",
            verbose=1,
        ),
        MetricsCallback(
            output_path=metrics_path,
            verbose=1,
        ),
    ])

    # ---- Train ----
    total_timesteps = config.get("total_timesteps", 100_000)
    print(f"[worker] Training for {total_timesteps} timesteps …")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=True)

    # ---- Save final model ----
    final_path = job_dir / "ppo_final"
    model.save(final_path)
    vec_env.save(job_dir / "vecnormalize.pkl")
    print(f"[worker] Done. Final model: {final_path}")


if __name__ == "__main__":
    main()
