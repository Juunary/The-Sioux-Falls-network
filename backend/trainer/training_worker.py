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
from backend.trainer.ppo_trainer import build_model, build_drt_model
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

    env_type = config.get("env_type", "sf")

    if env_type == "drt":
        # ---- DRT: Dynamic Ride-Sharing environment ----
        manifest_path  = config.get("manifest_path")   # Stage 0.5+ canonical field
        split          = config.get("split", "train")
        env_version    = config.get("env_version", "drt_env_v1")
        requests_path  = config.get("drt_requests_path")
        vehicle_pos_path = config.get("drt_vehicle_positions_path")
        od_matrix_path = config.get("drt_od_matrix_path")

        if manifest_path:
            # Manifest mode (Stage 0.5+): ignores individual CSV path fields
            print(
                f"[worker] job_id={job_id} | env_type=drt | "
                f"manifest={manifest_path} | split={split}"
            )
            model, vec_env = build_drt_model(
                manifest_path=manifest_path,
                split=split,
                env_version=env_version,
                n_envs=config.get("n_envs", 2),
                learning_rate=config.get("learning_rate", 3e-4),
                gamma=config.get("gamma", 0.99),
                clip_range=config.get("clip_range", 0.2),
                ent_coef=config.get("ent_coef", 0.01),
                n_steps=config.get("n_steps", 2048),
                batch_size=config.get("batch_size", 256),
                verbose=config.get("verbose", 0),
                seed=config.get("seed", 0),
            )
        else:
            # Single-CSV mode (backward-compatible with Stage 0 scripts)
            if not (requests_path and vehicle_pos_path and od_matrix_path):
                print(
                    "[worker] ERROR: env_type='drt' requires either manifest_path "
                    "or (drt_requests_path + drt_vehicle_positions_path + drt_od_matrix_path)",
                    file=sys.stderr,
                )
                sys.exit(1)

            print(
                f"[worker] job_id={job_id} | env_type=drt | requests={requests_path}"
            )
            model, vec_env = build_drt_model(
                requests_path=requests_path,
                vehicle_pos_path=vehicle_pos_path,
                od_matrix_path=od_matrix_path,
                n_envs=config.get("n_envs", 2),
                learning_rate=config.get("learning_rate", 3e-4),
                gamma=config.get("gamma", 0.99),
                clip_range=config.get("clip_range", 0.2),
                ent_coef=config.get("ent_coef", 0.01),
                n_steps=config.get("n_steps", 2048),
                batch_size=config.get("batch_size", 256),
                verbose=config.get("verbose", 0),
                seed=config.get("seed", 0),
            )

    else:
        # ---- SF: Sioux Falls bus routing environment (default) ----
        split = config.get("split", "train")
        scenario_paths = list_scenarios(split=split)
        if not scenario_paths:
            print(f"[worker] ERROR: No scenarios found for split='{split}'", file=sys.stderr)
            sys.exit(1)

        print(f"[worker] job_id={job_id} | {len(scenario_paths)} scenarios | split={split}")

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
