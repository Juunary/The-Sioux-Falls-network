# ============================================================
# SB3 training callbacks
#
# CheckpointCallback — save model every N timesteps
# MetricsCallback   — append episode metrics to a JSONL file
#                     (read by job_manager for WebSocket broadcast)
# ============================================================

from __future__ import annotations

import json
import pathlib
import time
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CheckpointCallback(BaseCallback):
    """Save model checkpoint every `save_freq` timesteps."""

    def __init__(
        self,
        save_freq: int,
        save_dir: pathlib.Path,
        name_prefix: str = "ppo",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = pathlib.Path(save_dir)
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            path = self.save_dir / f"{self.name_prefix}_{self.num_timesteps}"
            self.model.save(path)
            if self.verbose:
                print(f"[checkpoint] saved {path}")
        return True


class MetricsCallback(BaseCallback):
    """
    Append one JSONL record per episode to `output_path`.
    Record format:
    {
      "timestep": int,
      "episode": int,
      "reward": float,       # mean reward over recent episodes
      "ep_len": float,       # mean episode length (steps)
      "wall_time": float,    # wall-clock seconds since training started
    }
    """

    def __init__(
        self,
        output_path: pathlib.Path,
        log_freq: int = 1,   # write every N episodes
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.output_path = pathlib.Path(output_path)
        self.log_freq = log_freq
        self._episode_count = 0
        self._start_time = time.time()

    def _on_step(self) -> bool:
        # Check for episode completion in any env
        if "dones" in self.locals:
            for done in self.locals["dones"]:
                if done:
                    self._episode_count += 1
        return True

    def _on_rollout_end(self) -> None:
        """Write metrics after each rollout collection."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract mean reward from SB3 logger
        ep_rew_mean = None
        ep_len_mean = None
        if hasattr(self.model, "ep_info_buffer") and self.model.ep_info_buffer:
            rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            lengths = [ep["l"] for ep in self.model.ep_info_buffer]
            ep_rew_mean = float(np.mean(rewards))
            ep_len_mean = float(np.mean(lengths))

        record = {
            "timestep": self.num_timesteps,
            "episode": self._episode_count,
            "reward": ep_rew_mean,
            "ep_len": ep_len_mean,
            "wall_time": round(time.time() - self._start_time, 2),
        }

        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if self.verbose:
            print(f"[metrics] t={self.num_timesteps} ep_rew_mean={ep_rew_mean:.2f}")
