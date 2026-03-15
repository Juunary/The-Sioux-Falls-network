# ============================================================
# PPO training setup
#
# Uses MaskablePPO("MlpPolicy") from sb3-contrib.
# Observation: flat Box(96,) — use MlpPolicy (not MultiInputPolicy).
# Action masking via action_masks() method on the env.
#
# Training env: SubprocVecEnv over scenario files (multiple episodes
# in parallel for efficiency), wrapped with VecNormalize for reward
# normalization.
#
# Evaluation env: single DummyVecEnv without VecNormalize for metric
# comparability.
# ============================================================

from __future__ import annotations

import logging
import pathlib
from typing import Optional

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


class RotatingScenarioEnv(gym.Env):
    """
    Gymnasium-compatible wrapper that cycles through a list of scenario files.
    Each reset() advances to the next scenario (staggered by env_index so
    parallel workers cover different scenarios simultaneously).
    Compatible with SubprocVecEnv and MaskablePPO's action_masks() protocol.
    """

    def __init__(self, scenario_paths: list[pathlib.Path], env_index: int = 0) -> None:
        from backend.datasets.generator import load_scenario
        from backend.env.sioux_falls_env import SiouxFallsEnv

        self._paths = scenario_paths
        self._env_index = env_index
        self._episode_count = 0

        # Load first scenario to initialise spaces (required before SubprocVecEnv forks)
        scenario = load_scenario(self._paths[self._env_index % len(self._paths)])
        self._inner = SiouxFallsEnv(scenario)
        self.observation_space = self._inner.observation_space
        self.action_space = self._inner.action_space
        self.metadata = self._inner.metadata

    def reset(self, *, seed=None, options=None):
        from backend.datasets.generator import load_scenario
        from backend.env.sioux_falls_env import SiouxFallsEnv

        # Stagger: env 0 starts at index 0, env 1 at index 1, etc.
        idx = (self._env_index + self._episode_count) % len(self._paths)
        self._episode_count += 1
        scenario = load_scenario(self._paths[idx])
        self._inner = SiouxFallsEnv(scenario)
        return self._inner.reset(seed=seed, options=options)

    def step(self, action):
        return self._inner.step(action)

    def action_masks(self) -> np.ndarray:
        return self._inner.action_masks()

    def render(self):
        pass

    def close(self):
        if self._inner is not None:
            self._inner.close()


def make_rotating_env_fn(scenario_paths: list[pathlib.Path], env_index: int):
    """Return a callable that creates a RotatingScenarioEnv for SubprocVecEnv."""
    def _init():
        return RotatingScenarioEnv(scenario_paths, env_index)
    return _init


def build_model(
    scenario_paths: list[pathlib.Path],
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    gamma: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    n_steps: int = 2048,
    batch_size: int = 256,
    verbose: int = 1,
    seed: int = 0,
):
    """
    Build a MaskablePPO model with VecNormalize-wrapped vectorised envs.

    Args:
        scenario_paths: list of scenario JSON files to rotate through.
        n_envs:         number of parallel envs (SubprocVecEnv).
                        Each worker rotates through ALL scenario_paths independently,
                        staggered by env_index so workers see different scenarios
                        in each episode.

    Returns (model, vec_env).
    """
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    # Each worker gets a different starting offset so they cover different scenarios
    env_fns = [
        make_rotating_env_fn(scenario_paths, env_index=i)
        for i in range(n_envs)
    ]

    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,   # obs already in [0,1]
        norm_reward=True,
        clip_reward=10.0,
        gamma=gamma,
    )

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        gamma=gamma,
        clip_range=clip_range,
        ent_coef=ent_coef,
        n_steps=n_steps,
        batch_size=batch_size,
        verbose=verbose,
        seed=seed,
    )

    return model, vec_env


def load_model(
    checkpoint_path: pathlib.Path,
    vec_env=None,
):
    """Load a saved MaskablePPO model from a checkpoint."""
    from sb3_contrib import MaskablePPO
    return MaskablePPO.load(checkpoint_path, env=vec_env)


# ============================================================
# DRT (Dynamic Ride-Sharing) training support
# ============================================================

class RotatingDRTEpisodeEnv(gym.Env):
    """
    Gymnasium wrapper for DRTEnv that cycles through scenarios on reset().

    Two modes:
      Single-CSV mode (backward-compatible):
        Pass requests_path, vehicle_positions_path, od_matrix_path.
        Every reset() re-creates DRTEnv with the same three files.

      Manifest mode:
        Pass manifest_path + split (+ optionally env_version, env_index).
        __init__ loads manifest entries for the split and validates
        comparison_group against ENV_CONFIGS[env_version].
        reset() round-robins through valid entries, staggered by env_index.
    """

    def __init__(
        self,
        requests_path: str = "",
        vehicle_positions_path: str = "",
        od_matrix_path: str = "",
        manifest_path: Optional[str] = None,
        split: str = "train",
        env_index: int = 0,
        env_version: str = "drt_env_v1",
    ) -> None:
        from backend.env.drt_env import DRTEnv

        self._manifest_path = manifest_path
        self._split = split
        self._env_index = env_index
        self._env_version = env_version
        self._episode_count = 0
        self._entries: list[dict] = []

        if manifest_path is not None:
            self._entries = self._load_and_filter_entries(manifest_path, split, env_version)
            # Initialise spaces from the first entry (staggered by env_index)
            first_entry = self._entries[env_index % len(self._entries)]
            req, veh, od = self._resolve_entry_paths(first_entry, manifest_path)
            self._inner = DRTEnv(req, veh, od)
        else:
            # Single-CSV mode
            self._requests_path = requests_path
            self._vehicle_positions_path = vehicle_positions_path
            self._od_matrix_path = od_matrix_path
            self._inner = DRTEnv(requests_path, vehicle_positions_path, od_matrix_path)

        self.observation_space = self._inner.observation_space
        self.action_space = self._inner.action_space
        self.metadata = getattr(self._inner, "metadata", {})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_and_filter_entries(
        manifest_path: str,
        split: str,
        env_version: str,
    ) -> list[dict]:
        from backend.datasets.manifest_utils import load_manifest_entries
        from backend.env.drt_env_config import ENV_CONFIGS

        cfg = ENV_CONFIGS.get(env_version)
        expected_cg = cfg.comparison_group if cfg is not None else None
        return load_manifest_entries(
            manifest_path, split, expected_comparison_group=expected_cg
        )

    @staticmethod
    def _resolve_entry_paths(
        entry: dict,
        manifest_path: str,
    ) -> tuple[str, str, str]:
        from backend.datasets.manifest_utils import resolve_and_validate
        manifest_dir = pathlib.Path(manifest_path).parent
        req = str(resolve_and_validate(manifest_dir, entry["requests_path"]))
        veh = str(resolve_and_validate(manifest_dir, entry["vehicle_positions_path"]))
        od  = str(resolve_and_validate(manifest_dir, entry["od_matrix_path"]))
        return req, veh, od

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        from backend.env.drt_env import DRTEnv

        if self._manifest_path is not None:
            idx = (self._env_index + self._episode_count) % len(self._entries)
            self._episode_count += 1
            req, veh, od = self._resolve_entry_paths(
                self._entries[idx], self._manifest_path
            )
            self._inner = DRTEnv(req, veh, od)
        else:
            self._inner = DRTEnv(
                self._requests_path,
                self._vehicle_positions_path,
                self._od_matrix_path,
            )
        return self._inner.reset(seed=seed, options=options)

    def step(self, action):
        return self._inner.step(action)

    def action_masks(self) -> np.ndarray:
        return self._inner.action_masks()

    def render(self):
        pass

    def close(self):
        pass


def _make_drt_env_fn(
    requests_path: str = "",
    vehicle_positions_path: str = "",
    od_matrix_path: str = "",
    manifest_path: Optional[str] = None,
    split: str = "train",
    env_index: int = 0,
    env_version: str = "drt_env_v1",
):
    """Return a callable for SubprocVecEnv factory."""
    def _init():
        return RotatingDRTEpisodeEnv(
            requests_path=requests_path,
            vehicle_positions_path=vehicle_positions_path,
            od_matrix_path=od_matrix_path,
            manifest_path=manifest_path,
            split=split,
            env_index=env_index,
            env_version=env_version,
        )
    return _init


def build_drt_model(
    requests_path: str = "",
    vehicle_pos_path: str = "",
    od_matrix_path: str = "",
    manifest_path: Optional[str] = None,
    split: str = "train",
    env_version: str = "drt_env_v1",
    n_envs: int = 2,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    n_steps: int = 2048,
    batch_size: int = 256,
    verbose: int = 1,
    seed: int = 0,
):
    """
    Build a MaskablePPO model for the DRT environment.

    Manifest mode (Stage 0.5+):
        Pass manifest_path + split.  Each worker env round-robins through
        scenarios in the given split, staggered by env_index.

    Single-CSV mode (backward-compatible):
        Pass requests_path, vehicle_pos_path, od_matrix_path.

    Uses SubprocVecEnv wrapped with VecNormalize (norm_obs=False).
    Returns (model, vec_env).
    """
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    if manifest_path is not None:
        env_fns = [
            _make_drt_env_fn(
                manifest_path=manifest_path,
                split=split,
                env_index=i,
                env_version=env_version,
            )
            for i in range(n_envs)
        ]
    else:
        env_fns = [
            _make_drt_env_fn(
                requests_path=requests_path,
                vehicle_positions_path=vehicle_pos_path,
                od_matrix_path=od_matrix_path,
            )
            for _ in range(n_envs)
        ]

    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,   # obs already in [0, 1]
        norm_reward=True,
        clip_reward=10.0,
        gamma=gamma,
    )

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        gamma=gamma,
        clip_range=clip_range,
        ent_coef=ent_coef,
        n_steps=n_steps,
        batch_size=batch_size,
        verbose=verbose,
        seed=seed,
    )

    return model, vec_env
