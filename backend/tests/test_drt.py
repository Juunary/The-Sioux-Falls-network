# ============================================================
# BP-1: DRT environment tests
#
# Verifies:
# 1. reset/step determinism  — same seed → identical trajectory
# 2. obs ∈ [0, 1]           — full episode observation range
# 3. action_masks validity  — REJECT always True; greedy never picks masked-off action
# 4. greedy R8 serve_rate   — 5/8 = 0.625 reproduction
# ============================================================

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from backend.env.drt_env import DRTEnv
from backend.env.drt_config import OBS_DIM_DRT, POSSIBLE_ACTION
from backend.baseline.drt_greedy import drt_greedy_policy

# ---- Data paths (relative to this file: ../../KW_DRT/data) ----
_DATA = pathlib.Path(__file__).resolve().parents[2] / "KW_DRT" / "data"
_REQ_8 = str(_DATA / "requests_8.csv")
_VEH   = str(_DATA / "vehicle_positions.csv")
_OD    = str(_DATA / "od_matrix.csv")

_REJECT_ACTION = POSSIBLE_ACTION - 1   # 8


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def env_r8():
    return DRTEnv(_REQ_8, _VEH, _OD)


# ============================================================
# BP-1-a: reset/step determinism
# ============================================================

def test_determinism():
    """
    Two independent DRTEnv instances reset with seed=0 and driven by
    the same greedy policy must produce byte-identical observations,
    equal rewards, and the same done/truncated flags at every step.
    """
    env_a = DRTEnv(_REQ_8, _VEH, _OD)
    env_b = DRTEnv(_REQ_8, _VEH, _OD)

    obs_a, _ = env_a.reset(seed=0)
    obs_b, _ = env_b.reset(seed=0)

    np.testing.assert_array_equal(obs_a, obs_b, err_msg="Initial obs mismatch")

    total_a = total_b = 0.0
    for step_i in range(200):
        mask_a = env_a.action_masks()
        mask_b = env_b.action_masks()
        np.testing.assert_array_equal(
            mask_a, mask_b, err_msg=f"mask mismatch at step {step_i}"
        )

        action = drt_greedy_policy(obs_a, mask_a)
        assert action == drt_greedy_policy(obs_b, mask_b), (
            f"action mismatch at step {step_i}"
        )

        obs_a, r_a, done_a, trunc_a, _ = env_a.step(action)
        obs_b, r_b, done_b, trunc_b, _ = env_b.step(action)

        np.testing.assert_array_almost_equal(
            obs_a, obs_b, decimal=6,
            err_msg=f"obs mismatch at step {step_i}",
        )
        assert abs(r_a - r_b) < 1e-6, f"reward mismatch at step {step_i}"
        assert done_a == done_b and trunc_a == trunc_b, (
            f"done/trunc mismatch at step {step_i}"
        )

        total_a += r_a
        total_b += r_b

        if done_a or trunc_a:
            break

    assert abs(total_a - total_b) < 1e-5, (
        f"Cumulative reward mismatch: {total_a} vs {total_b}"
    )


# ============================================================
# BP-1-b: obs ∈ [0, 1] throughout a full episode
# ============================================================

def test_obs_range(env_r8):
    """
    Every observation returned by reset() and step() must be
    shape (157,) with all values in [0.0, 1.0].
    """
    obs, _ = env_r8.reset(seed=0)
    _assert_obs_valid(obs, step=-1)

    done = truncated = False
    step = 0
    while not (done or truncated):
        mask   = env_r8.action_masks()
        action = drt_greedy_policy(obs, mask)
        obs, _, done, truncated, _ = env_r8.step(action)
        _assert_obs_valid(obs, step)
        step += 1


def _assert_obs_valid(obs: np.ndarray, step: int) -> None:
    assert obs.shape == (OBS_DIM_DRT,), (
        f"Wrong obs shape at step {step}: {obs.shape}"
    )
    assert np.all(obs >= 0.0), (
        f"obs < 0 at step {step}: min={obs.min():.6f}"
    )
    assert np.all(obs <= 1.0), (
        f"obs > 1 at step {step}: max={obs.max():.6f}"
    )


# ============================================================
# BP-1-c: action_masks validity
# ============================================================

def test_action_masks(env_r8):
    """
    For every step of a full greedy episode:
    - REJECT (slot 8) is always True
    - At least one action is valid (mask.any())
    - The greedy policy never selects a mask-False action
    """
    obs, _ = env_r8.reset(seed=0)
    done = truncated = False
    step = 0
    while not (done or truncated):
        mask = env_r8.action_masks()

        assert mask[_REJECT_ACTION], (
            f"REJECT not in mask at step {step}"
        )
        assert mask.any(), f"No valid action at step {step}"

        action = drt_greedy_policy(obs, mask)
        assert mask[action], (
            f"Greedy chose masked-off action {action} at step {step}"
        )

        obs, _, done, truncated, _ = env_r8.step(action)
        step += 1


# ============================================================
# BP-1-d: greedy R8 serve_rate reproduction
# ============================================================

def test_greedy_r8_serve_rate(env_r8):
    """
    Greedy policy on requests_8.csv must reproduce serve_rate == 0.625 (5/8).
    This is the known result from the existing data/drt_eval_test/episodes.csv.
    """
    from backend.datasets.drt_evaluator import run_drt_episode

    metrics = run_drt_episode(
        env_r8,
        policy_fn=drt_greedy_policy,
        policy_name="drt_greedy_v1",
        episode_id="greedy_r8_check",
    )

    assert metrics.total_requests == 8, (
        f"Expected 8 total requests, got {metrics.total_requests}"
    )
    assert metrics.serve_rate == pytest.approx(0.625, abs=1e-4), (
        f"serve_rate={metrics.serve_rate:.4f}, expected 0.625"
    )
