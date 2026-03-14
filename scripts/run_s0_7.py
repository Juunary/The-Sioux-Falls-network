#!/usr/bin/env python
# ============================================================
# S0-7: PPO R8 500K evaluation
#
# Evaluates data/training/drt_r8_500k/ppo_final.zip on R8.
# Uses the same eval semantics as BP-4 (run_drt_benchmark.py).
#
# Does NOT touch:
#   - data/drt_benchmarks/ppo_r8/   (BP-4 smoke output)
#   - data/training/drt_r8_500k/    (S0-6 training output)
#
# Output:
#   data/drt_benchmarks/ppo_r8_500k/
#     episodes.csv
#     requests_8/requests.csv
#     requests_8/vehicles.csv
#
# S0-7 pass criteria (v13):
#   serve_rate  >= 0.625
#   total_reward >= 5.9633   (greedy R8 actual result)
# ============================================================

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DATA       = _ROOT / "KW_DRT" / "data"
_REQ_8      = str(_DATA / "requests_8.csv")
_VEH        = str(_DATA / "vehicle_positions.csv")
_OD         = str(_DATA / "od_matrix.csv")

_CHECKPOINT = _ROOT / "data" / "training" / "drt_r8_500k" / "ppo_final.zip"
_OUT        = _ROOT / "data" / "drt_benchmarks" / "ppo_r8_500k"


def main() -> None:
    from sb3_contrib import MaskablePPO
    from backend.env.drt_env import DRTEnv
    from backend.datasets.drt_evaluator import run_drt_episode, make_ppo_policy_fn

    if not _CHECKPOINT.exists():
        print(f"ERROR: checkpoint not found at {_CHECKPOINT}")
        sys.exit(1)

    print(f"[S0-7] Loading checkpoint: {_CHECKPOINT}")
    # VecNormalize NOT applied: norm_obs=False (obs already in [0,1]),
    # model.predict() works directly on raw obs.
    model = MaskablePPO.load(str(_CHECKPOINT))
    policy_fn = make_ppo_policy_fn(model)

    _OUT.mkdir(parents=True, exist_ok=True)
    env = DRTEnv(_REQ_8, _VEH, _OD)

    m = run_drt_episode(
        env,
        policy_fn=policy_fn,
        policy_name="maskable_ppo_500k",
        episode_id="requests_8",
        output_csv_dir=_OUT,
    )

    print(
        f"  serve_rate={m.serve_rate:.4f}  cancel_rate={m.cancel_rate:.4f}  "
        f"total_reward={m.total_reward:.4f}  steps={m.total_steps}  ticks={m.total_ticks}"
    )

    # S0-7 pass criteria
    greedy_serve  = 0.625
    greedy_reward = 5.9633
    pass_serve  = m.serve_rate  >= greedy_serve
    pass_reward = m.total_reward >= greedy_reward

    print(f"\n  serve_rate  {m.serve_rate:.4f} >= {greedy_serve}  -> {'PASS' if pass_serve  else 'FAIL'}")
    print(f"  total_reward {m.total_reward:.4f} >= {greedy_reward} -> {'PASS' if pass_reward else 'FAIL'}")

    if pass_serve and pass_reward:
        print("\n  S0-7 PASS")
    else:
        print("\n  S0-7 FAIL")


if __name__ == "__main__":
    main()
