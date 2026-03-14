#!/usr/bin/env python
# ============================================================
# S0-8: PPO R80 benchmark (model trained on R8, 500K steps)
#
# Evaluates data/training/drt_r8_500k/ppo_final.zip on R80.
# Records generalization performance; this is a benchmark
# record step, not a hard pass/fail gate.
#
# Does NOT touch:
#   - data/drt_benchmarks/ppo_r80/   (BP-5 smoke output)
#   - data/drt_benchmarks/ppo_r8_500k/   (S0-7 output)
#   - data/training/drt_r8_500k/    (S0-6 training output)
#
# Output:
#   data/drt_benchmarks/ppo_r80_from_r8_500k/
#     episodes.csv
#     requests_80/requests.csv
#     requests_80/vehicles.csv
#
# Greedy R80 reference (BP-3):
#   serve_rate  = 0.1875
#   cancel_rate = 0.8125
#   total_reward = -28.8767
#   steps = 46, ticks = 73
# ============================================================

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DATA       = _ROOT / "KW_DRT" / "data"
_REQ_80     = str(_DATA / "requests_80.csv")
_VEH        = str(_DATA / "vehicle_positions.csv")
_OD         = str(_DATA / "od_matrix.csv")

_CHECKPOINT = _ROOT / "data" / "training" / "drt_r8_500k" / "ppo_final.zip"
_OUT        = _ROOT / "data" / "drt_benchmarks" / "ppo_r80_from_r8_500k"


def main() -> None:
    from sb3_contrib import MaskablePPO
    from backend.env.drt_env import DRTEnv
    from backend.datasets.drt_evaluator import run_drt_episode, make_ppo_policy_fn

    if not _CHECKPOINT.exists():
        print(f"ERROR: checkpoint not found at {_CHECKPOINT}")
        sys.exit(1)

    print(f"[S0-8] Loading checkpoint: {_CHECKPOINT}")
    # VecNormalize NOT applied: norm_obs=False (obs already in [0,1]),
    # model.predict() works directly on raw obs.
    model = MaskablePPO.load(str(_CHECKPOINT))
    policy_fn = make_ppo_policy_fn(model)

    _OUT.mkdir(parents=True, exist_ok=True)
    env = DRTEnv(_REQ_80, _VEH, _OD)

    m = run_drt_episode(
        env,
        policy_fn=policy_fn,
        policy_name="maskable_ppo_500k",
        episode_id="requests_80",
        output_csv_dir=_OUT,
    )

    print(
        f"  serve_rate={m.serve_rate:.4f}  cancel_rate={m.cancel_rate:.4f}  "
        f"total_reward={m.total_reward:.4f}  steps={m.total_steps}  ticks={m.total_ticks}"
    )

    # Greedy R80 reference (BP-3)
    greedy_serve  = 0.1875
    greedy_reward = -28.8767

    print(f"\n  [S0-8] Greedy R80 reference:")
    print(f"    serve_rate  = {greedy_serve}")
    print(f"    total_reward = {greedy_reward}")

    print(f"\n  [S0-8] PPO (R8-trained, 500K) on R80:")
    print(f"    serve_rate  = {m.serve_rate:.4f}  (greedy: {greedy_serve}) "
          f"-> {'>= greedy' if m.serve_rate >= greedy_serve else '< greedy'}")
    print(f"    total_reward = {m.total_reward:.4f}  (greedy: {greedy_reward}) "
          f"-> {'>= greedy' if m.total_reward >= greedy_reward else '< greedy'}")

    print("\n  S0-8 recorded (benchmark record - no hard pass/fail gate)")


if __name__ == "__main__":
    main()
