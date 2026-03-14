#!/usr/bin/env python
# ============================================================
# BP-2 through BP-6: DRT Baseline Benchmark Pack
#
# Usage (run from project root):
#   python scripts/run_drt_benchmark.py
#
# Produces under data/drt_benchmarks/:
#   greedy_r8/    episodes.csv, requests_8/requests.csv, vehicles.csv  (BP-2)
#   greedy_r80/   episodes.csv, requests_80/requests.csv, vehicles.csv (BP-3)
#   ppo_r8/       ppo_final.zip, metrics.jsonl, episodes.csv,
#                 requests_8/requests.csv, vehicles.csv               (BP-4)
#   ppo_r80/      ppo_final.zip, metrics.jsonl, episodes.csv,
#                 requests_80/requests.csv, vehicles.csv              (BP-5)
#   export_sanity/onnx_vs_sb3_report.json                            (BP-6)
#
# NOTE (ppo_r8 vs ppo_r8_smoke): BP-4/5 only evaluate a pre-trained
# checkpoint — no new training is run — so _smoke suffix is dropped
# and directories are unified as ppo_r8 / ppo_r80.
#
# NOTE (metrics.jsonl in BP-4/5): The smoke checkpoint at
# data/training/drt_smoke_test/ was produced without MetricsCallback,
# so no training curve exists. metrics.jsonl contains a provenance
# stub and the eval result from this benchmark run.
#
# PPO checkpoint: data/training/drt_smoke_test/ppo_final.zip
# ONNX model:    data/training/drt_smoke_test/model.onnx
# ============================================================

from __future__ import annotations

import json
import pathlib
import shutil
import sys

import numpy as np

# Ensure project root is on sys.path so backend imports work
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---- Paths ----
_DATA         = _ROOT / "KW_DRT" / "data"
_REQ_8        = str(_DATA / "requests_8.csv")
_REQ_80       = str(_DATA / "requests_80.csv")
_VEH          = str(_DATA / "vehicle_positions.csv")
_OD           = str(_DATA / "od_matrix.csv")

_SMOKE        = _ROOT / "data" / "training" / "drt_smoke_test"
_CHECKPOINT   = _SMOKE / "ppo_final.zip"
_ONNX_MODEL   = _SMOKE / "model.onnx"

_OUT          = _ROOT / "data" / "drt_benchmarks"


# ============================================================
# Helpers
# ============================================================

def _print_metrics(label: str, m) -> None:
    print(
        f"  {label}: serve_rate={m.serve_rate:.4f}  "
        f"cancel_rate={m.cancel_rate:.4f}  "
        f"total_reward={m.total_reward:.4f}  "
        f"steps={m.total_steps}  ticks={m.total_ticks}"
    )


# ============================================================
# BP-2: Greedy R8
# ============================================================

def bp2_greedy_r8() -> None:
    print("\n[BP-2] Greedy R8 benchmark ...")
    from backend.env.drt_env import DRTEnv
    from backend.baseline.drt_greedy import drt_greedy_policy, BASELINE_NAME
    from backend.datasets.drt_evaluator import run_drt_episode

    out_dir = _OUT / "greedy_r8"
    env = DRTEnv(_REQ_8, _VEH, _OD)
    m = run_drt_episode(
        env,
        policy_fn=drt_greedy_policy,
        policy_name=BASELINE_NAME,
        episode_id="requests_8",
        output_csv_dir=out_dir,
    )
    _print_metrics("greedy_r8", m)
    assert m.serve_rate == 0.625, f"BP-2 FAIL: expected serve_rate=0.625, got {m.serve_rate}"
    print("  BP-2 PASS")


# ============================================================
# BP-3: Greedy R80
# ============================================================

def bp3_greedy_r80() -> None:
    print("\n[BP-3] Greedy R80 benchmark ...")
    from backend.env.drt_env import DRTEnv
    from backend.baseline.drt_greedy import drt_greedy_policy, BASELINE_NAME
    from backend.datasets.drt_evaluator import run_drt_episode

    out_dir = _OUT / "greedy_r80"
    env = DRTEnv(_REQ_80, _VEH, _OD)
    m = run_drt_episode(
        env,
        policy_fn=drt_greedy_policy,
        policy_name=BASELINE_NAME,
        episode_id="requests_80",
        output_csv_dir=out_dir,
    )
    _print_metrics("greedy_r80", m)
    # No fixed target for R80 — record for future comparison
    print("  BP-3 PASS (recorded)")


# ============================================================
# BP-4: PPO R8 smoke
# ============================================================

def bp4_ppo_r8() -> None:
    print("\n[BP-4] PPO R8 smoke benchmark ...")
    if not _CHECKPOINT.exists():
        print(f"  SKIP: checkpoint not found at {_CHECKPOINT}")
        return

    from sb3_contrib import MaskablePPO
    from backend.env.drt_env import DRTEnv
    from backend.datasets.drt_evaluator import run_drt_episode, make_ppo_policy_fn

    model = MaskablePPO.load(str(_CHECKPOINT))
    policy_fn = make_ppo_policy_fn(model)

    out_dir = _OUT / "ppo_r8"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = DRTEnv(_REQ_8, _VEH, _OD)
    m = run_drt_episode(
        env,
        policy_fn=policy_fn,
        policy_name="maskable_ppo_smoke",
        episode_id="requests_8",
        output_csv_dir=out_dir,
    )
    _print_metrics("ppo_r8", m)

    # Copy checkpoint into benchmark dir for traceability
    shutil.copy2(str(_CHECKPOINT), str(out_dir / "ppo_final.zip"))

    # Write metrics.jsonl stub — pre-trained checkpoint, no training curve
    metrics_stub = {
        "note": (
            "pre-trained smoke checkpoint; training curve not recorded. "
            "Run backend/trainer/training_worker.py with MetricsCallback for full metrics."
        ),
        "checkpoint_source": str(_CHECKPOINT),
        "eval_requests": "requests_8.csv",
        "eval_serve_rate": m.serve_rate,
        "eval_total_reward": m.total_reward,
        "eval_total_steps": m.total_steps,
    }
    (out_dir / "metrics.jsonl").write_text(json.dumps(metrics_stub) + "\n", encoding="utf-8")

    print("  BP-4 PASS (recorded)")


# ============================================================
# BP-5: PPO R80 smoke
# ============================================================

def bp5_ppo_r80() -> None:
    print("\n[BP-5] PPO R80 smoke benchmark ...")
    if not _CHECKPOINT.exists():
        print(f"  SKIP: checkpoint not found at {_CHECKPOINT}")
        return

    from sb3_contrib import MaskablePPO
    from backend.env.drt_env import DRTEnv
    from backend.datasets.drt_evaluator import run_drt_episode, make_ppo_policy_fn

    model = MaskablePPO.load(str(_CHECKPOINT))
    policy_fn = make_ppo_policy_fn(model)

    out_dir = _OUT / "ppo_r80"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = DRTEnv(_REQ_80, _VEH, _OD)
    m = run_drt_episode(
        env,
        policy_fn=policy_fn,
        policy_name="maskable_ppo_smoke",
        episode_id="requests_80",
        output_csv_dir=out_dir,
    )
    _print_metrics("ppo_r80", m)

    # Copy checkpoint into benchmark dir for traceability
    shutil.copy2(str(_CHECKPOINT), str(out_dir / "ppo_final.zip"))

    # Write metrics.jsonl stub — same pre-trained checkpoint as BP-4, evaluated on R80
    metrics_stub = {
        "note": (
            "pre-trained smoke checkpoint; training curve not recorded. "
            "Run backend/trainer/training_worker.py with MetricsCallback for full metrics."
        ),
        "checkpoint_source": str(_CHECKPOINT),
        "eval_requests": "requests_80.csv",
        "eval_serve_rate": m.serve_rate,
        "eval_total_reward": m.total_reward,
        "eval_total_steps": m.total_steps,
    }
    (out_dir / "metrics.jsonl").write_text(json.dumps(metrics_stub) + "\n", encoding="utf-8")

    print("  BP-5 PASS (recorded)")


# ============================================================
# BP-6: ONNX sanity check
# ============================================================

def bp6_onnx_sanity() -> None:
    """
    For each step in a short greedy-driven episode:
      SB3:  model.predict(obs, action_masks=mask, deterministic=True)
      ONNX: masked argmax over action_logits

    Both must choose the same action on every step.
    Saves report to data/drt_benchmarks/export_sanity/onnx_vs_sb3_report.json.
    """
    print("\n[BP-6] ONNX sanity check ...")
    if not _CHECKPOINT.exists():
        print(f"  SKIP: checkpoint not found at {_CHECKPOINT}")
        return
    if not _ONNX_MODEL.exists():
        print(f"  SKIP: ONNX model not found at {_ONNX_MODEL}")
        return

    import onnxruntime as ort
    from sb3_contrib import MaskablePPO
    from backend.env.drt_env import DRTEnv

    # Load SB3 model (deterministic=True → argmax over distribution)
    model = MaskablePPO.load(str(_CHECKPOINT))

    obs_dim    = int(model.observation_space.shape[0])   # 157
    action_dim = int(model.action_space.n)               # 9

    # Load ONNX session
    sess = ort.InferenceSession(
        str(_ONNX_MODEL),
        providers=["CPUExecutionProvider"],
    )

    env = DRTEnv(_REQ_8, _VEH, _OD)
    obs, _ = env.reset(seed=0)

    mismatches: list[dict] = []
    mask_violation_count = 0
    total_steps = 0
    done = truncated = False

    while not (done or truncated):
        mask = env.action_masks()

        # SB3 prediction — deterministic=True → argmax over policy distribution
        action_sb3, _ = model.predict(
            obs, action_masks=mask, deterministic=True
        )
        action_sb3 = int(action_sb3)

        # ONNX: forward pass → masked argmax (-1e9 for invalid actions)
        logits = sess.run(
            ["action_logits"],
            {"obs": obs.reshape(1, -1).astype(np.float32)},
        )[0][0]                        # shape: (action_dim,)
        masked_logits = logits.copy()
        masked_logits[~mask] = -1e9    # mask-off invalid actions before argmax
        action_onnx = int(np.argmax(masked_logits))

        # Track mask violations (ONNX chose a mask=False action)
        if not mask[action_onnx]:
            mask_violation_count += 1

        if action_sb3 != action_onnx:
            mismatches.append({
                "step":        total_steps,
                "action_sb3":  action_sb3,
                "action_onnx": action_onnx,
                "logits":      logits.tolist(),
                "mask":        mask.tolist(),
            })

        # Advance env with the SB3 action (authoritative)
        obs, _, done, truncated, _ = env.step(action_sb3)
        total_steps += 1

    all_match = len(mismatches) == 0
    report = {
        "checkpoint_path":    str(_CHECKPOINT),
        "onnx_path":          str(_ONNX_MODEL),
        "requests_file":      _REQ_8,
        "obs_dim":            obs_dim,
        "action_dim":         action_dim,
        "deterministic":      True,
        "masking_method":     "masked_argmax (-1e9 for invalid actions)",
        "total_steps":        total_steps,
        "match_count":        total_steps - len(mismatches),
        "mismatch_count":     len(mismatches),
        "mismatch_steps":     [m["step"] for m in mismatches],
        "mask_violation_count": mask_violation_count,
        "all_match":          all_match,
        "mismatches":         mismatches,
    }

    out_dir = _OUT / "export_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "onnx_vs_sb3_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(
        f"  steps={total_steps}  match={report['match_count']}  "
        f"mismatch={report['mismatch_count']}  all_match={all_match}"
    )
    print(f"  Report saved → {report_path}")

    if not all_match:
        print("  BP-6 FAIL: ONNX and SB3 actions diverge — see report for details")
    else:
        print("  BP-6 PASS")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    _OUT.mkdir(parents=True, exist_ok=True)

    bp2_greedy_r8()
    bp3_greedy_r80()
    bp4_ppo_r8()
    bp5_ppo_r80()
    bp6_onnx_sanity()

    print("\nDone. Results in:", _OUT)
