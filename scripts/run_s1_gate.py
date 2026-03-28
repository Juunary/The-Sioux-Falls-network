#!/usr/bin/env python
# ============================================================
# Stage 1 / 4.1 Gate Runner — S1-F1 through S1-F4
#
# Seven functional blocks:
#   1. Preflight   — provider selection, SDK, API key, manifest, imports
#   2. R8 eval     — 1-shot test_r8_original via llm_teacher
#   3. R80 eval    — full test split (11 scenarios) via llm_teacher
#   4. Consistency — same obs/mask 10× with temp=0, ≥90% match
#   5. Artifacts   — trace JSONL, cache DB, experiments DB check
#   6. Gates       — S1-F1..F4 pass/fail calculation
#   7. Reference   — greedy baseline comparison from S0.5-5
#
# Usage (from project root):
#   python scripts/run_s1_gate.py                     # interactive provider prompt
#   python scripts/run_s1_gate.py --provider anthropic # explicit provider
#   python scripts/run_s1_gate.py --provider gemini    # explicit provider
#   python scripts/run_s1_gate.py --provider gemini --model gemini-2.5-flash
#
# If the SDK or API key is missing/invalid, the script prints
# a preflight-only report and exits with code 2. No mock results.
# ============================================================

from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_MANIFEST = _ROOT / "KW_DRT" / "data" / "scenarios" / "manifest.jsonl"
_ENV_VER = "drt_env_v1"
_SPLIT = "test"
_PROMPT_VERSION = "v1"
_TEMPERATURE = 0.0
_MAX_TOKENS = 64
_CONSISTENCY_REPS = 10
_CONSISTENCY_THRESHOLD = 0.90
_VALID_ACTION_THRESHOLD = 0.95

# Default model per provider
_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-2.5-flash",
}


# ============================================================
# Provider selection (interactive / CLI)
# ============================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 / 4.1 Gate Runner")
    parser.add_argument(
        "--provider", choices=["anthropic", "gemini"], default=None,
        help="LLM provider. If not set, prompts interactively (TTY only).",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model ID override (default: provider-specific default).",
    )
    parser.add_argument(
        "--skip-key-verify", action="store_true",
        help="Skip the live API key verification call (env-var check only).",
    )
    return parser.parse_args()


def _select_provider_interactive() -> str:
    """Prompt user to select a provider.  Requires a TTY."""
    if not sys.stdin.isatty():
        print("ERROR: --provider is required in non-interactive mode.")
        print("  Usage: python scripts/run_s1_gate.py --provider anthropic")
        print("     or: python scripts/run_s1_gate.py --provider gemini")
        sys.exit(2)

    print()
    print("Select LLM provider:")
    print("  [1] anthropic  (Claude models)")
    print("  [2] gemini     (Google Gemini models)")
    print()
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return "anthropic"
        elif choice == "2":
            return "gemini"
        print("  Invalid choice. Enter 1 or 2.")


# ============================================================
# Block 1: Preflight
# ============================================================

def _preflight(provider: str, model_id: str, skip_key_verify: bool) -> dict:
    """Check all prerequisites.  Returns dict with pass/fail for each."""
    results = {"llm_provider": provider, "model_id": model_id}

    # 1a. manifest.jsonl exists
    results["manifest_exists"] = _MANIFEST.exists()

    # 1b. DRTEnv importable
    try:
        from backend.env.drt_env import DRTEnv  # noqa: F401
        results["drt_env_importable"] = True
    except Exception as exc:
        results["drt_env_importable"] = False
        results["drt_env_error"] = str(exc)

    # 1c. backend.llm modules importable (obs_to_text, text_to_action, cache)
    try:
        from backend.llm.obs_to_text import obs_to_text  # noqa: F401
        from backend.llm.text_to_action import text_to_action  # noqa: F401
        from backend.llm.cache import cache_lookup, cache_store  # noqa: F401
        from backend.llm.policy_context import PolicyContext  # noqa: F401
        results["llm_modules_importable"] = True
    except Exception as exc:
        results["llm_modules_importable"] = False
        results["llm_modules_error"] = str(exc)

    # 1d. Provider SDK installed
    from backend.llm.config import check_provider_sdk
    sdk_ok, sdk_msg = check_provider_sdk(provider)
    results["provider_sdk_installed"] = sdk_ok
    results["provider_sdk_message"] = sdk_msg

    # 1e. API key set (env var non-empty)
    from backend.llm.config import preflight_check_key
    key_ok, key_msg = preflight_check_key(provider)
    results["api_key_set"] = key_ok
    results["api_key_message"] = key_msg

    # 1f. Live API key verification (minimal cost call)
    if sdk_ok and key_ok and not skip_key_verify:
        from backend.llm.config import resolve_provider_api_key, verify_api_key
        api_key = resolve_provider_api_key(provider)
        verify_ok, verify_msg = verify_api_key(provider, api_key)
        results["api_key_verified"] = verify_ok
        results["api_key_verify_message"] = verify_msg
    elif skip_key_verify:
        results["api_key_verified"] = True  # assume valid if skipped
        results["api_key_verify_message"] = "Skipped (--skip-key-verify)"
    else:
        results["api_key_verified"] = False
        results["api_key_verify_message"] = "Cannot verify: SDK or key not available"

    # 1g. env_version in ENV_CONFIGS
    try:
        from backend.env.drt_env_config import ENV_CONFIGS
        results["env_config_valid"] = _ENV_VER in ENV_CONFIGS
    except Exception:
        results["env_config_valid"] = False

    # 1h. manifest has test entries
    if results["manifest_exists"]:
        try:
            from backend.datasets.manifest_utils import load_manifest_entries
            from backend.env.drt_env_config import ENV_CONFIGS
            cfg = ENV_CONFIGS[_ENV_VER]
            entries = load_manifest_entries(
                _MANIFEST, _SPLIT, expected_comparison_group=cfg.comparison_group
            )
            results["test_entries_count"] = len(entries)
        except Exception as exc:
            results["test_entries_count"] = 0
            results["manifest_error"] = str(exc)
    else:
        results["test_entries_count"] = 0

    # Overall: can we proceed?
    results["can_run"] = all([
        results["manifest_exists"],
        results["drt_env_importable"],
        results["llm_modules_importable"],
        results["provider_sdk_installed"],
        results["api_key_set"],
        results["api_key_verified"],
        results["env_config_valid"],
        results["test_entries_count"] > 0,
    ])

    return results


def _print_preflight(pf: dict) -> None:
    print("=" * 60)
    print("[PREFLIGHT] Stage 1 / 4.1 Gate Runner")
    print(f"  Provider: {pf.get('llm_provider', '???')}")
    print(f"  Model:    {pf.get('model_id', '???')}")
    print("=" * 60)
    skip_keys = {"can_run", "llm_provider", "model_id"}
    for k, v in pf.items():
        if k in skip_keys:
            continue
        if isinstance(v, bool):
            icon = "OK" if v else "FAIL"
            print(f"  [{icon:>4}] {k}")
        else:
            print(f"  [    ] {k} = {v}")
    print()
    if pf["can_run"]:
        print("  => All preflight checks PASSED. Proceeding to evaluation.")
    else:
        print("  => Preflight FAILED. Cannot run LLM evaluation.")
        provider = pf.get("llm_provider", "???")
        if not pf.get("provider_sdk_installed"):
            if provider == "anthropic":
                print("     SDK not installed: pip install -r backend/requirements-llm.txt")
            else:
                print("     SDK not installed: pip install google-generativeai")
        if not pf.get("api_key_set"):
            env_vars = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}
            print(f"     API key not set: export {env_vars.get(provider, '???')}=<your-key>")
        if pf.get("api_key_set") and not pf.get("api_key_verified"):
            print(f"     API key INVALID: {pf.get('api_key_verify_message', 'unknown error')}")
    print()


# ============================================================
# Block 2: R8 single-shot evaluation
# ============================================================

def _run_r8_eval(ctx, output_dir: pathlib.Path) -> dict:
    """Run llm_teacher on test_r8_original only.  Returns metrics dict."""
    from backend.env.drt_env import DRTEnv
    from backend.datasets.manifest_utils import load_manifest_entries, resolve_and_validate
    from backend.env.drt_env_config import ENV_CONFIGS
    from backend.llm.teacher import make_llm_teacher_policy_fn
    from backend.datasets.drt_evaluator import run_drt_episode

    cfg = ENV_CONFIGS[_ENV_VER]
    manifest_dir = _MANIFEST.parent
    entries = load_manifest_entries(
        _MANIFEST, _SPLIT, expected_comparison_group=cfg.comparison_group
    )

    r8_entry = next((e for e in entries if e["scenario_id"] == "test_r8_original"), None)
    if r8_entry is None:
        return {"error": "test_r8_original not found in manifest"}

    ctx.scenario_id = r8_entry["scenario_id"]
    ctx.episode_id = f"llm_teacher_{r8_entry['scenario_id']}"
    ctx.reset_episode_stats()

    policy_fn = make_llm_teacher_policy_fn(ctx)
    req = str(resolve_and_validate(manifest_dir, r8_entry["requests_path"]))
    veh = str(resolve_and_validate(manifest_dir, r8_entry["vehicle_positions_path"]))
    od = str(resolve_and_validate(manifest_dir, r8_entry["od_matrix_path"]))

    env = DRTEnv(req, veh, od)
    csv_dir = output_dir / "r8"
    metrics = run_drt_episode(
        env, policy_fn, "llm_teacher",
        episode_id="test_r8_original",
        output_csv_dir=csv_dir,
    )
    llm_stats = ctx.compute_llm_metrics()

    return {
        "metrics": metrics.to_dict(),
        "llm_stats": llm_stats,
        "completed": True,
    }


# ============================================================
# Block 3: R80 full test split evaluation
# ============================================================

def _run_full_test_eval(provider: str, model_id: str, output_dir: pathlib.Path) -> dict:
    """Run llm_teacher on full test split (11 scenarios)."""
    from backend.llm.policy_context import PolicyContext
    from backend.datasets.drt_evaluator import evaluate_drt_teacher_on_manifest
    from backend.experiments.tracker import (
        create_experiment, record_drt_metrics, record_drt_llm_stats,
        aggregate_drt_metrics, aggregate_llm_stats,
    )
    from backend.env.drt_env_config import ENV_CONFIGS
    from datetime import datetime, timezone

    ctx = PolicyContext(
        env_version=_ENV_VER,
        model_id=model_id,
        prompt_version=_PROMPT_VERSION,
        llm_provider=provider,
    )

    csv_dir = output_dir / "test_split"
    metrics_list, llm_stats_list = evaluate_drt_teacher_on_manifest(
        manifest_path=str(_MANIFEST),
        split=_SPLIT,
        ctx=ctx,
        policy_name="llm_teacher",
        output_csv_dir=csv_dir,
    )

    # Record in experiments DB
    cfg = ENV_CONFIGS[_ENV_VER]
    exp_id = f"s1_gate_llm_teacher_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    create_experiment(
        experiment_id=exp_id,
        policy=f"llm_teacher_{provider}_{model_id}_{_PROMPT_VERSION}",
        split=_SPLIT,
        config={
            "manifest_path": str(_MANIFEST),
            "split": _SPLIT,
            "env_version": _ENV_VER,
            "llm_provider": provider,
            "model_id": model_id,
            "prompt_version": _PROMPT_VERSION,
            "gate_run": True,
        },
        domain="drt",
        env_version=_ENV_VER,
        comparison_group=cfg.comparison_group,
    )
    record_drt_metrics(exp_id, metrics_list)
    for m, stats in zip(metrics_list, llm_stats_list):
        record_drt_llm_stats(exp_id, m.episode_id, stats)

    agg = aggregate_drt_metrics(exp_id)
    llm_agg = aggregate_llm_stats(exp_id)

    # Collect per-episode summaries
    episodes = []
    for m, stats in zip(metrics_list, llm_stats_list):
        episodes.append({
            "episode_id": m.episode_id,
            "serve_rate": m.serve_rate,
            "cancel_rate": m.cancel_rate,
            "total_reward": m.total_reward,
            "total_steps": m.total_steps,
            "invalid_action_rate": stats.get("invalid_action_rate"),
            "api_cost_per_episode": stats.get("api_cost_per_episode"),
            "episode_latency_p50": stats.get("episode_latency_p50"),
        })

    return {
        "experiment_id": exp_id,
        "episode_count": len(metrics_list),
        "episodes": episodes,
        "aggregate": agg,
        "llm_aggregate": llm_agg,
        "all_completed": len(metrics_list) == 11,
    }


# ============================================================
# Block 4: Consistency test (same obs/mask, 10 reps, temp=0)
# ============================================================

def _run_consistency_test(provider: str, model_id: str) -> dict:
    """Call teacher 10x on the same obs/mask with temp=0.

    Uses a single R8 obs (step 0) to test output stability.
    """
    from backend.env.drt_env import DRTEnv
    from backend.datasets.manifest_utils import load_manifest_entries, resolve_and_validate
    from backend.env.drt_env_config import ENV_CONFIGS
    from backend.llm.policy_context import PolicyContext
    from backend.llm.teacher import make_llm_teacher_policy_fn
    import numpy as np

    cfg = ENV_CONFIGS[_ENV_VER]
    manifest_dir = _MANIFEST.parent
    entries = load_manifest_entries(
        _MANIFEST, _SPLIT, expected_comparison_group=cfg.comparison_group
    )
    r8_entry = next((e for e in entries if e["scenario_id"] == "test_r8_original"), None)
    if r8_entry is None:
        return {"error": "test_r8_original not found", "match_rate": 0.0}

    req = str(resolve_and_validate(manifest_dir, r8_entry["requests_path"]))
    veh = str(resolve_and_validate(manifest_dir, r8_entry["vehicle_positions_path"]))
    od = str(resolve_and_validate(manifest_dir, r8_entry["od_matrix_path"]))

    env = DRTEnv(req, veh, od)
    obs, _ = env.reset()
    mask = env.action_masks()

    actions = []
    for i in range(_CONSISTENCY_REPS):
        ctx = PolicyContext(
            env_version=_ENV_VER,
            model_id=model_id,
            prompt_version=_PROMPT_VERSION,
            llm_provider=provider,
            scenario_id="consistency_test",
            episode_id=f"consistency_rep_{i}",
        )
        # Each rep gets a fresh policy_fn (fresh client, no cross-contamination)
        policy_fn = make_llm_teacher_policy_fn(ctx)
        action = policy_fn(obs, mask)
        actions.append(action)

    from collections import Counter
    counts = Counter(actions)
    most_common_action, most_common_count = counts.most_common(1)[0]
    match_rate = most_common_count / _CONSISTENCY_REPS

    return {
        "reps": _CONSISTENCY_REPS,
        "actions": actions,
        "most_common_action": most_common_action,
        "most_common_count": most_common_count,
        "match_rate": round(match_rate, 2),
        "passed": match_rate >= _CONSISTENCY_THRESHOLD,
    }


# ============================================================
# Block 5: Artifact verification (trace JSONL, cache DB, exp DB)
# ============================================================

def _verify_artifacts(exp_id: str) -> dict:
    """Check that trace files, cache DB, and experiment DB have data."""
    results = {}

    # 5a. Trace JSONL files exist
    trace_dir = _ROOT / "data" / "llm_traces"
    if trace_dir.exists():
        trace_files = list(trace_dir.glob("llm_teacher_*.jsonl"))
        results["trace_dir_exists"] = True
        results["trace_file_count"] = len(trace_files)
        # Check most recent trace has valid JSON lines
        if trace_files:
            latest = max(trace_files, key=lambda f: f.stat().st_mtime)
            try:
                lines = [l for l in latest.read_text(encoding="utf-8").splitlines() if l.strip()]
                entries = [json.loads(l) for l in lines]
                results["trace_latest_entries"] = len(entries)
                if entries:
                    sample = entries[0]
                    required_keys = {
                        "trace_id", "policy_type", "llm_provider", "model_id",
                        "prompt_version", "env_version", "parsed_action",
                        "parse_success", "mask_compliant", "cache_hit",
                    }
                    results["trace_schema_valid"] = required_keys.issubset(sample.keys())
                else:
                    results["trace_schema_valid"] = False
            except Exception as exc:
                results["trace_parse_error"] = str(exc)
                results["trace_schema_valid"] = False
    else:
        results["trace_dir_exists"] = False
        results["trace_file_count"] = 0

    # 5b. Cache DB
    cache_db = _ROOT / "data" / "llm_cache.db"
    if cache_db.exists():
        results["cache_db_exists"] = True
        try:
            conn = sqlite3.connect(cache_db)
            row = conn.execute("SELECT COUNT(*) FROM llm_action_cache").fetchone()
            results["cache_entries"] = row[0] if row else 0
            conn.close()
        except Exception as exc:
            results["cache_entries"] = 0
            results["cache_db_error"] = str(exc)
    else:
        results["cache_db_exists"] = False
        results["cache_entries"] = 0

    # 5c. Experiments DB — check v13 LLM columns are populated
    exp_db = _ROOT / "data" / "experiments.db"
    if exp_db.exists() and exp_id:
        try:
            conn = sqlite3.connect(exp_db)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT episode_id, llm_calls, invalid_action_rate,
                          api_cost_per_episode, episode_latency_p50
                   FROM drt_episode_metrics
                   WHERE experiment_id = ? AND llm_calls IS NOT NULL""",
                (exp_id,),
            ).fetchall()
            results["db_llm_rows"] = len(rows)
            if rows:
                sample = dict(rows[0])
                results["db_v13_cols_populated"] = all(
                    sample.get(c) is not None
                    for c in ["invalid_action_rate", "api_cost_per_episode",
                              "episode_latency_p50"]
                )
            else:
                results["db_v13_cols_populated"] = False
            conn.close()
        except Exception as exc:
            results["db_llm_rows"] = 0
            results["db_v13_cols_populated"] = False
            results["db_error"] = str(exc)
    else:
        results["db_llm_rows"] = 0
        results["db_v13_cols_populated"] = False

    return results


# ============================================================
# Block 6: Gate calculation — S1-F1..F4
# ============================================================

def _compute_gates(
    full_eval: dict,
    consistency: dict,
    artifacts: dict,
) -> dict:
    """Compute pass/fail for each gate.

    S1-F1: Valid action rate ≥ 95%  (from full eval invalid_action_rate)
    S1-F2: Episode completion 100%  (11/11 completed)
    S1-F3: Trace consistency ≥ 90%  (10 reps, temp=0)
    S1-F4: API cost per episode recorded (non-NULL in DB)
    """
    gates = {}

    # S1-F1: valid action rate ≥ 95%
    llm_agg = full_eval.get("llm_aggregate", {})
    avg_invalid_rate = llm_agg.get("avg_invalid_action_rate")
    if avg_invalid_rate is not None:
        valid_rate = 1.0 - avg_invalid_rate
        gates["S1-F1"] = {
            "name": "Valid action rate >= 95%",
            "value": round(valid_rate, 4),
            "threshold": _VALID_ACTION_THRESHOLD,
            "passed": valid_rate >= _VALID_ACTION_THRESHOLD,
        }
    else:
        gates["S1-F1"] = {
            "name": "Valid action rate >= 95%",
            "value": None,
            "threshold": _VALID_ACTION_THRESHOLD,
            "passed": False,
            "note": "avg_invalid_action_rate not available",
        }

    # S1-F2: Episode completion = 100%
    gates["S1-F2"] = {
        "name": "Episode completion 100% (R8 + R80)",
        "value": f"{full_eval.get('episode_count', 0)}/11",
        "passed": full_eval.get("all_completed", False),
    }

    # S1-F3: Trace consistency ≥ 90%
    gates["S1-F3"] = {
        "name": f"Trace consistency >= {_CONSISTENCY_THRESHOLD*100:.0f}%",
        "value": consistency.get("match_rate"),
        "threshold": _CONSISTENCY_THRESHOLD,
        "passed": consistency.get("passed", False),
    }

    # S1-F4: API cost per episode recorded
    cost = llm_agg.get("avg_api_cost_per_episode")
    gates["S1-F4"] = {
        "name": "API cost per episode recorded",
        "value": cost,
        "passed": cost is not None,
    }

    # Overall
    gates["all_passed"] = all(g["passed"] for g in gates.values() if isinstance(g, dict))

    return gates


# ============================================================
# Block 7: Reference metrics (greedy baseline from S0.5-5)
# ============================================================

def _load_greedy_reference() -> dict:
    """Load greedy baseline results from S0.5-5 output."""
    episodes_csv = _ROOT / "data" / "drt_benchmarks" / "greedy_test_split" / "episodes.csv"
    if not episodes_csv.exists():
        return {"available": False, "note": f"Not found: {episodes_csv}"}

    import csv
    with open(episodes_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"available": False, "note": "Empty CSV"}

    serve_rates = [float(r["serve_rate"]) for r in rows]
    rewards = [float(r["total_reward"]) for r in rows]

    import statistics
    return {
        "available": True,
        "episode_count": len(rows),
        "avg_serve_rate": round(statistics.mean(serve_rates), 4),
        "avg_total_reward": round(statistics.mean(rewards), 4),
        "min_serve_rate": round(min(serve_rates), 4),
        "max_serve_rate": round(max(serve_rates), 4),
    }


# ============================================================
# Report printer
# ============================================================

def _print_section(title: str, data: dict, indent: int = 2) -> None:
    prefix = " " * indent
    print(f"\n{'=' * 60}")
    print(f"[{title}]")
    print("=" * 60)
    for k, v in data.items():
        if isinstance(v, dict):
            print(f"{prefix}{k}:")
            for k2, v2 in v.items():
                print(f"{prefix}  {k2}: {v2}")
        elif isinstance(v, list) and len(v) > 10:
            print(f"{prefix}{k}: [{v[0]}, {v[1]}, ... ({len(v)} items)]")
        else:
            print(f"{prefix}{k}: {v}")


def _print_gates(gates: dict) -> None:
    print(f"\n{'=' * 60}")
    print("[GATES] Stage 1 / 4.1 -- S1-F1..F4")
    print("=" * 60)
    for key, g in gates.items():
        if key == "all_passed":
            continue
        if isinstance(g, dict):
            icon = "PASS" if g["passed"] else "FAIL"
            val = g.get("value", "N/A")
            print(f"  [{icon:>4}] {key}: {g['name']}")
            print(f"         value={val}  threshold={g.get('threshold', 'N/A')}")
            if "note" in g:
                print(f"         note: {g['note']}")
    print()
    if gates.get("all_passed"):
        print("  => ALL GATES PASSED.  Stage 1 / 4.1 exit criteria met.")
    else:
        print("  => SOME GATES FAILED. Review above.")
    print()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = _parse_args()

    # ---- Provider selection ----
    if args.provider:
        provider = args.provider
    else:
        provider = _select_provider_interactive()

    model_id = args.model or _DEFAULT_MODELS.get(provider, "claude-sonnet-4-6")

    print()
    print(f"  Selected provider: {provider}")
    print(f"  Selected model:    {model_id}")
    print()

    # ---- Preflight ----
    pf = _preflight(provider, model_id, args.skip_key_verify)
    _print_preflight(pf)

    # Block 7 (reference) is always available, run it first
    greedy_ref = _load_greedy_reference()
    _print_section("REFERENCE: Greedy Baseline (S0.5-5)", greedy_ref)

    if not pf["can_run"]:
        # Preflight-only report
        print()
        print("=" * 60)
        print("[PREFLIGHT-ONLY REPORT]")
        print("=" * 60)
        print("  Cannot run LLM evaluation due to missing prerequisites.")
        print("  Blocks 2-6 (R8 eval, R80 eval, consistency, artifacts, gates)")
        print("  are SKIPPED.")
        print()
        print("  To proceed:")
        if not pf.get("provider_sdk_installed"):
            if provider == "anthropic":
                print("    1. pip install -r backend/requirements-llm.txt")
            else:
                print("    1. pip install google-generativeai")
        if not pf.get("api_key_set"):
            env_vars = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}
            print(f"    2. export {env_vars.get(provider, '???')}=<your-key>")
        if pf.get("api_key_set") and not pf.get("api_key_verified"):
            print(f"    API key verification FAILED: {pf.get('api_key_verify_message')}")
            print("    Provide a valid API key and retry.")
        print()
        print("  Gate status: BLOCKED (preflight failed)")
        print()
        sys.exit(2)

    # ---- Blocks 2-6 require SDK + valid key ----
    from backend.llm.policy_context import PolicyContext

    output_dir = _ROOT / "data" / "s1_gate_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Block 2: R8 eval
    print("\n[BLOCK 2] Running R8 single-shot evaluation...")
    ctx_r8 = PolicyContext(
        env_version=_ENV_VER,
        model_id=model_id,
        prompt_version=_PROMPT_VERSION,
        llm_provider=provider,
    )
    t0 = time.time()
    r8_result = _run_r8_eval(ctx_r8, output_dir)
    r8_time = time.time() - t0
    r8_result["wall_time_sec"] = round(r8_time, 2)
    _print_section("R8 EVAL (test_r8_original)", r8_result)

    # Block 3: Full test split eval
    print("\n[BLOCK 3] Running full test split evaluation (11 scenarios)...")
    t0 = time.time()
    full_result = _run_full_test_eval(provider, model_id, output_dir)
    full_time = time.time() - t0
    full_result["wall_time_sec"] = round(full_time, 2)
    _print_section("FULL TEST SPLIT EVAL", {
        "experiment_id": full_result.get("experiment_id"),
        "episode_count": full_result.get("episode_count"),
        "all_completed": full_result.get("all_completed"),
        "wall_time_sec": full_result.get("wall_time_sec"),
        "aggregate": full_result.get("aggregate", {}),
        "llm_aggregate": full_result.get("llm_aggregate", {}),
    })
    # Print per-episode summary table
    print("\n  Per-episode breakdown:")
    print(f"  {'episode_id':<25} {'serve_rate':>10} {'reward':>10} "
          f"{'steps':>6} {'inv_rate':>8} {'cost_usd':>10} {'lat_p50':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*6} {'-'*8} {'-'*10} {'-'*8}")
    for ep in full_result.get("episodes", []):
        inv = ep.get("invalid_action_rate")
        cost = ep.get("api_cost_per_episode")
        lat = ep.get("episode_latency_p50")
        print(f"  {ep['episode_id']:<25} {ep['serve_rate']:>10.4f} {ep['total_reward']:>10.4f} "
              f"{ep['total_steps']:>6} "
              f"{inv if inv is not None else 'N/A':>8} "
              f"{f'${cost:.6f}' if cost is not None else 'N/A':>10} "
              f"{f'{lat:.1f}ms' if lat is not None else 'N/A':>8}")

    # Block 4: Consistency test
    print("\n[BLOCK 4] Running consistency test (10 reps, temp=0)...")
    t0 = time.time()
    consistency_result = _run_consistency_test(provider, model_id)
    consistency_result["wall_time_sec"] = round(time.time() - t0, 2)
    _print_section("CONSISTENCY TEST", consistency_result)

    # Block 5: Artifact verification
    print("\n[BLOCK 5] Verifying artifacts...")
    artifact_result = _verify_artifacts(full_result.get("experiment_id", ""))
    _print_section("ARTIFACT VERIFICATION", artifact_result)

    # Block 6: Gate calculation
    gates = _compute_gates(full_result, consistency_result, artifact_result)
    _print_gates(gates)

    # ---- Final summary JSON ----
    summary = {
        "preflight": pf,
        "r8_eval": r8_result,
        "full_eval": {
            "experiment_id": full_result.get("experiment_id"),
            "episode_count": full_result.get("episode_count"),
            "all_completed": full_result.get("all_completed"),
        },
        "consistency": consistency_result,
        "artifacts": artifact_result,
        "gates": gates,
        "greedy_reference": greedy_ref,
    }
    summary_path = output_dir / "s1_gate_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary written to: {summary_path}")

    # Exit code: 0 if all gates pass, 1 if any fail
    sys.exit(0 if gates.get("all_passed") else 1)


if __name__ == "__main__":
    main()
