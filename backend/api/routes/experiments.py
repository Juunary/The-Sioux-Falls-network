# ============================================================
# Experiments API routes
# POST /api/experiments/evaluate      — run SF baseline evaluation
# POST /api/experiments/evaluate/drt  — run DRT greedy evaluation
# GET  /api/experiments/              — list experiments
# GET  /api/experiments/{id}          — get experiment details + metrics
# GET  /api/experiments/{id}/aggregate — get aggregate statistics (SF or DRT)
# ============================================================

from __future__ import annotations

import pathlib
import threading
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from backend.baseline.demand_aware import BASELINE_NAME
from backend.datasets.evaluator import evaluate_baseline, greedy_policy_fn
from backend.experiments.tracker import (
    aggregate_drt_metrics,
    aggregate_metrics,
    create_experiment,
    get_experiment_metrics,
    list_experiments,
    record_drt_metrics,
    record_metrics,
)

router = APIRouter()

# Maximum number of concurrently running background evaluations.
_MAX_CONCURRENT_EVALS: int = 3
_active_evals: int = 0
_eval_lock = threading.Lock()

def _validate_drt_path(p: str) -> str:
    """Raise 400 if path is not under KW_DRT/data/. Normalises separators."""
    # Normalise to forward slashes before comparison (cross-platform)
    normalised = p.replace("\\", "/")
    if not normalised.startswith("KW_DRT/data/"):
        raise HTTPException(
            status_code=400,
            detail=f"Path must be under KW_DRT/data/: {p}",
        )
    return p


# ---- Request / Response schemas ----

class EvaluateRequest(BaseModel):
    policy: str = BASELINE_NAME
    split: Literal["train", "val", "test"] = "test"
    max_scenarios: Optional[int] = None


class DRTEvaluateRequest(BaseModel):
    requests_path: str = "KW_DRT/data/requests_8.csv"
    vehicle_positions_path: str = "KW_DRT/data/vehicle_positions.csv"
    od_matrix_path: str = "KW_DRT/data/od_matrix.csv"
    episode_id: str = "drt_eval"
    output_csv: bool = True


class EvaluateResponse(BaseModel):
    experiment_id: str
    status: str
    message: str


class ExperimentListResponse(BaseModel):
    experiments: list[dict]


class ExperimentDetailResponse(BaseModel):
    experiment: dict
    metrics: list[dict]


# ---- Routes ----

@router.post("/evaluate", response_model=EvaluateResponse)
async def run_evaluation(
    request: EvaluateRequest,
    background_tasks: BackgroundTasks,
) -> EvaluateResponse:
    """
    Evaluate a policy on the specified split.
    Currently supports: demand_aware_greedy_v1
    Runs in a background thread; up to _MAX_CONCURRENT_EVALS at once.
    """
    global _active_evals

    if request.policy not in (BASELINE_NAME, "greedy"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown policy '{request.policy}'. Supported: {BASELINE_NAME}",
        )

    with _eval_lock:
        if _active_evals >= _MAX_CONCURRENT_EVALS:
            raise HTTPException(
                status_code=429,
                detail=f"Too many concurrent evaluations (limit: {_MAX_CONCURRENT_EVALS}). Try again later.",
            )
        _active_evals += 1

    experiment_id = f"exp_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    create_experiment(
        experiment_id=experiment_id,
        policy=request.policy,
        split=request.split,
        config={"max_scenarios": request.max_scenarios},
        domain="sf",
    )

    def _run():
        global _active_evals
        try:
            from backend.datasets.generator import list_scenarios, load_scenario
            paths = list_scenarios(split=request.split)
            if request.max_scenarios:
                paths = paths[: request.max_scenarios]
            if not paths:
                return

            from backend.datasets.evaluator import evaluate_policy_on_scenarios
            output_jsonl = pathlib.Path("data") / "metrics" / f"{experiment_id}.jsonl"
            output_csv_dir = pathlib.Path("data") / "csv" / experiment_id
            metrics = evaluate_policy_on_scenarios(
                paths, greedy_policy_fn, request.policy, output_jsonl,
                output_csv_dir=output_csv_dir,
            )
            record_metrics(experiment_id, metrics)
        except Exception as exc:
            print(f"[evaluator] ERROR in experiment {experiment_id}: {exc}")
        finally:
            with _eval_lock:
                _active_evals -= 1

    background_tasks.add_task(_run)

    return EvaluateResponse(
        experiment_id=experiment_id,
        status="started",
        message=f"Evaluation started in background. experiment_id={experiment_id}",
    )


@router.post("/evaluate/drt", response_model=EvaluateResponse)
async def run_drt_evaluation(
    request: DRTEvaluateRequest,
    background_tasks: BackgroundTasks,
) -> EvaluateResponse:
    """
    Run one DRT episode with the greedy baseline and store results.
    Paths must be under KW_DRT/data/. Runs in a background thread.
    """
    global _active_evals

    # Validate all three paths
    _validate_drt_path(request.requests_path)
    _validate_drt_path(request.vehicle_positions_path)
    _validate_drt_path(request.od_matrix_path)

    with _eval_lock:
        if _active_evals >= _MAX_CONCURRENT_EVALS:
            raise HTTPException(
                status_code=429,
                detail=f"Too many concurrent evaluations (limit: {_MAX_CONCURRENT_EVALS}). Try again later.",
            )
        _active_evals += 1

    experiment_id = f"drt_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    create_experiment(
        experiment_id=experiment_id,
        policy="drt_greedy_v1",
        split="n/a",
        config={
            "requests_path": request.requests_path,
            "vehicle_positions_path": request.vehicle_positions_path,
            "od_matrix_path": request.od_matrix_path,
            "episode_id": request.episode_id,
        },
        domain="drt",
    )

    def _run_drt():
        global _active_evals
        try:
            from backend.datasets.drt_evaluator import evaluate_drt_baseline
            output_csv_dir = (
                pathlib.Path("data") / "drt_csv" / experiment_id
                if request.output_csv
                else None
            )
            metrics = evaluate_drt_baseline(
                requests_path=request.requests_path,
                vehicle_positions_path=request.vehicle_positions_path,
                od_matrix_path=request.od_matrix_path,
                episode_id=request.episode_id,
                output_csv_dir=output_csv_dir,
            )
            record_drt_metrics(experiment_id, [metrics])
        except Exception as exc:
            print(f"[drt_evaluator] ERROR in experiment {experiment_id}: {exc}")
        finally:
            with _eval_lock:
                _active_evals -= 1

    background_tasks.add_task(_run_drt)

    return EvaluateResponse(
        experiment_id=experiment_id,
        status="started",
        message=f"DRT evaluation started in background. experiment_id={experiment_id}",
    )


@router.get("/", response_model=ExperimentListResponse)
async def get_experiments() -> ExperimentListResponse:
    return ExperimentListResponse(experiments=list_experiments())


@router.get("/{experiment_id}", response_model=ExperimentDetailResponse)
async def get_experiment(experiment_id: str) -> ExperimentDetailResponse:
    experiments = list_experiments()
    exp = next((e for e in experiments if e["experiment_id"] == experiment_id), None)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
    metrics = get_experiment_metrics(experiment_id)
    return ExperimentDetailResponse(experiment=exp, metrics=metrics)


@router.get("/{experiment_id}/aggregate")
async def get_aggregate(experiment_id: str) -> dict:
    """
    Return aggregate statistics. Dispatches to DRT or SF aggregator
    based on the experiment's domain field.
    """
    experiments = list_experiments()
    exp = next((e for e in experiments if e["experiment_id"] == experiment_id), None)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")

    if exp.get("domain") == "drt":
        agg = aggregate_drt_metrics(experiment_id)
    else:
        agg = aggregate_metrics(experiment_id)

    if not agg:
        raise HTTPException(status_code=404, detail=f"No metrics for '{experiment_id}'")
    return agg
