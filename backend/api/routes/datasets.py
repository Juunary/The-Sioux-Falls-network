# ============================================================
# Dataset API routes
# POST /api/datasets/generate  — generate scenario batch
# GET  /api/datasets/list      — list existing scenarios
# GET  /api/datasets/{id}      — fetch single scenario JSON
# ============================================================

from __future__ import annotations

import logging
import pathlib

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

from backend.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    ScenarioData,
    ScenarioListItem,
    ScenarioListResponse,
)
from backend.datasets.generator import (
    generate_batch,
    list_scenarios,
    load_scenario,
)

router = APIRouter()

_DEFAULT_DATA_DIR = pathlib.Path(__file__).parent.parent.parent.parent / "data" / "scenarios"


@router.post("/generate", response_model=GenerateResponse)
async def generate_scenarios(request: GenerateRequest) -> GenerateResponse:
    """
    Generate a batch of deterministic scenario JSON files.
    Each scenario is stored under data/scenarios/<split>/<scenario_id>.json.
    """
    try:
        result = generate_batch(request)
        return result
    except Exception as exc:
        logger.exception("Scenario generation failed")
        raise HTTPException(status_code=500, detail="Scenario generation failed. Check server logs.") from exc


@router.get("/list", response_model=ScenarioListResponse)
async def list_scenario_files(
    split: str | None = Query(None, description="Filter by split: train, val, test"),
) -> ScenarioListResponse:
    """Return metadata for all existing scenario files."""
    paths = list_scenarios(split=split)
    items: list[ScenarioListItem] = []

    for p in paths:
        try:
            scenario = load_scenario(p)
            items.append(ScenarioListItem(
                scenario_id=scenario.meta.scenario_id,
                split=scenario.meta.split,
                difficulty=scenario.meta.difficulty,
                seed=scenario.meta.seed,
                bus_count=scenario.meta.bus_count,
                episode_length=scenario.meta.episode_length,
                event_count=len(scenario.passenger_events),
            ))
        except Exception:
            # Skip corrupt files — log but don't crash
            continue

    return ScenarioListResponse(scenarios=items, total=len(items))


@router.get("/{scenario_id}", response_model=ScenarioData)
async def get_scenario(scenario_id: str) -> ScenarioData:
    """Fetch a single scenario by ID (searches all splits)."""
    paths = list_scenarios()
    for p in paths:
        if p.stem == scenario_id:
            try:
                return load_scenario(p)
            except Exception as exc:
                logger.exception("Failed to load scenario '%s'", scenario_id)
                raise HTTPException(status_code=500, detail="Failed to load scenario. Check server logs.") from exc

    raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found")
