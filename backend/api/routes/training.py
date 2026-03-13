# ============================================================
# Training API routes
# POST /api/training/start      — start a training job
# POST /api/training/{id}/stop  — stop a training job
# GET  /api/training/           — list all jobs
# GET  /api/training/{id}/status — job status
# WS   /api/training/{id}/ws    — stream metrics (WebSocket)
# ============================================================

from __future__ import annotations

import asyncio
import logging
import uuid

from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from backend.worker.job_manager import job_manager

logger = logging.getLogger(__name__)

router = APIRouter()


# ---- Schemas ----

class TrainingConfig(BaseModel):
    env_type: str = "sf"   # "sf" | "drt"
    # SF-specific
    split: str = "train"
    # DRT-specific (required when env_type="drt")
    drt_requests_path: Optional[str] = None
    drt_vehicle_positions_path: Optional[str] = None
    drt_od_matrix_path: Optional[str] = None
    # Shared hyperparameters
    total_timesteps: int = Field(100_000, ge=1, le=5_000_000)
    n_envs: int = Field(2, ge=1, le=16)
    learning_rate: float = 3e-4
    gamma: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    n_steps: int = Field(2048, ge=64, le=4096)
    batch_size: int = Field(256, ge=32, le=1024)
    checkpoint_freq: int = Field(50_000, ge=1000, le=5_000_000)
    seed: int = 0
    verbose: int = 0


class StartResponse(BaseModel):
    job_id: str
    status: str


# ---- Routes ----

@router.post("/start", response_model=StartResponse)
async def start_training(config: TrainingConfig) -> StartResponse:
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    try:
        job_manager.start_job(job_id, config.model_dump())
    except RuntimeError as exc:
        # Concurrent job limit exceeded
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to start training job")
        raise HTTPException(status_code=500, detail="Failed to start training job.") from exc
    return StartResponse(job_id=job_id, status="running")


@router.post("/{job_id}/stop")
async def stop_training(job_id: str) -> dict:
    stopped = job_manager.stop_job(job_id)
    if not stopped:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found or not running")
    return {"job_id": job_id, "status": "stopped"}


@router.get("/", response_model=list[dict])
async def list_jobs() -> list[dict]:
    return job_manager.list_jobs()


@router.get("/{job_id}/status")
async def job_status(job_id: str) -> dict:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job.to_dict()


@router.websocket("/{job_id}/ws")
async def metrics_stream(websocket: WebSocket, job_id: str) -> None:
    """
    WebSocket endpoint: streams metrics JSONL records as they are written
    by the training subprocess.
    """
    await websocket.accept()
    job = job_manager.get_job(job_id)
    if job is None:
        await websocket.close(code=4004, reason=f"Job '{job_id}' not found")
        return

    queue = job_manager.subscribe(job_id)
    try:
        while True:
            try:
                record = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(record)
            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_json({"type": "ping"})
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        job_manager.unsubscribe(job_id, queue)
