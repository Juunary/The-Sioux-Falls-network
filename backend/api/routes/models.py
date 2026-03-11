# ============================================================
# Models API routes
#
# POST /api/models/{job_id}/export_onnx  — export SB3 → ONNX
# GET  /api/models/                      — list exported models
# GET  /api/models/{model_id}            — download ONNX file
# ============================================================

from __future__ import annotations

import asyncio
import logging
import pathlib
import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.trainer.export import export_to_onnx

logger = logging.getLogger(__name__)

router = APIRouter()

_MODELS_DIR = pathlib.Path("data/models")
_TRAINING_DIR = pathlib.Path("data/training")

# Whitelist: alphanumeric, underscore, hyphen; 1–64 chars.
# Prevents path traversal via job_id / model_id.
_SAFE_ID_RE = re.compile(r'^[a-zA-Z0-9_\-]{1,64}$')


def _validate_safe_id(value: str, label: str) -> None:
    """Raise HTTP 400 if *value* contains path-unsafe characters."""
    if not _SAFE_ID_RE.match(value):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {label}: must be 1–64 alphanumeric/underscore/hyphen characters.",
        )


def _assert_within(path: pathlib.Path, base: pathlib.Path, label: str) -> None:
    """Raise HTTP 400 if resolved *path* escapes *base* directory."""
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid {label}.")


@router.post("/{job_id}/export_onnx")
async def export_model_onnx(job_id: str) -> dict:
    """
    Export the final PPO checkpoint for *job_id* to ONNX.

    Looks for ``data/training/{job_id}/ppo_final.zip`` (written by
    training_worker.py on successful completion).
    Writes the ONNX file to ``data/models/{job_id}.onnx``.
    """
    _validate_safe_id(job_id, "job_id")
    ckpt_path = _TRAINING_DIR / job_id / "ppo_final.zip"
    _assert_within(ckpt_path, _TRAINING_DIR, "job_id")

    if not ckpt_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Final checkpoint not found for job '{job_id}'.",
        )

    output_path = _MODELS_DIR / f"{job_id}.onnx"
    try:
        # torch.onnx.export is CPU-bound — run in thread pool
        meta = await asyncio.to_thread(export_to_onnx, ckpt_path, output_path)
        return meta
    except Exception as exc:
        logger.exception("ONNX export failed for job '%s'", job_id)
        raise HTTPException(status_code=500, detail="Export failed. Check server logs.") from exc


@router.get("/")
async def list_models() -> dict:
    """Return stem names of all exported ONNX files."""
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = sorted(p.stem for p in _MODELS_DIR.glob("*.onnx"))
    return {"models": models}


@router.get("/{model_id}")
async def download_model(model_id: str) -> FileResponse:
    """Serve an ONNX model file for browser-side inference."""
    _validate_safe_id(model_id, "model_id")
    path = _MODELS_DIR / f"{model_id}.onnx"
    _assert_within(path, _MODELS_DIR, "model_id")

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    return FileResponse(
        str(path),
        media_type="application/octet-stream",
        filename=f"{model_id}.onnx",
    )
