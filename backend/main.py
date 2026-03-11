# ============================================================
# FastAPI application entry point
# Run: uvicorn backend.main:app --reload --port 8000
# ============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import datasets, experiments, training, models

app = FastAPI(
    title="Sioux Falls Simulator Backend",
    version="0.1.0",
    description="Dataset generation, training, and evaluation for the Sioux Falls PPO project.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["experiments"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(models.router, prefix="/api/models", tags=["models"])


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
