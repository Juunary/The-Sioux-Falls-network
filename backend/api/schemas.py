# ============================================================
# Pydantic schemas for API request / response bodies
# Mirrors src/types/network.ts for cross-language consistency
# ============================================================

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---- Scenario generation request ----

class GenerateRequest(BaseModel):
    count: int = Field(10, ge=1, le=1000, description="Number of scenarios to generate")
    seed_start: int = Field(0, ge=0, description="First seed (inclusive); each scenario uses seed_start + i")
    split: Literal["train", "val", "test"] = "train"
    difficulty: Literal["easy", "medium", "hard", "stress"] = "medium"

    # Episode settings
    bus_count: int = Field(6, ge=1, le=20)
    episode_length: float = Field(600.0, gt=0)
    spawn_rate: float = Field(0.8, gt=0)
    passenger_capacity: int = Field(6, ge=1)
    reboard_enabled: bool = False
    charging_enabled: bool = True
    low_battery_threshold: float = Field(20.0, ge=0, le=100)
    charge_duration: float = Field(8.0, gt=0)
    battery_drain_rate: float = Field(0.08, ge=0)


# ---- Domain types (mirrors TS) ----

class BusInitial(BaseModel):
    id: int
    start_node: int
    battery: float
    speed: float


class PassengerEvent(BaseModel):
    event_id: int
    spawn_time: float
    origin: int
    destination: int
    patience: float


class ScenarioMeta(BaseModel):
    scenario_id: str
    graph_version: str = "v1"
    seed: int
    split: Literal["train", "val", "test"]
    difficulty: Literal["easy", "medium", "hard", "stress"]
    episode_length: float
    sim_dt: float = 0.016
    bus_count: int
    spawn_rate: float
    reboard_enabled: bool
    charging_enabled: bool
    low_battery_threshold: float
    charge_duration: float
    battery_drain_rate: float
    passenger_capacity: int
    generator_version: str = "1.0"


class ScenarioData(BaseModel):
    meta: ScenarioMeta
    buses_initial: list[BusInitial]
    passenger_events: list[PassengerEvent]


# ---- API responses ----

class GenerateResponse(BaseModel):
    generated: int
    scenario_ids: list[str]
    output_dir: str


class ScenarioListItem(BaseModel):
    scenario_id: str
    split: str
    difficulty: str
    seed: int
    bus_count: int
    episode_length: float
    event_count: int


class ScenarioListResponse(BaseModel):
    scenarios: list[ScenarioListItem]
    total: int


# ---- Stage 1: LLM evaluation request schemas ----

class LLMTeacherRequest(BaseModel):
    """Request body for POST /evaluate/drt/batch with policy_type='llm_teacher'.

    Uses manifest_path + split as the canonical input interface (Stage 0.5+).
    llm_provider is required — API callers must explicitly select a provider.
    """
    model_config = {"protected_namespaces": ()}

    manifest_path: str = "KW_DRT/data/scenarios/manifest.jsonl"
    split: Literal["train", "val", "test"] = "test"
    policy_type: Literal["llm_teacher"] = "llm_teacher"
    env_version: str = "drt_env_v1"
    llm_provider: Literal["anthropic", "gemini"] = Field(
        ...,  # required — no default; API must explicitly choose
        description="LLM provider: 'anthropic' (Claude) or 'gemini' (Google)",
    )
    model_id: str = "claude-sonnet-4-6"
    prompt_version: str = "v1"
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(64, ge=1, le=4096)
    cache_enabled: bool = True
    output_csv: bool = True
