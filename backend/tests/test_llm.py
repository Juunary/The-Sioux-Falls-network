# ============================================================
# Stage 1 / 4.1 unit tests — no external API calls
#
# Tests:
#   1. obs_to_text smoke (v1 idle vehicle)
#   2. obs_to_text auto-detect fallback (v1 by dim)
#   3. obs_to_text unknown env_version raises ValueError
#   4. text_to_action bare integer parse success
#   5. text_to_action action:N pattern
#   6. text_to_action parse failure -> fallback (REJECT)
#   7. text_to_action mask violation -> fallback
#   8. text_to_action out-of-range integer -> fallback
#   9. cache store + lookup hit
#  10. cache miss returns None
#  11. cache key isolation (different model_id)
#  12. cache key isolation (different env_version)
#  13. trace JSONL produced (mocked LLM)
#  14. API schema LLMTeacherRequest validates correctly
#  15. PolicyContext reset_episode_stats clears counters
#  16. obs_to_text mask=all-valid shows all valid actions
#  17. provider validation (valid/invalid)
#  18. env var resolution per provider
#  19. cache key provider isolation
#  20. LLMTeacherRequest requires llm_provider
#  21. LLMConfig defaults and from_env with provider
#  22. PolicyContext llm_provider field
#  23. trace JSONL includes llm_provider field (mocked Anthropic)
#  24. Gemini teacher branch (mocked SDK)
#  25. Gemini cost estimation
#  26. check_provider_sdk (importability check)
#  27. preflight_check_key env var detection
# ============================================================

from __future__ import annotations

import json
import pathlib
import tempfile

import numpy as np
import pytest


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture()
def v1_obs() -> np.ndarray:
    """Minimal drt_env_v1 obs: vehicle at node 1, IDLE, full capacity, time=0, all slots empty."""
    obs = np.zeros(157, dtype=np.float32)
    obs[0] = 1.0   # node 1 (index 0)
    obs[48] = 1.0  # capacity_ratio = 1.0 (full)
    obs[49] = 1.0  # status IDLE
    return obs


@pytest.fixture()
def v1_mask_reject_only() -> np.ndarray:
    """Only REJECT (action 8) is valid."""
    mask = np.zeros(9, dtype=bool)
    mask[8] = True
    return mask


@pytest.fixture()
def v1_mask_slot0_and_reject(v1_obs) -> np.ndarray:
    """Slot 0 and REJECT valid; also fills in a PENDING request in obs[53:66]."""
    return np.array([True, False, False, False, False, False, False, False, True], dtype=bool)


@pytest.fixture()
def v1_obs_with_slot0(v1_obs):
    """v1_obs with a PENDING request in slot 0 (from_node=10, to_node=8)."""
    obs = v1_obs.copy()
    base = 53
    obs[base + 0] = 1.0   # PENDING
    obs[base + 3] = 1.0   # is_valid
    obs[base + 4] = 10 / 24  # from_node 10
    obs[base + 5] = 8 / 24   # to_node 8
    obs[base + 6] = 0.3      # wait 3/10
    obs[base + 7] = 0.5      # arrival_due_left ratio
    obs[base + 9] = 0.4      # dur_to_target ratio
    return obs


# ============================================================
# 1. obs_to_text smoke
# ============================================================

def test_obs_to_text_v1_smoke(v1_obs, v1_mask_reject_only):
    from backend.llm.obs_to_text import obs_to_text

    text = obs_to_text(v1_obs, v1_mask_reject_only, env_version="drt_env_v1")
    assert "node 1" in text
    assert "IDLE" in text
    assert "REJECT" in text


def test_obs_to_text_v1_with_slot(v1_obs_with_slot0):
    from backend.llm.obs_to_text import obs_to_text

    mask = np.array([True, False, False, False, False, False, False, False, True], dtype=bool)
    text = obs_to_text(v1_obs_with_slot0, mask, env_version="drt_env_v1")
    assert "Slot 0" in text
    assert "PENDING" in text
    assert "valid: PICKUP" in text


# ============================================================
# 2. obs_to_text auto-detect fallback (env_version=None)
# ============================================================

def test_obs_to_text_auto_detect(v1_obs, v1_mask_reject_only):
    from backend.llm.obs_to_text import obs_to_text

    text = obs_to_text(v1_obs, v1_mask_reject_only, env_version=None)
    assert "REJECT" in text


# ============================================================
# 3. obs_to_text unknown env_version raises ValueError
# ============================================================

def test_obs_to_text_unknown_env_version(v1_obs, v1_mask_reject_only):
    from backend.llm.obs_to_text import obs_to_text

    with pytest.raises(ValueError, match="unknown env_version"):
        obs_to_text(v1_obs, v1_mask_reject_only, env_version="drt_env_v99")


# ============================================================
# 4-8. text_to_action
# ============================================================

@pytest.fixture()
def v1_mask_all_valid():
    return np.ones(9, dtype=bool)


def test_text_to_action_bare_int(v1_mask_all_valid):
    from backend.llm.text_to_action import text_to_action

    action, parse_ok, mask_ok = text_to_action("0", v1_mask_all_valid, "drt_env_v1")
    assert action == 0
    assert parse_ok is True
    assert mask_ok is True


def test_text_to_action_action_pattern(v1_mask_all_valid):
    from backend.llm.text_to_action import text_to_action

    action, parse_ok, mask_ok = text_to_action("action: 3", v1_mask_all_valid, "drt_env_v1")
    assert action == 3
    assert parse_ok is True


def test_text_to_action_parse_fail_fallback():
    from backend.llm.text_to_action import text_to_action

    mask = np.zeros(9, dtype=bool)
    mask[8] = True  # only REJECT valid
    action, parse_ok, mask_ok = text_to_action("I don't know", mask, "drt_env_v1")
    assert parse_ok is False
    assert action == 8  # fallback to first valid = REJECT


def test_text_to_action_mask_violation_fallback():
    from backend.llm.text_to_action import text_to_action

    mask = np.zeros(9, dtype=bool)
    mask[8] = True  # only REJECT valid
    # LLM says action 2, but it's masked
    action, parse_ok, mask_ok = text_to_action("2", mask, "drt_env_v1")
    assert parse_ok is True
    assert mask_ok is False  # violation detected
    assert action == 8  # fallback


def test_text_to_action_out_of_range_fallback(v1_mask_all_valid):
    from backend.llm.text_to_action import text_to_action

    # 99 is out of range for v1 (max = 8)
    action, parse_ok, mask_ok = text_to_action("99", v1_mask_all_valid, "drt_env_v1")
    assert action == 0  # fallback to first valid


# ============================================================
# 9-12. cache
# ============================================================

def _make_temp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    return pathlib.Path(tmp.name)


def test_cache_store_and_lookup(v1_obs, v1_mask_reject_only):
    from backend.llm.cache import cache_store, cache_lookup

    db = _make_temp_db()
    cache_store(
        v1_obs, v1_mask_reject_only,
        env_version="drt_env_v1",
        model_id="m1",
        prompt_version="v1",
        policy_type="llm_teacher",
        action=8,
        reasoning="reject all",
        db_path=db,
    )
    result = cache_lookup(
        v1_obs, v1_mask_reject_only,
        env_version="drt_env_v1",
        model_id="m1",
        prompt_version="v1",
        policy_type="llm_teacher",
        db_path=db,
    )
    assert result is not None
    action, reasoning = result
    assert action == 8
    assert reasoning == "reject all"


def test_cache_miss_returns_none(v1_obs, v1_mask_reject_only):
    from backend.llm.cache import cache_lookup

    db = _make_temp_db()
    result = cache_lookup(
        v1_obs, v1_mask_reject_only,
        env_version="drt_env_v1",
        model_id="m_missing",
        prompt_version="v1",
        policy_type="llm_teacher",
        db_path=db,
    )
    assert result is None


def test_cache_key_isolation_model_id(v1_obs, v1_mask_reject_only):
    """Different model_id -> different cache entries."""
    from backend.llm.cache import cache_store, cache_lookup

    db = _make_temp_db()
    cache_store(
        v1_obs, v1_mask_reject_only,
        env_version="drt_env_v1", model_id="model_A", prompt_version="v1",
        policy_type="llm_teacher", action=0, db_path=db,
    )
    # model_B should not find the entry stored under model_A
    result = cache_lookup(
        v1_obs, v1_mask_reject_only,
        env_version="drt_env_v1", model_id="model_B", prompt_version="v1",
        policy_type="llm_teacher", db_path=db,
    )
    assert result is None


def test_cache_key_isolation_env_version(v1_obs, v1_mask_reject_only):
    """Different env_version -> different cache entries."""
    from backend.llm.cache import cache_store, cache_lookup

    db = _make_temp_db()
    cache_store(
        v1_obs, v1_mask_reject_only,
        env_version="drt_env_v1", model_id="m1", prompt_version="v1",
        policy_type="llm_teacher", action=0, db_path=db,
    )
    # Lookup with different env_version
    result = cache_lookup(
        v1_obs, v1_mask_reject_only,
        env_version="drt_env_v2", model_id="m1", prompt_version="v1",
        policy_type="llm_teacher", db_path=db,
    )
    assert result is None


# ============================================================
# 13. trace JSONL produced (mocked Anthropic)
# ============================================================

def test_teacher_produces_trace(v1_obs_with_slot0, monkeypatch, tmp_path):
    """Mock the Anthropic SDK via sys.modules to verify trace JSONL is written.

    Does NOT require the anthropic package to be installed.
    """
    import sys
    import types
    from backend.llm.policy_context import PolicyContext

    # ---- Build a minimal fake anthropic module ----
    class _FakeContent:
        text = "0"

    class _FakeUsage:
        input_tokens = 10
        output_tokens = 2

    class _FakeMessage:
        content = [_FakeContent()]
        usage = _FakeUsage()

    class _FakeMessages:
        def create(self, **kwargs):
            return _FakeMessage()

    class _FakeClient:
        messages = _FakeMessages()

    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = lambda **kw: _FakeClient()  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    # ---- Config pointing to tmp dirs ----
    from backend.llm.config import LLMConfig
    cfg = LLMConfig(
        llm_provider="anthropic",
        api_key="fake_key",
        model_id="test-model",
        prompt_version="v1",
        trace_dir=str(tmp_path / "traces"),
        cache_db=str(tmp_path / "cache.db"),
        cache_enabled=False,
    )

    ctx = PolicyContext(
        env_version="drt_env_v1",
        model_id="test-model",
        prompt_version="v1",
        llm_provider="anthropic",
        scenario_id="s0",
        episode_id="ep0",
    )

    import backend.llm.teacher as teacher_mod
    policy_fn = teacher_mod.make_llm_teacher_policy_fn(ctx, cfg)

    mask = np.array([True, False, False, False, False, False, False, False, True], dtype=bool)
    action = policy_fn(v1_obs_with_slot0, mask)
    assert action == 0  # parsed "0"

    # Verify trace file exists and has at least one valid entry
    trace_files = list((tmp_path / "traces").glob("*.jsonl"))
    assert len(trace_files) == 1, f"Expected 1 trace file, got {trace_files}"
    entries = [json.loads(line) for line in trace_files[0].read_text().splitlines() if line.strip()]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["parsed_action"] == 0
    assert entry["parse_success"] is True
    assert entry["cache_hit"] is False
    assert entry["scenario_id"] == "s0"
    assert entry["llm_provider"] == "anthropic"


# ============================================================
# 14. API schema validation
# ============================================================

def test_llm_teacher_request_schema():
    from backend.api.schemas import LLMTeacherRequest

    req = LLMTeacherRequest(
        manifest_path="KW_DRT/data/scenarios/manifest.jsonl",
        split="test",
        env_version="drt_env_v1",
        model_id="claude-sonnet-4-6",
        llm_provider="anthropic",
    )
    assert req.policy_type == "llm_teacher"
    assert req.temperature == 0.0
    assert req.cache_enabled is True
    assert req.llm_provider == "anthropic"


def test_llm_teacher_request_invalid_split():
    from pydantic import ValidationError
    from backend.api.schemas import LLMTeacherRequest

    with pytest.raises(ValidationError):
        LLMTeacherRequest(split="unknown", llm_provider="anthropic")


# ============================================================
# 15. PolicyContext reset_episode_stats
# ============================================================

def test_policy_context_reset():
    from backend.llm.policy_context import PolicyContext

    ctx = PolicyContext(env_version="drt_env_v1", model_id="m", prompt_version="v1")
    ctx.llm_calls = 5
    ctx.cache_hits = 3
    ctx.input_tokens = 100
    ctx.output_tokens = 20
    ctx.invalid_actions = 1
    ctx.step_latencies_ms = [10.0, 20.0, 30.0]
    ctx.reset_episode_stats()
    assert ctx.llm_calls == 0
    assert ctx.cache_hits == 0
    assert ctx.input_tokens == 0
    assert ctx.output_tokens == 0
    assert ctx.invalid_actions == 0
    assert ctx.step_latencies_ms == []

    stats = ctx.to_stats_dict()
    assert all(v == 0 for v in stats.values())


# ============================================================
# 15b. PolicyContext compute_llm_metrics (v13 canonical)
# ============================================================

def test_policy_context_compute_llm_metrics():
    from backend.llm.policy_context import PolicyContext

    ctx = PolicyContext(
        env_version="drt_env_v1",
        model_id="claude-sonnet-4-6",
        prompt_version="v1",
    )
    ctx.llm_calls = 8
    ctx.cache_hits = 2
    ctx.input_tokens = 1000
    ctx.output_tokens = 50
    ctx.invalid_actions = 1
    ctx.step_latencies_ms = [50.0, 100.0, 150.0, 200.0, 250.0,
                              300.0, 350.0, 400.0, 450.0, 500.0]

    m = ctx.compute_llm_metrics()

    # Legacy keys present
    assert m["llm_calls"] == 8
    assert m["cache_hits"] == 2
    assert m["input_tokens"] == 1000
    assert m["output_tokens"] == 50
    assert m["invalid_actions"] == 1

    # invalid_action_rate = 1/10 = 0.1
    assert m["invalid_action_rate"] == 0.1

    # Latency percentiles (10 values: 50-500)
    assert m["episode_latency_p50"] is not None
    assert m["episode_latency_p95"] is not None
    assert 200.0 < m["episode_latency_p50"] < 350.0
    assert 400.0 < m["episode_latency_p95"] < 600.0

    # Cost: claude-sonnet-4 pricing (3.0/M input, 15.0/M output)
    # 1000 * 3.0 / 1M + 50 * 15.0 / 1M = 0.003 + 0.00075 = 0.00375
    assert m["api_cost_per_episode"] is not None
    assert abs(m["api_cost_per_episode"] - 0.00375) < 0.0001

    # api_cost_per_decision = 0.00375 / 10 = 0.000375
    assert m["api_cost_per_decision"] is not None

    # Stage 2/3 fields are None
    assert m["teacher_agreement_rate"] is None
    assert m["planner_override_rate"] is None
    assert m["tool_call_success_rate"] is None


# ============================================================
# 16. obs_to_text mask=all-valid shows all valid actions
# ============================================================

def test_obs_to_text_all_valid_mask(v1_obs):
    from backend.llm.obs_to_text import obs_to_text

    mask = np.ones(9, dtype=bool)
    text = obs_to_text(v1_obs, mask, env_version="drt_env_v1")
    # All 9 actions should appear in the valid actions line
    for i in range(9):
        assert str(i) in text


# ============================================================
# 17. Provider validation
# ============================================================

def test_validate_provider_valid():
    from backend.llm.config import validate_provider
    validate_provider("anthropic")
    validate_provider("gemini")


def test_validate_provider_invalid():
    from backend.llm.config import validate_provider
    with pytest.raises(ValueError, match="Unknown llm_provider"):
        validate_provider("openai")


# ============================================================
# 18. Env var resolution per provider
# ============================================================

def test_resolve_provider_api_key_anthropic(monkeypatch):
    from backend.llm.config import resolve_provider_api_key
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert resolve_provider_api_key("anthropic") == "sk-ant-test"


def test_resolve_provider_api_key_gemini_primary(monkeypatch):
    from backend.llm.config import resolve_provider_api_key
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-gemini")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert resolve_provider_api_key("gemini") == "AIza-gemini"


def test_resolve_provider_api_key_gemini_fallback(monkeypatch):
    from backend.llm.config import resolve_provider_api_key
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-google")
    assert resolve_provider_api_key("gemini") == "AIza-google"


def test_resolve_provider_api_key_gemini_primary_wins(monkeypatch):
    from backend.llm.config import resolve_provider_api_key
    monkeypatch.setenv("GEMINI_API_KEY", "primary")
    monkeypatch.setenv("GOOGLE_API_KEY", "fallback")
    assert resolve_provider_api_key("gemini") == "primary"


def test_resolve_provider_api_key_unknown():
    from backend.llm.config import resolve_provider_api_key
    assert resolve_provider_api_key("openai") == ""


# ============================================================
# 19. Cache key provider isolation
# ============================================================

def test_cache_key_isolation_provider(v1_obs, v1_mask_reject_only):
    """Same obs/mask/model but different provider -> cache miss."""
    from backend.llm.cache import cache_store, cache_lookup

    db = _make_temp_db()
    cache_store(
        v1_obs, v1_mask_reject_only,
        env_version="drt_env_v1", model_id="m1", prompt_version="v1",
        policy_type="llm_teacher", llm_provider="anthropic",
        action=3, db_path=db,
    )
    # Same everything except provider=gemini -> miss
    result = cache_lookup(
        v1_obs, v1_mask_reject_only,
        env_version="drt_env_v1", model_id="m1", prompt_version="v1",
        policy_type="llm_teacher", llm_provider="gemini",
        db_path=db,
    )
    assert result is None

    # Same provider -> hit
    result = cache_lookup(
        v1_obs, v1_mask_reject_only,
        env_version="drt_env_v1", model_id="m1", prompt_version="v1",
        policy_type="llm_teacher", llm_provider="anthropic",
        db_path=db,
    )
    assert result is not None
    assert result[0] == 3


# ============================================================
# 20. LLMTeacherRequest requires llm_provider
# ============================================================

def test_llm_teacher_request_requires_provider():
    from pydantic import ValidationError
    from backend.api.schemas import LLMTeacherRequest

    # Missing llm_provider should fail (it's required, no default)
    with pytest.raises(ValidationError):
        LLMTeacherRequest(
            manifest_path="KW_DRT/data/scenarios/manifest.jsonl",
            split="test",
            model_id="claude-sonnet-4-6",
        )


def test_llm_teacher_request_gemini():
    from backend.api.schemas import LLMTeacherRequest

    req = LLMTeacherRequest(
        llm_provider="gemini",
        model_id="gemini-2.5-flash",
    )
    assert req.llm_provider == "gemini"
    assert req.policy_type == "llm_teacher"


def test_llm_teacher_request_invalid_provider():
    from pydantic import ValidationError
    from backend.api.schemas import LLMTeacherRequest

    with pytest.raises(ValidationError):
        LLMTeacherRequest(llm_provider="openai")


# ============================================================
# 21. LLMConfig defaults and from_env with provider
# ============================================================

def test_llm_config_default_provider():
    from backend.llm.config import LLMConfig
    cfg = LLMConfig(api_key="test")
    assert cfg.llm_provider == "anthropic"


def test_llm_config_gemini_provider():
    from backend.llm.config import LLMConfig
    cfg = LLMConfig(llm_provider="gemini", api_key="test")
    assert cfg.llm_provider == "gemini"


def test_llm_config_invalid_provider():
    from backend.llm.config import LLMConfig
    with pytest.raises(ValueError, match="Unknown llm_provider"):
        LLMConfig(llm_provider="openai", api_key="test")


def test_from_env_anthropic(monkeypatch):
    from backend.llm.config import from_env
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cfg = from_env(llm_provider="anthropic")
    assert cfg.llm_provider == "anthropic"
    assert cfg.api_key == "sk-test"


def test_from_env_gemini(monkeypatch):
    from backend.llm.config import from_env
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    cfg = from_env(llm_provider="gemini", model_id="gemini-2.5-flash")
    assert cfg.llm_provider == "gemini"
    assert cfg.api_key == "AIza-test"
    assert cfg.model_id == "gemini-2.5-flash"


# ============================================================
# 22. PolicyContext llm_provider field
# ============================================================

def test_policy_context_default_provider():
    from backend.llm.policy_context import PolicyContext
    ctx = PolicyContext(env_version="drt_env_v1", model_id="m", prompt_version="v1")
    assert ctx.llm_provider == "anthropic"


def test_policy_context_gemini_provider():
    from backend.llm.policy_context import PolicyContext
    ctx = PolicyContext(
        env_version="drt_env_v1", model_id="gemini-2.5-flash",
        prompt_version="v1", llm_provider="gemini",
    )
    assert ctx.llm_provider == "gemini"


# ============================================================
# 23. Trace JSONL includes llm_provider (mocked Anthropic)
# ============================================================

# (covered in test 13 above — asserts entry["llm_provider"] == "anthropic")


# ============================================================
# 24. Gemini teacher branch (mocked SDK)
# ============================================================

def test_teacher_gemini_branch(v1_obs_with_slot0, monkeypatch, tmp_path):
    """Mock the google.generativeai SDK to verify Gemini branch works."""
    import sys
    import types
    from backend.llm.policy_context import PolicyContext

    # ---- Build a minimal fake google.generativeai module ----
    class _FakeUsageMeta:
        prompt_token_count = 15
        candidates_token_count = 3

    class _FakeResponse:
        text = "0"
        usage_metadata = _FakeUsageMeta()

    class _FakeModel:
        def __init__(self, name):
            self.name = name

        def generate_content(self, prompt, generation_config=None):
            return _FakeResponse()

    class _FakeGenerationConfig:
        def __init__(self, **kwargs):
            pass

    class _FakeTypes:
        GenerationConfig = _FakeGenerationConfig

    fake_genai = types.ModuleType("google.generativeai")
    fake_genai.configure = lambda api_key: None  # type: ignore[attr-defined]
    fake_genai.GenerativeModel = _FakeModel  # type: ignore[attr-defined]
    fake_genai.types = _FakeTypes()  # type: ignore[attr-defined]
    fake_genai.list_models = lambda: [{"name": "gemini-2.5-flash"}]  # type: ignore[attr-defined]

    # Also register google and google.generativeai in sys.modules
    fake_google = types.ModuleType("google")
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)

    from backend.llm.config import LLMConfig
    cfg = LLMConfig(
        llm_provider="gemini",
        api_key="fake_gemini_key",
        model_id="gemini-2.5-flash",
        prompt_version="v1",
        trace_dir=str(tmp_path / "traces"),
        cache_db=str(tmp_path / "cache.db"),
        cache_enabled=False,
    )

    ctx = PolicyContext(
        env_version="drt_env_v1",
        model_id="gemini-2.5-flash",
        prompt_version="v1",
        llm_provider="gemini",
        scenario_id="s0",
        episode_id="ep0",
    )

    import backend.llm.teacher as teacher_mod
    policy_fn = teacher_mod.make_llm_teacher_policy_fn(ctx, cfg)

    mask = np.array([True, False, False, False, False, False, False, False, True], dtype=bool)
    action = policy_fn(v1_obs_with_slot0, mask)
    assert action == 0  # parsed "0"

    # Verify trace includes provider
    trace_files = list((tmp_path / "traces").glob("*.jsonl"))
    assert len(trace_files) == 1
    entries = [json.loads(line) for line in trace_files[0].read_text().splitlines() if line.strip()]
    assert len(entries) == 1
    assert entries[0]["llm_provider"] == "gemini"
    assert entries[0]["input_tokens"] == 15
    assert entries[0]["output_tokens"] == 3


# ============================================================
# 25. Gemini cost estimation
# ============================================================

def test_gemini_cost_estimation():
    from backend.llm.policy_context import PolicyContext

    ctx = PolicyContext(
        env_version="drt_env_v1",
        model_id="gemini-2.5-flash",
        prompt_version="v1",
        llm_provider="gemini",
    )
    ctx.llm_calls = 10
    ctx.input_tokens = 1000
    ctx.output_tokens = 50

    m = ctx.compute_llm_metrics()
    # gemini-2.5-flash: $0.15/M input, $0.60/M output
    # 1000 * 0.15 / 1M + 50 * 0.60 / 1M = 0.00015 + 0.00003 = 0.00018
    assert m["api_cost_per_episode"] is not None
    assert abs(m["api_cost_per_episode"] - 0.00018) < 0.00001


# ============================================================
# 26. check_provider_sdk
# ============================================================

def test_check_provider_sdk_unknown():
    from backend.llm.config import check_provider_sdk
    ok, msg = check_provider_sdk("openai")
    assert ok is False
    assert "Unknown" in msg


# ============================================================
# 27. preflight_check_key
# ============================================================

def test_preflight_check_key_missing(monkeypatch):
    from backend.llm.config import preflight_check_key
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ok, msg = preflight_check_key("anthropic")
    assert ok is False
    assert "not set" in msg


def test_preflight_check_key_not_set_value(monkeypatch):
    from backend.llm.config import preflight_check_key
    monkeypatch.setenv("ANTHROPIC_API_KEY", "NOT_SET")
    ok, msg = preflight_check_key("anthropic")
    assert ok is False


def test_preflight_check_key_present(monkeypatch):
    from backend.llm.config import preflight_check_key
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real")
    ok, msg = preflight_check_key("anthropic")
    assert ok is True
    assert "present" in msg


def test_preflight_check_key_gemini(monkeypatch):
    from backend.llm.config import preflight_check_key
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-real")
    ok, msg = preflight_check_key("gemini")
    assert ok is True
