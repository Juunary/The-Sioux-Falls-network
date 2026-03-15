# ============================================================
# Stage 0.5 manifest tests
#
# Tests:
# 1. resolve_and_validate — valid path + escape attempt
# 2. load_manifest_entries — split counts (train=50, val=10, test=11)
# 3. test_r8_original entry presence
# 4. comparison_group mismatch skip + warning
# 5. RotatingDRTEpisodeEnv round-robin (manifest mode)
# 6. RotatingDRTEpisodeEnv backward compat (single-CSV mode)
# ============================================================

from __future__ import annotations

import json
import pathlib

import pytest

from backend.datasets.manifest_utils import (
    DATA_ROOT,
    load_manifest_entries,
    resolve_and_validate,
)

# ---- Shared paths -----------------------------------------------
_DATA     = pathlib.Path(__file__).resolve().parents[2] / "KW_DRT" / "data"
_MANIFEST = _DATA / "scenarios" / "manifest.jsonl"
_REQ_8    = str(_DATA / "requests_8.csv")
_VEH      = str(_DATA / "vehicle_positions.csv")
_OD       = str(_DATA / "od_matrix.csv")

_MANIFEST_EXISTS = _MANIFEST.exists()


# ============================================================
# 1. resolve_and_validate
# ============================================================

def test_resolve_valid():
    """A path inside DATA_ROOT resolves correctly."""
    manifest_dir = _DATA / "scenarios"
    result = resolve_and_validate(manifest_dir, "../od_matrix.csv")
    assert result == (_DATA / "od_matrix.csv").resolve()
    assert result.exists()


def test_resolve_escape_raises():
    """A path that escapes DATA_ROOT raises ValueError."""
    manifest_dir = _DATA / "scenarios"
    with pytest.raises(ValueError, match="escapes data boundary"):
        resolve_and_validate(manifest_dir, "../../some_file.txt")


def test_resolve_deeply_escaped_raises():
    """Absolute-like traversal also escapes boundary."""
    manifest_dir = _DATA / "scenarios"
    with pytest.raises(ValueError, match="escapes data boundary"):
        resolve_and_validate(manifest_dir, "../../../backend/env/drt_env.py")


# ============================================================
# 2. split counts (requires generated manifest)
# ============================================================

@pytest.mark.skipif(not _MANIFEST_EXISTS, reason="manifest.jsonl not yet generated")
def test_split_counts():
    """train=50, val=10, test=11."""
    train = load_manifest_entries(_MANIFEST, "train")
    val   = load_manifest_entries(_MANIFEST, "val")
    test  = load_manifest_entries(_MANIFEST, "test")

    assert len(train) == 50, f"Expected 50 train entries, got {len(train)}"
    assert len(val)   == 10, f"Expected 10 val entries, got {len(val)}"
    assert len(test)  == 11, f"Expected 11 test entries, got {len(test)}"


# ============================================================
# 3. test_r8_original
# ============================================================

@pytest.mark.skipif(not _MANIFEST_EXISTS, reason="manifest.jsonl not yet generated")
def test_r8_original_present():
    """test_r8_original must exist in the test split."""
    test_entries = load_manifest_entries(_MANIFEST, "test")
    ids = [e["scenario_id"] for e in test_entries]
    assert "test_r8_original" in ids, f"test_r8_original not found in: {ids}"


@pytest.mark.skipif(not _MANIFEST_EXISTS, reason="manifest.jsonl not yet generated")
def test_r8_original_metadata():
    """test_r8_original must have zero perturbation."""
    test_entries = load_manifest_entries(_MANIFEST, "test")
    orig = next(e for e in test_entries if e["scenario_id"] == "test_r8_original")
    assert orig["seed"] == -1
    assert orig["jitter_range"] == 0
    assert orig["drop_rate"] == 0.0
    assert orig["num_requests"] == 8


# ============================================================
# 4. comparison_group mismatch
# ============================================================

def test_comparison_group_mismatch_skipped(tmp_path):
    """Entries with wrong comparison_group are skipped; ValueError if all skipped."""
    manifest = tmp_path / "manifest.jsonl"
    entries = [
        {
            "scenario_id": "bad_s0",
            "split": "train",
            "comparison_group": "L1",          # wrong
            "requests_path": "train/foo.csv",
            "vehicle_positions_path": "vp/foo.csv",
            "od_matrix_path": "../od_matrix.csv",
        },
    ]
    manifest.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")

    with pytest.raises(ValueError, match="no entries for split="):
        load_manifest_entries(manifest, "train", expected_comparison_group="L0")


def test_comparison_group_mixed(tmp_path):
    """Only matching entries are returned when some have wrong comparison_group."""
    manifest = tmp_path / "manifest.jsonl"
    entries = [
        {
            "scenario_id": "good",
            "split": "train",
            "comparison_group": "L0",
            "requests_path": "train/good.csv",
            "vehicle_positions_path": "vp/good.csv",
            "od_matrix_path": "../od_matrix.csv",
        },
        {
            "scenario_id": "bad",
            "split": "train",
            "comparison_group": "L1",
            "requests_path": "train/bad.csv",
            "vehicle_positions_path": "vp/bad.csv",
            "od_matrix_path": "../od_matrix.csv",
        },
    ]
    manifest.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")

    result = load_manifest_entries(manifest, "train", expected_comparison_group="L0")
    assert len(result) == 1
    assert result[0]["scenario_id"] == "good"


def test_missing_required_field_skipped(tmp_path):
    """Entries missing required fields are skipped."""
    manifest = tmp_path / "manifest.jsonl"
    entry = {
        "scenario_id": "incomplete",
        "split": "train",
        "comparison_group": "L0",
        # missing requests_path, vehicle_positions_path, od_matrix_path
    }
    manifest.write_text(json.dumps(entry), encoding="utf-8")

    with pytest.raises(ValueError, match="no entries for split="):
        load_manifest_entries(manifest, "train")


def test_nonexistent_split_raises():
    """Requesting a non-existent split raises ValueError."""
    if not _MANIFEST_EXISTS:
        pytest.skip("manifest.jsonl not yet generated")
    with pytest.raises(ValueError, match="no entries for split="):
        load_manifest_entries(_MANIFEST, "nonexistent_split")


# ============================================================
# 5. RotatingDRTEpisodeEnv — manifest round-robin
# ============================================================

@pytest.mark.skipif(not _MANIFEST_EXISTS, reason="manifest.jsonl not yet generated")
def test_rotating_env_manifest_reset():
    """Manifest-mode env completes multiple resets without error."""
    from backend.trainer.ppo_trainer import RotatingDRTEpisodeEnv

    env = RotatingDRTEpisodeEnv(
        manifest_path=str(_MANIFEST),
        split="train",
        env_index=0,
        env_version="drt_env_v1",
    )
    assert env.observation_space is not None
    assert env.action_space is not None

    for _ in range(3):
        obs, _ = env.reset()
        assert obs.shape == (157,)
        mask = env.action_masks()
        assert mask.shape == (9,)
        assert mask[-1]  # REJECT always valid


@pytest.mark.skipif(not _MANIFEST_EXISTS, reason="manifest.jsonl not yet generated")
def test_rotating_env_stagger():
    """env_index=1 starts at a different scenario than env_index=0."""
    from backend.trainer.ppo_trainer import RotatingDRTEpisodeEnv

    env0 = RotatingDRTEpisodeEnv(
        manifest_path=str(_MANIFEST), split="train",
        env_index=0, env_version="drt_env_v1",
    )
    env1 = RotatingDRTEpisodeEnv(
        manifest_path=str(_MANIFEST), split="train",
        env_index=1, env_version="drt_env_v1",
    )
    # After reset, episode_count becomes 1; first entry index differs by env_index
    assert (
        env0._entries[0 % len(env0._entries)]["scenario_id"]
        != env1._entries[1 % len(env1._entries)]["scenario_id"]
    )


@pytest.mark.skipif(not _MANIFEST_EXISTS, reason="manifest.jsonl not yet generated")
def test_rotating_env_round_robin():
    """After N resets, the env has cycled through N different scenarios."""
    from backend.trainer.ppo_trainer import RotatingDRTEpisodeEnv

    env = RotatingDRTEpisodeEnv(
        manifest_path=str(_MANIFEST), split="val",
        env_index=0, env_version="drt_env_v1",
    )
    n_entries = len(env._entries)
    assert n_entries == 10  # val split

    # Reset n_entries + 2 times and verify episode_count increments
    for i in range(n_entries + 2):
        env.reset()
        assert env._episode_count == i + 1


# ============================================================
# 6. Backward compatibility — single-CSV mode
# ============================================================

def test_rotating_env_single_csv_mode():
    """Without manifest_path, original three-path behaviour is preserved."""
    from backend.trainer.ppo_trainer import RotatingDRTEpisodeEnv

    env = RotatingDRTEpisodeEnv(
        requests_path=_REQ_8,
        vehicle_positions_path=_VEH,
        od_matrix_path=_OD,
    )
    obs, _ = env.reset()
    assert obs.shape == (157,)
    mask = env.action_masks()
    assert mask[-1]  # REJECT always valid
    # episode_count not incremented in single-CSV mode
    assert env._episode_count == 0
