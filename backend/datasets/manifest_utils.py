# ============================================================
# Manifest utilities — shared by training_worker, drt_evaluator, API
#
# resolve_and_validate  : relative path → absolute path with
#                         KW_DRT/data/ boundary check
# load_manifest_entries : parse manifest.jsonl, filter by split,
#                         optionally filter by comparison_group
# ============================================================

from __future__ import annotations

import json
import logging
import pathlib

logger = logging.getLogger(__name__)

# All manifest-referenced paths must stay within this root
DATA_ROOT: pathlib.Path = (
    pathlib.Path(__file__).resolve().parents[2] / "KW_DRT" / "data"
)

# Fields that every manifest entry must provide
_REQUIRED_FIELDS = frozenset({
    "scenario_id",
    "split",
    "comparison_group",
    "requests_path",
    "vehicle_positions_path",
    "od_matrix_path",
})


def resolve_and_validate(
    manifest_dir: pathlib.Path,
    relative_path: str,
) -> pathlib.Path:
    """Resolve a manifest-relative path and verify it stays within DATA_ROOT.

    Args:
        manifest_dir: directory that contains manifest.jsonl
        relative_path: relative path string from a manifest entry

    Returns:
        Resolved absolute Path

    Raises:
        ValueError: if the resolved path escapes KW_DRT/data/
    """
    resolved = (manifest_dir / relative_path).resolve()
    if not resolved.is_relative_to(DATA_ROOT):
        raise ValueError(
            f"manifest path escapes data boundary: {resolved}"
        )
    return resolved


def load_manifest_entries(
    manifest_path: pathlib.Path | str,
    split: str,
    expected_comparison_group: str | None = None,
) -> list[dict]:
    """Parse manifest.jsonl and return entries matching split.

    Structural validation only — path resolution is NOT performed here.
    Callers (RotatingDRTEpisodeEnv, evaluate_drt_policy_on_manifest)
    call resolve_and_validate() when they actually open the files.

    Args:
        manifest_path: path to manifest.jsonl
        split: "train" | "val" | "test"
        expected_comparison_group: if provided, entries whose
            comparison_group does not match are skipped with a warning.
            Pass None to accept all comparison_groups.

    Returns:
        List of matching entry dicts (never empty — raises if empty)

    Raises:
        ValueError: if no entries remain after filtering
    """
    manifest_path = pathlib.Path(manifest_path)
    entries: list[dict] = []

    with open(manifest_path, encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "manifest %s line %d: JSON parse error: %s — skipping",
                    manifest_path, line_no, exc,
                )
                continue

            # Split filter
            if entry.get("split") != split:
                continue

            # Required fields check
            missing = _REQUIRED_FIELDS - entry.keys()
            if missing:
                logger.warning(
                    "manifest %s line %d (scenario_id=%r): missing fields %s — skipping",
                    manifest_path, line_no, entry.get("scenario_id", "?"), missing,
                )
                continue

            # comparison_group filter (only when caller supplies expected value)
            if expected_comparison_group is not None:
                got = entry.get("comparison_group")
                if got != expected_comparison_group:
                    logger.warning(
                        "manifest %s line %d (scenario_id=%r): "
                        "comparison_group mismatch — expected %r, got %r — skipping",
                        manifest_path, line_no, entry["scenario_id"],
                        expected_comparison_group, got,
                    )
                    continue

            entries.append(entry)

    if not entries:
        raise ValueError(
            f"no entries for split={split!r} in {manifest_path}"
        )

    return entries
