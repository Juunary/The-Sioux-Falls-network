# ============================================================
# text_to_action — parse LLM response text into a valid DRT action
#
# text_to_action(text, mask, env_version) -> int
#
# Parsing strategy (in order):
#   1. Look for a bare integer in the response (e.g. "0", "  8  ")
#   2. Look for patterns like "action: 3", "answer: 5", "select 2"
#   3. If parsed action is out of range or mask=0, apply fallback:
#      first valid action in mask order (ascending), with warning logged.
# ============================================================

from __future__ import annotations

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

# env_version -> action_dim
_ACTION_DIMS: dict[str, int] = {
    "drt_env_v1":               9,
    "drt_env_v1_hybrid":        9,
    "drt_env_v1_scaled":       17,
    "drt_env_v1_scaled_hybrid": 17,
    "drt_env_v2":              33,
    "drt_env_v2_hybrid":       33,
}

# Regex patterns for extracting an integer from LLM output
_PATTERNS = [
    re.compile(r"^\s*(\d+)\s*$"),                        # bare integer line
    re.compile(r"action[:\s]+(\d+)", re.IGNORECASE),
    re.compile(r"answer[:\s]+(\d+)", re.IGNORECASE),
    re.compile(r"select[:\s]+(\d+)", re.IGNORECASE),
    re.compile(r"choose[:\s]+(\d+)", re.IGNORECASE),
    re.compile(r"\b(\d+)\b"),                             # first integer anywhere
]


def _first_valid_action(mask: np.ndarray) -> int:
    """Return the lowest-index valid action (fallback)."""
    for i, m in enumerate(mask):
        if m:
            return i
    return 0  # REJECT always valid as last slot; should not reach here


def text_to_action(
    text: str,
    mask: np.ndarray,
    env_version: str,
) -> tuple[int, bool, bool]:
    """Parse LLM response text into a valid DRT action.

    Returns:
        (action, parse_success, mask_compliant)

        parse_success:   True if an integer was found in the text.
        mask_compliant:  True if the parsed (or fallback) action is in mask.

    Fallback behaviour on parse failure OR mask violation:
        - Selects the first valid action (lowest index where mask=True).
        - Logs a warning.
    """
    action_dim = _ACTION_DIMS.get(env_version)
    if action_dim is None:
        raise ValueError(f"text_to_action: unknown env_version '{env_version}'")

    parsed: int | None = None

    for pat in _PATTERNS:
        m = pat.search(text)
        if m:
            candidate = int(m.group(1))
            if 0 <= candidate < action_dim:
                parsed = candidate
                break

    # Evaluate parse success
    parse_success = parsed is not None

    if not parse_success:
        logger.warning(
            "text_to_action: parse failed for env_version=%s, text=%r — "
            "using fallback action",
            env_version, text[:100],
        )
        fallback = _first_valid_action(mask)
        return fallback, False, bool(mask[fallback])

    # Check mask compliance
    if parsed < len(mask) and mask[parsed]:
        return parsed, True, True

    # Mask violation: fallback
    logger.warning(
        "text_to_action: action %d is mask=0 for env_version=%s — "
        "using fallback action",
        parsed, env_version,
    )
    fallback = _first_valid_action(mask)
    return fallback, True, False   # parse_success=True, mask_compliant=False
