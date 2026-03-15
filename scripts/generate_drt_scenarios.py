#!/usr/bin/env python
# ============================================================
# Stage 0.5: generate diversified DRT scenarios + manifest.jsonl
#
# Output: KW_DRT/data/scenarios/
#   manifest.jsonl
#   train/  (50 scenarios, seeds 0-49)
#   val/    (10 scenarios, seeds 50-59)
#   test/   (10 R80 scenarios seeds 60-69 + requests_8_original)
#   vehicle_positions/  (perturbed per scenario)
# ============================================================

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DATA   = _ROOT / "KW_DRT" / "data"
_OUT    = _DATA / "scenarios"


def main() -> None:
    from backend.datasets.drt_scenario_generator import generate_all_scenarios

    manifest_path = generate_all_scenarios(
        requests_80_path     = _DATA / "requests_80.csv",
        requests_8_path      = _DATA / "requests_8.csv",
        vehicle_positions_path = _DATA / "vehicle_positions.csv",
        od_matrix_path       = _DATA / "od_matrix.csv",
        travel_time_path     = _DATA / "travel_time.csv",
        output_dir           = _OUT,
    )
    print(f"[generate] Done. Manifest at: {manifest_path}")


if __name__ == "__main__":
    main()
