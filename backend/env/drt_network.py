# ============================================================
# DRTNetwork — loads OD travel-time matrix from CSV
# Ported from KW_DRT/app/network.py (import path only changed)
# Node IDs are 1-based integers (as in od_matrix.csv)
# ============================================================

import pandas as pd


class DRTNetwork:
    def __init__(self) -> None:
        self.num_nodes: int = 0
        self.od_dur_mat: dict[int, dict[int, float]] = {}
        self.max_duration: float = -1.0

    def set_od_matrix(self, path: str) -> None:
        df = pd.read_csv(path, index_col=0)
        self.num_nodes = len(df.index)
        self.od_dur_mat = {
            int(o): {int(d): float(df.loc[o, d]) for d in df.columns}
            for o in df.index
        }
        self.max_duration = max(
            float(df.loc[o, d]) for o in df.index for d in df.columns
        )

    def get_duration(self, from_node_id: int, to_node_id: int) -> float:
        return self.od_dur_mat[from_node_id][to_node_id]
