# ============================================================
# Sioux Falls graph topology — Python mirror of src/data/network.ts
#
# BENCHMARK TOPOLOGY v1 — DO NOT CHANGE AFTER DATASET GENERATION
# 24 nodes (0-indexed), 76 directed edges.
# Charging stations: 2, 4, 7, 10, 14, 16, 19, 23
# Edges 74/75 (nodes 5 ↔ 6) are retained as part of v1.
#
# Node coordinates match NODE_COORDS in network.ts exactly.
# Edge weights = Euclidean pixel distance between node coords.
# ============================================================

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    id: int
    x: float
    y: float
    is_charging_station: bool


@dataclass(frozen=True)
class Edge:
    id: int
    source: int
    target: int
    weight: float


# ----------------------------------------------------------
# Node coordinates — must match NODE_COORDS in network.ts
# ----------------------------------------------------------
_NODE_COORDS: dict[int, tuple[float, float]] = {
    0:  (100, 130),
    1:  (680, 130),
    2:  (100, 260),
    3:  (246, 260),
    4:  (420, 260),
    5:  (586, 260),
    6:  (860, 340),
    7:  (586, 370),
    8:  (420, 370),
    9:  (320, 470),
    10: (226, 370),
    11: (100, 370),
    12: (100, 730),
    13: (226, 540),
    14: (380, 540),
    15: (490, 470),
    16: (490, 600),
    17: (860, 470),
    18: (606, 600),
    19: (674, 730),
    20: (490, 730),
    21: (380, 654),
    22: (226, 654),
    23: (294, 730),
}

_CHARGING_STATIONS: frozenset[int] = frozenset({2, 4, 7, 10, 14, 16, 19, 23})

NODES: list[Node] = [
    Node(
        id=i,
        x=_NODE_COORDS[i][0],
        y=_NODE_COORDS[i][1],
        is_charging_station=(i in _CHARGING_STATIONS),
    )
    for i in range(24)
]

# ----------------------------------------------------------
# Edge topology — must match RAW_EDGES in network.ts exactly
# ----------------------------------------------------------
_RAW_EDGES: list[tuple[int, int, int]] = [
    (0,   0,  1),
    (1,   0,  2),
    (2,   1,  0),
    (3,   1,  5),
    (4,   2,  0),
    (5,   2,  3),
    (6,   2, 11),
    (7,   3,  2),
    (8,   3,  4),
    (9,   3, 10),
    (10,  4,  3),
    (11,  4,  8),
    (12,  4,  9),
    (13,  5,  1),
    (14,  5,  7),
    (15,  6,  7),
    (16,  6, 17),
    (17,  7,  5),
    (18,  7,  6),
    (19,  7,  8),
    (20,  7, 15),
    (21,  8,  4),
    (22,  8,  7),
    (23,  8,  9),
    (24,  9,  4),
    (25,  9,  8),
    (26,  9, 10),
    (27,  9, 15),
    (28,  9, 16),
    (29, 10,  3),
    (30, 10,  9),
    (31, 10, 11),
    (32, 10, 13),
    (33, 11,  2),
    (34, 11, 10),
    (35, 11, 12),
    (36, 12, 11),
    (37, 12, 23),
    (38, 13, 10),
    (39, 13, 14),
    (40, 13, 22),
    (41, 14, 13),
    (42, 14, 18),
    (43, 14, 21),
    (44, 15,  7),
    (45, 15,  9),
    (46, 15, 16),
    (47, 15, 17),
    (48, 16,  9),
    (49, 16, 15),
    (50, 16, 18),
    (51, 17,  6),
    (52, 17, 15),
    (53, 17, 19),
    (54, 18, 14),
    (55, 18, 16),
    (56, 18, 19),
    (57, 19, 17),
    (58, 19, 18),
    (59, 19, 20),
    (60, 19, 21),
    (61, 20, 19),
    (62, 20, 21),
    (63, 20, 23),
    (64, 21, 14),
    (65, 21, 19),
    (66, 21, 20),
    (67, 21, 22),
    (68, 22, 13),
    (69, 22, 21),
    (70, 22, 23),
    (71, 23, 12),
    (72, 23, 20),
    (73, 23, 22),
    # UNCERTAIN edges (part of v1 topology):
    (74,  5,  6),
    (75,  6,  5),
]


def _euclidean(a: Node, b: Node) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return math.sqrt(dx * dx + dy * dy)


EDGES: list[Edge] = [
    Edge(id=eid, source=src, target=tgt, weight=_euclidean(NODES[src], NODES[tgt]))
    for eid, src, tgt in _RAW_EDGES
]

# ----------------------------------------------------------
# Adjacency: node_id → list of outgoing edge ids (in edge order)
# ----------------------------------------------------------
ADJACENCY: dict[int, list[int]] = {i: [] for i in range(24)}
for e in EDGES:
    ADJACENCY[e.source].append(e.id)

# Convenience: edge lookup by (source, target)
_EDGE_LOOKUP: dict[tuple[int, int], Edge] = {(e.source, e.target): e for e in EDGES}

# Sorted node IDs
ALL_NODE_IDS: list[int] = list(range(24))
CHARGING_STATION_IDS: list[int] = sorted(_CHARGING_STATIONS)


def find_edge(source: int, target: int) -> Edge | None:
    """Return the edge from source → target, or None if it doesn't exist."""
    return _EDGE_LOOKUP.get((source, target))
