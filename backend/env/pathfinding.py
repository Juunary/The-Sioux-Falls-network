# ============================================================
# Pathfinding — Python mirror of src/utils/pathfinding.ts
#
# Tie-break rules (must match TypeScript exactly):
#   Dijkstra: equal-distance nodes → smaller node_id wins.
#   heapq pushes (dist, node_id) tuples; Python tuple comparison
#   naturally applies node_id as secondary key — matches TS behaviour
#   where queue.sort() is stable and leftmost minimum wins.
#
# Insertion heuristic tie-breaks:
#   nearest-k tie → smallest node id
#   cheapest insertion position tie → smallest (lower) index
# ============================================================

from __future__ import annotations

import heapq
import math
from typing import Optional

from backend.env.network import (
    ADJACENCY,
    ALL_NODE_IDS,
    CHARGING_STATION_IDS,
    EDGES,
    Edge,
    find_edge,
)


# ============================================================
# Dijkstra
# ============================================================

def dijkstra(
    source: int,
    all_node_ids: list[int] | None = None,
) -> tuple[dict[int, float], dict[int, Optional[int]]]:
    """
    Run Dijkstra from `source` on the Sioux Falls graph.

    Returns:
        dist: node_id → shortest distance from source (float)
        prev: node_id → edge_id used to reach it on shortest path (None = no predecessor)

    Tie-break: (dist, node_id) min-heap → smaller node_id wins on equal distance.
    This matches TS: sort()-based queue where smaller node ids appear first among
    equal-distance entries (stable sort, lower-id nodes inserted first).
    """
    if all_node_ids is None:
        all_node_ids = ALL_NODE_IDS

    dist: dict[int, float] = {n: math.inf for n in all_node_ids}
    prev: dict[int, Optional[int]] = {n: None for n in all_node_ids}
    dist[source] = 0.0

    # heap entries: (distance, node_id)
    heap: list[tuple[float, int]] = [(0.0, source)]
    visited: set[int] = set()

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)

        for edge_id in ADJACENCY.get(u, []):
            edge = EDGES[edge_id]
            v = edge.target
            if v in visited:
                continue
            alt = d + edge.weight
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = edge_id
                heapq.heappush(heap, (alt, v))

    return dist, prev


def reconstruct_path(
    source: int,
    target: int,
    prev: dict[int, Optional[int]],
) -> list[int]:
    """
    Reconstruct the node-id path from source to target using the
    predecessor map from dijkstra(). Returns [] if no path exists.
    """
    path: list[int] = []
    current = target

    while current != source:
        edge_id = prev.get(current)
        if edge_id is None:
            return []
        edge = EDGES[edge_id]
        path.append(current)
        current = edge.source

    path.append(source)
    path.reverse()
    return path


# ============================================================
# Nearest charging station
# ============================================================

def nearest_charging_station(
    from_node: int,
) -> Optional[tuple[int, list[int]]]:
    """
    Find the nearest reachable charging station from `from_node`.

    Returns (node_id, path) or None if unreachable.
    Tie-break: smaller node_id wins (Dijkstra heap secondary key).
    """
    dist, prev = dijkstra(from_node)

    best_node = -1
    best_dist = math.inf

    for cs_id in CHARGING_STATION_IDS:
        d = dist.get(cs_id, math.inf)
        if d < best_dist:
            best_dist = d
            best_node = cs_id

    if best_node == -1 or math.isinf(best_dist):
        return None

    path = reconstruct_path(from_node, best_node, prev)
    return (best_node, path) if path else None


def shortest_distance(src: int, dst: int) -> float:
    """
    Return the shortest path distance in pixels from src to dst.
    Uses Dijkstra on the Sioux Falls graph.
    Returns math.inf if dst is unreachable from src.
    Unit: px (same as edge weights).
    """
    dist_map, _ = dijkstra(src)
    return dist_map.get(dst, math.inf)


# ============================================================
# Nearest-Insertion Heuristic tour
# ============================================================

def insertion_heuristic_tour(
    start: int,
    waypoints: list[int],
) -> list[int]:
    """
    Nearest-insertion heuristic for multi-stop routing.
    Returns ordered waypoints (excluding start).

    Tie-break rules (matching TS insertionHeuristicTour):
      - Nearest k among remaining: smaller node_id wins on equal distance
      - Cheapest insertion position: lower index wins on equal cost
    """
    if not waypoints:
        return []
    if len(waypoints) == 1:
        return list(waypoints)

    # Pre-compute Dijkstra distances from start and each waypoint
    dist_from: dict[int, dict[int, float]] = {}
    for src in [start] + waypoints:
        dist_from[src], _ = dijkstra(src)

    def d(a: int, b: int) -> float:
        return dist_from.get(a, {}).get(b, math.inf)

    # Seed: waypoint nearest to start — tie-break: smallest node id
    remaining = set(waypoints)
    seed_node = min(
        waypoints,
        key=lambda w: (d(start, w), w),
    )
    tour: list[int] = [seed_node]
    remaining.remove(seed_node)

    # Nearest-insertion loop
    while remaining:
        tour_nodes = [start] + tour

        # Find uninserted waypoint k nearest to any node in tour_nodes
        # tie-break: smallest node id
        best_k = min(
            remaining,
            key=lambda k: (min(d(t, k) for t in tour_nodes), k),
        )

        # Find cheapest insertion position in [start, ...tour]
        # Insert best_k at tour index `insert_idx`
        # full_seq[insert_idx] → best_k → full_seq[insert_idx + 1]
        full_seq = [start] + tour
        n_seq = len(full_seq)

        # Default: append at end (after last element of tour)
        best_cost = d(full_seq[-1], best_k)
        best_insert_idx = len(tour)

        for i in range(n_seq - 1):
            cost = d(full_seq[i], best_k) + d(best_k, full_seq[i + 1]) - d(full_seq[i], full_seq[i + 1])
            if cost < best_cost:
                best_cost = cost
                best_insert_idx = i  # insert before tour[i] (i.e. after full_seq[i])

        tour.insert(best_insert_idx, best_k)
        remaining.remove(best_k)

    return tour


# ============================================================
# Reachability matrix (precomputed for action masking)
# ============================================================

def compute_reachability() -> list[list[bool]]:
    """
    Returns a 24×24 boolean matrix: reachable[i][j] = True iff node j is
    reachable from node i (including i itself with dist=0).
    """
    matrix: list[list[bool]] = []
    for src in ALL_NODE_IDS:
        dist, _ = dijkstra(src)
        row = [not math.isinf(dist.get(tgt, math.inf)) for tgt in ALL_NODE_IDS]
        matrix.append(row)
    return matrix


# Precomputed once at import time (24-node graph is trivial to compute)
REACHABILITY: list[list[bool]] = compute_reachability()


def max_dijkstra_dist() -> float:
    """Return the maximum finite shortest-path distance across all node pairs."""
    max_d = 0.0
    for src in ALL_NODE_IDS:
        dist, _ = dijkstra(src)
        for tgt in ALL_NODE_IDS:
            d = dist.get(tgt, math.inf)
            if not math.isinf(d) and d > max_d:
                max_d = d
    return max_d


MAX_DIST: float = max_dijkstra_dist()
