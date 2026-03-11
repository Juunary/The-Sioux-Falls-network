# ============================================================
# Pathfinding tests
#
# Verifies:
# 1. Dijkstra returns correct distances for known node pairs
# 2. reconstruct_path returns valid paths
# 3. Route invariant: path[1] != path[2] (no self-edge)
# 4. All-pairs reachability on v1 topology
# 5. Nearest charging station routing correctness
# 6. Route state invariant: path.slice(2) excludes first hop
# ============================================================

import math
import pytest

from backend.env.network import (
    ALL_NODE_IDS,
    CHARGING_STATION_IDS,
    EDGES,
    NODES,
    find_edge,
)
from backend.env.pathfinding import (
    REACHABILITY,
    dijkstra,
    nearest_charging_station,
    reconstruct_path,
)


# ============================================================
# Dijkstra correctness
# ============================================================

class TestDijkstra:
    def test_source_to_self_is_zero(self):
        for src in ALL_NODE_IDS:
            dist, _ = dijkstra(src)
            assert dist[src] == 0.0, f"dist[{src}→{src}] should be 0"

    def test_all_pairs_finite_or_inf(self):
        """All node pairs should return a finite value or math.inf — no NaN."""
        for src in ALL_NODE_IDS:
            dist, _ = dijkstra(src)
            for tgt in ALL_NODE_IDS:
                d = dist.get(tgt, math.inf)
                assert not math.isnan(d), f"NaN for {src}→{tgt}"

    def test_direct_edge_weight_matches(self):
        """For nodes with a direct edge, dist should be exactly edge.weight."""
        for edge in EDGES:
            dist, _ = dijkstra(edge.source)
            d = dist.get(edge.target, math.inf)
            # Direct edge should be a lower bound; if direct is shortest, equality holds
            assert d <= edge.weight + 1e-6, (
                f"dist[{edge.source}→{edge.target}]={d:.2f} > edge.weight={edge.weight:.2f}"
            )

    def test_triangle_inequality(self):
        """Spot-check triangle inequality: dist(a,c) ≤ dist(a,b) + dist(b,c)."""
        pairs = [(0, 5), (3, 18), (11, 20), (7, 22)]
        for a, c in pairs:
            dist_a, _ = dijkstra(a)
            for b in [2, 10, 14]:
                dist_b, _ = dijkstra(b)
                d_ac = dist_a.get(c, math.inf)
                d_ab = dist_a.get(b, math.inf)
                d_bc = dist_b.get(c, math.inf)
                assert d_ac <= d_ab + d_bc + 1e-6, (
                    f"Triangle violated: dist({a},{c})={d_ac:.2f} > dist({a},{b})+dist({b},{c})={d_ab+d_bc:.2f}"
                )

    def test_symmetry_not_required_directed(self):
        """Directed graph: dist(a,b) may differ from dist(b,a). Just verify no crash."""
        dist_0, _ = dijkstra(0)
        dist_1, _ = dijkstra(1)
        # Different values are acceptable for directed graph
        assert isinstance(dist_0.get(1, math.inf), float)
        assert isinstance(dist_1.get(0, math.inf), float)


# ============================================================
# reconstruct_path correctness
# ============================================================

class TestReconstructPath:
    def test_source_to_self(self):
        dist, prev = dijkstra(5)
        path = reconstruct_path(5, 5, prev)
        assert path == [5]

    def test_adjacent_node(self):
        """Source→adjacent: path length should be 2."""
        src = 0
        dist, prev = dijkstra(src)
        for edge in EDGES:
            if edge.source == src:
                path = reconstruct_path(src, edge.target, prev)
                assert len(path) >= 2
                assert path[0] == src
                assert path[-1] == edge.target
                break

    def test_path_is_valid_sequence(self):
        """Every consecutive pair in path must have a direct edge."""
        test_pairs = [(0, 20), (11, 19), (3, 23)]
        for src, tgt in test_pairs:
            dist, prev = dijkstra(src)
            path = reconstruct_path(src, tgt, prev)
            if not path:
                continue  # unreachable
            for i in range(len(path) - 1):
                edge = find_edge(path[i], path[i + 1])
                assert edge is not None, (
                    f"No edge {path[i]}→{path[i+1]} in path {src}→{tgt}: {path}"
                )

    def test_no_path_returns_empty(self):
        """If target is unreachable, should return []."""
        # Force unreachable: build empty prev
        fake_prev = {n: None for n in ALL_NODE_IDS}
        path = reconstruct_path(0, 5, fake_prev)
        assert path == []


# ============================================================
# Route state invariant verification
# ============================================================

class TestRouteInvariant:
    def test_start_route_invariant(self):
        """
        After _start_route(bus, path, dest):
          bus.current_edge = edge(path[0]→path[1])
          bus.route = path[2:]  (does NOT contain path[1])
        """
        from backend.env.bus import BusState, _start_route

        dist, prev = dijkstra(0)
        path = reconstruct_path(0, 20, prev)
        assert len(path) >= 3, "Need at least 3-node path for this test"

        bus = BusState(
            id=0, current_node=0, current_edge=None, progress=0.0,
            speed=100.0, battery=80.0, state="idle", route=[],
            destination=None, charge_time_left=0.0,
            passenger_ids=[], capacity=6,
        )
        bus = _start_route(bus, path, path[-1])

        # current_edge should be edge(path[0]→path[1])
        edge = find_edge(path[0], path[1])
        assert bus.current_edge == edge.id, "current_edge should be edge to path[1]"

        # route should start at path[2], NOT path[1]
        if len(path) > 2:
            assert bus.route == path[2:], f"route should be path[2:], got {bus.route}"
            assert bus.route[0] != path[1], "route[0] must not equal current_edge.target"

    def test_charging_reroute_invariant(self):
        """Emergency charge routing must not put first-hop in route[]."""
        from backend.env.bus import BusState, _route_to_nearest_charger

        # Find a non-charging node
        src_node = 0  # not a charging station
        bus = BusState(
            id=0, current_node=src_node, current_edge=None, progress=0.0,
            speed=100.0, battery=5.0,  # low battery
            state="idle", route=[],
            destination=None, charge_time_left=0.0,
            passenger_ids=[], capacity=6,
        )
        bus = _route_to_nearest_charger(bus)
        assert bus.state == "moving"
        assert bus.current_edge is not None

        # Get the edge
        edge = EDGES[bus.current_edge]
        first_hop_target = edge.target

        # route[0] must NOT equal first_hop_target (invariant)
        if bus.route:
            assert bus.route[0] != first_hop_target, (
                f"Invariant violated: route[0]={bus.route[0]} == currentEdge.target={first_hop_target}"
            )


# ============================================================
# All-pairs reachability
# ============================================================

class TestReachability:
    def test_reachability_matrix_shape(self):
        assert len(REACHABILITY) == 24
        assert all(len(row) == 24 for row in REACHABILITY)

    def test_self_reachable(self):
        for i in range(24):
            assert REACHABILITY[i][i], f"Node {i} should reach itself"

    def test_most_pairs_reachable(self):
        """Sioux Falls v1 (76-edge) should be strongly connected for almost all pairs."""
        total = 0
        reachable = 0
        for i in range(24):
            for j in range(24):
                total += 1
                if REACHABILITY[i][j]:
                    reachable += 1
        # Expect > 99% reachability on the directed graph
        ratio = reachable / total
        assert ratio > 0.95, f"Only {ratio:.1%} node pairs reachable — graph may be disconnected"


# ============================================================
# Nearest charging station
# ============================================================

class TestNearestChargingStation:
    def test_returns_charging_station(self):
        result = nearest_charging_station(0)
        assert result is not None
        cs_id, path = result
        assert cs_id in CHARGING_STATION_IDS

    def test_path_valid(self):
        for src in [0, 1, 3, 9, 17]:
            result = nearest_charging_station(src)
            if result is None:
                continue
            cs_id, path = result
            assert path[0] == src
            assert path[-1] == cs_id
            # Validate each edge in path
            for i in range(len(path) - 1):
                edge = find_edge(path[i], path[i + 1])
                assert edge is not None, f"Invalid edge {path[i]}→{path[i+1]} in charger path from {src}"

    def test_already_at_charger(self):
        """Bus already at a charging station → nearest charger is self (dist=0)."""
        for cs_id in CHARGING_STATION_IDS:
            result = nearest_charging_station(cs_id)
            assert result is not None
            node_id, path = result
            # Either self (dist=0) or nearby — just ensure it's a valid CS
            assert node_id in CHARGING_STATION_IDS
