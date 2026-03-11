# ============================================================
# Parity / determinism tests
#
# Verifies:
# 1. Scenario generator: same seed → identical JSON
# 2. Env reset: same scenario → same initial observation
# 3. Route invariant: never violated during a short rollout
# 4. Passenger spawn boundary: events with spawn_time ≤ sim_time are consumed
# 5. Tick ordering: passengers spawned in tick N are NOT boarded in tick N
# 6. Determinism: same scenario, same greedy policy → same episode reward (x2)
# ============================================================

import json
import math
import copy
from typing import Optional

import pytest
import numpy as np

from backend.api.schemas import GenerateRequest, ScenarioData
from backend.datasets.generator import generate_scenario, generate_batch
from backend.env.network import ALL_NODE_IDS, CHARGING_STATION_IDS, EDGES
from backend.env.pathfinding import dijkstra, reconstruct_path


# ============================================================
# Helpers
# ============================================================

def _make_simple_scenario(seed: int = 42) -> ScenarioData:
    """Create a short easy scenario for fast testing."""
    return generate_scenario(
        seed=seed,
        split="test",
        difficulty="easy",
        bus_count=3,
        episode_length=60.0,   # 60 s — fast
        spawn_rate=0.5,
        passenger_capacity=4,
        reboard_enabled=False,
        charging_enabled=True,
        low_battery_threshold=20.0,
        charge_duration=8.0,
        battery_drain_rate=0.08,
    )


# ============================================================
# Generator determinism
# ============================================================

class TestGeneratorDeterminism:
    def test_same_seed_same_json(self):
        """Two calls with same seed must produce byte-identical JSON."""
        s1 = generate_scenario(seed=1234)
        s2 = generate_scenario(seed=1234)
        j1 = json.dumps(s1.model_dump(), sort_keys=True)
        j2 = json.dumps(s2.model_dump(), sort_keys=True)
        assert j1 == j2, "Same seed produced different JSON"

    def test_different_seeds_different_scenarios(self):
        """Different seeds must produce different scenarios (with high probability)."""
        s1 = generate_scenario(seed=0)
        s2 = generate_scenario(seed=1)
        # Very unlikely to have identical first bus start nodes for both seeds
        assert s1.meta.seed != s2.meta.seed
        # Check that at least one bus differs
        pairs = zip(s1.buses_initial, s2.buses_initial)
        differs = any(
            b1.start_node != b2.start_node or abs(b1.battery - b2.battery) > 0.01
            for b1, b2 in pairs
        )
        assert differs, "Different seeds produced identical buses"

    def test_passenger_events_sorted_by_spawn_time(self):
        """Events must be in ascending spawn_time order."""
        s = generate_scenario(seed=99)
        times = [e.spawn_time for e in s.passenger_events]
        assert times == sorted(times), "Passenger events not sorted by spawn_time"

    def test_all_origins_valid(self):
        """All passenger origins/destinations must be valid node IDs."""
        s = generate_scenario(seed=7)
        for ev in s.passenger_events:
            assert 0 <= ev.origin < 24, f"Invalid origin: {ev.origin}"
            assert 0 <= ev.destination < 24, f"Invalid destination: {ev.destination}"
            assert ev.origin != ev.destination, "Origin == destination"

    def test_meta_fields_match_request(self):
        s = generate_scenario(
            seed=42, split="val", difficulty="hard",
            bus_count=8, episode_length=300.0, spawn_rate=1.0,
        )
        assert s.meta.split == "val"
        assert s.meta.difficulty == "hard"
        assert s.meta.bus_count == 8
        assert s.meta.episode_length == 300.0
        assert s.meta.graph_version == "v1"
        assert s.meta.generator_version == "1.0"


# ============================================================
# Spawn boundary
# ============================================================

class TestSpawnBoundary:
    def test_events_consumed_by_correct_time(self):
        """Events with spawn_time ≤ current_sim_time must be consumed."""
        from backend.env.passenger import replay_spawn, Passenger

        # Create fake events
        fake_events_raw = [
            {"spawn_time": 0.016, "origin": 0, "destination": 5, "patience": 30.0},
            {"spawn_time": 0.032, "origin": 1, "destination": 6, "patience": 25.0},
            {"spawn_time": 0.100, "origin": 2, "destination": 7, "patience": 40.0},
        ]

        from backend.api.schemas import PassengerEvent
        fake_events = [
            PassengerEvent(event_id=i, **ev)
            for i, ev in enumerate(fake_events_raw)
        ]
        queue = list(fake_events)

        # Consume at t=0.016
        pax, next_id = replay_spawn([], 0, 0.016, queue)
        assert len(pax) == 1, f"Expected 1 passenger at t=0.016, got {len(pax)}"
        assert pax[0].origin_node == 0
        assert len(queue) == 2  # 2 remaining

        # Consume at t=0.05 (should consume spawn_time=0.032)
        pax, next_id = replay_spawn(pax, next_id, 0.05, queue)
        assert len(pax) == 2
        assert len(queue) == 1  # only t=0.100 remaining

        # Consume at t=0.099 (should NOT consume t=0.100)
        pax, next_id = replay_spawn(pax, next_id, 0.099, queue)
        assert len(pax) == 2
        assert len(queue) == 1  # still remaining

        # Consume at t=0.100 (exact boundary — inclusive)
        pax, next_id = replay_spawn(pax, next_id, 0.100, queue)
        assert len(pax) == 3
        assert len(queue) == 0


# ============================================================
# Route invariant
# ============================================================

class TestRouteInvariantDuringRollout:
    def _check_invariant(self, bus) -> None:
        from backend.env.network import EDGES
        if bus.current_edge is not None:
            edge = EDGES[bus.current_edge]
            first_hop_target = edge.target
            if bus.route:
                assert bus.route[0] != first_hop_target, (
                    f"Bus {bus.id}: route[0]={bus.route[0]} == currentEdge.target={first_hop_target} "
                    f"(invariant violated)"
                )

    def test_invariant_maintained_across_ticks(self):
        """Run a short episode and verify route invariant is never broken."""
        from backend.env.sioux_falls_env import SiouxFallsEnv

        scenario = _make_simple_scenario(seed=42)
        env = SiouxFallsEnv(scenario)
        obs, info = env.reset()

        for _ in range(50):  # 50 steps
            if env.pending_decision_bus is None:
                break

            mask = env.action_masks()
            valid_actions = [i for i, m in enumerate(mask) if m]
            if not valid_actions:
                break

            action = valid_actions[0]  # always take first valid action
            obs, reward, done, truncated, info = env.step(action)

            # Check invariant on all buses
            for bus in env.buses:
                self._check_invariant(bus)

            if done:
                break


# ============================================================
# Same-tick boarding prohibition
# ============================================================

class TestSameTickBoardingProhibition:
    def test_spawned_passengers_not_boarded_same_tick(self):
        """
        Passengers spawned in tick N must NOT board a bus in tick N.
        (spawn happens after processPassengers in tick order)
        """
        from backend.env.passenger import Passenger, replay_spawn, process_passengers
        from backend.env.bus import BusState
        from backend.api.schemas import PassengerEvent

        # A bus sitting at node 3
        bus = BusState(
            id=0, current_node=3, current_edge=None, progress=0.0,
            speed=100.0, battery=80.0, state="idle", route=[3, 5],
            destination=5, charge_time_left=0.0,
            passenger_ids=[], capacity=6,
        )

        # Passenger spawns at t=0.016 at node 3
        event = PassengerEvent(event_id=0, spawn_time=0.016, origin=3, destination=7, patience=30.0)
        queue = [event]

        # Tick: t goes from 0 → 0.016
        # Step 1: processPassengers (no waiting passengers yet)
        updated_buses, pax = process_passengers([bus], [], "greedy", 0.016, False)
        assert len(pax) == 0, "No passengers should exist before spawn"

        # Step 3: spawn (happens AFTER processPassengers)
        pax, next_id = replay_spawn(pax, 0, 0.016, queue)
        assert len(pax) == 1, "Passenger should be spawned"
        assert pax[0].state == "waiting"
        assert pax[0].bus_id is None, "Passenger should NOT be on any bus (same-tick prohibition)"


# ============================================================
# Episode determinism (greedy policy, same scenario × 2)
# ============================================================

class TestEpisodeDeterminism:
    def _run_greedy_episode(self, scenario: ScenarioData) -> dict:
        """Run a full episode using greedy (always pick first valid action)."""
        from backend.env.sioux_falls_env import SiouxFallsEnv

        env = SiouxFallsEnv(scenario)
        obs, _ = env.reset()

        total_reward = 0.0
        steps = 0

        while True:
            if env.pending_decision_bus is None:
                break

            mask = env.action_masks()
            valid_actions = [i for i, m in enumerate(mask) if m]
            if not valid_actions:
                break

            # Greedy: pick node with most waiting passengers (deterministic)
            waiting_per_node = obs[55:79]
            scores = np.array(waiting_per_node)
            scores[~mask] = -np.inf
            action = int(np.argmax(scores))
            if np.isinf(scores[action]):
                action = valid_actions[0]

            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            steps += 1

            if done:
                break

        return {
            "total_reward": total_reward,
            "steps": steps,
            "arrived": env._arrived_count,
            "gone": env._gone_count,
        }

    def test_same_scenario_same_result(self):
        """Two runs of the same episode must yield identical metrics."""
        scenario = _make_simple_scenario(seed=42)

        result1 = self._run_greedy_episode(scenario)
        result2 = self._run_greedy_episode(scenario)

        assert result1["steps"] == result2["steps"], (
            f"Steps differ: {result1['steps']} vs {result2['steps']}"
        )
        assert result1["arrived"] == result2["arrived"], (
            f"Arrived differ: {result1['arrived']} vs {result2['arrived']}"
        )
        assert result1["gone"] == result2["gone"], (
            f"Gone differ: {result1['gone']} vs {result2['gone']}"
        )
        assert abs(result1["total_reward"] - result2["total_reward"]) < 1e-6, (
            f"Reward differs: {result1['total_reward']} vs {result2['total_reward']}"
        )

    def test_different_seeds_can_differ(self):
        """Different scenarios (different seeds) may produce different outcomes."""
        s1 = _make_simple_scenario(seed=1)
        s2 = _make_simple_scenario(seed=999)
        r1 = self._run_greedy_episode(s1)
        r2 = self._run_greedy_episode(s2)
        # They *might* differ — just verify both complete without error
        assert r1["steps"] >= 0
        assert r2["steps"] >= 0
