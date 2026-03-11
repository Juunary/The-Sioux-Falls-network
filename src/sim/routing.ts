// ============================================================
// Routing decision layer
// Replaces advanceBusRoute() in store.ts with a context-aware version.
//
// Route State Invariant (must be maintained by ALL branches):
//   IF bus.currentEdge != null:
//     route[] contains only nodes AFTER currentEdge.target
//     (i.e. route[0] !== currentEdge.target)
//   IF bus.currentEdge == null:
//     route[0] is the next node to move toward (if route is non-empty)
//     empty route → bus is idle and needs a new decision
// ============================================================

import type { Bus, SimSettings, DecisionContext } from '../types/network';
import { NODES, EDGES, ADJACENCY } from '../data/network';
import { ALL_NODE_IDS, CHARGING_STATION_IDS, randomOutgoingEdge } from '../utils/graph';
import {
  dijkstra,
  reconstructPath,
  nearestChargingStation,
  insertionHeuristicTour,
} from '../utils/pathfinding';
import { fisherYatesSample } from '../utils/rng';
import {
  getPpoAction,
  consumePpoAction,
  isModelLoaded,
  schedulePpoInference,
} from '../utils/onnxInference';

// ---- Internal helpers ----

function findEdge(source: number, target: number) {
  return EDGES.find((e) => e.source === source && e.target === target) ?? null;
}

/** Apply route invariant: set currentEdge to the first hop, route to the remainder. */
function startRoute(bus: Bus, path: number[], destination: number): Bus {
  // path = [currentNode, node1, node2, ..., dest]
  if (path.length < 2) return { ...bus, state: 'idle' };
  const firstHop = path[1];
  const edge = findEdge(path[0], firstHop);
  if (!edge) return { ...bus, state: 'idle' };
  return {
    ...bus,
    state: 'moving',
    currentEdge: edge.id,
    progress: 0,
    // Route invariant: path.slice(2) — firstHop is currentEdge.target, excluded
    route: path.slice(2),
    destination,
  };
}

// ---- Emergency charge routing ----

/** Route bus directly to nearest charging station. Respects route invariant. */
function routeToNearestCharger(bus: Bus): Bus {
  const result = nearestChargingStation(
    bus.currentNode, CHARGING_STATION_IDS, ALL_NODE_IDS, ADJACENCY, EDGES,
  );
  if (!result || result.path.length < 2) return { ...bus, state: 'idle' };
  return startRoute(bus, result.path, result.nodeId);
}

// ---- Mode-specific destination decisions ----

function randomDecision(bus: Bus, context: DecisionContext): Bus {
  const edge = randomOutgoingEdge(bus.currentNode);
  if (!edge) return { ...bus, state: 'idle' };
  return {
    ...bus,
    state: 'moving',
    currentEdge: edge.id,
    progress: 0,
    route: [],
    destination: edge.target,
  };
}

function shortestDecision(bus: Bus, context: DecisionContext): Bus {
  const dest = Math.floor(context.rng() * NODES.length);
  const { prev } = dijkstra(bus.currentNode, ALL_NODE_IDS, ADJACENCY, EDGES);
  const path = reconstructPath(bus.currentNode, dest, prev, EDGES);
  if (path.length < 2) return { ...bus, state: 'idle' };
  return startRoute(bus, path, dest);
}

function insertionDecision(bus: Bus, context: DecisionContext): Bus {
  const NUM_WAYPOINTS = 4;
  const candidates = ALL_NODE_IDS.filter((id) => id !== bus.currentNode);
  const waypoints = fisherYatesSample(candidates, NUM_WAYPOINTS, context.rng);
  const ordered = insertionHeuristicTour(bus.currentNode, waypoints, ALL_NODE_IDS, ADJACENCY, EDGES);

  let fullPath: number[] = [bus.currentNode];
  let cur = bus.currentNode;
  for (const wp of ordered) {
    const { prev } = dijkstra(cur, ALL_NODE_IDS, ADJACENCY, EDGES);
    const seg = reconstructPath(cur, wp, prev, EDGES);
    if (seg.length > 1) fullPath = [...fullPath, ...seg.slice(1)];
    cur = wp;
  }

  if (fullPath.length < 2) return { ...bus, state: 'idle' };
  return startRoute(bus, fullPath, fullPath[fullPath.length - 1]);
}

function greedyDecision(bus: Bus, context: DecisionContext): Bus {
  // Demand-aware greedy: pick node with most waiting passengers,
  // excluding current node and unreachable nodes.
  // Ties broken by node id ascending (deterministic).
  const waitingByNode = new Map<number, number>();
  for (const p of context.passengers) {
    if (p.state === 'waiting') {
      waitingByNode.set(p.currentNode, (waitingByNode.get(p.currentNode) ?? 0) + 1);
    }
  }

  const { dist } = dijkstra(bus.currentNode, ALL_NODE_IDS, ADJACENCY, EDGES);

  let bestNode = -1;
  let bestScore = -Infinity;
  for (const nodeId of ALL_NODE_IDS) {
    if (nodeId === bus.currentNode) continue;
    const reachable = (dist.get(nodeId) ?? Infinity) < Infinity;
    if (!reachable) continue;
    // Score = waiting passengers at node (fleet concentration penalty added in Python env)
    const score = waitingByNode.get(nodeId) ?? 0;
    if (score > bestScore || (score === bestScore && nodeId < bestNode)) {
      bestScore = score;
      bestNode = nodeId;
    }
  }

  if (bestNode === -1) return randomDecision(bus, context);

  const { prev } = dijkstra(bus.currentNode, ALL_NODE_IDS, ADJACENCY, EDGES);
  const path = reconstructPath(bus.currentNode, bestNode, prev, EDGES);
  if (path.length < 2) return { ...bus, state: 'idle' };
  return startRoute(bus, path, bestNode);
}

function ppoDecision(bus: Bus, context: DecisionContext): Bus {
  // 1. No model loaded → greedy (only valid fallback case).
  if (!isModelLoaded()) return greedyDecision(bus, context);

  // 2. Cached action exists for THIS snapshot (busId + currentNode).
  //    Consume it immediately so it is never reused at a different node.
  const action = getPpoAction(bus.id, bus.currentNode);
  if (action !== null && action !== bus.currentNode) {
    const { prev } = dijkstra(bus.currentNode, ALL_NODE_IDS, ADJACENCY, EDGES);
    const path = reconstructPath(bus.currentNode, action, prev, EDGES);
    if (path.length >= 2) {
      consumePpoAction(bus.id, bus.currentNode);
      return startRoute(bus, path, action);
    }
  }

  // 3. Model loaded but no result yet for this snapshot.
  //    Schedule inference and return the bus in rerouting state (no route, no currentEdge)
  //    so decideNextRoute() retries this decision next tick.
  void schedulePpoInference([bus], context);
  return {
    ...bus,
    state: 'rerouting',
    route: [],
    currentEdge: null,
    progress: 0,
    destination: null,
  };
}

// ============================================================
// Public API
// ============================================================

/**
 * Decide the next route for a bus that is idle or rerouting.
 *
 * Decision order:
 *  1. Charge in place if at a charging station with low battery
 *  2. Battery emergency → route to nearest charger (no PPO step)
 *  3. Continue existing route[] if non-empty
 *  4. Choose new destination based on routingMode
 *
 * NOTE: rerouting is treated identically to idle (MVP simplification).
 */
export function decideNextRoute(
  bus: Bus,
  context: DecisionContext,
  settings: SimSettings,
): Bus {
  const { chargingEnabled, lowBatteryThreshold, chargeDuration } = settings;
  const node = NODES[bus.currentNode];

  // 1. Charge in place
  if (
    chargingEnabled &&
    bus.state !== 'charging' &&
    node.isChargingStation &&
    bus.battery <= lowBatteryThreshold + 5
  ) {
    return {
      ...bus,
      state: 'charging',
      currentEdge: null,
      progress: 0,
      route: [],
      destination: null,
      chargeTimeLeft: chargeDuration,
    };
  }

  // 2. Battery emergency → charger (no PPO step consumed)
  if (chargingEnabled && bus.battery < lowBatteryThreshold && !node.isChargingStation) {
    return routeToNearestCharger(bus);
  }

  // 3. Continue existing planned route
  if (bus.route.length > 0) {
    const nextNode = bus.route[0];
    const edge = findEdge(bus.currentNode, nextNode);
    if (edge) {
      return {
        ...bus,
        state: 'moving',
        currentEdge: edge.id,
        progress: 0,
        route: bus.route.slice(1),
      };
    }
  }

  // 4. Choose new destination
  switch (settings.routingMode) {
    case 'random':    return randomDecision(bus, context);
    case 'shortest':  return shortestDecision(bus, context);
    case 'insertion': return insertionDecision(bus, context);
    case 'greedy':    return greedyDecision(bus, context);
    case 'ppo':       return ppoDecision(bus, context);
    default:          return randomDecision(bus, context);
  }
}
