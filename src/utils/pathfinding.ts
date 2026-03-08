// ============================================================
// Dijkstra's shortest-path algorithm on the directed network
// ============================================================

import type { Edge } from '../types/network';

interface DijkstraResult {
  /** dist[nodeId] = shortest distance from source */
  dist: Map<number, number>;
  /** prev[nodeId] = the edge id used to reach nodeId on shortest path */
  prev: Map<number, number | null>;
}

/**
 * Run Dijkstra from `sourceNode` over the given directed edges.
 * Returns distance and predecessor maps.
 */
export function dijkstra(
  sourceNode: number,
  allNodeIds: number[],
  adjacency: Map<number, number[]>,
  edges: Edge[],
): DijkstraResult {
  const dist = new Map<number, number>();
  const prev = new Map<number, number | null>();
  // Simple priority queue via sorted array (fine for 24 nodes)
  const visited = new Set<number>();

  for (const id of allNodeIds) {
    dist.set(id, Infinity);
    prev.set(id, null);
  }
  dist.set(sourceNode, 0);

  const queue = [...allNodeIds];

  while (queue.length > 0) {
    // Pick unvisited node with smallest tentative distance
    queue.sort((a, b) => (dist.get(a) ?? Infinity) - (dist.get(b) ?? Infinity));
    const u = queue.shift()!;
    if (visited.has(u)) continue;
    visited.add(u);

    const outgoing = adjacency.get(u) ?? [];
    for (const edgeId of outgoing) {
      const edge = edges[edgeId];
      const v = edge.target;
      if (visited.has(v)) continue;
      const alt = (dist.get(u) ?? Infinity) + edge.weight;
      if (alt < (dist.get(v) ?? Infinity)) {
        dist.set(v, alt);
        prev.set(v, edgeId);
      }
    }
  }

  return { dist, prev };
}

/**
 * Reconstruct an ordered array of node IDs from source to target
 * using the predecessor map returned by dijkstra().
 * Returns [] if no path exists.
 */
export function reconstructPath(
  source: number,
  target: number,
  prev: Map<number, number | null>,
  edges: Edge[],
): number[] {
  const path: number[] = [];
  let current = target;

  while (current !== source) {
    const edgeId = prev.get(current);
    if (edgeId == null) return []; // no path
    const edge = edges[edgeId];
    path.unshift(current);
    current = edge.source;
  }
  path.unshift(source);
  return path;
}

/**
 * Nearest-Insertion Heuristic for multi-stop routing.
 *
 * Given a start node and a list of waypoints to visit, returns an ordered
 * sequence of those waypoints (excluding start) that approximates the
 * minimum total travel distance using shortest-path distances.
 *
 * Algorithm:
 *  1. Compute Dijkstra distances from start and each waypoint.
 *  2. Seed tour with the waypoint nearest to start.
 *  3. Repeat: find the unvisited waypoint nearest to any node already
 *     in the tour, then insert it at the position with minimum
 *     insertion cost  d(i,k) + d(k,j) − d(i,j).
 */
export function insertionHeuristicTour(
  start: number,
  waypoints: number[],
  allNodeIds: number[],
  adjacency: Map<number, number[]>,
  edges: Edge[],
): number[] {
  if (waypoints.length === 0) return [];
  if (waypoints.length === 1) return [...waypoints];

  // Pre-compute shortest-path distances from start and every waypoint
  const distFrom = new Map<number, Map<number, number>>();
  for (const src of [start, ...waypoints]) {
    distFrom.set(src, dijkstra(src, allNodeIds, adjacency, edges).dist);
  }
  const d = (a: number, b: number): number =>
    distFrom.get(a)?.get(b) ?? Infinity;

  // Seed: waypoint nearest to start
  const remaining = new Set(waypoints);
  let seedNode = waypoints[0];
  let seedDist = d(start, waypoints[0]);
  for (const w of waypoints) {
    const dist = d(start, w);
    if (dist < seedDist) { seedDist = dist; seedNode = w; }
  }
  const tour: number[] = [seedNode];
  remaining.delete(seedNode);

  // Nearest-insertion loop
  while (remaining.size > 0) {
    // Find uninserted waypoint k nearest to any node in [start, ...tour]
    const tourNodes = [start, ...tour];
    let bestK = -1;
    let bestKDist = Infinity;
    for (const k of remaining) {
      for (const t of tourNodes) {
        const dist = d(t, k);
        if (dist < bestKDist) { bestKDist = dist; bestK = k; }
      }
    }
    if (bestK === -1) break;

    // Find cheapest insertion position
    // fullSeq[i] → fullSeq[i+1] becomes fullSeq[i] → bestK → fullSeq[i+1]
    // Inserting between fullSeq[i] and fullSeq[i+1] maps to tour.splice(i, 0, bestK)
    const fullSeq = [start, ...tour];
    let bestInsertIdx = tour.length; // default: append at end
    let bestCost = d(fullSeq[fullSeq.length - 1], bestK);

    for (let i = 0; i < fullSeq.length - 1; i++) {
      const cost =
        d(fullSeq[i], bestK) +
        d(bestK, fullSeq[i + 1]) -
        d(fullSeq[i], fullSeq[i + 1]);
      if (cost < bestCost) { bestCost = cost; bestInsertIdx = i; }
    }

    tour.splice(bestInsertIdx, 0, bestK);
    remaining.delete(bestK);
  }

  return tour;
}

/**
 * Find the nearest reachable charging station from `fromNode`.
 * Returns { nodeId, path } or null if none reachable.
 */
export function nearestChargingStation(
  fromNode: number,
  chargingStationIds: number[],
  allNodeIds: number[],
  adjacency: Map<number, number[]>,
  edges: Edge[],
): { nodeId: number; path: number[] } | null {
  if (chargingStationIds.length === 0) return null;

  const { dist, prev } = dijkstra(fromNode, allNodeIds, adjacency, edges);

  let bestNode = -1;
  let bestDist = Infinity;
  for (const csId of chargingStationIds) {
    const d = dist.get(csId) ?? Infinity;
    if (d < bestDist) {
      bestDist = d;
      bestNode = csId;
    }
  }

  if (bestNode === -1 || bestDist === Infinity) return null;

  const path = reconstructPath(fromNode, bestNode, prev, edges);
  return path.length > 0 ? { nodeId: bestNode, path } : null;
}
