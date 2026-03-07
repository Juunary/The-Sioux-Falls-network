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
