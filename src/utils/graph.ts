// ============================================================
// Graph helper utilities
// ============================================================

import type { Node, Edge } from '../types/network';
import { NODES, EDGES, ADJACENCY } from '../data/network';
import { simRng } from './rng';

/** Get a node by id — throws if not found */
export function getNode(id: number): Node {
  const n = NODES[id];
  if (!n) throw new Error(`Node ${id} not found`);
  return n;
}

/** Get an edge by id — throws if not found */
export function getEdge(id: number): Edge {
  const e = EDGES[id];
  if (!e) throw new Error(`Edge ${id} not found`);
  return e;
}

/** List outgoing edges from a node */
export function outgoingEdges(nodeId: number): Edge[] {
  return (ADJACENCY.get(nodeId) ?? []).map((eid) => EDGES[eid]);
}

/** Pick a random outgoing edge from a node. Returns null if dead-end. */
export function randomOutgoingEdge(nodeId: number): Edge | null {
  const edges = outgoingEdges(nodeId);
  if (edges.length === 0) return null;
  return edges[Math.floor(simRng() * edges.length)];
}

/** Interpolate a point along an edge at fraction t (0–1) */
export function interpolateEdge(edge: Edge, t: number): { x: number; y: number } {
  const src = NODES[edge.source];
  const tgt = NODES[edge.target];
  return {
    x: src.x + (tgt.x - src.x) * t,
    y: src.y + (tgt.y - src.y) * t,
  };
}

/**
 * Compute a perpendicular offset position on edge at fraction t.
 * `laneOffset` is signed pixels — used to separate buses on same edge.
 */
export function interpolateEdgeWithOffset(
  edge: Edge,
  t: number,
  laneOffset: number,
): { x: number; y: number } {
  const src = NODES[edge.source];
  const tgt = NODES[edge.target];
  const dx = tgt.x - src.x;
  const dy = tgt.y - src.y;
  const len = Math.sqrt(dx * dx + dy * dy) || 1;
  // Perpendicular unit vector (rotate 90°)
  const px = -dy / len;
  const py = dx / len;

  const base = interpolateEdge(edge, t);
  return {
    x: base.x + px * laneOffset,
    y: base.y + py * laneOffset,
  };
}

/** All node IDs */
export const ALL_NODE_IDS = NODES.map((n) => n.id);

/** IDs of charging stations */
export const CHARGING_STATION_IDS = NODES.filter((n) => n.isChargingStation).map((n) => n.id);
