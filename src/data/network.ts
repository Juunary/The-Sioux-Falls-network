// ============================================================
// Sioux Falls Transportation Network
// Source: LeBlanc, Morlok & Pierskalla (1975)
// 24 nodes, 76 directed links (74 confirmed + 2 marked uncertain)
//
// Node IDs are 0-indexed throughout (standard 1-indexed labels - 1).
// Charging stations: nodes 2, 4, 7, 10, 14, 16, 19, 23 (0-indexed)
//   → marked yellow in the reference image.
//
// Coordinates are scaled to a 960 × 800 SVG canvas.
// Edit NODE_COORDS to reposition nodes; edge topology is separate.
// ============================================================

import type { Node, Edge } from '../types/network';

// ----------------------------------------------------------
// Node coordinates (edit here to reposition)
// ----------------------------------------------------------
// Layout matches the standard Sioux Falls depiction:
//   Row y=760: nodes 0–4  (bottom)
//   Row y=590: nodes 5–9
//   Row y=420: nodes 10–14
//   Row y=250: nodes 15–19
//   Row y=80:  nodes 20–23 (top, 4 nodes)
// ----------------------------------------------------------
const NODE_COORDS: Record<number, { x: number; y: number }> = {
  0:  { x: 90,  y: 760 },
  1:  { x: 290, y: 760 },
  2:  { x: 480, y: 840 }, // slightly lower — matches image
  3:  { x: 680, y: 760 },
  4:  { x: 880, y: 800 },
  5:  { x: 60,  y: 590 },
  6:  { x: 230, y: 590 },
  7:  { x: 430, y: 590 },
  8:  { x: 630, y: 590 },
  9:  { x: 830, y: 590 },
  10: { x: 60,  y: 420 },
  11: { x: 230, y: 420 },
  12: { x: 430, y: 420 },
  13: { x: 630, y: 420 },
  14: { x: 830, y: 420 },
  15: { x: 60,  y: 250 },
  16: { x: 230, y: 250 },
  17: { x: 430, y: 250 },
  18: { x: 630, y: 250 },
  19: { x: 830, y: 250 },
  20: { x: 230, y: 80  },
  21: { x: 430, y: 80  },
  22: { x: 630, y: 80  },
  23: { x: 830, y: 80  },
};

const CHARGING_STATIONS = new Set([2, 4, 7, 10, 14, 16, 19, 23]);

// ----------------------------------------------------------
// Node definitions
// ----------------------------------------------------------
export const NODES: Node[] = Array.from({ length: 24 }, (_, i) => ({
  id: i,
  x: NODE_COORDS[i].x,
  y: NODE_COORDS[i].y,
  isChargingStation: CHARGING_STATIONS.has(i),
  label: String(i),
}));

// ----------------------------------------------------------
// Edge definitions
// [id, source, target]  — all 0-indexed
// Edge IDs 0–73 map to the 74 confirmed standard Sioux Falls links.
// UNCERTAIN: edges 74 and 75 (marked below) — topology is plausible
//   but could not be fully confirmed from the reference image alone.
// ----------------------------------------------------------
const RAW_EDGES: [number, number, number][] = [
  // id  src  tgt
  [0,   0,   1],  // 1→2
  [1,   0,   2],  // 1→3
  [2,   1,   0],  // 2→1
  [3,   1,   5],  // 2→6
  [4,   2,   0],  // 3→1
  [5,   2,   3],  // 3→4
  [6,   2,  11],  // 3→12
  [7,   3,   2],  // 4→3
  [8,   3,   4],  // 4→5
  [9,   3,  10],  // 4→11
  [10,  4,   3],  // 5→4
  [11,  4,   8],  // 5→9
  [12,  4,   9],  // 5→10
  [13,  5,   1],  // 6→2
  [14,  5,   7],  // 6→8
  [15,  6,   7],  // 7→8
  [16,  6,  17],  // 7→18
  [17,  7,   5],  // 8→6
  [18,  7,   6],  // 8→7
  [19,  7,   8],  // 8→9
  [20,  7,  15],  // 8→16
  [21,  8,   4],  // 9→5
  [22,  8,   7],  // 9→8
  [23,  8,   9],  // 9→10
  [24,  9,   4],  // 10→5
  [25,  9,   8],  // 10→9
  [26,  9,  10],  // 10→11
  [27,  9,  15],  // 10→16
  [28,  9,  16],  // 10→17
  [29, 10,   3],  // 11→4
  [30, 10,   9],  // 11→10
  [31, 10,  11],  // 11→12
  [32, 10,  13],  // 11→14
  [33, 11,   2],  // 12→3
  [34, 11,  10],  // 12→11
  [35, 11,  12],  // 12→13
  [36, 12,  11],  // 13→12
  [37, 12,  23],  // 13→24
  [38, 13,  10],  // 14→11
  [39, 13,  14],  // 14→15
  [40, 13,  22],  // 14→23
  [41, 14,  13],  // 15→14
  [42, 14,  18],  // 15→19
  [43, 14,  21],  // 15→22
  [44, 15,   7],  // 16→8
  [45, 15,   9],  // 16→10
  [46, 15,  16],  // 16→17
  [47, 15,  17],  // 16→18
  [48, 16,   9],  // 17→10
  [49, 16,  15],  // 17→16
  [50, 16,  18],  // 17→19
  [51, 17,   6],  // 18→7
  [52, 17,  15],  // 18→16
  [53, 17,  19],  // 18→20
  [54, 18,  14],  // 19→15
  [55, 18,  16],  // 19→17
  [56, 18,  19],  // 19→20
  [57, 19,  17],  // 20→18
  [58, 19,  18],  // 20→19
  [59, 19,  20],  // 20→21
  [60, 19,  21],  // 20→22
  [61, 20,  19],  // 21→20
  [62, 20,  21],  // 21→22
  [63, 20,  23],  // 21→24
  [64, 21,  14],  // 22→15
  [65, 21,  19],  // 22→20
  [66, 21,  20],  // 22→21
  [67, 21,  22],  // 22→23
  [68, 22,  13],  // 23→14
  [69, 22,  21],  // 23→22
  [70, 22,  23],  // 23→24
  [71, 23,  12],  // 24→13
  [72, 23,  20],  // 24→21
  [73, 23,  22],  // 24→23
  // UNCERTAIN: The following two edges complete the 76-link standard network.
  // They connect node 5 (0-idx) to node 6 (0-idx) bidirectionally.
  // Present in some published versions; may be absent in others.
  [74,  5,   6],  // UNCERTAIN: 6→7
  [75,  6,   5],  // UNCERTAIN: 7→6
];

// ----------------------------------------------------------
// Build edges with computed Euclidean weights
// ----------------------------------------------------------
function dist(a: Node, b: Node): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

export const EDGES: Edge[] = RAW_EDGES.map(([id, source, target]) => ({
  id,
  source,
  target,
  weight: dist(NODES[source], NODES[target]),
}));

// ----------------------------------------------------------
// Adjacency map: nodeId → list of outgoing edge ids
// ----------------------------------------------------------
export const ADJACENCY: Map<number, number[]> = new Map();
NODES.forEach((n) => ADJACENCY.set(n.id, []));
EDGES.forEach((e) => {
  ADJACENCY.get(e.source)!.push(e.id);
});
