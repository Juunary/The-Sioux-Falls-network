// ============================================================
// Core domain types for the Sioux Falls network simulator
// ============================================================

export interface Node {
  id: number;
  /** SVG/canvas x coordinate (pixels) */
  x: number;
  /** SVG/canvas y coordinate (pixels) */
  y: number;
  isChargingStation: boolean;
  label?: string;
}

export interface Edge {
  /** Edge ID (0-indexed, matching the link labels in the reference image) */
  id: number;
  source: number; // node id
  target: number; // node id
  /** Euclidean distance derived from node positions (set at init time) */
  weight: number;
}

export type BusState = 'moving' | 'charging' | 'idle' | 'rerouting';

export interface Bus {
  id: number;
  /** Node the bus most recently departed from (or is currently at if idle) */
  currentNode: number;
  /** Edge being traversed, null if idle/charging */
  currentEdge: number | null;
  /** 0–1 fraction along currentEdge */
  progress: number;
  /** Speed in "edge-length-per-second" units (adjusted by sim speed multiplier) */
  speed: number;
  /** Battery level 0–100 */
  battery: number;
  state: BusState;
  /** Planned node route, index 0 is next waypoint */
  route: number[];
  /** Current destination node (for shortest-path mode) */
  destination: number | null;
  /** Color string for rendering */
  color: string;
  /** Remaining charge time in seconds */
  chargeTimeLeft: number;
  /** Small perpendicular offset to prevent visual stacking on same edge */
  laneOffset: number;
}

export type RoutingMode = 'random' | 'shortest' | 'insertion';

export interface SimSettings {
  busCount: number;
  speedMultiplier: number;
  routingMode: RoutingMode;
  chargingEnabled: boolean;
  /** Battery % at which bus seeks charging station */
  lowBatteryThreshold: number;
  /** Seconds to fully charge a bus */
  chargeDuration: number;
  /** Battery drain per unit distance traveled */
  batteryDrainRate: number;
  showLabels: boolean;
  showEdgeIds: boolean;
  overlayImage: boolean;
}

export interface SelectionState {
  selectedBusId: number | null;
  selectedNodeId: number | null;
  selectedEdgeId: number | null;
}

export interface NetworkGraph {
  nodes: Node[];
  edges: Edge[];
  /** Adjacency: nodeId -> list of outgoing edge ids */
  adjacency: Map<number, number[]>;
}
