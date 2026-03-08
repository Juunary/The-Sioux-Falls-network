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
  /** Edge ID (0-indexed) */
  id: number;
  source: number;
  target: number;
  /** Euclidean distance derived from node positions */
  weight: number;
}

export type BusState = 'moving' | 'charging' | 'idle' | 'rerouting';

export interface Bus {
  id: number;
  currentNode: number;
  currentEdge: number | null;
  progress: number;
  speed: number;
  battery: number;
  state: BusState;
  route: number[];
  destination: number | null;
  color: string;
  chargeTimeLeft: number;
  laneOffset: number;
  /** IDs of passengers currently on board */
  passengerIds: number[];
  /** Maximum passenger capacity */
  capacity: number;
}

// ---- Passenger ----

export type PassengerState = 'waiting' | 'riding' | 'arrived' | 'gone';

export interface Passenger {
  id: number;
  /** Node where this passenger spawned / is currently waiting */
  originNode: number;
  /** Node the passenger wants to reach */
  destinationNode: number;
  /** Current node (relevant when waiting; set to last node when riding) */
  currentNode: number;
  /** Bus ID when riding, null otherwise */
  busId: number | null;
  state: PassengerState;
  /** simTime when this passenger started waiting at currentNode */
  waitingSince: number;
  /** Seconds the passenger will wait before giving up */
  patience: number;
  /** Color for rendering and tracking */
  color: string;
}

export type RoutingMode = 'random' | 'shortest' | 'insertion';

export interface SimSettings {
  busCount: number;
  speedMultiplier: number;
  routingMode: RoutingMode;
  chargingEnabled: boolean;
  lowBatteryThreshold: number;
  chargeDuration: number;
  batteryDrainRate: number;
  showLabels: boolean;
  showEdgeIds: boolean;
  overlayImage: boolean;
  /** Passengers spawned per second across the whole network */
  passengerSpawnRate: number;
  /** Max passengers per bus */
  passengerCapacity: number;
}

export interface SelectionState {
  selectedBusId: number | null;
  selectedNodeId: number | null;
  selectedEdgeId: number | null;
  /** Passenger being tracked (highlighted and shown in InfoPanel) */
  trackedPassengerId: number | null;
}

export interface NetworkGraph {
  nodes: Node[];
  edges: Edge[];
  adjacency: Map<number, number[]>;
}
