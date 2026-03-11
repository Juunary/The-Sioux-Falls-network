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

// 'greedy' = demand-aware greedy baseline (demand_aware_greedy_v1)
// 'ppo'    = ONNX-exported PPO policy (Phase 8)
export type RoutingMode = 'random' | 'shortest' | 'insertion' | 'greedy' | 'ppo';

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
  /**
   * When true (free-play default): riders who reach their destination have a
   * 50% chance of becoming a new waiting passenger with a fresh destination.
   * When false (benchmark mode): every rider permanently exits on arrival.
   */
  reboardEnabled: boolean;
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

// ---- Scenario / Replay types ----

export interface BusInitial {
  id: number;
  start_node: number;
  battery: number;
  speed: number;
}

export interface PassengerEvent {
  event_id: number;
  /** Simulation time at which this passenger spawns */
  spawn_time: number;
  origin: number;
  destination: number;
  patience: number;
}

export interface ScenarioMeta {
  scenario_id: string;
  graph_version: string;
  seed: number;
  split: 'train' | 'val' | 'test';
  difficulty: 'easy' | 'medium' | 'hard' | 'stress';
  episode_length: number;
  sim_dt: number;
  bus_count: number;
  spawn_rate: number;
  reboard_enabled: boolean;
  charging_enabled: boolean;
  low_battery_threshold: number;
  charge_duration: number;
  battery_drain_rate: number;
  passenger_capacity: number;
  generator_version: string;
}

export interface ScenarioData {
  meta: ScenarioMeta;
  buses_initial: BusInitial[];
  passenger_events: PassengerEvent[];
}

/**
 * Context passed to the routing decision function.
 * Contains all global simulation state needed by demand-aware and PPO policies.
 */
export interface DecisionContext {
  allBuses: Bus[];
  passengers: Passenger[];
  simTime: number;
  rng: () => number;
}

/** Metrics collected at the end of a benchmark episode. */
export interface EpisodeMetrics {
  scenario_id: string;
  policy: string;
  episode_reward: number;
  passengers_spawned: number;
  passengers_served: number;
  passengers_gone: number;
  service_rate: number;
  unserved_rate: number;
  avg_wait_time_sec: number;
  charge_events: number;
  total_distance_px: number;
}
