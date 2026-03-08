// ============================================================
// Zustand store — shared simulation state for 2D and 3D views
// ============================================================

import { create } from 'zustand';
import type { Bus, Passenger, SimSettings, SelectionState } from '../types/network';
import { NODES, EDGES, ADJACENCY } from '../data/network';
import { ALL_NODE_IDS, CHARGING_STATION_IDS, randomOutgoingEdge } from '../utils/graph';
import { dijkstra, reconstructPath, nearestChargingStation, insertionHeuristicTour } from '../utils/pathfinding';

// ---- Color palettes ----
const BUS_COLORS = [
  '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
  '#1abc9c', '#e67e22', '#e91e63', '#00bcd4', '#8bc34a',
  '#ff5722', '#607d8b', '#795548', '#ff9800', '#673ab7',
];

export const PASSENGER_COLORS = [
  '#ff6b6b', '#ffa94d', '#ffe066', '#69db7c', '#4dabf7',
  '#da77f2', '#f783ac', '#a9e34b', '#63e6be', '#74c0fc',
  '#ff8787', '#ffec99', '#b2f2bb', '#a5d8ff', '#d0bfff',
  '#ffa8a8', '#ffda79', '#8ce99a', '#91a7ff', '#f3d9fa',
];

function busColor(id: number): string {
  return BUS_COLORS[id % BUS_COLORS.length];
}

export function passengerColor(id: number): string {
  return PASSENGER_COLORS[id % PASSENGER_COLORS.length];
}

function randomNode(): number {
  return Math.floor(Math.random() * NODES.length);
}

function makeBus(id: number, capacity: number): Bus {
  const startNode = randomNode();
  const laneOffset = ((id % 5) - 2) * 6;
  return {
    id,
    currentNode: startNode,
    currentEdge: null,
    progress: 0,
    speed: 80 + Math.random() * 40,
    battery: 60 + Math.random() * 40,
    state: 'idle',
    route: [],
    destination: null,
    color: busColor(id),
    chargeTimeLeft: 0,
    laneOffset,
    passengerIds: [],
    capacity,
  };
}

function createBuses(count: number, capacity: number): Bus[] {
  return Array.from({ length: count }, (_, i) => makeBus(i, capacity));
}

// ---- Default settings ----
const DEFAULT_SETTINGS: SimSettings = {
  busCount: 6,
  speedMultiplier: 1,
  routingMode: 'random',
  chargingEnabled: true,
  lowBatteryThreshold: 20,
  chargeDuration: 8,
  batteryDrainRate: 0.08,
  showLabels: true,
  showEdgeIds: true,
  overlayImage: false,
  passengerSpawnRate: 0.8,   // passengers per second
  passengerCapacity: 6,      // max per bus
};

// ---- Store types ----
export interface SimStore {
  buses: Bus[];
  passengers: Passenger[];
  /** Rolling 120-point history of waiting passenger count */
  waitingHistory: { t: number; count: number }[];
  simTime: number;
  nextPassengerId: number;

  settings: SimSettings;
  selection: SelectionState;
  isPlaying: boolean;

  // Actions
  setPlaying: (v: boolean) => void;
  reset: () => void;
  updateBuses: (buses: Bus[]) => void;
  updatePassengers: (passengers: Passenger[]) => void;
  updateSettings: (patch: Partial<SimSettings>) => void;
  setSelection: (patch: Partial<SelectionState>) => void;
  setBusCount: (n: number) => void;
}

export const useSimStore = create<SimStore>((set, get) => ({
  buses: createBuses(DEFAULT_SETTINGS.busCount, DEFAULT_SETTINGS.passengerCapacity),
  passengers: [],
  waitingHistory: [],
  simTime: 0,
  nextPassengerId: 0,

  settings: { ...DEFAULT_SETTINGS },
  selection: {
    selectedBusId: null,
    selectedNodeId: null,
    selectedEdgeId: null,
    trackedPassengerId: null,
  },
  isPlaying: true,

  setPlaying: (v) => set({ isPlaying: v }),

  reset: () => {
    const { settings } = get();
    set({
      buses: createBuses(settings.busCount, settings.passengerCapacity),
      passengers: [],
      waitingHistory: [],
      simTime: 0,
      nextPassengerId: 0,
      isPlaying: true,
    });
  },

  updateBuses: (buses) => set({ buses }),
  updatePassengers: (passengers) => set({ passengers }),

  updateSettings: (patch) =>
    set((state) => ({ settings: { ...state.settings, ...patch } })),

  setSelection: (patch) =>
    set((state) => ({ selection: { ...state.selection, ...patch } })),

  setBusCount: (n) => {
    const clamped = Math.max(1, Math.min(20, n));
    set((state) => {
      const current = state.buses;
      if (clamped === current.length) return state;
      let next: Bus[];
      if (clamped > current.length) {
        const extras = Array.from({ length: clamped - current.length }, (_, i) =>
          makeBus(current.length + i, state.settings.passengerCapacity),
        );
        next = [...current, ...extras];
      } else {
        next = current.slice(0, clamped);
      }
      return { buses: next, settings: { ...state.settings, busCount: clamped } };
    });
  },
}));

// ============================================================
// Routing helpers
// ============================================================

export function advanceBusRoute(bus: Bus, settings: SimSettings): Bus {
  const { routingMode, chargingEnabled, lowBatteryThreshold, chargeDuration } = settings;
  const node = NODES[bus.currentNode];

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

  if (chargingEnabled && bus.battery < lowBatteryThreshold && !node.isChargingStation) {
    const result = nearestChargingStation(
      bus.currentNode, CHARGING_STATION_IDS, ALL_NODE_IDS, ADJACENCY, EDGES,
    );
    if (result && result.path.length > 1) {
      const route = result.path.slice(1);
      const edge = findEdge(bus.currentNode, route[0]);
      if (edge) {
        return { ...bus, state: 'moving', route, destination: result.nodeId, currentEdge: edge.id, progress: 0 };
      }
    }
  }

  if (bus.route.length > 0) {
    const nextNode = bus.route[0];
    const edge = findEdge(bus.currentNode, nextNode);
    if (edge) {
      return { ...bus, state: 'moving', route: bus.route.slice(1), currentEdge: edge.id, progress: 0 };
    }
  }

  if (routingMode === 'shortest') {
    const dest = randomNode();
    const { prev } = dijkstra(bus.currentNode, ALL_NODE_IDS, ADJACENCY, EDGES);
    const path = reconstructPath(bus.currentNode, dest, prev, EDGES);
    if (path.length > 1) {
      const route = path.slice(1);
      const edge = findEdge(bus.currentNode, route[0]);
      if (edge) {
        return { ...bus, state: 'moving', route: route.slice(1), destination: dest, currentEdge: edge.id, progress: 0 };
      }
    }
  }

  if (routingMode === 'insertion') {
    const NUM_WAYPOINTS = 4;
    const candidates = ALL_NODE_IDS.filter((id) => id !== bus.currentNode);
    const waypoints = [...candidates].sort(() => Math.random() - 0.5).slice(0, NUM_WAYPOINTS);

    const ordered = insertionHeuristicTour(bus.currentNode, waypoints, ALL_NODE_IDS, ADJACENCY, EDGES);

    let fullRoute: number[] = [];
    let cur = bus.currentNode;
    for (const wp of ordered) {
      const { prev } = dijkstra(cur, ALL_NODE_IDS, ADJACENCY, EDGES);
      const path = reconstructPath(cur, wp, prev, EDGES);
      if (path.length > 1) fullRoute = [...fullRoute, ...path.slice(1)];
      cur = wp;
    }

    if (fullRoute.length > 0) {
      const edge = findEdge(bus.currentNode, fullRoute[0]);
      if (edge) {
        return {
          ...bus,
          state: 'moving',
          route: fullRoute.slice(1),
          destination: fullRoute[fullRoute.length - 1],
          currentEdge: edge.id,
          progress: 0,
        };
      }
    }
  }

  const edge = randomOutgoingEdge(bus.currentNode);
  if (edge) {
    return { ...bus, state: 'moving', currentEdge: edge.id, progress: 0, route: [] };
  }

  return { ...bus, state: 'idle' };
}

function findEdge(source: number, target: number) {
  return EDGES.find((e) => e.source === source && e.target === target) ?? null;
}
