// ============================================================
// Zustand store — shared simulation state for 2D and 3D views
// ============================================================

import { create } from 'zustand';
import type { Bus, SimSettings, SelectionState } from '../types/network';
import { NODES, EDGES, ADJACENCY } from '../data/network';
import { ALL_NODE_IDS, CHARGING_STATION_IDS, randomOutgoingEdge } from '../utils/graph';
import { dijkstra, reconstructPath, nearestChargingStation, insertionHeuristicTour } from '../utils/pathfinding';

// ---- Bus color palette ----
const BUS_COLORS = [
  '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
  '#1abc9c', '#e67e22', '#e91e63', '#00bcd4', '#8bc34a',
  '#ff5722', '#607d8b', '#795548', '#ff9800', '#673ab7',
];

function busColor(id: number): string {
  return BUS_COLORS[id % BUS_COLORS.length];
}

function randomNode(): number {
  return Math.floor(Math.random() * NODES.length);
}

function makeBus(id: number, busCount: number): Bus {
  const startNode = randomNode();
  // Stagger lane offsets so buses on the same edge don't stack
  const laneOffset = ((id % 5) - 2) * 6; // -12 to +12 px
  return {
    id,
    currentNode: startNode,
    currentEdge: null,
    progress: 0,
    speed: 80 + Math.random() * 40, // 80–120 px/s base speed
    battery: 60 + Math.random() * 40, // start 60–100%
    state: 'idle',
    route: [],
    destination: null,
    color: busColor(id),
    chargeTimeLeft: 0,
    laneOffset,
  };
}

function createBuses(count: number): Bus[] {
  return Array.from({ length: count }, (_, i) => makeBus(i, count));
}

// ---- Default settings ----
const DEFAULT_SETTINGS: SimSettings = {
  busCount: 6,
  speedMultiplier: 1,
  routingMode: 'random',
  chargingEnabled: true,
  lowBatteryThreshold: 20,
  chargeDuration: 8,       // seconds
  batteryDrainRate: 0.08,  // % per px traveled
  showLabels: true,
  showEdgeIds: true,
  overlayImage: false,
};

// ---- Store types ----
export interface SimStore {
  buses: Bus[];
  settings: SimSettings;
  selection: SelectionState;
  isPlaying: boolean;

  // Actions
  setPlaying: (v: boolean) => void;
  reset: () => void;
  updateBuses: (buses: Bus[]) => void;
  updateSettings: (patch: Partial<SimSettings>) => void;
  setSelection: (patch: Partial<SelectionState>) => void;
  setBusCount: (n: number) => void;
}

export const useSimStore = create<SimStore>((set, get) => ({
  buses: createBuses(DEFAULT_SETTINGS.busCount),
  settings: { ...DEFAULT_SETTINGS },
  selection: { selectedBusId: null, selectedNodeId: null, selectedEdgeId: null },
  isPlaying: true,

  setPlaying: (v) => set({ isPlaying: v }),

  reset: () => {
    const { settings } = get();
    set({ buses: createBuses(settings.busCount), isPlaying: true });
  },

  updateBuses: (buses) => set({ buses }),

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
          makeBus(current.length + i, clamped),
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
// Routing helpers (called by the simulation engine)
// ============================================================

/**
 * Choose the next edge for a bus that just arrived at a node.
 * Handles low-battery rerouting to charging stations.
 * Mutates the bus object in place and returns it.
 */
export function advanceBusRoute(bus: Bus, settings: SimSettings): Bus {
  const { routingMode, chargingEnabled, lowBatteryThreshold, chargeDuration } = settings;
  const node = NODES[bus.currentNode];

  // --- Arrived at charging station while needing charge ---
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

  // --- Low battery: reroute to nearest charging station ---
  if (chargingEnabled && bus.battery < lowBatteryThreshold && !node.isChargingStation) {
    const result = nearestChargingStation(
      bus.currentNode,
      CHARGING_STATION_IDS,
      ALL_NODE_IDS,
      ADJACENCY,
      EDGES,
    );
    if (result && result.path.length > 1) {
      const route = result.path.slice(1); // exclude current node
      const nextNode = route[0];
      const edge = findEdge(bus.currentNode, nextNode);
      if (edge) {
        return {
          ...bus,
          state: 'moving',
          route,
          destination: result.nodeId,
          currentEdge: edge.id,
          progress: 0,
        };
      }
    }
  }

  // --- Follow existing route (shortest-path mode) ---
  if (bus.route.length > 0) {
    const nextNode = bus.route[0];
    const edge = findEdge(bus.currentNode, nextNode);
    if (edge) {
      return {
        ...bus,
        state: 'moving',
        route: bus.route.slice(1),
        currentEdge: edge.id,
        progress: 0,
      };
    }
  }

  // --- Pick next move ---
  if (routingMode === 'shortest') {
    // Choose a random far-away destination and route to it
    const dest = randomNode();
    const { prev } = dijkstra(bus.currentNode, ALL_NODE_IDS, ADJACENCY, EDGES);
    const path = reconstructPath(bus.currentNode, dest, prev, EDGES);
    if (path.length > 1) {
      const route = path.slice(1);
      const nextNode = route[0];
      const edge = findEdge(bus.currentNode, nextNode);
      if (edge) {
        return {
          ...bus,
          state: 'moving',
          route: route.slice(1),
          destination: dest,
          currentEdge: edge.id,
          progress: 0,
        };
      }
    }
  }

  if (routingMode === 'insertion') {
    // Pick 4 random unique waypoints and order them with Nearest-Insertion Heuristic,
    // then chain Dijkstra shortest paths between consecutive stops into one full route.
    const NUM_WAYPOINTS = 4;
    const candidates = ALL_NODE_IDS.filter((id) => id !== bus.currentNode);
    const waypoints = [...candidates]
      .sort(() => Math.random() - 0.5)
      .slice(0, NUM_WAYPOINTS);

    const ordered = insertionHeuristicTour(
      bus.currentNode,
      waypoints,
      ALL_NODE_IDS,
      ADJACENCY,
      EDGES,
    );

    // Build a flat node sequence by chaining Dijkstra paths between stops
    let fullRoute: number[] = [];
    let cur = bus.currentNode;
    for (const wp of ordered) {
      const { prev } = dijkstra(cur, ALL_NODE_IDS, ADJACENCY, EDGES);
      const path = reconstructPath(cur, wp, prev, EDGES);
      if (path.length > 1) fullRoute = [...fullRoute, ...path.slice(1)];
      cur = wp;
    }

    if (fullRoute.length > 0) {
      const nextNode = fullRoute[0];
      const edge = findEdge(bus.currentNode, nextNode);
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

  // --- Random walk (default / fallback) ---
  const edge = randomOutgoingEdge(bus.currentNode);
  if (edge) {
    return { ...bus, state: 'moving', currentEdge: edge.id, progress: 0, route: [] };
  }

  // Dead end — stay idle
  return { ...bus, state: 'idle' };
}

function findEdge(source: number, target: number) {
  return EDGES.find((e) => e.source === source && e.target === target) ?? null;
}
