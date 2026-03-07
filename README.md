# Sioux Falls Network Simulator

An interactive 2D/3D city-network simulator built on the well-known Sioux Falls transportation benchmark.

## Quick Start

```bash
npm install
npm run dev
```

Open http://localhost:5173.

## Project Structure

```
src/
  types/
    network.ts          Core domain types (Node, Edge, Bus, SimSettings)
  data/
    network.ts          Graph topology — edit nodes and edges here
  utils/
    graph.ts            Graph helpers (adjacency, interpolation, offsets)
    pathfinding.ts      Dijkstra + path reconstruction + nearest-CS finder
  sim/
    store.ts            Zustand store — shared state for 2D and 3D
    engine.ts           requestAnimationFrame simulation loop
  components/
    Network2D.tsx       D3-powered SVG canvas (pan/zoom, interactions)
    Network3D.tsx       @react-three/fiber scene (OrbitControls)
    Controls.tsx        Left sidebar — playback and settings
    InfoPanel.tsx       Right sidebar — status and selection details
  App.tsx               Root component — mounts engine, handles view toggle
  main.tsx              Entry point
  index.css             All styles
```

## Where to Edit Graph Data

**`src/data/network.ts`**

- `NODE_COORDS` — x/y pixel positions of every node on the 960×880 canvas.
- `RAW_EDGES` — `[id, source, target]` triples (0-indexed). Add or remove edges here.
- `CHARGING_STATIONS` — Set of node IDs that are charging stations.

Edge weights are computed automatically from Euclidean node distances.

## Where to Edit Simulation Rules

**`src/sim/store.ts`**

- `DEFAULT_SETTINGS` — change default speed, bus count, battery drain rate, etc.
- `advanceBusRoute()` — all routing and charging decision logic lives here.

**`src/sim/engine.ts`**

- `stepMoving()` / `stepCharging()` — per-frame bus movement and battery update.

## Network Details

- 24 nodes (0–23), 76 directed edges (74 confirmed + 2 marked uncertain in source).
- Source: LeBlanc, Morlok & Pierskalla (1975).
- Charging stations (yellow nodes): 2, 4, 7, 10, 14, 16, 19, 23.

## Features

| Feature | Location |
|---|---|
| D3 pan/zoom on 2D view | `Network2D.tsx` |
| Click nodes/edges/buses | `Network2D.tsx`, `Network3D.tsx` |
| Random walk routing | `store.ts → advanceBusRoute` |
| Dijkstra shortest-path | `utils/pathfinding.ts` |
| Low-battery rerouting | `store.ts → advanceBusRoute` |
| Charging station dwell | `engine.ts → stepCharging` |
| Bus lane offsets | `utils/graph.ts → interpolateEdgeWithOffset` |
| Configurable N buses | Controls slider |
| Configurable speed | Controls slider |

## Key Assumptions

1. Edge weights = Euclidean pixel distance between nodes. No traffic model.
2. Buses move at constant speed; no acceleration/deceleration.
3. "Nearest charging station" uses Dijkstra distance, not geographic proximity.
4. In random-walk mode, buses pick uniformly at random from valid outgoing edges.
5. Charging restores battery to 100% after `chargeDuration` seconds.
6. Edges 74/75 (nodes 5↔6) are marked uncertain and are included but flagged in comments.

## What Can Be Extended

- Real traffic assignment with flow-dependent travel times
- Bus schedules and timetables
- Passenger demand overlay (OD matrix)
- Historical playback and recording
- Import/export graph as JSON
- Custom node/edge labels from a reference image overlay (toggle exists in settings)
