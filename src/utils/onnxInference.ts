// ============================================================
// ONNX inference — browser-side PPO policy execution
//
// Observation schema (96-dim flat Box, must match Python env):
//   [0:24]  self_node one-hot
//   [24]    battery / 100
//   [25]    onboard passengers / capacity
//   [26:30] state one-hot  [moving, charging, idle, rerouting]
//   [30]    (speed - 80) / 40
//   [31:55] onboard destination distribution / capacity
//   [55:79] waiting passengers per node / 5  (clipped to 1.0)
//   [79:87] fleet count at each charger / bus_count
//   [87:95] Dijkstra distance to each charger / MAX_DIST
//   [95]    sim_time / episode_length
//
// Model I/O:
//   Input  "obs"           shape [batch, 96]  float32
//   Output "action_logits" shape [batch, 24]  float32
//
// Action masking (current_node invalid, unreachable invalid) is
// applied here before argmax — the model never sees the mask.
// ============================================================

import * as ort from 'onnxruntime-web';
import type { Bus, DecisionContext } from '../types/network';
import { EDGES, ADJACENCY } from '../data/network';
import { ALL_NODE_IDS, CHARGING_STATION_IDS } from './graph';
import { dijkstra } from './pathfinding';

// ---- WASM configuration ----
// Use CDN-hosted binaries so no vite.config.ts changes are needed.
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.0/dist/';

// ---- Constants ----
const OBS_DIM = 96;
const N_NODES = 24;
const N_CHARGERS = 8;
const DEFAULT_EPISODE_LENGTH = 600.0;   // seconds used in free-play mode
const SPEED_MIN = 80;
const SPEED_RANGE = 40;
const WAIT_NORM = 5.0;

const STATE_ORDER = ['moving', 'charging', 'idle', 'rerouting'] as const;
type KnownState = typeof STATE_ORDER[number];

// ---- Module-level session state ----
let _session: ort.InferenceSession | null = null;
let _loadedModelId: string | null = null;
let _maxDist: number | null = null;

/** Cache: snapshot key (`${busId}:${node}`) → inferred next-node */
const _ppoCache = new Map<string, number>();
/** Guard: skip snapshots with in-flight inference to avoid duplicate runs */
const _pendingSnapshots = new Set<string>();

function _snapshotKey(busId: number, node: number): string {
  return `${busId}:${node}`;
}

// ---- MAX_DIST (lazy, computed once on first PPO call) ----
function _getMaxDist(): number {
  if (_maxDist !== null) return _maxDist;
  let max = 0;
  for (const src of ALL_NODE_IDS) {
    const { dist } = dijkstra(src, ALL_NODE_IDS, ADJACENCY, EDGES);
    for (const d of dist.values()) {
      if (d < Infinity && d > max) max = d;
    }
  }
  _maxDist = max > 0 ? max : 1;
  return _maxDist;
}

// ============================================================
// Observation builder
// ============================================================

/**
 * Build a 96-dim Float32Array observation for one bus.
 * Must produce values identical to Python `build_observation()`.
 */
export function buildObservation(
  bus: Bus,
  context: DecisionContext,
  episodeLength = DEFAULT_EPISODE_LENGTH,
): Float32Array {
  const obs = new Float32Array(OBS_DIM);
  const busCount = context.allBuses.length || 1;
  const maxDist = _getMaxDist();

  // [0:24] self_node one-hot
  if (bus.currentNode >= 0 && bus.currentNode < N_NODES) {
    obs[bus.currentNode] = 1;
  }

  // [24] battery / 100
  obs[24] = Math.min(bus.battery / 100, 1);

  // [25] onboard passengers / capacity
  obs[25] = bus.passengerIds.length / (bus.capacity || 1);

  // [26:30] state one-hot
  const stateIdx = STATE_ORDER.indexOf(bus.state as KnownState);
  if (stateIdx >= 0) obs[26 + stateIdx] = 1;

  // [30] speed normalised
  obs[30] = (bus.speed - SPEED_MIN) / SPEED_RANGE;

  // [31:55] onboard destination distribution / capacity
  for (const p of context.passengers) {
    if (p.busId === bus.id && p.state === 'riding') {
      const dest = p.destinationNode;
      if (dest >= 0 && dest < N_NODES) obs[31 + dest] += 1;
    }
  }
  const cap = bus.capacity || 1;
  for (let i = 31; i < 55; i++) obs[i] /= cap;

  // [55:79] waiting passengers per node / WAIT_NORM (clipped)
  const waitCounts = new Float32Array(N_NODES);
  for (const p of context.passengers) {
    if (p.state === 'waiting' && p.currentNode >= 0 && p.currentNode < N_NODES) {
      waitCounts[p.currentNode] += 1;
    }
  }
  for (let i = 0; i < N_NODES; i++) {
    obs[55 + i] = Math.min(waitCounts[i] / WAIT_NORM, 1.0);
  }

  // [79:87] fleet count at charger nodes / bus_count
  // Only count buses stationary at a charger (currentEdge === null) — matches Python env.
  const fleetCounts = new Float32Array(N_CHARGERS);
  for (const b of context.allBuses) {
    const ci = CHARGING_STATION_IDS.indexOf(b.currentNode);
    if (ci >= 0 && b.currentEdge === null) fleetCounts[ci] += 1;
  }
  for (let i = 0; i < N_CHARGERS; i++) {
    obs[79 + i] = fleetCounts[i] / busCount;
  }

  // [87:95] Dijkstra distance to each charger / MAX_DIST
  const { dist } = dijkstra(bus.currentNode, ALL_NODE_IDS, ADJACENCY, EDGES);
  for (let i = 0; i < N_CHARGERS; i++) {
    const d = dist.get(CHARGING_STATION_IDS[i]) ?? Infinity;
    obs[87 + i] = d < Infinity ? d / maxDist : 1.0;
  }

  // [95] time normalised
  obs[95] = Math.min(context.simTime / episodeLength, 1.0);

  return obs;
}

// ============================================================
// Internal helpers
// ============================================================

/**
 * Masked argmax.
 * current_node is always invalid; unreachable nodes are invalid.
 * Ties broken by lowest node index (deterministic, matches Python env).
 */
function _maskedArgmax(
  logits: Float32Array,
  currentNode: number,
  reachable: boolean[],
): number {
  let bestIdx = -1;
  let bestVal = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (i === currentNode || !reachable[i]) continue;
    if (logits[i] > bestVal) {
      bestVal = logits[i];
      bestIdx = i;
    }
  }
  return bestIdx;
}

/** Boolean reachability vector: dist < Infinity for each node. */
function _reachability(srcNode: number): boolean[] {
  const { dist } = dijkstra(srcNode, ALL_NODE_IDS, ADJACENCY, EDGES);
  return ALL_NODE_IDS.map((id) => (dist.get(id) ?? Infinity) < Infinity);
}

// ============================================================
// Public session management
// ============================================================

/**
 * Download an ONNX file from `url` and initialise the inference session.
 * Clears the action cache so stale decisions are not reused.
 */
export async function loadOnnxModel(url: string, modelId: string): Promise<void> {
  _session = await ort.InferenceSession.create(url, {
    executionProviders: ['wasm'],
  });
  _loadedModelId = modelId;
  _ppoCache.clear();
  _pendingSnapshots.clear();
}

/** Release the ONNX session and clear all caches. */
export function unloadOnnxModel(): void {
  _session = null;
  _loadedModelId = null;
  _ppoCache.clear();
  _pendingSnapshots.clear();
}

/** Return the model ID currently loaded, or null. */
export function getLoadedModelId(): string | null {
  return _loadedModelId;
}

/** True iff an ONNX session is active and ready for inference. */
export function isModelLoaded(): boolean {
  return _session !== null;
}

// ============================================================
// Public inference API
// ============================================================

/**
 * Retrieve the cached PPO action for a bus at a specific node.
 * Returns null if no result exists for this exact (busId, currentNode) snapshot.
 */
export function getPpoAction(busId: number, currentNode: number): number | null {
  return _ppoCache.get(_snapshotKey(busId, currentNode)) ?? null;
}

/**
 * Consume (clear) the cached action for a snapshot after it has been used for routing.
 * Prevents the same result from being applied to a later, different idle event.
 */
export function consumePpoAction(busId: number, currentNode: number): void {
  _ppoCache.delete(_snapshotKey(busId, currentNode));
}

/**
 * Schedule async ONNX inference for the given buses.
 *
 * Results are stored in the cache; `getPpoAction()` returns them on
 * the next routing decision.  This function is fire-and-forget — it
 * does NOT block the synchronous simulation tick.
 *
 * Buses with in-flight inference are skipped to avoid duplicate runs.
 * Inference runs sequentially per bus to stay within WASM memory limits.
 */
export async function schedulePpoInference(
  buses: Bus[],
  context: DecisionContext,
  episodeLength = DEFAULT_EPISODE_LENGTH,
): Promise<void> {
  if (!_session) return;

  for (const bus of buses) {
    const key = _snapshotKey(bus.id, bus.currentNode);
    if (_pendingSnapshots.has(key)) continue;      // same snapshot already in flight
    _pendingSnapshots.add(key);
    try {
      const obs = buildObservation(bus, context, episodeLength);
      const tensor = new ort.Tensor('float32', obs, [1, OBS_DIM]);
      const result = await _session.run({ obs: tensor });
      const logits = result['action_logits'].data as Float32Array;
      const reachable = _reachability(bus.currentNode);
      const action = _maskedArgmax(logits, bus.currentNode, reachable);
      if (action >= 0) _ppoCache.set(key, action);  // store under snapshot key
    } catch {
      // Silently ignore per-bus errors (model not yet loaded, etc.)
    } finally {
      _pendingSnapshots.delete(key);
    }
  }
}
