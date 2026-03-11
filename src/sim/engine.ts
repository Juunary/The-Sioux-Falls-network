// ============================================================
// Simulation engine — RAF loop + passenger lifecycle
//
// Architecture:
//  - simulateTick(): deterministic simulation function (no DOM, no RAF, no Zustand).
//    Same state + same RNG sequence → same result.
//    In benchmark/replay mode, passenger_events are pre-generated so rng is rarely called.
//  - useSimulationEngine(): React hook that drives UI mode via RAF.
//  - runEpisode(): fixed-dt replay runner for benchmark / parity testing.
//
// Tick order (must match Python env exactly):
//  1. processPassengers()  — patience, alighting, boarding
//  2. stepBus() for all buses in bus.id ascending order
//  3. maybeSpawnPassengers() / replaySpawn()
// ============================================================

import { useEffect, useRef } from 'react';
import type {
  Bus,
  Passenger,
  PassengerState,
  SimSettings,
  DecisionContext,
  ScenarioData,
  PassengerEvent,
} from '../types/network';
import { EDGES } from '../data/network';
import { ALL_NODE_IDS } from '../utils/graph';
import { useSimStore, passengerColor, PASSENGER_COLORS } from './store';
import { decideNextRoute } from './routing';
import { simRng } from '../utils/rng';

const MAX_PASSENGER_POOL = 300;

// ---- Exported for benchmark / parity runner ----
export const SIM_DT = 0.016; // fixed time step for deterministic replay (≈60fps)

export interface SimState {
  buses: Bus[];
  passengers: Passenger[];
  simTime: number;
  nextPassengerId: number;
}

// ============================================================
// simulateTick — deterministic simulation function
// ============================================================

/**
 * Advance simulation by `dt` seconds.
 *
 * deterministic: given identical (state, dt, settings, rng sequence, eventQueue),
 * produces identical output. Does not access DOM, RAF, or Zustand.
 *
 * In benchmark/replay mode rng is barely used because all passenger events
 * are pre-generated in the scenario file.
 */
export function simulateTick(
  state: SimState,
  dt: number,
  settings: SimSettings,
  rng: () => number,
  replayQueue?: PassengerEvent[], // if provided, spawn from queue instead of Poisson
): SimState {
  // 1. Passenger events (patience, alighting, boarding)
  const { buses: busesAfterPax, passengers: pax1 } = processPassengers(
    state.buses,
    state.passengers,
    settings,
    state.simTime + dt,
    rng,
  );

  // Build context AFTER processPassengers so routing decisions (greedy/ppo)
  // see the post-tick passenger and bus states, not the stale pre-tick snapshot.
  const context: DecisionContext = {
    allBuses: busesAfterPax,
    passengers: pax1,
    simTime: state.simTime + dt,
    rng,
  };

  // 2. Step buses — bus.id ascending order (deterministic)
  const sortedBuses = [...busesAfterPax].sort((a, b) => a.id - b.id);
  const steppedBuses = sortedBuses.map((bus) => stepBus(bus, dt, settings, context));

  // 3. Spawn passengers
  const { passengers: finalPax, nextId } = replayQueue
    ? replaySpawn(pax1, state.nextPassengerId, state.simTime + dt, replayQueue)
    : maybeSpawnPassengers(pax1, state.nextPassengerId, settings, dt, state.simTime + dt, rng);

  return {
    buses: steppedBuses,
    passengers: trimPassengerPool(finalPax),
    simTime: state.simTime + dt,
    nextPassengerId: nextId,
  };
}

// ============================================================
// runEpisode — fixed-dt benchmark runner (TypeScript reference)
// ============================================================

export interface EpisodeResult {
  scenario_id: string;
  total_reward: number;
  passengers_served: number;
  passengers_gone: number;
  passengers_spawned: number;
}

/**
 * Run a complete episode from scenario data using fixed SIM_DT steps.
 * Returns metrics without any UI side effects.
 */
export function runEpisode(scenario: ScenarioData, settings: SimSettings): EpisodeResult {
  // Build initial state from scenario
  const buses: Bus[] = scenario.buses_initial.map((b) => ({
    id: b.id,
    currentNode: b.start_node,
    currentEdge: null,
    progress: 0,
    speed: b.speed,
    battery: b.battery,
    state: 'idle' as const,
    route: [],
    destination: null,
    color: '#3498db',
    chargeTimeLeft: 0,
    laneOffset: ((b.id % 5) - 2) * 6,
    passengerIds: [],
    capacity: scenario.meta.passenger_capacity,
  }));

  let state: SimState = {
    buses,
    passengers: [],
    simTime: 0,
    nextPassengerId: 0,
  };

  // Apply scenario settings
  const episodeSettings: SimSettings = {
    ...settings,
    busCount: scenario.meta.bus_count,
    reboardEnabled: scenario.meta.reboard_enabled,
    chargingEnabled: scenario.meta.charging_enabled,
    lowBatteryThreshold: scenario.meta.low_battery_threshold,
    chargeDuration: scenario.meta.charge_duration,
    batteryDrainRate: scenario.meta.battery_drain_rate,
    passengerCapacity: scenario.meta.passenger_capacity,
    passengerSpawnRate: scenario.meta.spawn_rate,
  };

  // Replay queue: copy events, will be consumed as simTime advances
  const replayQueue: PassengerEvent[] = [...scenario.passenger_events];

  // Fixed-dt loop (no rng needed for spawn — pre-generated)
  const noop = () => 0;
  while (state.simTime < scenario.meta.episode_length) {
    state = simulateTick(state, SIM_DT, episodeSettings, noop, replayQueue);
  }

  const served = state.passengers.filter((p) => p.state === 'arrived').length;
  const gone = state.passengers.filter((p) => p.state === 'gone').length;
  const spawned = state.nextPassengerId;

  return {
    scenario_id: scenario.meta.scenario_id,
    total_reward: served * 2.0 - gone * 1.5,
    passengers_served: served,
    passengers_gone: gone,
    passengers_spawned: spawned,
  };
}

// ============================================================
// Bus stepping
// ============================================================

function stepBus(
  bus: Bus,
  dt: number,
  settings: SimSettings,
  context: DecisionContext,
): Bus {
  switch (bus.state) {
    case 'charging':  return stepCharging(bus, dt, settings, context);
    case 'moving':    return stepMoving(bus, dt, settings, context);
    case 'idle':
    // rerouting treated identically to idle in MVP
    case 'rerouting': return decideNextRoute(bus, context, settings);
    default:          return bus;
  }
}

function stepCharging(bus: Bus, dt: number, settings: SimSettings, context: DecisionContext): Bus {
  const chargeRate = 100 / settings.chargeDuration;
  const newBattery = Math.min(100, bus.battery + chargeRate * dt);
  const newTimeLeft = bus.chargeTimeLeft - dt;
  if (newTimeLeft <= 0) {
    return decideNextRoute({ ...bus, battery: 100, chargeTimeLeft: 0 }, context, settings);
  }
  return { ...bus, battery: newBattery, chargeTimeLeft: newTimeLeft };
}

function stepMoving(bus: Bus, dt: number, settings: SimSettings, context: DecisionContext): Bus {
  if (bus.currentEdge == null) return decideNextRoute(bus, context, settings);
  const edge = EDGES[bus.currentEdge];
  if (!edge) return decideNextRoute(bus, context, settings);

  const progressPerSecond = (bus.speed * settings.speedMultiplier) / edge.weight;
  const newProgress = bus.progress + progressPerSecond * dt;
  const distanceTraveled = progressPerSecond * dt * edge.weight;
  const newBattery = Math.max(0, bus.battery - settings.batteryDrainRate * distanceTraveled);

  if (newProgress >= 1) {
    // Arrive at target node — stay idle for one tick so processPassengers can board/alight
    return {
      ...bus,
      currentNode: edge.target,
      currentEdge: null,
      progress: 0,
      battery: newBattery,
      state: 'idle',
    };
  }

  return { ...bus, progress: newProgress, battery: newBattery };
}

// ============================================================
// Passenger lifecycle
// ============================================================

function processPassengers(
  buses: Bus[],
  passengers: Passenger[],
  settings: SimSettings,
  simTime: number,
  rng: () => number,
): { buses: Bus[]; passengers: Passenger[] } {
  // 1. Patience check
  let pax = passengers.map((p) => {
    if (p.state === 'waiting' && simTime - p.waitingSince > p.patience) {
      return { ...p, state: 'gone' as PassengerState };
    }
    return p;
  });

  // bus.id ascending order for determinism
  const updatedBuses = [...buses]
    .sort((a, b) => a.id - b.id)
    .map((b) => ({ ...b, passengerIds: [...b.passengerIds] }));

  // 2. Alighting: riders who reached their destination
  for (const bus of updatedBuses) {
    if (bus.currentEdge !== null) continue;

    const ridersAtDest = pax.filter(
      (p) => p.state === 'riding' && p.busId === bus.id && p.destinationNode === bus.currentNode,
    );

    for (const p of ridersAtDest) {
      bus.passengerIds = bus.passengerIds.filter((id) => id !== p.id);

      if (settings.reboardEnabled && rng() < 0.5) {
        // 50%: get a new random destination and wait again (free-play mode only)
        const newDest = randomNodeExcluding(bus.currentNode, rng);
        pax = pax.map((pp) =>
          pp.id === p.id
            ? {
                ...pp,
                state: 'waiting' as PassengerState,
                busId: null,
                originNode: bus.currentNode,
                currentNode: bus.currentNode,
                destinationNode: newDest,
                waitingSince: simTime,
                patience: 20 + rng() * 40,
              }
            : pp,
        );
      } else {
        // arrived and permanently exit (always in benchmark; 50% in free-play)
        pax = pax.map((pp) =>
          pp.id === p.id ? { ...pp, state: 'arrived' as PassengerState, busId: null } : pp,
        );
      }
    }
  }

  // 3. Boarding: bus.id ascending order for determinism
  for (const bus of updatedBuses) {
    if (bus.currentEdge !== null) continue;
    if (bus.state === 'charging') continue;

    const waitingHere = pax.filter(
      (p) => p.state === 'waiting' && p.currentNode === bus.currentNode,
    );

    for (const p of waitingHere) {
      if (bus.passengerIds.length >= bus.capacity) break;
      if (!shouldBoard(bus, p, settings, rng)) continue;

      bus.passengerIds = [...bus.passengerIds, p.id];
      pax = pax.map((pp) =>
        pp.id === p.id
          ? { ...pp, state: 'riding' as PassengerState, busId: bus.id }
          : pp,
      );
    }
  }

  return { buses: updatedBuses, passengers: pax };
}

/**
 * Decide whether a waiting passenger boards a given bus.
 * - random mode: 80% chance (optimistic boarding — route is unpredictable)
 * - shortest/insertion/greedy: board only if destination is in the bus's planned route
 */
function shouldBoard(
  bus: Bus,
  passenger: Passenger,
  settings: SimSettings,
  rng: () => number,
): boolean {
  if (settings.routingMode === 'random') {
    return rng() < 0.8;
  }
  return (
    bus.route.includes(passenger.destinationNode) ||
    bus.destination === passenger.destinationNode
  );
}

// ---- Spawning: free-play mode (Poisson) ----

function maybeSpawnPassengers(
  passengers: Passenger[],
  nextId: number,
  settings: SimSettings,
  dt: number,
  simTime: number,
  rng: () => number,
): { passengers: Passenger[]; nextId: number } {
  const expected = settings.passengerSpawnRate * dt;
  // Poisson approximation: floor + probabilistic extra
  const count = Math.floor(expected) + (rng() < expected % 1 ? 1 : 0);
  if (count === 0) return { passengers, nextId };

  const newPax = [...passengers];
  let id = nextId;

  for (let i = 0; i < count; i++) {
    const origin = ALL_NODE_IDS[Math.floor(rng() * ALL_NODE_IDS.length)];
    const destOptions = ALL_NODE_IDS.filter((n) => n !== origin);
    const dest = destOptions[Math.floor(rng() * destOptions.length)];

    newPax.push({
      id,
      originNode: origin,
      destinationNode: dest,
      currentNode: origin,
      busId: null,
      state: 'waiting',
      waitingSince: simTime,
      patience: 20 + rng() * 40,
      color: PASSENGER_COLORS[id % PASSENGER_COLORS.length],
    });
    id++;
  }

  return { passengers: newPax, nextId: id };
}

// ---- Spawning: benchmark/replay mode (pre-generated events) ----

/**
 * Consume events from the pre-generated replay queue.
 * Spawn boundary: event.spawn_time ∈ (prevSimTime, currentSimTime].
 * Newly spawned passengers are NOT eligible for same-tick boarding
 * (spawn happens after processPassengers in tick order).
 */
function replaySpawn(
  passengers: Passenger[],
  nextId: number,
  currentSimTime: number,
  queue: PassengerEvent[], // mutable: consumed events are removed from front
): { passengers: Passenger[]; nextId: number } {
  const newPax = [...passengers];
  let id = nextId;

  while (queue.length > 0 && queue[0].spawn_time <= currentSimTime) {
    const ev = queue.shift()!;
    newPax.push({
      id,
      originNode: ev.origin,
      destinationNode: ev.destination,
      currentNode: ev.origin,
      busId: null,
      state: 'waiting',
      waitingSince: ev.spawn_time,
      patience: ev.patience,
      color: PASSENGER_COLORS[id % PASSENGER_COLORS.length],
    });
    id++;
  }

  return { passengers: newPax, nextId: id };
}

// ---- Helpers ----

function randomNodeExcluding(excludeId: number, rng: () => number): number {
  const options = ALL_NODE_IDS.filter((id) => id !== excludeId);
  return options[Math.floor(rng() * options.length)];
}

function trimPassengerPool(passengers: Passenger[]): Passenger[] {
  if (passengers.length <= MAX_PASSENGER_POOL) return passengers;
  const active = passengers.filter((p) => p.state === 'waiting' || p.state === 'riding');
  const inactive = passengers.filter((p) => p.state === 'arrived' || p.state === 'gone');
  const keep = inactive.slice(-(MAX_PASSENGER_POOL - active.length));
  return [...active, ...keep];
}

// ============================================================
// React hook — UI mode (RAF, variable dt)
// ============================================================

export function useSimulationEngine(): void {
  const rafRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number | null>(null);
  const historyTimerRef = useRef<number>(0);

  useEffect(() => {
    function tick(timestamp: number) {
      const store = useSimStore.getState();

      if (!store.isPlaying) {
        lastTimeRef.current = null;
        rafRef.current = requestAnimationFrame(tick);
        return;
      }

      const dt = lastTimeRef.current == null
        ? 0
        : Math.min((timestamp - lastTimeRef.current) / 1000, 0.1);
      lastTimeRef.current = timestamp;

      if (dt <= 0) {
        rafRef.current = requestAnimationFrame(tick);
        return;
      }

      const { buses, passengers, settings, nextPassengerId, simTime, waitingHistory } = store;

      const prevState: SimState = { buses, passengers, simTime, nextPassengerId };
      const nextState = simulateTick(prevState, dt, settings, simRng);

      // Update waiting history snapshot every 0.5 s
      historyTimerRef.current -= dt;
      let newHistory = waitingHistory;
      if (historyTimerRef.current <= 0) {
        historyTimerRef.current = 0.5;
        const waitingCount = nextState.passengers.filter((p) => p.state === 'waiting').length;
        newHistory = [...waitingHistory.slice(-119), { t: nextState.simTime, count: waitingCount }];
      }

      useSimStore.setState({
        buses: nextState.buses,
        passengers: nextState.passengers,
        nextPassengerId: nextState.nextPassengerId,
        simTime: nextState.simTime,
        waitingHistory: newHistory,
      });

      rafRef.current = requestAnimationFrame(tick);
    }

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, []);
}

// Re-export for components that import passengerColor from engine
export { passengerColor };
