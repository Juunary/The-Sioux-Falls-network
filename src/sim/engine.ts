// ============================================================
// Simulation engine — RAF loop + passenger lifecycle
// ============================================================

import { useEffect, useRef } from 'react';
import type { Bus, Passenger, PassengerState, SimSettings } from '../types/network';
import { EDGES } from '../data/network';
import { ALL_NODE_IDS } from '../utils/graph';
import { useSimStore, advanceBusRoute, passengerColor, PASSENGER_COLORS } from './store';

const MAX_PASSENGER_POOL = 300; // trim gone/arrived passengers beyond this

export function useSimulationEngine(): void {
  const rafRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number | null>(null);
  const historyTimerRef = useRef<number>(0); // countdown to next history snapshot

  useEffect(() => {
    function tick(timestamp: number) {
      const state = useSimStore.getState();

      if (!state.isPlaying) {
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

      const { buses, passengers, settings, nextPassengerId, simTime, waitingHistory } = state;
      const newSimTime = simTime + dt;

      // 1. Handle passenger events (alighting, boarding) before buses depart
      //    Buses that arrived last tick are idle (currentEdge === null), so
      //    passengers can board before advanceBusRoute sends them off again.
      const { buses: busesAfterPax, passengers: updatedPax } = processPassengers(
        buses,
        passengers,
        settings,
        newSimTime,
      );

      // 2. Step buses (movement / charging / routing) — depart after boarding
      const steppedBuses = busesAfterPax.map((bus) =>
        stepBus(bus, dt, settings.speedMultiplier, settings),
      );

      // 3. Spawn new passengers
      const { passengers: spawnedPax, nextId } = maybeSpawnPassengers(
        updatedPax,
        nextPassengerId,
        settings,
        dt,
        newSimTime,
      );

      // 4. Trim old arrived/gone passengers if pool grows too large
      const finalPax = trimPassengerPool(spawnedPax);

      // 5. Update waiting history snapshot every 0.5 s
      historyTimerRef.current -= dt;
      let newHistory = waitingHistory;
      if (historyTimerRef.current <= 0) {
        historyTimerRef.current = 0.5;
        const waitingCount = finalPax.filter((p) => p.state === 'waiting').length;
        newHistory = [...waitingHistory.slice(-119), { t: newSimTime, count: waitingCount }];
      }

      useSimStore.setState({
        buses: steppedBuses,
        passengers: finalPax,
        nextPassengerId: nextId,
        simTime: newSimTime,
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

// ============================================================
// Bus stepping (unchanged logic)
// ============================================================

function stepBus(
  bus: Bus,
  dt: number,
  speedMultiplier: number,
  settings: SimSettings,
): Bus {
  switch (bus.state) {
    case 'charging':  return stepCharging(bus, dt, settings);
    case 'moving':    return stepMoving(bus, dt, speedMultiplier, settings);
    case 'idle':
    case 'rerouting': return advanceBusRoute(bus, settings);
    default:          return bus;
  }
}

function stepCharging(bus: Bus, dt: number, settings: SimSettings): Bus {
  const chargeRate = 100 / settings.chargeDuration;
  const newBattery = Math.min(100, bus.battery + chargeRate * dt);
  const newTimeLeft = bus.chargeTimeLeft - dt;
  if (newTimeLeft <= 0) {
    return advanceBusRoute({ ...bus, battery: 100, chargeTimeLeft: 0 }, settings);
  }
  return { ...bus, battery: newBattery, chargeTimeLeft: newTimeLeft };
}

function stepMoving(bus: Bus, dt: number, speedMultiplier: number, settings: SimSettings): Bus {
  if (bus.currentEdge == null) return advanceBusRoute(bus, settings);
  const edge = EDGES[bus.currentEdge];
  if (!edge) return advanceBusRoute(bus, settings);

  const progressPerSecond = (bus.speed * speedMultiplier) / edge.weight;
  const newProgress = bus.progress + progressPerSecond * dt;
  const distanceTraveled = progressPerSecond * dt * edge.weight;
  const newBattery = Math.max(0, bus.battery - settings.batteryDrainRate * distanceTraveled);

  if (newProgress >= 1) {
    // Stay idle for one tick so processPassengers can handle boarding/alighting
    // before advanceBusRoute sends the bus off on the next tick.
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
): { buses: Bus[]; passengers: Passenger[] } {
  // --- 1. Patience check ---
  let pax = passengers.map((p) => {
    if (p.state === 'waiting' && simTime - p.waitingSince > p.patience) {
      return { ...p, state: 'gone' as PassengerState };
    }
    return p;
  });

  // Make mutable bus copies (only passengerIds changes)
  const updatedBuses = buses.map((b) => ({ ...b, passengerIds: [...b.passengerIds] }));

  // --- 2. Alighting: riders who reached their destination ---
  for (const bus of updatedBuses) {
    if (bus.currentEdge !== null) continue; // still moving

    const ridersAtDest = pax.filter(
      (p) => p.state === 'riding' && p.busId === bus.id && p.destinationNode === bus.currentNode,
    );

    for (const p of ridersAtDest) {
      bus.passengerIds = bus.passengerIds.filter((id) => id !== p.id);

      if (Math.random() < 0.5) {
        // 50%: get a new random destination and wait again
        const newDest = randomNodeExcluding(bus.currentNode);
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
                patience: 20 + Math.random() * 40,
              }
            : pp,
        );
      } else {
        // 50%: arrived and disappear
        pax = pax.map((pp) =>
          pp.id === p.id ? { ...pp, state: 'arrived' as PassengerState, busId: null } : pp,
        );
      }
    }
  }

  // --- 3. Boarding: waiting passengers try to board a bus at their node ---
  for (const bus of updatedBuses) {
    if (bus.currentEdge !== null) continue;
    if (bus.state === 'charging') continue;

    const waitingHere = pax.filter(
      (p) => p.state === 'waiting' && p.currentNode === bus.currentNode,
    );

    for (const p of waitingHere) {
      if (bus.passengerIds.length >= bus.capacity) break;
      if (!shouldBoard(bus, p, settings)) continue;

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
 * - random mode: 35% chance (uncertain route, hop on and hope)
 * - shortest/insertion: board only if destination is in the bus's planned route
 */
function shouldBoard(bus: Bus, passenger: Passenger, settings: SimSettings): boolean {
  if (settings.routingMode === 'random') {
    return Math.random() < 0.8;
  }
  return (
    bus.route.includes(passenger.destinationNode) ||
    bus.destination === passenger.destinationNode
  );
}

// ---- Spawning ----

function maybeSpawnPassengers(
  passengers: Passenger[],
  nextId: number,
  settings: SimSettings,
  dt: number,
  simTime: number,
): { passengers: Passenger[]; nextId: number } {
  const expected = settings.passengerSpawnRate * dt;
  // Poisson approximation: floor + probabilistic extra
  const count = Math.floor(expected) + (Math.random() < expected % 1 ? 1 : 0);
  if (count === 0) return { passengers, nextId };

  const newPax = [...passengers];
  let id = nextId;

  for (let i = 0; i < count; i++) {
    const origin = ALL_NODE_IDS[Math.floor(Math.random() * ALL_NODE_IDS.length)];
    const destOptions = ALL_NODE_IDS.filter((n) => n !== origin);
    const dest = destOptions[Math.floor(Math.random() * destOptions.length)];

    newPax.push({
      id,
      originNode: origin,
      destinationNode: dest,
      currentNode: origin,
      busId: null,
      state: 'waiting',
      waitingSince: simTime,
      patience: 20 + Math.random() * 40,
      color: PASSENGER_COLORS[id % PASSENGER_COLORS.length],
    });
    id++;
  }

  return { passengers: newPax, nextId: id };
}

// ---- Helpers ----

function randomNodeExcluding(excludeId: number): number {
  const options = ALL_NODE_IDS.filter((id) => id !== excludeId);
  return options[Math.floor(Math.random() * options.length)];
}

function trimPassengerPool(passengers: Passenger[]): Passenger[] {
  if (passengers.length <= MAX_PASSENGER_POOL) return passengers;
  // Remove oldest arrived/gone first
  const active = passengers.filter((p) => p.state === 'waiting' || p.state === 'riding');
  const inactive = passengers.filter((p) => p.state === 'arrived' || p.state === 'gone');
  const keep = inactive.slice(-(MAX_PASSENGER_POOL - active.length));
  return [...active, ...keep];
}
