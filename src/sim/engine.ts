// ============================================================
// Simulation engine — RAF loop, independent of rendering
// ============================================================

import { useEffect, useRef } from 'react';
import type { Bus } from '../types/network';
import { EDGES } from '../data/network';
import { useSimStore, advanceBusRoute } from './store';

/**
 * useSimulationEngine
 * Mounts a requestAnimationFrame loop that advances all buses each tick.
 * Must be called once at the root level.
 */
export function useSimulationEngine(): void {
  const rafRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number | null>(null);

  useEffect(() => {
    function tick(timestamp: number) {
      const { isPlaying, buses, settings, updateBuses } = useSimStore.getState();

      if (isPlaying) {
        const dt = lastTimeRef.current == null
          ? 0
          : Math.min((timestamp - lastTimeRef.current) / 1000, 0.1); // cap at 100ms
        lastTimeRef.current = timestamp;

        if (dt > 0) {
          const nextBuses = buses.map((bus) => stepBus(bus, dt, settings.speedMultiplier, settings));
          updateBuses(nextBuses);
        } else {
          lastTimeRef.current = timestamp;
        }
      } else {
        lastTimeRef.current = null;
      }

      rafRef.current = requestAnimationFrame(tick);
    }

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, []);
}

// ---- Per-bus simulation step ----

function stepBus(
  bus: Bus,
  dt: number,
  speedMultiplier: number,
  settings: ReturnType<typeof useSimStore.getState>['settings'],
): Bus {
  switch (bus.state) {
    case 'charging':
      return stepCharging(bus, dt, settings);
    case 'moving':
      return stepMoving(bus, dt, speedMultiplier, settings);
    case 'idle':
    case 'rerouting':
      // Advance route immediately
      return advanceBusRoute(bus, settings);
    default:
      return bus;
  }
}

function stepCharging(bus: Bus, dt: number, settings: typeof bus extends Bus ? ReturnType<typeof useSimStore.getState>['settings'] : never): Bus {
  // Charge battery while stopped
  const chargeRate = 100 / settings.chargeDuration; // % per second
  const newBattery = Math.min(100, bus.battery + chargeRate * dt);
  const newTimeLeft = bus.chargeTimeLeft - dt;

  if (newTimeLeft <= 0) {
    // Done charging — resume routing
    return advanceBusRoute(
      { ...bus, battery: 100, chargeTimeLeft: 0 },
      settings,
    );
  }
  return { ...bus, battery: newBattery, chargeTimeLeft: newTimeLeft };
}

function stepMoving(
  bus: Bus,
  dt: number,
  speedMultiplier: number,
  settings: ReturnType<typeof useSimStore.getState>['settings'],
): Bus {
  if (bus.currentEdge == null) {
    return advanceBusRoute(bus, settings);
  }

  const edge = EDGES[bus.currentEdge];
  if (!edge) return advanceBusRoute(bus, settings);

  // Progress increment: speed (px/s) * multiplier / edge.weight (px)
  const progressPerSecond = (bus.speed * speedMultiplier) / edge.weight;
  const newProgress = bus.progress + progressPerSecond * dt;

  // Battery drain proportional to distance traveled this frame
  const distanceTraveled = progressPerSecond * dt * edge.weight; // px
  const newBattery = Math.max(0, bus.battery - settings.batteryDrainRate * distanceTraveled);

  if (newProgress >= 1) {
    // Arrived at target node
    const arrivedBus: Bus = {
      ...bus,
      currentNode: edge.target,
      currentEdge: null,
      progress: 0,
      battery: newBattery,
      state: 'idle',
    };
    return advanceBusRoute(arrivedBus, settings);
  }

  return { ...bus, progress: newProgress, battery: newBattery };
}
