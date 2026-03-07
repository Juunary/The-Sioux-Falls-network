// ============================================================
// Controls panel — play/pause, speed, bus count, routing mode
// ============================================================

import React from 'react';
import { useSimStore } from '../sim/store';
import type { RoutingMode } from '../types/network';

interface ControlsProps {
  viewMode: '2d' | '3d';
  onToggleView: () => void;
}

export default function Controls({ viewMode, onToggleView }: ControlsProps) {
  const isPlaying = useSimStore((s) => s.isPlaying);
  const settings = useSimStore((s) => s.settings);
  const setPlaying = useSimStore((s) => s.setPlaying);
  const reset = useSimStore((s) => s.reset);
  const updateSettings = useSimStore((s) => s.updateSettings);
  const setBusCount = useSimStore((s) => s.setBusCount);

  return (
    <div className="controls-panel">
      <div className="controls-title">Sioux Falls Network Sim</div>

      {/* View toggle */}
      <div className="control-row">
        <button className="btn btn-view" onClick={onToggleView}>
          {viewMode === '2d' ? '3D View' : '2D View'}
        </button>
      </div>

      <div className="control-divider" />

      {/* Playback */}
      <div className="control-row">
        <button
          className={`btn ${isPlaying ? 'btn-pause' : 'btn-play'}`}
          onClick={() => setPlaying(!isPlaying)}
        >
          {isPlaying ? '⏸ Pause' : '▶ Play'}
        </button>
        <button className="btn btn-reset" onClick={reset}>
          ↺ Reset
        </button>
      </div>

      {/* Speed */}
      <div className="control-group">
        <label>Speed: {settings.speedMultiplier.toFixed(1)}×</label>
        <input
          type="range"
          min={0.1}
          max={5}
          step={0.1}
          value={settings.speedMultiplier}
          onChange={(e) => updateSettings({ speedMultiplier: parseFloat(e.target.value) })}
        />
      </div>

      {/* Bus count */}
      <div className="control-group">
        <label>Buses (N): {settings.busCount}</label>
        <input
          type="range"
          min={1}
          max={20}
          step={1}
          value={settings.busCount}
          onChange={(e) => setBusCount(parseInt(e.target.value, 10))}
        />
      </div>

      <div className="control-divider" />

      {/* Routing mode */}
      <div className="control-group">
        <label>Routing Mode</label>
        <div className="radio-group">
          {(['random', 'shortest'] as RoutingMode[]).map((mode) => (
            <label key={mode} className="radio-label">
              <input
                type="radio"
                name="routing"
                value={mode}
                checked={settings.routingMode === mode}
                onChange={() => updateSettings({ routingMode: mode })}
              />
              {mode === 'random' ? 'Random Walk' : 'Shortest Path'}
            </label>
          ))}
        </div>
      </div>

      <div className="control-divider" />

      {/* Charging */}
      <div className="control-group">
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={settings.chargingEnabled}
            onChange={(e) => updateSettings({ chargingEnabled: e.target.checked })}
          />
          Charging Behavior
        </label>
      </div>

      {settings.chargingEnabled && (
        <>
          <div className="control-group">
            <label>Low Battery: {settings.lowBatteryThreshold}%</label>
            <input
              type="range"
              min={5}
              max={50}
              step={5}
              value={settings.lowBatteryThreshold}
              onChange={(e) =>
                updateSettings({ lowBatteryThreshold: parseInt(e.target.value, 10) })
              }
            />
          </div>
          <div className="control-group">
            <label>Charge Duration: {settings.chargeDuration}s</label>
            <input
              type="range"
              min={2}
              max={30}
              step={1}
              value={settings.chargeDuration}
              onChange={(e) =>
                updateSettings({ chargeDuration: parseInt(e.target.value, 10) })
              }
            />
          </div>
        </>
      )}

      <div className="control-divider" />

      {/* Display */}
      <div className="control-group">
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={settings.showLabels}
            onChange={(e) => updateSettings({ showLabels: e.target.checked })}
          />
          Node Labels
        </label>
        {viewMode === '2d' && (
          <>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={settings.showEdgeIds}
                onChange={(e) => updateSettings({ showEdgeIds: e.target.checked })}
              />
              Edge IDs
            </label>
          </>
        )}
      </div>
    </div>
  );
}
