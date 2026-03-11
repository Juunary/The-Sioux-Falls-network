// ============================================================
// Controls panel — play/pause, speed, bus count, routing mode,
//                  PPO model management (Phase 8)
// ============================================================

import React, { useState, useEffect } from 'react';
import { useSimStore } from '../sim/store';
import type { RoutingMode } from '../types/network';
import { loadOnnxModel, getLoadedModelId } from '../utils/onnxInference';
import { API_BASE } from '../api/client';

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

  // ---- PPO model management state ----
  const [ppoModels, setPpoModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [exportJobId, setExportJobId] = useState('');
  const [exportPending, setExportPending] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [loadPending, setLoadPending] = useState(false);
  const [loadStatus, setLoadStatus] = useState<'idle' | 'loaded' | 'error'>('idle');
  const [loadError, setLoadError] = useState<string | null>(null);

  // Fetch model list whenever the user switches to PPO mode
  useEffect(() => {
    if (settings.routingMode !== 'ppo') return;
    fetch(`${API_BASE}/models/`)
      .then((r) => r.json())
      .then((d: { models: string[] }) => {
        const models = d.models ?? [];
        setPpoModels(models);
        setSelectedModel((prev) =>
          models.includes(prev) ? prev : (models[0] ?? ''),
        );
      })
      .catch(() => {});
  // Only re-run when mode changes to/from 'ppo'
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings.routingMode]);

  async function handleExport() {
    if (!exportJobId) return;
    setExportPending(true);
    setExportError(null);
    try {
      const res = await fetch(`${API_BASE}/models/${exportJobId}/export_onnx`, {
        method: 'POST',
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error((err as { detail?: string }).detail ?? 'Export failed');
      }
      // Refresh the list and auto-select the freshly exported model
      const listData = await fetch(`${API_BASE}/models/`).then((r) => r.json());
      const models: string[] = listData.models ?? [];
      setPpoModels(models);
      if (models.includes(exportJobId)) setSelectedModel(exportJobId);
      setExportJobId('');
    } catch (e) {
      setExportError(String(e));
    } finally {
      setExportPending(false);
    }
  }

  async function handleLoadModel() {
    if (!selectedModel) return;
    setLoadPending(true);
    setLoadError(null);
    try {
      await loadOnnxModel(`${API_BASE}/models/${selectedModel}`, selectedModel);
      setLoadStatus('loaded');
    } catch (e) {
      setLoadStatus('error');
      setLoadError(String(e));
    } finally {
      setLoadPending(false);
    }
  }

  const loadedId = getLoadedModelId();

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
          {(
            [
              ['random',    'Random Walk'],
              ['shortest',  'Shortest Path'],
              ['insertion', 'Insertion Heuristic'],
              ['greedy',    'Demand-Aware Greedy'],
              ['ppo',       'PPO Agent'],
            ] as [RoutingMode, string][]
          ).map(([mode, label]) => (
            <label key={mode} className="radio-label">
              <input
                type="radio"
                name="routing"
                value={mode}
                checked={settings.routingMode === mode}
                onChange={() => updateSettings({ routingMode: mode })}
              />
              {label}
            </label>
          ))}
        </div>
      </div>

      {/* PPO model management — visible only when PPO mode is selected */}
      {settings.routingMode === 'ppo' && (
        <div className="control-group" style={{ gap: 5 }}>
          <label style={{ fontSize: 10, color: 'var(--accent2)', fontWeight: 'bold' }}>
            PPO Model
          </label>

          {/* Export: job_id → ONNX */}
          <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
            <input
              type="text"
              className="form-input"
              style={{ fontSize: 9, flex: 1, minWidth: 0 }}
              placeholder="job id to export"
              value={exportJobId}
              onChange={(e) => setExportJobId(e.target.value)}
            />
            <button
              className="btn btn-sm"
              onClick={handleExport}
              disabled={!exportJobId || exportPending}
              style={{ flexShrink: 0 }}
            >
              {exportPending ? '…' : 'Export'}
            </button>
          </div>
          {exportError && (
            <span className="status-err" style={{ fontSize: 9 }}>{exportError}</span>
          )}

          {/* Model selector */}
          {ppoModels.length > 0 ? (
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="form-input"
              style={{ fontSize: 9 }}
            >
              {ppoModels.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          ) : (
            <span style={{ fontSize: 9, color: 'var(--text-dim)' }}>
              No exported models yet
            </span>
          )}

          <button
            className="btn btn-primary btn-sm"
            onClick={handleLoadModel}
            disabled={!selectedModel || loadPending}
          >
            {loadPending
              ? 'Loading…'
              : loadStatus === 'loaded' && loadedId === selectedModel
              ? '✓ Loaded'
              : 'Load Model'}
          </button>
          {loadStatus === 'error' && (
            <span className="status-err" style={{ fontSize: 9 }}>{loadError}</span>
          )}
          {loadStatus === 'loaded' && loadedId === selectedModel && (
            <span className="status-ok" style={{ fontSize: 9 }}>Model active</span>
          )}
          {settings.routingMode === 'ppo' && !loadedId && (
            <span style={{ fontSize: 9, color: 'var(--accent2)' }}>
              No model loaded — using greedy fallback
            </span>
          )}
          {settings.routingMode === 'ppo' && loadedId && loadedId !== selectedModel && (
            <span style={{ fontSize: 9, color: 'var(--text-dim)' }}>
              (different model active — reload to switch)
            </span>
          )}
        </div>
      )}

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
