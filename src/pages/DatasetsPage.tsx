// ============================================================
// DatasetsPage — generate and browse scenario files
// ============================================================

import React, { useState, useEffect, useCallback } from 'react';
import {
  generateScenarios,
  listScenarios,
  type ScenarioListItem,
  type GenerateRequest,
} from '../api/datasets';

const DEFAULT_GEN: GenerateRequest = {
  count: 10,
  seed_start: 0,
  split: 'train',
  difficulty: 'medium',
  bus_count: 6,
  episode_length: 600,
  spawn_rate: 0.8,
  passenger_capacity: 6,
  reboard_enabled: false,
  charging_enabled: true,
  low_battery_threshold: 20,
  charge_duration: 8,
  battery_drain_rate: 0.08,
};

export default function DatasetsPage() {
  const [splitFilter, setSplitFilter] = useState<string>('');
  const [scenarios, setScenarios] = useState<ScenarioListItem[]>([]);
  const [loadingList, setLoadingList] = useState(false);
  const [listError, setListError] = useState<string | null>(null);

  const [genConfig, setGenConfig] = useState<GenerateRequest>(DEFAULT_GEN);
  const [generating, setGenerating] = useState(false);
  const [genResult, setGenResult] = useState<string | null>(null);
  const [genError, setGenError] = useState<string | null>(null);

  const loadScenarios = useCallback(async () => {
    setLoadingList(true);
    setListError(null);
    try {
      const res = await listScenarios(splitFilter || undefined);
      setScenarios(res.scenarios);
    } catch (e) {
      setListError(String(e));
    } finally {
      setLoadingList(false);
    }
  }, [splitFilter]);

  useEffect(() => { loadScenarios(); }, [loadScenarios]);

  async function handleGenerate(e: React.FormEvent) {
    e.preventDefault();
    setGenerating(true);
    setGenResult(null);
    setGenError(null);
    try {
      const res = await generateScenarios(genConfig);
      setGenResult(`Generated ${res.generated} scenarios → ${res.output_dir}`);
      loadScenarios();
    } catch (e) {
      setGenError(String(e));
    } finally {
      setGenerating(false);
    }
  }

  function setField<K extends keyof GenerateRequest>(key: K, value: GenerateRequest[K]) {
    setGenConfig((c) => ({ ...c, [key]: value }));
  }

  const splitCounts = scenarios.reduce<Record<string, number>>((acc, s) => {
    acc[s.split] = (acc[s.split] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="page-scrollable">
      <div className="page-inner">
        <h1 className="page-title">Datasets</h1>

        <section className="card">
          <h2 className="card-title">Generate Scenarios</h2>
          <form onSubmit={handleGenerate}>
            <div className="form-grid">
              <label className="form-label">
                Count
                <input type="number" min={1} max={1000} value={genConfig.count}
                  onChange={(e) => setField('count', parseInt(e.target.value))}
                  className="form-input" />
              </label>
              <label className="form-label">
                Seed start
                <input type="number" min={0} value={genConfig.seed_start}
                  onChange={(e) => setField('seed_start', parseInt(e.target.value))}
                  className="form-input" />
              </label>
              <label className="form-label">
                Split
                <select value={genConfig.split}
                  onChange={(e) => setField('split', e.target.value as GenerateRequest['split'])}
                  className="form-input">
                  <option value="train">train</option>
                  <option value="val">val</option>
                  <option value="test">test</option>
                </select>
              </label>
              <label className="form-label">
                Difficulty
                <select value={genConfig.difficulty}
                  onChange={(e) => setField('difficulty', e.target.value as GenerateRequest['difficulty'])}
                  className="form-input">
                  <option value="easy">easy</option>
                  <option value="medium">medium</option>
                  <option value="hard">hard</option>
                  <option value="stress">stress</option>
                </select>
              </label>
              <label className="form-label">
                Bus count
                <input type="number" min={1} max={20} value={genConfig.bus_count}
                  onChange={(e) => setField('bus_count', parseInt(e.target.value))}
                  className="form-input" />
              </label>
              <label className="form-label">
                Episode length (s)
                <input type="number" min={10} value={genConfig.episode_length}
                  onChange={(e) => setField('episode_length', parseFloat(e.target.value))}
                  className="form-input" />
              </label>
              <label className="form-label">
                Spawn rate
                <input type="number" min={0.01} step={0.1} value={genConfig.spawn_rate}
                  onChange={(e) => setField('spawn_rate', parseFloat(e.target.value))}
                  className="form-input" />
              </label>
              <label className="form-label">
                Pax capacity
                <input type="number" min={1} max={30} value={genConfig.passenger_capacity}
                  onChange={(e) => setField('passenger_capacity', parseInt(e.target.value))}
                  className="form-input" />
              </label>
            </div>
            <div className="form-actions">
              <button type="submit" className="btn btn-primary" disabled={generating}>
                {generating ? 'Generating...' : 'Generate'}
              </button>
              {genResult && <span className="status-ok">{genResult}</span>}
              {genError  && <span className="status-err">{genError}</span>}
            </div>
          </form>
        </section>

        <section className="card">
          <div className="card-header-row">
            <h2 className="card-title">Existing Scenarios</h2>
            <div className="header-actions">
              <select value={splitFilter}
                onChange={(e) => setSplitFilter(e.target.value)}
                className="form-input form-input-sm">
                <option value="">All splits</option>
                <option value="train">train ({splitCounts['train'] ?? 0})</option>
                <option value="val">val ({splitCounts['val'] ?? 0})</option>
                <option value="test">test ({splitCounts['test'] ?? 0})</option>
              </select>
              <button className="btn" onClick={loadScenarios} disabled={loadingList}>Refresh</button>
            </div>
          </div>
          {loadingList && <div className="status-info">Loading...</div>}
          {listError   && <div className="status-err">{listError}</div>}
          {!loadingList && scenarios.length === 0 && (
            <div className="status-info">No scenarios found. Generate some first.</div>
          )}
          {scenarios.length > 0 && (
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>ID</th><th>Split</th><th>Difficulty</th>
                    <th>Seed</th><th>Buses</th><th>Length (s)</th><th>Events</th>
                  </tr>
                </thead>
                <tbody>
                  {scenarios.map((s) => (
                    <tr key={s.scenario_id}>
                      <td className="mono">{s.scenario_id}</td>
                      <td><span className={`badge badge-${s.split}`}>{s.split}</span></td>
                      <td>{s.difficulty}</td>
                      <td className="mono">{s.seed}</td>
                      <td>{s.bus_count}</td>
                      <td>{s.episode_length}</td>
                      <td>{s.event_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
