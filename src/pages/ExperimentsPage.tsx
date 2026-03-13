// ============================================================
// ExperimentsPage — run and browse baseline evaluations
// ============================================================

import React, { useState, useEffect, useCallback } from 'react';
import {
  runEvaluation,
  runDRTEvaluation,
  runDRTBatchEvaluation,
  listExperiments,
  getExperiment,
  type MetricRow,
} from '../api/experiments';

const NA = <span style={{ color: '#7a7a9a' }}>N/A</span>;

function fmtNum(v: number | undefined | null, decimals = 3): React.ReactNode {
  if (v === undefined || v === null) return NA;
  return v.toFixed(decimals);
}

export default function ExperimentsPage() {
  const [policy, setPolicy] = useState('demand_aware_greedy_v1');
  const [split, setSplit] = useState('test');
  const [maxScenarios, setMaxScenarios] = useState<number | ''>('');

  const [running, setRunning] = useState(false);
  const [runResult, setRunResult] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  // DRT batch eval form state
  const [batchPaths, setBatchPaths] = useState(
    'KW_DRT/data/requests_8.csv,KW_DRT/data/requests_80.csv',
  );
  const [batchPolicyType, setBatchPolicyType] = useState<'greedy' | 'ppo'>('greedy');
  const [batchJobId, setBatchJobId] = useState('');
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchResult, setBatchResult] = useState<string | null>(null);
  const [batchError, setBatchError] = useState<string | null>(null);

  // DRT single eval form state
  const [drtRequestsPath, setDrtRequestsPath] = useState('KW_DRT/data/requests_8.csv');
  const [drtVehiclePath, setDrtVehiclePath] = useState('KW_DRT/data/vehicle_positions.csv');
  const [drtOdPath, setDrtOdPath] = useState('KW_DRT/data/od_matrix.csv');
  const [drtEpisodeId, setDrtEpisodeId] = useState('drt_eval');
  const [drtRunning, setDrtRunning] = useState(false);
  const [drtResult, setDrtResult] = useState<string | null>(null);
  const [drtError, setDrtError] = useState<string | null>(null);

  const [experiments, setExperiments] = useState<Record<string, unknown>[]>([]);
  const [loadingExps, setLoadingExps] = useState(false);

  const [selectedExpId, setSelectedExpId] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const [loadingMetrics, setLoadingMetrics] = useState(false);

  const loadExperiments = useCallback(async () => {
    setLoadingExps(true);
    try {
      const res = await listExperiments();
      setExperiments(res.experiments);
    } catch {
      // backend may not be running
    } finally {
      setLoadingExps(false);
    }
  }, []);

  useEffect(() => { loadExperiments(); }, [loadExperiments]);

  async function loadMetrics(expId: string) {
    setLoadingMetrics(true);
    setMetrics([]);
    try {
      const res = await getExperiment(expId);
      setMetrics(res.metrics);
    } catch {
      setMetrics([]);
    } finally {
      setLoadingMetrics(false);
    }
  }

  async function handleRun(e: React.FormEvent) {
    e.preventDefault();
    setRunning(true);
    setRunResult(null);
    setRunError(null);
    try {
      const res = await runEvaluation({
        policy,
        split,
        max_scenarios: maxScenarios === '' ? null : maxScenarios,
      });
      setRunResult(`Started: ${res.experiment_id}`);
      setTimeout(loadExperiments, 1500);
    } catch (e) {
      setRunError(String(e));
    } finally {
      setRunning(false);
    }
  }

  async function handleBatchRun(e: React.FormEvent) {
    e.preventDefault();
    setBatchRunning(true);
    setBatchResult(null);
    setBatchError(null);
    try {
      const paths = batchPaths.split(',').map((p) => p.trim()).filter(Boolean);
      const res = await runDRTBatchEvaluation({
        requests_paths: paths,
        policy_type: batchPolicyType,
        job_id: batchPolicyType === 'ppo' ? batchJobId : undefined,
        output_csv: true,
      });
      setBatchResult(`Started: ${res.experiment_id}`);
      setTimeout(loadExperiments, 2000);
    } catch (e) {
      setBatchError(String(e));
    } finally {
      setBatchRunning(false);
    }
  }

  async function handleDRTRun(e: React.FormEvent) {
    e.preventDefault();
    setDrtRunning(true);
    setDrtResult(null);
    setDrtError(null);
    try {
      const res = await runDRTEvaluation({
        requests_path: drtRequestsPath,
        vehicle_positions_path: drtVehiclePath,
        od_matrix_path: drtOdPath,
        episode_id: drtEpisodeId,
        output_csv: true,
      });
      setDrtResult(`Started: ${res.experiment_id}`);
      setTimeout(loadExperiments, 2000);
    } catch (e) {
      setDrtError(String(e));
    } finally {
      setDrtRunning(false);
    }
  }

  function handleSelectExp(id: string) {
    setSelectedExpId(id);
    loadMetrics(id);
  }

  return (
    <div className="page-scrollable">
      <div className="page-inner">
        <h1 className="page-title">Experiments</h1>

        <section className="card">
          <h2 className="card-title">Run Evaluation</h2>
          <form onSubmit={handleRun}>
            <div className="form-grid">
              <label className="form-label">
                Policy
                <select value={policy} onChange={(e) => setPolicy(e.target.value)}
                  className="form-input">
                  <option value="demand_aware_greedy_v1">demand_aware_greedy_v1</option>
                </select>
              </label>
              <label className="form-label">
                Split
                <select value={split} onChange={(e) => setSplit(e.target.value)}
                  className="form-input">
                  <option value="train">train</option>
                  <option value="val">val</option>
                  <option value="test">test</option>
                </select>
              </label>
              <label className="form-label">
                Max scenarios (blank = all)
                <input type="number" min={1} value={maxScenarios}
                  onChange={(e) => setMaxScenarios(e.target.value === '' ? '' : parseInt(e.target.value))}
                  className="form-input" placeholder="all" />
              </label>
            </div>
            <div className="form-actions">
              <button type="submit" className="btn btn-primary" disabled={running}>
                {running ? 'Starting...' : 'Run Evaluation'}
              </button>
              {runResult && <span className="status-ok">{runResult}</span>}
              {runError  && <span className="status-err">{runError}</span>}
            </div>
          </form>
        </section>

        <section className="card">
          <h2 className="card-title">Run DRT Evaluation</h2>
          <form onSubmit={handleDRTRun}>
            <div className="form-grid">
              <label className="form-label">
                Requests CSV
                <input type="text" value={drtRequestsPath}
                  onChange={(e) => setDrtRequestsPath(e.target.value)}
                  className="form-input" />
              </label>
              <label className="form-label">
                Vehicle Positions CSV
                <input type="text" value={drtVehiclePath}
                  onChange={(e) => setDrtVehiclePath(e.target.value)}
                  className="form-input" />
              </label>
              <label className="form-label">
                OD Matrix CSV
                <input type="text" value={drtOdPath}
                  onChange={(e) => setDrtOdPath(e.target.value)}
                  className="form-input" />
              </label>
              <label className="form-label">
                Episode ID
                <input type="text" value={drtEpisodeId}
                  onChange={(e) => setDrtEpisodeId(e.target.value)}
                  className="form-input" />
              </label>
            </div>
            <div className="form-actions">
              <button type="submit" className="btn btn-primary" disabled={drtRunning}>
                {drtRunning ? 'Starting...' : 'Run DRT Eval'}
              </button>
              {drtResult && <span className="status-ok">{drtResult}</span>}
              {drtError  && <span className="status-err">{drtError}</span>}
            </div>
          </form>
        </section>

        <section className="card">
          <h2 className="card-title">Run DRT Batch Evaluation</h2>
          <form onSubmit={handleBatchRun}>
            <div className="form-grid">
              <label className="form-label" style={{ gridColumn: '1 / -1' }}>
                Requests CSV paths (comma-separated)
                <input type="text" value={batchPaths}
                  onChange={(e) => setBatchPaths(e.target.value)}
                  className="form-input" placeholder="KW_DRT/data/requests_8.csv, KW_DRT/data/requests_80.csv" />
              </label>
              <label className="form-label">
                Policy type
                <select value={batchPolicyType}
                  onChange={(e) => setBatchPolicyType(e.target.value as 'greedy' | 'ppo')}
                  className="form-input">
                  <option value="greedy">greedy (drt_greedy_v1)</option>
                  <option value="ppo">ppo (trained checkpoint)</option>
                </select>
              </label>
              {batchPolicyType === 'ppo' && (
                <label className="form-label">
                  Job ID
                  <input type="text" value={batchJobId}
                    onChange={(e) => setBatchJobId(e.target.value)}
                    className="form-input" placeholder="job_20240101_120000_abc123" />
                </label>
              )}
            </div>
            <div className="form-actions">
              <button type="submit" className="btn btn-primary" disabled={batchRunning}>
                {batchRunning ? 'Starting…' : 'Run Batch Eval'}
              </button>
              {batchResult && <span className="status-ok">{batchResult}</span>}
              {batchError  && <span className="status-err">{batchError}</span>}
            </div>
          </form>
        </section>

        <section className="card">
          <div className="card-header-row">
            <h2 className="card-title">Past Experiments</h2>
            <button className="btn" onClick={loadExperiments} disabled={loadingExps}>Refresh</button>
          </div>
          {loadingExps && <div className="status-info">Loading...</div>}
          {!loadingExps && experiments.length === 0 && (
            <div className="status-info">No experiments yet.</div>
          )}
          {experiments.length > 0 && (
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr><th>Experiment ID</th><th>Domain</th><th>Policy</th><th>Split</th><th>Created</th></tr>
                </thead>
                <tbody>
                  {experiments.map((exp) => {
                    const id = exp['experiment_id'] as string;
                    const domain = (exp['domain'] as string | undefined) ?? 'sf';
                    return (
                      <tr key={id}
                        className={id === selectedExpId ? 'row-selected' : ''}
                        onClick={() => handleSelectExp(id)}
                        style={{ cursor: 'pointer' }}>
                        <td className="mono">{id}</td>
                        <td><span style={{ fontWeight: domain === 'drt' ? 600 : 400 }}>{domain}</span></td>
                        <td>{exp['policy'] as string}</td>
                        <td>{exp['split'] as string}</td>
                        <td className="mono">{exp['created_at'] as string}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {selectedExpId && (
          <section className="card">
            <h2 className="card-title">Metrics — {selectedExpId}</h2>
            {loadingMetrics && <div className="status-info">Loading metrics...</div>}
            {!loadingMetrics && metrics.length === 0 && (
              <div className="status-info">No metrics yet (evaluation may still be running).</div>
            )}
            {metrics.length > 0 && (
              <>
                <div className="metrics-summary">
                  <span>Scenarios: {metrics.length}</span>
                  <span>
                    Avg reward: {(metrics.reduce((s, m) => s + m.episode_reward, 0) / metrics.length).toFixed(2)}
                  </span>
                  <span>
                    Avg service rate:{' '}
                    {(metrics.reduce((s, m) => s + m.service_rate, 0) / metrics.length * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Scenario</th>
                        <th>Reward</th>
                        <th>Spawned</th>
                        <th>Served</th>
                        <th>Gone</th>
                        <th>Service %</th>
                        <th>Avg Wait</th>
                        <th>Avg Detour (px)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {metrics.map((m) => (
                        <tr key={m.scenario_id}>
                          <td className="mono">{m.scenario_id}</td>
                          <td>{m.episode_reward.toFixed(1)}</td>
                          <td>{m.passengers_spawned}</td>
                          <td>{m.passengers_served}</td>
                          <td>{m.passengers_gone}</td>
                          <td>{(m.service_rate * 100).toFixed(1)}%</td>
                          <td>{fmtNum(m.avg_wait_time_sec, 1)}</td>
                          <td>{fmtNum(m.avg_detour_px, 1)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </section>
        )}
      </div>
    </div>
  );
}
