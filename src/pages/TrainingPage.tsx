// ============================================================
// TrainingPage — configure, start, and monitor PPO training
// ============================================================

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  startTraining,
  stopTraining,
  listJobs,
  connectMetricsWs,
  type TrainingConfig,
  type JobInfo,
  type MetricsRecord,
} from '../api/training';
import TrainingChart from '../components/TrainingChart';

const DEFAULT_CONFIG: TrainingConfig = {
  split: 'train',
  total_timesteps: 100_000,
  n_envs: 2,
  learning_rate: 3e-4,
  gamma: 0.95,
  clip_range: 0.2,
  ent_coef: 0.01,
  n_steps: 2048,
  batch_size: 256,
  checkpoint_freq: 50_000,
  seed: 0,
};

function fmtStatus(s: string) {
  const colors: Record<string, string> = {
    running: '#2ecc71',
    completed: '#3498db',
    failed: '#e74c3c',
    pending: '#f39c12',
  };
  return <span style={{ color: colors[s] ?? '#c8cce8' }}>{s}</span>;
}

export default function TrainingPage() {
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_CONFIG);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [metricsHistory, setMetricsHistory] = useState<MetricsRecord[]>([]);

  const wsRef = useRef<WebSocket | null>(null);

  const refreshJobs = useCallback(async () => {
    try {
      const list = await listJobs();
      setJobs(list);
    } catch {
      // ignore — backend may not be running
    }
  }, []);

  // Poll job list every 5s
  useEffect(() => {
    refreshJobs();
    const id = setInterval(refreshJobs, 5000);
    return () => clearInterval(id);
  }, [refreshJobs]);

  // WebSocket: open/close only when selectedJobId changes (not on every jobs poll)
  useEffect(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setMetricsHistory([]);

    if (!selectedJobId) return;

    const ws = connectMetricsWs(
      selectedJobId,
      (r) => setMetricsHistory((h) => [...h, r]),
      () => { wsRef.current = null; },
    );
    wsRef.current = ws;
    return () => { ws.close(); };
  }, [selectedJobId]);

  // Close WS when selected job transitions out of 'running' (detected by polling)
  useEffect(() => {
    if (!selectedJobId || !wsRef.current) return;
    const job = jobs.find((j) => j.job_id === selectedJobId);
    if (job && job.status !== 'running') {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, [jobs, selectedJobId]);

  function setField<K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) {
    setConfig((c) => ({ ...c, [key]: value }));
  }

  async function handleStart(e: React.FormEvent) {
    e.preventDefault();
    setStarting(true);
    setStartError(null);
    try {
      const res = await startTraining(config);
      setSelectedJobId(res.job_id);
      await refreshJobs();
    } catch (e) {
      setStartError(String(e));
    } finally {
      setStarting(false);
    }
  }

  async function handleStop(jobId: string) {
    try {
      await stopTraining(jobId);
      await refreshJobs();
    } catch (e) {
      alert(String(e));
    }
  }

  const selectedJob = jobs.find((j) => j.job_id === selectedJobId) ?? null;

  return (
    <div className="page-scrollable">
      <div className="page-inner">
        <h1 className="page-title">Training</h1>

        <section className="card">
          <h2 className="card-title">Start New Job</h2>
          <form onSubmit={handleStart}>
            <div className="form-grid">
              <label className="form-label">
                Split
                <select value={config.split}
                  onChange={(e) => setField('split', e.target.value)}
                  className="form-input">
                  <option value="train">train</option>
                  <option value="val">val</option>
                </select>
              </label>
              <label className="form-label">
                Total timesteps
                <input type="number" min={1000} step={1000} value={config.total_timesteps}
                  onChange={(e) => setField('total_timesteps', parseInt(e.target.value))}
                  className="form-input" />
              </label>
              <label className="form-label">
                n_envs
                <input type="number" min={1} max={16} value={config.n_envs}
                  onChange={(e) => setField('n_envs', parseInt(e.target.value))}
                  className="form-input" />
              </label>
              <label className="form-label">
                Learning rate
                <input type="number" min={1e-6} max={1} step={1e-5} value={config.learning_rate}
                  onChange={(e) => setField('learning_rate', parseFloat(e.target.value))}
                  className="form-input" />
              </label>
              <label className="form-label">
                Gamma
                <input type="number" min={0} max={1} step={0.01} value={config.gamma}
                  onChange={(e) => setField('gamma', parseFloat(e.target.value))}
                  className="form-input" />
              </label>
              <label className="form-label">
                Clip range
                <input type="number" min={0.01} max={0.5} step={0.01} value={config.clip_range}
                  onChange={(e) => setField('clip_range', parseFloat(e.target.value))}
                  className="form-input" />
              </label>
              <label className="form-label">
                Entropy coef
                <input type="number" min={0} max={0.1} step={0.001} value={config.ent_coef}
                  onChange={(e) => setField('ent_coef', parseFloat(e.target.value))}
                  className="form-input" />
              </label>
              <label className="form-label">
                n_steps
                <input type="number" min={64} step={64} value={config.n_steps}
                  onChange={(e) => setField('n_steps', parseInt(e.target.value))}
                  className="form-input" />
              </label>
              <label className="form-label">
                Batch size
                <input type="number" min={32} step={32} value={config.batch_size}
                  onChange={(e) => setField('batch_size', parseInt(e.target.value))}
                  className="form-input" />
              </label>
              <label className="form-label">
                Checkpoint freq
                <input type="number" min={1000} step={1000} value={config.checkpoint_freq}
                  onChange={(e) => setField('checkpoint_freq', parseInt(e.target.value))}
                  className="form-input" />
              </label>
              <label className="form-label">
                Seed
                <input type="number" min={0} value={config.seed}
                  onChange={(e) => setField('seed', parseInt(e.target.value))}
                  className="form-input" />
              </label>
            </div>
            <div className="form-actions">
              <button type="submit" className="btn btn-primary" disabled={starting}>
                {starting ? 'Starting...' : 'Start Training'}
              </button>
              {startError && <span className="status-err">{startError}</span>}
            </div>
          </form>
        </section>

        {/* Job list */}
        {jobs.length > 0 && (
          <section className="card">
            <div className="card-header-row">
              <h2 className="card-title">Jobs</h2>
              <button className="btn" onClick={refreshJobs}>Refresh</button>
            </div>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Job ID</th><th>Status</th><th>Split</th>
                    <th>Timesteps</th><th>n_envs</th><th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map((j) => (
                    <tr key={j.job_id}
                      className={j.job_id === selectedJobId ? 'row-selected' : ''}
                      onClick={() => setSelectedJobId(j.job_id)}
                      style={{ cursor: 'pointer' }}>
                      <td className="mono">{j.job_id}</td>
                      <td>{fmtStatus(j.status)}</td>
                      <td>{j.config.split ?? '—'}</td>
                      <td>{j.config.total_timesteps?.toLocaleString() ?? '—'}</td>
                      <td>{j.config.n_envs ?? '—'}</td>
                      <td>
                        {j.status === 'running' && (
                          <button className="btn btn-danger btn-sm"
                            onClick={(e) => { e.stopPropagation(); handleStop(j.job_id); }}>
                            Stop
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* Live metrics for selected job */}
        {selectedJob && (
          <section className="card">
            <h2 className="card-title">
              Metrics — {selectedJob.job_id}
              {selectedJob.error && (
                <span className="status-err" style={{ marginLeft: 8, fontWeight: 'normal', fontSize: 11 }}>
                  Error: {selectedJob.error}
                </span>
              )}
            </h2>
            <div className="metrics-summary">
              <span>Status: {fmtStatus(selectedJob.status)}</span>
              <span>Rollouts: {metricsHistory.length}</span>
              {metricsHistory.length > 0 && metricsHistory[metricsHistory.length - 1].reward !== null && (
                <span>
                  Last reward: {(metricsHistory[metricsHistory.length - 1].reward as number).toFixed(2)}
                </span>
              )}
            </div>
            <TrainingChart data={metricsHistory} />
            {selectedJob.status !== 'running' && metricsHistory.length === 0 && (
              <div className="status-info" style={{ marginTop: 8 }}>
                No metrics collected yet for this job.
              </div>
            )}
          </section>
        )}
      </div>
    </div>
  );
}
