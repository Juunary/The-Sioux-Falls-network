// ============================================================
// ComparisonPage — compare two experiments side by side
// ============================================================

import React, { useState, useEffect, useCallback } from 'react';
import {
  listExperiments,
  getAggregate,
  type AggregateStats,
} from '../api/experiments';
import ComparisonChart from '../components/ComparisonChart';

const EMPTY_STATS: AggregateStats = {
  count: 0,
  avg_reward: 0,
  avg_service_rate: 0,
  avg_wait_time_sec: 0,
};

export default function ComparisonPage() {
  const [experiments, setExperiments] = useState<Record<string, unknown>[]>([]);
  const [idA, setIdA] = useState('');
  const [idB, setIdB] = useState('');
  const [statsA, setStatsA] = useState<AggregateStats | null>(null);
  const [statsB, setStatsB] = useState<AggregateStats | null>(null);
  const [loadingA, setLoadingA] = useState(false);
  const [loadingB, setLoadingB] = useState(false);
  const [errorA, setErrorA] = useState<string | null>(null);
  const [errorB, setErrorB] = useState<string | null>(null);

  const loadExperiments = useCallback(async () => {
    try {
      const res = await listExperiments();
      setExperiments(res.experiments);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => { loadExperiments(); }, [loadExperiments]);

  async function loadStats(
    id: string,
    setStats: (s: AggregateStats) => void,
    setLoading: (v: boolean) => void,
    setError: (e: string | null) => void,
  ) {
    if (!id) return;
    setLoading(true);
    setError(null);
    try {
      const s = await getAggregate(id);
      setStats(s);
    } catch (e) {
      setError(String(e));
      setStats(EMPTY_STATS);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (idA) loadStats(idA, setStatsA, setLoadingA, setErrorA);
    else setStatsA(null);
  }, [idA]);

  useEffect(() => {
    if (idB) loadStats(idB, setStatsB, setLoadingB, setErrorB);
    else setStatsB(null);
  }, [idB]);

  const expOptions = experiments.map((e) => ({
    id: e['experiment_id'] as string,
    label: `${e['experiment_id']} (${e['policy']}, ${e['split']})`,
  }));

  const NA = <span style={{ color: '#7a7a9a' }}>N/A</span>;

  // naIfZero=true: show N/A when value is 0 (used for known-incomplete metrics)
  function statRow(
    label: string,
    valA: number | null | undefined,
    valB: number | null | undefined,
    fmt?: (v: number) => string,
    naIfZero = false,
  ) {
    const f = fmt ?? ((v: number) => v.toFixed(3));
    const cell = (v: number | null | undefined) => {
      if (v == null) return NA;
      if (naIfZero && v === 0) return NA;
      return f(v);
    };
    return (
      <tr>
        <td className="stat-label">{label}</td>
        <td className="stat-val">{cell(valA)}</td>
        <td className="stat-val">{cell(valB)}</td>
      </tr>
    );
  }

  return (
    <div className="page-scrollable">
      <div className="page-inner">
        <h1 className="page-title">Comparison</h1>

        <section className="card">
          <h2 className="card-title">Select Experiments</h2>
          {experiments.length === 0 && (
            <div className="status-info">
              No experiments available. Run evaluations on the Experiments page first.
            </div>
          )}
          <div className="form-grid">
            <label className="form-label">
              Experiment A
              <select value={idA} onChange={(e) => setIdA(e.target.value)} className="form-input">
                <option value="">— select —</option>
                {expOptions.filter((o) => o.id !== idB).map((o) => (
                  <option key={o.id} value={o.id}>{o.label}</option>
                ))}
              </select>
              {errorA && <span className="status-err">{errorA}</span>}
            </label>
            <label className="form-label">
              Experiment B
              <select value={idB} onChange={(e) => setIdB(e.target.value)} className="form-input">
                <option value="">— select —</option>
                {expOptions.filter((o) => o.id !== idA).map((o) => (
                  <option key={o.id} value={o.id}>{o.label}</option>
                ))}
              </select>
              {errorB && <span className="status-err">{errorB}</span>}
            </label>
          </div>
        </section>

        {statsA && statsB && (
          <>
            <section className="card">
              <h2 className="card-title">Aggregate Comparison</h2>
              <div className="comparison-grid">
                <table className="data-table comparison-table">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>{idA.slice(0, 20)}…</th>
                      <th>{idB.slice(0, 20)}…</th>
                    </tr>
                  </thead>
                  <tbody>
                    {statRow('Scenarios', statsA.count, statsB.count, (v) => String(Math.round(v)))}
                    {statRow('Avg Reward', statsA.avg_reward, statsB.avg_reward, (v) => v.toFixed(2))}
                    {statRow('Avg Service Rate', statsA.avg_service_rate, statsB.avg_service_rate,
                      (v) => (v * 100).toFixed(1) + '%')}
                    {statRow('Avg Wait (s)', statsA.avg_wait_time_sec, statsB.avg_wait_time_sec,
                      (v) => v.toFixed(1), true)}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="card">
              <h2 className="card-title">Chart</h2>
              {(loadingA || loadingB) && <div className="status-info">Loading...</div>}
              <ComparisonChart
                labelA={idA.slice(0, 18)}
                labelB={idB.slice(0, 18)}
                statsA={statsA}
                statsB={statsB}
              />
            </section>
          </>
        )}

        {(idA && !statsA && !loadingA) && (
          <div className="status-info">
            No aggregate data for experiment A yet. Evaluation may still be running.
          </div>
        )}
      </div>
    </div>
  );
}
