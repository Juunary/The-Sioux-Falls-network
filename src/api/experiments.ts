// ============================================================
// Experiments API — /api/experiments/*
// ============================================================

import { apiFetch } from './client';

export interface EvaluateRequest {
  policy?: string;
  split?: string;
  max_scenarios?: number | null;
}

export interface EvaluateResponse {
  experiment_id: string;
  status: string;
  message: string;
}

export interface MetricRow {
  scenario_id: string;
  policy: string;
  episode_reward: number;
  passengers_spawned: number;
  passengers_served: number;
  passengers_gone: number;
  service_rate: number;
  unserved_rate: number;
  avg_wait_time_sec: number;   // may be 0.0 (not yet implemented)
  charge_events: number;
  total_distance_px: number;   // may be 0.0 (not yet implemented)
}

export interface ExperimentDetail {
  experiment: Record<string, unknown>;
  metrics: MetricRow[];
}

export interface AggregateStats {
  count: number;
  avg_reward: number;
  avg_service_rate: number;
  avg_wait_time_sec: number;   // may be 0.0 (not yet implemented in evaluator.py)
  [key: string]: unknown;
}

export function runEvaluation(req: EvaluateRequest = {}): Promise<EvaluateResponse> {
  return apiFetch('/experiments/evaluate', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export function listExperiments(): Promise<{ experiments: Record<string, unknown>[] }> {
  return apiFetch('/experiments/');
}

export function getExperiment(id: string): Promise<ExperimentDetail> {
  return apiFetch(`/experiments/${id}`);
}

export function getAggregate(id: string): Promise<AggregateStats> {
  return apiFetch(`/experiments/${id}/aggregate`);
}
