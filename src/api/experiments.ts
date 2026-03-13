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
  avg_wait_time_sec: number;
  charge_events: number;
  total_distance_px: number;
  avg_in_vehicle_sec: number;
  avg_detour_px: number;
}

export interface ExperimentDetail {
  experiment: Record<string, unknown>;
  metrics: MetricRow[];
}

export interface AggregateStats {
  count: number;
  avg_reward: number;
  // SF fields
  avg_service_rate?: number;
  avg_wait_time_sec?: number;
  avg_detour_px?: number;
  // DRT fields
  domain?: string;
  avg_serve_rate?: number;
  avg_cancel_rate?: number;
  avg_wait_ticks?: number;
  avg_ivt_ticks?: number;
  avg_detour_ticks?: number;
  [key: string]: unknown;
}

export interface DRTEvaluateRequest {
  requests_path?: string;
  vehicle_positions_path?: string;
  od_matrix_path?: string;
  episode_id?: string;
  output_csv?: boolean;
}

export interface DRTBatchEvaluateRequest {
  requests_paths: string[];
  vehicle_positions_path?: string;
  od_matrix_path?: string;
  /** 'greedy' | 'ppo' */
  policy_type?: string;
  /** Required when policy_type === 'ppo' */
  job_id?: string;
  output_csv?: boolean;
}

export interface ExportModelResult {
  model_id: string;
  onnx_path: string;
  input_shape: number[];
  output_shape: number[];
  opset_version: number;
}

export function runEvaluation(req: EvaluateRequest = {}): Promise<EvaluateResponse> {
  return apiFetch('/experiments/evaluate', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export function runDRTEvaluation(req: DRTEvaluateRequest = {}): Promise<EvaluateResponse> {
  return apiFetch('/experiments/evaluate/drt', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export function runDRTBatchEvaluation(req: DRTBatchEvaluateRequest): Promise<EvaluateResponse> {
  return apiFetch('/experiments/evaluate/drt/batch', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export function exportModel(jobId: string): Promise<ExportModelResult> {
  return apiFetch(`/models/${jobId}/export_onnx`, { method: 'POST' });
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
