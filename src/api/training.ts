// ============================================================
// Training API — /api/training/*
// ============================================================

import { apiFetch, WS_BASE } from './client';

export interface TrainingConfig {
  split?: string;
  total_timesteps?: number;
  n_envs?: number;
  learning_rate?: number;
  gamma?: number;
  clip_range?: number;
  ent_coef?: number;
  n_steps?: number;
  batch_size?: number;
  checkpoint_freq?: number;
  seed?: number;
  verbose?: number;
}

export type JobStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface JobInfo {
  job_id: string;
  config: TrainingConfig;
  status: JobStatus;
  pid: number | null;
  started_at: number;
  ended_at: number | null;
  error: string | null;
}

/** One row written to metrics.jsonl by MetricsCallback. */
export interface MetricsRecord {
  timestep: number;
  episode: number;
  reward: number | null;
  ep_len: number | null;
  wall_time: number;
}

export function startTraining(config: TrainingConfig): Promise<{ job_id: string; status: string }> {
  return apiFetch('/training/start', {
    method: 'POST',
    body: JSON.stringify(config),
  });
}

export function stopTraining(jobId: string): Promise<{ job_id: string; status: string }> {
  return apiFetch(`/training/${jobId}/stop`, { method: 'POST' });
}

export function listJobs(): Promise<JobInfo[]> {
  return apiFetch('/training/');
}

export function getJobStatus(jobId: string): Promise<JobInfo> {
  return apiFetch(`/training/${jobId}/status`);
}

/**
 * Open a WebSocket to stream metrics from a running job.
 * Returns the WebSocket — caller is responsible for closing it.
 */
export function connectMetricsWs(
  jobId: string,
  onRecord: (r: MetricsRecord) => void,
  onClose?: () => void,
): WebSocket {
  const ws = new WebSocket(`${WS_BASE}/training/${jobId}/ws`);
  ws.onmessage = (e) => {
    try {
      const r = JSON.parse(e.data as string) as MetricsRecord & { type?: string };
      if (r.type !== 'ping') onRecord(r);
    } catch {
      // ignore malformed
    }
  };
  ws.onclose = () => onClose?.();
  return ws;
}
