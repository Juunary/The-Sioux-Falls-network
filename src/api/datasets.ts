// ============================================================
// Datasets API — /api/datasets/*
// ============================================================

import { apiFetch } from './client';

export interface GenerateRequest {
  count?: number;
  seed_start?: number;
  split?: 'train' | 'val' | 'test';
  difficulty?: 'easy' | 'medium' | 'hard' | 'stress';
  bus_count?: number;
  episode_length?: number;
  spawn_rate?: number;
  passenger_capacity?: number;
  reboard_enabled?: boolean;
  charging_enabled?: boolean;
  low_battery_threshold?: number;
  charge_duration?: number;
  battery_drain_rate?: number;
}

export interface GenerateResponse {
  generated: number;
  scenario_ids: string[];
  output_dir: string;
}

export interface ScenarioListItem {
  scenario_id: string;
  split: string;
  difficulty: string;
  seed: number;
  bus_count: number;
  episode_length: number;
  event_count: number;
}

export interface ScenarioListResponse {
  scenarios: ScenarioListItem[];
  total: number;
}

export function generateScenarios(req: GenerateRequest = {}): Promise<GenerateResponse> {
  return apiFetch('/datasets/generate', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export function listScenarios(split?: string): Promise<ScenarioListResponse> {
  const qs = split ? `?split=${split}` : '';
  return apiFetch(`/datasets/list${qs}`);
}
