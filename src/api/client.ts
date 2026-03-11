// ============================================================
// API client — base URL config and fetch helper
// ============================================================

export const API_BASE = 'http://localhost:8000/api';
export const WS_BASE  = 'ws://localhost:8000/api';

export async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}
