// ============================================================
// TrainingChart — live episode reward chart for a training job
// Uses recharts LineChart.
// ============================================================

import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import type { MetricsRecord } from '../api/training';

interface Props {
  data: MetricsRecord[];
}

export default function TrainingChart({ data }: Props) {
  if (data.length === 0) {
    return (
      <div className="chart-placeholder">
        Waiting for training metrics…
      </div>
    );
  }

  // Filter to records that have a reward value
  const chartData = data
    .filter((r) => r.reward !== null)
    .map((r) => ({ ...r, reward: r.reward as number }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2a2a45" />
        <XAxis
          dataKey="timestep"
          tick={{ fill: '#7a7a9a', fontSize: 10 }}
          tickFormatter={(v: number) => `${(v / 1000).toFixed(0)}k`}
          label={{ value: 'Timesteps', position: 'insideBottom', offset: -2, fill: '#7a7a9a', fontSize: 10 }}
        />
        <YAxis
          tick={{ fill: '#7a7a9a', fontSize: 10 }}
          width={40}
        />
        <Tooltip
          contentStyle={{ background: '#161628', border: '1px solid #2a2a45', fontSize: 11 }}
          labelStyle={{ color: '#7a7a9a' }}
          itemStyle={{ color: '#3498db' }}
          labelFormatter={(v) => `t=${v}`}
          formatter={(v: number) => [v.toFixed(2), 'ep_rew_mean']}
        />
        <ReferenceLine y={0} stroke="#2a2a45" />
        <Line
          type="monotone"
          dataKey="reward"
          stroke="#3498db"
          dot={false}
          strokeWidth={1.5}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
