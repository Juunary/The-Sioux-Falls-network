// ============================================================
// ComparisonChart — side-by-side bar chart for experiment metrics
// Shows key metrics for two selected experiments.
// ============================================================

import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { AggregateStats } from '../api/experiments';

interface Props {
  labelA: string;
  labelB: string;
  statsA: AggregateStats;
  statsB: AggregateStats;
}

export default function ComparisonChart({ labelA, labelB, statsA, statsB }: Props) {
  const metrics = [
    {
      name: 'Avg Reward',
      a: statsA.avg_reward ?? 0,
      b: statsB.avg_reward ?? 0,
    },
    {
      name: 'Service Rate',
      a: (statsA.avg_service_rate ?? 0) * 100,
      b: (statsB.avg_service_rate ?? 0) * 100,
    },
    {
      name: 'Scenarios',
      a: statsA.count ?? 0,
      b: statsB.count ?? 0,
    },
  ];

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={metrics} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2a2a45" />
        <XAxis dataKey="name" tick={{ fill: '#7a7a9a', fontSize: 11 }} />
        <YAxis tick={{ fill: '#7a7a9a', fontSize: 10 }} width={44} />
        <Tooltip
          contentStyle={{ background: '#161628', border: '1px solid #2a2a45', fontSize: 11 }}
          labelStyle={{ color: '#c8cce8' }}
        />
        <Legend
          wrapperStyle={{ fontSize: 11, color: '#c8cce8' }}
        />
        <Bar dataKey="a" name={labelA} fill="#3498db" maxBarSize={40} />
        <Bar dataKey="b" name={labelB} fill="#f39c12" maxBarSize={40} />
      </BarChart>
    </ResponsiveContainer>
  );
}
