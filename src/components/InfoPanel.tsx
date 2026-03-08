// ============================================================
// InfoPanel — status overview, passenger waiting chart,
//             selected item details, passenger tracking
// ============================================================

import { useSimStore } from '../sim/store';
import { NODES, EDGES } from '../data/network';
import type { Passenger } from '../types/network';

export default function InfoPanel() {
  const buses      = useSimStore((s) => s.buses);
  const passengers = useSimStore((s) => s.passengers);
  const history    = useSimStore((s) => s.waitingHistory);
  const selection  = useSimStore((s) => s.selection);
  const setSelection = useSimStore((s) => s.setSelection);

  const moving   = buses.filter((b) => b.state === 'moving').length;
  const charging = buses.filter((b) => b.state === 'charging').length;
  const idle     = buses.filter((b) => b.state === 'idle' || b.state === 'rerouting').length;

  const waiting  = passengers.filter((p) => p.state === 'waiting').length;
  const riding   = passengers.filter((p) => p.state === 'riding').length;

  const selectedBus  = buses.find((b) => b.id === selection.selectedBusId);
  const selectedNode = selection.selectedNodeId != null ? NODES[selection.selectedNodeId] : null;
  const selectedEdge = selection.selectedEdgeId != null ? EDGES[selection.selectedEdgeId] : null;
  const trackedPax   = passengers.find((p) => p.id === selection.trackedPassengerId) ?? null;

  return (
    <div className="info-panel">

      {/* ---- Waiting passenger line chart ---- */}
      <div className="info-section">
        <div className="info-title">Waiting Passengers</div>
        <WaitingChart history={history} />
        <div className="pax-stat-row">
          <span className="pax-stat">
            <span className="pax-dot" style={{ background: '#f39c12' }} />
            Waiting: {waiting}
          </span>
          <span className="pax-stat">
            <span className="pax-dot" style={{ background: '#2ecc71' }} />
            Riding: {riding}
          </span>
        </div>
      </div>

      {/* ---- Network status ---- */}
      <div className="info-section">
        <div className="info-title">Network Status</div>
        <div className="info-row">
          <span className="info-label">Buses</span>
          <span className="info-value">{buses.length}</span>
        </div>
        <div className="info-row">
          <span className="info-label" style={{ color: '#3498db' }}>Moving</span>
          <span className="info-value">{moving}</span>
        </div>
        <div className="info-row">
          <span className="info-label" style={{ color: '#f1c40f' }}>Charging</span>
          <span className="info-value">{charging}</span>
        </div>
        <div className="info-row">
          <span className="info-label" style={{ color: '#888' }}>Idle</span>
          <span className="info-value">{idle}</span>
        </div>
      </div>

      {/* ---- Tracked passenger ---- */}
      {trackedPax && (
        <div className="info-section" style={{ borderColor: trackedPax.color }}>
          <div className="info-title" style={{ color: trackedPax.color, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>Passenger #{trackedPax.id}</span>
            <button
              className="btn-untrack"
              onClick={() => setSelection({ trackedPassengerId: null })}
            >
              ✕
            </button>
          </div>
          <PassengerDetail pax={trackedPax} buses={buses} />
        </div>
      )}

      {/* ---- Selected bus ---- */}
      {selectedBus && (
        <div className="info-section">
          <div className="info-title" style={{ color: selectedBus.color }}>
            Bus {selectedBus.id}
          </div>
          <div className="info-row">
            <span className="info-label">State</span>
            <span className="info-value" style={{ textTransform: 'capitalize' }}>{selectedBus.state}</span>
          </div>
          <div className="info-row">
            <span className="info-label">At node</span>
            <span className="info-value">{selectedBus.currentNode}</span>
          </div>
          {selectedBus.currentEdge != null && (
            <div className="info-row">
              <span className="info-label">On edge</span>
              <span className="info-value">
                {selectedBus.currentEdge} ({EDGES[selectedBus.currentEdge].source}→{EDGES[selectedBus.currentEdge].target})
              </span>
            </div>
          )}
          <div className="info-row">
            <span className="info-label">Battery</span>
            <BatteryBar value={selectedBus.battery} />
          </div>
          <div className="info-row">
            <span className="info-label">Passengers</span>
            <span className="info-value">{selectedBus.passengerIds.length} / {selectedBus.capacity}</span>
          </div>
          {selectedBus.passengerIds.length > 0 && (
            <div className="info-row">
              <span className="info-label">On board</span>
              <span className="pax-id-list">
                {selectedBus.passengerIds.map((pid) => {
                  const p = passengers.find((x) => x.id === pid);
                  return (
                    <span
                      key={pid}
                      className="pax-id-badge"
                      style={{ background: p?.color ?? '#888' }}
                      title={`Pax ${pid} → Node ${p?.destinationNode}`}
                      onClick={() => setSelection({ trackedPassengerId: pid })}
                    >
                      {pid}
                    </span>
                  );
                })}
              </span>
            </div>
          )}
          {selectedBus.state === 'charging' && (
            <div className="info-row">
              <span className="info-label">Charge left</span>
              <span className="info-value">{selectedBus.chargeTimeLeft.toFixed(1)}s</span>
            </div>
          )}
          {selectedBus.destination != null && (
            <div className="info-row">
              <span className="info-label">Destination</span>
              <span className="info-value">{selectedBus.destination}</span>
            </div>
          )}
          {selectedBus.route.length > 0 && (
            <div className="info-row">
              <span className="info-label">Route</span>
              <span className="info-value route-path">
                {selectedBus.currentNode} → {selectedBus.route.join(' → ')}
              </span>
            </div>
          )}
        </div>
      )}

      {/* ---- Selected node ---- */}
      {selectedNode && (
        <div className="info-section">
          <div className="info-title">Node {selectedNode.id}</div>
          <div className="info-row">
            <span className="info-label">Charging station</span>
            <span className="info-value" style={{ color: selectedNode.isChargingStation ? '#f1c40f' : '#aaa' }}>
              {selectedNode.isChargingStation ? 'Yes' : 'No'}
            </span>
          </div>
          <div className="info-row">
            <span className="info-label">Buses here</span>
            <span className="info-value">
              {buses.filter((b) => b.currentNode === selectedNode.id && b.currentEdge == null).length}
            </span>
          </div>
          <div className="info-row">
            <span className="info-label">Waiting pax</span>
            <span className="info-value">
              {passengers.filter((p) => p.state === 'waiting' && p.currentNode === selectedNode.id).length}
            </span>
          </div>
        </div>
      )}

      {/* ---- Selected edge ---- */}
      {selectedEdge && (
        <div className="info-section">
          <div className="info-title">Edge {selectedEdge.id}</div>
          <div className="info-row">
            <span className="info-label">Direction</span>
            <span className="info-value">{selectedEdge.source} → {selectedEdge.target}</span>
          </div>
          <div className="info-row">
            <span className="info-label">Weight</span>
            <span className="info-value">{selectedEdge.weight.toFixed(1)} px</span>
          </div>
          <div className="info-row">
            <span className="info-label">Buses on edge</span>
            <span className="info-value">
              {buses.filter((b) => b.currentEdge === selectedEdge.id).length}
            </span>
          </div>
        </div>
      )}

    </div>
  );
}

// ---- Passenger detail ----
function PassengerDetail({
  pax,
  buses,
}: {
  pax: Passenger;
  buses: ReturnType<typeof useSimStore.getState>['buses'];
}) {
  const STATE_LABEL: Record<string, string> = {
    waiting: 'Waiting',
    riding: 'Riding',
    arrived: 'Arrived',
    gone: 'Left (gave up)',
  };
  const onBus = pax.busId != null ? buses.find((b) => b.id === pax.busId) : null;

  return (
    <>
      <div className="info-row">
        <span className="info-label">Status</span>
        <span className="info-value" style={{ color: stateColor(pax.state) }}>
          {STATE_LABEL[pax.state]}
        </span>
      </div>
      <div className="info-row">
        <span className="info-label">Origin</span>
        <span className="info-value">Node {pax.originNode}</span>
      </div>
      <div className="info-row">
        <span className="info-label">Destination</span>
        <span className="info-value">Node {pax.destinationNode}</span>
      </div>
      {pax.state === 'waiting' && (
        <div className="info-row">
          <span className="info-label">At node</span>
          <span className="info-value">{pax.currentNode}</span>
        </div>
      )}
      {onBus && (
        <div className="info-row">
          <span className="info-label">On bus</span>
          <span className="info-value" style={{ color: onBus.color }}>Bus {onBus.id}</span>
        </div>
      )}
    </>
  );
}

function stateColor(state: string): string {
  switch (state) {
    case 'waiting': return '#f39c12';
    case 'riding':  return '#2ecc71';
    case 'arrived': return '#3498db';
    default:        return '#888';
  }
}

// ---- Waiting passenger line chart ----
function WaitingChart({ history }: { history: { t: number; count: number }[] }) {
  const W = 192, H = 60, PAD_X = 22, PAD_Y = 6;

  if (history.length < 2) {
    return <div className="chart-placeholder">Collecting data…</div>;
  }

  const maxCount = Math.max(...history.map((h) => h.count), 1);
  const n = history.length;
  const chartW = W - PAD_X - 4;
  const chartH = H - PAD_Y * 2;

  const toXY = (h: { count: number }, i: number) => ({
    x: PAD_X + (i / (n - 1)) * chartW,
    y: PAD_Y + (1 - h.count / maxCount) * chartH,
  });

  const pts = history.map((h, i) => {
    const { x, y } = toXY(h, i);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');

  const bottomY = PAD_Y + chartH;
  const areaPath =
    `M${PAD_X},${bottomY} ` +
    history.map((h, i) => { const { x, y } = toXY(h, i); return `L${x.toFixed(1)},${y.toFixed(1)}`; }).join(' ') +
    ` L${(PAD_X + chartW).toFixed(1)},${bottomY} Z`;

  const latest = history[history.length - 1].count;

  return (
    <svg width={W} height={H} className="waiting-chart">
      <defs>
        <linearGradient id="chartGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#3498db" stopOpacity="0.4" />
          <stop offset="100%" stopColor="#3498db" stopOpacity="0.02" />
        </linearGradient>
      </defs>

      {/* Axes */}
      <line x1={PAD_X} y1={PAD_Y} x2={PAD_X} y2={bottomY} stroke="#2a2a45" strokeWidth={1} />
      <line x1={PAD_X} y1={bottomY} x2={W - 4} y2={bottomY} stroke="#2a2a45" strokeWidth={1} />

      {/* Y labels */}
      <text x={PAD_X - 3} y={PAD_Y + 4} fontSize={7} fill="#7a7a9a" textAnchor="end">{maxCount}</text>
      <text x={PAD_X - 3} y={bottomY} fontSize={7} fill="#7a7a9a" textAnchor="end">0</text>

      {/* Area fill */}
      <path d={areaPath} fill="url(#chartGrad)" />

      {/* Line */}
      <polyline points={pts} fill="none" stroke="#3498db" strokeWidth={1.5} strokeLinejoin="round" />

      {/* Current value */}
      <text x={W - 2} y={PAD_Y + 7} fontSize={8} fill="#3498db" textAnchor="end" fontWeight="bold">
        {latest}
      </text>
    </svg>
  );
}

// ---- Battery bar ----
function BatteryBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(100, value));
  const color = pct > 40 ? '#2ecc71' : pct > 20 ? '#f39c12' : '#e74c3c';
  return (
    <span className="battery-bar">
      <span className="battery-fill" style={{ width: `${pct}%`, background: color }} />
      <span className="battery-text">{pct.toFixed(0)}%</span>
    </span>
  );
}
