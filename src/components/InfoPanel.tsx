// ============================================================
// InfoPanel — status overview + selected item details
// ============================================================

import React from 'react';
import { useSimStore } from '../sim/store';
import { NODES, EDGES } from '../data/network';

export default function InfoPanel() {
  const buses = useSimStore((s) => s.buses);
  const selection = useSimStore((s) => s.selection);

  const moving = buses.filter((b) => b.state === 'moving').length;
  const charging = buses.filter((b) => b.state === 'charging').length;
  const idle = buses.filter((b) => b.state === 'idle' || b.state === 'rerouting').length;

  const selectedBus = buses.find((b) => b.id === selection.selectedBusId);
  const selectedNode = selection.selectedNodeId != null ? NODES[selection.selectedNodeId] : null;
  const selectedEdge = selection.selectedEdgeId != null ? EDGES[selection.selectedEdgeId] : null;

  return (
    <div className="info-panel">
      {/* Summary */}
      <div className="info-section">
        <div className="info-title">Network Status</div>
        <div className="info-row">
          <span className="info-label">Total buses</span>
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

      {/* Selected bus */}
      {selectedBus && (
        <div className="info-section">
          <div className="info-title" style={{ color: selectedBus.color }}>
            Bus {selectedBus.id}
          </div>
          <div className="info-row">
            <span className="info-label">State</span>
            <span className="info-value" style={{ textTransform: 'capitalize' }}>
              {selectedBus.state}
            </span>
          </div>
          <div className="info-row">
            <span className="info-label">At node</span>
            <span className="info-value">{selectedBus.currentNode}</span>
          </div>
          {selectedBus.currentEdge != null && (
            <div className="info-row">
              <span className="info-label">On edge</span>
              <span className="info-value">
                {selectedBus.currentEdge} ({EDGES[selectedBus.currentEdge].source}→
                {EDGES[selectedBus.currentEdge].target})
              </span>
            </div>
          )}
          <div className="info-row">
            <span className="info-label">Progress</span>
            <span className="info-value">{(selectedBus.progress * 100).toFixed(0)}%</span>
          </div>
          <div className="info-row">
            <span className="info-label">Battery</span>
            <BatteryBar value={selectedBus.battery} />
          </div>
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

      {/* Selected node */}
      {selectedNode && (
        <div className="info-section">
          <div className="info-title">Node {selectedNode.id}</div>
          <div className="info-row">
            <span className="info-label">Position</span>
            <span className="info-value">
              ({selectedNode.x}, {selectedNode.y})
            </span>
          </div>
          <div className="info-row">
            <span className="info-label">Charging station</span>
            <span className="info-value" style={{ color: selectedNode.isChargingStation ? '#f1c40f' : '#aaa' }}>
              {selectedNode.isChargingStation ? 'Yes ⚡' : 'No'}
            </span>
          </div>
          <div className="info-row">
            <span className="info-label">Buses here</span>
            <span className="info-value">
              {buses.filter((b) => b.currentNode === selectedNode.id && b.currentEdge == null).length}
            </span>
          </div>
        </div>
      )}

      {/* Selected edge */}
      {selectedEdge && (
        <div className="info-section">
          <div className="info-title">Edge {selectedEdge.id}</div>
          <div className="info-row">
            <span className="info-label">Direction</span>
            <span className="info-value">
              {selectedEdge.source} → {selectedEdge.target}
            </span>
          </div>
          <div className="info-row">
            <span className="info-label">Weight (dist)</span>
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

      {/* Legend */}
      <div className="info-section">
        <div className="info-title">Legend</div>
        <div className="legend-row">
          <span className="legend-dot" style={{ background: '#f39c12' }} />
          <span>Charging station node</span>
        </div>
        <div className="legend-row">
          <span className="legend-dot" style={{ background: '#f1c40f' }} />
          <span>Bus charging</span>
        </div>
        <div className="legend-row">
          <span className="legend-dot" style={{ background: '#2ecc71' }} />
          <span>Battery &gt; 40%</span>
        </div>
        <div className="legend-row">
          <span className="legend-dot" style={{ background: '#f39c12' }} />
          <span>Battery 20–40%</span>
        </div>
        <div className="legend-row">
          <span className="legend-dot" style={{ background: '#e74c3c' }} />
          <span>Battery &lt; 20%</span>
        </div>
      </div>
    </div>
  );
}

function BatteryBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(100, value));
  const color = pct > 40 ? '#2ecc71' : pct > 20 ? '#f39c12' : '#e74c3c';
  return (
    <span className="battery-bar">
      <span
        className="battery-fill"
        style={{ width: `${pct}%`, background: color }}
      />
      <span className="battery-text">{pct.toFixed(0)}%</span>
    </span>
  );
}
