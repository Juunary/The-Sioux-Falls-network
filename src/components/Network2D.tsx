// ============================================================
// Network2D — Sioux Falls style layout
// ============================================================

import { useEffect, useMemo, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import { useSimStore } from '../sim/store';
import { NODES, EDGES } from '../data/network';
import type { Bus } from '../types/network';

const PAX_R = 5;          // passenger dot radius
const PAX_MAX_VISIBLE = 6; // max dots shown per node (rest shown as "+N")

const SVG_W = 850;
const SVG_H = 1111;
const NODE_R = 24;
const EDGE_OFFSET = 6;
const EDGE_LABEL_OFFSET = 14;

const ARROW_ID = 'arrowhead';
const ARROW_ROUTE_ID = 'arrowhead-route';
const ARROW_SELECTED_ID = 'arrowhead-selected';

export const SIOUX_FALLS_LAYOUT: Record<number, { x: number; y: number }> = {
  0: { x: 41, y: 36 },
  1: { x: 617, y: 36 },

  2: { x: 41, y: 179 },
  3: { x: 209, y: 177 },
  4: { x: 407, y: 178 },
  5: { x: 617, y: 177 },

  6: { x: 819, y: 301 },
  7: { x: 617, y: 301 },
  8: { x: 407, y: 299 },

  9: { x: 407, y: 428 },
  10: { x: 211, y: 428 },
  11: { x: 41, y: 428 },
  15: { x: 617, y: 427 },
  17: { x: 819, y: 428 },

  16: { x: 617, y: 553 },

  13: { x: 211, y: 701 },
  14: { x: 407, y: 701 },
  18: { x: 617, y: 701 },

  22: { x: 209, y: 829 },
  21: { x: 407, y: 829 },

  12: { x: 41, y: 991 },
  23: { x: 211, y: 991 },
  20: { x: 407, y: 991 },
  19: { x: 617, y: 991 },
};

export default function Network2D() {
  const svgRef = useRef<SVGSVGElement>(null);
  const gRef = useRef<SVGGElement>(null);

  const buses = useSimStore((s) => s.buses);
  const passengers = useSimStore((s) => s.passengers);
  const settings = useSimStore((s) => s.settings);
  const selection = useSimStore((s) => s.selection);
  const setSelection = useSimStore((s) => s.setSelection);

  const displayNodes = useMemo(
    () =>
      NODES.map((node) => ({
        ...node,
        ...(SIOUX_FALLS_LAYOUT[node.id] ?? {}),
      })),
    [],
  );

  const nodeById = useMemo(
    () => new Map(displayNodes.map((node) => [node.id, node])),
    [displayNodes],
  );

  const edgeById = useMemo(
    () => new Map(EDGES.map((edge) => [edge.id, edge])),
    [],
  );

  const showLabels = settings.showLabels !== false;
  const showEdgeIds = settings.showEdgeIds !== false;

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);

    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.4, 5])
      .on('zoom', (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
        if (gRef.current) {
          d3.select(gRef.current).attr('transform', event.transform.toString());
        }
      });

    svg.call(zoom);
    svg.call(zoom.transform, d3.zoomIdentity);

    return () => {
      svg.on('.zoom', null);
    };
  }, []);

  const handleSvgClick = useCallback(() => {
    setSelection({
      selectedBusId: null,
      selectedNodeId: null,
      selectedEdgeId: null,
    });
  }, [setSelection]);

  const interpolateDisplayEdge = useCallback(
    (sourceId: number, targetId: number, progress: number, laneOffset = 0) => {
      const src = nodeById.get(sourceId);
      const tgt = nodeById.get(targetId);

      if (!src || !tgt) return { x: 0, y: 0 };

      const t = Math.max(0, Math.min(1, progress));
      const x = src.x + (tgt.x - src.x) * t;
      const y = src.y + (tgt.y - src.y) * t;

      const dx = tgt.x - src.x;
      const dy = tgt.y - src.y;
      const len = Math.hypot(dx, dy) || 1;

      const ox = (-dy / len) * laneOffset;
      const oy = (dx / len) * laneOffset;

      return { x: x + ox, y: y + oy };
    },
    [nodeById],
  );

  function busPosition(bus: Bus): { x: number; y: number } {
    if (bus.currentEdge != null) {
      const edge = edgeById.get(bus.currentEdge);
      if (edge) {
        return interpolateDisplayEdge(
          edge.source,
          edge.target,
          bus.progress,
          bus.laneOffset,
        );
      }
    }

    const node = nodeById.get(bus.currentNode);
    return node ? { x: node.x, y: node.y } : { x: 0, y: 0 };
  }

  const selectedBus = buses.find((b) => b.id === selection.selectedBusId) ?? null;

  const routeEdgeSet = useMemo(() => {
    const result = new Set<number>();
    if (!selectedBus) return result;

    if (selectedBus.currentEdge != null) {
      result.add(selectedBus.currentEdge);
    }

    let cur =
      selectedBus.currentEdge != null
        ? edgeById.get(selectedBus.currentEdge)?.target ?? selectedBus.currentNode
        : selectedBus.currentNode;

    for (const next of selectedBus.route) {
      const edge = EDGES.find((e) => e.source === cur && e.target === next);
      if (edge) result.add(edge.id);
      cur = next;
    }

    return result;
  }, [selectedBus, edgeById]);

  const routeNodeSet = useMemo(() => {
    const result = new Set<number>();
    if (!selectedBus) return result;

    result.add(selectedBus.currentNode);

    if (selectedBus.currentEdge != null) {
      const edge = edgeById.get(selectedBus.currentEdge);
      if (edge) result.add(edge.target);
    }

    for (const nodeId of selectedBus.route) {
      result.add(nodeId);
    }

    return result;
  }, [selectedBus, edgeById]);

  return (
    <svg
      ref={svgRef}
      width="100%"
      height="100%"
      viewBox={`0 0 ${SVG_W} ${SVG_H}`}
      preserveAspectRatio="xMidYMid meet"
      style={{ background: '#e8e8e8', cursor: 'grab' }}
      onClick={handleSvgClick}
    >
      <defs>
        <marker
          id={ARROW_ID}
          viewBox="0 0 10 10"
          markerWidth="7"
          markerHeight="7"
          refX="8"
          refY="5"
          orient="auto"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#111" />
        </marker>

        <marker
          id={ARROW_ROUTE_ID}
          viewBox="0 0 10 10"
          markerWidth="7"
          markerHeight="7"
          refX="8"
          refY="5"
          orient="auto"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#d94a4a" />
        </marker>

        <marker
          id={ARROW_SELECTED_ID}
          viewBox="0 0 10 10"
          markerWidth="7"
          markerHeight="7"
          refX="8"
          refY="5"
          orient="auto"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb" />
        </marker>
      </defs>

      <g ref={gRef}>
        {/* ---- Edges ---- */}
        {EDGES.map((edge) => {
          const src = nodeById.get(edge.source);
          const tgt = nodeById.get(edge.target);
          if (!src || !tgt) return null;

          const isSelected = edge.id === selection.selectedEdgeId;
          const isRouteEdge = routeEdgeSet.has(edge.id);

          const dx = tgt.x - src.x;
          const dy = tgt.y - src.y;
          const len = Math.hypot(dx, dy) || 1;

          const ox = (-dy / len) * EDGE_OFFSET;
          const oy = (dx / len) * EDGE_OFFSET;

          const x1 = src.x + ox;
          const y1 = src.y + oy;
          const x2 = tgt.x + ox;
          const y2 = tgt.y + oy;

          const shorten = NODE_R + 6;
          const ex = (x2 - x1) / len;
          const ey = (y2 - y1) / len;

          const sx1 = x1 + ex * shorten;
          const sy1 = y1 + ey * shorten;
          const sx2 = x2 - ex * shorten;
          const sy2 = y2 - ey * shorten;

          const midX = (sx1 + sx2) / 2 - ey * EDGE_LABEL_OFFSET;
          const midY = (sy1 + sy2) / 2 + ex * EDGE_LABEL_OFFSET;

          const stroke = isSelected ? '#2563eb' : isRouteEdge ? '#d94a4a' : '#111';
          const markerId = isSelected
            ? ARROW_SELECTED_ID
            : isRouteEdge
            ? ARROW_ROUTE_ID
            : ARROW_ID;

          return (
            <g key={edge.id}>
              <line
                x1={sx1}
                y1={sy1}
                x2={sx2}
                y2={sy2}
                stroke={stroke}
                strokeWidth={isSelected ? 3.2 : isRouteEdge ? 2.4 : 1.8}
                markerEnd={`url(#${markerId})`}
                strokeLinecap="round"
                style={{ cursor: 'pointer' }}
                onClick={(e) => {
                  e.stopPropagation();
                  setSelection({
                    selectedEdgeId: edge.id,
                    selectedBusId: null,
                    selectedNodeId: null,
                  });
                }}
              />

              <line
                x1={sx1}
                y1={sy1}
                x2={sx2}
                y2={sy2}
                stroke="transparent"
                strokeWidth={14}
                style={{ cursor: 'pointer' }}
                onClick={(e) => {
                  e.stopPropagation();
                  setSelection({
                    selectedEdgeId: edge.id,
                    selectedBusId: null,
                    selectedNodeId: null,
                  });
                }}
              />

              {showEdgeIds && (
                <text
                  x={midX}
                  y={midY}
                  fontSize={16}
                  fill="#111"
                  textAnchor="middle"
                  dominantBaseline="middle"
                  style={{ pointerEvents: 'none', userSelect: 'none' }}
                >
                  {edge.id}
                </text>
              )}
            </g>
          );
        })}

        {/* ---- Nodes ---- */}
        {displayNodes.map((node) => {
          const isSelected = node.id === selection.selectedNodeId;
          const isOnRoute = routeNodeSet.has(node.id);
          const isCS = node.isChargingStation;

          const stroke = isSelected ? '#2563eb' : isOnRoute ? '#d94a4a' : '#111';

          return (
            <g
              key={node.id}
              transform={`translate(${node.x},${node.y})`}
              style={{ cursor: 'pointer' }}
              onClick={(e) => {
                e.stopPropagation();
                setSelection({
                  selectedNodeId: node.id,
                  selectedBusId: null,
                  selectedEdgeId: null,
                });
              }}
            >
              <circle
                r={NODE_R}
                fill={isCS ? '#f4ec00' : '#fff'}
                stroke={stroke}
                strokeWidth={isSelected ? 3 : isOnRoute ? 2.4 : 1.8}
              />

              {showLabels && (
                <text
                  fontSize={20}
                  fill="#111"
                  textAnchor="middle"
                  dominantBaseline="middle"
                  style={{
                    pointerEvents: 'none',
                    userSelect: 'none',
                    fontWeight: 600,
                  }}
                >
                  {node.id}
                </text>
              )}
            </g>
          );
        })}

        {/* ---- Buses ---- */}
        {buses.map((bus) => {
          const pos = busPosition(bus);
          const isSelected = bus.id === selection.selectedBusId;

          return (
            <g
              key={bus.id}
              transform={`translate(${pos.x},${pos.y})`}
              style={{ cursor: 'pointer' }}
              onClick={(e) => {
                e.stopPropagation();
                setSelection({
                  selectedBusId: bus.id,
                  selectedNodeId: null,
                  selectedEdgeId: null,
                });
              }}
            >
              {isSelected && (
                <circle r={11} fill="none" stroke="#111" strokeWidth={2} opacity={0.9} />
              )}

              <circle
                r={8}
                fill={bus.state === 'charging' ? '#f1c40f' : bus.color}
                stroke={isSelected ? '#111' : 'rgba(0,0,0,0.45)'}
                strokeWidth={isSelected ? 2 : 1}
                opacity={0.95}
              />

              <BatteryArc battery={bus.battery} />

              {isSelected && (
                <text
                  y={-16}
                  fontSize={11}
                  fill="#111"
                  textAnchor="middle"
                  style={{ pointerEvents: 'none', userSelect: 'none' }}
                >
                  B{bus.id}
                </text>
              )}

              {/* ---- Riding passengers — row below bus circle ---- */}
              {(() => {
                const riders = bus.passengerIds
                  .map((pid) => passengers.find((p) => p.id === pid))
                  .filter(Boolean) as typeof passengers;
                if (riders.length === 0) return null;

                const BUS_R = 8;
                const SELECTED_RING_R = 11; // outer ring radius when bus is selected
                const spacing = PAX_R * 2 + 2;
                const totalW = riders.length * spacing - 2;
                const startX = -totalW / 2 + PAX_R;
                const dotY = SELECTED_RING_R + PAX_R + 5; // 11 + 5 + 5 = 21, clears the selection ring

                return (
                  <g>
                    {riders.map((p, i) => {
                      const isTracked = p.id === selection.trackedPassengerId;
                      return (
                        <g
                          key={p.id}
                          style={{ cursor: 'pointer' }}
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelection({
                              trackedPassengerId: isTracked ? null : p.id,
                              selectedBusId: null,
                              selectedNodeId: null,
                              selectedEdgeId: null,
                            });
                          }}
                        >
                          {isTracked && (
                            <circle
                              cx={startX + i * spacing}
                              cy={dotY}
                              r={PAX_R + 2.5}
                              fill="none"
                              stroke={p.color}
                              strokeWidth={1.5}
                            />
                          )}
                          <circle
                            cx={startX + i * spacing}
                            cy={dotY}
                            r={PAX_R}
                            fill={p.color}
                            stroke="rgba(0,0,0,0.4)"
                            strokeWidth={0.8}
                          />
                          <title>Pax #{p.id} → Node {p.destinationNode}</title>
                        </g>
                      );
                    })}
                  </g>
                );
              })()}
            </g>
          );
        })}

        {/* ---- Waiting Passengers at Nodes ---- */}
        {displayNodes.map((node) => {
          const waitingHere = passengers.filter(
            (p) => p.state === 'waiting' && p.currentNode === node.id,
          );
          if (waitingHere.length === 0) return null;

          const visible = waitingHere.slice(0, PAX_MAX_VISIBLE);
          const overflow = waitingHere.length - visible.length;
          // Arrange dots in a row above the node
          const spacing = PAX_R * 2 + 2;
          const totalW = visible.length * spacing + (overflow > 0 ? 18 : 0);
          const startX = -totalW / 2 + PAX_R;
          const dotY = -(NODE_R + PAX_R + 4);

          return (
            <g key={`pax-${node.id}`} transform={`translate(${node.x},${node.y})`}>
              {visible.map((p, i) => {
                const isTracked = p.id === selection.trackedPassengerId;
                return (
                  <g
                    key={p.id}
                    style={{ cursor: 'pointer' }}
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelection({
                        trackedPassengerId: isTracked ? null : p.id,
                        selectedBusId: null,
                        selectedNodeId: null,
                        selectedEdgeId: null,
                      });
                    }}
                  >
                    {isTracked && (
                      <circle
                        cx={startX + i * spacing}
                        cy={dotY}
                        r={PAX_R + 3}
                        fill="none"
                        stroke={p.color}
                        strokeWidth={1.5}
                        opacity={0.9}
                      />
                    )}
                    <circle
                      cx={startX + i * spacing}
                      cy={dotY}
                      r={PAX_R}
                      fill={p.color}
                      stroke="#111"
                      strokeWidth={0.8}
                      opacity={0.92}
                    />
                    <title>Pax #{p.id} → Node {p.destinationNode}</title>
                  </g>
                );
              })}
              {overflow > 0 && (
                <text
                  x={startX + visible.length * spacing + 2}
                  y={dotY + 4}
                  fontSize={8}
                  fill="#555"
                  style={{ pointerEvents: 'none', userSelect: 'none' }}
                >
                  +{overflow}
                </text>
              )}
            </g>
          );
        })}

        <Legend />
      </g>
    </svg>
  );
}

function Legend() {
  return (
    <g transform="translate(205 1040)" pointerEvents="none">
      <rect width={376} height={70} fill="#fff" stroke="#111" strokeWidth={2} />
      <circle cx={50} cy={35} r={23} fill="#f4ec00" stroke="#111" strokeWidth={2} />
      <text
        x={92}
        y={44}
        fontSize={26}
        fill="#111"
        style={{ fontFamily: 'Times New Roman, serif' }}
      >
        : Charging station
      </text>
    </g>
  );
}

function BatteryArc({ battery }: { battery: number }) {
  const pct = Math.max(0, Math.min(1, battery / 100));
  const color = pct > 0.4 ? '#2ecc71' : pct > 0.2 ? '#f39c12' : '#e74c3c';
  const r = 8;
  const angle = pct * 2 * Math.PI - Math.PI / 2;
  const x = r * Math.cos(angle);
  const y = r * Math.sin(angle);
  const largeArc = pct > 0.5 ? 1 : 0;

  if (pct <= 0) return null;
  if (pct >= 1) {
    return <circle r={r} fill="none" stroke={color} strokeWidth={2} opacity={0.8} />;
  }

  return (
    <path
      d={`M 0 ${-r} A ${r} ${r} 0 ${largeArc} 1 ${x} ${y}`}
      fill="none"
      stroke={color}
      strokeWidth={2}
      opacity={0.8}
      style={{ pointerEvents: 'none' }}
    />
  );
}
