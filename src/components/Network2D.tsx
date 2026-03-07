// ============================================================
// Network2D — D3-powered SVG visualization
// ============================================================

import React, { useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import { useSimStore } from '../sim/store';
import { NODES, EDGES } from '../data/network';
import { interpolateEdgeWithOffset, getEdge } from '../utils/graph';
import type { Bus } from '../types/network';

const SVG_W = 960;
const SVG_H = 880;
const NODE_R = 14;

// Arrowhead marker id
const ARROW_ID = 'arrowhead';
const ARROW_CS_ID = 'arrowhead-cs';

export default function Network2D() {
  const svgRef = useRef<SVGSVGElement>(null);
  const gRef = useRef<SVGGElement | null>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);

  const buses = useSimStore((s) => s.buses);
  const settings = useSimStore((s) => s.settings);
  const selection = useSimStore((s) => s.selection);
  const setSelection = useSimStore((s) => s.setSelection);

  // ---- Setup D3 zoom once ----
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 5])
      .on('zoom', (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
        if (gRef.current) {
          d3.select(gRef.current).attr('transform', event.transform.toString());
        }
      });

    svg.call(zoom);
    zoomRef.current = zoom;

    // Initial fit
    const initialT = d3.zoomIdentity
      .translate(20, 20)
      .scale(0.95);
    svg.call(zoom.transform, initialT);

    return () => { svg.on('.zoom', null); };
  }, []);

  // ---- Click on SVG background clears selection ----
  const handleSvgClick = useCallback(() => {
    setSelection({ selectedBusId: null, selectedNodeId: null, selectedEdgeId: null });
  }, [setSelection]);

  // ---- Compute bus screen position ----
  function busPosition(bus: Bus): { x: number; y: number } {
    if (bus.currentEdge != null) {
      const edge = EDGES[bus.currentEdge];
      return interpolateEdgeWithOffset(edge, bus.progress, bus.laneOffset);
    }
    const node = NODES[bus.currentNode];
    return { x: node.x, y: node.y };
  }

  const selectedBus = buses.find((b) => b.id === selection.selectedBusId);
  const routeNodeSet = new Set<number>(
    selectedBus
      ? [selectedBus.currentNode, ...selectedBus.route,
         ...(selectedBus.currentEdge != null ? [getEdge(selectedBus.currentEdge).target] : [])]
      : [],
  );

  return (
    <svg
      ref={svgRef}
      width="100%"
      height="100%"
      viewBox={`0 0 ${SVG_W} ${SVG_H}`}
      style={{ background: '#1a1a2e', cursor: 'grab' }}
      onClick={handleSvgClick}
    >
      <defs>
        {/* Standard arrowhead */}
        <marker
          id={ARROW_ID}
          markerWidth="8"
          markerHeight="8"
          refX="6"
          refY="3"
          orient="auto"
        >
          <path d="M0,0 L0,6 L8,3 z" fill="#556" />
        </marker>
        {/* Highlighted arrowhead */}
        <marker
          id={ARROW_CS_ID}
          markerWidth="8"
          markerHeight="8"
          refX="6"
          refY="3"
          orient="auto"
        >
          <path d="M0,0 L0,6 L8,3 z" fill="#f39c12" />
        </marker>
      </defs>

      <g ref={gRef as React.RefObject<SVGGElement>}>

        {/* ---- Edges ---- */}
        {EDGES.map((edge) => {
          const src = NODES[edge.source];
          const tgt = NODES[edge.target];
          const isSelected = edge.id === selection.selectedEdgeId;
          const isRouteEdge =
            selectedBus?.currentEdge === edge.id ||
            (selectedBus &&
              selectedBus.route.length > 0 &&
              (() => {
                // check if edge is on selected bus route
                let cur = selectedBus.currentNode;
                if (selectedBus.currentEdge != null) cur = EDGES[selectedBus.currentEdge].target;
                for (const next of selectedBus.route) {
                  if (EDGES.find((e) => e.source === cur && e.target === next)?.id === edge.id)
                    return true;
                  cur = next;
                }
                return false;
              })());

          // Offset lines slightly so bidirectional pairs don't overlap
          const dx = tgt.x - src.x;
          const dy = tgt.y - src.y;
          const len = Math.sqrt(dx * dx + dy * dy) || 1;
          const ox = (-dy / len) * 5;
          const oy = (dx / len) * 5;

          const x1 = src.x + ox;
          const y1 = src.y + oy;
          const x2 = tgt.x + ox;
          const y2 = tgt.y + oy;

          // Shorten line so it doesn't overlap arrowhead or node circle
          const shorten = NODE_R + 4;
          const ex = (x2 - x1) / len;
          const ey = (y2 - y1) / len;
          const sx1 = x1 + ex * shorten;
          const sy1 = y1 + ey * shorten;
          const sx2 = x2 - ex * shorten;
          const sy2 = y2 - ey * shorten;

          const midX = (sx1 + sx2) / 2 - ey * 10;
          const midY = (sy1 + sy2) / 2 + ex * 10;

          return (
            <g key={edge.id}>
              <line
                x1={sx1} y1={sy1} x2={sx2} y2={sy2}
                stroke={isSelected ? '#f39c12' : isRouteEdge ? '#e74c3c' : '#445'}
                strokeWidth={isSelected ? 3 : isRouteEdge ? 2.5 : 1.5}
                markerEnd={`url(#${isSelected || isRouteEdge ? ARROW_CS_ID : ARROW_ID})`}
                style={{ cursor: 'pointer' }}
                onClick={(e) => {
                  e.stopPropagation();
                  setSelection({ selectedEdgeId: edge.id, selectedBusId: null, selectedNodeId: null });
                }}
              />
              {/* Invisible wider hit area */}
              <line
                x1={sx1} y1={sy1} x2={sx2} y2={sy2}
                stroke="transparent"
                strokeWidth={12}
                style={{ cursor: 'pointer' }}
                onClick={(e) => {
                  e.stopPropagation();
                  setSelection({ selectedEdgeId: edge.id, selectedBusId: null, selectedNodeId: null });
                }}
              />
              {settings.showEdgeIds && (
                <text
                  x={midX}
                  y={midY}
                  fontSize={9}
                  fill="#778"
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
        {NODES.map((node) => {
          const isSelected = node.id === selection.selectedNodeId;
          const isOnRoute = routeNodeSet.has(node.id);
          const isCS = node.isChargingStation;

          return (
            <g
              key={node.id}
              transform={`translate(${node.x},${node.y})`}
              style={{ cursor: 'pointer' }}
              onClick={(e) => {
                e.stopPropagation();
                setSelection({ selectedNodeId: node.id, selectedBusId: null, selectedEdgeId: null });
              }}
            >
              {isCS && (
                <circle
                  r={NODE_R + 5}
                  fill="none"
                  stroke="#f1c40f"
                  strokeWidth={2}
                  opacity={0.6}
                />
              )}
              <circle
                r={NODE_R}
                fill={isCS ? '#f39c12' : isSelected ? '#3498db' : isOnRoute ? '#e74c3c' : '#2c3e50'}
                stroke={isSelected ? '#ecf0f1' : isCS ? '#f1c40f' : '#445'}
                strokeWidth={isSelected ? 2.5 : 1.5}
              />
              {settings.showLabels && (
                <text
                  fontSize={10}
                  fill="#ecf0f1"
                  textAnchor="middle"
                  dominantBaseline="middle"
                  style={{ pointerEvents: 'none', userSelect: 'none', fontWeight: 'bold' }}
                >
                  {node.id}
                </text>
              )}
              {isCS && (
                <text
                  y={NODE_R + 12}
                  fontSize={8}
                  fill="#f1c40f"
                  textAnchor="middle"
                  style={{ pointerEvents: 'none', userSelect: 'none' }}
                >
                  ⚡
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
                setSelection({ selectedBusId: bus.id, selectedNodeId: null, selectedEdgeId: null });
              }}
            >
              {isSelected && (
                <circle r={11} fill="none" stroke="#ecf0f1" strokeWidth={2} opacity={0.8} />
              )}
              <circle
                r={8}
                fill={bus.state === 'charging' ? '#f1c40f' : bus.color}
                stroke={isSelected ? '#fff' : 'rgba(0,0,0,0.5)'}
                strokeWidth={isSelected ? 2 : 1}
                opacity={0.92}
              />
              {/* Battery indicator arc */}
              <BatteryArc battery={bus.battery} />
              {isSelected && (
                <text
                  y={-14}
                  fontSize={9}
                  fill="#ecf0f1"
                  textAnchor="middle"
                  style={{ pointerEvents: 'none' }}
                >
                  B{bus.id}
                </text>
              )}
            </g>
          );
        })}
      </g>
    </svg>
  );
}

// ---- Battery arc indicator ----
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
    return <circle r={r} fill="none" stroke={color} strokeWidth={2} opacity={0.7} />;
  }

  return (
    <path
      d={`M 0 ${-r} A ${r} ${r} 0 ${largeArc} 1 ${x} ${y}`}
      fill="none"
      stroke={color}
      strokeWidth={2}
      opacity={0.7}
      style={{ pointerEvents: 'none' }}
    />
  );
}
