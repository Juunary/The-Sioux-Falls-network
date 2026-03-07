// ============================================================
// Network3D — @react-three/fiber visualization
// Matches the Sioux Falls layout from Network2D
// ============================================================

import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Line } from '@react-three/drei';
import * as THREE from 'three';
import { useSimStore } from '../sim/store';
import { NODES, EDGES } from '../data/network';
import { SIOUX_FALLS_LAYOUT } from './Network2D';
import type { Bus } from '../types/network';

// ---- Coordinate mapping ----
// SVG space (0..850 x 0..1111) → 3D space centered around origin
// Center: (425, 555)  Scale: 0.02
// X: [0,850] → [-8.5, 8.5]   Y flipped so top is +Y → [-5.56, +5.56]
const SCALE = 0.02;
const CX = 425;
const CY = 555;

function toWorld(x: number, y: number, z = 0): [number, number, number] {
  return [(x - CX) * SCALE, -(y - CY) * SCALE, z];
}

// Build a node position map using SIOUX_FALLS_LAYOUT, fallback to NODES data
function getLayoutPos(nodeId: number): { x: number; y: number } {
  return SIOUX_FALLS_LAYOUT[nodeId] ?? { x: 0, y: 0 };
}

// ============================================================
// Scene contents (inner component, inside Canvas)
// ============================================================
function Scene() {
  const buses = useSimStore((s) => s.buses);
  const settings = useSimStore((s) => s.settings);
  const selection = useSimStore((s) => s.selection);
  const setSelection = useSimStore((s) => s.setSelection);

  // Build edge offset in 3D (matches 2D EDGE_OFFSET = 6px)
  const EDGE_OFFSET_3D = 6 * SCALE;

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={1.0} />
      <directionalLight position={[8, 12, 8]} intensity={1.0} />
      <pointLight position={[-8, -8, 4]} intensity={0.3} color="#aaccff" />

      {/* Edges */}
      {EDGES.map((edge) => {
        const srcPos = getLayoutPos(edge.source);
        const tgtPos = getLayoutPos(edge.target);

        const dx = tgtPos.x - srcPos.x;
        const dy = tgtPos.y - srcPos.y;
        const len = Math.hypot(dx, dy) || 1;

        // Perpendicular offset (same as 2D EDGE_OFFSET=6)
        const ox = (-dy / len) * 6;
        const oy = (dx / len) * 6;

        const p1 = toWorld(srcPos.x + ox, srcPos.y + oy);
        const p2 = toWorld(tgtPos.x + ox, tgtPos.y + oy);

        const isSelected = edge.id === selection.selectedEdgeId;
        const isRouteEdge = false; // route highlight handled by bus selection
        const color = isSelected ? '#2563eb' : '#333';
        const lineWidth = isSelected ? 3 : 1.5;

        return (
          <Line
            key={edge.id}
            points={[p1, p2]}
            color={color}
            lineWidth={lineWidth}
            onClick={(e) => {
              e.stopPropagation();
              setSelection({ selectedEdgeId: edge.id, selectedBusId: null, selectedNodeId: null });
            }}
          />
        );
      })}

      {/* Nodes */}
      {NODES.map((node) => {
        const pos2D = getLayoutPos(node.id);
        const pos = toWorld(pos2D.x, pos2D.y, 0);
        const isCS = node.isChargingStation;
        const isSelected = node.id === selection.selectedNodeId;

        // Colors match 2D: white node, yellow CS, blue border when selected
        const fillColor = isCS ? '#f4ec00' : '#ffffff';
        const emissiveColor = isSelected ? '#2563eb' : isCS ? '#c8b800' : '#000000';

        return (
          <group key={node.id} position={pos}>
            {isSelected && (
              <mesh>
                <sphereGeometry args={[0.36, 16, 16]} />
                <meshStandardMaterial
                  color="#2563eb"
                  transparent
                  opacity={0.25}
                />
              </mesh>
            )}
            <mesh
              onClick={(e) => {
                e.stopPropagation();
                setSelection({ selectedNodeId: node.id, selectedBusId: null, selectedEdgeId: null });
              }}
            >
              <sphereGeometry args={[0.22, 16, 16]} />
              <meshStandardMaterial
                color={fillColor}
                emissive={emissiveColor}
                emissiveIntensity={isSelected ? 0.5 : isCS ? 0.3 : 0}
              />
            </mesh>
            {settings.showLabels && (
              <Text
                position={[0, 0.32, 0.1]}
                fontSize={0.16}
                color="#111"
                anchorX="center"
                anchorY="bottom"
              >
                {String(node.id)}
              </Text>
            )}
          </group>
        );
      })}

      {/* Buses */}
      {buses.map((bus) => (
        <Bus3D
          key={bus.id}
          bus={bus}
          isSelected={bus.id === selection.selectedBusId}
          onSelect={() =>
            setSelection({ selectedBusId: bus.id, selectedNodeId: null, selectedEdgeId: null })
          }
        />
      ))}
    </>
  );
}

// ---- Animated bus mesh ----
function Bus3D({
  bus,
  isSelected,
  onSelect,
}: {
  bus: Bus;
  isSelected: boolean;
  onSelect: () => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame(() => {
    if (!meshRef.current) return;

    let pos2D: { x: number; y: number };

    if (bus.currentEdge != null) {
      const edge = EDGES[bus.currentEdge];
      if (edge) {
        const srcPos = getLayoutPos(edge.source);
        const tgtPos = getLayoutPos(edge.target);
        const t = Math.max(0, Math.min(1, bus.progress));
        const dx = tgtPos.x - srcPos.x;
        const dy = tgtPos.y - srcPos.y;
        const len = Math.hypot(dx, dy) || 1;
        const ox = (-dy / len) * bus.laneOffset;
        const oy = (dx / len) * bus.laneOffset;
        pos2D = {
          x: srcPos.x + dx * t + ox,
          y: srcPos.y + dy * t + oy,
        };
      } else {
        pos2D = getLayoutPos(bus.currentNode);
      }
    } else {
      pos2D = getLayoutPos(bus.currentNode);
    }

    const [wx, wy, wz] = toWorld(pos2D.x, pos2D.y, 0.18);
    meshRef.current.position.set(wx, wy, wz);
  });

  const color = bus.state === 'charging' ? '#f1c40f' : bus.color;

  return (
    <mesh
      ref={meshRef}
      onClick={(e) => {
        e.stopPropagation();
        onSelect();
      }}
    >
      <boxGeometry args={[0.2, 0.14, 0.12]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={isSelected ? 0.9 : 0.35}
      />
    </mesh>
  );
}

// ============================================================
// Public component
// ============================================================
export default function Network3D() {
  return (
    <Canvas
      camera={{ position: [0, 0, 20], fov: 50 }}
      style={{ background: '#d0d0d0' }}
      onClick={() => {
        useSimStore.getState().setSelection({
          selectedBusId: null,
          selectedNodeId: null,
          selectedEdgeId: null,
        });
      }}
    >
      <Scene />
      <OrbitControls
        enablePan
        enableZoom
        enableRotate
        minDistance={3}
        maxDistance={50}
      />
    </Canvas>
  );
}
