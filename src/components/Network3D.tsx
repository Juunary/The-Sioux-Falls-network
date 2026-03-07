// ============================================================
// Network3D — @react-three/fiber visualization
// Same graph data + same simulation state as Network2D
// ============================================================

import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Line } from '@react-three/drei';
import * as THREE from 'three';
import { useSimStore } from '../sim/store';
import { NODES, EDGES } from '../data/network';
import { interpolateEdgeWithOffset } from '../utils/graph';
import type { Bus } from '../types/network';

// ---- Coordinate mapping ----
// SVG space (0..960 x 0..880) → 3D space centered around origin
// X: [0,960] → [-9.6, 9.6]   Y: [0,880] → [8.8, -8.8] (flip Y so top is +Y)
const SCALE = 0.02;
function toWorld(x: number, y: number, z = 0): [number, number, number] {
  return [(x - 480) * SCALE, -(y - 440) * SCALE, z];
}

// ============================================================
// Scene contents (inner component, inside Canvas)
// ============================================================
function Scene() {
  const buses = useSimStore((s) => s.buses);
  const settings = useSimStore((s) => s.settings);
  const selection = useSimStore((s) => s.selection);
  const setSelection = useSimStore((s) => s.setSelection);

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 15, 10]} intensity={1.2} />
      <pointLight position={[-10, -10, 5]} intensity={0.4} color="#4466ff" />

      {/* Edges */}
      {EDGES.map((edge) => {
        const src = NODES[edge.source];
        const tgt = NODES[edge.target];
        // Offset lines as in 2D (perpendicular 5px → 0.1 world units)
        const dx = tgt.x - src.x;
        const dy = tgt.y - src.y;
        const len = Math.sqrt(dx * dx + dy * dy) || 1;
        const ox = (-dy / len) * 5;
        const oy = (dx / len) * 5;

        const p1 = toWorld(src.x + ox, src.y + oy);
        const p2 = toWorld(tgt.x + ox, tgt.y + oy);

        const isSelected = edge.id === selection.selectedEdgeId;
        const color = isSelected ? '#f39c12' : '#445577';
        const lineWidth = isSelected ? 3 : 1;

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
        const pos = toWorld(node.x, node.y, 0);
        const isCS = node.isChargingStation;
        const isSelected = node.id === selection.selectedNodeId;

        return (
          <group key={node.id} position={pos}>
            {isCS && (
              <mesh>
                <sphereGeometry args={[0.28, 16, 16]} />
                <meshStandardMaterial
                  color="#f39c12"
                  emissive="#f39c12"
                  emissiveIntensity={0.5}
                  transparent
                  opacity={0.3}
                />
              </mesh>
            )}
            <mesh
              onClick={(e) => {
                e.stopPropagation();
                setSelection({ selectedNodeId: node.id, selectedBusId: null, selectedEdgeId: null });
              }}
            >
              <sphereGeometry args={[0.18, 16, 16]} />
              <meshStandardMaterial
                color={isCS ? '#f39c12' : isSelected ? '#3498db' : '#2c3e50'}
                emissive={isCS ? '#f39c12' : isSelected ? '#3498db' : '#000000'}
                emissiveIntensity={isCS ? 0.4 : isSelected ? 0.6 : 0}
              />
            </mesh>
            {settings.showLabels && (
              <Text
                position={[0, 0.28, 0.1]}
                fontSize={0.15}
                color="#ecf0f1"
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
      pos2D = interpolateEdgeWithOffset(edge, bus.progress, bus.laneOffset);
    } else {
      const node = NODES[bus.currentNode];
      pos2D = { x: node.x, y: node.y };
    }

    const [wx, wy, wz] = toWorld(pos2D.x, pos2D.y, 0.15);
    meshRef.current.position.set(wx, wy, wz);
  });

  const color = bus.state === 'charging' ? '#f1c40f' : bus.color;

  return (
    <mesh ref={meshRef} onClick={(e) => { e.stopPropagation(); onSelect(); }}>
      <boxGeometry args={[0.18, 0.12, 0.1]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={isSelected ? 0.8 : 0.3}
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
      camera={{ position: [0, 0, 18], fov: 55 }}
      style={{ background: '#0f0f23' }}
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
        maxDistance={40}
      />
    </Canvas>
  );
}
