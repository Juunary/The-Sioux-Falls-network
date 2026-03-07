// ============================================================
// App — root component, mounts the simulation engine,
//        manages 2D/3D view toggle, lays out UI
// ============================================================

import React, { useState, Suspense } from 'react';
import { useSimulationEngine } from './sim/engine';
import Network2D from './components/Network2D';
import Network3D from './components/Network3D';
import Controls from './components/Controls';
import InfoPanel from './components/InfoPanel';

export default function App() {
  const [viewMode, setViewMode] = useState<'2d' | '3d'>('2d');

  // Mount the simulation RAF loop once at the root
  useSimulationEngine();

  return (
    <div className="app-layout">
      {/* Left sidebar — controls */}
      <aside className="sidebar sidebar-left">
        <Controls
          viewMode={viewMode}
          onToggleView={() => setViewMode((v) => (v === '2d' ? '3d' : '2d'))}
        />
      </aside>

      {/* Main canvas area */}
      <main className="canvas-area">
        {viewMode === '2d' ? (
          <Network2D />
        ) : (
          <Suspense fallback={<div className="loading">Loading 3D...</div>}>
            <Network3D />
          </Suspense>
        )}
      </main>

      {/* Right sidebar — info panel */}
      <aside className="sidebar sidebar-right">
        <InfoPanel />
      </aside>
    </div>
  );
}
