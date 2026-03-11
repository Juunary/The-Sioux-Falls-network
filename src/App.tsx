// ============================================================
// App — router root. Wraps all pages with BrowserRouter + NavBar.
// ============================================================

import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import NavBar from './components/NavBar';
import SimulatorPage from './pages/SimulatorPage';
import DatasetsPage from './pages/DatasetsPage';
import TrainingPage from './pages/TrainingPage';
import ExperimentsPage from './pages/ExperimentsPage';
import ComparisonPage from './pages/ComparisonPage';

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <NavBar />
        <div className="page-content">
          <Routes>
            <Route path="/" element={<Navigate to="/simulator" replace />} />
            <Route path="/simulator"   element={<SimulatorPage />} />
            <Route path="/datasets"    element={<DatasetsPage />} />
            <Route path="/training"    element={<TrainingPage />} />
            <Route path="/experiments" element={<ExperimentsPage />} />
            <Route path="/comparison"  element={<ComparisonPage />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  );
}
