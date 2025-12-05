import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/layout/Navbar';
import Sidebar from './components/layout/Sidebar';
import Dashboard from './pages/Dashboard';
import Settings from './pages/Settings';
import LoadingScreen from './components/LoadingScreen';
import './App.css';

// Lazy load the pages
const BrainTumorAnalysis = lazy(() => import('./pages/BrainTumorAnalysis'));
const PneumoniaAnalysis = lazy(() => import('./pages/PneumoniaAnalysis'));
const Analysis3D = lazy(() => import('./pages/Analysis3D'));

const App: React.FC = () => {
  return (
    <Router>
      <div className="min-h-screen bg-medical-background">
        <Navbar />
        <div className="flex">
          <Sidebar />
          <main className="flex-1 p-8">
            <Suspense fallback={<LoadingScreen />}>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/brain-tumor" element={<BrainTumorAnalysis />} />
                <Route path="/pneumonia" element={<PneumoniaAnalysis />} />
                <Route path="/analysis-3d" element={<Analysis3D />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </Suspense>
          </main>
        </div>
      </div>
    </Router>
  );
};

export default App;