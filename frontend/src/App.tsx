import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/layout/Navbar';
import PadcomChat from './pages/PadcomChat';
import MedicalAnalysis from './pages/MedicalAnalysis';
import './App.css';

const App: React.FC = () => {
  return (
    <Router>
      <div className="min-h-screen bg-gray-900">
        <Navbar />
        <Routes>
          <Route path="/" element={<PadcomChat />} />
          <Route path="/medical" element={<MedicalAnalysis />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;