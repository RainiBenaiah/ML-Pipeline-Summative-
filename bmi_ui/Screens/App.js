// App.js - Main component
import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, Navigate } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

// Import components for each screen
import LandingPage from './components/LandingPage';
import PredictionScreen from './components/PredictionScreen';
import UploadScreen from './components/UploadScreen';
import RetrainScreen from './components/RetrainScreen';
import FeatureImportanceScreen from './components/FeatureImportanceScreen';
import VisualizationScreen from './components/VisualizationScreen';
import Navbar from './components/Navbar';

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <div className="container mt-4">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/predict" element={<PredictionScreen />} />
            <Route path="/upload" element={<UploadScreen />} />
            <Route path="/retrain" element={<RetrainScreen />} />
            <Route path="/feature-importance" element={<FeatureImportanceScreen />} />
            <Route path="/visualization" element={<VisualizationScreen />} />
            <Route path="*" element={<Navigate to="/" />} />
          </Routes>
        </div>
        <footer className="bg-dark text-white text-center py-3 mt-5">
          <div className="container">
            <p className="mb-0">© 2025 TRUTH WEIGHTS - Obesity Classification & Risk Assessment</p>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;