// App.js - Main component
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

// Import components for each screen
import LandingPage from './Screens/LandingPage';
import PredictionScreen from './Screens/PredictionScreen';
import UploadScreen from './Screens/UploadScreen';
import RetrainScreen from './Screens/RetrainScreen';
import FeatureImportanceScreen from './Screens/FeatureImportanceScreen';
import VisualizationScreen from './Screens/VisualizationScreen';


function App() {
  return (
    <Router>
      <div className="App">
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
