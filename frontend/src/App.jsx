import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import DisasterAnalysis from './pages/DisasterAnalysis';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app-container">
        <Sidebar />
        <main className="content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/analysis" element={<DisasterAnalysis />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
