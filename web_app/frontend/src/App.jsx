import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import DatasetView from './pages/DatasetView';
import PredictionView from './pages/PredictionView';
import FSDatasetView from './pages/FSDatasetView';
import FSPredictView from './pages/FSPredictView';

function NavBar() {
  const location = useLocation();
  return (
    <nav className="glass-panel" style={{ borderRadius: 0, borderTop: 0, borderLeft: 0, borderRight: 0 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700 }}>
          DisasterDL <span className="gradient-text">Dashboard</span>
        </h1>
      </div>
      <div className="nav-links">
        <Link to="/fs-dataset" className={location.pathname === '/fs-dataset' || location.pathname === '/' ? 'active' : ''}>Flood Dataset</Link>
        <Link to="/fs-predict" className={location.pathname === '/fs-predict' ? 'active' : ''}>Flood Predict</Link>
        <Link to="/bd-dataset" className={location.pathname === '/bd-dataset' ? 'active' : ''}>BD Dataset</Link>
        <Link to="/bd-predict" className={location.pathname === '/bd-predict' ? 'active' : ''}>BD Predict</Link>
      </div>
    </nav>
  );
}

function App() {
  return (
    <Router>
      <div className="app-container">
        <NavBar />
        <div className="page-container">
          <Routes>
            <Route path="/" element={<FSDatasetView />} />
            <Route path="/fs-dataset" element={<FSDatasetView />} />
            <Route path="/fs-predict" element={<FSPredictView />} />
            <Route path="/bd-dataset" element={<DatasetView />} />
            <Route path="/bd-predict" element={<PredictionView />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
