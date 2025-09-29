import React from 'react';
import Map from '../components/Map';

const DisasterAnalysis = () => {
  return (
    <div>
      <h1>Disaster Analysis</h1>
      <div className="analysis-container">
        <div className="card map-card">
          <Map />
        </div>
        <div className="card image-card">
          <h2>Affected Area Images</h2>
          <p>Satellite images with damage assessment markings will be displayed here.</p>
          <div className="image-placeholder">Images will be highlighted here</div>
        </div>
      </div>
    </div>
  );
};

export default DisasterAnalysis;
