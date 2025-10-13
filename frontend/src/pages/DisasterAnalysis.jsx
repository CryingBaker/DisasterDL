import React, { useState } from 'react';
import Map from '../components/Map';
import axios from 'axios';

const DisasterAnalysis = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mapCenter, setMapCenter] = useState([19.0760, 72.8777]); // Default center

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setPredictionResult(null);
    setError(null);
  };

  const handlePrediction = async () => {
    if (!selectedFile) {
      setError('Please select a GeoTIFF file first.');
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://localhost:8000/predict/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        responseType: 'blob',
      });

      const imageBlob = new Blob([response.data], { type: 'image/tiff' });
      setPredictionResult(imageBlob);

    } catch (err) {
      setError('An error occurred during prediction. Please ensure the backend is running and the file is a valid GeoTIFF.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <h1>Disaster Analysis</h1>
      <div className="analysis-container">
        <div className="card map-card">
          <Map prediction={predictionResult} center={mapCenter} />
        </div>
        <div className="card controls-card">
          <h2>Upload SAR Image</h2>
          <p>Upload a GeoTIFF SAR image to see the flood prediction.</p>
          <div className="upload-section">
            <input type="file" accept=".tif,.tiff" onChange={handleFileChange} />
            <button onClick={handlePrediction} disabled={isLoading}>
              {isLoading ? 'Analyzing...' : 'Analyze Disaster Area'}
            </button>
          </div>
          {error && <p className="error-message">{error}</p>}
        </div>
      </div>
    </div>
  );
};

export default DisasterAnalysis;
