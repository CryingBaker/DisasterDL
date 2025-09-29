import React from 'react';
import { MapContainer, TileLayer, Polygon, CircleMarker, Popup, LayerGroup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

// --- Demo Data (Mumbai) ---
const center = [19.0760, 72.8777];

// A sample flood zone polygon
const floodZone = [
  [19.08, 72.87],
  [19.08, 72.88],
  [19.07, 72.88],
  [19.07, 72.87],
];

// Sample damaged buildings data
const damagedBuildings = [
  { id: 1, pos: [19.075, 72.875], severity: 'high', info: 'Building A: Severe structural damage' },
  { id: 2, pos: [19.078, 72.878], severity: 'medium', info: 'Building B: Partial collapse' },
  { id: 3, pos: [19.076, 72.87], severity: 'low', info: 'Building C: Minor damage' },
];

const severityColors = {
  high: 'red',
  medium: 'orange',
  low: 'yellow',
};

const Map = () => {
  return (
    <MapContainer center={center} zoom={15} style={{ height: '100%', width: '100%' }}>
      {/* Base Map Layer */}
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />

      {/* Disaster Data Layers */}
      <LayerGroup>
        {/* Flood Zone Polygon */}
        <Polygon pathOptions={{ color: 'blue', fillColor: 'lightblue' }} positions={floodZone} >
            <Popup>Affected Flood Zone</Popup>
        </Polygon>

        {/* Damaged Buildings Markers */}
        {damagedBuildings.map(building => (
          <CircleMarker
            key={building.id}
            center={building.pos}
            radius={8}
            pathOptions={{ color: severityColors[building.severity], fillColor: severityColors[building.severity], fillOpacity: 0.7 }}
          >
            <Popup>{building.info}</Popup>
          </CircleMarker>
        ))}
      </LayerGroup>
    </MapContainer>
  );
};

export default Map;
