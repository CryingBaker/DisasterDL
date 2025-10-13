import React, { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import parseGeoraster from 'georaster';
import GeoRasterLayer from 'georaster-layer-for-leaflet';

// Custom component to handle the GeoRaster layer
const GeoRasterComponent = ({ prediction }) => {
  const map = useMap();
  const layerRef = useRef(null);

  useEffect(() => {
    if (prediction) {
      const reader = new FileReader();
      reader.onload = (event) => {
        parseGeoraster(event.target.result).then(georaster => {
          const layer = new GeoRasterLayer({
            georaster,
            opacity: 0.7,
            pixelValuesToColorFn: values => values[0] === 1 ? '#ff0000' : null,
            resolution: 256,
          });

          if (layerRef.current) {
            map.removeLayer(layerRef.current);
          }

          layer.addTo(map);
          layerRef.current = layer;

          // Fit map to the bounds of the new layer
          map.fitBounds(layer.getBounds());
        });
      };
      reader.readAsArrayBuffer(prediction);
    }
  }, [prediction, map]);

  return null;
};

const Map = ({ prediction, center }) => {
  return (
    <MapContainer center={center} zoom={13} style={{ height: '100%', width: '100%' }}>
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />
      {prediction && <GeoRasterComponent prediction={prediction} />}
    </MapContainer>
  );
};

export default Map;
