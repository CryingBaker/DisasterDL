import { MapContainer, TileLayer, ImageOverlay, useMap, LayersControl } from 'react-leaflet';
import { useEffect } from 'react';
import 'leaflet/dist/leaflet.css';

// Component to auto-center map
function CenterMap({ bounds }) {
    const map = useMap();
    useEffect(() => {
        if (bounds) {
            map.fitBounds(bounds);
        }
    }, [bounds, map]);
    return null;
}

const { BaseLayer } = LayersControl;

export default function MapViewer({
    bounds,
    groundTruthUrl,
    predictionUrl,
    opacity = 0.6,
    children
}) {
    if (!bounds) return <div className="glass-panel" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>Select a sample to view map</div>;

    return (
        <div className="glass-panel" style={{ height: '500px', width: '100%', overflow: 'hidden', padding: 0, position: 'relative' }}>
            <MapContainer
                bounds={bounds}
                style={{ height: '100%', width: '100%', background: '#f8fafc' }}
                zoomControl={true}
            >
                <LayersControl position="topright">
                    <BaseLayer checked name="Street Map">
                        <TileLayer
                            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
                        />
                    </BaseLayer>
                    <BaseLayer name="Satellite">
                        <TileLayer
                            attribution='&copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
                            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                        />
                    </BaseLayer>
                    <BaseLayer name="Terrain">
                        <TileLayer
                            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                            url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
                        />
                    </BaseLayer>
                </LayersControl>

                <CenterMap bounds={bounds} />
                {children}
            </MapContainer>

            {/* Legend Overlay */}
            <div style={{
                position: 'absolute',
                bottom: '20px',
                left: '20px',
                background: 'rgba(255, 255, 255, 0.95)',
                padding: '0.75rem',
                borderRadius: '8px',
                border: '1px solid var(--glass-border)',
                zIndex: 1000,
                boxShadow: '0 4px 6px rgba(0,0,0,0.05)',
                fontSize: '0.8rem'
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                    <div style={{ width: '16px', height: '16px', background: 'rgba(0, 100, 255, 0.7)', borderRadius: '3px' }}></div>
                    <span style={{ fontWeight: 600 }}>Ground Truth</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                    <div style={{ width: '16px', height: '16px', background: 'rgba(255, 60, 60, 0.7)', borderRadius: '3px' }}></div>
                    <span style={{ fontWeight: 600 }}>Model Prediction</span>
                </div>
                <div style={{ color: 'var(--text-secondary)', fontSize: '0.7rem', marginTop: '4px' }}>
                    Overlap appears purple
                </div>
            </div>
        </div>
    );
}
