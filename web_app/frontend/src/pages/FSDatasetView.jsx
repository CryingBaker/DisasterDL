import { useState, useEffect } from 'react';
import axios from 'axios';
import { Loader2, Database, ChevronDown, Globe } from 'lucide-react';
import { MapContainer, TileLayer, ImageOverlay, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

const API_BASE = 'http://localhost:5000/api/fs_dataset';

function FitBounds({ bounds }) {
    const map = useMap();
    useEffect(() => { if (bounds) map.fitBounds(bounds); }, [bounds, map]);
    return null;
}

export default function FSDatasetView() {
    const [dataset, setDataset] = useState([]);
    const [loading, setLoading] = useState(true);
    const [filterSplit, setFilterSplit] = useState('all');
    const [selectedItem, setSelectedItem] = useState(null);
    const [itemDetails, setItemDetails] = useState(null);
    const [itemLoading, setItemLoading] = useState(false);

    useEffect(() => {
        axios.get(`${API_BASE}/list`)
            .then(r => setDataset(r.data.data))
            .catch(console.error)
            .finally(() => setLoading(false));
    }, []);

    const fetchDetails = async (item) => {
        if (selectedItem?.uid === item.uid) return;
        setSelectedItem(item);
        setItemDetails(null);
        setItemLoading(true);
        try {
            const r = await axios.get(`${API_BASE}/image/${item.uid}`);
            setItemDetails(r.data);
        } catch (e) {
            console.error(e);
        } finally {
            setItemLoading(false);
        }
    };

    const filteredDataset = dataset.filter(i => filterSplit === 'all' || i.split === filterSplit);

    const SC = { 
        train: { bg: 'rgba(37,99,235,0.1)', fg: '#3b82f6' }, 
        val: { bg: 'rgba(234,179,8,0.1)', fg: '#ca8a04' }, 
        test: { bg: 'rgba(124,58,237,0.1)', fg: '#a855f7' } 
    };

    return (
        <div className="content-grid">
            {/* Sidebar list */}
            <div className="sidebar">
                <div className="glass-panel" style={{ height: 'calc(100vh - 120px)', display: 'flex', flexDirection: 'column' }}>
                    <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--glass-border)' }}>
                        <h2 style={{ margin: 0, fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <Database size={18} className="text-accent" /> Dataset Explorer
                        </h2>
                        <div style={{ display: 'flex', gap: '4px', marginTop: '1rem' }}>
                            {['all', 'train', 'val', 'test'].map(s => (
                                <button key={s} onClick={() => setFilterSplit(s)} 
                                    className={filterSplit === s ? 'btn-primary' : 'btn-secondary'}
                                    style={{ flex: 1, padding: '4px', fontSize: '0.7rem' }}>{s}</button>
                            ))}
                        </div>
                    </div>

                    <div style={{ flex: 1, overflowY: 'auto', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        {loading ? <div style={{ textAlign: 'center', padding: '2rem' }}><Loader2 className="animate-spin" /></div>
                        : filteredDataset.map(item => (
                            <div 
                                key={item.uid} 
                                onClick={() => fetchDetails(item)} 
                                style={{ 
                                    padding: '0.75rem', borderRadius: '8px', cursor: 'pointer', 
                                    background: selectedItem?.uid === item.uid ? 'var(--bg-secondary)' : 'rgba(255,255,255,0.02)',
                                    border: `1px solid ${selectedItem?.uid === item.uid ? 'var(--accent-primary)' : 'var(--glass-border)'}`,
                                    transition: 'all 0.15s'
                                }}
                            >
                                <div style={{ fontSize: '0.75rem', fontWeight: 600, fontFamily: 'monospace', marginBottom: '4px' }}>{item.uid}</div>
                                <span style={{ background: SC[item.split]?.bg, color: SC[item.split]?.fg, padding: '2px 6px', borderRadius: '4px', fontSize: '0.65rem', fontWeight: 700, textTransform: 'uppercase' }}>{item.split}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <div className="main-content">
                {!selectedItem ? (
                    <div className="glass-panel" style={{ height: '400px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', opacity: 0.3 }}>
                        <Globe size={48} style={{ marginBottom: '1rem' }} />
                        <p>Select a tile from the explorer to view details</p>
                    </div>
                ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                        <div className="glass-panel" style={{ padding: '1rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <h2 style={{ margin: 0, fontSize: '1rem', fontFamily: 'monospace' }}>{selectedItem.uid}</h2>
                            <span style={{ background: SC[selectedItem.split]?.bg, color: SC[selectedItem.split]?.fg, padding: '2px 8px', borderRadius: '4px', fontSize: '0.7rem', fontWeight: 700 }}>{selectedItem.split.toUpperCase()}</span>
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                            <div className="glass-panel" style={{ padding: '1rem' }}>
                                <h3 style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '1rem' }}>Ground Truth Overlay</h3>
                                <div style={{ height: '320px', borderRadius: '8px', overflow: 'hidden' }}>
                                    {itemDetails?.bounds ? (
                                        <MapContainer bounds={itemDetails.bounds} style={{ height: '100%' }}>
                                            <TileLayer url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" attribution="Esri" />
                                            <FitBounds bounds={itemDetails.bounds} />
                                            {itemDetails.gt_overlay && <ImageOverlay url={itemDetails.gt_overlay} bounds={itemDetails.bounds} opacity={0.8} />}
                                        </MapContainer>
                                    ) : <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000' }}>{itemLoading ? <Loader2 className="animate-spin" /> : 'Loading...'}</div>}
                                </div>
                            </div>

                            <div className="glass-panel" style={{ padding: '1rem' }}>
                                <h3 style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '1rem' }}>Satellite Imagery (Optical)</h3>
                                <div style={{ height: '320px', borderRadius: '8px', overflow: 'hidden', background: '#000' }}>
                                    {itemDetails?.post_s2_image ? (
                                        <img src={itemDetails.post_s2_image} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                                    ) : <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{itemLoading ? <Loader2 className="animate-spin" /> : 'Image N/A'}</div>}
                                </div>
                            </div>
                        </div>

                        {itemDetails && (
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
                                {[
                                    { label: 'Pre-Event SAR', src: itemDetails.pre_s1_image },
                                    { label: 'Pre-Event Optical', src: itemDetails.pre_s2_image },
                                    { label: 'Post-Event SAR', src: itemDetails.post_s1_image },
                                    { label: 'Post-Event Optical', src: itemDetails.post_s2_image }
                                ].map((img, i) => (
                                    <div key={i} className="glass-panel" style={{ padding: '0.5rem' }}>
                                        <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginBottom: '4px', textAlign: 'center' }}>{img.label}</div>
                                        <div style={{ aspectRatio: '1/1', background: '#000', borderRadius: '4px', overflow: 'hidden' }}>
                                            {img.src ? <img src={img.src} style={{ width: '100%', height: '100%', objectFit: 'contain' }} /> : <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.1 }}>N/A</div>}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
