import { useState, useRef } from 'react';
import { MapContainer, TileLayer, ImageOverlay } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import {
    Upload, Layers, Loader2, AlertCircle, CheckCircle2, Satellite, Waves
} from 'lucide-react';

const PREDICT_API = 'http://localhost:5000/api/fs_predict';

// ── Channel definitions ──────────────────────────────────────────────────────
const CHANNELS = [
    {
        key: 'post_s1', label: 'Post-Event S1', sublabel: 'SAR (VV, VH)',
        desc: 'Sentinel-1 SAR acquired after the flood event',
        bands: '2 bands', icon: '📡', required: true,
    },
    {
        key: 'post_s2', label: 'Post-Event S2', sublabel: 'Optical (B2,B3,B4,B8,B11,B12)',
        desc: 'Sentinel-2 multispectral image acquired after the flood',
        bands: '6 bands', icon: '🛰️', required: false,
    },
    {
        key: 'pre_s1', label: 'Pre-Event S1', sublabel: 'SAR (VV, VH)',
        desc: 'Sentinel-1 SAR composite from 3–33 days before the flood',
        bands: '2 bands', icon: '📡', required: false,
    },
    {
        key: 'pre_s2', label: 'Pre-Event S2', sublabel: 'Optical (B2,B3,B4,B8,B11,B12)',
        desc: 'Pre-flood Sentinel-2 cloud-masked composite',
        bands: '6 bands', icon: '🛰️', required: false,
    },
    {
        key: 'aux', label: 'Auxiliary', sublabel: 'SRTM + MERIT HAND + JRC',
        desc: 'Elevation, Height Above Nearest Drainage, JRC water occurrence/seasonality',
        bands: '4 bands', icon: '🗺️', required: false,
    },
];

// ── Preview image key for each channel ──────────────────────────────────────
const PREVIEW_KEY = {
    post_s1: 'post_s1_image',
    post_s2: 'post_s2_image',
    pre_s1:  'pre_s1_image',
    pre_s2:  'pre_s2_image',
};

export default function FSPredictView() {
    const [files, setFiles]       = useState({});  // key → File
    const [dragOver, setDragOver] = useState(null); // which slot is being hovered
    const [loading, setLoading]   = useState(false);
    const [error, setError]       = useState(null);
    const [results, setResults]   = useState(null);
    const fileInputRefs = useRef({});

    const handleFileChange = (key, file) => {
        if (!file) return;
        setFiles(prev => ({ ...prev, [key]: file }));
        setResults(null);
        setError(null);
    };

    const handleDrop = (key, e) => {
        e.preventDefault();
        setDragOver(null);
        const file = e.dataTransfer.files[0];
        if (file) handleFileChange(key, file);
    };

    const clearFile = (key) => {
        setFiles(prev => { const n = { ...prev }; delete n[key]; return n; });
        if (fileInputRefs.current[key]) fileInputRefs.current[key].value = '';
    };

    const handlePredict = async () => {
        if (!files.post_s1 && !files.post_s2) {
            setError('Please upload at least one post-event file (Post S1 or Post S2).');
            return;
        }
        setLoading(true);
        setError(null);
        setResults(null);

        const form = new FormData();
        for (const [key, file] of Object.entries(files)) {
            form.append(key, file);
        }

        try {
            const res = await fetch(`${PREDICT_API}/run`, { method: 'POST', body: form });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Prediction failed');
            setResults(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const bounds = results?.bounds;
    const hasMap = bounds && results?.pred_overlay;

    return (
        <div className="content-grid">
            {/* ── Sidebar ─────────────────────────────────────────────────── */}
            <div className="sidebar" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

                {/* Upload slots */}
                <div className="glass-panel" style={{ padding: '1.5rem' }}>
                    <h2 style={{ marginTop: 0, marginBottom: '0.5rem', fontSize: '1.15rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Layers size={20} className="text-accent" /> Upload Channels
                    </h2>
                    <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '1.25rem', lineHeight: 1.5 }}>
                        Upload GeoTIFFs for each channel. At minimum, provide <strong>Post-Event S1</strong> or <strong>Post-Event S2</strong>.
                    </p>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                        {CHANNELS.map(ch => {
                            const hasFile = !!files[ch.key];
                            const isOver  = dragOver === ch.key;
                            return (
                                <div key={ch.key}>
                                    <input
                                        ref={el => fileInputRefs.current[ch.key] = el}
                                        type="file" accept=".tif,.tiff"
                                        style={{ display: 'none' }}
                                        onChange={e => handleFileChange(ch.key, e.target.files[0])}
                                    />
                                    <div
                                        onClick={() => !hasFile && fileInputRefs.current[ch.key]?.click()}
                                        onDragOver={e => { e.preventDefault(); setDragOver(ch.key); }}
                                        onDragLeave={() => setDragOver(null)}
                                        onDrop={e => handleDrop(ch.key, e)}
                                        style={{
                                            padding: '0.75rem 1rem',
                                            borderRadius: '10px',
                                            border: hasFile
                                                ? '1px solid rgba(0,200,100,0.4)'
                                                : isOver
                                                    ? '1.5px dashed var(--accent-primary)'
                                                    : '1.5px dashed rgba(255,255,255,0.12)',
                                            background: hasFile
                                                ? 'rgba(0,180,80,0.07)'
                                                : isOver
                                                    ? 'rgba(99,179,237,0.07)'
                                                    : 'rgba(255,255,255,0.02)',
                                            cursor: hasFile ? 'default' : 'pointer',
                                            transition: 'all 0.15s ease',
                                            display: 'flex', alignItems: 'center', gap: '0.75rem',
                                        }}
                                    >
                                        {/* Status icon */}
                                        <div style={{ fontSize: '1.25rem', flexShrink: 0 }}>
                                            {hasFile ? '✅' : ch.icon}
                                        </div>

                                        {/* Labels */}
                                        <div style={{ flex: 1, minWidth: 0 }}>
                                            <div style={{ fontWeight: 600, fontSize: '0.88rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                                {ch.label}
                                                {ch.required && <span style={{ fontSize: '0.7rem', color: '#ff6b6b', fontWeight: 700 }}>required*</span>}
                                            </div>
                                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '1px' }}>
                                                {hasFile ? files[ch.key].name : `${ch.sublabel} · ${ch.bands}`}
                                            </div>
                                        </div>

                                        {/* Action button */}
                                        {hasFile ? (
                                            <button
                                                onClick={e => { e.stopPropagation(); clearFile(ch.key); }}
                                                style={{ background: 'rgba(255,100,100,0.15)', border: 'none', borderRadius: '6px', padding: '4px 8px', color: '#ff8080', cursor: 'pointer', fontSize: '0.75rem', fontWeight: 600 }}
                                            >
                                                ✕
                                            </button>
                                        ) : (
                                            <Upload size={15} style={{ color: 'var(--text-secondary)', flexShrink: 0 }} />
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    {/* Run button */}
                    <button
                        className="btn-primary"
                        style={{ width: '100%', marginTop: '1.25rem', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.5rem' }}
                        onClick={handlePredict}
                        disabled={loading || (!files.post_s1 && !files.post_s2)}
                    >
                        {loading
                            ? <><Loader2 size={18} className="animate-spin" /> Running Inference…</>
                            : <><Waves size={18} /> Run Flood Prediction</>
                        }
                    </button>

                    {error && (
                        <div style={{ marginTop: '1rem', padding: '0.85rem', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.25)', borderRadius: '8px', color: '#ef4444', fontSize: '0.85rem', display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
                            <AlertCircle size={15} style={{ marginTop: '2px', flexShrink: 0 }} />
                            <span>{error}</span>
                        </div>
                    )}
                </div>

                {/* Stats panel */}
                {results && (
                    <div className="glass-panel" style={{ padding: '1.5rem' }}>
                        <h2 style={{ marginTop: 0, marginBottom: '1rem', fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <CheckCircle2 size={18} style={{ color: '#4ade80' }} /> Flood Assessment
                        </h2>

                        <div style={{ marginBottom: '1.25rem', paddingBottom: '1rem', borderBottom: '1px solid var(--glass-border)' }}>
                            <div style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', fontWeight: 500, marginBottom: '4px' }}>Estimated Flooded Area</div>
                            <div style={{ fontSize: '1.6rem', fontWeight: 700 }}>
                                {results.estimated_area_km2.toFixed(3)}
                                <span style={{ fontSize: '0.95rem', fontWeight: 500, color: 'var(--text-secondary)', marginLeft: '4px' }}>km²</span>
                            </div>
                        </div>

                        {Object.entries(results.breakdown || {}).map(([cls, d]) => (
                            <div key={cls} style={{ marginBottom: '1rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px', fontSize: '0.88rem' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                        <div style={{ width: 10, height: 10, borderRadius: '50%', background: cls === 'Flooded' ? '#ff4444' : '#555' }} />
                                        <span style={{ fontWeight: 600 }}>{cls}</span>
                                    </div>
                                    <span style={{ fontWeight: 700 }}>{d.percentage}%</span>
                                </div>
                                <div style={{ height: 6, background: 'rgba(255,255,255,0.07)', borderRadius: 3, overflow: 'hidden' }}>
                                    <div style={{ height: '100%', width: `${d.percentage}%`, background: cls === 'Flooded' ? 'linear-gradient(90deg,#ff4444,#ff8800)' : '#444', borderRadius: 3, transition: 'width 0.5s ease' }} />
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* ── Main panel ──────────────────────────────────────────────── */}
            <div className="main-content" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700 }}>Prediction Results</h2>

                {/* Map */}
                <div className="glass-panel" style={{ padding: '1rem' }}>
                    <h3 style={{ margin: '0 0 0.75rem', fontSize: '0.88rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        Flood Prediction Map
                    </h3>
                    {hasMap ? (
                        <>
                            <div style={{ height: '360px', borderRadius: '10px', overflow: 'hidden', border: '1px solid var(--glass-border)' }}>
                                <MapContainer bounds={bounds} style={{ height: '100%', width: '100%' }} scrollWheelZoom>
                                    <TileLayer
                                        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                                        attribution="Tiles © Esri"
                                        maxZoom={18}
                                    />
                                    <ImageOverlay url={results.pred_overlay} bounds={bounds} opacity={1} />
                                </MapContainer>
                            </div>
                            <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.65rem', flexWrap: 'wrap' }}>
                                {[
                                    { color: 'rgba(255,0,0,0.7)', label: 'Predicted Flood' },
                                    { color: 'rgba(0,0,0,0)', label: 'Dry / No Data', border: '1px dashed rgba(255,255,255,0.2)' },
                                ].map(({ color, label, border }) => (
                                    <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.82rem', fontWeight: 600 }}>
                                        <div style={{ width: 14, height: 14, borderRadius: 3, background: color, border: border || 'none' }} />
                                        <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
                                    </div>
                                ))}
                            </div>
                        </>
                    ) : (
                        <div style={{ height: '360px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1rem', color: 'var(--text-secondary)', borderRadius: '10px', border: '1px dashed rgba(255,255,255,0.08)' }}>
                            {loading
                                ? <><Loader2 size={36} className="animate-spin" /><span style={{ fontWeight: 500 }}>Analysing 20-channel inputs…</span></>
                                : <><Satellite size={40} opacity={0.3} /><span style={{ fontWeight: 500, fontSize: '0.95rem' }}>Upload files → Run Prediction to see the flood map</span></>
                            }
                        </div>
                    )}
                </div>

                {/* Preview images */}
                {results && (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem' }}>
                        {CHANNELS.filter(c => c.key !== 'aux' && results[PREVIEW_KEY[c.key]]).map(ch => (
                            <div key={ch.key} className="glass-panel" style={{ padding: '0.85rem' }}>
                                <h4 style={{ margin: '0 0 0.6rem', fontSize: '0.8rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                    {ch.icon} {ch.label}
                                </h4>
                                <img
                                    src={results[PREVIEW_KEY[ch.key]]}
                                    alt={ch.label}
                                    style={{ width: '100%', aspectRatio: '1/1', objectFit: 'contain', borderRadius: '8px', background: '#000' }}
                                />
                            </div>
                        ))}
                    </div>
                )}

                {/* Empty state */}
                {!results && !loading && (
                    <div className="glass-panel" style={{ padding: '3rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem', color: 'var(--text-secondary)', textAlign: 'center' }}>
                        <div style={{ display: 'flex', gap: '1rem' }}>
                            {['📡', '🛰️', '🗺️'].map(e => <span key={e} style={{ fontSize: '2rem', opacity: 0.4 }}>{e}</span>)}
                        </div>
                        <div>
                            <div style={{ fontWeight: 600, marginBottom: '0.5rem', fontSize: '1rem' }}>Upload GeoTIFFs to begin</div>
                            <div style={{ fontSize: '0.85rem', maxWidth: 380, lineHeight: 1.6 }}>
                                Upload your Sentinel-1, Sentinel-2, and auxiliary GeoTIFF files using the panel on the left, then click <strong>Run Flood Prediction</strong>.
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
