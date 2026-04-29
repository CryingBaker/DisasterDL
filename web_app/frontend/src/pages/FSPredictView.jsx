import { useState, useRef, useEffect } from 'react';
import { MapContainer, TileLayer, ImageOverlay } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import {
    Upload, Layers, Loader2, AlertCircle, CheckCircle2, Satellite, Waves, ChevronDown, FileText,
    TrendingUp, ArrowUpCircle, Activity, Info, Download
} from 'lucide-react';
import FloodReportDialog from '../components/FloodReportDialog';

const PREDICT_API = 'http://localhost:5000/api/fs_predict';
const DATASET_API = 'http://localhost:5000/api/fs_dataset';

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
    const [files, setFiles]       = useState({});
    const [dragOver, setDragOver] = useState(null);
    const [loading, setLoading]   = useState(false);
    const [error, setError]       = useState(null);
    const [results, setResults]   = useState(null);
    const [reportOpen, setReportOpen] = useState(false);
    const fileInputRefs = useRef({});

    // ── Water Rise Simulation ────────────────────────────────────────────
    const [waterRise, setWaterRise]   = useState(2.0);
    const [simLoading, setSimLoading] = useState(false);
    const [simResults, setSimResults] = useState(null);
    const [simError, setSimError]     = useState(null);
    const [overlayMode, setOverlayMode] = useState('original'); // 'original' | 'simulated'

    // ── Model selection ──────────────────────────────────────────────────
    const [models, setModels]             = useState([]);
    const [selectedModel, setSelectedModel] = useState('');

    useEffect(() => {
        fetch(`${DATASET_API}/models`)
            .then(r => r.json())
            .then(data => {
                setModels(data.models || []);
                setSelectedModel(data.default || '');
            })
            .catch(console.error);
    }, []);

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
        setSimResults(null);
        setOverlayMode('original');

        const form = new FormData();
        for (const [key, file] of Object.entries(files)) {
            form.append(key, file);
        }
        if (selectedModel) form.append('model', selectedModel);

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

    // ── Water Rise Simulation handler ────────────────────────────────────
    const handleSimulate = async () => {
        if (!results?.mask_b64 || !files.aux) {
            setSimError('Auxiliary GeoTIFF with elevation data is required for simulation.');
            return;
        }
        setSimLoading(true);
        setSimError(null);
        setSimResults(null);

        const form = new FormData();
        form.append('aux', files.aux);
        form.append('mask_b64', results.mask_b64);
        form.append('mask_shape', JSON.stringify(results.mask_shape));
        form.append('water_rise', waterRise.toString());

        try {
            const res = await fetch(`${PREDICT_API}/simulate`, { method: 'POST', body: form });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Simulation failed');
            setSimResults(data);
            setOverlayMode('simulated');
        } catch (err) {
            setSimError(err.message);
        } finally {
            setSimLoading(false);
        }
    };

    const bounds = results?.bounds;
    const hasMap = bounds && results?.pred_overlay;
    const activeOverlay = overlayMode === 'simulated' && simResults?.sim_overlay
        ? simResults.sim_overlay
        : results?.pred_overlay;

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

                    {/* Model Selector */}
                    {models.length > 0 && (
                        <div style={{ marginBottom: '1rem' }}>
                            <div style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.35rem' }}>Model</div>
                            <div style={{ position: 'relative' }}>
                                <select
                                    id="fs-predict-model-selector"
                                    value={selectedModel}
                                    onChange={e => { setSelectedModel(e.target.value); setResults(null); }}
                                    style={{ width: '100%', padding: '0.45rem 2rem 0.45rem 0.6rem', borderRadius: '7px', border: '1px solid var(--glass-border)', background: 'rgba(255,255,255,0.04)', color: 'var(--text-primary)', fontSize: '0.75rem', fontWeight: 600, cursor: 'pointer', appearance: 'none', WebkitAppearance: 'none', outline: 'none' }}
                                >
                                    {models.map(m => <option key={m} value={m} style={{ background: '#1a1a2e', color: '#e0e0e0' }}>{m}</option>)}
                                </select>
                                <ChevronDown size={14} style={{ position: 'absolute', right: '0.5rem', top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', color: 'var(--text-secondary)' }} />
                            </div>
                        </div>
                    )}

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
                                            padding: '0.75rem 1rem', borderRadius: '10px',
                                            border: hasFile ? '1px solid rgba(0,200,100,0.4)' : isOver ? '1.5px dashed var(--accent-primary)' : '1.5px dashed rgba(255,255,255,0.12)',
                                            background: hasFile ? 'rgba(0,180,80,0.07)' : isOver ? 'rgba(99,179,237,0.07)' : 'rgba(255,255,255,0.02)',
                                            cursor: hasFile ? 'default' : 'pointer', transition: 'all 0.15s ease',
                                            display: 'flex', alignItems: 'center', gap: '0.75rem',
                                        }}
                                    >
                                        <div style={{ fontSize: '1.25rem', flexShrink: 0 }}>{hasFile ? '✅' : ch.icon}</div>
                                        <div style={{ flex: 1, minWidth: 0 }}>
                                            <div style={{ fontWeight: 600, fontSize: '0.88rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                                {ch.label}
                                                {ch.required && <span style={{ fontSize: '0.7rem', color: '#ff6b6b', fontWeight: 700 }}>required*</span>}
                                            </div>
                                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '1px' }}>
                                                {hasFile ? files[ch.key].name : `${ch.sublabel} · ${ch.bands}`}
                                            </div>
                                        </div>
                                        {hasFile ? (
                                            <button onClick={e => { e.stopPropagation(); clearFile(ch.key); }}
                                                style={{ background: 'rgba(255,100,100,0.15)', border: 'none', borderRadius: '6px', padding: '4px 8px', color: '#ff8080', cursor: 'pointer', fontSize: '0.75rem', fontWeight: 600 }}>✕</button>
                                        ) : (
                                            <Upload size={15} style={{ color: 'var(--text-secondary)', flexShrink: 0 }} />
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    <button className="btn-primary" style={{ width: '100%', marginTop: '1.25rem', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.5rem' }}
                        onClick={handlePredict} disabled={loading || (!files.post_s1 && !files.post_s2)}>
                        {loading ? <><Loader2 size={18} className="animate-spin" /> Running Inference…</> : <><Waves size={18} /> Run Flood Prediction</>}
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
                        {results.model_used && (
                            <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.5rem', fontWeight: 500 }}>
                                🧠 {results.model_used}
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* ── Main panel ──────────────────────────────────────────────── */}
            <div className="main-content" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

                {/* Heading row with action buttons */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700 }}>Prediction Results</h2>
                    {results && (
                        <button
                            id="fs-generate-report-btn"
                            onClick={() => setReportOpen(true)}
                            style={{
                                display: 'flex', alignItems: 'center', gap: '0.45rem',
                                padding: '0.55rem 1.1rem', borderRadius: '10px', border: 'none',
                                background: 'linear-gradient(135deg, #2563eb, #7c3aed)',
                                color: '#fff', fontWeight: 600, fontSize: '0.82rem',
                                cursor: 'pointer',
                                transition: 'all 0.2s cubic-bezier(0.4,0,0.2,1)',
                                boxShadow: '0 4px 12px rgba(37,99,235,0.25)',
                            }}
                            onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = '0 8px 20px rgba(37,99,235,0.35)'; }}
                            onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 4px 12px rgba(37,99,235,0.25)'; }}
                        >
                            <Download size={15} /> Export Report
                        </button>
                    )}
                </div>

                {/* Map panel */}
                <div className="glass-panel" style={{ padding: '1rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                        <h3 style={{ margin: 0, fontSize: '0.88rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Flood Prediction Map</h3>

                        {/* Overlay Toggle */}
                        {simResults && (
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', background: 'rgba(0,0,0,0.04)', borderRadius: '8px', padding: '3px' }}>
                                {[
                                    { key: 'original', label: 'Original' },
                                    { key: 'simulated', label: 'Simulated' },
                                ].map(opt => (
                                    <button
                                        key={opt.key}
                                        onClick={() => setOverlayMode(opt.key)}
                                        style={{
                                            padding: '0.3rem 0.7rem', borderRadius: '6px',
                                            border: 'none', fontSize: '0.72rem', fontWeight: 600,
                                            cursor: 'pointer', transition: 'all 0.15s',
                                            background: overlayMode === opt.key
                                                ? 'linear-gradient(135deg, #2563eb, #7c3aed)'
                                                : 'transparent',
                                            color: overlayMode === opt.key ? '#fff' : 'var(--text-secondary)',
                                            boxShadow: overlayMode === opt.key ? '0 2px 8px rgba(37,99,235,0.25)' : 'none',
                                        }}
                                    >
                                        {opt.label}
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>
                    {hasMap ? (
                        <>
                            <div style={{ height: '360px', borderRadius: '10px', overflow: 'hidden', border: '1px solid var(--glass-border)' }}>
                                <MapContainer bounds={bounds} style={{ height: '100%', width: '100%' }} scrollWheelZoom>
                                    <TileLayer url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" attribution="Tiles © Esri" maxZoom={18} />
                                    <ImageOverlay url={activeOverlay} bounds={bounds} opacity={1} />
                                </MapContainer>
                            </div>
                            <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.65rem', flexWrap: 'wrap' }}>
                                {[
                                    { color: 'rgba(255,0,0,0.7)', label: 'Predicted Flood' },
                                    ...(overlayMode === 'simulated' && simResults ? [{ color: 'rgba(255,165,0,0.7)', label: 'Simulated Flood' }] : []),
                                    { color: 'rgba(0,0,0,0)', label: 'Dry / No Data', border: '1px dashed rgba(0,0,0,0.15)' },
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
                            {loading ? <><Loader2 size={36} className="animate-spin" /><span style={{ fontWeight: 500 }}>Analysing inputs…</span></>
                            : <><Satellite size={40} opacity={0.3} /><span style={{ fontWeight: 500, fontSize: '0.95rem' }}>Upload files → Run Prediction to see the flood map</span></>}
                        </div>
                    )}
                </div>

                {/* ── Water Rise Simulator — wide horizontal layout ────────── */}
                {results && (
                    <div className="glass-panel" style={{ padding: '1.25rem 1.5rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: files.aux ? '1rem' : '0' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                                <div style={{
                                    width: 32, height: 32, borderRadius: 8,
                                    background: 'linear-gradient(135deg, #06b6d4, #0ea5e9)',
                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                }}>
                                    <TrendingUp size={16} color="#fff" />
                                </div>
                                <div>
                                    <div style={{ fontWeight: 700, fontSize: '0.95rem', color: 'var(--text-primary)' }}>Water Rise Simulator</div>
                                    <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>Elevation-based flood expansion using SRTM data</div>
                                </div>
                            </div>
                            {!files.aux && (
                                <div style={{
                                    padding: '0.4rem 0.85rem', borderRadius: '8px',
                                    background: 'rgba(14,165,233,0.06)',
                                    border: '1px solid rgba(14,165,233,0.15)',
                                    display: 'flex', alignItems: 'center', gap: '0.4rem',
                                    fontSize: '0.75rem', color: '#0284c7', fontWeight: 500,
                                }}>
                                    <Info size={13} />
                                    Upload Auxiliary GeoTIFF to enable
                                </div>
                            )}
                        </div>

                        {files.aux && (
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: '1.5rem', alignItems: 'start' }}>
                                {/* Left: Controls */}
                                <div>
                                    {/* Slider row */}
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
                                        {/* Elevation stats chips */}
                                        {results.elevation_stats && (
                                            <div style={{ display: 'flex', gap: '0.5rem', flexShrink: 0 }}>
                                                {[
                                                    { label: 'Ref', val: `${results.elevation_stats.median.toFixed(1)}m` },
                                                    { label: 'Range', val: `${results.elevation_stats.min.toFixed(1)}–${results.elevation_stats.max.toFixed(1)}m` },
                                                ].map(s => (
                                                    <div key={s.label} style={{
                                                        padding: '0.35rem 0.6rem', borderRadius: '7px',
                                                        background: 'rgba(14,165,233,0.06)',
                                                        border: '1px solid rgba(14,165,233,0.12)',
                                                        fontSize: '0.68rem', lineHeight: 1.3,
                                                    }}>
                                                        <div style={{ fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', fontSize: '0.58rem', letterSpacing: '0.04em' }}>{s.label}</div>
                                                        <div style={{ fontWeight: 700, color: '#0369a1' }}>{s.val}</div>
                                                    </div>
                                                ))}
                                            </div>
                                        )}

                                        {/* Slider */}
                                        <div style={{ flex: 1 }}>
                                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                                                <span style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Water Level Rise</span>
                                                <span style={{
                                                    fontSize: '1.05rem', fontWeight: 800,
                                                    background: 'linear-gradient(135deg, #0ea5e9, #6366f1)',
                                                    WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                                                    backgroundClip: 'text',
                                                }}>
                                                    +{waterRise.toFixed(1)}m
                                                </span>
                                            </div>
                                            <input
                                                id="fs-water-rise-slider"
                                                type="range" min="0" max="10" step="0.5"
                                                value={waterRise}
                                                onChange={e => setWaterRise(parseFloat(e.target.value))}
                                                style={{
                                                    width: '100%', height: '6px',
                                                    appearance: 'none', WebkitAppearance: 'none',
                                                    borderRadius: '3px', outline: 'none',
                                                    background: `linear-gradient(90deg, #0ea5e9 ${waterRise * 10}%, #e2e8f0 ${waterRise * 10}%)`,
                                                    cursor: 'pointer',
                                                }}
                                            />
                                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem', color: '#94a3b8', marginTop: '2px' }}>
                                                <span>0m</span><span>5m</span><span>10m</span>
                                            </div>
                                        </div>

                                        {/* Simulate button */}
                                        <button
                                            id="fs-simulate-btn"
                                            onClick={handleSimulate}
                                            disabled={simLoading}
                                            style={{
                                                display: 'flex', alignItems: 'center', gap: '0.4rem',
                                                padding: '0.55rem 1.1rem', borderRadius: '10px', border: 'none',
                                                background: 'linear-gradient(135deg, #06b6d4, #0ea5e9)',
                                                color: '#fff', fontWeight: 600, fontSize: '0.8rem',
                                                cursor: simLoading ? 'wait' : 'pointer',
                                                transition: 'all 0.2s', opacity: simLoading ? 0.7 : 1,
                                                boxShadow: '0 4px 12px rgba(6,182,212,0.25)',
                                                whiteSpace: 'nowrap', flexShrink: 0,
                                            }}
                                        >
                                            {simLoading
                                                ? <><Loader2 size={15} className="animate-spin" /> Simulating…</>
                                                : <><ArrowUpCircle size={15} /> Simulate</>
                                            }
                                        </button>
                                    </div>

                                    {simError && (
                                        <div style={{ marginTop: '0.6rem', padding: '0.55rem 0.75rem', background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.15)', borderRadius: '8px', color: '#ef4444', fontSize: '0.78rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                                            <AlertCircle size={13} style={{ flexShrink: 0 }} />
                                            <span>{simError}</span>
                                        </div>
                                    )}
                                </div>

                                {/* Right: Simulation Results */}
                                {simResults && (
                                    <div style={{
                                        display: 'flex', alignItems: 'center', gap: '0.75rem',
                                        padding: '0.6rem 1rem', borderRadius: '10px',
                                        background: 'rgba(249,115,22,0.05)',
                                        border: '1px solid rgba(249,115,22,0.15)',
                                    }}>
                                        <Activity size={16} style={{ color: '#ea580c', flexShrink: 0 }} />
                                        <div style={{ display: 'flex', gap: '1.25rem' }}>
                                            <div>
                                                <div style={{ fontSize: '0.58rem', fontWeight: 600, color: '#9a3412', textTransform: 'uppercase', letterSpacing: '0.04em' }}>New Flooding</div>
                                                <div style={{ fontSize: '1rem', fontWeight: 800, color: '#ea580c', lineHeight: 1.2 }}>
                                                    +{simResults.additional_area_km2.toFixed(3)}
                                                    <span style={{ fontSize: '0.65rem', fontWeight: 500, color: '#9a3412', marginLeft: '2px' }}>km²</span>
                                                </div>
                                            </div>
                                            <div style={{ width: 1, background: 'rgba(249,115,22,0.2)' }} />
                                            <div>
                                                <div style={{ fontSize: '0.58rem', fontWeight: 600, color: '#991b1b', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Total Flooded</div>
                                                <div style={{ fontSize: '1rem', fontWeight: 800, color: '#dc2626', lineHeight: 1.2 }}>
                                                    {simResults.total_area_km2.toFixed(3)}
                                                    <span style={{ fontSize: '0.65rem', fontWeight: 500, color: '#991b1b', marginLeft: '2px' }}>km²</span>
                                                </div>
                                            </div>
                                            <div style={{ width: 1, background: 'rgba(249,115,22,0.2)' }} />
                                            <div>
                                                <div style={{ fontSize: '0.58rem', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Threshold</div>
                                                <div style={{ fontSize: '1rem', fontWeight: 800, color: '#0369a1', lineHeight: 1.2 }}>
                                                    {simResults.threshold}
                                                    <span style={{ fontSize: '0.65rem', fontWeight: 500, color: '#64748b', marginLeft: '2px' }}>m</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                )}

                {results && (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem' }}>
                        {CHANNELS.filter(c => c.key !== 'aux' && results[PREVIEW_KEY[c.key]]).map(ch => (
                            <div key={ch.key} className="glass-panel" style={{ padding: '0.85rem' }}>
                                <h4 style={{ margin: '0 0 0.6rem', fontSize: '0.8rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{ch.icon} {ch.label}</h4>
                                <img src={results[PREVIEW_KEY[ch.key]]} alt={ch.label} style={{ width: '100%', aspectRatio: '1/1', objectFit: 'contain', borderRadius: '8px', background: '#000' }} />
                            </div>
                        ))}
                    </div>
                )}

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

            {/* Flood Report Dialog */}
            <FloodReportDialog
                open={reportOpen}
                onClose={() => setReportOpen(false)}
                results={results}
                uploadedFiles={files}
            />
        </div>
    );
}
