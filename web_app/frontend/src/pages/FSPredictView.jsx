import { useState, useRef, useEffect } from 'react';
import { MapContainer, TileLayer, ImageOverlay } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import {
    Upload, Loader2, AlertCircle, CheckCircle2, Waves, Satellite, FileText
} from 'lucide-react';
import MissionBriefing from '../components/MissionBriefing';

const PREDICT_API = 'http://localhost:5000/api/fs_predict';

export default function FSPredictView() {
    const [files, setFiles] = useState({ post_s1: null, post_s2: null, pre_s1: null, pre_s2: null });
    const [models, setModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState("Default (ResNet-34)");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [isBriefingOpen, setIsBriefingOpen] = useState(false);
    const [results, setResults] = useState(() => {
        const saved = sessionStorage.getItem('last_flood_result');
        return saved ? JSON.parse(saved) : null;
    });

    useEffect(() => {
        const fetchModels = async () => {
            try {
                const res = await fetch('http://localhost:5000/api/fs_dataset/models');
                const data = await res.json();
                setModels(data.models || []);
            } catch (err) {
                console.error("Failed to load models", err);
            }
        };
        fetchModels();
    }, []);

    const handleFileChange = (e, key) => {
        const selected = e.target.files[0];
        if (selected) {
            setFiles(prev => ({ ...prev, [key]: selected }));
            setResults(null);
            setError(null);
        }
    };

    const handlePredict = async () => {
        if (!files.post_s1 && !files.post_s2) {
            setError('Please upload at least one post-disaster image (SAR or Optical).');
            return;
        }
        setLoading(true);
        setError(null);
        setResults(null);

        const form = new FormData();
        Object.entries(files).forEach(([k, v]) => {
            if (v) form.append(k, v);
        });
        form.append('model', selectedModel);

        try {
            const res = await fetch(`${PREDICT_API}/run`, { method: 'POST', body: form });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Prediction failed');
            setResults(data);
            sessionStorage.setItem('last_flood_result', JSON.stringify(data));
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
            {/* ── Sidebar ── */}
            <div className="sidebar" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <div className="glass-panel" style={{ padding: '2rem' }}>
                    <h2 style={{ marginTop: 0, marginBottom: '0.5rem', fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <Satellite size={24} className="text-accent" /> Flood Prediction
                    </h2>
                    
                    {/* Model Selector */}
                    <div style={{ marginBottom: '1.5rem' }}>
                        <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.5rem', textTransform: 'uppercase' }}>Analysis Model</div>
                        <select 
                            value={selectedModel}
                            onChange={(e) => setSelectedModel(e.target.value)}
                            style={{ width: '100%', padding: '0.75rem', background: 'rgba(255,255,255,0.05)', border: '1px solid var(--glass-border)', borderRadius: '8px', color: 'white' }}
                        >
                            {models.map(m => <option key={m} value={m}>{m}</option>)}
                        </select>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        {[
                            { id: 'post_s1', label: 'Post-Event SAR (required)', icon: <Waves size={16}/> },
                            { id: 'post_s2', label: 'Post-Event Optical (optional)', icon: <Satellite size={16}/> },
                            { id: 'pre_s1', label: 'Pre-Event SAR (optional)', icon: <Waves size={16}/> },
                            { id: 'pre_s2', label: 'Pre-Event Optical (optional)', icon: <Satellite size={16}/> },
                        ].map(field => (
                            <div key={field.id}>
                                <div style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.4rem', textTransform: 'uppercase' }}>{field.label}</div>
                                <div 
                                    onClick={() => document.getElementById(field.id).click()}
                                    style={{ 
                                        padding: '0.75rem', borderRadius: '8px', border: files[field.id] ? '1px solid #4ade80' : '1px dashed var(--glass-border)', 
                                        background: 'rgba(255,255,255,0.02)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem' 
                                    }}
                                >
                                    <input id={field.id} type="file" hidden onChange={(e) => handleFileChange(e, field.id)} />
                                    {files[field.id] ? <CheckCircle2 size={16} stroke="#4ade80"/> : field.icon}
                                    <span style={{ fontSize: '0.8rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                        {files[field.id] ? files[field.id].name : 'Select File'}
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>

                    <button 
                        className="btn-primary" 
                        style={{ width: '100%', marginTop: '2rem', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.75rem', padding: '1rem' }}
                        onClick={handlePredict} 
                        disabled={loading}
                    >
                        {loading ? <><Loader2 size={20} className="animate-spin" /> Processing…</> : <><Waves size={20} /> Start Prediction</>}
                    </button>

                    {error && (
                        <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: '10px', color: '#ef4444', fontSize: '0.85rem', display: 'flex', gap: '0.75rem' }}>
                            <AlertCircle size={18} style={{ flexShrink: 0 }} />
                            <span>{error}</span>
                        </div>
                    )}
                </div>

                {/* Stats Panel */}
                {results && (
                    <div className="glass-panel animate-in" style={{ padding: '1.5rem' }}>
                        <h3 style={{ marginTop: 0, marginBottom: '1.25rem', fontSize: '1rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-secondary)' }}>Analysis Result</h3>
                        <div style={{ marginBottom: '1.5rem' }}>
                            <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '4px' }}>Flooded Area</div>
                            <div style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                                {results.estimated_area_km2.toFixed(3)}
                                <span style={{ fontSize: '1rem', fontWeight: 500, color: 'var(--text-secondary)', marginLeft: '6px' }}>km²</span>
                            </div>
                        </div>
                        {Object.entries(results.breakdown || {}).map(([cls, d]) => (
                            <div key={cls} style={{ marginBottom: '1.25rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', fontSize: '0.9rem' }}>
                                    <span style={{ fontWeight: 600 }}>{cls}</span>
                                    <span style={{ fontWeight: 700, color: cls === 'Flooded' ? '#ff4444' : 'inherit' }}>{d.percentage}%</span>
                                </div>
                                <div style={{ height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                                    <div 
                                        style={{ 
                                            height: '100%', 
                                            width: `${d.percentage}%`, 
                                            background: cls === 'Flooded' ? 'linear-gradient(90deg, #ff4d4d, #f97316)' : 'rgba(255,255,255,0.2)', 
                                            borderRadius: '4px', 
                                            transition: 'width 1s cubic-bezier(0.4, 0, 0.2, 1)' 
                                        }} 
                                    />
                                </div>
                            </div>
                        ))}
                        {results && (
                            <button 
                                onClick={() => setIsBriefingOpen(true)}
                                className="btn-primary" 
                                style={{ width: '100%', marginTop: '1rem', background: 'rgba(59, 130, 246, 0.1)', border: '1px solid #3b82f6', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.75rem', padding: '0.8rem' }}
                            >
                                <FileText size={18} /> Generate Mission Briefing
                            </button>
                        )}
                    </div>
                )}
            </div>

            {/* ── Map Display ── */}
            <div className="main-content" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <div className="glass-panel" style={{ flex: 1, padding: '1rem', display: 'flex', flexDirection: 'column' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                         <h3 style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Visualization</h3>
                         {results && <div style={{ fontSize: '0.75rem', color: '#4ade80', fontWeight: 700 }}>● Analysis Complete</div>}
                    </div>
                    
                    {hasMap ? (
                        <div style={{ flex: 1, borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--glass-border)', background: '#000' }}>
                            <MapContainer bounds={bounds} style={{ height: '100%', width: '100%' }}>
                                <TileLayer url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" attribution="Esri" />
                                <ImageOverlay url={results.pred_overlay} bounds={bounds} opacity={0.8} />
                            </MapContainer>
                        </div>
                    ) : (
                        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1.5rem', borderRadius: '12px', border: '1px dashed rgba(255,255,255,0.1)', background: 'rgba(0,0,0,0.2)' }}>
                            {loading ? (
                                <><Loader2 size={48} className="animate-spin text-accent" /><div style={{ fontSize: '1.1rem', fontWeight: 500 }}>Generating Flood Map...</div></>
                            ) : (
                                <><Satellite size={64} opacity={0.15} /><div style={{ fontSize: '1rem', color: 'var(--text-secondary)', maxWidth: '280px', textAlign: 'center' }}>Upload a SAR image and click "Start Prediction" to view results on the interactive map.</div></>
                            )}
                        </div>
                    )}

                    {results && (
                        <div style={{ display: 'flex', gap: '2rem', marginTop: '1.5rem', padding: '0.5rem' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <div style={{ width: 14, height: 14, borderRadius: '3px', background: 'rgba(255,0,0,0.8)' }} />
                                <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Predicted Flood Zone</span>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <div style={{ width: 14, height: 14, borderRadius: '3px', border: '1px dashed rgba(255,255,255,0.3)' }} />
                                <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Dry / Ground</span>
                            </div>
                        </div>
                    )}
                </div>

                {results?.post_s1_image && (
                    <div className="glass-panel" style={{ height: '240px', padding: '1rem' }}>
                        <h4 style={{ margin: '0 0 1rem', fontSize: '0.8rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Source Preview (SAR)</h4>
                        <img src={results.post_s1_image} alt="Source" style={{ height: '160px', borderRadius: '8px', border: '1px solid var(--glass-border)' }} />
                    </div>
                )}
            </div>

            <MissionBriefing 
                isOpen={isBriefingOpen} 
                onClose={() => setIsBriefingOpen(false)} 
                data={results} 
                type="flood" 
            />
        </div>
    );
}
