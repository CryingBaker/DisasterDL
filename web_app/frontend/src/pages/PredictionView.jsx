import { useState, useRef } from 'react';
import axios from 'axios';
import { Upload, Loader2, AlertCircle, ImageIcon, FileJson, CheckCircle2, FileText } from 'lucide-react';
import MissionBriefing from '../components/MissionBriefing';

const API_BASE = 'http://localhost:5000/api/bd_predict';

const DAMAGE_COLORS = {
    'no-damage':    { hex: '#00C851', label: 'No Damage' },
    'minor-damage': { hex: '#ffbb33', label: 'Minor Damage' },
    'major-damage': { hex: '#ff8800', label: 'Major Damage' },
    'destroyed':    { hex: '#CC0000', label: 'Destroyed' },
};

export default function PredictionView() {
    const [files, setFiles] = useState({ pre_image: null, post_image: null, pre_json: null, post_json: null });
    const [previews, setPreviews] = useState({ pre: null, post: null });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [isBriefingOpen, setIsBriefingOpen] = useState(false);
    const [results, setResults] = useState(() => {
        const saved = sessionStorage.getItem('last_damage_result');
        return saved ? JSON.parse(saved) : null;
    });

    const gtCanvasRef = useRef(null);
    const predCanvasRef = useRef(null);

    const handleFile = (e, type) => {
        const file = e.target.files[0];
        if (!file) return;

        setFiles(prev => ({ ...prev, [type]: file }));
        if (type.includes('image')) {
            const key = type.startsWith('pre') ? 'pre' : 'post';
            setPreviews(prev => ({ ...prev, [key]: URL.createObjectURL(file) }));
        }
        setResults(null);
    };

    const handlePredict = async () => {
        if (!files.pre_image || !files.post_image || !files.pre_json || !files.post_json) {
            setError('Please upload all 4 required files.');
            return;
        }

        setLoading(true);
        setError(null);
        
        const formData = new FormData();
        Object.entries(files).forEach(([k, v]) => formData.append(k, v));

        try {
            const res = await axios.post(`${API_BASE}/predict`, formData);
            setResults(res.data);
            sessionStorage.setItem('last_damage_result', JSON.stringify(res.data));
        } catch (err) {
            setError(err.response?.data?.error || 'Prediction failed');
        } finally {
            setLoading(false);
        }
    };

    const drawPolygons = (canvas, polygons) => {
        if (!canvas || !polygons) return;
        const ctx = canvas.getContext('2d');
        const { width: cw, height: ch } = canvas;
        ctx.clearRect(0, 0, cw, ch);

        const imgW = results?.image_size?.[0] || 1024;
        const imgH = results?.image_size?.[1] || 1024;
        const sx = cw / imgW;
        const sy = ch / imgH;

        polygons.forEach(p => {
            const coords = p.polygon_coords;
            if (!coords) return;
            ctx.beginPath();
            ctx.moveTo(coords[0][0] * sx, coords[0][1] * sy);
            coords.slice(1).forEach(c => ctx.lineTo(c[0] * sx, c[1] * sy));
            ctx.closePath();
            ctx.strokeStyle = p.color;
            ctx.lineWidth = 2;
            ctx.stroke();
            const hex = p.color.replace('#', '');
            ctx.fillStyle = `rgba(${parseInt(hex.substr(0,2),16)},${parseInt(hex.substr(2,2),16)},${parseInt(hex.substr(4,2),16)},0.2)`;
            ctx.fill();
        });
    };

    return (
        <div className="content-grid" style={{ padding: '2rem' }}>
            {/* ── Left: Simplified Controls ── */}
            <div className="sidebar">
                <div className="glass-panel" style={{ padding: '2rem' }}>
                    <h2 style={{ margin: '0 0 1.5rem', fontSize: '1.2rem', fontWeight: 700 }}>Building Damage</h2>
                    
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        {[
                            { id: 'pre_img', label: 'Pre-Event Image', type: 'pre_image', icon: <ImageIcon size={16}/> },
                            { id: 'post_img', label: 'Post-Event Image', type: 'post_image', icon: <ImageIcon size={16}/> },
                            { id: 'pre_js', label: 'Pre-Event JSON', type: 'pre_json', icon: <FileJson size={16}/> },
                            { id: 'post_js', label: 'Post-Event JSON', type: 'post_json', icon: <FileJson size={16}/> }
                        ].map(field => (
                            <div key={field.id}>
                                <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.5rem', textTransform: 'uppercase' }}>{field.label}</div>
                                <div 
                                    onClick={() => document.getElementById(field.id).click()}
                                    style={{ 
                                        padding: '0.75rem', borderRadius: '8px', border: files[field.type] ? '1px solid #4ade80' : '1px dashed var(--glass-border)', 
                                        background: 'rgba(255,255,255,0.02)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem' 
                                    }}
                                >
                                    <input id={field.id} type="file" hidden onChange={(e) => handleFile(e, field.type)} />
                                    {files[field.type] ? <CheckCircle2 size={16} stroke="#4ade80"/> : field.icon}
                                    <span style={{ fontSize: '0.8rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                        {files[field.type] ? files[field.type].name : 'Select File'}
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>

                    <button 
                        className="btn-primary" 
                        style={{ width: '100%', marginTop: '2rem', padding: '1rem' }}
                        onClick={handlePredict}
                        disabled={loading}
                    >
                        {loading ? <Loader2 className="animate-spin" size={20}/> : 'Analyze Damage'}
                    </button>

                    {error && <div style={{ marginTop: '1rem', color: '#ff4444', fontSize: '0.8rem' }}>⚠️ {error}</div>}
                    
                    {results && (
                        <div style={{ marginTop: '2rem', padding: '1.5rem', background: 'rgba(255,255,255,0.03)', borderRadius: '12px' }}>
                            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Accuracy Score</div>
                            <div style={{ fontSize: '2rem', fontWeight: 900, color: '#4ade80' }}>{results.accuracy}%</div>
                            <button 
                                onClick={() => setIsBriefingOpen(true)}
                                className="btn-primary" 
                                style={{ width: '100%', marginTop: '1.5rem', background: 'rgba(59, 130, 246, 0.1)', border: '1px solid #3b82f6', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.75rem', padding: '0.8rem' }}
                            >
                                <FileText size={18} /> Generate Mission Briefing
                            </button>
                        </div>
                    )}
                </div>
            </div>

            {/* ── Right: Clear Visualization ── */}
            <div className="main-content">
                <div className="glass-panel" style={{ padding: '1.5rem', minHeight: '500px' }}>
                    {!results ? (
                        <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', opacity: 0.3 }}>
                            <Upload size={64} style={{ marginBottom: '1rem' }} />
                            <p>Upload files to start building damage assessment</p>
                        </div>
                    ) : (
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                            <div>
                                <h3 style={{ fontSize: '0.8rem', textTransform: 'uppercase', marginBottom: '1rem' }}>Ground Truth</h3>
                                <div style={{ position: 'relative', background: '#000', borderRadius: '8px', overflow: 'hidden' }}>
                                    <img src={results.post_image} onLoad={() => drawPolygons(gtCanvasRef.current, results.ground_truth_polygons)} style={{ width: '100%' }} />
                                    <canvas ref={gtCanvasRef} width={1024} height={1024} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} />
                                </div>
                            </div>
                            <div>
                                <h3 style={{ fontSize: '0.8rem', textTransform: 'uppercase', marginBottom: '1rem' }}>Model Prediction</h3>
                                <div style={{ position: 'relative', background: '#000', borderRadius: '8px', overflow: 'hidden' }}>
                                    <img src={results.post_image} onLoad={() => drawPolygons(predCanvasRef.current, results.predicted_polygons)} style={{ width: '100%' }} />
                                    <canvas ref={predCanvasRef} width={1024} height={1024} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} />
                                </div>
                            </div>
                        </div>
                    )}

                    {results && (
                        <div style={{ marginTop: '2rem', display: 'flex', justifyContent: 'center', gap: '2rem', flexWrap: 'wrap' }}>
                             {Object.entries(DAMAGE_COLORS).map(([k,v]) => (
                                 <div key={k} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8rem', fontWeight: 600 }}>
                                     <div style={{ width: 12, height: 12, borderRadius: '50%', background: v.hex }} />
                                     {v.label}
                                 </div>
                             ))}
                        </div>
                    )}
                </div>
            </div>

            <MissionBriefing 
                isOpen={isBriefingOpen} 
                onClose={() => setIsBriefingOpen(false)} 
                data={results} 
                type="damage" 
            />
        </div>
    );
}
