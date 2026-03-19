import { useState } from 'react';
import axios from 'axios';
import { Upload, Image as ImageIcon, Map as MapIcon, Loader2, AlertCircle } from 'lucide-react';

const API_BASE = 'http://localhost:5000/api/bd_predict';

const DAMAGE_COLORS = {
    'No Damage': 'rgb(0, 255, 0)',
    'Minor Damage': 'rgb(255, 255, 0)',
    'Major Damage': 'rgb(255, 165, 0)',
    'Destroyed': 'rgb(255, 0, 0)',
};

export default function PredictionView() {
    const [preImage, setPreImage] = useState(null);
    const [postImage, setPostImage] = useState(null);
    const [prePreview, setPrePreview] = useState(null);
    const [postPreview, setPostPreview] = useState(null);
    
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [results, setResults] = useState(null);
    const [showOverlay, setShowOverlay] = useState(true);

    const handleFileDrop = (e, type) => {
        e.preventDefault();
        const file = e.dataTransfer ? e.dataTransfer.files[0] : e.target.files[0];
        if (file) {
            const previewUrl = URL.createObjectURL(file);
            if (type === 'pre') {
                setPreImage(file);
                setPrePreview(previewUrl);
            } else {
                setPostImage(file);
                setPostPreview(previewUrl);
            }
        }
    };

    const handlePredict = async () => {
        if (!preImage || !postImage) {
            setError('Please upload both Pre and Post disaster images.');
            return;
        }

        setLoading(true);
        setError(null);
        setResults(null);

        const formData = new FormData();
        formData.append('pre_image', preImage);
        formData.append('post_image', postImage);

        try {
            const res = await axios.post(`${API_BASE}/predict`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            setResults(res.data);
        } catch (err) {
            console.error('Prediction failed', err);
            setError(err.response?.data?.error || 'Prediction failed to connect to server');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="content-grid">
            {/* Sidebar with Controls */}
            <div className="sidebar" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <div className="glass-panel" style={{ padding: '1.5rem' }}>
                    <h2 style={{ marginTop: 0, marginBottom: '1.5rem', fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Upload size={20} className="text-accent" /> Upload Images
                    </h2>
                    
                    {/* Pre Image Upload */}
                    <div style={{ marginBottom: '1.5rem' }}>
                        <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Pre-Disaster</label>
                        <div 
                            onDrop={(e) => handleFileDrop(e, 'pre')}
                            onDragOver={(e) => e.preventDefault()}
                            style={{ 
                                border: '2px dashed var(--glass-border)', borderRadius: '12px', padding: '1.5rem', textAlign: 'center', 
                                background: prePreview ? `url(${prePreview}) center/cover` : 'rgba(255,255,255,0.5)',
                                color: prePreview ? 'transparent' : 'var(--text-secondary)',
                                minHeight: '120px', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer',
                                transition: 'all 0.2s'
                            }}
                            onClick={() => document.getElementById('pre-upload').click()}
                        >
                            <input id="pre-upload" type="file" hidden accept="image/*" onChange={(e) => handleFileDrop(e, 'pre')} />
                            {!prePreview && <span style={{ fontSize: '0.9rem' }}>Drag & Drop or Click</span>}
                        </div>
                    </div>

                    {/* Post Image Upload */}
                    <div style={{ marginBottom: '1.5rem' }}>
                        <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Post-Disaster</label>
                        <div 
                            onDrop={(e) => handleFileDrop(e, 'post')}
                            onDragOver={(e) => e.preventDefault()}
                            style={{ 
                                border: '2px dashed var(--glass-border)', borderRadius: '12px', padding: '1.5rem', textAlign: 'center', 
                                background: postPreview ? `url(${postPreview}) center/cover` : 'rgba(255,255,255,0.5)',
                                color: postPreview ? 'transparent' : 'var(--text-secondary)',
                                minHeight: '120px', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer',
                                transition: 'all 0.2s'
                            }}
                            onClick={() => document.getElementById('post-upload').click()}
                        >
                            <input id="post-upload" type="file" hidden accept="image/*" onChange={(e) => handleFileDrop(e, 'post')} />
                            {!postPreview && <span style={{ fontSize: '0.9rem' }}>Drag & Drop or Click</span>}
                        </div>
                    </div>

                    <button 
                        className="btn-primary" 
                        style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.5rem' }}
                        onClick={handlePredict}
                        disabled={loading}
                    >
                        {loading ? <><Loader2 size={18} className="animate-spin" /> Processing...</> : 'Run Prediction'}
                    </button>
                    
                    {error && (
                        <div style={{ marginTop: '1rem', padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', borderRadius: '8px', color: '#ef4444', fontSize: '0.9rem', display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
                            <AlertCircle size={16} style={{ marginTop: '2px', flexShrink: 0 }} />
                            <span>{error}</span>
                        </div>
                    )}
                </div>

                {/* Results Summary */}
                {results && (
                    <div className="glass-panel" style={{ padding: '1.5rem' }}>
                        <h2 style={{ marginTop: 0, marginBottom: '1.5rem', fontSize: '1.2rem' }}>Damage Summary</h2>
                        
                        <div style={{ marginBottom: '1.5rem', paddingBottom: '1rem', borderBottom: '1px solid var(--glass-border)' }}>
                            <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', fontWeight: 500, marginBottom: '0.25rem' }}>Estimated Affected Area</div>
                            <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)' }}>{results.estimated_area_km2} <span style={{ fontSize: '1rem', fontWeight: 500, color: 'var(--text-secondary)' }}>km²</span></div>
                        </div>

                        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                            {Object.entries(results.breakdown || {}).filter(([k]) => k !== 'No Damage').map(([className, data]) => (
                                <div key={className}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem', fontSize: '0.9rem' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                            <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: DAMAGE_COLORS[className] }} />
                                            <span style={{ fontWeight: 500 }}>{className}</span>
                                        </div>
                                        <span style={{ fontWeight: 600 }}>{data.percentage}%</span>
                                    </div>
                                    <div style={{ height: '6px', background: 'rgba(0,0,0,0.05)', borderRadius: '3px', overflow: 'hidden' }}>
                                        <div style={{ height: '100%', width: `${data.percentage}%`, background: DAMAGE_COLORS[className], borderRadius: '3px' }} />
                                    </div>
                                </div>
                            ))}
                            
                            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '1rem', cursor: 'pointer', fontSize: '0.9rem', fontWeight: 500, color: 'var(--text-secondary)' }}>
                                <input 
                                    type="checkbox" 
                                    checked={showOverlay} 
                                    onChange={(e) => setShowOverlay(e.target.checked)} 
                                    style={{ accentColor: 'var(--accent-primary)', width: '16px', height: '16px' }}
                                />
                                Show heatmap overlay
                            </label>
                        </div>
                    </div>
                )}
            </div>

            {/* Main Visualizations */}
            <div className="main-content" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 700 }}>Results Panel</h2>
                </div>

                {prePreview && postPreview ? (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                        {/* Pre Image */}
                        <div className="glass-panel" style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                            <h3 style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                                <ImageIcon size={16} /> Pre-Disaster
                            </h3>
                            <div style={{ flex: 1, minHeight: '300px', background: 'var(--bg-primary)', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--glass-border)' }}>
                                <img src={prePreview} alt="Pre-Disaster" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                            </div>
                        </div>

                        {/* Post Image & Overlay */}
                        <div className="glass-panel" style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                            <h3 style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                                <MapIcon size={16} /> Post-Disaster / Prediction
                            </h3>
                            <div style={{ flex: 1, minHeight: '300px', background: 'var(--bg-primary)', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--glass-border)', position: 'relative' }}>
                                <img src={postPreview} alt="Post-Disaster" style={{ width: '100%', height: '100%', objectFit: 'contain', position: 'absolute', inset: 0 }} />
                                {results && results.mask && showOverlay && (
                                    <img 
                                        src={results.mask} 
                                        alt="Damage Mask" 
                                        style={{ width: '100%', height: '100%', objectFit: 'contain', position: 'absolute', inset: 0, opacity: 0.65, mixBlendMode: 'multiply' }} 
                                    />
                                )}
                                {loading && (
                                    <div style={{ position: 'absolute', inset: 0, background: 'rgba(255,255,255,0.7)', backdropFilter: 'blur(4px)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1rem' }}>
                                        <Loader2 size={32} className="animate-spin text-accent" />
                                        <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Analyzing architectural damage...</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="glass-panel" style={{ height: '400px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1rem', color: 'var(--text-secondary)' }}>
                        <ImageIcon size={48} opacity={0.3} />
                        <span style={{ fontSize: '1.1rem', fontWeight: 500 }}>Upload image pair to start prediction</span>
                    </div>
                )}
            </div>
        </div>
    );
}
