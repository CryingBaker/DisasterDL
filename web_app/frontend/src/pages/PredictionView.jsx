import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Upload, Loader2, AlertCircle, FileJson, ImageIcon, BarChart3 } from 'lucide-react';

const API_BASE = 'http://localhost:5000/api/bd_predict';

const DAMAGE_COLORS = {
    'no-damage':    { hex: '#00C851', label: 'No Damage' },
    'minor-damage': { hex: '#ffbb33', label: 'Minor Damage' },
    'major-damage': { hex: '#ff8800', label: 'Major Damage' },
    'destroyed':    { hex: '#CC0000', label: 'Destroyed' },
};

const BREAKDOWN_COLORS = {
    'No Damage':    '#00C851',
    'Minor Damage': '#ffbb33',
    'Major Damage': '#ff8800',
    'Destroyed':    '#CC0000',
};

export default function PredictionView() {
    const [preImage, setPreImage] = useState(null);
    const [postImage, setPostImage] = useState(null);
    const [preJson, setPreJson] = useState(null);
    const [postJson, setPostJson] = useState(null);

    const [prePreview, setPrePreview] = useState(null);
    const [postPreview, setPostPreview] = useState(null);

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [results, setResults] = useState(null);

    const gtCanvasRef = useRef(null);
    const predCanvasRef = useRef(null);

    // ── File handlers ─────────────────────────────────────────
    const handleFile = (e, type) => {
        const file = e.dataTransfer ? e.dataTransfer.files[0] : e.target.files[0];
        if (!file) return;

        if (type === 'pre_image') {
            setPreImage(file);
            setPrePreview(URL.createObjectURL(file));
        } else if (type === 'post_image') {
            setPostImage(file);
            setPostPreview(URL.createObjectURL(file));
        } else if (type === 'pre_json') {
            setPreJson(file);
        } else if (type === 'post_json') {
            setPostJson(file);
        }
    };

    // ── Prediction ────────────────────────────────────────────
    const handlePredict = async () => {
        if (!preImage || !postImage || !preJson || !postJson) {
            setError('Please upload all four files: Pre/Post images and Pre/Post JSON labels.');
            return;
        }

        setLoading(true);
        setError(null);
        setResults(null);

        const formData = new FormData();
        formData.append('pre_image', preImage);
        formData.append('post_image', postImage);
        formData.append('pre_json', preJson);
        formData.append('post_json', postJson);

        try {
            const res = await axios.post(`${API_BASE}/predict`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            setResults(res.data);
        } catch (err) {
            console.error('Prediction failed', err);
            setError(err.response?.data?.error || 'Prediction failed — is the backend running?');
        } finally {
            setLoading(false);
        }
    };

    // ── Polygon drawing ───────────────────────────────────────
    const drawPolygons = (canvas, polygons, label) => {
        if (!canvas || !polygons || polygons.length === 0) return;
        const ctx = canvas.getContext('2d');
        const cw = canvas.width;
        const ch = canvas.height;
        ctx.clearRect(0, 0, cw, ch);

        const imgW = results?.image_size?.[0] || 1024;
        const imgH = results?.image_size?.[1] || 1024;
        const scaleX = cw / imgW;
        const scaleY = ch / imgH;

        polygons.forEach(p => {
            const coords = p.polygon_coords;
            if (!coords || coords.length < 3) return;

            ctx.beginPath();
            ctx.moveTo(coords[0][0] * scaleX, coords[0][1] * scaleY);
            for (let i = 1; i < coords.length; i++) {
                ctx.lineTo(coords[i][0] * scaleX, coords[i][1] * scaleY);
            }
            ctx.closePath();
            ctx.strokeStyle = p.color;
            ctx.lineWidth = 2;
            ctx.stroke();

            const hex = (p.color || '#888888').replace('#', '');
            if (hex.length === 6) {
                const r = parseInt(hex.substring(0, 2), 16);
                const g = parseInt(hex.substring(2, 4), 16);
                const b = parseInt(hex.substring(4, 6), 16);
                ctx.fillStyle = `rgba(${r},${g},${b},0.25)`;
                ctx.fill();
            }
        });
    };

    const handleGtImageLoad = () => {
        drawPolygons(gtCanvasRef.current, results?.ground_truth_polygons, 'GT');
    };

    const handlePredImageLoad = () => {
        drawPolygons(predCanvasRef.current, results?.predicted_polygons, 'Pred');
    };

    useEffect(() => {
        if (!results) return;
        const t = setTimeout(() => {
            drawPolygons(gtCanvasRef.current, results.ground_truth_polygons, 'GT-effect');
            if (results.predicted_polygons)
                drawPolygons(predCanvasRef.current, results.predicted_polygons, 'Pred-effect');
        }, 200);
        return () => clearTimeout(t);
    }, [results]);

    // ── Count classes ─────────────────────────────────────────
    const countClasses = (polys) => {
        if (!polys) return {};
        const counts = {};
        polys.forEach(p => {
            counts[p.damage_class] = (counts[p.damage_class] || 0) + 1;
        });
        return counts;
    };

    // ── Upload zone component ─────────────────────────────────
    const UploadZone = ({ id, label, icon, accept, file, type, isImage, preview }) => (
        <div style={{ marginBottom: '1rem' }}>
            <label style={{
                display: 'flex', alignItems: 'center', gap: '0.4rem',
                marginBottom: '0.4rem', fontSize: '0.8rem',
                color: 'var(--text-secondary)', fontWeight: 600,
                textTransform: 'uppercase', letterSpacing: '0.03em'
            }}>
                {icon} {label}
            </label>
            <div
                onDrop={(e) => { e.preventDefault(); handleFile(e, type); }}
                onDragOver={(e) => e.preventDefault()}
                onClick={() => document.getElementById(id).click()}
                style={{
                    border: file ? '2px solid var(--accent-primary)' : '2px dashed var(--glass-border)',
                    borderRadius: '10px', padding: '0.75rem', textAlign: 'center',
                    background: isImage && preview
                        ? `url(${preview}) center/cover`
                        : file
                            ? 'rgba(37, 99, 235, 0.06)'
                            : 'rgba(255,255,255,0.03)',
                    minHeight: isImage ? '90px' : '44px',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    cursor: 'pointer', transition: 'all 0.2s',
                    color: isImage && preview ? 'transparent' : 'var(--text-secondary)',
                    position: 'relative'
                }}
            >
                <input
                    id={id}
                    type="file"
                    hidden
                    accept={accept}
                    onChange={(e) => handleFile(e, type)}
                />
                {file ? (
                    isImage && preview ? null : (
                        <span style={{
                            fontSize: '0.78rem', fontWeight: 600,
                            color: 'var(--accent-primary)',
                            maxWidth: '100%', overflow: 'hidden',
                            textOverflow: 'ellipsis', whiteSpace: 'nowrap'
                        }}>
                            ✓ {file.name}
                        </span>
                    )
                ) : (
                    <span style={{ fontSize: '0.8rem', opacity: 0.6 }}>
                        {isImage ? 'Drop image or click' : 'Drop JSON or click'}
                    </span>
                )}
            </div>
        </div>
    );

    const allFilesReady = preImage && postImage && preJson && postJson;

    return (
        <div style={{ display: 'flex', gap: '1.5rem', height: '100%', padding: '1.5rem', boxSizing: 'border-box', overflowY: 'hidden' }}>

            {/* ─── LEFT SIDEBAR ───────────────────────────────── */}
            <div className="glass-panel" style={{
                width: '300px', display: 'flex', flexDirection: 'column',
                flexShrink: 0, height: 'calc(100vh - 120px)'
            }}>
                <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--glass-border)' }}>
                    <h2 style={{
                        marginTop: 0, marginBottom: '1.25rem', fontSize: '1.15rem',
                        fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem'
                    }}>
                        <Upload size={20} className="text-accent" /> Upload &amp; Predict
                    </h2>

                    {/* Image uploads */}
                    <UploadZone
                        id="pre-upload" label="Pre-Disaster Image"
                        icon={<ImageIcon size={13} />}
                        accept="image/*" file={preImage} type="pre_image"
                        isImage preview={prePreview}
                    />
                    <UploadZone
                        id="post-upload" label="Post-Disaster Image"
                        icon={<ImageIcon size={13} />}
                        accept="image/*" file={postImage} type="post_image"
                        isImage preview={postPreview}
                    />

                    {/* JSON uploads */}
                    <UploadZone
                        id="pre-json-upload" label="Pre-Disaster JSON"
                        icon={<FileJson size={13} />}
                        accept=".json,application/json" file={preJson}
                        type="pre_json"
                    />
                    <UploadZone
                        id="post-json-upload" label="Post-Disaster JSON"
                        icon={<FileJson size={13} />}
                        accept=".json,application/json" file={postJson}
                        type="post_json"
                    />

                    <button
                        className="btn-primary"
                        style={{
                            width: '100%', display: 'flex', justifyContent: 'center',
                            alignItems: 'center', gap: '0.5rem', marginTop: '0.25rem',
                            opacity: allFilesReady ? 1 : 0.5
                        }}
                        onClick={handlePredict}
                        disabled={loading || !allFilesReady}
                    >
                        {loading ? (
                            <><Loader2 size={18} className="animate-spin" /> Processing...</>
                        ) : (
                            'Run Prediction'
                        )}
                    </button>

                    {error && (
                        <div style={{
                            marginTop: '0.75rem', padding: '0.75rem',
                            background: 'rgba(239, 68, 68, 0.08)',
                            border: '1px solid rgba(239, 68, 68, 0.2)',
                            borderRadius: '8px', color: '#ef4444', fontSize: '0.8rem',
                            display: 'flex', alignItems: 'flex-start', gap: '0.5rem'
                        }}>
                            <AlertCircle size={14} style={{ marginTop: '2px', flexShrink: 0 }} />
                            <span>{error}</span>
                        </div>
                    )}
                </div>

                {/* ── Results Summary ──────────────────────────── */}
                {results && (
                    <div style={{ flex: 1, overflowY: 'auto', padding: '1.25rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                            <BarChart3 size={18} className="text-accent" />
                            <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 700 }}>Analysis</h3>
                        </div>

                        {/* Total buildings */}
                        <div style={{
                            background: 'rgba(255,255,255,0.03)', padding: '1rem',
                            borderRadius: '10px', marginBottom: '1rem',
                            border: '1px solid rgba(255,255,255,0.05)'
                        }}>
                            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '0.25rem' }}>
                                Buildings Analyzed
                            </div>
                            <div style={{ fontSize: '2rem', fontWeight: 900, color: 'var(--primary)' }}>
                                {results.total_buildings}
                            </div>
                        </div>

                        {/* Accuracy */}
                        {results.accuracy !== null && (
                            <div style={{
                                background: 'rgba(255,255,255,0.03)', padding: '1rem',
                                borderRadius: '10px', marginBottom: '1rem',
                                border: '1px solid rgba(255,255,255,0.05)'
                            }}>
                                <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '0.25rem' }}>
                                    Accuracy (vs Ground Truth)
                                </div>
                                <div style={{ fontSize: '2rem', fontWeight: 900, color: 'hsl(142, 71%, 45%)' }}>
                                    {results.accuracy}%
                                </div>
                            </div>
                        )}

                        {/* Per-class breakdown by building count */}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                            {Object.entries(results.breakdown || {}).map(([className, data]) => (
                                <div key={className}>
                                    <div style={{
                                        display: 'flex', justifyContent: 'space-between',
                                        marginBottom: '4px', fontSize: '0.82rem'
                                    }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                            <div style={{
                                                width: '10px', height: '10px', borderRadius: '50%',
                                                background: BREAKDOWN_COLORS[className] || '#888'
                                            }} />
                                            <span style={{ fontWeight: 500 }}>{className}</span>
                                        </div>
                                        <span style={{ fontWeight: 700, fontFamily: 'monospace' }}>
                                            {data.count} <span style={{ fontWeight: 400, color: 'var(--text-secondary)' }}>({data.percentage}%)</span>
                                        </span>
                                    </div>
                                    <div style={{
                                        height: '5px', background: 'rgba(0,0,0,0.08)',
                                        borderRadius: '3px', overflow: 'hidden'
                                    }}>
                                        <div style={{
                                            height: '100%',
                                            width: `${data.percentage}%`,
                                            background: BREAKDOWN_COLORS[className] || '#888',
                                            borderRadius: '3px',
                                            transition: 'width 0.5s ease'
                                        }} />
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Class count comparison */}
                        <div style={{ marginTop: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                            <div style={{
                                fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)',
                                textTransform: 'uppercase', marginBottom: '0.25rem'
                            }}>GT vs Prediction</div>
                            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                                {Object.entries(countClasses(results.ground_truth_polygons)).map(([cls, count]) =>
                                    count > 0 && (
                                        <div key={`gt-${cls}`} style={{
                                            background: 'rgba(255,255,255,0.05)', padding: '3px 8px',
                                            borderRadius: '4px', fontSize: '0.72rem', fontWeight: 600,
                                            border: `1px solid ${DAMAGE_COLORS[cls]?.hex || '#888'}40`
                                        }}>
                                            GT {DAMAGE_COLORS[cls]?.label || cls}: {count}
                                        </div>
                                    )
                                )}
                            </div>
                            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                                {Object.entries(countClasses(results.predicted_polygons)).map(([cls, count]) =>
                                    count > 0 && (
                                        <div key={`pred-${cls}`} style={{
                                            background: 'rgba(255,255,255,0.05)', padding: '3px 8px',
                                            borderRadius: '4px', fontSize: '0.72rem', fontWeight: 600,
                                            border: `1px solid ${DAMAGE_COLORS[cls]?.hex || '#888'}40`
                                        }}>
                                            Pred {DAMAGE_COLORS[cls]?.label || cls}: {count}
                                        </div>
                                    )
                                )}
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* ─── RIGHT MAIN AREA ────────────────────────────── */}
            <div className="main-content" style={{
                flex: 1, padding: '1.5rem', display: 'flex', flexDirection: 'column',
                gap: '1.5rem', overflowY: 'auto', background: 'var(--bg-primary)'
            }}>
                {!results ? (
                    <div style={{
                        flex: 1, display: 'flex', flexDirection: 'column',
                        alignItems: 'center', justifyContent: 'center',
                        gap: '1.5rem', color: 'var(--text-secondary)', opacity: 0.5
                    }}>
                        <Upload size={56} strokeWidth={1} />
                        <div style={{ textAlign: 'center' }}>
                            <p style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.5rem' }}>
                                Upload an image pair to analyze building damage
                            </p>
                            <p style={{ fontSize: '0.9rem', maxWidth: '420px' }}>
                                Provide pre &amp; post disaster images along with their
                                corresponding xBD JSON label files containing building polygons.
                            </p>
                        </div>
                    </div>
                ) : (
                    <>
                        {/* ── Three-panel image row ────────────── */}
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1fr)',
                            gap: '1rem', flexShrink: 0
                        }}>
                            {/* Pre-Event */}
                            <div className="glass-panel" style={{ padding: '0.75rem', display: 'flex', flexDirection: 'column' }}>
                                <div style={{
                                    width: '100%', aspectRatio: '1/1',
                                    background: '#111', borderRadius: '8px',
                                    position: 'relative', overflow: 'hidden'
                                }}>
                                    <img
                                        src={results.pre_image}
                                        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                                        alt="pre"
                                    />
                                </div>
                                <div style={{
                                    textAlign: 'center', marginTop: '0.75rem',
                                    fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.9rem'
                                }}>Pre-Event</div>
                            </div>

                            {/* Ground Truth */}
                            <div className="glass-panel" style={{ padding: '0.75rem', display: 'flex', flexDirection: 'column' }}>
                                <div style={{
                                    width: '100%', aspectRatio: '1/1',
                                    background: '#111', borderRadius: '8px',
                                    position: 'relative', overflow: 'hidden'
                                }}>
                                    <img
                                        src={results.post_image}
                                        onLoad={handleGtImageLoad}
                                        style={{
                                            width: '100%', height: '100%', objectFit: 'contain',
                                            position: 'absolute', inset: 0
                                        }}
                                        alt="post gt"
                                    />
                                    <canvas
                                        ref={gtCanvasRef}
                                        width={results.image_size?.[0] || 1024}
                                        height={results.image_size?.[1] || 1024}
                                        style={{
                                            position: 'absolute', top: 0, left: 0,
                                            width: '100%', height: '100%', zIndex: 10
                                        }}
                                    />
                                </div>
                                <div style={{
                                    textAlign: 'center', marginTop: '0.75rem',
                                    fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.9rem'
                                }}>Ground Truth</div>
                            </div>

                            {/* Prediction */}
                            <div className="glass-panel" style={{ padding: '0.75rem', display: 'flex', flexDirection: 'column' }}>
                                <div style={{
                                    width: '100%', aspectRatio: '1/1',
                                    background: '#111', borderRadius: '8px',
                                    position: 'relative', overflow: 'hidden'
                                }}>
                                    <img
                                        src={results.post_image}
                                        onLoad={handlePredImageLoad}
                                        style={{
                                            width: '100%', height: '100%', objectFit: 'contain',
                                            position: 'absolute', inset: 0
                                        }}
                                        alt="post pred"
                                    />
                                    <canvas
                                        ref={predCanvasRef}
                                        width={results.image_size?.[0] || 1024}
                                        height={results.image_size?.[1] || 1024}
                                        style={{
                                            position: 'absolute', top: 0, left: 0,
                                            width: '100%', height: '100%', zIndex: 10
                                        }}
                                    />
                                </div>
                                <div style={{
                                    textAlign: 'center', marginTop: '0.75rem',
                                    fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.9rem'
                                }}>Prediction</div>
                            </div>
                        </div>

                        {/* ── Legend ────────────────────────────── */}
                        <div className="glass-panel" style={{
                            padding: '0.75rem 1rem', display: 'flex',
                            justifyContent: 'center', gap: '1.5rem',
                            flexWrap: 'wrap', flexShrink: 0
                        }}>
                            {Object.entries(DAMAGE_COLORS).map(([key, { hex, label }]) => (
                                <div key={key} style={{
                                    display: 'flex', alignItems: 'center', gap: '6px',
                                    fontSize: '0.85rem', fontWeight: 600
                                }}>
                                    <div style={{
                                        width: '12px', height: '12px',
                                        borderRadius: '50%', background: hex
                                    }} />
                                    {label}
                                </div>
                            ))}
                        </div>

                        {/* ── Per-building table ───────────────── */}
                        {results.predicted_polygons && results.predicted_polygons.length > 0 && (
                            <div className="glass-panel" style={{ padding: '1rem', flexShrink: 0 }}>
                                <div style={{
                                    fontSize: '0.9rem', fontWeight: 700,
                                    marginBottom: '0.75rem'
                                }}>Per-Building Results</div>
                                <div style={{
                                    maxHeight: '240px', overflowY: 'auto',
                                    borderRadius: '8px', border: '1px solid var(--glass-border)'
                                }}>
                                    <table style={{
                                        width: '100%', borderCollapse: 'collapse',
                                        fontSize: '0.78rem'
                                    }}>
                                        <thead>
                                            <tr style={{ background: 'rgba(255,255,255,0.03)' }}>
                                                <th style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 700, borderBottom: '1px solid var(--glass-border)' }}>#</th>
                                                <th style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 700, borderBottom: '1px solid var(--glass-border)' }}>Predicted</th>
                                                <th style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 700, borderBottom: '1px solid var(--glass-border)' }}>Ground Truth</th>
                                                <th style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 700, borderBottom: '1px solid var(--glass-border)' }}>Match</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {results.predicted_polygons.map((p, i) => {
                                                const isCorrect = p.damage_class === p.ground_truth;
                                                const gtColor = DAMAGE_COLORS[p.ground_truth]?.hex || '#888';
                                                const predColor = DAMAGE_COLORS[p.damage_class]?.hex || '#888';
                                                return (
                                                    <tr key={i} style={{
                                                        borderBottom: '1px solid rgba(255,255,255,0.03)'
                                                    }}>
                                                        <td style={{ padding: '6px 12px', fontFamily: 'monospace', color: 'var(--text-secondary)' }}>{i + 1}</td>
                                                        <td style={{ padding: '6px 12px' }}>
                                                            <span style={{
                                                                display: 'inline-flex', alignItems: 'center', gap: '5px'
                                                            }}>
                                                                <span style={{
                                                                    width: '8px', height: '8px',
                                                                    borderRadius: '50%', background: predColor,
                                                                    display: 'inline-block'
                                                                }} />
                                                                {DAMAGE_COLORS[p.damage_class]?.label || p.damage_class}
                                                            </span>
                                                        </td>
                                                        <td style={{ padding: '6px 12px' }}>
                                                            <span style={{
                                                                display: 'inline-flex', alignItems: 'center', gap: '5px'
                                                            }}>
                                                                <span style={{
                                                                    width: '8px', height: '8px',
                                                                    borderRadius: '50%', background: gtColor,
                                                                    display: 'inline-block'
                                                                }} />
                                                                {DAMAGE_COLORS[p.ground_truth]?.label || p.ground_truth}
                                                            </span>
                                                        </td>
                                                        <td style={{
                                                            padding: '6px 12px', textAlign: 'center',
                                                            fontSize: '1rem'
                                                        }}>
                                                            {isCorrect ? '✓' : '✗'}
                                                        </td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
}
