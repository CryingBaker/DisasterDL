import { useState, useEffect } from 'react';
import axios from 'axios';
import { Loader2, Activity, Database, ChevronDown, BarChart3, TrendingUp, Globe } from 'lucide-react';
import { MapContainer, TileLayer, ImageOverlay, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

const API_BASE = 'http://localhost:5000/api/fs_dataset';

function FitBounds({ bounds }) {
    const map = useMap();
    useEffect(() => { if (bounds) map.fitBounds(bounds); }, [bounds, map]);
    return null;
}

/* ── Mini SVG line chart ── */
function MiniChart({ data, yKey, color = '#3b82f6', height = 80, label }) {
    if (!data || data.length === 0) return null;
    const vals = data.map(d => d[yKey]).filter(v => v != null);
    if (vals.length === 0) return null;
    const mn = Math.min(...vals), mx = Math.max(...vals);
    const range = mx - mn || 1;
    const w = 280, h = height;
    const pts = vals.map((v, i) => `${(i / Math.max(vals.length - 1, 1)) * w},${h - ((v - mn) / range) * (h - 8) - 4}`).join(' ');
    return (
        <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', marginBottom: '2px' }}>
                <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>{label}</span>
                <span style={{ fontWeight: 700, color }}>{vals[vals.length - 1]?.toFixed(4)}</span>
            </div>
            <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{ borderRadius: '4px', background: 'rgba(0,0,0,0.15)' }}>
                <polyline points={pts} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" />
            </svg>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.58rem', color: 'rgba(255,255,255,0.3)', marginTop: '1px' }}>
                <span>Ep 1</span><span>Ep {vals.length}</span>
            </div>
        </div>
    );
}

const MetricBar = ({ label, value, max = 1, color = '#3b82f6' }) => (
    <div style={{ marginBottom: '0.4rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', marginBottom: '2px' }}>
            <span style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>{label}</span>
            <span style={{ fontWeight: 700, color }}>{typeof value === 'number' ? value.toFixed(4) : '—'}</span>
        </div>
        <div style={{ height: 4, background: 'rgba(255,255,255,0.06)', borderRadius: 2, overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${Math.min((value / max) * 100, 100)}%`, background: color, borderRadius: 2, transition: 'width 0.4s ease' }} />
        </div>
    </div>
);

/* ── Split color dot ── */
const SplitDot = ({ split, size = 8 }) => {
    const c = { train: '#3b82f6', val: '#ca8a04', test: '#a855f7' }[split] || '#666';
    return <span style={{ width: size, height: size, borderRadius: '50%', background: c, display: 'inline-block', flexShrink: 0 }} />;
};

export default function FSDatasetView() {
    const [dataset, setDataset] = useState([]);
    const [loading, setLoading] = useState(true);
    const [filterSplit, setFilterSplit] = useState('all');
    const [filterRegion, setFilterRegion] = useState('all');
    const [selectedItem, setSelectedItem] = useState(null);
    const [itemDetails, setItemDetails] = useState(null);
    const [itemLoading, setItemLoading] = useState(false);
    const [stats, setStats] = useState({});
    const [models, setModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState('');
    const [modelInfo, setModelInfo] = useState([]);
    const [trainingCurves, setTrainingCurves] = useState({});

    useEffect(() => {
        axios.get(`${API_BASE}/models`).then(r => { setModels(r.data.models || []); setSelectedModel(r.data.default || ''); setModelInfo(r.data.model_info || []); }).catch(console.error);
        axios.get(`${API_BASE}/list`).then(r => { setDataset(r.data.data); setStats(r.data.stats || {}); }).catch(console.error).finally(() => setLoading(false));
        axios.get(`${API_BASE}/training_curves`).then(r => setTrainingCurves(r.data || {})).catch(console.error);
    }, []);

    const fetchDetails = async (item, modelOverride) => {
        const m = modelOverride || selectedModel;
        if (!modelOverride && selectedItem?.uid === item.uid) return;
        setSelectedItem(item); setItemDetails(null); setItemLoading(true);
        try { const r = await axios.get(`${API_BASE}/image/${item.uid}`, { params: { model: m } }); setItemDetails(r.data); }
        catch (e) { console.error(e); } finally { setItemLoading(false); }
    };
    const handleModelChange = (nm) => { setSelectedModel(nm); if (selectedItem) fetchDetails(selectedItem, nm); };

    const filteredDataset = dataset.filter(i =>
        (filterSplit === 'all' || i.split === filterSplit) &&
        (filterRegion === 'all' || i.region === filterRegion)
    );

    const cmi = modelInfo.find(m => m.name === selectedModel);
    const curves = trainingCurves[selectedModel] || [];
    const regions = [...new Set(dataset.map(d => d.region))].sort();
    const rb = stats.region_breakdown || {};

    const SC = { train: { bg: 'rgba(37,99,235,0.12)', fg: '#3b82f6' }, val: { bg: 'rgba(234,179,8,0.12)', fg: '#ca8a04' }, test: { bg: 'rgba(124,58,237,0.12)', fg: '#a855f7' } };
    const LC = { hand: { bg: 'rgba(34,197,94,0.12)', fg: '#16a34a' }, both: { bg: 'rgba(34,197,94,0.12)', fg: '#16a34a' }, weak: { bg: 'rgba(234,179,8,0.12)', fg: '#ca8a04' } };
    const Tag = ({ text, colors }) => <span style={{ background: colors?.bg, color: colors?.fg, padding: '1px 7px', borderRadius: '4px', fontSize: '0.65rem', fontWeight: 700, textTransform: 'uppercase' }}>{text}</span>;

    const RawImg = ({ src, label }) => (
        <div className="glass-panel" style={{ padding: '0.65rem', display: 'flex', flexDirection: 'column', minWidth: 0 }}>
            <div style={{ width: '100%', aspectRatio: '1/1', background: '#0a0a0a', borderRadius: '6px', overflow: 'hidden', position: 'relative' }}>
                {itemLoading ? <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}><Loader2 size={18} className="animate-spin" style={{ color: 'var(--accent-primary)' }} /></div>
                : src ? <img src={src} alt={label} style={{ width: '100%', height: '100%', objectFit: 'contain', imageRendering: 'pixelated' }} />
                : <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'rgba(255,255,255,0.15)', fontSize: '0.75rem' }}>N/A</div>}
            </div>
            <div style={{ textAlign: 'center', marginTop: '0.45rem', fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</div>
        </div>
    );

    const bounds = itemDetails?.bounds;
    const ps = itemDetails?.pred_stats;

    return (
        <div style={{ display: 'flex', gap: '1.5rem', height: '100%', padding: '1.5rem', boxSizing: 'border-box', overflow: 'hidden' }}>
            {/* SIDEBAR */}
            <div className="glass-panel" style={{ width: '270px', display: 'flex', flexDirection: 'column', flexShrink: 0, height: 'calc(100vh - 120px)' }}>
                <div style={{ padding: '1.1rem', borderBottom: '1px solid var(--glass-border)' }}>
                    <h2 style={{ margin: '0 0 0.85rem', fontSize: '1rem', fontWeight: 700 }}>Flood Segmentation</h2>
                    {models.length > 0 && (
                        <div style={{ marginBottom: '0.85rem' }}>
                            <div style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.35rem' }}>Model</div>
                            <div style={{ position: 'relative' }}>
                                <select id="fs-model-selector" value={selectedModel} onChange={e => handleModelChange(e.target.value)}
                                    style={{ width: '100%', padding: '0.45rem 2rem 0.45rem 0.6rem', borderRadius: '7px', border: '1px solid var(--glass-border)', background: 'rgba(255,255,255,0.04)', color: 'var(--text-primary)', fontSize: '0.75rem', fontWeight: 600, cursor: 'pointer', appearance: 'none', WebkitAppearance: 'none', outline: 'none' }}>
                                    {models.map(m => <option key={m} value={m} style={{ background: '#1a1a2e', color: '#e0e0e0' }}>{m}</option>)}
                                </select>
                                <ChevronDown size={14} style={{ position: 'absolute', right: '0.5rem', top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', color: 'var(--text-secondary)' }} />
                            </div>
                            {cmi && (
                                <div style={{ marginTop: '0.45rem', padding: '0.5rem 0.6rem', background: 'rgba(255,255,255,0.025)', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.04)' }}>
                                    <div style={{ fontSize: '0.65rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '0.3rem', display: 'flex', alignItems: 'center', gap: '4px' }}><BarChart3 size={10} /> Model Metrics</div>
                                    <MetricBar label="Val Flood IoU" value={cmi.val_flood_iou} color="#3b82f6" />
                                    <MetricBar label="Test Flood IoU" value={cmi.test_flood_iou} color="#a855f7" />
                                    <MetricBar label="Val Mean IoU" value={cmi.val_mean_iou} color="#8b5cf6" />
                                    <div style={{ fontSize: '0.62rem', color: 'rgba(255,255,255,0.3)', marginTop: '0.2rem' }}>{cmi.arch} · {cmi.in_channels}ch</div>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Region filter */}
                    <div style={{ marginBottom: '0.5rem' }}>
                        <div style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.35rem' }}>Region</div>
                        <div style={{ position: 'relative' }}>
                            <select value={filterRegion} onChange={e => setFilterRegion(e.target.value)}
                                style={{ width: '100%', padding: '0.4rem 2rem 0.4rem 0.6rem', borderRadius: '7px', border: '1px solid var(--glass-border)', background: 'rgba(255,255,255,0.04)', color: 'var(--text-primary)', fontSize: '0.72rem', fontWeight: 500, cursor: 'pointer', appearance: 'none', WebkitAppearance: 'none', outline: 'none' }}>
                                <option value="all" style={{ background: '#1a1a2e', color: '#e0e0e0' }}>All Regions</option>
                                {regions.map(r => <option key={r} value={r} style={{ background: '#1a1a2e', color: '#e0e0e0' }}>{r} ({rb[r]?.total || 0})</option>)}
                            </select>
                            <ChevronDown size={14} style={{ position: 'absolute', right: '0.5rem', top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', color: 'var(--text-secondary)' }} />
                        </div>
                    </div>

                    <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap' }}>
                        {['all', 'train', 'val', 'test'].map(s => (
                            <button key={s} onClick={() => setFilterSplit(s)} className={filterSplit === s ? 'btn-primary' : 'btn-secondary'}
                                style={{ flex: '1 1 40%', padding: '0.35rem', fontSize: '0.72rem', textTransform: 'capitalize' }}>{s}</button>
                        ))}
                    </div>
                    <div style={{ marginTop: '0.6rem', fontSize: '0.72rem', color: 'var(--text-secondary)' }}>{filteredDataset.length} / {dataset.length} tiles</div>
                </div>
                <div style={{ flex: 1, overflowY: 'auto', padding: '0.65rem', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                    {loading ? <div style={{ textAlign: 'center', padding: '2rem' }}><Loader2 className="animate-spin" style={{ color: 'var(--accent-primary)' }} /></div>
                    : filteredDataset.map(item => {
                        const active = selectedItem?.uid === item.uid;
                        return (
                            <div key={item.uid} onClick={() => fetchDetails(item)} style={{ padding: '0.6rem 0.7rem', borderRadius: '7px', cursor: 'pointer', background: active ? 'var(--bg-secondary)' : 'rgba(255,255,255,0.025)', border: `1px solid ${active ? 'var(--accent-primary)' : 'var(--glass-border)'}`, transition: 'all 0.12s' }}>
                                <div style={{ fontSize: '0.75rem', fontFamily: 'monospace', whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden', fontWeight: 600, marginBottom: '4px' }}>{item.uid}</div>
                                <div style={{ display: 'flex', gap: '3px', flexWrap: 'wrap' }}><Tag text={item.split} colors={SC[item.split]} /><Tag text={item.label_quality} colors={LC[item.set_type]} /></div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* MAIN */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '1rem', overflowY: 'auto', minWidth: 0 }}>
                {!selectedItem ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        <div className="glass-panel" style={{ padding: '1.5rem' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.25rem' }}>
                                <Activity size={26} style={{ color: 'var(--accent-primary)' }} />
                                <h2 style={{ margin: 0, fontSize: '1.4rem', fontWeight: 800 }}>Flood Model Dashboard</h2>
                            </div>

                            {/* ── Region breakdown table ── */}
                            <div style={{ overflowX: 'auto' }}>
                                <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: 0, fontSize: '0.78rem' }}>
                                    <thead>
                                        <tr>
                                            {['Region', 'Total', 'Train', 'Val', 'Test', 'Hand', 'Weak'].map(h => (
                                                <th key={h} style={{ padding: '0.5rem 0.65rem', textAlign: h === 'Region' ? 'left' : 'right', fontWeight: 700, color: 'var(--text-secondary)', borderBottom: '2px solid rgba(255,255,255,0.08)', textTransform: 'uppercase', fontSize: '0.65rem', letterSpacing: '0.05em' }}>{h}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {Object.entries(rb).map(([region, d], i) => (
                                            <tr key={region} onClick={() => setFilterRegion(region === filterRegion ? 'all' : region)} style={{ cursor: 'pointer', background: region === filterRegion ? 'rgba(59,130,246,0.08)' : i % 2 === 0 ? 'rgba(255,255,255,0.015)' : 'transparent', transition: 'background 0.12s' }}>
                                                <td style={{ padding: '0.45rem 0.65rem', fontWeight: 600, borderBottom: '1px solid rgba(255,255,255,0.04)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                                    <Globe size={12} style={{ color: 'var(--text-secondary)', flexShrink: 0 }} />{region}
                                                </td>
                                                <td style={{ padding: '0.45rem 0.65rem', textAlign: 'right', fontWeight: 700, borderBottom: '1px solid rgba(255,255,255,0.04)' }}>{d.total}</td>
                                                <td style={{ padding: '0.45rem 0.65rem', textAlign: 'right', color: '#3b82f6', fontWeight: 600, borderBottom: '1px solid rgba(255,255,255,0.04)' }}>{d.train}</td>
                                                <td style={{ padding: '0.45rem 0.65rem', textAlign: 'right', color: '#ca8a04', fontWeight: 600, borderBottom: '1px solid rgba(255,255,255,0.04)' }}>{d.val || '—'}</td>
                                                <td style={{ padding: '0.45rem 0.65rem', textAlign: 'right', color: '#a855f7', fontWeight: 600, borderBottom: '1px solid rgba(255,255,255,0.04)' }}>{d.test || '—'}</td>
                                                <td style={{ padding: '0.45rem 0.65rem', textAlign: 'right', color: '#16a34a', fontWeight: 600, borderBottom: '1px solid rgba(255,255,255,0.04)' }}>{d.hand}</td>
                                                <td style={{ padding: '0.45rem 0.65rem', textAlign: 'right', color: '#ca8a04', fontWeight: 600, borderBottom: '1px solid rgba(255,255,255,0.04)' }}>{d.weak}</td>
                                            </tr>
                                        ))}
                                        {/* Totals row */}
                                        <tr style={{ background: 'rgba(255,255,255,0.03)' }}>
                                            <td style={{ padding: '0.55rem 0.65rem', fontWeight: 800, borderTop: '2px solid rgba(255,255,255,0.1)' }}>Total</td>
                                            <td style={{ padding: '0.55rem 0.65rem', textAlign: 'right', fontWeight: 800, borderTop: '2px solid rgba(255,255,255,0.1)' }}>{dataset.length}</td>
                                            <td style={{ padding: '0.55rem 0.65rem', textAlign: 'right', fontWeight: 800, color: '#3b82f6', borderTop: '2px solid rgba(255,255,255,0.1)' }}>{stats.training_tiles}</td>
                                            <td style={{ padding: '0.55rem 0.65rem', textAlign: 'right', fontWeight: 800, color: '#ca8a04', borderTop: '2px solid rgba(255,255,255,0.1)' }}>{stats.val_tiles}</td>
                                            <td style={{ padding: '0.55rem 0.65rem', textAlign: 'right', fontWeight: 800, color: '#a855f7', borderTop: '2px solid rgba(255,255,255,0.1)' }}>{stats.test_tiles}</td>
                                            <td style={{ padding: '0.55rem 0.65rem', textAlign: 'right', fontWeight: 800, color: '#16a34a', borderTop: '2px solid rgba(255,255,255,0.1)' }}>{Object.values(rb).reduce((s, d) => s + d.hand, 0)}</td>
                                            <td style={{ padding: '0.55rem 0.65rem', textAlign: 'right', fontWeight: 800, color: '#ca8a04', borderTop: '2px solid rgba(255,255,255,0.1)' }}>{Object.values(rb).reduce((s, d) => s + d.weak, 0)}</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        {/* Model metrics table + Training curves */}
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                            {/* Model comparison table */}
                            <div className="glass-panel" style={{ padding: '1.25rem' }}>
                                <h3 style={{ fontSize: '0.88rem', fontWeight: 700, margin: '0 0 0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}><BarChart3 size={16} style={{ color: 'var(--accent-primary)' }} /> Model Comparison (Flood IoU)</h3>
                                <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: 0, fontSize: '0.76rem' }}>
                                    <thead>
                                        <tr>
                                            {['Model', 'Val', 'Test', 'Gap'].map(h => (
                                                <th key={h} style={{ padding: '0.45rem 0.55rem', textAlign: h === 'Model' ? 'left' : 'right', fontWeight: 700, color: 'var(--text-secondary)', borderBottom: '2px solid rgba(255,255,255,0.08)', fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{h}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {[...modelInfo].sort((a, b) => b.val_flood_iou - a.val_flood_iou).map((m, i) => {
                                            const gap = m.val_flood_iou - m.test_flood_iou;
                                            const isActive = m.name === selectedModel;
                                            return (
                                                <tr key={m.name} onClick={() => handleModelChange(m.name)} style={{ cursor: 'pointer', background: isActive ? 'rgba(59,130,246,0.08)' : i % 2 === 0 ? 'rgba(255,255,255,0.015)' : 'transparent', transition: 'background 0.12s' }}>
                                                    <td style={{ padding: '0.5rem 0.55rem', fontWeight: 600, borderBottom: '1px solid rgba(255,255,255,0.04)', maxWidth: '140px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m.name.replace(' (Ablation)', '')}</td>
                                                    <td style={{ padding: '0.5rem 0.55rem', textAlign: 'right', fontWeight: 700, color: '#3b82f6', fontFamily: 'monospace', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>{m.val_flood_iou.toFixed(4)}</td>
                                                    <td style={{ padding: '0.5rem 0.55rem', textAlign: 'right', fontWeight: 700, color: '#a855f7', fontFamily: 'monospace', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>{m.test_flood_iou.toFixed(4)}</td>
                                                    <td style={{ padding: '0.5rem 0.55rem', textAlign: 'right', fontWeight: 700, fontFamily: 'monospace', borderBottom: '1px solid rgba(255,255,255,0.04)', color: gap > 0.05 ? '#ef4444' : gap < -0.05 ? '#10b981' : 'var(--text-secondary)' }}>{gap >= 0 ? '+' : ''}{gap.toFixed(3)}</td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>

                            {/* Training Curves */}
                            <div className="glass-panel" style={{ padding: '1.25rem' }}>
                                <h3 style={{ fontSize: '0.88rem', fontWeight: 700, margin: '0 0 0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}><TrendingUp size={16} style={{ color: '#10b981' }} /> Training Curves — {selectedModel?.replace(' (Ablation)', '') || 'Select model'}</h3>
                                {curves.length > 0 ? (
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                                        <MiniChart data={curves} yKey="mean_iou" color="#8b5cf6" label="Val Mean IoU" />
                                        <MiniChart data={curves} yKey="flood_iou" color="#3b82f6" label="Val Flood IoU" />
                                        <MiniChart data={curves} yKey="loss" color="#ef4444" label="Val Loss" />
                                    </div>
                                ) : <div style={{ height: '260px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'rgba(255,255,255,0.25)', fontSize: '0.85rem' }}>No training curves for this model</div>}
                            </div>
                        </div>

                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', gap: '0.75rem', opacity: 0.4, padding: '1rem' }}><Database size={32} strokeWidth={1} /><span style={{ fontWeight: 500 }}>Select a tile from the sidebar to view predictions</span></div>
                    </div>
                ) : (
                    <>
                        <div className="glass-panel" style={{ padding: '0.65rem 1.1rem', display: 'flex', alignItems: 'center', gap: '0.6rem', flexShrink: 0, flexWrap: 'wrap' }}>
                            <button onClick={() => { setSelectedItem(null); setItemDetails(null); }} style={{ background: 'rgba(255,255,255,0.06)', border: 'none', borderRadius: '6px', padding: '4px 10px', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '0.75rem', fontWeight: 600 }}>← Dashboard</button>
                            <h2 style={{ margin: 0, fontSize: '1rem', fontFamily: 'monospace', fontWeight: 700 }}>{selectedItem.uid}</h2>
                            <Tag text={selectedItem.split} colors={SC[selectedItem.split]} /><Tag text={selectedItem.label_quality} colors={LC[selectedItem.set_type]} />
                            {itemDetails?.model_used && <span style={{ marginLeft: 'auto', fontSize: '0.7rem', color: 'var(--text-secondary)', fontWeight: 600, background: 'rgba(99,179,237,0.1)', padding: '2px 8px', borderRadius: '4px' }}>🧠 {itemDetails.model_used}</span>}
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.85rem', flexShrink: 0 }}>
                            {[{ title: 'Ground Truth', overlay: itemDetails?.gt_overlay, color: 'rgba(0,100,255,0.8)', legend: 'Blue = Flood', pctKey: 'gt_flood_pct' }, { title: 'Model Prediction', overlay: itemDetails?.pred_overlay, color: 'rgba(255,60,60,0.85)', legend: 'Red = Predicted', pctKey: 'flood_pct' }].map(m => (
                                <div key={m.title} className="glass-panel" style={{ padding: '0.75rem' }}>
                                    <div style={{ marginBottom: '0.5rem', fontWeight: 700, fontSize: '0.82rem', textTransform: 'uppercase', color: 'var(--text-secondary)', letterSpacing: '0.05em', display: 'flex', alignItems: 'center', gap: '6px' }}><span style={{ width: '10px', height: '10px', borderRadius: '2px', background: m.color, display: 'inline-block' }} />{m.title}</div>
                                    <div style={{ height: '320px', borderRadius: '8px', overflow: 'hidden' }}>
                                        {bounds ? <MapContainer bounds={bounds} style={{ height: '100%', width: '100%' }} zoomControl><TileLayer url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" attribution="© Esri" /><FitBounds bounds={bounds} />{m.overlay && <ImageOverlay url={m.overlay} bounds={bounds} opacity={1} />}</MapContainer>
                                        : <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#111', borderRadius: '8px', color: 'rgba(255,255,255,0.3)' }}>{itemLoading ? <Loader2 className="animate-spin" size={22} style={{ color: 'var(--accent-primary)' }} /> : 'No geo-coordinates'}</div>}
                                    </div>
                                    <div style={{ marginTop: '0.4rem', fontSize: '0.72rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '5px' }}><span style={{ width: '10px', height: '10px', borderRadius: '2px', background: m.color, display: 'inline-block' }} />{m.legend}{ps?.[m.pctKey] != null && <span style={{ marginLeft: 'auto', fontWeight: 700 }}>{ps[m.pctKey]}%</span>}</div>
                                </div>
                            ))}
                        </div>
                        {ps && (
                            <div className="glass-panel" style={{ padding: '1rem 1.25rem', flexShrink: 0 }}>
                                <div style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '5px' }}><BarChart3 size={13} /> Tile Prediction Metrics</div>
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '0.75rem' }}>
                                    {[{ label: 'Mean IoU', value: ps.mean_iou, color: '#8b5cf6' }, { label: 'Flood IoU', value: ps.flood_iou, color: '#3b82f6' }, { label: 'Accuracy', value: ps.accuracy, color: '#f59e0b' }, { label: 'Precision', value: ps.precision, color: '#10b981' }, { label: 'Recall', value: ps.recall, color: '#06b6d4' }, { label: 'F1 Score', value: ps.f1, color: '#ec4899' }].filter(m => m.value != null).map(m => (
                                        <div key={m.label} style={{ background: 'rgba(255,255,255,0.025)', padding: '0.6rem 0.75rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.04)' }}>
                                            <div style={{ fontSize: '0.65rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '0.25rem' }}>{m.label}</div>
                                            <div style={{ fontSize: '1.3rem', fontWeight: 800, color: m.color }}>{m.value.toFixed(4)}</div>
                                            <div style={{ height: 3, background: 'rgba(255,255,255,0.06)', borderRadius: 2, marginTop: '4px', overflow: 'hidden' }}><div style={{ height: '100%', width: `${m.value * 100}%`, background: m.color, borderRadius: 2 }} /></div>
                                        </div>
                                    ))}
                                </div>
                                {ps.gt_flood_pct != null && (
                                    <div style={{ marginTop: '0.6rem', display: 'flex', gap: '1.5rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                                        <span>GT: <strong>{ps.gt_flood_pct}%</strong> ({ps.gt_flood_pixels?.toLocaleString()} px)</span>
                                        <span>Pred: <strong>{ps.flood_pct}%</strong> ({ps.flood_pixels?.toLocaleString()} px)</span>
                                        <span>Area: <strong>{ps.area_km2} km²</strong></span>
                                    </div>
                                )}
                            </div>
                        )}
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: '0.85rem', flexShrink: 0 }}>
                            <RawImg src={itemDetails?.pre_s1_image} label="Pre-Event S1" /><RawImg src={itemDetails?.pre_s2_image} label="Pre-Event S2" /><RawImg src={itemDetails?.post_s1_image} label="Post-Event S1" /><RawImg src={itemDetails?.post_s2_image} label="Post-Event S2" />
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}
