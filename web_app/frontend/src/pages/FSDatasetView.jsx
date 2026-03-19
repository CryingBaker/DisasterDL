import { useState, useEffect } from 'react';
import axios from 'axios';
import { Loader2, Activity, Database } from 'lucide-react';
import { MapContainer, TileLayer, ImageOverlay, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

const API_BASE = 'http://localhost:5000/api/fs_dataset';

function FitBounds({ bounds }) {
    const map = useMap();
    useEffect(() => { if (bounds) map.fitBounds(bounds); }, [bounds, map]);
    return null;
}

export default function FSDatasetView() {
    const [dataset, setDataset]       = useState([]);
    const [loading, setLoading]       = useState(true);
    const [filterSplit, setFilterSplit] = useState('all');
    const [selectedItem, setSelectedItem] = useState(null);
    const [itemDetails, setItemDetails]   = useState(null);
    const [itemLoading, setItemLoading]   = useState(false);
    const [stats, setStats] = useState({});

    useEffect(() => {
        axios.get(`${API_BASE}/list`)
            .then(r => { setDataset(r.data.data); setStats(r.data.stats || {}); })
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
        } catch (e) { console.error(e); }
        finally { setItemLoading(false); }
    };

    const filteredDataset = dataset.filter(item =>
        filterSplit === 'all' ? true : item.split === filterSplit
    );

    const SPLITS = ['all', 'train', 'val', 'test'];
    const SPLIT_COLOR = {
        train: { bg: 'rgba(37,99,235,0.12)',  fg: '#3b82f6' },
        val:   { bg: 'rgba(234,179,8,0.12)',  fg: '#ca8a04' },
        test:  { bg: 'rgba(124,58,237,0.12)', fg: '#a855f7' },
    };
    const LABEL_COLOR = {
        hand: { bg: 'rgba(34,197,94,0.12)',  fg: '#16a34a' },
        both: { bg: 'rgba(34,197,94,0.12)',  fg: '#16a34a' },
        weak: { bg: 'rgba(234,179,8,0.12)',  fg: '#ca8a04' },
    };

    const Tag = ({ text, colors }) => (
        <span style={{ background: colors.bg, color: colors.fg, padding: '1px 7px', borderRadius: '4px', fontSize: '0.65rem', fontWeight: 700, textTransform: 'uppercase' }}>{text}</span>
    );

    const RawImg = ({ src, label }) => (
        <div className="glass-panel" style={{ padding: '0.65rem', display: 'flex', flexDirection: 'column', minWidth: 0 }}>
            <div style={{ width: '100%', aspectRatio: '1/1', background: '#0a0a0a', borderRadius: '6px', overflow: 'hidden', position: 'relative' }}>
                {itemLoading ? (
                    <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <Loader2 size={18} className="animate-spin" style={{ color: 'var(--accent-primary)' }} />
                    </div>
                ) : src ? (
                    <img src={src} alt={label} style={{ width: '100%', height: '100%', objectFit: 'contain', imageRendering: 'pixelated' }} />
                ) : (
                    <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'rgba(255,255,255,0.15)', fontSize: '0.75rem' }}>N/A</div>
                )}
            </div>
            <div style={{ textAlign: 'center', marginTop: '0.45rem', fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</div>
        </div>
    );

    const bounds = itemDetails?.bounds;

    return (
        <div style={{ display: 'flex', gap: '1.5rem', height: '100%', padding: '1.5rem', boxSizing: 'border-box', overflow: 'hidden' }}>

            {/* ── SIDEBAR ── */}
            <div className="glass-panel" style={{ width: '250px', display: 'flex', flexDirection: 'column', flexShrink: 0, height: 'calc(100vh - 120px)' }}>
                <div style={{ padding: '1.1rem', borderBottom: '1px solid var(--glass-border)' }}>
                    <h2 style={{ margin: '0 0 0.85rem', fontSize: '1rem', fontWeight: 700 }}>Flood Segmentation</h2>
                    <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap' }}>
                        {SPLITS.map(s => (
                            <button key={s}
                                onClick={() => setFilterSplit(s)}
                                className={filterSplit === s ? 'btn-primary' : 'btn-secondary'}
                                style={{ flex: '1 1 40%', padding: '0.35rem', fontSize: '0.72rem', textTransform: 'capitalize' }}>
                                {s}
                            </button>
                        ))}
                    </div>
                    <div style={{ marginTop: '0.6rem', fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
                        {filteredDataset.length} / {dataset.length} tiles
                    </div>
                </div>

                <div style={{ flex: 1, overflowY: 'auto', padding: '0.65rem', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                    {loading ? (
                        <div style={{ textAlign: 'center', padding: '2rem' }}>
                            <Loader2 className="animate-spin" style={{ color: 'var(--accent-primary)' }} />
                        </div>
                    ) : filteredDataset.map(item => {
                        const sc = SPLIT_COLOR[item.split] || {};
                        const lc = LABEL_COLOR[item.set_type] || LABEL_COLOR.weak;
                        const active = selectedItem?.uid === item.uid;
                        return (
                            <div key={item.uid} onClick={() => fetchDetails(item)} style={{
                                padding: '0.6rem 0.7rem', borderRadius: '7px', cursor: 'pointer',
                                background: active ? 'var(--bg-secondary)' : 'rgba(255,255,255,0.025)',
                                border: `1px solid ${active ? 'var(--accent-primary)' : 'var(--glass-border)'}`,
                                transition: 'all 0.12s',
                            }}>
                                <div style={{ fontSize: '0.75rem', fontFamily: 'monospace', whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden', fontWeight: 600, marginBottom: '4px' }}>{item.uid}</div>
                                <div style={{ display: 'flex', gap: '3px', flexWrap: 'wrap' }}>
                                    <Tag text={item.split} colors={sc} />
                                    <Tag text={item.label_quality} colors={lc} />
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* ── MAIN CONTENT ── */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '1rem', overflowY: 'auto', minWidth: 0 }}>
                {!selectedItem ? (
                    <div className="glass-panel" style={{ flex: 1, padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <Activity size={26} style={{ color: 'var(--accent-primary)' }} />
                            <h2 style={{ margin: 0, fontSize: '1.4rem', fontWeight: 800 }}>Flood Model Overview</h2>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.25rem' }}>
                            {[
                                { label: 'Train', val: stats.training_tiles, color: '#3b82f6' },
                                { label: 'Val',   val: stats.val_tiles,      color: '#ca8a04' },
                                { label: 'Test',  val: stats.test_tiles,     color: '#a855f7' },
                            ].map(c => (
                                <div key={c.label} style={{ background: 'rgba(255,255,255,0.03)', padding: '1.25rem', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.05)' }}>
                                    <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '0.4rem' }}>{c.label} Tiles</div>
                                    <div style={{ fontSize: '2.2rem', fontWeight: 900, color: c.color }}>{c.val ?? '-'}</div>
                                </div>
                            ))}
                        </div>
                        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', flexDirection: 'column', gap: '1rem', opacity: 0.45 }}>
                            <Database size={44} strokeWidth={1} />
                            <p style={{ fontWeight: 500 }}>Select a tile to explore flood segmentations</p>
                        </div>
                    </div>
                ) : (
                    <>
                        {/* Title bar */}
                        <div className="glass-panel" style={{ padding: '0.65rem 1.1rem', display: 'flex', alignItems: 'center', gap: '0.6rem', flexShrink: 0 }}>
                            <h2 style={{ margin: 0, fontSize: '1rem', fontFamily: 'monospace', fontWeight: 700 }}>{selectedItem.uid}</h2>
                            <Tag text={selectedItem.split} colors={SPLIT_COLOR[selectedItem.split] || {}} />
                            <Tag text={selectedItem.label_quality} colors={LABEL_COLOR[selectedItem.set_type] || LABEL_COLOR.weak} />
                        </div>

                        {/* ── SINGLE COMBINED MAP ── */}
                        <div className="glass-panel" style={{ padding: '0.75rem', flexShrink: 0 }}>
                            <div style={{ marginBottom: '0.5rem', fontWeight: 700, fontSize: '0.82rem', textTransform: 'uppercase', color: 'var(--text-secondary)', letterSpacing: '0.05em' }}>
                                Flood Map — Ground Truth + Model Prediction
                            </div>
                            <div style={{ height: '380px', borderRadius: '8px', overflow: 'hidden', position: 'relative' }}>
                                {bounds ? (
                                    <MapContainer bounds={bounds} style={{ height: '100%', width: '100%' }} zoomControl>
                                        {/* Satellite base */}
                                        <TileLayer
                                            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                                            attribution="© Esri"
                                        />
                                        <FitBounds bounds={bounds} />
                                        {/* Ground truth overlay — BLUE */}
                                        {itemDetails?.gt_overlay && (
                                            <ImageOverlay url={itemDetails.gt_overlay} bounds={bounds} opacity={1} />
                                        )}
                                        {/* Model prediction overlay — RED */}
                                        {itemDetails?.pred_overlay && (
                                            <ImageOverlay url={itemDetails.pred_overlay} bounds={bounds} opacity={1} />
                                        )}
                                        {/* Overlap overlay — YELLOW (GT ∩ Prediction) */}
                                        {itemDetails?.overlap_overlay && (
                                            <ImageOverlay url={itemDetails.overlap_overlay} bounds={bounds} opacity={1} />
                                        )}
                                    </MapContainer>
                                ) : (
                                    <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#111', borderRadius: '8px', color: 'rgba(255,255,255,0.3)' }}>
                                        {itemLoading ? <Loader2 className="animate-spin" size={22} style={{ color: 'var(--accent-primary)' }} /> : 'No geo-coordinates in this tile'}
                                    </div>
                                )}
                            </div>

                            {/* Map legend */}
                            <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.65rem', paddingTop: '0.65rem', borderTop: '1px solid var(--glass-border)', flexWrap: 'wrap' }}>
                                {[
                                    { color: 'rgba(0,100,255,0.7)',   label: 'Ground Truth (Flood)' },
                                    { color: 'rgba(255,0,0,0.7)',     label: 'Prediction (Flood)' },
                                    { color: 'rgba(255,220,0,0.8)',   label: 'Overlap (GT ∩ Pred)' },
                                    { color: 'rgba(0,0,0,0)',         label: 'No-Flood / Ignored', border: '1px dashed rgba(255,255,255,0.2)' },
                                ].map(({ color, label, border }) => (
                                    <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.82rem', fontWeight: 600 }}>
                                        <div style={{ width: '14px', height: '14px', borderRadius: '3px', background: color, border: border || 'none' }} />
                                        {label}
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* ── 4 RAW IMAGES ── */}
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: '0.85rem', flexShrink: 0 }}>
                            <RawImg src={itemDetails?.pre_s1_image}  label="Pre-Event S1 (SAR)" />
                            <RawImg src={itemDetails?.pre_s2_image}  label="Pre-Event S2 (RGB)" />
                            <RawImg src={itemDetails?.post_s1_image} label="Post-Event S1 (SAR)" />
                            <RawImg src={itemDetails?.post_s2_image} label="Post-Event S2 (RGB)" />
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}
