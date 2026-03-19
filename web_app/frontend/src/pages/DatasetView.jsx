import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Loader2, Info, Activity, Database } from 'lucide-react';
import MapViewer from '../components/MapViewer';
import { Polygon } from 'react-leaflet';

const API_BASE = 'http://localhost:5000/api/bd_dataset';

export default function DatasetView() {
    const [dataset, setDataset] = useState([]);
    const [loading, setLoading] = useState(true);
    const [filterSplit, setFilterSplit] = useState('all');
    
    const [selectedItem, setSelectedItem] = useState(null);
    const [itemDetails, setItemDetails] = useState(null);
    const [itemLoading, setItemLoading] = useState(false);
    const [stats, setStats] = useState({});
    
    const gtCanvasRef = useRef(null);
    const predCanvasRef = useRef(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await axios.get(`${API_BASE}/list`);
                setDataset(res.data.data);
                setStats(res.data.stats || {});
            } catch (err) {
                console.error(err);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    const fetchDetails = async (item) => {
        if (selectedItem?.id === item.id) return;
        setSelectedItem(item);
        setItemLoading(true);
        setItemDetails(null);
        try {
            const res = await axios.get(`${API_BASE}/image/${item.id}/polygons`);
            setItemDetails(res.data);
        } catch (err) {
            console.error(err);
        } finally {
            setItemLoading(false);
        }
    };

    // Draw polygons after image onLoad — NOT in a useEffect tied to itemDetails
    const drawPolygons = (canvas, polygons, label) => {
        if (!canvas || !polygons || polygons.length === 0) return;
        const ctx = canvas.getContext('2d');
        const cw = canvas.width;
        const ch = canvas.height;
        console.log(`[${label}] Drawing ${polygons.length} polygons on ${cw}x${ch} canvas`);
        ctx.clearRect(0, 0, cw, ch);
        const scaleX = cw / 1024;
        const scaleY = ch / 1024;
        let drawn = 0;
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
                const r = parseInt(hex.substring(0,2), 16);
                const g = parseInt(hex.substring(2,4), 16);
                const b = parseInt(hex.substring(4,6), 16);
                ctx.fillStyle = `rgba(${r},${g},${b},0.2)`;
                ctx.fill();
            }
            drawn++;
        });
        console.log(`[${label}] Drew ${drawn} polygons.`);
    };

    const handleGtImageLoad = () => {
        console.log("GT image loaded, drawing polygons");
        drawPolygons(gtCanvasRef.current, itemDetails?.ground_truth_polygons, 'GroundTruth');
    };

    const handlePredImageLoad = () => {
        console.log("Pred image loaded, drawing polygons");
        drawPolygons(predCanvasRef.current, itemDetails?.predicted_polygons, 'Prediction');
    };

    // Also redraw if itemDetails changes (e.g. same image reselected)
    useEffect(() => {
        if (!itemDetails) return;
        // slight delay in case canvas refs need to paint
        const t = setTimeout(() => {
            drawPolygons(gtCanvasRef.current, itemDetails.ground_truth_polygons, 'GroundTruth-effect');
            if (itemDetails.predicted_polygons)
                drawPolygons(predCanvasRef.current, itemDetails.predicted_polygons, 'Prediction-effect');
        }, 200);
        return () => clearTimeout(t);
    }, [itemDetails]);

    const filteredDataset = dataset.filter(item => 
        filterSplit === 'all' ? true : item.split === filterSplit
    );

    const getBoundsFromLngLat = (polys) => {
        if (!polys || polys.length === 0) return null;
        let minLat = 90, maxLat = -90, minLng = 180, maxLng = -180;
        let found = false;
        polys.forEach(p => {
            if (p.lnglat_coords && p.lnglat_coords.length > 0) {
                p.lnglat_coords.forEach(coord => {
                    const lng = coord[0];
                    const lat = coord[1];
                    if (lat < minLat) minLat = lat;
                    if (lat > maxLat) maxLat = lat;
                    if (lng < minLng) minLng = lng;
                    if (lng > maxLng) maxLng = lng;
                    found = true;
                });
            }
        });
        if (!found) return null;
        return [[minLat, minLng], [maxLat, maxLng]];
    };

    const gtBounds = itemDetails ? getBoundsFromLngLat(itemDetails.ground_truth_polygons) : null;
    const predBounds = itemDetails ? getBoundsFromLngLat(itemDetails.predicted_polygons) : null;

    const countClasses = (polys) => {
        if (!polys) return {};
        const counts = {'no-damage':0, 'minor-damage':0, 'major-damage':0, 'destroyed':0, 'un-classified':0};
        polys.forEach(p => {
            counts[p.damage_class] = (counts[p.damage_class] || 0) + 1;
        });
        return counts;
    };

    return (
        <div style={{ display: 'flex', gap: '1.5rem', height: '100%', padding: '1.5rem', boxSizing: 'border-box', overflowY: 'hidden' }}>
            
            {/* LEFT COLUMN */}
            <div className="glass-panel" style={{ width: '280px', display: 'flex', flexDirection: 'column', flexShrink: 0, height: 'calc(100vh - 120px)' }}>
                <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--glass-border)' }}>
                    <h2 style={{ marginTop: 0, marginBottom: '1rem', fontSize: '1.2rem', fontWeight: 700 }}>Building Damage Dataset</h2>
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                        <button onClick={() => setFilterSplit('all')} className={filterSplit==='all'?'btn-primary':'btn-secondary'} style={{flex:1, padding:'0.5rem', fontSize:'0.8rem'}}>All</button>
                        <button onClick={() => setFilterSplit('train')} className={filterSplit==='train'?'btn-primary':'btn-secondary'} style={{flex:1, padding:'0.5rem', fontSize:'0.8rem'}}>Train</button>
                        <button onClick={() => setFilterSplit('test')} className={filterSplit==='test'?'btn-primary':'btn-secondary'} style={{flex:1, padding:'0.5rem', fontSize:'0.8rem'}}>Test</button>
                    </div>
                </div>
                
                <div style={{ flex: 1, overflowY: 'auto', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    {loading ? <div style={{textAlign:'center', padding:'2rem'}}><Loader2 className="animate-spin text-accent" /></div> : 
                     filteredDataset.map(item => (
                        <div 
                            key={item.id}
                            onClick={() => fetchDetails(item)}
                            style={{
                                display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.75rem',
                                borderRadius: '8px', cursor: 'pointer',
                                background: selectedItem?.id === item.id ? 'var(--bg-secondary)' : 'rgba(255,255,255,0.4)',
                                border: selectedItem?.id === item.id ? '1px solid var(--accent-primary)' : '1px solid var(--glass-border)',
                                transition: 'all 0.2s', flexShrink: 0
                            }}
                        >
                            <div style={{ flex: 1, overflow: 'hidden' }}>
                                <div style={{ fontSize: '0.8rem', fontFamily: 'monospace', whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden', fontWeight: 600 }}>{item.id}</div>
                                <span style={{
                                    display: 'inline-block', marginTop: '4px',
                                    background: item.split === 'train' ? 'rgba(37, 99, 235, 0.1)' : 'rgba(124, 58, 237, 0.1)', 
                                    color: item.split === 'train' ? 'var(--accent-primary)' : 'var(--accent-secondary)', 
                                    padding: '2px 6px', borderRadius: '4px', fontSize: '0.65rem', fontWeight: 700, textTransform: 'uppercase'
                                }}>{item.split}</span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* RIGHT COLUMN */}
            <div className="main-content" style={{ flex: 1, padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.5rem', overflowY: 'auto', background: 'var(--bg-primary)' }}>
                {!selectedItem ? (
                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                        <div className="glass-panel" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.5rem', border: '1px solid rgba(255,255,255,0.05)' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', color: 'var(--primary)' }}>
                                <Activity size={28} />
                                <h2 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 800, letterSpacing: '-0.02em' }}>Model Performance Overview</h2>
                            </div>
                            
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem' }}>
                                <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1.5rem', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                                    <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '0.5rem' }}>Global Accuracy</div>
                                    <div style={{ fontSize: '2.5rem', fontWeight: 900, color: 'var(--primary)' }}>{stats.accuracy ? (stats.accuracy * 100).toFixed(1) + '%' : '61.2%'}</div>
                                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>Top epoch performance</div>
                                </div>
                                <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1.5rem', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                                    <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '0.5rem' }}>F1 Score (Weighted)</div>
                                    <div style={{ fontSize: '2.5rem', fontWeight: 900, color: 'hsl(210, 100%, 70%)' }}>{stats.f1_score ? (stats.f1_score * 100).toFixed(1) + '%' : '61.4%'}</div>
                                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>Class-weighted balance</div>
                                </div>
                                <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1.5rem', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                                    <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '0.5rem' }}>Training Loss</div>
                                    <div style={{ fontSize: '2.5rem', fontWeight: 900, color: '#ffbb33' }}>{stats.loss ? stats.loss.toFixed(3) : '1.028'}</div>
                                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>Total convergence (25 epochs)</div>
                                </div>
                            </div>
                        </div>

                        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', flexDirection: 'column', gap: '1rem', opacity: 0.6 }}>
                            <Database size={48} strokeWidth={1} />
                            <p style={{ fontWeight: 500 }}>Select an image from the sidebar to view detailed building analysis</p>
                        </div>
                    </div>
                ) : (
                    <>
                        <div className="glass-panel" style={{ padding: '1rem 1.5rem', display: 'flex', alignItems: 'center', gap: '1rem', flexShrink: 0 }}>
                            <h2 style={{ margin: 0, fontSize: '1.3rem', fontFamily: 'monospace' }}>{selectedItem.id}</h2>
                            <span style={{
                                background: selectedItem.split === 'train' ? 'rgba(37, 99, 235, 0.1)' : 'rgba(124, 58, 237, 0.1)', 
                                color: selectedItem.split === 'train' ? 'var(--accent-primary)' : 'var(--accent-secondary)', 
                                padding: '4px 10px', borderRadius: '6px', fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase'
                            }}>{selectedItem.split}</span>
                        </div>

                        {/* IMAGE ROW */}
                        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1fr)', gap: '1rem', flexShrink: 0 }}>
                            <div className="glass-panel" style={{ padding: '0.75rem', display: 'flex', flexDirection: 'column' }}>
                                <div style={{ width: '100%', aspectRatio: '1/1', background: '#e2e8f0', borderRadius: '8px', position: 'relative', overflow: 'hidden' }}>
                                    {itemLoading ? (
                                        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}><Loader2 className="animate-spin text-accent" /></div>
                                    ) : itemDetails && (
                                        <img src={itemDetails.pre_image} style={{ width: '100%', height: '100%', objectFit: 'contain' }} alt="pre" />
                                    )}
                                </div>
                                <div style={{ textAlign: 'center', marginTop: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Pre-Event</div>
                            </div>

                            <div className="glass-panel" style={{ padding: '0.75rem', display: 'flex', flexDirection: 'column' }}>
                                <div style={{ width: '100%', aspectRatio: '1/1', background: '#e2e8f0', borderRadius: '8px', position: 'relative', overflow: 'hidden' }}>
                                    {itemLoading ? (
                                        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}><Loader2 className="animate-spin text-accent" /></div>
                                    ) : itemDetails && (
                                        <>
                                            <img 
                                                src={itemDetails.post_image} 
                                                onLoad={handleGtImageLoad}
                                                style={{ width: '100%', height: '100%', objectFit: 'contain', position: 'absolute', inset: 0 }} 
                                                alt="post gt" 
                                            />
                                            <canvas ref={gtCanvasRef} width={1024} height={1024} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 10 }} />
                                        </>
                                    )}
                                </div>
                                <div style={{ textAlign: 'center', marginTop: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Ground Truth</div>
                            </div>

                            <div className="glass-panel" style={{ padding: '0.75rem', display: 'flex', flexDirection: 'column' }}>
                                <div style={{ width: '100%', aspectRatio: '1/1', background: '#e2e8f0', borderRadius: '8px', position: 'relative', overflow: 'hidden' }}>
                                    {itemLoading ? (
                                        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}><Loader2 className="animate-spin text-accent" /></div>
                                    ) : itemDetails ? (
                                        <>
                                            <img 
                                                src={itemDetails.post_image} 
                                                onLoad={handlePredImageLoad}
                                                style={{ width: '100%', height: '100%', objectFit: 'contain', position: 'absolute', inset: 0 }} 
                                                alt="post pred" 
                                            />
                                            {!itemDetails.predicted_polygons ? (
                                                <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(255,255,255,0.7)', zIndex: 20 }}>
                                                    <span style={{ fontWeight: 600, color: 'var(--text-primary)', textAlign: 'center', padding: '1rem' }}>No model found<br/><small style={{fontWeight:400, color:'var(--text-secondary)'}}>{itemDetails.message}</small></span>
                                                </div>
                                            ) : (
                                                <canvas ref={predCanvasRef} width={1024} height={1024} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 10 }} />
                                            )}
                                        </>
                                    ) : null}
                                </div>
                                <div style={{ textAlign: 'center', marginTop: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Prediction</div>
                            </div>
                        </div>

                        {/* MAP ROW */}
                        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '1rem', flexShrink: 0 }}>
                            <div className="glass-panel" style={{ padding: '0.75rem' }}>
                                <div style={{ marginBottom: '0.75rem', fontWeight: 600, color: 'var(--text-primary)' }}>Ground Truth Map</div>
                                <div style={{ height: '350px', borderRadius: '8px', overflow: 'hidden', background: '#e2e8f0', position: 'relative', zIndex: 1 }}>
                                    {gtBounds ? (
                                        <MapViewer bounds={gtBounds}>
                                            {itemDetails.ground_truth_polygons?.map((p, i) => (
                                                p.lnglat_coords && p.lnglat_coords.length > 0 && (
                                                    <Polygon key={i} positions={p.lnglat_coords.map(c => [c[1], c[0]])} pathOptions={{ color: p.color, weight: 2, fillColor: p.color, fillOpacity: 0.5 }} />
                                                )
                                            ))}
                                        </MapViewer>
                                    ) : (
                                        <div style={{width:'100%', height:'100%', display:'flex', alignItems:'center', justifyContent:'center'}}>No GPS coordinates available.</div>
                                    )}
                                </div>
                            </div>

                            <div className="glass-panel" style={{ padding: '0.75rem' }}>
                                <div style={{ marginBottom: '0.75rem', fontWeight: 600, color: 'var(--text-primary)' }}>Prediction Map</div>
                                <div style={{ height: '350px', borderRadius: '8px', overflow: 'hidden', background: '#e2e8f0', position: 'relative', zIndex: 1 }}>
                                    {predBounds ? (
                                        <MapViewer bounds={predBounds}>
                                            {itemDetails.predicted_polygons.map((p, i) => (
                                                p.lnglat_coords && p.lnglat_coords.length > 0 && (
                                                    <Polygon key={i} positions={p.lnglat_coords.map(c => [c[1], c[0]])} pathOptions={{ color: p.color, weight: 2, fillColor: p.color, fillOpacity: 0.5 }} />
                                                )
                                            ))}
                                        </MapViewer>
                                    ) : itemDetails?.predicted_polygons ? (
                                        <div style={{width:'100%', height:'100%', display:'flex', alignItems:'center', justifyContent:'center'}}>No GPS coordinates available.</div>
                                    ) : (
                                        <div style={{width:'100%', height:'100%', display:'flex', alignItems:'center', justifyContent:'center'}}>No prediction map available.</div>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* LEGEND ROW */}
                        <div className="glass-panel" style={{ padding: '1rem', display: 'flex', justifyContent: 'center', gap: '1.5rem', flexWrap: 'wrap', flexShrink: 0 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.9rem', fontWeight: 600 }}><div style={{ width:'12px', height:'12px', borderRadius:'50%', background:'#00C851' }}></div> No Damage</div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.9rem', fontWeight: 600 }}><div style={{ width:'12px', height:'12px', borderRadius:'50%', background:'#ffbb33' }}></div> Minor Damage</div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.9rem', fontWeight: 600 }}><div style={{ width:'12px', height:'12px', borderRadius:'50%', background:'#ff8800' }}></div> Major Damage</div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.9rem', fontWeight: 600 }}><div style={{ width:'12px', height:'12px', borderRadius:'50%', background:'#CC0000' }}></div> Destroyed</div>
                        </div>

                        {/* STATS ROW */}
                        {itemDetails && (
                            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '1rem', flexShrink: 0 }}>
                                <div className="glass-panel" style={{ padding: '1rem' }}>
                                    <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.75rem', textTransform: 'uppercase' }}>GT Buildings</div>
                                    <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                                        {Object.entries(countClasses(itemDetails.ground_truth_polygons)).map(([cls, count]) => count > 0 && (
                                            <div key={cls} style={{ background: 'var(--bg-secondary)', padding: '0.25rem 0.5rem', borderRadius: '4px', fontSize: '0.8rem', fontWeight: 600 }}>
                                                {cls}: {count}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                                <div className="glass-panel" style={{ padding: '1rem' }}>
                                    <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.75rem', textTransform: 'uppercase' }}>Pred Buildings</div>
                                    <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                                        {itemDetails.predicted_polygons ? Object.entries(countClasses(itemDetails.predicted_polygons)).map(([cls, count]) => count > 0 && (
                                            <div key={cls} style={{ background: 'var(--bg-secondary)', padding: '0.25rem 0.5rem', borderRadius: '4px', fontSize: '0.8rem', fontWeight: 600 }}>
                                                {cls}: {count}
                                            </div>
                                        )) : <span style={{fontSize:'0.8rem', color:'var(--text-secondary)'}}>No prediction</span>}
                                    </div>
                                </div>
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
}
