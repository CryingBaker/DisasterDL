import { useState, useEffect } from 'react';
import axios from 'axios';
import { 
    Activity, ShieldAlert, Waves, Building2, ChevronRight, 
    BarChart3, Globe2, AlertTriangle, CheckCircle2, Info
} from 'lucide-react';
import { Link } from 'react-router-dom';

const INTEL_API = 'http://localhost:5000/api/intelligence/summary';

export default function DashboardView() {
    const [data, setData] = useState(() => {
        const saved = localStorage.getItem('last_intelligence_summary');
        return saved ? JSON.parse(saved) : null;
    });
    const [loading, setLoading] = useState(!data);

    useEffect(() => {
        const fetchStats = async () => {
            try {
                const res = await axios.get(INTEL_API);
                setData(res.data);
                localStorage.setItem('last_intelligence_summary', JSON.stringify(res.data));
            } catch (err) {
                console.error("Failed to fetch intelligence summary", err);
            } finally {
                setLoading(false);
            }
        };
        fetchStats();
        const interval = setInterval(fetchStats, 10000); // Polling every 10s
        return () => clearInterval(interval);
    }, []);

    if (loading) {
        return (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh', color: 'var(--accent-primary)' }}>
                <Activity className="animate-spin" size={48} />
            </div>
        );
    }

    const crisisIndex = data?.crisis_index || 0;
    const level = data?.level || "Unknown";
    const levelColor = crisisIndex > 60 ? "#ff4444" : crisisIndex > 30 ? "#ffbb33" : "#00C851";

    return (
        <div style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto' }}>
            {/* ── HEADER ── */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '2.5rem' }}>
                <div>
                    <h1 style={{ fontSize: '2.5rem', fontWeight: 800, margin: 0, letterSpacing: '-0.02em' }}>
                        Operation <span className="gradient-text">Control Center</span>
                    </h1>
                    <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', fontSize: '1.1rem' }}>
                        Unified Disaster Intelligence & Crisis Assessment Protocol
                    </p>
                </div>
                <div className="glass-panel" style={{ padding: '0.5rem 1rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <div className="pulse" style={{ width: '8px', height: '8px', background: '#00C851', borderRadius: '50%' }} />
                    <span style={{ fontSize: '0.85rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        System Status: <span style={{ color: '#00C851' }}>Operational</span>
                    </span>
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '2rem' }}>
                
                {/* ── CRISIS INDEX GAUGE ── */}
                <div className="glass-panel" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                    <h3 style={{ margin: '0 0 1.5rem', fontSize: '0.9rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.1em', width: '100%', textAlign: 'left' }}>
                        <ShieldAlert size={16} inline="true" style={{ verticalAlign: 'middle', marginRight: '0.5rem' }} /> Crisis Severity Index
                    </h3>
                    <div style={{ position: 'relative', width: '200px', height: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <svg width="200" height="200" viewBox="0 0 100 100">
                            <circle cx="50" cy="50" r="45" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="8" />
                            <circle 
                                cx="50" cy="50" r="45" fill="none" 
                                stroke={levelColor} strokeWidth="8" 
                                strokeDasharray={`${crisisIndex * 2.82} 282`}
                                strokeLinecap="round" transform="rotate(-90 50 50)"
                                style={{ transition: 'stroke-dasharray 1s ease-out' }}
                            />
                        </svg>
                        <div style={{ position: 'absolute', textAlign: 'center' }}>
                            <div style={{ fontSize: '3.5rem', fontWeight: 900, lineHeight: 1 }}>{Math.round(crisisIndex)}</div>
                            <div style={{ fontSize: '0.8rem', fontWeight: 700, opacity: 0.6, marginTop: '0.2rem' }}>POINTS</div>
                        </div>
                    </div>
                    <div style={{ marginTop: '1.5rem', fontSize: '1.5rem', fontWeight: 800, color: levelColor, textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                        {level} RISK
                    </div>
                </div>

                {/* ── INTELLIGENCE SUMMARY ── */}
                <div className="glass-panel" style={{ padding: '2rem' }}>
                    <h3 style={{ margin: '0 0 1.5rem', fontSize: '0.9rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                        <Info size={16} inline="true" style={{ verticalAlign: 'middle', marginRight: '0.5rem' }} /> Mission Reconnaissance
                    </h3>
                    
                    <div style={{ display: 'grid', gap: '1rem' }}>
                        <div style={{ display: 'flex', gap: '1rem' }}>
                            <div className="glass-panel" style={{ flex: 1, padding: '1rem', background: 'rgba(59, 130, 246, 0.05)' }}>
                                <Waves size={20} style={{ color: '#3b82f6', marginBottom: '0.5rem' }} />
                                <div style={{ fontSize: '0.75rem', fontWeight: 600, opacity: 0.6 }}>Flood Zone</div>
                                <div style={{ fontSize: '1.25rem', fontWeight: 700 }}>{data?.latest?.flood?.estimated_area_km2 || '0.00' } km²</div>
                            </div>
                            <div className="glass-panel" style={{ flex: 1, padding: '1rem', background: 'rgba(239, 68, 68, 0.05)' }}>
                                <Building2 size={20} style={{ color: '#ef4444', marginBottom: '0.5rem' }} />
                                <div style={{ fontSize: '0.75rem', fontWeight: 600, opacity: 0.6 }}>Structures Hit</div>
                                <div style={{ fontSize: '1.25rem', fontWeight: 700 }}>{data?.latest?.damage?.total_buildings || '0' } units</div>
                            </div>
                        </div>

                        <div className="glass-panel" style={{ padding: '1.25rem' }}>
                            <div style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <BarChart3 size={16} className="text-accent" /> Damage Distribution
                            </div>
                            <div style={{ display: 'flex', height: '12px', borderRadius: '6px', overflow: 'hidden', background: 'rgba(255,255,255,0.05)' }}>
                                {Object.entries(data?.latest?.damage?.breakdown || {}).map(([name, stats]) => (
                                    <div 
                                        key={name}
                                        style={{ 
                                            width: `${stats.percentage}%`, 
                                            background: name === 'Destroyed' ? '#CC0000' : name === 'Major Damage' ? '#ff8800' : name === 'Minor Damage' ? '#ffbb33' : '#00C851',
                                            transition: 'width 0.5s ease'
                                        }} 
                                        title={`${name}: ${stats.percentage}%`}
                                    />
                                ))}
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.5rem', fontSize: '0.7rem', fontWeight: 600, opacity: 0.5 }}>
                                <span>No Damage</span>
                                <span>Destroyed</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* ── RECOMMENDATIONS ── */}
            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '2rem' }}>
                <div className="glass-panel" style={{ padding: '2rem' }}>
                    <h3 style={{ margin: '0 0 1.25rem', fontSize: '1rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                        <CheckCircle2 size={20} className="text-accent" /> Operational Directives
                    </h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                        {data?.recommendations?.map((rec, i) => (
                            <div key={i} style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', borderLeft: `4px solid ${crisisIndex > 50 ? '#ef4444' : 'var(--accent-primary)'}`, fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                                {rec}
                            </div>
                        ))}
                        {(!data?.recommendations || data.recommendations.length === 0) && (
                            <div style={{ padding: '1rem', color: 'var(--text-secondary)', fontSize: '0.9rem', fontStyle: 'italic' }}>
                                Awaiting mission data for situational analysis...
                            </div>
                        )}
                    </div>
                </div>

                <div className="glass-panel" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                    <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 700 }}>Quick Launch</h3>
                    <Link to="/fs-predict" className="btn-primary" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', textDecoration: 'none' }}>
                        <span>Flood Analysis</span>
                        <ChevronRight size={18} />
                    </Link>
                    <Link to="/bd-predict" className="btn-primary" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', textDecoration: 'none', background: 'transparent', border: '1px solid var(--accent-primary)' }}>
                        <span>Damage Detection</span>
                        <ChevronRight size={18} />
                    </Link>
                    <div style={{ marginTop: 'auto', paddingTop: '1rem', borderTop: '1px solid var(--glass-border)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                            <Globe2 size={14} /> Global Satellite Feed Active
                        </div>
                    </div>
                </div>
            </div>

            <style dangerouslySetInnerHTML={{ __html: `
                .pulse {
                    box-shadow: 0 0 0 0 rgba(0, 200, 81, 0.7);
                    animation: pulse-green 2s infinite;
                }
                @keyframes pulse-green {
                    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 200, 81, 0.7); }
                    70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(0, 200, 81, 0); }
                    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 200, 81, 0); }
                }
                .gradient-text {
                    background: linear-gradient(135deg, var(--accent-primary), #a855f7);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }
            ` }} />
        </div>
    );
}
