import { useState } from 'react';
import { Eye, X } from 'lucide-react';

export default function ImageViewer({ title, src, isLoading }) {
    const [isExpanded, setIsExpanded] = useState(false);

    return (
        <>
            <div className="glass-panel" style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h3 style={{ margin: 0, fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{title}</h3>
                    <button
                        onClick={() => setIsExpanded(true)}
                        style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '4px', borderRadius: '4px', transition: 'background 0.2s' }}
                        className="hover-bg"
                    >
                        <Eye size={16} className="text-secondary" />
                    </button>
                </div>
                <div style={{
                    aspectRatio: '1/1',
                    background: '#f1f5f9',
                    borderRadius: '10px',
                    overflow: 'hidden',
                    position: 'relative',
                    border: '1px solid #e2e8f0'
                }}>
                    {isLoading ? (
                        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8' }}>
                            Loading...
                        </div>
                    ) : src ? (
                        <img src={src} alt={title} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                    ) : (
                        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8', fontSize: '0.8rem' }}>
                            No Data
                        </div>
                    )}
                </div>
            </div>

            {/* Expansion Modal */}
            {isExpanded && (
                <div style={{
                    position: 'fixed',
                    inset: 0,
                    zIndex: 2000,
                    background: 'rgba(255, 255, 255, 0.95)',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '2rem',
                    backdropFilter: 'blur(10px)'
                }}>
                    <div style={{ width: '100%', maxWidth: '1000px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                        <h2 style={{ margin: 0 }}>{title}</h2>
                        <button
                            onClick={() => setIsExpanded(false)}
                            style={{ background: 'rgba(0,0,0,0.05)', border: 'none', borderRadius: '50%', padding: '0.5rem', cursor: 'pointer' }}
                        >
                            <X size={24} />
                        </button>
                    </div>
                    <div style={{ flex: 1, width: '100%', maxWidth: '1000px', background: '#000', borderRadius: '16px', overflow: 'hidden', boxShadow: '0 25px 50px -12px rgba(0,0,0,0.25)' }}>
                        <img src={src} alt={title} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                    </div>
                </div>
            )}
        </>
    );
}
