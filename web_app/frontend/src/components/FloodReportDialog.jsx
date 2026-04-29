import { useRef, useState, useCallback } from 'react';
import { X, Download, Loader2, FileText, Droplets, MapPin, Clock, Cpu } from 'lucide-react';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

const SEVERITY_CONFIG = {
    low: { label: 'Low', color: '#22c55e', bg: 'rgba(34,197,94,0.1)', border: 'rgba(34,197,94,0.3)' },
    moderate: { label: 'Moderate', color: '#f59e0b', bg: 'rgba(245,158,11,0.1)', border: 'rgba(245,158,11,0.3)' },
    high: { label: 'High', color: '#ef4444', bg: 'rgba(239,68,68,0.1)', border: 'rgba(239,68,68,0.3)' },
    critical: { label: 'Critical', color: '#dc2626', bg: 'rgba(220,38,38,0.15)', border: 'rgba(220,38,38,0.4)' },
};

function getSeverity(pct) {
    if (pct >= 60) return SEVERITY_CONFIG.critical;
    if (pct >= 30) return SEVERITY_CONFIG.high;
    if (pct >= 10) return SEVERITY_CONFIG.moderate;
    return SEVERITY_CONFIG.low;
}

const CHANNELS_META = {
    post_s1_image: { label: 'Post-Event SAR (S1)', icon: '📡' },
    post_s2_image: { label: 'Post-Event Optical (S2)', icon: '🛰️' },
    pre_s1_image: { label: 'Pre-Event SAR (S1)', icon: '📡' },
    pre_s2_image: { label: 'Pre-Event Optical (S2)', icon: '🛰️' },
};

// ── Section-aware PDF generator ─────────────────────────────────────────────
// Renders each report section individually and places them on pages so that
// no section is ever split across a page boundary.
async function generateSectionPDF(sectionRefs, reportId) {
    const pdf = new jsPDF('p', 'mm', 'a4');
    const pdfW = pdf.internal.pageSize.getWidth();
    const pdfH = pdf.internal.pageSize.getHeight();
    const margin = 10;           // mm
    const contentW = pdfW - margin * 2;
    const pageContentH = pdfH - margin * 2;
    const gapMM = 3;             // gap between sections

    let cursorY = margin;        // current Y position on the page
    let isFirstPage = true;

    for (const ref of sectionRefs) {
        if (!ref) continue;

        const canvas = await html2canvas(ref, {
            scale: 2,
            useCORS: true,
            backgroundColor: '#ffffff',
            logging: false,
            windowWidth: 740,
        });

        const imgData = canvas.toDataURL('image/png');
        const sectionH_mm = (canvas.height * contentW) / canvas.width;

        // If this section won't fit on the current page, start a new page
        if (cursorY + sectionH_mm > pdfH - margin && !isFirstPage) {
            pdf.addPage();
            cursorY = margin;
        }
        isFirstPage = false;

        // If a single section is taller than a full page, scale it down
        if (sectionH_mm > pageContentH) {
            const scale = pageContentH / sectionH_mm;
            const scaledW = contentW * scale;
            const scaledH = sectionH_mm * scale;
            const xOffset = margin + (contentW - scaledW) / 2;
            pdf.addImage(imgData, 'PNG', xOffset, cursorY, scaledW, scaledH);
            cursorY += scaledH + gapMM;
        } else {
            pdf.addImage(imgData, 'PNG', margin, cursorY, contentW, sectionH_mm);
            cursorY += sectionH_mm + gapMM;
        }
    }

    pdf.save(`Flood_Report_${reportId}.pdf`);
}

export default function FloodReportDialog({ open, onClose, results, uploadedFiles }) {
    // Refs for each section so we can render them individually for the PDF
    const headerRef     = useRef(null);
    const summaryRef    = useRef(null);
    const breakdownRef  = useRef(null);
    const mapRef        = useRef(null);
    const channelsRef   = useRef(null);
    const filesRef      = useRef(null);
    const footerRef     = useRef(null);

    const [generating, setGenerating] = useState(false);

    const floodedPct = results?.breakdown?.Flooded?.percentage ?? 0;
    const severity = getSeverity(floodedPct);
    const timestamp = new Date().toLocaleString('en-US', {
        dateStyle: 'long',
        timeStyle: 'short',
    });
    const reportId = `FR-${Date.now().toString(36).toUpperCase()}`;

    const channelImages = Object.entries(CHANNELS_META).filter(
        ([key]) => results?.[key]
    );

    const handleDownload = useCallback(async () => {
        setGenerating(true);
        try {
            const refs = [
                headerRef.current,
                summaryRef.current,
                breakdownRef.current,
                mapRef.current,
                channelsRef.current,
                filesRef.current,
                footerRef.current,
            ].filter(Boolean);

            await generateSectionPDF(refs, reportId);
        } catch (err) {
            console.error('PDF generation failed:', err);
        } finally {
            setGenerating(false);
        }
    }, [results, reportId]);

    if (!open || !results) return null;

    // Shared section style for consistency in the rendered report
    const sectionStyle = {
        width: '740px',
        background: '#fff',
        fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
        color: '#0f172a',
        padding: '0 2rem',
    };

    return (
        <div style={{
            position: 'fixed', inset: 0, zIndex: 9999,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'rgba(0,0,0,0.5)',
            backdropFilter: 'blur(6px)',
            animation: 'fadeIn 0.25s ease',
        }}
            onClick={onClose}
        >
            {/* Dialog Container */}
            <div
                onClick={e => e.stopPropagation()}
                style={{
                    width: '860px', maxWidth: '95vw',
                    maxHeight: '92vh',
                    background: '#fff',
                    borderRadius: '20px',
                    boxShadow: '0 25px 80px rgba(0,0,0,0.25), 0 0 0 1px rgba(0,0,0,0.05)',
                    display: 'flex', flexDirection: 'column',
                    animation: 'slideUp 0.3s cubic-bezier(0.4,0,0.2,1)',
                    overflow: 'hidden',
                }}
            >
                {/* Dialog Header */}
                <div style={{
                    padding: '1.25rem 1.5rem',
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    borderBottom: '1px solid rgba(0,0,0,0.08)',
                    background: 'linear-gradient(135deg, rgba(37,99,235,0.04), rgba(124,58,237,0.04))',
                    flexShrink: 0,
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <div style={{
                            width: 36, height: 36, borderRadius: 10,
                            background: 'linear-gradient(135deg, #2563eb, #7c3aed)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                        }}>
                            <FileText size={18} color="#fff" />
                        </div>
                        <div>
                            <div style={{ fontWeight: 700, fontSize: '1.05rem', color: '#0f172a' }}>Flood Assessment Report</div>
                            <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Report ID: {reportId}</div>
                        </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <button
                            onClick={handleDownload}
                            disabled={generating}
                            style={{
                                display: 'flex', alignItems: 'center', gap: '0.5rem',
                                padding: '0.55rem 1.25rem',
                                borderRadius: '10px', border: 'none',
                                background: 'linear-gradient(135deg, #2563eb, #7c3aed)',
                                color: '#fff', fontWeight: 600, fontSize: '0.85rem',
                                cursor: generating ? 'wait' : 'pointer',
                                transition: 'all 0.2s',
                                boxShadow: '0 4px 12px rgba(37,99,235,0.3)',
                                opacity: generating ? 0.7 : 1,
                            }}
                        >
                            {generating ? <Loader2 size={16} className="animate-spin" /> : <Download size={16} />}
                            {generating ? 'Generating…' : 'Download PDF'}
                        </button>
                        <button onClick={onClose} style={{
                            width: 36, height: 36, borderRadius: 10,
                            border: '1px solid rgba(0,0,0,0.1)',
                            background: 'rgba(0,0,0,0.03)',
                            cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
                            transition: 'all 0.15s',
                        }}>
                            <X size={18} color="#64748b" />
                        </button>
                    </div>
                </div>

                {/* Scrollable Report Body */}
                <div style={{ overflowY: 'auto', flex: 1, padding: '2rem 0', background: '#f8fafc' }}>
                    <div style={{ width: '740px', margin: '0 auto' }}>

                        {/* ═══ Section 1: Header Banner ═══ */}
                        <div ref={headerRef} style={{ ...sectionStyle, padding: '0 2rem 1.5rem' }}>
                            <div style={{
                                background: 'linear-gradient(135deg, #1e3a5f, #2563eb, #7c3aed)',
                                borderRadius: '16px', padding: '2rem 2.5rem',
                                color: '#fff',
                                position: 'relative', overflow: 'hidden',
                            }}>
                                <div style={{
                                    position: 'absolute', top: -30, right: -30,
                                    width: 120, height: 120, borderRadius: '50%',
                                    background: 'rgba(255,255,255,0.08)',
                                }} />
                                <div style={{
                                    position: 'absolute', bottom: -20, left: '40%',
                                    width: 80, height: 80, borderRadius: '50%',
                                    background: 'rgba(255,255,255,0.05)',
                                }} />
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
                                    <Droplets size={28} />
                                    <span style={{ fontSize: '1.6rem', fontWeight: 800, letterSpacing: '-0.02em' }}>
                                        Flood Assessment Report
                                    </span>
                                </div>
                                <div style={{ fontSize: '0.85rem', opacity: 0.85, fontWeight: 500 }}>
                                    DisasterDL — Satellite-Based Flood Segmentation Analysis
                                </div>
                                <div style={{
                                    marginTop: '1.25rem', display: 'flex', gap: '2rem',
                                    fontSize: '0.78rem', opacity: 0.75,
                                }}>
                                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                        <Clock size={13} /> {timestamp}
                                    </span>
                                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                        <FileText size={13} /> {reportId}
                                    </span>
                                    {results.model_used && (
                                        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                            <Cpu size={13} /> {results.model_used}
                                        </span>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* ═══ Section 2: Summary Cards ═══ */}
                        <div ref={summaryRef} style={{ ...sectionStyle, paddingBottom: '0.5rem' }}>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem' }}>
                                <div style={{
                                    padding: '1.25rem', borderRadius: '14px',
                                    border: '1px solid rgba(37,99,235,0.15)',
                                    background: 'linear-gradient(135deg, rgba(37,99,235,0.04), rgba(37,99,235,0.08))',
                                }}>
                                    <div style={{ fontSize: '0.7rem', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.5rem' }}>
                                        Estimated Flooded Area
                                    </div>
                                    <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#2563eb', lineHeight: 1.1 }}>
                                        {results.estimated_area_km2.toFixed(3)}
                                        <span style={{ fontSize: '0.85rem', fontWeight: 500, color: '#64748b', marginLeft: '4px' }}>km²</span>
                                    </div>
                                </div>
                                <div style={{
                                    padding: '1.25rem', borderRadius: '14px',
                                    border: `1px solid ${severity.border}`,
                                    background: severity.bg,
                                }}>
                                    <div style={{ fontSize: '0.7rem', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.5rem' }}>
                                        Flood Coverage
                                    </div>
                                    <div style={{ fontSize: '1.8rem', fontWeight: 800, color: severity.color, lineHeight: 1.1 }}>
                                        {floodedPct}
                                        <span style={{ fontSize: '0.85rem', fontWeight: 500, color: '#64748b', marginLeft: '2px' }}>%</span>
                                    </div>
                                </div>
                                <div style={{
                                    padding: '1.25rem', borderRadius: '14px',
                                    border: `1px solid ${severity.border}`,
                                    background: severity.bg,
                                    display: 'flex', flexDirection: 'column', justifyContent: 'center',
                                }}>
                                    <div style={{ fontSize: '0.7rem', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.5rem' }}>
                                        Severity Level
                                    </div>
                                    <div style={{
                                        display: 'inline-flex', alignItems: 'center', gap: '6px',
                                        padding: '0.35rem 0.85rem', borderRadius: '8px',
                                        background: severity.color, color: '#fff',
                                        fontWeight: 700, fontSize: '0.95rem',
                                        width: 'fit-content',
                                    }}>
                                        ⚠ {severity.label}
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* ═══ Section 3: Classification Breakdown ═══ */}
                        <div ref={breakdownRef} style={{ ...sectionStyle, paddingTop: '1rem', paddingBottom: '0.5rem' }}>
                            <div style={{
                                padding: '1.5rem', borderRadius: '14px',
                                border: '1px solid rgba(0,0,0,0.08)',
                                background: '#fafbfc',
                            }}>
                                <h3 style={{ margin: '0 0 1rem', fontSize: '0.95rem', fontWeight: 700, color: '#0f172a' }}>
                                    Classification Breakdown
                                </h3>
                                {Object.entries(results.breakdown || {}).map(([cls, d]) => {
                                    const isFlood = cls === 'Flooded';
                                    const barColor = isFlood ? 'linear-gradient(90deg, #ef4444, #f97316)' : 'linear-gradient(90deg, #94a3b8, #cbd5e1)';
                                    return (
                                        <div key={cls} style={{ marginBottom: '0.85rem' }}>
                                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', fontSize: '0.85rem' }}>
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                                    <div style={{
                                                        width: 12, height: 12, borderRadius: '50%',
                                                        background: isFlood ? '#ef4444' : '#94a3b8',
                                                    }} />
                                                    <span style={{ fontWeight: 600 }}>{cls}</span>
                                                </div>
                                                <div>
                                                    <span style={{ fontWeight: 700 }}>{d.percentage}%</span>
                                                    <span style={{ color: '#94a3b8', marginLeft: '6px', fontSize: '0.78rem' }}>
                                                        ({d.pixels?.toLocaleString() ?? '—'} px)
                                                    </span>
                                                </div>
                                            </div>
                                            <div style={{ height: 10, background: 'rgba(0,0,0,0.06)', borderRadius: 6, overflow: 'hidden' }}>
                                                <div style={{
                                                    height: '100%', width: `${d.percentage}%`,
                                                    background: barColor, borderRadius: 6,
                                                }} />
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>

                        {/* ═══ Section 4: Prediction Map ═══ */}
                        {results.pred_overlay && (
                            <div ref={mapRef} style={{ ...sectionStyle, paddingTop: '1rem', paddingBottom: '0.5rem' }}>
                                <div style={{
                                    padding: '1.5rem', borderRadius: '14px',
                                    border: '1px solid rgba(0,0,0,0.08)',
                                    background: '#fafbfc',
                                }}>
                                    <h3 style={{ margin: '0 0 1rem', fontSize: '0.95rem', fontWeight: 700, color: '#0f172a', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                        <MapPin size={16} color="#2563eb" /> Flood Prediction Overlay
                                    </h3>
                                    <img
                                        src={results.pred_overlay}
                                        alt="Flood prediction overlay"
                                        crossOrigin="anonymous"
                                        style={{
                                            width: '100%', borderRadius: '10px',
                                            border: '1px solid rgba(0,0,0,0.08)',
                                        }}
                                    />
                                    <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.75rem' }}>
                                        {[
                                            { color: 'rgba(255,0,0,0.7)', label: 'Predicted Flood' },
                                            { color: 'transparent', label: 'Dry / No Data', border: '1px dashed #cbd5e1' },
                                        ].map(({ color, label, border }) => (
                                            <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.78rem', fontWeight: 600 }}>
                                                <div style={{ width: 14, height: 14, borderRadius: 3, background: color, border: border || 'none' }} />
                                                <span style={{ color: '#64748b' }}>{label}</span>
                                            </div>
                                        ))}
                                    </div>
                                    {results.bounds && (
                                        <div style={{ marginTop: '0.75rem', fontSize: '0.72rem', color: '#94a3b8', fontWeight: 500 }}>
                                            📍 Bounds: [{results.bounds[0].map(v => v.toFixed(4)).join(', ')}] → [{results.bounds[1].map(v => v.toFixed(4)).join(', ')}]
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}

                        {/* ═══ Section 5: Channel Previews ═══ */}
                        {channelImages.length > 0 && (
                            <div ref={channelsRef} style={{ ...sectionStyle, paddingTop: '1rem', paddingBottom: '0.5rem' }}>
                                <div style={{
                                    padding: '1.5rem', borderRadius: '14px',
                                    border: '1px solid rgba(0,0,0,0.08)',
                                    background: '#fafbfc',
                                }}>
                                    <h3 style={{ margin: '0 0 1rem', fontSize: '0.95rem', fontWeight: 700, color: '#0f172a' }}>
                                        Input Channel Previews
                                    </h3>
                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem' }}>
                                        {channelImages.map(([key, meta]) => (
                                            <div key={key} style={{
                                                borderRadius: '10px', overflow: 'hidden',
                                                border: '1px solid rgba(0,0,0,0.06)',
                                            }}>
                                                <div style={{
                                                    padding: '0.5rem 0.75rem',
                                                    background: 'rgba(0,0,0,0.03)',
                                                    fontSize: '0.75rem', fontWeight: 600, color: '#475569',
                                                }}>
                                                    {meta.icon} {meta.label}
                                                </div>
                                                <img
                                                    src={results[key]}
                                                    alt={meta.label}
                                                    crossOrigin="anonymous"
                                                    style={{ width: '100%', aspectRatio: '1/1', objectFit: 'contain', background: '#f1f5f9' }}
                                                />
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* ═══ Section 6: Uploaded Files ═══ */}
                        {uploadedFiles && Object.keys(uploadedFiles).length > 0 && (
                            <div ref={filesRef} style={{ ...sectionStyle, paddingTop: '1rem', paddingBottom: '0.5rem' }}>
                                <div style={{
                                    padding: '1.25rem 1.5rem', borderRadius: '14px',
                                    border: '1px solid rgba(0,0,0,0.08)',
                                    background: '#fafbfc',
                                }}>
                                    <h3 style={{ margin: '0 0 0.75rem', fontSize: '0.95rem', fontWeight: 700, color: '#0f172a' }}>
                                        Input Files
                                    </h3>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                                        {Object.entries(uploadedFiles).map(([key, file]) => (
                                            <div key={key} style={{
                                                display: 'flex', justifyContent: 'space-between',
                                                padding: '0.45rem 0.75rem', borderRadius: '8px',
                                                background: 'rgba(0,0,0,0.02)',
                                                fontSize: '0.78rem', fontWeight: 500,
                                            }}>
                                                <span style={{ color: '#64748b', textTransform: 'uppercase', fontWeight: 600, fontSize: '0.7rem', letterSpacing: '0.04em' }}>{key}</span>
                                                <span style={{ color: '#0f172a' }}>{file.name} <span style={{ color: '#94a3b8' }}>({(file.size / 1024 / 1024).toFixed(2)} MB)</span></span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* ═══ Section 7: Footer ═══ */}
                        <div ref={footerRef} style={{ ...sectionStyle, paddingTop: '1rem' }}>
                            <div style={{
                                borderTop: '1px solid rgba(0,0,0,0.08)',
                                paddingTop: '1.25rem',
                                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                fontSize: '0.72rem', color: '#94a3b8', fontWeight: 500,
                            }}>
                                <div>
                                    Generated by <span style={{ fontWeight: 700, color: '#475569' }}>DisasterDL</span> — Satellite Flood Analysis Platform
                                </div>
                                <div>{timestamp}</div>
                            </div>
                        </div>

                    </div>
                </div>
            </div>

            {/* CSS Animations */}
            <style>{`
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                @keyframes slideUp {
                    from { opacity: 0; transform: translateY(20px) scale(0.98); }
                    to { opacity: 1; transform: translateY(0) scale(1); }
                }
                .animate-spin {
                    animation: spin 1s linear infinite;
                }
                @keyframes spin {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }
            `}</style>
        </div>
    );
}
