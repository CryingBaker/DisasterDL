import React, { useRef } from 'react';
import { 
    FileText, X, Printer, ShieldAlert, 
    Globe2, Activity, Map, Navigation2, Loader2
} from 'lucide-react';

export default function MissionBriefing({ isOpen, onClose, data, type }) {
    const reportRef = useRef(null);
    const [exporting, setExporting] = React.useState(false);

    if (!isOpen || !data) return null;

    const missionId = `MISSION-${new Date().toISOString().substring(0, 10).replace(/-/g, '')}-${Math.random().toString(36).substring(2, 6).toUpperCase()}`;
    const timestamp = new Date().toLocaleString();

    const handleExportPDF = async () => {
        const element = reportRef.current;
        if (!element) return;

        setExporting(true);

        try {
            // Dynamically import to avoid SSR issues
            const html2canvas = (await import('https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.esm.min.js')).default;
            const { jsPDF } = await import('https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js');

            // Capture the white report document only
            const canvas = await html2canvas(element, {
                scale: 2,               // retina quality
                useCORS: true,          // allow cross-origin images
                allowTaint: false,
                backgroundColor: '#ffffff',
                logging: false,
                // Scroll the element into view before capture
                scrollY: -window.scrollY,
                windowWidth: element.scrollWidth,
                windowHeight: element.scrollHeight,
            });

            const imgData = canvas.toDataURL('image/png');
            const pdf = new jsPDF({
                orientation: 'portrait',
                unit: 'mm',
                format: 'a4',
            });

            const pdfWidth = pdf.internal.pageSize.getWidth();
            const pdfHeight = pdf.internal.pageSize.getHeight();

            // Calculate how many pages we need
            const imgWidth = canvas.width;
            const imgHeight = canvas.height;
            const ratio = pdfWidth / (imgWidth / 2); // divide by scale factor
            const scaledHeight = (imgHeight / 2) * ratio;

            let position = 0;
            let remainingHeight = scaledHeight;

            // Add first page
            pdf.addImage(imgData, 'PNG', 0, position, pdfWidth, scaledHeight);
            remainingHeight -= pdfHeight;

            // Add extra pages if content overflows
            while (remainingHeight > 0) {
                position -= pdfHeight;
                pdf.addPage();
                pdf.addImage(imgData, 'PNG', 0, position, pdfWidth, scaledHeight);
                remainingHeight -= pdfHeight;
            }

            pdf.save(`${missionId}.pdf`);
        } catch (err) {
            console.error('PDF export failed:', err);
            // Graceful fallback to print dialog
            window.print();
        } finally {
            setExporting(false);
        }
    };

    return (
        <div className="report-modal-overlay">
            <div className="report-container glass-panel">
                {/* ── HEADER (Action Bar) ── */}
                <div className="report-actions no-print">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: 'var(--accent-primary)' }}>
                        <FileText size={20} />
                        <span style={{ fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Intelligence Briefing Mode</span>
                    </div>
                    <div style={{ display: 'flex', gap: '1rem' }}>
                        <button
                            onClick={handleExportPDF}
                            disabled={exporting}
                            className="btn-primary"
                            style={{ padding: '0.5rem 1rem', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
                        >
                            {exporting
                                ? <><Loader2 size={16} className="animate-spin" /> Generating PDF…</>
                                : <><Printer size={16} /> Export to PDF</>
                            }
                        </button>
                        <button onClick={onClose} style={{ background: 'transparent', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                            <X size={24} />
                        </button>
                    </div>
                </div>

                {/* ── THE DOCUMENT (this is what gets captured) ── */}
                <div className="report-document" id="printable-report" ref={reportRef}>
                    <div className="report-header">
                        <div className="header-top">
                            <span className="security-tag">RESTRICTED // OPERATIONAL INTELLIGENCE</span>
                            <span className="timestamp">{timestamp}</span>
                        </div>
                        <div className="header-main">
                            <h1>DISASTER IMPACT ASSESSMENT</h1>
                            <div className="mission-id">REFERENCE: {missionId}</div>
                        </div>
                    </div>

                    <div className="report-body">
                        {/* Summary Section */}
                        <section>
                            <h3><Activity size={16} /> SITUATIONAL OVERVIEW</h3>
                            <div className="info-grid">
                                <div className="info-item">
                                    <label>Analysis Type</label>
                                    <value>{type === 'flood' ? 'Flood Inundation Mapping' : 'Building Damage Classification'}</value>
                                </div>
                                <div className="info-item">
                                    <label>Satellite Constellation</label>
                                    <value>{type === 'flood' ? 'Sentinel-1 (SAR)' : 'Sentinel-2 (Optical Multi-Spectral)'}</value>
                                </div>
                                <div className="info-item">
                                    <label>Processing Engine</label>
                                    <value>{data.model_used || 'Hybrid U-Net/Siamese ResNet'}</value>
                                </div>
                                <div className="info-item">
                                    <label>Status</label>
                                    <value style={{ color: '#00C851' }}>CONFIRMED</value>
                                </div>
                            </div>
                        </section>

                        {/* Results Section */}
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginTop: '2rem' }}>
                            <section>
                                <h3><Navigation2 size={16} /> QUANTIFIED IMPACT</h3>
                                <div className="stats-box">
                                    {type === 'flood' ? (
                                        <>
                                            <div style={{ fontSize: '2rem', fontWeight: 800 }}>{data.estimated_area_km2} <span style={{ fontSize: '1rem' }}>km²</span></div>
                                            <p style={{ margin: '0.5rem 0 0', opacity: 0.7 }}>Estimated total inundated area within specified AOI.</p>
                                        </>
                                    ) : (
                                        <>
                                            <div style={{ fontSize: '2rem', fontWeight: 800 }}>{data.accuracy}% <span style={{ fontSize: '1rem' }}>Confidence</span></div>
                                            <p style={{ margin: '0.5rem 0 0', opacity: 0.7 }}>Multi-class building damage detection accuracy.</p>
                                        </>
                                    )}
                                </div>
                                <table className="report-table">
                                    <thead>
                                        <tr>
                                            <th>Classification Class</th>
                                            <th align="right">Intensity / Value</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {Object.entries(data.breakdown || {}).map(([name, stats]) => (
                                            <tr key={name}>
                                                <td>{name}</td>
                                                <td align="right">{stats.percentage}%</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </section>

                            <section>
                                <h3><Map size={16} /> GEOSPATIAL BOUNDS</h3>
                                <div className="bounds-box">
                                    {data.bounds ? (
                                        <div style={{ fontSize: '0.75rem', fontFamily: 'monospace', lineHeight: 2 }}>
                                            <div style={{ borderBottom: '1px solid rgba(0,0,0,0.1)', paddingBottom: '4px' }}>
                                                NORTHEAST: {data.bounds[1][0].toFixed(6)}, {data.bounds[1][1].toFixed(6)}
                                            </div>
                                            <div style={{ paddingTop: '4px' }}>
                                                SOUTHWEST: {data.bounds[0][0].toFixed(6)}, {data.bounds[0][1].toFixed(6)}
                                            </div>
                                        </div>
                                    ) : (
                                        <p>Coordinates could not be resolved from source metadata.</p>
                                    )}
                                </div>
                                <div className="mission-map-preview">
                                    {data.pred_overlay && (
                                        <img
                                            src={data.pred_overlay}
                                            alt="Reconnaissance Overlay"
                                            crossOrigin="anonymous"
                                            style={{ width: '100%', borderRadius: '4px', filter: 'grayscale(1)' }}
                                        />
                                    )}
                                    <div style={{ position: 'absolute', inset: 0, border: '2px solid rgba(0,0,0,0.2)', pointerEvents: 'none' }} />
                                    <div style={{ position: 'absolute', bottom: '8px', right: '8px', background: 'rgba(0,0,0,0.5)', padding: '2px 4px', fontSize: '8px', color: 'white' }}>Mission Overlay Alpha</div>
                                </div>
                            </section>
                        </div>

                        {/* Directives Section */}
                        <section style={{ marginTop: '2rem' }}>
                            <h3><ShieldAlert size={16} /> OPERATIONAL DIRECTIVES</h3>
                            <div className="directives">
                                {type === 'flood' ? (
                                    <>
                                        <div className="directive-item">• Prioritize evacuation of coastal residential quadrants with area &gt; 0.5km².</div>
                                        <div className="directive-item">• Deploy shallow-draft maritime rescue units to high-saturation zones.</div>
                                    </>
                                ) : (
                                    <>
                                        <div className="directive-item">• Immediate deployment of search &amp; rescue teams to "Destroyed" structures.</div>
                                        <div className="directive-item">• Strategic assessment required for critical infrastructure within radius.</div>
                                    </>
                                )}
                            </div>
                        </section>
                    </div>

                    <div className="report-footer">
                        <div style={{ borderTop: '2px solid #000', paddingTop: '1rem', display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontWeight: 700 }}>
                            <span>DISASTERDL // UNIT 734-ALPHA</span>
                            <span>END OF BRIEFING</span>
                        </div>
                    </div>
                </div>
            </div>

            <style dangerouslySetInnerHTML={{ __html: `
                .report-modal-overlay {
                    position: fixed;
                    inset: 0;
                    background: rgba(0, 0, 0, 0.85);
                    backdrop-filter: blur(8px);
                    z-index: 1000;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 2rem;
                    animation: fadeIn 0.3s ease;
                }
                .report-container {
                    width: 100%;
                    max-width: 900px;
                    max-height: 90vh;
                    display: flex;
                    flex-direction: column;
                    overflow: hidden;
                }
                .report-actions {
                    padding: 1rem 2rem;
                    border-bottom: 1px solid var(--glass-border);
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    flex-shrink: 0;
                }
                .report-document {
                    background: white;
                    color: black;
                    padding: 3rem;
                    overflow-y: auto;
                    font-family: 'Inter', sans-serif;
                    flex: 1;
                }
                .report-header {
                    border-bottom: 3px solid black;
                    padding-bottom: 1.5rem;
                    margin-bottom: 2rem;
                }
                .header-top {
                    display: flex;
                    justify-content: space-between;
                    font-size: 0.7rem;
                    font-weight: 800;
                    letter-spacing: 0.05em;
                    margin-bottom: 1rem;
                }
                .security-tag { color: #d32f2f; }
                .report-header h1 {
                    margin: 0;
                    font-size: 2.5rem;
                    font-weight: 900;
                    letter-spacing: -0.02em;
                }
                .mission-id {
                    margin-top: 0.5rem;
                    font-size: 0.9rem;
                    font-weight: 700;
                    opacity: 0.6;
                }
                .report-body section { margin-bottom: 2rem; }
                .report-body h3 {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    font-size: 0.9rem;
                    font-weight: 800;
                    text-transform: uppercase;
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 0.5rem;
                    margin-bottom: 1rem;
                }
                .info-grid {
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 1rem;
                }
                .info-item label {
                    display: block;
                    font-size: 0.65rem;
                    font-weight: 700;
                    color: #666;
                    text-transform: uppercase;
                    margin-bottom: 0.2rem;
                }
                .info-item value {
                    font-size: 0.9rem;
                    font-weight: 700;
                }
                .stats-box, .bounds-box {
                    background: #f8f9fa;
                    padding: 1.5rem;
                    border: 1px solid #eee;
                    border-radius: 4px;
                    margin-bottom: 1rem;
                }
                .report-table {
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 0.85rem;
                }
                .report-table th, .report-table td {
                    padding: 0.75rem 0;
                    border-bottom: 1px solid #eee;
                }
                .mission-map-preview {
                    position: relative;
                    background: #eee;
                    aspect-ratio: 16/9;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    overflow: hidden;
                }
                .directives {
                    background: #fff8f8;
                    border-left: 4px solid #d32f2f;
                    padding: 1.5rem;
                }
                .directive-item {
                    font-size: 0.9rem;
                    font-weight: 600;
                    margin-bottom: 0.5rem;
                }
                .report-footer {
                    margin-top: 3rem;
                }
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
            ` }} />
        </div>
    );
}