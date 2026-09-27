'use client';

export default function LargeShowcase() {
  return (
    <section className="large-showcase-section" id="showcase">
      <div className="container">
        <div className="large-showcase-card">
          <div className="showcase-inner-grid">
            <div>
              <div className="section-eyebrow">Integrated Pipeline</div>
              <h2 className="section-title" style={{ fontSize: '2.2rem' }}>
                Engineered For Complex Baseband Ingestion & Signal Analysis
              </h2>
              <p className="section-desc" style={{ marginBottom: '24px' }}>
                Ingest baseband recordings with world-class DSP algorithms, robust noise floor estimation, and real-time visualization.
              </p>
              <a href="https://github.com/SumitKumar00113/sigma-signal-analysis/releases/latest" target="_blank" rel="noreferrer" className="btn-hero-primary">
                Explore Workbench Features
              </a>
            </div>

            <div style={{ background: '#020208', border: '1px solid var(--border-subtle)', borderRadius: '16px', padding: '20px' }}>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', color: 'var(--purple-light)', marginBottom: '12px' }}>
                [PIPELINE_RUNNER] Ingestion → Parameters → Hybrid ML → Decoded Payload
              </div>
              <div style={{ background: '#050514', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '8px', padding: '12px', fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div>✔ SNR PSD Estimate: <span style={{ color: '#34D399' }}>15.2 dB</span></div>
                <div>✔ Carrier Offset: <span style={{ color: '#34D399' }}>+1,842.3 Hz</span></div>
                <div>✔ Hybrid Classifier: <span style={{ color: '#C084FC', fontWeight: 'bold' }}>2-FSK (99.8% Confidence)</span></div>
                <div>✔ Frame Sync Discovery: <span style={{ color: '#38BDF8' }}>ZCZC ASM Found</span></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
