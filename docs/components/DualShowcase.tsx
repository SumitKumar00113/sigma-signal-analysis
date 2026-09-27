'use client';

export default function DualShowcase() {
  return (
    <section className="dual-showcase-section" id="benchmarks">
      <div className="container">
        <div className="section-header-center">
          <div className="section-eyebrow">Field Proven</div>
          <h2 className="section-title">A Signal Environment Built For Testing And Decoding</h2>
          <p className="section-desc">Analyze raw baseband recordings, recover unknown FEC polynomials, and extract telemetry off-air.</p>
        </div>

        <div className="dual-grid">
          {/* Card 1 */}
          <div className="dual-card">
            <h3>Blind FEC Identification & Solvers</h3>
            <p>Convolutional codes are identified from dual code parity checks across K=3..9. Stride scanning resolves block, diagonal, and pseudo-random interleavers.</p>
            <div className="code-editor-box">
              <div className="code-row"><span className="num">1</span><span><span className="kw">from</span> src.decoding.fec_id <span className="kw">import</span> identify_fec</span></div>
              <div className="code-row"><span className="num">2</span><span>code = identify_fec(demod_bits)</span></div>
              <div className="code-row"><span className="num">3</span><span><span className="cm"># Found: Conv K=7, Rate 1/2, BER: 0.00%</span></span></div>
            </div>
          </div>

          {/* Card 2 */}
          <div className="dual-card">
            <h3>Off-Air Telemetry & Image Recovery</h3>
            <p>Built-in decoders for SITOR-B / NAVTEX marine warnings, Baudot RTTY, RS41 Radiosonde telemetry, and NOAA APT weather satellite imagery (FM subcarrier to PNG).</p>
            <div className="code-editor-box">
              <div className="code-row"><span className="num">1</span><span><span className="kw">from</span> src.decoding.apt <span className="kw">import</span> decode_apt</span></div>
              <div className="code-row"><span className="num">2</span><span>img = decode_apt(<span className="str">&quot;noaa18.wav&quot;</span>)</span></div>
              <div className="code-row"><span className="num">3</span><span><span className="cm"># Saved: noaa18_apt.png (1,097 Lines)</span></span></div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
