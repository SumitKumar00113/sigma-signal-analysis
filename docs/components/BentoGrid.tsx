'use client';

import GemGraphic from './bento/GemGraphic';

export default function BentoGrid() {
  return (
    <section className="bento-section" id="features">
      <div className="container">
        <div className="section-header-center">
          <div className="section-eyebrow">All-In-One Platform</div>
          <h2 className="section-title">All Of Your Signals In One Place</h2>
          <p className="section-desc">A unified desktop environment engineered for RF inspection and blind telemetry decoding.</p>
        </div>

        <div className="bento-grid">
          {/* Bento Card 1 */}
          <div className="bento-card">
            <div>
              <h3 className="bento-card-title">Real-Signal Ingestion & Hilbert DSP</h3>
              <p className="bento-card-desc">
                Ingest `.wav` (Mono/Stereo IQ), raw `.iq`, and `.sigmf` files. Mono audio files automatically run through a 255-tap Hilbert FIR filter with &gt;80 dB image rejection.
              </p>
            </div>
            <div className="bento-code-preview">
              from src.ingestion import FileReader<br />
              reader = FileReader.from_file(&quot;recording.wav&quot;)<br />
              iq_data = reader.read_analytic_signal() # &gt;80dB Rejection
            </div>
          </div>

          {/* Bento Card 2 */}
          <div className="bento-card">
            <div>
              <h3 className="bento-card-title">Hybrid ML Modulation Classifier</h3>
              <p className="bento-card-desc">
                Identifies 18 modulation types (PSK, QAM, FSK, MSK, GMSK, ASK, AM, FM, SSB). Combines explainable rules with a 38-feature gradient-boosted model achieving 99% accuracy at 4 dB.
              </p>
            </div>
            <GemGraphic />
          </div>

          {/* Bento Card 3 */}
          <div className="bento-card">
            <div>
              <h3 className="bento-card-title">Demodulation & Audio Recovery</h3>
              <p className="bento-card-desc">
                Digital demodulation with RRC matched filtering, Gardner timing, Costas loop, EVM analysis, and 8 kHz audio output for AM/FM/SSB voice communications.
              </p>
            </div>
            <div className="node-diagram">
              <div className="node-pill">PSK</div>
              <div className="node-pill">FSK</div>
              <div className="node-pill center">DEMOD</div>
              <div className="node-pill">QAM</div>
              <div className="node-pill">AUDIO</div>
            </div>
          </div>

          {/* Bento Card 4 */}
          <div className="bento-card">
            <div>
              <h3 className="bento-card-title">One-Click Auto-Analyse Pipeline</h3>
              <p className="bento-card-desc">
                Automated end-to-end processing chain: ingestion → parameters → hybrid ML → demodulation → blind interleaver/FEC scan → frame sync discovery.
              </p>
            </div>
            <div className="toggle-list">
              <div className="toggle-row">
                <span>Blind FEC Code Identification</span>
                <span className="status-badge-active">ACTIVE</span>
              </div>
              <div className="toggle-row">
                <span>Interleaver Stride & Comb Search</span>
                <span className="status-badge-active">ACTIVE</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
