'use client';

export default function FinalCta() {
  return (
    <section className="final-cta-section">
      <div className="container">
        <div className="floating-orb-wrapper">
          <div className="developer-orb-3d" />
        </div>

        <h2 className="cta-heading">Ready To Join A New Dimension?</h2>
        <p className="cta-subhead">
          Get started with SIGMA Signal Analysis today on Windows or macOS. Download the desktop release or build from source.
        </p>

        <div className="download-options">
          <div className="download-option">
            <a href="https://github.com/SumitKumar00113/sigma-signal-analysis/releases/download/v0.2.0/SIGMA-Signal-Analysis-v0.2.1-Windows-x64.zip" className="btn-hero-primary">
              Download for Windows
            </a>
            <p className="download-requirement">Windows 10/11, 64-bit</p>
          </div>
          <div className="download-option">
            <a href="https://github.com/SumitKumar00113/sigma-signal-analysis/releases/download/v0.2.1/Sigma-Signal-Analysis-0.2.1-macOS-arm64.zip" className="btn-hero-secondary">
              Download for macOS
            </a>
            <p className="download-requirement">Apple Silicon (M1 or later), macOS 11+, unsigned — right-click → Open the first time.</p>
          </div>
        </div>
      </div>
    </section>
  );
}
