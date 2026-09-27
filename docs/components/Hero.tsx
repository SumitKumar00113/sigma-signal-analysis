'use client';

export default function Hero() {
  return (
    <section className="hero">
      <div className="container">
        <a href="https://github.com/SumitKumar00113/sigma-signal-analysis/releases/latest" target="_blank" rel="noreferrer" className="hero-eyebrow">
          <span>⚡ Phase 3 Released • Windows & macOS Desktop Support</span>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <polyline points="9 18 15 12 9 6" />
          </svg>
        </a>

        <h1 className="hero-heading">
          The Best Place To Ingest, Classify, And Decode <span className="highlight">RF Signal Baseband.</span>
        </h1>

        <p className="hero-subhead">
          A PROFESSIONAL DESKTOP PLATFORM FOR RF ENGINEERS, SIGINT ANALYSTS, AND SDR DEVELOPERS
        </p>

        <div className="hero-cta-group">
          <div className="download-option">
            <a href="https://github.com/SumitKumar00113/sigma-signal-analysis/releases/download/v0.2.0/SIGMA-Setup.exe" className="btn-hero-primary">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M0 3.449L9.75 2.1v9.451H0m10.949-9.602L24 0v11.4H10.949M0 12.6h9.75v9.451L0 20.699M10.949 12.6H24V24l-13.051-1.8" />
              </svg>
              Download for Windows
            </a>
            <p className="download-requirement">Windows 10/11, 64-bit</p>
          </div>
          <div className="download-option">
            <a href="https://github.com/SumitKumar00113/sigma-signal-analysis/releases/download/v0.2.1/Sigma-Signal-Analysis-0.2.1-macOS-arm64.zip" className="btn-hero-secondary">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M18.71 19.5c-.83 1.24-1.71 2.45-3.05 2.47-1.34.03-1.77-.79-3.29-.79-1.53 0-2 .77-3.27.82-1.31.05-2.3-1.32-3.14-2.53C4.25 17 2.94 12.45 4.7 9.39c.87-1.52 2.43-2.48 4.12-2.51 1.28-.02 2.5.87 3.29.87.78 0 2.26-1.07 3.81-.91.65.03 2.47.26 3.64 1.98-.09.06-2.17 1.28-2.15 3.81.03 3.02 2.65 4.03 2.68 4.04-.03.07-.42 1.44-1.38 2.83M15.97 6.32c.62-.75 1.04-1.8 0.92-2.85-.9.04-2 .6-2.65 1.36-.58.68-1.09 1.77-.95 2.81 1.01.08 2.06-.57 2.68-1.32z" />
              </svg>
              Download for macOS
            </a>
            <p className="download-requirement">Apple Silicon (M1 or later), macOS 11+, unsigned — right-click → Open the first time.</p>
          </div>
        </div>
      </div>
    </section>
  );
}
