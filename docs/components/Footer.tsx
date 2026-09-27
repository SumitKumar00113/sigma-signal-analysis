'use client';

export default function Footer() {
  return (
    <footer className="footer">
      <div className="bottom-purple-arc" />
      <div className="container">
        <div className="footer-grid">
          <div className="footer-brand">
            <a href="#" className="logo">
              <div className="logo-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#E9D5FF" strokeWidth="2.5">
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                </svg>
              </div>
              SIGMA
            </a>
            <p>Advanced desktop platform for RF signal inspection, hybrid ML classification, and blind decoding on Windows and macOS.</p>
          </div>

          <div className="footer-column">
            <h4>About</h4>
            <ul>
              <li><a href="#features">Capabilities</a></li>
              <li><a href="#workbench">Workbench</a></li>
              <li><a href="#showcase">Platform</a></li>
            </ul>
          </div>

          <div className="footer-column">
            <h4>Services</h4>
            <ul>
              <li><a href="#benchmarks">Off-Air Testing</a></li>
              <li><a href="#code">Blind Code ID</a></li>
              <li><a href="#features">Auto-Analyse</a></li>
            </ul>
          </div>

          <div className="footer-column">
            <h4>Use Cases</h4>
            <ul>
              <li><a>RF Signals Intelligence</a></li>
              <li><a>Telemetry Decoding</a></li>
              <li><a>SDR Research</a></li>
            </ul>
          </div>

          <div className="footer-column">
            <h4>Resources</h4>
            <ul>
              <li><a href="https://github.com/SumitKumar00113/sigma-signal-analysis" target="_blank" rel="noreferrer">GitHub Repo</a></li>
              <li><a href="https://github.com/SumitKumar00113/sigma-signal-analysis/releases" target="_blank" rel="noreferrer">Releases</a></li>
              <li><a href="https://github.com/SumitKumar00113/sigma-signal-analysis/issues" target="_blank" rel="noreferrer">Issue Tracker</a></li>
            </ul>
          </div>
        </div>

        <div className="footer-bottom">
          <div>© 2026 SIGMA Signal Analysis. All rights reserved.</div>
          <div>Built for RF Engineers & SIGINT Analysts</div>
        </div>
      </div>
    </footer>
  );
}
