'use client';

export default function Navbar() {
  return (
    <header className="navbar">
      <div className="container nav-content">
        <a href="#" className="logo">
          <div className="logo-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#E9D5FF" strokeWidth="2.5">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
          </div>
          SIGMA
        </a>

        <ul className="nav-links">
          <li><a href="#features">Capabilities</a></li>
          <li><a href="#workbench">Workbench</a></li>
          <li><a href="#showcase">Platform</a></li>
          <li><a href="#benchmarks">Off-Air Tests</a></li>
          <li><a href="#code">Architecture</a></li>
        </ul>

        <div className="nav-actions">
          <a href="https://github.com/SumitKumar00113/sigma-signal-analysis" target="_blank" rel="noreferrer" className="btn-nav-github">
            GitHub
          </a>
          <a href="https://github.com/SumitKumar00113/sigma-signal-analysis/releases/latest" target="_blank" rel="noreferrer" className="btn-nav-primary">
            Download App
          </a>
        </div>
      </div>
    </header>
  );
}
