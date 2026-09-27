'use client';

import { FILE_PROFILES, FileProfile } from '@/data/fileProfiles';

interface SidebarProps {
  currentProfileKey: string;
  onSelectFile: (key: string) => void;
  currentProfile: FileProfile;
}

export default function Sidebar({ currentProfileKey, onSelectFile, currentProfile }: SidebarProps) {
  const profilesList = Object.values(FILE_PROFILES);

  return (
    <div className="editor-sidebar">
      <div>
        <div className="sidebar-heading">Off-Air Captures</div>
        {profilesList.map((prof) => (
          <div
            key={prof.key}
            className={`tree-item ${currentProfileKey === prof.key ? 'active' : ''}`}
            onClick={() => onSelectFile(prof.key)}
          >
            <span>📄 {prof.filename}</span>
            <span className="tree-badge">{prof.badge}</span>
          </div>
        ))}
      </div>

      <div>
        <div className="sidebar-heading">DSP Parameters</div>
        <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-subtle)', padding: '12px', borderRadius: '8px', fontFamily: 'var(--font-mono)', fontSize: '0.75rem', display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <div>SNR: <span style={{ color: 'var(--cyan-subtle)' }}>{currentProfile.snr}</span></div>
          <div>Carrier: <span style={{ color: 'var(--cyan-subtle)' }}>{currentProfile.cfo}</span></div>
          <div>Rate: <span style={{ color: 'var(--cyan-subtle)' }}>{currentProfile.rate}</span></div>
          <div style={{ color: 'var(--purple-light)', fontWeight: 600, marginTop: '4px' }}>{currentProfile.cls}</div>
        </div>
      </div>
    </div>
  );
}
