'use client';

import { FileProfile } from '@/data/fileProfiles';

interface DecodeTerminalProps {
  currentProfile: FileProfile;
}

export default function DecodeTerminal({ currentProfile }: DecodeTerminalProps) {
  return (
    <div style={{ background: '#010105', border: '1px solid rgba(16,185,129,0.35)', borderRadius: '10px', padding: '16px', fontFamily: 'var(--font-mono)', fontSize: '0.78rem', color: '#34D399', lineHeight: '1.65', boxShadow: 'inset 0 0 20px rgba(16,185,129,0.05)' }}>
      <div style={{ whiteSpace: 'pre-line' }}>
        {currentProfile.decodedText}
      </div>
      <div style={{ color: '#A7F3D0', fontWeight: 600, marginTop: '10px', borderTop: '1px solid rgba(16,185,129,0.2)', paddingTop: '6px' }}>
        {currentProfile.decodedStatus}
      </div>
    </div>
  );
}
