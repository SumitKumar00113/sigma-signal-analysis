'use client';

import { useState } from 'react';
import { FILE_PROFILES } from '@/data/fileProfiles';
import Sidebar from './Sidebar';
import SpectrumCanvas from './SpectrumCanvas';
import ConstellationCanvas from './ConstellationCanvas';
import DecodeTerminal from './DecodeTerminal';

export default function Workbench() {
  const [selectedFileKey, setSelectedFileKey] = useState<string>('navtex');
  const [activeTab, setActiveTab] = useState<'psd' | 'const' | 'decode'>('psd');

  const currentProfile = FILE_PROFILES[selectedFileKey] || FILE_PROFILES.navtex;

  return (
    <div className="mockup-container" id="workbench">
      <div className="perspective-grid-floor" />

      <div className="editor-card">
        {/* Window Bar */}
        <div className="editor-header">
          <div className="mac-dots">
            <span className="mac-dot mac-dot-red" />
            <span className="mac-dot mac-dot-yellow" />
            <span className="mac-dot mac-dot-green" />
          </div>
          <div className="editor-title">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
            </svg>
            SIGMA Signal Analysis v3.0 - [{currentProfile.filename}]
          </div>
          <div className="editor-status-badge">
            ● ANALYSED • {currentProfile.rate}
          </div>
        </div>

        {/* Editor Body */}
        <div className="editor-body">
          <Sidebar
            currentProfileKey={selectedFileKey}
            onSelectFile={(key) => setSelectedFileKey(key)}
            currentProfile={currentProfile}
          />

          {/* Main Panel Viewers */}
          <div className="editor-main">
            <div className="editor-tabs">
              <button
                className={`tab-btn ${activeTab === 'psd' ? 'active' : ''}`}
                onClick={() => setActiveTab('psd')}
              >
                Spectrum & Waterfall
              </button>
              <button
                className={`tab-btn ${activeTab === 'const' ? 'active' : ''}`}
                onClick={() => setActiveTab('const')}
              >
                Constellation & EVM
              </button>
              <button
                className={`tab-btn ${activeTab === 'decode' ? 'active' : ''}`}
                onClick={() => setActiveTab('decode')}
              >
                Decoding Workbench
              </button>
            </div>

            <div className="editor-content">
              {activeTab === 'psd' && <SpectrumCanvas currentProfile={currentProfile} />}
              {activeTab === 'const' && <ConstellationCanvas currentProfile={currentProfile} />}
              {activeTab === 'decode' && <DecodeTerminal currentProfile={currentProfile} />}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
