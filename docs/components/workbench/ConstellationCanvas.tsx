'use client';

import { useEffect, useRef } from 'react';
import { FileProfile } from '@/data/fileProfiles';

interface ConstellationCanvasProps {
  currentProfile: FileProfile;
}

export default function ConstellationCanvas({ currentProfile }: ConstellationCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let constTime = 0;
    let animationFrameId: number;

    const render = () => {
      const w = canvas.width;
      const h = canvas.height;
      const cx = w / 2;
      const cy = h / 2;

      ctx.fillStyle = '#010107';
      ctx.fillRect(0, 0, w, h);

      // Radial grid
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(cx, cy, 65, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(cx, cy, 35, 0, Math.PI * 2);
      ctx.stroke();

      // Axes
      ctx.beginPath();
      ctx.moveTo(cx, 10); ctx.lineTo(cx, h - 10);
      ctx.moveTo(10, cy); ctx.lineTo(w - 10, cy);
      ctx.stroke();

      constTime += 0.03;

      // Draw constellation points cluster
      ctx.fillStyle = '#C084FC';
      const numSymbols = 120;

      let centers: { x: number; y: number }[] = [];
      if (currentProfile.key === 'navtex' || currentProfile.key === 'rtty') {
        centers = [{ x: cx - 40, y: cy }, { x: cx + 40, y: cy }];
      } else if (currentProfile.key === 'rs41') {
        centers = [{ x: cx - 45, y: cy - 20 }, { x: cx + 45, y: cy + 20 }];
      } else {
        // Circle ring for FM subcarrier
        for (let a = 0; a < Math.PI * 2; a += Math.PI / 4) {
          centers.push({ x: cx + Math.cos(a) * 50, y: cy + Math.sin(a) * 50 });
        }
      }

      for (let i = 0; i < numSymbols; i++) {
        const c = centers[i % centers.length];
        const jitterX = Math.sin(i * 1.7 + constTime) * 3 + (Math.random() - 0.5) * 5;
        const jitterY = Math.cos(i * 2.3 + constTime) * 3 + (Math.random() - 0.5) * 5;

        ctx.beginPath();
        ctx.arc(c.x + jitterX, c.y + jitterY, 2, 0, Math.PI * 2);
        ctx.fill();
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [currentProfile]);

  return (
    <div style={{ background: '#020208', border: '1px solid var(--border-subtle)', borderRadius: '12px', padding: '16px', display: 'flex', alignItems: 'center', justifyContent: 'space-around', gap: '16px', flexWrap: 'wrap' }}>
      <div style={{ position: 'relative' }}>
        <canvas ref={canvasRef} width={180} height={180} style={{ width: '180px', height: '180px', display: 'block', borderRadius: '50%', border: '1px solid rgba(139, 92, 246, 0.3)', background: '#010107' }} />
      </div>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', color: 'var(--text-muted)', minWidth: '200px' }}>
        <div style={{ color: 'var(--purple-light)', fontWeight: 600, fontSize: '0.88rem', marginBottom: '8px' }}>Demodulator EVM & Phase</div>
        <div style={{ marginBottom: '4px' }}>RMS EVM: <span style={{ color: '#34D399', fontWeight: 600 }}>{currentProfile.evm}</span></div>
        <div style={{ marginBottom: '4px' }}>Peak EVM: <span style={{ color: '#38BDF8' }}>{currentProfile.evmPeak}</span></div>
        <div style={{ marginBottom: '4px' }}>Phase Jitter: <span style={{ color: '#34D399' }}>{currentProfile.phaseErr}</span></div>
        <div style={{ marginBottom: '4px' }}>Costas Lock: <span style={{ color: '#10B981', fontWeight: 600 }}>{currentProfile.lock}</span></div>
      </div>
    </div>
  );
}
