'use client';

import { useEffect, useRef } from 'react';
import { FileProfile } from '@/data/fileProfiles';

interface SpectrumCanvasProps {
  currentProfile: FileProfile;
}

export default function SpectrumCanvas({ currentProfile }: SpectrumCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animTime = 0;
    let animationFrameId: number;

    const render = () => {
      const w = canvas.width;
      const h = canvas.height;

      ctx.fillStyle = '#020208';
      ctx.fillRect(0, 0, w, h);

      // Grid lines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      for (let y = 30; y < h; y += 30) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }
      for (let x = 100; x < w; x += 100) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }

      animTime += 0.05;

      // Draw FFT line
      ctx.beginPath();
      const numPoints = 120;
      for (let i = 0; i <= numPoints; i++) {
        const x = (i / numPoints) * w;
        const normX = i / numPoints;

        const noise = Math.sin(i * 0.8 + animTime) * 3 + (Math.random() - 0.5) * 4;
        let signalAmp = 0;

        currentProfile.peaks.forEach((peakX) => {
          const dist = Math.abs(normX - peakX);
          if (dist < 0.08) {
            signalAmp += Math.exp(-Math.pow(dist / 0.025, 2)) * 95;
          }
        });

        const y = h - 20 - signalAmp - noise;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }

      const grad = ctx.createLinearGradient(0, 0, w, 0);
      grad.addColorStop(0, '#38BDF8');
      grad.addColorStop(0.5, '#8B5CF6');
      grad.addColorStop(1, '#E9D5FF');

      ctx.strokeStyle = grad;
      ctx.lineWidth = 2.5;
      ctx.stroke();

      // Highlight peak box
      if (currentProfile.peaks.length > 0) {
        const peakX = currentProfile.peaks[0] * w;
        ctx.fillStyle = 'rgba(124, 58, 237, 0.12)';
        ctx.strokeStyle = 'rgba(139, 92, 246, 0.4)';
        ctx.setLineDash([4, 4]);
        ctx.fillRect(peakX - 35, 10, 70, h - 25);
        ctx.strokeRect(peakX - 35, 10, 70, h - 25);
        ctx.setLineDash([]);
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [currentProfile]);

  return (
    <div style={{ background: '#020208', border: '1px solid var(--border-subtle)', borderRadius: '12px', padding: '12px', position: 'relative', overflow: 'hidden' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
        <span>FFT REAL-TIME BASEBAND SCAN (BW: 48 kHz)</span>
        <span style={{ color: 'var(--cyan-subtle)', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#10B981', display: 'inline-block' }} />
          LIVE FFT • 60 FPS
        </span>
      </div>
      <canvas ref={canvasRef} width={600} height={150} style={{ width: '100%', height: '150px', display: 'block', borderRadius: '6px' }} />
    </div>
  );
}
