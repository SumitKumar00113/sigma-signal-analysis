'use client';

export default function GemGraphic() {
  return (
    <div className="gem-wrapper">
      <svg className="gem-svg-icon" viewBox="0 0 100 100">
        <polygon points="50,5 90,30 90,70 50,95 10,70 10,30" fill="url(#gemGrad1)" stroke="rgba(255,255,255,0.4)" strokeWidth="1.5" />
        <polygon points="50,5 50,95 90,30" fill="url(#gemGrad2)" opacity="0.7" />
        <polygon points="50,5 10,30 50,95" fill="url(#gemGrad3)" opacity="0.6" />
        <defs>
          <linearGradient id="gemGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#C084FC" />
            <stop offset="50%" stopColor="#4F46E5" />
            <stop offset="100%" stopColor="#8B5CF6" />
          </linearGradient>
          <linearGradient id="gemGrad2" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#FFFFFF" />
            <stop offset="100%" stopColor="#7C3AED" />
          </linearGradient>
          <linearGradient id="gemGrad3" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#38BDF8" />
            <stop offset="100%" stopColor="#4C1D95" />
          </linearGradient>
        </defs>
      </svg>
    </div>
  );
}
