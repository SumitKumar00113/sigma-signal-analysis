'use client';

import { useState, useEffect, useCallback } from 'react';

export default function CarouselShowcase() {
  const [activeIndex, setActiveIndex] = useState<number>(1);

  const rotateCarousel = useCallback((dir: number) => {
    setActiveIndex((prev) => {
      let next = prev + dir;
      if (next < 0) next = 2;
      if (next > 2) next = 0;
      return next;
    });
  }, []);

  useEffect(() => {
    const timer = setInterval(() => {
      rotateCarousel(1);
    }, 6000);

    return () => clearInterval(timer);
  }, [rotateCarousel]);

  return (
    <section className="carousel-section">
      <div className="container">
        <div className="section-header-center">
          <div className="section-eyebrow">Workbench Suite</div>
          <h2 className="section-title">Complete Signal Processing Chain</h2>
          <p className="section-desc">Explore the specialized workbench engines powering SIGMA Signal Analysis.</p>
        </div>

        <div className="carousel-wrapper">
          {/* Card 0 */}
          <div className={`carousel-card-item ${activeIndex === 0 ? 'active' : ''}`}>
            <h3>18-Class Hybrid Classifier</h3>
            <p>38-feature gradient-boosted ML classifier combined with explainable decision rules.</p>
            {activeIndex === 0 && (
              <a href="https://github.com/SumitKumar00113/sigma-signal-analysis/releases/latest" target="_blank" rel="noreferrer" className="btn-nav-primary">
                Explore Classifier
              </a>
            )}
          </div>

          {/* Card 1 */}
          <div className={`carousel-card-item ${activeIndex === 1 ? 'active' : ''}`}>
            <h3>⚡ Auto-Analyse Engine</h3>
            <p>One-click background pipeline running parameter estimation, demodulation, interleaver scan, Viterbi/LDPC decoding, and sync discovery.</p>
            {activeIndex === 1 && (
              <a href="https://github.com/SumitKumar00113/sigma-signal-analysis/releases/latest" target="_blank" rel="noreferrer" className="btn-nav-primary">
                Explore Pipeline
              </a>
            )}
          </div>

          {/* Card 2 */}
          <div className={`carousel-card-item ${activeIndex === 2 ? 'active' : ''}`}>
            <h3>Blind Code Identification</h3>
            <p>Blind identification of convolutional codes, Reed-Solomon GF(2⁸), and sparse LDPC matrices.</p>
            {activeIndex === 2 && (
              <a href="https://github.com/SumitKumar00113/sigma-signal-analysis/releases/latest" target="_blank" rel="noreferrer" className="btn-nav-primary">
                Explore Code ID
              </a>
            )}
          </div>
        </div>

        <div className="carousel-dots">
          <button className="carousel-arrow-btn" onClick={() => rotateCarousel(-1)} aria-label="Previous Slide">
            ‹
          </button>
          {[0, 1, 2].map((idx) => (
            <button
              key={idx}
              className={`carousel-dot-btn ${activeIndex === idx ? 'active' : ''}`}
              onClick={() => setActiveIndex(idx)}
              aria-label={`Go to slide ${idx + 1}`}
            />
          ))}
          <button className="carousel-arrow-btn" onClick={() => rotateCarousel(1)} aria-label="Next Slide">
            ›
          </button>
        </div>
      </div>
    </section>
  );
}
