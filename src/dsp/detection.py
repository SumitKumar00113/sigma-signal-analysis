"""Signal region detection.

Identifies regions of interest containing potential signals based on energy
and spectral characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.enums import ConfidenceLevel, DetectionMethod
from src.core.models import SignalRegion
from src.dsp.spectral import compute_psd, detect_peaks, estimate_noise_floor


@dataclass
class DetectionConfig:
    """Configuration for signal detection."""

    method: DetectionMethod = DetectionMethod.SPECTRAL_PEAK
    threshold_db: float = 10.0
    min_bandwidth_hz: float = 1000.0
    merge_gap_hz: float = 5000.0
    fft_size: int = 4096


def detect_signal_regions(
    samples: np.ndarray,
    sample_rate: float,
    config: DetectionConfig | None = None,
) -> list[SignalRegion]:
    """Detect regions of interest within the signal.

    Currently relies on spectral peaks and maps them back to the entire
    time domain (since the input chunk is assumed to be stationary for now).
    Time-domain energy envelope detection will be added in future versions.
    """
    if config is None:
        config = DetectionConfig()

    regions: list[SignalRegion] = []
    
    if len(samples) < config.fft_size:
        return regions

    # Compute PSD to find signals
    freqs, psd_db = compute_psd(
        samples, sample_rate, fft_size=config.fft_size, window="hann"
    )
    
    noise_floor = estimate_noise_floor(psd_db)
    
    # Detect peaks
    peaks = detect_peaks(
        freqs, psd_db, noise_floor_db=noise_floor, threshold_db=config.threshold_db
    )
    
    if not peaks:
        return regions

    # Sort peaks by frequency to merge adjacent ones
    peaks.sort(key=lambda p: p.frequency_hz)
    
    # Merge peaks that are close to each other into wider signal regions
    current_start = peaks[0].frequency_hz - (peaks[0].bandwidth_hz / 2)
    current_end = peaks[0].frequency_hz + (peaks[0].bandwidth_hz / 2)
    current_snr = peaks[0].snr_db
    
    for i in range(1, len(peaks)):
        p = peaks[i]
        p_start = p.frequency_hz - (p.bandwidth_hz / 2)
        p_end = p.frequency_hz + (p.bandwidth_hz / 2)
        
        # If overlapping or close enough, merge
        if p_start <= current_end + config.merge_gap_hz:
            current_end = max(current_end, p_end)
            # Take max SNR for the merged region
            current_snr = max(current_snr, p.snr_db)
        else:
            # Save the previous region and start a new one
            # We map frequency regions back to a "SignalRegion" which in our core models
            # tracks time (start/end sample). For now, a detected spectral peak
            # implies the signal is present for the whole sample duration.
            bw = current_end - current_start
            if bw >= config.min_bandwidth_hz:
                regions.append(
                    SignalRegion(
                        start_sample=0,
                        end_sample=len(samples),
                        snr_db=current_snr,
                        confidence=0.8 if current_snr > 15 else 0.5
                    )
                )
            
            current_start = p_start
            current_end = p_end
            current_snr = p.snr_db

    # Don't forget the last region
    bw = current_end - current_start
    if bw >= config.min_bandwidth_hz:
        regions.append(
            SignalRegion(
                start_sample=0,
                end_sample=len(samples),
                snr_db=current_snr,
                confidence=0.8 if current_snr > 15 else 0.5
            )
        )

    return regions
