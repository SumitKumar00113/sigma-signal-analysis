"""Feature-based modulation classification.

A two-stage decision tree, deliberately simple so its verdicts can be
explained to an analyst:

1. **Constant-envelope test.**  FSK/CPM signals have a flat envelope and a
   multi-modal instantaneous-frequency histogram.  Pulse-shaped linear
   modulations have a fluctuating envelope and a heavy-tailed, unimodal
   instantaneous frequency.

2. **Cumulant test** (linear modulations only).  After timing recovery and
   carrier-frequency removal, the normalised fourth-order cumulants
   separate the linear families:

   ============  ======  ======  ======
   Modulation    |C20|   |C40|   C42
   ============  ======  ======  ======
   BPSK            1       2      -2
   QPSK            0       1      -1
   8-PSK           0       0      -1
   16-QAM          0     0.68   -0.68
   64-QAM          0     0.62   -0.62
   Gaussian        0       0       0
   ============  ======  ======  ======

   Noise shrinks each cumulant by a known factor of the SNR, which is
   corrected before thresholding.

Every result carries a ranked candidate list and the raw feature values
so the GUI can show *why* a decision was made.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sp_stats

from src.core.enums import ModulationType
from src.dsp.measurements import (
    bandlimit_to_signal,
    compute_instantaneous_frequency,
    estimate_frequency_offset,
    estimate_snr_inband,
)
from src.dsp.preprocessing import translate_frequency
from src.dsp.sync import (
    apply_cfo,
    estimate_cfo_mpower,
    gardner_timing_recovery,
    matched_filter,
    moving_average,
    resample_to_sps,
)

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ClassificationResult:
    """Ranked modulation hypotheses plus the features that produced them."""

    modulation: ModulationType = ModulationType.UNKNOWN
    confidence: float = 0.0
    candidates: list[tuple[ModulationType, float]] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    symbols: np.ndarray | None = None      # timing-recovered, CFO-corrected symbols


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def envelope_features(samples: np.ndarray) -> dict[str, float]:
    """Envelope coefficient of variation and kurtosis."""
    env = np.abs(samples)
    mean = float(np.mean(env))
    if mean <= 0:
        return {"env_cv": 0.0, "env_kurtosis": 0.0}
    return {
        "env_cv": float(np.std(env) / mean),
        "env_kurtosis": float(sp_stats.kurtosis(env, fisher=False)),
    }


def instfreq_features(
    samples: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
) -> dict[str, float]:
    """Instantaneous-frequency statistics after symbol-length smoothing.

    ``if_kurtosis`` < ~2.2 indicates a multi-level (FSK-like) distribution;
    a Gaussian is 3 and PSK transition spikes push it well above 3.
    ``if_modes`` is the number of clear peaks in the histogram.
    """
    inst = compute_instantaneous_frequency(samples, sample_rate)
    sps = sample_rate / symbol_rate if symbol_rate > 0 else 8.0
    smooth = moving_average(inst, max(1, int(sps * 0.6)))
    smooth = smooth[int(sps):-int(sps)] if len(smooth) > 4 * sps else smooth
    if len(smooth) < 32:
        return {"if_kurtosis": 3.0, "if_modes": 1.0, "if_spread_hz": 0.0}

    kurt = float(sp_stats.kurtosis(smooth, fisher=False))

    # Count histogram modes
    hist, edges = np.histogram(smooth, bins=64)
    hist = np.convolve(hist, np.ones(3) / 3.0, mode="same")
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(hist, prominence=0.15 * np.max(hist), distance=4)

    return {
        "if_kurtosis": kurt,
        "if_modes": float(len(peaks)),
        "if_spread_hz": float(np.std(smooth)),
    }


def cumulant_features(symbols: np.ndarray, snr_db: float | None = None) -> dict[str, float]:
    """Normalised second/fourth-order cumulants of unit-power symbols.

    If *snr_db* is given the cumulants are corrected for the shrinkage
    caused by additive Gaussian noise.
    """
    s = np.asarray(symbols, dtype=np.complex128)
    if len(s) == 0:
        return {"c20": 0.0, "c40": 0.0, "c42": 0.0, "c21": 0.0}
    s = s / (np.sqrt(np.mean(np.abs(s) ** 2)) + 1e-12)

    c20 = np.mean(s**2)
    c21 = np.mean(np.abs(s) ** 2)
    c40 = np.mean(s**4) - 3 * c20**2
    c42 = np.mean(np.abs(s) ** 4) - np.abs(c20) ** 2 - 2 * c21**2

    if snr_db is not None:
        snr = 10 ** (snr_db / 10.0)
        rho = snr / (1.0 + snr)             # signal fraction of total power
        rho = max(rho, 0.2)
        c20 = c20 / rho
        c40 = c40 / rho**2
        c42 = c42 / rho**2

    return {
        "c20": float(np.abs(c20)),
        "c40": float(np.abs(c40)),
        "c42": float(np.real(c42)),
        "c21": float(c21),
    }


def amplitude_level_count(symbols: np.ndarray, max_levels: int = 9) -> int:
    """Estimate how many distinct amplitude rings the constellation has."""
    amp = np.abs(symbols)
    amp = amp / (np.mean(amp) + 1e-12)
    hist, _ = np.histogram(amp, bins=48, range=(0, 2.0))
    hist = np.convolve(hist, np.ones(3) / 3.0, mode="same")
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(hist, prominence=0.12 * np.max(hist), distance=3)
    return int(min(max(len(peaks), 1), max_levels))


def _fsk_order_by_clustering(
    samples: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    prior_order: int,
) -> int:
    """Decide between 2- and 4-FSK by comparing k-means fits.

    Four clusters are accepted only if they cut the within-cluster
    variance by a large factor *and* the centroids are roughly evenly
    spaced; otherwise a two-level fit is returned.
    """
    inst = compute_instantaneous_frequency(samples, sample_rate)
    sps = sample_rate / symbol_rate if symbol_rate > 0 else 8.0
    smooth = moving_average(inst, max(1, int(sps * 0.8)))
    smooth = smooth[int(sps):-int(sps)] if len(smooth) > 4 * sps else smooth
    # Keep only "settled" samples: discard the transition regions by
    # dropping values whose slope is large
    slope = np.abs(np.gradient(smooth))
    keep = slope < np.percentile(slope, 60)
    vals = smooth[keep]
    if len(vals) < 64:
        return prior_order

    def _fit(k: int) -> tuple[np.ndarray, float]:
        lo, hi = np.percentile(vals, 3), np.percentile(vals, 97)
        c = np.linspace(lo, hi, k)
        for _ in range(25):
            a = np.argmin(np.abs(vals[:, None] - c[None, :]), axis=1)
            new = np.array([vals[a == i].mean() if np.any(a == i) else c[i] for i in range(k)])
            if np.allclose(new, c):
                break
            c = new
        a = np.argmin(np.abs(vals[:, None] - c[None, :]), axis=1)
        wcss = float(np.sum((vals - c[a]) ** 2))
        return np.sort(c), wcss

    c2, w2 = _fit(2)
    c4, w4 = _fit(4)
    if w2 <= 0:
        return 2
    gaps = np.diff(c4)
    even = np.min(gaps) > 0.4 * np.max(gaps) if np.max(gaps) > 0 else False
    if w4 < 0.25 * w2 and even:
        return 4
    return 2


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def _linear_symbols(
    samples: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    cfo_hz: float,
) -> np.ndarray:
    """Timing-recovered, CFO-corrected symbols for cumulant analysis."""
    x = translate_frequency(samples, sample_rate, -cfo_hz)
    x, sps = resample_to_sps(x, sample_rate, symbol_rate, target_sps=8)
    x = matched_filter(x, sps)
    timing = gardner_timing_recovery(x, sps)
    syms = timing.symbols[min(len(timing.symbols) // 10, 200):]
    if len(syms) < 32:
        return syms
    syms = syms / (np.sqrt(np.mean(np.abs(syms) ** 2)) + 1e-12)

    # Residual CFO: try the 4th power; if no clear line, the 8th (8-PSK)
    cfo4 = estimate_cfo_mpower(syms, 4)
    cand4 = apply_cfo(syms, cfo4)
    line4 = np.abs(np.mean(cand4**4))
    if line4 < 0.25:
        cfo8 = estimate_cfo_mpower(syms, 8)
        cand8 = apply_cfo(syms, cfo8)
        if np.abs(np.mean(cand8**8)) > line4:
            return cand8
    return cand4


def classify_modulation(
    samples: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    snr_db: float | None = None,
    cfo_hz: float | None = None,
) -> ClassificationResult:
    """Classify the modulation of *samples*.

    Parameters
    ----------
    symbol_rate : estimated symbol rate (needed for timing recovery).
    snr_db : optional SNR estimate, used to de-bias the cumulants.
    cfo_hz : optional carrier offset estimate; computed if omitted.
    """
    result = ClassificationResult()
    if len(samples) < 1024 or symbol_rate <= 0 or sample_rate <= 0:
        result.evidence.append("Insufficient samples or unknown symbol rate.")
        return result

    if cfo_hz is None:
        cfo_hz = estimate_frequency_offset(samples, sample_rate)

    # Work in-band so out-of-band noise does not dominate the features
    band = bandlimit_to_signal(samples, sample_rate)
    inband_snr, _ = estimate_snr_inband(samples, sample_rate)
    if snr_db is None:
        snr_db = inband_snr

    feats: dict[str, float] = {}
    feats.update(envelope_features(band))
    feats.update(instfreq_features(translate_frequency(band, sample_rate, -cfo_hz),
                                   sample_rate, symbol_rate))
    feats["snr_inband_db"] = float(inband_snr)

    # ---- Stage 1: constant envelope? ----------------------------------
    env_cv = feats["env_cv"]
    if_kurt = feats["if_kurtosis"]
    if_modes = feats["if_modes"]

    fsk_score = 0.0
    if env_cv < 0.22:
        fsk_score += 0.5 * (1.0 - env_cv / 0.22)
    if if_kurt < 2.4:
        fsk_score += 0.4 * (1.0 - max(if_kurt - 1.0, 0.0) / 1.4)
    if if_modes >= 2:
        fsk_score += 0.2
    fsk_score = min(fsk_score, 1.0)

    if fsk_score >= 0.55:
        # A two-level distribution has kurtosis ≈ 1.0; four equiprobable
        # levels ≈ 1.64.  Noise pushes both upward, so split at 1.45 and
        # confirm with a cluster-count test on the smoothed frequency.
        order = 4 if if_kurt > 1.45 else 2
        order = _fsk_order_by_clustering(
            translate_frequency(band, sample_rate, -cfo_hz), sample_rate, symbol_rate, order
        )
        feats["fsk_order"] = float(order)
        mod = ModulationType.FSK4 if order == 4 else ModulationType.FSK2
        result.modulation = mod
        result.confidence = fsk_score
        result.candidates = [(mod, fsk_score),
                             (ModulationType.FSK4 if order == 2 else ModulationType.FSK2,
                              max(0.0, fsk_score - 0.5)),
                             (ModulationType.GMSK, max(0.0, fsk_score - 0.6))]
        result.evidence.append(
            f"Constant envelope (CV {env_cv:.2f}) with {int(if_modes)}-modal instantaneous "
            f"frequency (kurtosis {if_kurt:.2f}) → frequency-shift keying."
        )
        result.features = feats
        return result

    # ---- Stage 2: linear modulation cumulants ---------------------------
    syms = _linear_symbols(samples, sample_rate, symbol_rate, cfo_hz)
    if len(syms) < 64:
        result.evidence.append("Timing recovery produced too few symbols for classification.")
        result.features = feats
        return result

    # Cumulants use the *symbol* SNR which is higher than the in-band SNR
    # by the matched-filter gain; approximate with in-band SNR + 3 dB
    cum = cumulant_features(syms, snr_db=inband_snr + 3.0)
    feats.update(cum)
    rings = amplitude_level_count(syms)
    feats["amp_rings"] = float(rings)

    c20, c40, c42 = cum["c20"], cum["c40"], cum["c42"]

    scores: dict[ModulationType, float] = {}
    # Distance-based scoring against theoretical cumulant vectors
    theory = {
        ModulationType.BPSK: (1.0, 2.0, -2.0),
        ModulationType.QPSK: (0.0, 1.0, -1.0),
        ModulationType.PSK8: (0.0, 0.0, -1.0),
        ModulationType.QAM16: (0.0, 0.68, -0.68),
        ModulationType.QAM64: (0.0, 0.62, -0.62),
    }
    for mod, (t20, t40, t42) in theory.items():
        d = np.sqrt((c20 - t20) ** 2 + (c40 - t40) ** 2 + (c42 - t42) ** 2)
        scores[mod] = float(np.exp(-(d / 0.35) ** 2))

    # Ring count disambiguates QAM orders and PSK vs QAM
    if rings <= 1:
        scores[ModulationType.QAM16] *= 0.5
        scores[ModulationType.QAM64] *= 0.3
    elif rings == 3 or rings == 2:
        scores[ModulationType.QAM16] *= 1.3
        scores[ModulationType.QAM64] *= 0.7
    elif rings >= 4:
        scores[ModulationType.QAM64] *= 1.3
        scores[ModulationType.QAM16] *= 0.8
        scores[ModulationType.QPSK] *= 0.5
        scores[ModulationType.BPSK] *= 0.5

    total = sum(scores.values())
    if total <= 1e-9:
        result.evidence.append(
            f"Cumulants (|C20|={c20:.2f}, |C40|={c40:.2f}, C42={c42:.2f}) match no known "
            "linear constellation; possibly OFDM, noise, or an unsupported scheme."
        )
        result.candidates = [(ModulationType.OFDM, 0.3)] if abs(c42) < 0.3 else []
        result.features = feats
        result.symbols = syms
        return result

    ranked = sorted(((m, s / total) for m, s in scores.items()), key=lambda x: x[1], reverse=True)
    best_mod, best_p = ranked[0]
    # Absolute goodness (not just relative) must also be reasonable
    absolute = scores[best_mod]
    confidence = float(min(1.0, best_p * min(1.0, absolute / 0.5)))

    result.modulation = best_mod
    result.confidence = confidence
    result.candidates = [(m, float(p)) for m, p in ranked]
    result.evidence.append(
        f"Fluctuating envelope (CV {env_cv:.2f}) → linear modulation. "
        f"Cumulants |C20|={c20:.2f}, |C40|={c40:.2f}, C42={c42:.2f}; "
        f"{rings} amplitude ring(s)."
    )
    result.features = feats
    result.symbols = syms
    return result
