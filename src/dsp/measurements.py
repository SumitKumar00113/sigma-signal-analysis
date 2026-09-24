"""Signal measurement utilities.

Measures specific properties of a detected signal like SNR,
frequency offset, and symbol rate.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

from src.dsp.spectral import compute_psd, estimate_noise_floor, live_bins

# ---------------------------------------------------------------------------
# SNR
# ---------------------------------------------------------------------------


def estimate_snr_m2m4(samples: np.ndarray) -> float:
    """Estimate SNR with the M2M4 (2nd & 4th moment) estimator.

    Accurate only for constant-envelope modulations (FSK, unshaped PSK).
    For pulse-shaped PSK/QAM it reads several dB low; prefer
    :func:`estimate_snr_psd` in the general case.
    """
    if len(samples) == 0:
        return 0.0

    m2 = np.mean(np.abs(samples) ** 2)
    m4 = np.mean(np.abs(samples) ** 4)

    if m2 == 0 or m4 >= 2 * (m2 ** 2):
        return 0.0

    sqrt_term = np.sqrt(max(0, 2 * m2**2 - m4))
    noise_power = m2 - sqrt_term
    if noise_power <= 0:
        return 100.0

    snr_linear = sqrt_term / noise_power
    if snr_linear <= 0:
        return 0.0
    return float(10 * np.log10(snr_linear))


def estimate_snr_psd(
    samples: np.ndarray,
    sample_rate: float,
    fft_size: int = 1024,
    noise_percentile: float = 20.0,
) -> float:
    """Estimate full-band SNR from the power spectral density.

    The noise floor is taken as a low percentile of the PSD (bins the signal
    does not occupy).  Total noise power is the floor integrated across the
    whole band; signal power is whatever remains above it.  This matches the
    conventional definition ``P_signal / P_noise`` used when AWGN is added to
    a recording, and is independent of the modulation's envelope statistics.
    """
    if len(samples) < 64:
        return 0.0
    fft_size = min(fft_size, len(samples))

    _, psd_db = compute_psd(samples, sample_rate, fft_size=fft_size)
    psd_lin = 10.0 ** (psd_db.astype(np.float64) / 10.0)

    noise_floor_db = estimate_noise_floor(psd_db, percentile=noise_percentile)
    noise_bin = 10.0 ** (noise_floor_db / 10.0)
    # Percentile of a chi-squared-ish distribution underestimates the mean.
    # For Welch with many averages the per-bin distribution is tight; the
    # 20th percentile is ~0.9 of the mean for typical segment counts.
    noise_bin *= 1.1

    total = float(np.sum(psd_lin))
    # Dead bins (empty half of an analytic signal, receiver stop-band) hold
    # no noise, so integrate the floor over the live bins only
    noise_total = noise_bin * int(live_bins(psd_db).sum())
    signal_total = total - noise_total

    if noise_total <= 0:
        return 100.0
    if signal_total <= 0:
        return -30.0
    return float(10 * np.log10(signal_total / noise_total))


def estimate_snr_inband(
    samples: np.ndarray,
    sample_rate: float,
    fft_size: int = 1024,
    threshold_db: float = 3.0,
) -> tuple[float, float]:
    """Estimate in-band SNR and occupied bandwidth.

    Returns ``(snr_db, bandwidth_hz)`` where the SNR is measured only over
    the bins the signal occupies (those above the noise floor by
    *threshold_db*).
    """
    if len(samples) < 64:
        return 0.0, 0.0
    fft_size = min(fft_size, len(samples))
    freqs, psd_db = compute_psd(samples, sample_rate, fft_size=fft_size)
    psd_lin = 10.0 ** (psd_db.astype(np.float64) / 10.0)
    noise_floor_db = estimate_noise_floor(psd_db, percentile=20.0)
    noise_bin = 10.0 ** (noise_floor_db / 10.0) * 1.1

    mask = psd_db > noise_floor_db + threshold_db
    n_sig = int(mask.sum())
    if n_sig == 0:
        return -30.0, 0.0
    sig = float(np.sum(psd_lin[mask])) - noise_bin * n_sig
    noise = noise_bin * n_sig
    bw = n_sig * float(abs(freqs[1] - freqs[0]))
    if sig <= 0:
        return -30.0, bw
    return float(10 * np.log10(sig / noise)), bw


def estimate_snr(samples: np.ndarray, sample_rate: float | None = None) -> float:
    """Estimate SNR in dB.

    With a sample rate the PSD-based estimator is used (accurate for all
    modulations).  Without one the M2M4 estimator is used as a fallback.
    """
    if sample_rate is not None and sample_rate > 0:
        return estimate_snr_psd(samples, sample_rate)
    return estimate_snr_m2m4(samples)


# ---------------------------------------------------------------------------
# Frequency offset
# ---------------------------------------------------------------------------


def estimate_frequency_offset(
    samples: np.ndarray,
    sample_rate: float,
    fft_size: int = 4096
) -> float:
    """Estimate the frequency offset from DC (0 Hz).

    Finds the power-weighted spectral centroid of the region above the
    noise floor.  Works for any modulation whose spectrum is symmetric
    about its carrier (PSK, QAM, FSK).
    """
    if len(samples) < fft_size:
        fft_size = len(samples)

    if fft_size < 16:
        return 0.0

    window = sp_signal.windows.hann(fft_size)
    num_blocks = max(1, len(samples) // fft_size)

    psd_sum = np.zeros(fft_size)
    for i in range(num_blocks):
        chunk = samples[i*fft_size : (i+1)*fft_size]
        if len(chunk) < fft_size:
            chunk = np.pad(chunk, (0, fft_size - len(chunk)))
        fft_out = np.fft.fftshift(np.fft.fft(chunk * window))
        psd_sum += np.abs(fft_out) ** 2

    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, 1 / sample_rate))

    # Noise floor from low percentile; keep only bins clearly above it
    psd_db = 10.0 * np.log10(psd_sum + 1e-300)
    noise = np.percentile(psd_sum[live_bins(psd_db)], 20.0) * 1.1
    above = psd_sum - noise
    above[above < 0] = 0.0
    thresh = 0.1 * np.max(above)
    weights = np.where(above > thresh, above, 0.0)

    if np.sum(weights) == 0:
        return float(freqs[int(np.argmax(psd_sum))])

    # Split the occupied bins into contiguous segments and keep every
    # segment carrying a meaningful share of the strongest one.  A lone
    # emitter gives one segment; FSK mark/space tones give two (or more)
    # with a null between them, and the carrier sits at their centroid.
    occupied = np.flatnonzero(weights)
    breaks = np.flatnonzero(np.diff(occupied) > 1)
    segments = np.split(occupied, breaks + 1)
    seg_power = np.array([weights[seg].sum() for seg in segments])
    keep = [seg for seg, pw in zip(segments, seg_power, strict=True)
            if pw >= 0.3 * seg_power.max()]
    lo = int(min(seg[0] for seg in keep))
    hi = int(max(seg[-1] for seg in keep))

    w = weights[lo:hi + 1]
    f = freqs[lo:hi + 1]
    return float(np.sum(f * w) / np.sum(w))


# ---------------------------------------------------------------------------
# Symbol rate
# ---------------------------------------------------------------------------


def _spectral_lines(
    feature: np.ndarray,
    sample_rate: float,
    min_rate_hz: float = 100.0,
) -> list[tuple[float, float]]:
    """Find periodic lines in a real-valued feature signal.

    Returns ``(frequency_hz, confidence)`` pairs sorted by confidence.
    Confidence is the line's prominence relative to the strongest line,
    scaled by how far it stands above the local spectrum median.
    """
    feature = np.asarray(feature, dtype=np.float64)
    feature = feature - np.mean(feature)
    if len(feature) < 1024:
        return []

    fft_size = min(1 << 18, 1 << int(np.log2(len(feature))))
    window = sp_signal.windows.blackman(fft_size)

    # Average across blocks for a cleaner line spectrum
    n_blocks = max(1, len(feature) // fft_size)
    mag = np.zeros(fft_size // 2)
    for b in range(n_blocks):
        chunk = feature[b * fft_size:(b + 1) * fft_size]
        mag += np.abs(np.fft.fft(chunk * window))[: fft_size // 2]
    freqs = np.fft.fftfreq(fft_size, 1 / sample_rate)[: fft_size // 2]

    # Ignore DC and rates below the minimum
    min_idx = int(np.searchsorted(freqs, min_rate_hz))
    mag[:min_idx] = 0.0
    if np.max(mag) == 0:
        return []

    # Normalise against a smoothed background so broad humps don't win
    background = sp_signal.medfilt(mag, kernel_size=min(101, (len(mag) // 2) * 2 - 1))
    excess = mag - background
    excess[excess < 0] = 0.0
    if np.max(excess) == 0:
        return []

    peaks, props = sp_signal.find_peaks(
        excess, distance=max(1, fft_size // 200), prominence=np.max(excess) * 0.15
    )
    if len(peaks) == 0:
        return []

    prom = props["prominences"]
    strength = prom / np.max(prom)
    # Absolute significance: how many standard deviations of the
    # background residual the line rises.  Noise alone tops out around
    # 5–8σ for tens of thousands of bins; a real symbol-rate line is
    # typically 30σ or more.
    resid = excess[excess > 0]
    sigma = float(np.std(resid)) if len(resid) > 10 else 1.0
    z = excess[peaks] / max(sigma, 1e-12)
    significance = np.clip((z - 8.0) / 40.0, 0.0, 1.0)
    conf = np.clip(strength * significance, 0.0, 1.0)

    out = [(float(freqs[p]), float(c)) for p, c in zip(peaks, conf, strict=True)]

    # Prefer a fundamental over its harmonics: if a candidate at f has a
    # comparably strong partner at f/2 the partner is the symbol rate and
    # this one is the second harmonic of the transition pulse train.
    boosted: list[tuple[float, float]] = []
    for f, c in out:
        for f2, c2 in out:
            if abs(f2 - f / 2.0) / max(f, 1.0) < 0.01 and c2 >= 0.4 * c:
                c = c * 0.6
                break
        boosted.append((f, c))
    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted


def estimate_symbol_rate_envelope(
    samples: np.ndarray,
    sample_rate: float,
) -> list[tuple[float, float]]:
    """Symbol rate candidates from the amplitude envelope.

    Works for pulse-shaped linear modulations (PSK, QAM, ASK) whose
    envelope dips at symbol transitions.  Returns nothing useful for
    constant-envelope signals (FSK, GMSK).
    """
    if len(samples) < 1024:
        return []
    # Out-of-band noise beats against the signal (and against the carrier
    # of ASK/AM) in |x|²; confine it to the occupied band first
    return _spectral_lines(np.abs(bandlimit_to_signal(samples, sample_rate)) ** 2, sample_rate)


def estimate_symbol_rate_instfreq(
    samples: np.ndarray,
    sample_rate: float,
) -> list[tuple[float, float]]:
    """Symbol rate candidates from instantaneous-frequency transitions.

    Suited to FSK/CPM: the instantaneous frequency is piecewise constant
    and jumps at symbol boundaries, so ``|d/dt f_inst|`` is a train of
    pulses spaced at the symbol period.  Also responds to PSK phase jumps.
    """
    if len(samples) < 1024:
        return []
    # Confine the discriminator to the occupied band.  At high oversampling
    # the out-of-band noise otherwise swamps the transition pulses.
    filtered = bandlimit_to_signal(samples, sample_rate)
    inst_freq = compute_instantaneous_frequency(filtered, sample_rate)
    feature = np.abs(np.diff(inst_freq))
    return _spectral_lines(feature, sample_rate)


def estimate_symbol_rate_transitions(
    samples: np.ndarray,
    sample_rate: float,
) -> list[tuple[float, float]]:
    """Symbol rate candidates from envelope *transitions*, ``(d|x|/dt)²``.

    For carrier-bearing amplitude keying (OOK/ASK) the envelope is the data
    itself, whose strong baseband spectrum buries the symbol-rate line of
    ``|x|²``.  Differentiating suppresses that hump while the transitions
    at symbol boundaries keep their periodicity.
    """
    if len(samples) < 1024:
        return []
    env = np.abs(bandlimit_to_signal(samples, sample_rate))
    return _spectral_lines(np.diff(env) ** 2, sample_rate)


def mpower_carrier(
    samples: np.ndarray,
    sample_rate: float,
    order: int = 4,
) -> tuple[float, float, float]:
    """Carrier line of ``x^order`` (M-PSK / OQPSK carrier recovery).

    Returns ``(carrier_hz, carrier_phase_rad, line_ratio)`` where the ratio
    compares the peak with its local spectral neighbourhood; the phase is
    ambiguous by multiples of 2π/order.  A ratio well above ~30 means the
    line is real.
    """
    x = np.asarray(samples, dtype=np.complex128)
    n = len(x)
    if n < 256:
        return 0.0, 0.0, 0.0
    spec = np.fft.fft(x ** order * np.hanning(n))
    mag = np.abs(spec)
    k = int(np.argmax(mag))
    # Parabolic interpolation of the peak for a sub-bin frequency
    a, b, c = mag[k - 1], mag[k], mag[(k + 1) % n]
    delta = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    f_line = (np.fft.fftfreq(n, 1.0 / sample_rate)[k] + delta * sample_rate / n)
    t = np.arange(n) / sample_rate
    phase = float(np.angle(np.sum(x ** order * np.exp(-2j * np.pi * f_line * t))))
    # A spectral *line* must stand out from its own neighbourhood, not
    # just from the (noise-dominated) median of the whole band
    w = max(8, min(n // 50, 400))
    idx = np.arange(k - w, k + w + 1) % n
    neighbours = mag[idx[np.abs(np.arange(-w, w + 1)) > 3]]
    ratio = float(b / (np.median(neighbours) + 1e-300))
    return float(f_line / order), phase / order, ratio


def mpower_line_pair(
    samples: np.ndarray,
    sample_rate: float,
    spacing_hz: float,
    order: int = 4,
) -> tuple[float, float, float]:
    """Look for two equal ``x^order`` lines *spacing_hz* apart.

    π/4-DQPSK alternates between two QPSK grids 45° apart, so its 4th
    power flips sign every symbol: instead of one line at 4·f₀ it shows a
    pair at 4·f₀ ± Rs/2 (spacing Rs).  Returns ``(carrier_hz,
    partner_strength, line_ratio)`` where the carrier is the pair's
    midpoint / *order* and the strength is partner/peak magnitude.

    QPSK's 4th power also has (weaker) lines Rs either side of its main
    line, but *symmetrically*; a π/4-DQPSK peak has one equal partner and a
    much weaker line on the other side.  The strength returned is therefore
    0 unless the stronger side is at least 1.5× the weaker one.
    """
    x = np.asarray(samples, dtype=np.complex128)
    n = len(x)
    if n < 256 or spacing_hz <= 0:
        return 0.0, 0.0, 0.0
    mag = np.abs(np.fft.fft(x ** order * np.hanning(n)))
    freqs = np.fft.fftfreq(n, 1.0 / sample_rate)
    k = int(np.argmax(mag))
    w = max(8, min(n // 50, 400))
    idx = np.arange(k - w, k + w + 1) % n
    ratio = float(mag[k] / (np.median(mag[idx[np.abs(np.arange(-w, w + 1)) > 3]]) + 1e-300))
    step = spacing_hz * n / sample_rate
    sides = []
    for sign in (1, -1):
        centre = int(round(k + sign * step)) % n
        near = np.arange(centre - 2, centre + 3) % n
        j = near[int(np.argmax(mag[near]))]
        sides.append((float(mag[j] / mag[k]), float(freqs[j])))
    sides.sort(reverse=True)
    (strong, f_partner), (weak, _) = sides
    carrier = (freqs[k] + f_partner) / 2.0 / order
    strength = strong if strong >= 1.5 * weak else 0.0
    return float(carrier), strength, ratio


def estimate_symbol_rate_quadrature(
    samples: np.ndarray,
    sample_rate: float,
    min_line_ratio: float = 30.0,
) -> list[tuple[float, float]]:
    """Symbol rate of QPSK-family signals from a single rail.

    OQPSK's envelope barely varies, so |x|² shows almost no symbol-rate
    line.  After carrier recovery from the 4th-power line, the real part
    of the de-rotated signal is one rail (I or Q) – a plain PAM signal
    whose square has a strong line at the symbol rate.
    """
    if len(samples) < 1024:
        return []
    band = bandlimit_to_signal(samples, sample_rate)
    # A plain carrier line (ASK, AM) also shows up in x⁴; not this case
    if mpower_carrier(band, sample_rate, 1)[2] >= min_line_ratio:
        return []
    # MSK/GMSK are OQPSK-like with two-symbol pulses per rail, so their
    # rails would report half the symbol rate; they have a flat envelope
    env = np.abs(band)
    if np.std(env) < 0.18 * np.mean(env):
        return []
    fc, phase, ratio = mpower_carrier(band, sample_rate, 4)
    if ratio < min_line_ratio:
        return []
    # QPSK points sit at ±45° (x⁴ ≈ −1): rotate them there, which puts
    # the I and Q rails on the real and imaginary axes
    phase -= np.pi / 4
    t = np.arange(len(band)) / sample_rate
    rail = np.real(band * np.exp(-1j * (2 * np.pi * fc * t + phase)))
    return _spectral_lines(rail ** 2, sample_rate)


def bandlimit_to_signal(
    samples: np.ndarray,
    sample_rate: float,
    margin: float = 1.5,
) -> np.ndarray:
    """Low-pass *samples* to the occupied bandwidth (centred on its offset).

    The signal is first shifted to DC using the centroid estimate, filtered
    with a FIR whose cutoff is ``margin`` × half the occupied bandwidth, then
    shifted back.  If the occupied bandwidth is nearly the whole band the
    samples are returned unchanged.
    """
    from src.dsp.preprocessing import apply_lowpass, translate_frequency

    _, bw = estimate_snr_inband(samples, sample_rate)
    if bw <= 0 or bw * margin >= 0.9 * sample_rate:
        return samples
    offset = estimate_frequency_offset(samples, sample_rate)
    cutoff = max(bw * margin / 2.0, sample_rate / 500.0)
    centred = translate_frequency(samples, sample_rate, -offset)
    lp = apply_lowpass(centred, sample_rate, cutoff, order=128)
    return translate_frequency(lp, sample_rate, offset)


def estimate_symbol_rate_candidates(
    samples: np.ndarray,
    sample_rate: float,
    method: str = "auto",
) -> list[tuple[float, float]]:
    """Estimate potential symbol rates.

    Parameters
    ----------
    method:
        ``"envelope"`` (PSK/QAM), ``"instfreq"`` (FSK), ``"transitions"``
        (ASK/OOK), or ``"auto"`` which runs all three and merges the
        candidate lists, preferring whichever produced the cleaner line.

    Returns a list of ``(rate_hz, confidence)`` tuples, best first.
    """
    if len(samples) < 1024:
        return []

    if method == "envelope":
        return estimate_symbol_rate_envelope(samples, sample_rate)[:5]
    if method == "instfreq":
        return estimate_symbol_rate_instfreq(samples, sample_rate)[:5]
    if method == "transitions":
        return estimate_symbol_rate_transitions(samples, sample_rate)[:5]

    env = estimate_symbol_rate_envelope(samples, sample_rate)
    ifr = estimate_symbol_rate_instfreq(samples, sample_rate)
    trn = estimate_symbol_rate_transitions(samples, sample_rate)
    quad = estimate_symbol_rate_quadrature(samples, sample_rate)

    merged: dict[float, float] = {}

    def _add(cands: list[tuple[float, float]], weight: float) -> None:
        for rate, conf in cands:
            # Merge candidates within 1 % of each other, keeping the rate
            # reported by the stronger (hence more precise) detection
            key = next((k for k in merged if abs(k - rate) / max(k, 1.0) < 0.01), None)
            if key is None:
                merged[rate] = conf * weight
            elif conf * weight > merged[key]:
                del merged[key]
                merged[rate] = conf * weight

    _add(env, 1.0)
    _add(ifr, 1.0)
    _add(trn, 1.0)
    # When it fires at all (4th-power carrier line, no plain carrier) the
    # single-rail estimate is the most direct one for the QPSK family, and
    # the only one that sees OQPSK
    _add(quad, 2.0)

    # Agreement between methods is strong evidence
    for a, b in ((env, ifr), (env, trn), (ifr, trn), (env, quad), (trn, quad)):
        for r_e, c_e in a[:3]:
            for r_i, c_i in b[:3]:
                if abs(r_e - r_i) / max(r_e, 1.0) < 0.01:
                    # The merge keeps the stronger detection's rate, which may
                    # have drifted just outside 1 % of r_e: look near either
                    key = next((k for k in merged
                                if min(abs(k - r_e), abs(k - r_i)) / max(k, 1.0) < 0.01), None)
                    if key is not None:
                        merged[key] = min(1.0, merged[key] + 0.5 * min(c_e, c_i))

    out = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
    return [(float(r), float(c)) for r, c in out[:5]]


# ---------------------------------------------------------------------------
# Instantaneous quantities
# ---------------------------------------------------------------------------


def compute_instantaneous_frequency(
    samples: np.ndarray,
    sample_rate: float
) -> np.ndarray:
    """Compute the instantaneous frequency of the analytic signal (Hz).

    Uses the conjugate-product discriminator, which does not require
    phase unwrapping and is robust to noise-induced wraps.
    """
    if len(samples) < 2:
        return np.zeros(len(samples))
    prod = samples[1:] * np.conj(samples[:-1])
    inst_freq = np.angle(prod) / (2.0 * np.pi) * sample_rate
    return np.append(inst_freq, inst_freq[-1]).astype(np.float64)


def compute_instantaneous_amplitude(samples: np.ndarray) -> np.ndarray:
    """Compute the instantaneous amplitude (envelope) of the signal."""
    return np.abs(samples)


def compute_instantaneous_phase(samples: np.ndarray) -> np.ndarray:
    """Compute the unwrapped instantaneous phase of the signal."""
    return np.unwrap(np.angle(samples))
