"""Synchronisation primitives for demodulation.

Provides the building blocks a demodulator needs before it can slice
symbols: pulse-shape filtering, rational resampling to a convenient
samples-per-symbol, Gardner timing recovery, feed-forward carrier
frequency estimation, and decision-directed carrier phase tracking.

All functions are pure NumPy/SciPy and operate on 1-D complex arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy import signal as sp_signal

# ---------------------------------------------------------------------------
# Pulse shaping
# ---------------------------------------------------------------------------


def rrc_taps(sps: float, beta: float = 0.35, span_symbols: int = 8) -> np.ndarray:
    """Root-raised-cosine filter taps, unit energy.

    Parameters
    ----------
    sps : samples per symbol (may be fractional; taps are sampled at 1/sps).
    beta : roll-off factor in (0, 1].
    span_symbols : filter length in symbols (total taps ≈ span × sps).
    """
    n_taps = int(round(span_symbols * sps)) | 1  # force odd
    t = (np.arange(n_taps) - (n_taps - 1) / 2.0) / sps  # in symbols
    taps = np.zeros(n_taps)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            taps[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0 and abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-9:
            taps[i] = (beta / np.sqrt(2.0)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            taps[i] = num / den

    return taps / np.sqrt(np.sum(taps**2))


def matched_filter(samples: np.ndarray, sps: float, beta: float = 0.35) -> np.ndarray:
    """Apply an RRC matched filter (assumes the transmitter used RRC)."""
    taps = rrc_taps(sps, beta)
    return sp_signal.fftconvolve(samples, taps, mode="same").astype(np.complex64)


def moving_average(x: np.ndarray, length: int) -> np.ndarray:
    """Boxcar (integrate-and-dump style) filter, same length output."""
    length = max(1, int(length))
    kernel = np.ones(length) / length
    return sp_signal.fftconvolve(x, kernel, mode="same")


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def resample_to_sps(
    samples: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    target_sps: int = 8,
    max_denominator: int = 2000,
) -> tuple[np.ndarray, float]:
    """Rationally resample so the signal has ≈ *target_sps* samples/symbol.

    Returns ``(resampled, actual_sps)``.  The actual value differs from the
    target only by the rational approximation error, which the timing loop
    absorbs.
    """
    current_sps = sample_rate / symbol_rate
    if current_sps <= 0:
        raise ValueError("sample_rate and symbol_rate must be positive")

    ratio = Fraction(target_sps / current_sps).limit_denominator(max_denominator)
    up, down = ratio.numerator, ratio.denominator
    if up == down:
        return samples, current_sps

    out = sp_signal.resample_poly(samples, up, down)
    actual_sps = current_sps * up / down
    return out.astype(np.complex64), float(actual_sps)


# ---------------------------------------------------------------------------
# Timing recovery
# ---------------------------------------------------------------------------


@dataclass
class TimingResult:
    """Output of the timing-recovery loop."""

    symbols: np.ndarray
    strobe_positions: np.ndarray            # fractional sample index of each symbol
    period_estimate: float = 0.0            # final estimate of samples/symbol
    timing_error: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _cubic_interp(x: np.ndarray, pos: float) -> complex:
    """4-point Lagrange (cubic) interpolation at fractional index *pos*."""
    k = int(np.floor(pos))
    mu = pos - k
    if k < 1 or k + 2 >= len(x):
        k = min(max(k, 0), len(x) - 1)
        return x[k]
    xm1, x0, x1, x2 = x[k - 1], x[k], x[k + 1], x[k + 2]
    # Farrow-form cubic
    c0 = x0
    c1 = -xm1 / 3.0 - x0 / 2.0 + x1 - x2 / 6.0
    c2 = xm1 / 2.0 - x0 + x1 / 2.0
    c3 = -xm1 / 6.0 + x0 / 2.0 - x1 / 2.0 + x2 / 6.0
    return ((c3 * mu + c2) * mu + c1) * mu + c0


def gardner_timing_recovery(
    samples: np.ndarray,
    sps: float,
    loop_bandwidth: float = 0.05,
    damping: float = 1.0,
    max_period_deviation: float = 0.05,
) -> TimingResult:
    """Gardner timing-error-detector with a second-order PI loop.

    The Gardner detector is decision-independent, so it works for any
    linear modulation (PSK, QAM) and for real-valued PAM such as the
    frequency-discriminator output of an FSK signal.  Requires roughly
    2 or more samples per symbol.

    Parameters
    ----------
    sps : nominal samples per symbol.
    loop_bandwidth : normalised loop bandwidth (fraction of symbol rate).
    damping : loop damping factor (1.0 ≈ critically damped).
    max_period_deviation : clamp on how far the tracked period may drift
        from the nominal, as a fraction.
    """
    x = np.asarray(samples)
    n = len(x)
    if n < int(3 * sps) + 4:
        return TimingResult(symbols=np.zeros(0, dtype=x.dtype),
                            strobe_positions=np.zeros(0))

    # Normalise power so loop gains are signal-independent
    rms = np.sqrt(np.mean(np.abs(x) ** 2))
    if rms > 0:
        x = x / rms

    # PI gains from bandwidth/damping (standard second-order design).
    # Detector gain for a unit-power Gardner TED is ≈ 2.
    kd = 2.0
    theta = loop_bandwidth / (damping + 1.0 / (4.0 * damping))
    denom = 1.0 + 2.0 * damping * theta + theta**2
    kp = (4.0 * damping * theta / denom) / kd
    ki = (4.0 * theta**2 / denom) / kd

    period = float(sps)
    period_min = sps * (1.0 - max_period_deviation)
    period_max = sps * (1.0 + max_period_deviation)

    pos = float(sps)  # first strobe
    prev = _cubic_interp(x, 0.0)
    integrator = 0.0

    est_symbols = int(n / sps) + 2
    symbols = np.zeros(est_symbols, dtype=np.complex128)
    strobes = np.zeros(est_symbols)
    errors = np.zeros(est_symbols)
    count = 0

    while pos + 2 < n - 1 and count < est_symbols:
        cur = _cubic_interp(x, pos)
        mid = _cubic_interp(x, pos - period / 2.0)

        # Gardner TED: e = Re{ conj(mid) · (prev − cur) }.  With this sign a
        # positive error means the strobe is early, so it is *added* to the
        # position below.
        err = float(np.real(np.conj(mid) * (prev - cur)))
        err = max(-1.0, min(1.0, err))

        integrator += ki * err
        integrator = max(-max_period_deviation * sps, min(max_period_deviation * sps, integrator))
        period = float(sps) + integrator
        period = max(period_min, min(period_max, period))

        symbols[count] = cur
        strobes[count] = pos
        errors[count] = err
        count += 1

        pos += period + kp * err
        prev = cur

    return TimingResult(
        symbols=symbols[:count],
        strobe_positions=strobes[:count],
        period_estimate=period,
        timing_error=errors[:count],
    )


def maximum_energy_timing(samples: np.ndarray, sps: int) -> tuple[np.ndarray, int]:
    """Pick the integer sampling phase with the highest symbol energy.

    A fast, open-loop fallback for integer *sps* when no clock drift is
    expected.  Returns ``(symbols, phase)``.
    """
    sps = int(sps)
    best_phase, best_energy = 0, -1.0
    for ph in range(sps):
        e = float(np.sum(np.abs(samples[ph::sps]) ** 2))
        if e > best_energy:
            best_energy, best_phase = e, ph
    return samples[best_phase::sps], best_phase


# ---------------------------------------------------------------------------
# Carrier recovery
# ---------------------------------------------------------------------------


def estimate_cfo_mpower(
    symbols: np.ndarray,
    order: int,
    zero_pad: int = 8,
) -> float:
    """Feed-forward carrier frequency estimate via the M-th power method.

    Raising M-PSK symbols to the M-th power collapses the modulation to a
    single tone at M·Δf.  Returns Δf in cycles per symbol (multiply by the
    symbol rate for Hz).  Ambiguous beyond ±1/(2M) cycles/symbol.
    """
    if len(symbols) < 16:
        return 0.0
    s = np.asarray(symbols, dtype=np.complex128)
    s = s / (np.sqrt(np.mean(np.abs(s) ** 2)) + 1e-12)
    powered = s**order
    n_fft = 1 << int(np.ceil(np.log2(len(powered) * zero_pad)))
    spec = np.fft.fftshift(np.fft.fft(powered * np.hanning(len(powered)), n_fft))
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft))
    peak = int(np.argmax(np.abs(spec)))

    # Parabolic interpolation on the peak for sub-bin accuracy
    if 0 < peak < n_fft - 1:
        a, b, c = np.abs(spec[peak - 1]), np.abs(spec[peak]), np.abs(spec[peak + 1])
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    f_m = freqs[peak] + delta * (freqs[1] - freqs[0])
    return float(f_m / order)


def apply_cfo(symbols: np.ndarray, cfo_cycles_per_symbol: float) -> np.ndarray:
    """Remove a carrier frequency offset expressed in cycles/symbol."""
    n = np.arange(len(symbols))
    return symbols * np.exp(-1j * 2.0 * np.pi * cfo_cycles_per_symbol * n)


def psk_decision(sym: complex, order: int) -> complex:
    """Nearest M-PSK constellation point (unit circle, phase k·2π/M)."""
    ang = np.angle(sym)
    k = np.round(ang / (2 * np.pi / order))
    return np.exp(1j * k * 2 * np.pi / order)


def qam_decision(sym: complex, levels: np.ndarray) -> complex:
    """Nearest square-QAM point given per-axis *levels*."""
    i = levels[np.argmin(np.abs(levels - sym.real))]
    q = levels[np.argmin(np.abs(levels - sym.imag))]
    return complex(i, q)


@dataclass
class CarrierResult:
    """Output of the carrier phase-tracking loop."""

    symbols: np.ndarray
    phase: np.ndarray
    cfo_cycles_per_symbol: float = 0.0
    lock_metric: float = 0.0     # mean |error| in the final quarter; lower = better


def costas_loop(
    symbols: np.ndarray,
    order: int,
    loop_bandwidth: float = 0.02,
    damping: float = 0.707,
    qam_levels: np.ndarray | None = None,
) -> CarrierResult:
    """Decision-directed carrier phase tracking (generalised Costas loop).

    For M-PSK (``order`` = 2, 4, 8) the decision is the nearest PSK point.
    For QAM pass ``qam_levels`` (e.g. ``[-3, -1, 1, 3]``) and the decision
    is the nearest square-QAM point.  The residual carrier frequency is
    removed feed-forward first via :func:`estimate_cfo_mpower`, so the loop
    only has to track phase noise and the small leftover frequency.
    """
    s = np.asarray(symbols, dtype=np.complex128)
    if len(s) == 0:
        return CarrierResult(symbols=s, phase=np.zeros(0))

    # Normalise to unit average power
    s = s / (np.sqrt(np.mean(np.abs(s) ** 2)) + 1e-12)

    # Feed-forward CFO removal. 4th power works for QAM too (weaker line).
    m_power = order if qam_levels is None else 4
    cfo = estimate_cfo_mpower(s, m_power)
    s = apply_cfo(s, cfo)

    if qam_levels is not None:
        # Scale so that the decision grid matches unit-power QAM
        lv = np.asarray(qam_levels, dtype=np.float64)
        grid_power = np.mean(lv[:, None] ** 2 + lv[None, :] ** 2)
        lv = lv / np.sqrt(grid_power)
    else:
        lv = None

    theta_n = loop_bandwidth / (damping + 1.0 / (4.0 * damping))
    denom = 1.0 + 2.0 * damping * theta_n + theta_n**2
    kp = 4.0 * damping * theta_n / denom
    ki = 4.0 * theta_n**2 / denom

    phase = 0.0
    freq = 0.0
    out = np.zeros_like(s)
    ph_track = np.zeros(len(s))
    errs = np.zeros(len(s))

    # For QAM, only the outer-corner points give an unambiguous phase
    # error at moderate SNR; inner points are too easily mis-decided.
    corner_thresh = 0.0
    if lv is not None:
        corner_thresh = 0.85 * np.sqrt(2.0) * float(np.max(np.abs(lv)))

    for i, sym in enumerate(s):
        y = sym * np.exp(-1j * phase)
        if lv is not None:
            if abs(y) >= corner_thresh:
                d = complex(np.sign(y.real), np.sign(y.imag))
                e = float(np.angle(y * np.conj(d)))
            else:
                e = 0.0
        else:
            d = psk_decision(y, order)
            # Phase error = angle between received and decided
            e = float(np.angle(y * np.conj(d)))
        freq += ki * e
        phase += freq + kp * e
        out[i] = y
        ph_track[i] = phase
        errs[i] = abs(e)

    tail = errs[len(errs) * 3 // 4:] if len(errs) >= 4 else errs
    return CarrierResult(
        symbols=out,
        phase=ph_track,
        cfo_cycles_per_symbol=cfo,
        lock_metric=float(np.mean(tail)) if len(tail) else 0.0,
    )


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


def evm_percent(symbols: np.ndarray, reference: np.ndarray) -> float:
    """Error-vector magnitude in percent, RMS, relative to reference power."""
    if len(symbols) == 0:
        return 0.0
    err = np.mean(np.abs(symbols - reference) ** 2)
    ref = np.mean(np.abs(reference) ** 2)
    if ref <= 0:
        return 0.0
    return float(100.0 * np.sqrt(err / ref))
