"""Demodulators for PSK, QAM, and FSK.

Each demodulator takes complex baseband samples plus the parameters the
analysis stage has already estimated (symbol rate, frequency offset) and
produces symbol-rate samples, hard-decision bits, and quality metrics.

The processing chain for linear modulations (PSK/QAM) is:

    coarse CFO removal → RRC matched filter → rational resample to 8 sps
    → Gardner timing recovery → M-th power CFO estimate → Costas phase
    tracking → slicing → Gray de-mapping

For FSK:

    band-limit → frequency discriminator → integrate over one symbol
    → Gardner timing recovery on the real-valued discriminator output
    → level slicing (thresholds from a k-means on the sampled frequencies)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.core.enums import ModulationType
from src.core.exceptions import DemodulationError, InsufficientSamplesError
from src.dsp.measurements import (
    bandlimit_to_signal,
    compute_instantaneous_frequency,
    estimate_frequency_offset,
    estimate_snr_inband,
)
from src.dsp.preprocessing import translate_frequency
from src.dsp.sync import (
    costas_loop,
    evm_percent,
    gardner_timing_recovery,
    matched_filter,
    moving_average,
    psk_decision,
    qam_decision,
    resample_to_sps,
)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class DemodResult:
    """Everything a demodulator produces."""

    modulation: ModulationType
    symbols: np.ndarray                     # complex, carrier-corrected, unit power
    bits: np.ndarray                        # uint8 hard decisions
    symbol_rate_hz: float
    samples_per_symbol: float
    bits_per_symbol: int
    evm_percent: float = 0.0
    residual_cfo_hz: float = 0.0
    coarse_cfo_hz: float = 0.0
    timing_lock: float = 0.0                # mean |timing error| tail; lower = better
    carrier_lock: float = 0.0               # mean |phase error| tail; lower = better
    fsk_levels_hz: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    audio: np.ndarray | None = None         # analog modulations: recovered message
    audio_rate_hz: float = 0.0
    carrier_hz: float | None = None         # e.g. the suppressed SSB carrier used

    @property
    def num_symbols(self) -> int:
        return int(len(self.symbols))

    @property
    def num_bits(self) -> int:
        return int(len(self.bits))


# ---------------------------------------------------------------------------
# Bit mapping
# ---------------------------------------------------------------------------

_MOD_ORDER: dict[ModulationType, int] = {
    ModulationType.BPSK: 2,
    ModulationType.QPSK: 4,
    ModulationType.PSK8: 8,
    ModulationType.PSK16: 16,
    ModulationType.QAM16: 16,
    ModulationType.QAM64: 64,
    ModulationType.QAM256: 256,
    ModulationType.FSK2: 2,
    ModulationType.FSK4: 4,
    ModulationType.MSK: 2,
    ModulationType.GMSK: 2,
    ModulationType.GFSK: 2,
    ModulationType.OQPSK: 4,
    ModulationType.DPSK: 2,
    ModulationType.PI4_DQPSK: 4,
    ModulationType.OOK: 2,
}

#: Modulations whose output is audio rather than bits
ANALOG_MODULATIONS = frozenset({ModulationType.AM, ModulationType.FM,
                                ModulationType.SSB_USB, ModulationType.SSB_LSB})


def modulation_order(mod: ModulationType) -> int:
    return _MOD_ORDER.get(mod, 2)


def _gray_encode(n: int) -> int:
    return n ^ (n >> 1)


def _int_to_bits(values: np.ndarray, width: int) -> np.ndarray:
    """Vectorised integer → MSB-first bit array."""
    shifts = np.arange(width - 1, -1, -1)
    return ((values[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)


def psk_symbols_to_bits(symbols: np.ndarray, order: int) -> np.ndarray:
    """Gray-coded M-PSK hard decision.

    Constellation points sit at phase k·2π/M (k = 0..M-1) and carry the
    Gray code of k.  BPSK therefore maps +1 → 0, −1 → 1.
    """
    k_bits = int(np.log2(order))
    ang = np.angle(symbols)
    k = np.round(ang / (2 * np.pi / order)).astype(int) % order
    gray = np.array([_gray_encode(v) for v in range(order)])
    return _int_to_bits(gray[k], k_bits)


def qam_levels(order: int) -> np.ndarray:
    """Per-axis levels for square QAM, e.g. 16-QAM → [-3, -1, 1, 3]."""
    m = int(np.sqrt(order))
    return np.arange(-(m - 1), m, 2, dtype=np.float64)


def qam_symbols_to_bits(symbols: np.ndarray, order: int) -> np.ndarray:
    """Gray-coded square-QAM hard decision, I bits then Q bits per symbol."""
    m = int(np.sqrt(order))
    bits_per_axis = int(np.log2(m))
    levels = qam_levels(order)
    # Scale received (unit power) symbols to the integer grid
    grid_power = np.mean(levels[:, None] ** 2 + levels[None, :] ** 2)
    s = symbols * np.sqrt(grid_power)

    gray = np.array([_gray_encode(v) for v in range(m)])
    i_idx = np.argmin(np.abs(levels[None, :] - s.real[:, None]), axis=1)
    q_idx = np.argmin(np.abs(levels[None, :] - s.imag[:, None]), axis=1)
    i_bits = _int_to_bits(gray[i_idx], bits_per_axis).reshape(-1, bits_per_axis)
    q_bits = _int_to_bits(gray[q_idx], bits_per_axis).reshape(-1, bits_per_axis)
    return np.hstack([i_bits, q_bits]).reshape(-1)


def differential_decode_bits(bits: np.ndarray) -> np.ndarray:
    """XOR successive bits — removes a BPSK 180° phase ambiguity if the
    transmitter used differential encoding."""
    if len(bits) < 2:
        return bits.copy()
    return np.concatenate([[bits[0]], bits[1:] ^ bits[:-1]]).astype(np.uint8)


# ---------------------------------------------------------------------------
# Linear modulations
# ---------------------------------------------------------------------------


def _linear_front_end(
    samples: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    coarse_cfo_hz: float | None,
    rrc_beta: float,
    target_sps: int,
) -> tuple[np.ndarray, float, float, float]:
    """Shared PSK/QAM chain up to and including timing recovery.

    Returns ``(symbols, actual_sps, coarse_cfo_hz, timing_lock)``.
    """
    if symbol_rate <= 0:
        raise DemodulationError("Symbol rate must be positive.")
    if sample_rate / symbol_rate < 1.5:
        raise DemodulationError(
            f"Only {sample_rate / symbol_rate:.2f} samples/symbol; need ≥ 1.5 to demodulate."
        )
    if len(samples) < 64 * (sample_rate / symbol_rate):
        raise InsufficientSamplesError("Need at least 64 symbols worth of samples.")

    if coarse_cfo_hz is None:
        coarse_cfo_hz = estimate_frequency_offset(samples, sample_rate)
    x = translate_frequency(samples, sample_rate, -coarse_cfo_hz)

    # Resample first so the matched filter is short regardless of the
    # recording's oversampling ratio
    x, sps = resample_to_sps(x, sample_rate, symbol_rate, target_sps=target_sps)
    x = matched_filter(x, sps, beta=rrc_beta)

    timing = gardner_timing_recovery(x, sps)
    if len(timing.symbols) < 16:
        raise DemodulationError("Timing recovery produced too few symbols.")
    tail = timing.timing_error[len(timing.timing_error) * 3 // 4:]
    lock = float(np.mean(np.abs(tail))) if len(tail) else 0.0
    return timing.symbols, sps, coarse_cfo_hz, lock


def demodulate_psk(
    samples: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    order: int = 2,
    coarse_cfo_hz: float | None = None,
    rrc_beta: float = 0.35,
    target_sps: int = 8,
    differential: bool = False,
) -> DemodResult:
    """Demodulate M-PSK (order 2, 4, or 8)."""
    if order not in (2, 4, 8, 16):
        raise DemodulationError(f"Unsupported PSK order {order}.")

    syms, sps, cfo, t_lock = _linear_front_end(
        samples, sample_rate, symbol_rate, coarse_cfo_hz, rrc_beta, target_sps
    )
    carrier = costas_loop(syms, order)
    out = carrier.symbols

    # Discard the loop's acquisition transient
    settle = min(len(out) // 10, 200)
    out = out[settle:]

    ref = np.array([psk_decision(s, order) for s in out])
    bits = psk_symbols_to_bits(out, order)
    if differential:
        bits = differential_decode_bits(bits)

    mod = {2: ModulationType.BPSK, 4: ModulationType.QPSK,
           8: ModulationType.PSK8, 16: ModulationType.PSK16}[order]
    warnings = [f"M-PSK carrier phase is ambiguous to multiples of {360 // order}°; "
                "bit values may be rotated unless differential encoding was used."]
    if carrier.lock_metric > 0.4:
        warnings.append("Carrier loop did not converge well (high residual phase error).")

    return DemodResult(
        modulation=mod,
        symbols=out.astype(np.complex64),
        bits=bits,
        symbol_rate_hz=symbol_rate,
        samples_per_symbol=sps,
        bits_per_symbol=int(np.log2(order)),
        evm_percent=evm_percent(out, ref),
        residual_cfo_hz=carrier.cfo_cycles_per_symbol * symbol_rate,
        coarse_cfo_hz=cfo,
        timing_lock=t_lock,
        carrier_lock=carrier.lock_metric,
        warnings=warnings,
    )


def demodulate_qam(
    samples: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    order: int = 16,
    coarse_cfo_hz: float | None = None,
    rrc_beta: float = 0.35,
    target_sps: int = 8,
) -> DemodResult:
    """Demodulate square M-QAM (order 16, 64, 256)."""
    if order not in (16, 64, 256):
        raise DemodulationError(f"Unsupported QAM order {order}.")

    syms, sps, cfo, t_lock = _linear_front_end(
        samples, sample_rate, symbol_rate, coarse_cfo_hz, rrc_beta, target_sps
    )
    levels = qam_levels(order)
    carrier = costas_loop(syms, 4, qam_levels=levels)
    out = carrier.symbols
    settle = min(len(out) // 10, 200)
    out = out[settle:]

    grid_power = np.mean(levels[:, None] ** 2 + levels[None, :] ** 2)
    lv_unit = levels / np.sqrt(grid_power)
    ref = np.array([qam_decision(s, lv_unit) for s in out])
    bits = qam_symbols_to_bits(out, order)

    mod = {16: ModulationType.QAM16, 64: ModulationType.QAM64,
           256: ModulationType.QAM256}[order]
    warnings = ["QAM carrier phase is ambiguous to multiples of 90°."]
    if carrier.lock_metric > 0.3:
        warnings.append("Carrier loop did not converge well (high residual phase error).")

    return DemodResult(
        modulation=mod,
        symbols=out.astype(np.complex64),
        bits=bits,
        symbol_rate_hz=symbol_rate,
        samples_per_symbol=sps,
        bits_per_symbol=int(np.log2(order)),
        evm_percent=evm_percent(out, ref),
        residual_cfo_hz=carrier.cfo_cycles_per_symbol * symbol_rate,
        coarse_cfo_hz=cfo,
        timing_lock=t_lock,
        carrier_lock=carrier.lock_metric,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# FSK
# ---------------------------------------------------------------------------


def _kmeans_1d(values: np.ndarray, k: int, iters: int = 30) -> np.ndarray:
    """Tiny 1-D k-means; returns sorted centroids."""
    lo, hi = np.percentile(values, 2), np.percentile(values, 98)
    centroids = np.linspace(lo, hi, k)
    for _ in range(iters):
        assign = np.argmin(np.abs(values[:, None] - centroids[None, :]), axis=1)
        new = np.array([
            values[assign == i].mean() if np.any(assign == i) else centroids[i]
            for i in range(k)
        ])
        if np.allclose(new, centroids):
            break
        centroids = new
    return np.sort(centroids)


@dataclass
class FSKFrontEnd:
    """Symbol-centre instantaneous frequencies of a CPFSK-like signal."""

    freq_symbols_hz: np.ndarray       # relative to the coarse carrier
    samples_per_symbol: float
    coarse_cfo_hz: float
    timing_lock: float


def fsk_front_end(
    samples: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    coarse_cfo_hz: float | None = None,
    target_sps: int = 8,
) -> FSKFrontEnd:
    """Mix to DC → band-limit → resample → discriminate → integrate over a
    symbol → Gardner timing; shared by the FSK/MSK/GMSK demodulators and
    the classifier."""
    if symbol_rate <= 0:
        raise DemodulationError("Symbol rate must be positive.")
    if len(samples) < 64 * (sample_rate / symbol_rate):
        raise InsufficientSamplesError("Need at least 64 symbols worth of samples.")

    if coarse_cfo_hz is None:
        coarse_cfo_hz = estimate_frequency_offset(samples, sample_rate)

    # Mix the tone pair to DC *before* resampling: at target_sps the new
    # rate can be far below the carrier (e.g. a 1700 Hz audio modem at
    # 8 × 300 Bd = 2400 Hz), which would alias the tones.
    x = translate_frequency(samples, sample_rate, -coarse_cfo_hz)
    x = bandlimit_to_signal(x, sample_rate)
    # Keep the whole tone spread inside the resampled band
    _, occupied = estimate_snr_inband(x, sample_rate)
    if occupied > 0:
        target_sps = max(target_sps, int(np.ceil(1.25 * occupied / symbol_rate)))
    target_sps = min(target_sps, max(2, int(sample_rate / symbol_rate)))
    x, sps = resample_to_sps(x, sample_rate, symbol_rate, target_sps=target_sps)
    fs_eff = sps * symbol_rate
    inst_freq = compute_instantaneous_frequency(x, fs_eff)

    # Integrate over one symbol (matched filter for a rectangular pulse)
    integrated = moving_average(inst_freq, int(round(sps)))

    timing = gardner_timing_recovery(integrated.astype(np.complex128), sps)
    if len(timing.symbols) < 16:
        raise DemodulationError("Timing recovery produced too few symbols.")
    freq_syms = np.real(timing.symbols)
    settle = min(len(freq_syms) // 10, 200)
    freq_syms = freq_syms[settle:]

    # Gardner normalised power, so recover Hz by comparing to the
    # discriminator output's RMS
    scale = np.sqrt(np.mean(integrated**2)) if np.any(integrated) else 1.0
    tail = timing.timing_error[len(timing.timing_error) * 3 // 4:]
    t_lock = float(np.mean(np.abs(tail))) if len(tail) else 0.0
    return FSKFrontEnd(freq_syms * scale, sps, float(coarse_cfo_hz), t_lock)


def demodulate_fsk(
    samples: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    order: int = 2,
    coarse_cfo_hz: float | None = None,
    target_sps: int = 8,
    modulation: ModulationType | None = None,
) -> DemodResult:
    """Demodulate M-FSK (order 2 or 4) – or MSK/GMSK, which are 2-level
    CPFSK – with a frequency discriminator."""
    if order not in (2, 4):
        raise DemodulationError(f"Unsupported FSK order {order}.")
    fe = fsk_front_end(samples, sample_rate, symbol_rate, coarse_cfo_hz, target_sps)
    freq_syms_hz, sps, coarse_cfo_hz = fe.freq_symbols_hz, fe.samples_per_symbol, fe.coarse_cfo_hz

    centroids = _kmeans_1d(freq_syms_hz, order)
    thresholds = (centroids[:-1] + centroids[1:]) / 2.0
    level_idx = np.searchsorted(thresholds, freq_syms_hz)

    # Gray-code the level index (natural binary for 2-FSK)
    k_bits = int(np.log2(order))
    gray = np.array([_gray_encode(v) for v in range(order)])
    bits = _int_to_bits(gray[level_idx], k_bits)

    # Present frequency symbols on the real axis, scaled to the level grid
    deviation = float(np.max(np.abs(centroids))) or 1.0
    symbols = (freq_syms_hz / deviation).astype(np.complex64)

    # EVM analogue: spread around the level centroids
    ref = centroids[level_idx]
    evm = float(100.0 * np.sqrt(np.mean((freq_syms_hz - ref) ** 2)) / deviation)

    t_lock = fe.timing_lock

    mod = modulation or (ModulationType.FSK2 if order == 2 else ModulationType.FSK4)
    return DemodResult(
        modulation=mod,
        symbols=symbols,
        bits=bits,
        symbol_rate_hz=symbol_rate,
        samples_per_symbol=sps,
        bits_per_symbol=k_bits,
        evm_percent=evm,
        residual_cfo_hz=float(np.mean(centroids)),
        coarse_cfo_hz=coarse_cfo_hz,
        timing_lock=t_lock,
        carrier_lock=0.0,
        fsk_levels_hz=[float(c) + coarse_cfo_hz for c in centroids],
        warnings=[],
    )


# ---------------------------------------------------------------------------
# Amplitude-shift keying (OOK / ASK)
# ---------------------------------------------------------------------------


@dataclass
class LevelFit:
    """Result of fitting discrete levels to symbol-centre values."""

    order: int                   # 1 (no structure), 2 or 4
    centroids: np.ndarray
    separation: float            # 2-level: centroid gap / within-cluster std
    variance_ratio: float        # within-cluster variance, 4 vs 2 levels
    even_spacing: bool           # 4-level centroids roughly evenly spaced


def fit_levels(values: np.ndarray, four_level_ratio: float = 0.27) -> LevelFit:
    """Decide whether *values* sit on 2 or 4 discrete levels.

    Thresholds were set on synthetic OOK/ASK/FSK (≥ 6 dB SNR) against
    analog AM/FM, whose symbol-centre values are continuous: a continuous
    (Gaussian-like) distribution gives a 2-level separation of ≈ 2.4 and a
    4/2 variance ratio of ≈ 0.33.  *four_level_ratio* is the variance
    ratio below which 4 levels are accepted (4-ASK at 6 dB ≈ 0.22; 4-FSK
    ≈ 0.01, while GMSK's ISI fakes ≈ 0.2, hence a stricter value there).
    """
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 32:
        return LevelFit(1, np.array([np.mean(v) if len(v) else 0.0]), 0.0, 1.0, False)

    def within(c: np.ndarray) -> float:
        a = np.argmin(np.abs(v[:, None] - c[None, :]), axis=1)
        return float(np.mean((v - c[a]) ** 2))

    c2 = _kmeans_1d(v, 2)
    c4 = _kmeans_1d(v, 4)
    w2, w4 = within(c2), within(c4)
    sep = float((c2[1] - c2[0]) / np.sqrt(max(w2, 1e-30)))
    ratio = w4 / max(w2, 1e-30)
    gaps = np.diff(c4)
    even = bool(np.max(gaps) > 0 and np.min(gaps) > 0.6 * np.max(gaps))
    if ratio < four_level_ratio and even:
        return LevelFit(4, c4, sep, ratio, even)
    if sep >= 5.0:
        return LevelFit(2, c2, sep, ratio, even)
    return LevelFit(1, c2, sep, ratio, even)


def envelope_symbols(
    samples: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    coarse_cfo_hz: float | None = None,
    target_sps: int = 8,
) -> tuple[np.ndarray, float, float, float]:
    """Envelope at the symbol centres: mix to DC → band-limit → resample →
    |x| → Gardner timing.  Returns ``(values, sps, coarse_cfo_hz, timing_lock)``."""
    if symbol_rate <= 0:
        raise DemodulationError("Symbol rate must be positive.")
    if len(samples) < 64 * (sample_rate / symbol_rate):
        raise InsufficientSamplesError("Need at least 64 symbols worth of samples.")
    if coarse_cfo_hz is None:
        coarse_cfo_hz = estimate_frequency_offset(samples, sample_rate)
    x = translate_frequency(samples, sample_rate, -coarse_cfo_hz)
    x = bandlimit_to_signal(x, sample_rate)
    x, sps = resample_to_sps(x, sample_rate, symbol_rate, target_sps=target_sps)
    env = np.abs(x).astype(np.float64)
    mean = float(np.mean(env))
    ac = env - mean
    timing = gardner_timing_recovery(ac.astype(np.complex128), sps)
    if len(timing.symbols) < 16:
        raise DemodulationError("Timing recovery produced too few symbols.")
    v = np.real(timing.symbols)
    # Gardner normalises power; restore the envelope scale and its mean
    v = v * np.sqrt(np.mean(ac ** 2)) / (np.sqrt(np.mean(v ** 2)) + 1e-12) + mean
    v = v[min(len(v) // 10, 200):]
    tail = timing.timing_error[len(timing.timing_error) * 3 // 4:]
    t_lock = float(np.mean(np.abs(tail))) if len(tail) else 0.0
    return v, sps, float(coarse_cfo_hz), t_lock


def demodulate_ask(
    samples: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    order: int | None = None,
    coarse_cfo_hz: float | None = None,
    target_sps: int = 8,
) -> DemodResult:
    """Demodulate unipolar ASK / OOK with an envelope detector.

    *order* (2 or 4) is detected from the envelope levels when omitted.
    Levels map to Gray codes in ascending amplitude, so for OOK
    "carrier off" → 0 and "carrier on" → 1.
    """
    v, sps, cfo, t_lock = envelope_symbols(samples, sample_rate, symbol_rate, coarse_cfo_hz,
                                           target_sps)
    fit = fit_levels(v)
    if order is None:
        order = fit.order if fit.order in (2, 4) else 2
    if order not in (2, 4):
        raise DemodulationError(f"Unsupported ASK order {order}.")
    centroids = _kmeans_1d(v, order)
    idx = np.searchsorted((centroids[:-1] + centroids[1:]) / 2.0, v)
    k_bits = int(np.log2(order))
    gray = np.array([_gray_encode(i) for i in range(order)])
    bits = _int_to_bits(gray[idx], k_bits)

    peak = float(centroids[-1]) or 1.0
    symbols = (v / peak).astype(np.complex64)
    evm = float(100.0 * np.sqrt(np.mean((v - centroids[idx]) ** 2)) / peak)
    ook = order == 2 and centroids[0] < 0.25 * centroids[-1]
    return DemodResult(
        modulation=ModulationType.OOK if ook else ModulationType.ASK,
        symbols=symbols, bits=bits, symbol_rate_hz=symbol_rate, samples_per_symbol=sps,
        bits_per_symbol=k_bits, evm_percent=evm, coarse_cfo_hz=cfo, timing_lock=t_lock,
        warnings=[f"{order}-level amplitude keying, levels "
                  + ", ".join(f"{c / peak:.2f}" for c in centroids) + " (relative)."],
    )


# ---------------------------------------------------------------------------
# Offset QPSK
# ---------------------------------------------------------------------------


def _feedforward_phase(y: np.ndarray, order: int, window: int = 64,
                       offset: float = np.pi) -> np.ndarray:
    """Viterbi-Viterbi phase: de-rotate *y* using a sliding average of
    ``y^order`` (whose ideal angle is *offset*).  Tracks slow drift."""
    z = y ** order
    avg = np.convolve(z, np.ones(window) / window, mode="same")
    theta = np.unwrap(np.angle(avg) - offset) / order
    return y * np.exp(-1j * theta)


def oqpsk_rails(
    samples: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    target_sps: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Carrier-recovered OQPSK-style rails.

    Returns ``(offset_symbols, aligned_symbols, strobes, sps, carrier_hz,
    timing_lock)``: I at its symbol centres paired with Q half a symbol
    later (OQPSK) or at the same instant (QPSK) – whichever pairing gives
    a clean QPSK constellation reveals the modulation.
    """
    from src.dsp.measurements import mpower_carrier

    if symbol_rate <= 0:
        raise DemodulationError("Symbol rate must be positive.")
    if len(samples) < 64 * (sample_rate / symbol_rate):
        raise InsufficientSamplesError("Need at least 64 symbols worth of samples.")
    band = bandlimit_to_signal(samples, sample_rate)
    fc, phase, _ = mpower_carrier(band, sample_rate, 4)
    t = np.arange(len(band)) / sample_rate
    # QPSK points at ±45° put the rails on the axes (ambiguity k·90°)
    x = band * np.exp(-1j * (2 * np.pi * fc * t + phase - np.pi / 4))
    x, sps = resample_to_sps(x, sample_rate, symbol_rate, target_sps=target_sps)
    x = matched_filter(x, sps)
    timing = gardner_timing_recovery(np.real(x).astype(np.complex128), sps)
    if len(timing.symbols) < 16:
        raise DemodulationError("Timing recovery produced too few symbols.")
    strobes = timing.strobe_positions
    idx = np.arange(len(x))
    i = np.interp(strobes, idx, np.real(x))
    q_off = np.interp(strobes + sps / 2.0, idx, np.imag(x))
    q_al = np.interp(strobes, idx, np.imag(x))
    settle = min(len(i) // 10, 200)
    tail = timing.timing_error[len(timing.timing_error) * 3 // 4:]
    t_lock = float(np.mean(np.abs(tail))) if len(tail) else 0.0
    off = (i + 1j * q_off)[settle:]
    al = (i + 1j * q_al)[settle:]
    return off, al, strobes[settle:], sps, float(fc), t_lock


def qpsk_fit(symbols: np.ndarray) -> float:
    """|E[s⁴]| / E[|s|⁴]: ≈ 1 for a clean QPSK constellation, → 0 otherwise."""
    s = np.asarray(symbols)
    den = float(np.mean(np.abs(s) ** 4))
    return float(np.abs(np.mean(s ** 4)) / den) if den > 0 else 0.0


def demodulate_oqpsk(
    samples: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    coarse_cfo_hz: float | None = None,
    target_sps: int = 8,
) -> DemodResult:
    """Offset-QPSK: carrier from the 4th-power line, timing from the I rail,
    Q sampled half a symbol later; bits I then Q per symbol (+ → 1)."""
    off, _, _, sps, fc, t_lock = oqpsk_rails(samples, sample_rate, symbol_rate, target_sps)
    y = off / (np.sqrt(np.mean(np.abs(off) ** 2)) + 1e-12)
    y = _feedforward_phase(y, 4)
    bits = np.empty(2 * len(y), dtype=np.uint8)
    bits[0::2] = (y.real > 0).astype(np.uint8)
    bits[1::2] = (y.imag > 0).astype(np.uint8)
    ref = (np.sign(y.real) + 1j * np.sign(y.imag)) / np.sqrt(2)
    return DemodResult(
        modulation=ModulationType.OQPSK, symbols=y.astype(np.complex64), bits=bits,
        symbol_rate_hz=symbol_rate, samples_per_symbol=sps, bits_per_symbol=2,
        evm_percent=evm_percent(y, ref), coarse_cfo_hz=fc, timing_lock=t_lock,
        warnings=["OQPSK carrier phase is ambiguous to multiples of 90° (I/Q may be "
                  "swapped or inverted), and which Q pairs with which I is ambiguous "
                  "by one symbol."],
    )


# ---------------------------------------------------------------------------
# Differential PSK (DBPSK, π/4-DQPSK)
# ---------------------------------------------------------------------------

#: π/4-DQPSK phase steps → dibits (TETRA / IS-54 Gray mapping)
_PI4_STEPS = np.array([np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4])
_PI4_BITS = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)


def differential_products(symbols: np.ndarray) -> np.ndarray:
    """``s[k]·conj(s[k−1])`` normalised to unit magnitude."""
    s = np.asarray(symbols, dtype=np.complex128)
    d = s[1:] * np.conj(s[:-1])
    return d / (np.abs(d) + 1e-12)


def demodulate_dpsk(
    samples: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    variant: ModulationType = ModulationType.DPSK,
    coarse_cfo_hz: float | None = None,
    rrc_beta: float = 0.35,
    target_sps: int = 8,
) -> DemodResult:
    """Differential detection – no carrier phase needed, so no ambiguity.

    ``DPSK`` (DBPSK): bit 1 = 180° phase change.  ``PI4_DQPSK``: phase steps
    ±45°/±135° carry two bits each.
    """
    syms, sps, cfo, t_lock = _linear_front_end(
        samples, sample_rate, symbol_rate, coarse_cfo_hz, rrc_beta, target_sps
    )
    d = differential_products(syms[min(len(syms) // 10, 200):])
    if variant == ModulationType.PI4_DQPSK:
        # Residual CFO rotates every product by the same angle: remove it
        d = d * np.exp(-1j * np.angle(-np.mean(d ** 4)) / 4)
        k = np.argmin(np.abs(np.angle(d[:, None] * np.exp(-1j * _PI4_STEPS[None, :]))), axis=1)
        bits = _PI4_BITS[k].reshape(-1)
        ref = np.exp(1j * _PI4_STEPS[k])
        bps = 2
    else:
        d = d * np.exp(-1j * np.angle(np.mean(d ** 2)) / 2)
        bits = (d.real < 0).astype(np.uint8)
        ref = np.sign(d.real) + 0j
        bps = 1
    return DemodResult(
        modulation=variant, symbols=d.astype(np.complex64), bits=bits,
        symbol_rate_hz=symbol_rate, samples_per_symbol=sps, bits_per_symbol=bps,
        evm_percent=evm_percent(d, ref), coarse_cfo_hz=cfo, timing_lock=t_lock,
        warnings=["Differential detection: the constellation shows phase *changes* "
                  "between symbols."],
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def demodulate(
    samples: np.ndarray,
    sample_rate: float,
    modulation: ModulationType,
    symbol_rate: float,
    coarse_cfo_hz: float | None = None,
    **kwargs: object,
) -> DemodResult:
    """Demodulate *samples* according to *modulation*.

    Analog modulations (AM, FM, SSB) return the recovered message in
    ``DemodResult.audio`` and need no symbol rate.  Raises
    :class:`DemodulationError` for modulations without a demodulator.
    """
    if modulation in (ModulationType.BPSK, ModulationType.QPSK,
                      ModulationType.PSK8, ModulationType.PSK16):
        return demodulate_psk(samples, sample_rate, symbol_rate,
                              order=modulation_order(modulation),
                              coarse_cfo_hz=coarse_cfo_hz, **kwargs)  # type: ignore[arg-type]
    if modulation in (ModulationType.QAM16, ModulationType.QAM64, ModulationType.QAM256):
        return demodulate_qam(samples, sample_rate, symbol_rate,
                              order=modulation_order(modulation),
                              coarse_cfo_hz=coarse_cfo_hz, **kwargs)  # type: ignore[arg-type]
    if modulation in (ModulationType.FSK2, ModulationType.FSK4):
        return demodulate_fsk(samples, sample_rate, symbol_rate,
                              order=modulation_order(modulation),
                              coarse_cfo_hz=coarse_cfo_hz, **kwargs)  # type: ignore[arg-type]
    if modulation in (ModulationType.MSK, ModulationType.GMSK, ModulationType.GFSK):
        return demodulate_fsk(samples, sample_rate, symbol_rate, order=2,
                              coarse_cfo_hz=coarse_cfo_hz, modulation=modulation,
                              **kwargs)  # type: ignore[arg-type]
    if modulation in (ModulationType.ASK, ModulationType.OOK):
        return demodulate_ask(samples, sample_rate, symbol_rate,
                              order=2 if modulation == ModulationType.OOK else None,
                              coarse_cfo_hz=coarse_cfo_hz, **kwargs)  # type: ignore[arg-type]
    if modulation == ModulationType.OQPSK:
        return demodulate_oqpsk(samples, sample_rate, symbol_rate,
                                coarse_cfo_hz=coarse_cfo_hz, **kwargs)  # type: ignore[arg-type]
    if modulation in (ModulationType.DPSK, ModulationType.PI4_DQPSK):
        return demodulate_dpsk(samples, sample_rate, symbol_rate, variant=modulation,
                               coarse_cfo_hz=coarse_cfo_hz, **kwargs)  # type: ignore[arg-type]
    if modulation in ANALOG_MODULATIONS:
        from src.dsp.analog import demodulate_analog

        return demodulate_analog(samples, sample_rate, modulation,
                                 coarse_cfo_hz=coarse_cfo_hz, **kwargs)  # type: ignore[arg-type]
    raise DemodulationError(f"No demodulator available for {modulation.value}.")
