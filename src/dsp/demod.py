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
}


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


def demodulate_fsk(
    samples: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    order: int = 2,
    coarse_cfo_hz: float | None = None,
    target_sps: int = 8,
) -> DemodResult:
    """Demodulate M-FSK (order 2 or 4) with a frequency discriminator."""
    if order not in (2, 4):
        raise DemodulationError(f"Unsupported FSK order {order}.")
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

    # Rescale: Gardner normalised power, so recover Hz by comparing to the
    # discriminator output's RMS
    scale = np.sqrt(np.mean(integrated**2)) if np.any(integrated) else 1.0
    freq_syms_hz = freq_syms * scale

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

    tail = timing.timing_error[len(timing.timing_error) * 3 // 4:]
    t_lock = float(np.mean(np.abs(tail))) if len(tail) else 0.0

    mod = ModulationType.FSK2 if order == 2 else ModulationType.FSK4
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

    Raises :class:`DemodulationError` for modulations without a demodulator.
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
    raise DemodulationError(f"No demodulator available for {modulation.value}.")
