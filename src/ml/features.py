"""Fixed-length feature vector for the learned modulation classifier.

Every feature is computed for every signal (unlike the rule-based tree,
which only measures what its branch needs) and is normalised so it does
not depend on the sample rate or symbol rate: frequencies are expressed
in symbol-rate or bandwidth units, line strengths as log ratios.  The
building blocks are the same tested measurements the rule-based
classifier uses, so the two agree on what they "see".

Inputs are the pipeline's own estimates (symbol rate + confidence, carrier
offset), not ground truth, so the model learns to live with their errors.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as sp_stats

from src.core.exceptions import SigmaError
from src.dsp.analog import analyse_sideband
from src.dsp.classification import (
    _linear_symbols,
    am_fm_ratio,
    amplitude_level_count,
    carrier_line_fraction,
    cumulant_features,
    instfreq_features,
)
from src.dsp.demod import (
    differential_products,
    envelope_symbols,
    fit_levels,
    fsk_front_end,
    oqpsk_rails,
    qpsk_fit,
)
from src.dsp.measurements import (
    bandlimit_to_signal,
    estimate_snr_inband,
    mpower_carrier,
    mpower_line_pair,
)
from src.dsp.preprocessing import translate_frequency
from src.dsp.sync import moving_average

FEATURE_NAMES: tuple[str, ...] = (
    # envelope
    "env_cv", "env_cv_signal", "env_kurtosis", "env_cv_1ms",
    # carrier / power-law lines
    "carrier_fraction", "log_am_fm_ratio", "log_line1", "log_line2", "log_line4",
    "log_line8", "line4_pair",
    # spectrum
    "sideband_offset", "sideband_asym", "snr_inband", "bw_over_rate", "bw_over_fs",
    "rate_conf", "has_rate",
    # instantaneous frequency
    "if_kurtosis", "if_modes", "if_std_over_rate",
    # linear symbols
    "c20", "c40", "c42", "c21_spread", "amp_rings", "qpsk_fit", "oqpsk_fit",
    "oqpsk_gain", "diff_m4_real", "diff_m4_abs",
    # FSK symbol-centre frequencies
    "fsk_log_sep", "fsk_var_ratio", "fsk_fill", "fsk_h",
    # ASK symbol-centre envelope
    "ask_log_sep", "ask_var_ratio", "ask_low_high",
)

N_FEATURES = len(FEATURE_NAMES)


def _log_ratio(r: float) -> float:
    return float(math.log10(max(r, 1e-3)))


def _clip(v: float, lo: float, hi: float) -> float:
    if not np.isfinite(v):
        return 0.0
    return float(min(max(v, lo), hi))


def extract_features(
    samples: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    symbol_rate_confidence: float,
    cfo_hz: float,
) -> np.ndarray:
    """Feature vector (``float32``, ordered as :data:`FEATURE_NAMES`)."""
    f = dict.fromkeys(FEATURE_NAMES, 0.0)
    x = np.asarray(samples, dtype=np.complex64).reshape(-1)
    fs = float(sample_rate)
    rs = max(float(symbol_rate), 0.0)
    if len(x) < 1024 or fs <= 0:
        return np.zeros(N_FEATURES, dtype=np.float32)

    band = bandlimit_to_signal(x, fs)
    core = band[256:-256] if len(band) > 4096 else band
    snr_db, occupied = estimate_snr_inband(x, fs)
    snr_lin = 10 ** (snr_db / 10.0)

    env = np.abs(core)
    mean_env = float(np.mean(env)) + 1e-12
    env_cv = float(np.std(env) / mean_env)
    f["env_cv"] = _clip(env_cv, 0, 3)
    f["env_cv_signal"] = _clip(math.sqrt(max(env_cv ** 2 - 0.5 / max(snr_lin, 1e-3), 0.0)), 0, 3)
    f["env_kurtosis"] = _clip(float(sp_stats.kurtosis(env, fisher=False)), 0, 50)
    w = max(1, int(fs / 1000))
    smooth = moving_average(env, w)
    f["env_cv_1ms"] = _clip(float(np.std(smooth) / (np.mean(smooth) + 1e-12)), 0, 3)

    f["carrier_fraction"] = _clip(carrier_line_fraction(band, fs, cfo_hz), 0, 1)
    f["log_am_fm_ratio"] = _clip(_log_ratio(am_fm_ratio(band, fs, cfo_hz)), -3, 6)
    for order in (1, 2, 4, 8):
        f[f"log_line{order}"] = _clip(_log_ratio(mpower_carrier(band, fs, order)[2]), -3, 8)

    try:
        sb = analyse_sideband(x, fs)
        f["sideband_offset"] = _clip(sb.position - 0.5, -0.5, 0.5)
        f["sideband_asym"] = abs(f["sideband_offset"])
    except Exception:  # noqa: BLE001
        pass
    f["snr_inband"] = _clip(snr_db, -10, 60)
    f["bw_over_fs"] = _clip(occupied / fs, 0, 1)
    f["rate_conf"] = _clip(float(symbol_rate_confidence), 0, 1)
    f["has_rate"] = 1.0 if rs > 0 else 0.0

    if rs <= 0:
        return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float32)

    f["bw_over_rate"] = _clip(occupied / rs, 0, 50)
    f["line4_pair"] = _clip(mpower_line_pair(band, fs, rs, 4)[1], 0, 2)

    centred = translate_frequency(band, fs, -cfo_hz)
    inst = instfreq_features(centred, fs, rs)
    f["if_kurtosis"] = _clip(inst["if_kurtosis"], 0, 50)
    f["if_modes"] = _clip(inst["if_modes"], 0, 10)
    f["if_std_over_rate"] = _clip(inst["if_spread_hz"] / rs, 0, 20)

    # Linear-modulation view
    try:
        raw: list[np.ndarray] = []
        syms = _linear_symbols(x, fs, rs, cfo_hz, raw_out=raw)
        if len(syms) >= 64:
            cum = cumulant_features(syms, snr_db=snr_db + 3.0)
            f["c20"] = _clip(cum["c20"], 0, 3)
            f["c40"] = _clip(cum["c40"], 0, 3)
            f["c42"] = _clip(cum["c42"], -3, 3)
            mag2 = np.abs(syms) ** 2
            f["c21_spread"] = _clip(float(np.std(mag2) / (np.mean(mag2) + 1e-12)), 0, 5)
            f["amp_rings"] = float(amplitude_level_count(syms))
        if raw:
            m4d = np.mean(differential_products(raw[0]) ** 4)
            f["diff_m4_real"] = _clip(float(np.real(m4d)), -1, 1)
            f["diff_m4_abs"] = _clip(float(np.abs(m4d)), 0, 1)
    except SigmaError:
        pass
    try:
        off, aligned, *_ = oqpsk_rails(x, fs, rs)
        fo, fa = qpsk_fit(off), qpsk_fit(aligned)
        f["qpsk_fit"], f["oqpsk_fit"] = _clip(fa, 0, 1), _clip(fo, 0, 1)
        f["oqpsk_gain"] = _clip(fo / (fa + 1e-3), 0, 10)
    except SigmaError:
        pass

    # FSK view
    try:
        freqs = fsk_front_end(x, fs, rs, cfo_hz).freq_symbols_hz
        if len(freqs) >= 64:
            fit = fit_levels(freqs, four_level_ratio=0.12)
            c = fit.centroids
            spread = np.abs(freqs - np.mean(c))
            f["fsk_log_sep"] = _clip(_log_ratio(fit.separation), -3, 3)
            f["fsk_var_ratio"] = _clip(fit.variance_ratio, 0, 1.5)
            f["fsk_fill"] = _clip(float(np.std(spread) / (np.mean(spread) + 1e-12)), 0, 5)
            f["fsk_h"] = _clip(float((c[-1] - c[0]) / rs), 0, 10)
    except SigmaError:
        pass

    # ASK view
    try:
        v = envelope_symbols(x, fs, rs, cfo_hz)[0]
        if len(v) >= 64:
            fit = fit_levels(v)
            f["ask_log_sep"] = _clip(_log_ratio(fit.separation), -3, 3)
            f["ask_var_ratio"] = _clip(fit.variance_ratio, 0, 1.5)
            f["ask_low_high"] = _clip(float(fit.centroids[0] / (fit.centroids[-1] + 1e-12)), -1, 1)
    except SigmaError:
        pass

    return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float32)
