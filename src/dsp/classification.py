"""Feature-based modulation classification.

A decision tree, deliberately simple so its verdicts can be explained to
an analyst.  Each branch is confirmed by the structure the modulation
should show at its symbol centres, which also separates digital keying
from analog modulation (whose symbol-centre values are continuous):

1. **Discrete carrier line with the message in the envelope** (envelope
   variation ≫ frequency variation) → OOK / 2-ASK / 4-ASK when the
   symbol-centre envelope sits on levels, otherwise AM.
2. **Constant envelope** → CPFSK family or analog FM.  Symbol-centre
   frequencies on 4 even levels → 4-FSK; 2 tight levels → 2-FSK, or MSK
   when the modulation index h ≈ 0.5; 2 smeared levels (Gaussian
   pre-filter ISI) → GMSK/GFSK; continuous → FM.
3. **Suppressed carrier, fluctuating envelope** → linear digital
   modulation when there is symbol structure (symbol-rate line or a
   4th-power carrier line): OQPSK if pairing Q half a symbol after I gives
   the cleaner QPSK, π/4-DQPSK if the symbol-to-symbol phase steps are odd
   multiples of 45°, otherwise the cumulant test below.  Without symbol
   structure: SSB if the occupied band is lop-sided, else unknown.

**Cumulant test** (linear modulations).  After timing recovery and
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
from src.core.exceptions import SigmaError
from src.dsp.measurements import (
    bandlimit_to_signal,
    compute_instantaneous_frequency,
    estimate_frequency_offset,
    estimate_snr_inband,
    estimate_symbol_rate_quadrature,
    mpower_carrier,
    mpower_line_pair,
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
    # Refinements the classifier could make from the modulation's structure
    # (used by the pipeline for demodulation when set)
    carrier_hz: float | None = None
    symbol_rate_hz: float | None = None


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


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def _linear_symbols(
    samples: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    cfo_hz: float,
    raw_out: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Timing-recovered, CFO-corrected symbols for cumulant analysis.

    If *raw_out* is given, the symbols *before* the M-th-power CFO
    correction are appended to it (the differential test needs them: for
    π/4-DQPSK that correction locks onto the alternating ±45° pattern).
    """
    x = translate_frequency(samples, sample_rate, -cfo_hz)
    x, sps = resample_to_sps(x, sample_rate, symbol_rate, target_sps=8)
    x = matched_filter(x, sps)
    timing = gardner_timing_recovery(x, sps)
    syms = timing.symbols[min(len(timing.symbols) // 10, 200):]
    if len(syms) < 32:
        return syms
    syms = syms / (np.sqrt(np.mean(np.abs(syms) ** 2)) + 1e-12)
    if raw_out is not None:
        raw_out.append(syms)

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


def carrier_line_fraction(band: np.ndarray, sample_rate: float, cfo_hz: float,
                          search_hz: float = 30.0) -> float:
    """Fraction of the power in a discrete carrier line near *cfo_hz*.

    ≈ 1/(1+m²P) for AM, ≈ 0.5 for OOK, 0 for suppressed-carrier modulations.
    """
    z = translate_frequency(band, sample_rate, -cfo_hz).astype(np.complex128)
    n = len(z)
    if n < 64:
        return 0.0
    nfft = 1 << int(np.ceil(np.log2(4 * n)))
    spec = np.abs(np.fft.fft(z, nfft)) ** 2 / n ** 2
    freqs = np.fft.fftfreq(nfft, 1.0 / sample_rate)
    near = np.abs(freqs) <= max(search_hz, 3 * sample_rate / n)
    total = float(np.mean(np.abs(z) ** 2))
    return float(np.max(spec[near]) / total) if total > 0 else 0.0


def am_fm_ratio(band: np.ndarray, sample_rate: float, cfo_hz: float) -> float:
    """Which parameter carries the message: envelope variation over
    instantaneous-frequency variation (normalised by the occupied band),
    both smoothed over 1 ms.  ≳ 25 for AM/ASK, ≲ 4 for FM (3–25 dB SNR)."""
    x = translate_frequency(band, sample_rate, -cfo_hz)
    w = max(1, int(sample_rate / 1000))
    env = moving_average(np.abs(x), w)
    env_cv = float(np.std(env) / (np.mean(env) + 1e-12))
    f = moving_average(compute_instantaneous_frequency(x, sample_rate), w)
    # Frequency is meaningless where there is no signal (OOK "off" periods)
    present = env[: len(f)] >= 0.5 * np.mean(env)
    edge = 5 * w if len(f) > 20 * w else 0
    present[:edge] = False
    present[len(present) - edge:] = False
    f = f[present]
    _, bw = estimate_snr_inband(x, sample_rate)
    f_norm = float(np.std(f) / bw) if bw > 0 and len(f) else 0.0
    return env_cv / (f_norm + 1e-6)


def _result(mod: ModulationType, confidence: float, alternatives: list[ModulationType],
            evidence: str, feats: dict[str, float],
            symbols: np.ndarray | None = None) -> ClassificationResult:
    cands = [(mod, float(confidence))]
    rest = max(0.0, 1.0 - confidence)
    for i, alt in enumerate(alternatives):
        cands.append((alt, float(rest / (2 ** (i + 1)))))
    return ClassificationResult(mod, float(confidence), cands, feats, [evidence], symbols)


AUDIO_TONE_MIN_HZ = 300.0       # below this the discriminator output is data spectrum
# Share of discriminator power in narrow tones: digital ≲ 0.04, AFSK ≈ 0.15
AUDIO_TONE_FRACTION = 0.1
AUDIO_TONE_MAX_BILEVEL = 0.7    # keyed FSK sits on two levels (real captures ≥ 0.83)
AUDIO_TONE_EVIDENCE = "FM discriminator output is dominated by an audio tone"


def discriminator_tonality(samples: np.ndarray, sample_rate: float,
                           min_hz: float = AUDIO_TONE_MIN_HZ) -> tuple[float, float]:
    """Share of the FM-discriminator output's power in its (up to three)
    strongest narrow spectral lines above *min_hz*, and the strongest
    line's frequency.

    Keyed FSK/CPM gives a random-data discriminator output with a smooth
    spectrum (≲ 0.1).  FM carrying audio – a subcarrier (NOAA APT's
    2400 Hz), AFSK or SSTV tones, a test tone – has sharp lines (≳ 0.3).
    """
    from scipy import signal as sp_signal

    from src.dsp.measurements import bandlimit_to_signal, compute_instantaneous_frequency

    if len(samples) < 4096:
        return 0.0, 0.0
    f = compute_instantaneous_frequency(bandlimit_to_signal(samples, sample_rate), sample_rate)
    f = f - np.mean(f)
    nper = int(min(len(f) // 8, 2 ** int(np.log2(max(256.0, sample_rate / 5.0)))))
    freqs, p = sp_signal.welch(f, sample_rate, nperseg=nper)
    p[:3] = 0.0                                      # residual carrier offset / drift
    total = float(p.sum())
    if total <= 0:
        return 0.0, 0.0
    # Tones are looked for above min_hz, but measured against all the power
    # (low-rate data keeps most of its power below).  Only the power a line
    # adds above the local smooth spectrum counts, so a data spectrum's hump
    # scores ≈ 0 while two or three AFSK / SSTV tones add up.
    background = sp_signal.medfilt(p, kernel_size=61)
    excess = np.clip(p - background, 0.0, None)
    excess = np.where(freqs >= min_hz, excess, 0.0)
    peaks, _ = sp_signal.find_peaks(excess, distance=8)
    if len(peaks) == 0:
        return 0.0, 0.0
    top = peaks[np.argsort(excess[peaks])[::-1][:3]]
    share = sum(float(excess[max(0, k - 3): k + 4].sum()) for k in top) / total
    return share, float(freqs[top[0]])


def discriminator_bilevel(samples: np.ndarray, sample_rate: float) -> float:
    """Fraction of the FM-discriminator output within 20 % (of its 5–95 %
    range) of either extreme.

    Keyed FSK is a square wave between two frequencies (≳ 0.8 on real
    captures, even when the data is periodic and so looks tonal); an
    audio tone is a sinusoid, which spends ≈ 40 % of its time there.
    """
    from src.dsp.measurements import bandlimit_to_signal, compute_instantaneous_frequency

    if len(samples) < 1024:
        return 0.0
    f = compute_instantaneous_frequency(bandlimit_to_signal(samples, sample_rate), sample_rate)
    lo, hi = np.percentile(f, [5, 95])
    r = hi - lo
    if r <= 0:
        return 0.0
    return float(np.mean((np.abs(f - lo) < 0.2 * r) | (np.abs(f - hi) < 0.2 * r)))


def _classify_constant_envelope(samples: np.ndarray, sample_rate: float, symbol_rate: float,
                                rate_conf: float, cfo_hz: float,
                                feats: dict[str, float]) -> ClassificationResult:
    """CPFSK family (2/4-FSK, MSK, GMSK) versus analog FM."""
    from src.dsp.demod import fit_levels, fsk_front_end

    env_cv = feats["env_cv"]
    # The level test itself separates keying from FM (whose symbol-centre
    # frequencies are continuous), so run it whenever a rate was found
    if symbol_rate > 0 and rate_conf >= 0.05:
        try:
            fe = fsk_front_end(samples, sample_rate, symbol_rate, cfo_hz)
            f = fe.freq_symbols_hz
        except SigmaError:
            f = np.zeros(0)
        if len(f) >= 64:
            fit = fit_levels(f, four_level_ratio=0.12)
            c = fit.centroids
            spread = np.abs(f - np.mean(c))
            fill = float(np.std(spread) / (np.mean(spread) + 1e-12))
            h = float((c[-1] - c[0]) / symbol_rate) if len(c) > 1 else 0.0
            feats.update({"fsk_separation": fit.separation, "fsk_var_ratio": fit.variance_ratio,
                          "fsk_fill": fill, "mod_index_h": h})
            if fit.order == 4:
                feats["fsk_order"] = 4.0
                return _result(
                    ModulationType.FSK4, 0.9, [ModulationType.FSK2, ModulationType.GFSK],
                    f"Constant envelope (CV {env_cv:.2f}); symbol-centre frequencies on 4 "
                    f"evenly spaced levels (variance ratio {fit.variance_ratio:.2f}) → 4-FSK.",
                    feats)
            if fit.separation >= 10.0 and fill < 0.25:
                feats["fsk_order"] = 2.0
                if 0.4 <= h <= 0.6:
                    return _result(
                        ModulationType.MSK, 0.85, [ModulationType.FSK2, ModulationType.GMSK],
                        f"Constant envelope; 2 sharp frequency levels with modulation index "
                        f"h = {h:.2f} ≈ 0.5 → MSK (minimum-shift 2-FSK).", feats)
                return _result(
                    ModulationType.FSK2, 0.9, [ModulationType.MSK, ModulationType.FSK4],
                    f"Constant envelope (CV {env_cv:.2f}); 2 sharp symbol-centre frequency "
                    f"levels (separation {fit.separation:.1f}σ), h = {h:.2f} → 2-FSK.", feats)
            if fit.separation >= 3.5:
                feats["fsk_order"] = 2.0
                mod = ModulationType.GMSK if h < 0.45 else ModulationType.GFSK
                return _result(
                    mod, 0.7, [ModulationType.MSK, ModulationType.FSK2],
                    f"Constant envelope; 2 frequency levels smeared by inter-symbol "
                    f"interference (separation {fit.separation:.1f}σ, spread {fill:.2f}) → "
                    f"Gaussian-filtered FSK, apparent h = {h:.2f}.", feats)
    return _result(
        ModulationType.FM, 0.7, [ModulationType.GMSK, ModulationType.FSK2],
        f"Constant envelope (CV {env_cv:.2f}) with a continuous instantaneous frequency "
        f"and no symbol structure (symbol-rate confidence {rate_conf:.2f}) → analog FM.",
        feats)


def _classify_carrier(samples: np.ndarray, sample_rate: float, symbol_rate: float,
                      rate_conf: float, cfo_hz: float,
                      feats: dict[str, float]) -> ClassificationResult:
    """Carrier-bearing amplitude modulation: OOK / ASK versus AM."""
    from src.dsp.demod import envelope_symbols, fit_levels

    cf = feats["carrier_fraction"]
    # As for FSK, the level test separates keying from AM on its own
    if symbol_rate > 0 and rate_conf >= 0.05:
        try:
            v = envelope_symbols(samples, sample_rate, symbol_rate, cfo_hz)[0]
        except SigmaError:
            v = np.zeros(0)
        if len(v) >= 64:
            fit = fit_levels(v)
            c = fit.centroids
            feats.update({"ask_separation": fit.separation,
                          "ask_var_ratio": fit.variance_ratio,
                          "ask_levels": float(fit.order)})
            if fit.order == 2 and c[0] < 0.25 * c[-1]:
                return _result(
                    ModulationType.OOK, 0.9, [ModulationType.ASK, ModulationType.AM],
                    f"Carrier line ({cf:.0%} of power) keyed on/off: envelope at symbol "
                    f"centres on 2 levels ({c[0] / c[-1]:.2f}, 1.00) → OOK.", feats)
            if fit.order in (2, 4):
                return _result(
                    ModulationType.ASK, 0.85, [ModulationType.OOK, ModulationType.AM],
                    f"Carrier line ({cf:.0%} of power); envelope at symbol centres on "
                    f"{fit.order} levels → {fit.order}-ASK.", feats)
    return _result(
        ModulationType.AM, 0.75, [ModulationType.ASK, ModulationType.SSB_USB],
        f"Discrete carrier ({cf:.0%} of power) with a continuously varying envelope and "
        f"no symbol structure → analog AM.", feats)


def classify_modulation(
    samples: np.ndarray,
    sample_rate: float,
    symbol_rate: float,
    snr_db: float | None = None,
    cfo_hz: float | None = None,
    symbol_rate_confidence: float | None = None,
    rate_candidates: list[tuple[float, float]] | None = None,
) -> ClassificationResult:
    """Classify the modulation of *samples*.

    Parameters
    ----------
    symbol_rate : estimated symbol rate (0 if none was found – analog
        modulations do not need one).
    snr_db : optional SNR estimate, used to de-bias the cumulants.
    cfo_hz : optional carrier offset estimate; computed if omitted.
    symbol_rate_confidence : confidence of the symbol-rate estimate.  If
        omitted, a positive *symbol_rate* is taken as reliable.
    rate_candidates : further ``(rate, confidence)`` candidates; for the
        QPSK family the one giving the cleanest constellation is adopted
        (reported in ``result.symbol_rate_hz``).
    """
    result = ClassificationResult()
    if len(samples) < 1024 or sample_rate <= 0:
        result.evidence.append("Insufficient samples.")
        return result
    symbol_rate = max(float(symbol_rate), 0.0)
    rate_conf = (1.0 if symbol_rate > 0 else 0.0) if symbol_rate_confidence is None \
        else float(symbol_rate_confidence)

    if cfo_hz is None:
        cfo_hz = estimate_frequency_offset(samples, sample_rate)

    # Work in-band so out-of-band noise does not dominate the features
    band = bandlimit_to_signal(samples, sample_rate)
    inband_snr, occupied_hz = estimate_snr_inband(samples, sample_rate)
    if snr_db is None:
        snr_db = inband_snr

    feats: dict[str, float] = {"symbol_rate_confidence": rate_conf}
    # Skip the band-limiting filter's start-up and tail transients
    core = band[256:-256] if len(band) > 4096 else band
    feats.update(envelope_features(core))
    if symbol_rate > 0:
        feats.update(instfreq_features(translate_frequency(band, sample_rate, -cfo_hz),
                                       sample_rate, symbol_rate))
    feats["snr_inband_db"] = float(inband_snr)
    feats["carrier_fraction"] = carrier_line_fraction(band, sample_rate, cfo_hz)
    feats["mpower4_line"] = mpower_carrier(band, sample_rate, 4)[2]
    # Noise alone gives a unit-modulus signal an envelope CV of ≈ √(1/(2·SNR));
    # remove that before judging how much the envelope really varies
    snr_lin = 10 ** (inband_snr / 10.0)
    env_cv_signal = float(np.sqrt(max(feats["env_cv"] ** 2 - 0.5 / max(snr_lin, 1e-3), 0.0)))
    feats["env_cv_signal"] = env_cv_signal

    # ---- 0: unmodulated carrier ----------------------------------------------
    # (a lightly modulated AM carrier can look similar, but has sidebands;
    # without noise the "occupied band" is just window leakage, so the
    # bandwidth condition only applies at realistic SNRs)
    narrow = occupied_hz <= 5 * sample_rate / 1024 or inband_snr >= 60.0
    if feats["carrier_fraction"] >= 0.9 and env_cv_signal < 0.03 and narrow:
        result.evidence.append(
            f"Unmodulated carrier: {feats['carrier_fraction']:.0%} of the power in a single "
            f"line at {cfo_hz:+,.1f} Hz and a flat envelope (CW / pilot tone).")
        result.features = feats
        return result

    # ---- 1: discrete carrier line, message in the envelope -----------------
    # (checked first: lightly modulated AM has a nearly flat envelope, while
    # narrowband FM also has a carrier line but its message is in frequency)
    # FM carrying audio (subcarrier, AFSK, SSTV): the discriminator output
    # is tones – continuous-valued, unlike FSK sending periodic data.
    # Checked before the envelope branches – fading and burst edges spoil
    # the flat envelope of a real FM packet – and before the FSK level
    # test, which a sampled tone passes (it is bimodal).
    # Digital and amplitude modulations score ≲ 0.04 here.
    tonal, tone_hz = discriminator_tonality(samples, sample_rate)
    feats["fm_tonality"] = tonal
    feats["fm_tone_hz"] = tone_hz
    if tonal >= AUDIO_TONE_FRACTION and \
            discriminator_bilevel(samples, sample_rate) < AUDIO_TONE_MAX_BILEVEL:
        return _result(
            ModulationType.FM, 0.85, [ModulationType.GFSK, ModulationType.FSK2],
            f"{AUDIO_TONE_EVIDENCE} at {tone_hz:,.0f} Hz ({tonal:.0%} of its power in narrow "
            "lines) → analog FM carrying audio (subcarrier, AFSK or SSTV tones); demodulate "
            "to audio.", feats)

    if feats["carrier_fraction"] >= 0.15:
        feats["am_fm_ratio"] = am_fm_ratio(band, sample_rate, cfo_hz)
        # A strongly fluctuating envelope (keyed carrier) can never be FM
        if feats["am_fm_ratio"] >= 8.0 or feats["env_cv"] >= 0.3:
            return _classify_carrier(samples, sample_rate, symbol_rate, rate_conf, cfo_hz,
                                     feats)

    # ---- 2: constant envelope -----------------------------------------------
    if env_cv_signal < 0.22:
        return _classify_constant_envelope(samples, sample_rate, symbol_rate, rate_conf,
                                           cfo_hz, feats)

    # ---- 3: suppressed carrier ---------------------------------------------
    from src.dsp.analog import analyse_sideband

    sb = analyse_sideband(samples, sample_rate)
    feats["sideband_position"] = sb.position
    line4 = feats["mpower4_line"] >= 30.0
    if sb.sideband != "symmetric" and not line4:
        mod = ModulationType.SSB_USB if sb.sideband == "usb" else ModulationType.SSB_LSB
        feats["ssb_carrier_hz"] = sb.carrier_hz
        where = "low (carrier) edge" if sb.sideband == "usb" else "high (carrier) edge"
        return _result(
            mod, 0.7, [ModulationType.SSB_LSB if sb.sideband == "usb" else ModulationType.SSB_USB,
                       ModulationType.UNKNOWN],
            f"No carrier, fluctuating envelope, and a lop-sided band "
            f"({sb.low_edge_hz:,.0f}–{sb.high_edge_hz:,.0f} Hz, power centred at "
            f"{sb.position:.2f} of the width, near the {where}) → single sideband, "
            f"{sb.sideband.upper()}; suppressed carrier ≈ {sb.carrier_hz:,.0f} Hz.", feats)

    # Lop-sided (SSB) spectra were caught above; a symmetric suppressed-
    # carrier signal with any symbol structure is taken as linear digital
    digital = symbol_rate > 0 and (rate_conf >= 0.2 or line4)
    if not digital:
        result.evidence.append(
            f"Suppressed carrier, fluctuating envelope, symmetric band and no symbol "
            f"structure (symbol-rate confidence {rate_conf:.2f}): possibly DSB-SC, OFDM, "
            "noise, or an unsupported scheme.")
        result.candidates = [(ModulationType.OFDM, 0.2)]
        result.features = feats
        return result

    refined_rate: float | None = None
    carrier: float | None = None
    fc2, _, line2_ratio = mpower_carrier(band, sample_rate, 2)
    feats["mpower2_line"] = line2_ratio
    if line2_ratio >= 30.0:
        # BPSK family: a squared-signal line.  (A real-valued BPSK symbol
        # stream also "fits" QPSK at any rate, so skip the QPSK checks.)
        carrier = fc2
        # Short bursts: random peaks of the data spectrum can beat the
        # symbol-rate line.  Keep the candidate whose symbols have the most
        # constant magnitude: at the right rate and timing raised-cosine BPSK
        # has no inter-symbol interference (±A); elsewhere samples land at
        # random points of the pulses.  (|E[s²]| cannot tell: de-rotated
        # BPSK is real-valued at any sampling instant.)
        rates = [symbol_rate] + [r for r, _ in (rate_candidates or [])[:4]]
        scores: dict[float, float] = {}
        for r in dict.fromkeys(round(v, 1) for v in rates if v > 0):
            syms = _linear_symbols(samples, sample_rate, r, carrier)
            if len(syms) >= 64:
                p = np.abs(syms) ** 2
                scores[r] = float(1.0 - np.std(p) / (np.mean(p) + 1e-12))
        if scores:
            best_rate = max(scores, key=scores.get)
            base = scores.get(round(symbol_rate, 1), 0.0)
            feats["bpsk_fit"] = scores[best_rate]
            if best_rate != round(symbol_rate, 1) and scores[best_rate] > base + 0.15:
                refined_rate = symbol_rate = best_rate
    elif line4:
        # The 4th-power line pins the carrier of the QPSK family exactly
        carrier = mpower_carrier(band, sample_rate, 4)[0]
        from src.dsp.demod import oqpsk_rails, qpsk_fit

        # Verify the symbol rate by the constellation it produces: OQPSK's
        # rate line is weak, so a spurious candidate may have won
        rates = [symbol_rate] + [r for r, _ in (rate_candidates or [])[:3]] + \
            [r for r, _ in estimate_symbol_rate_quadrature(samples, sample_rate)[:3]]
        best: tuple[float, float, float, float] | None = None       # (score, rate, off, al)
        original: tuple[float, float, float, float] | None = None
        for r in dict.fromkeys(round(v, 1) for v in rates if v > 0):
            try:
                off, aligned, *_ = oqpsk_rails(samples, sample_rate, r)
            except SigmaError:
                continue
            fo, fa = qpsk_fit(off), qpsk_fit(aligned)
            entry = (max(fo, fa), r, fo, fa)
            if original is None:
                original = entry
            if best is None or entry[0] > best[0]:
                best = entry
        if best is not None and original is not None:
            # Only move away from the estimated rate for a clearly better fit
            if best[0] < original[0] + 0.15:
                best = original
            _, rate, fit_off, fit_al = best
            feats.update({"oqpsk_fit": fit_off, "qpsk_fit": fit_al})
            if abs(rate - symbol_rate) > 0.01 * symbol_rate and max(fit_off, fit_al) > 0.5:
                refined_rate = symbol_rate = rate
            if fit_off > 0.5 and fit_off > 1.3 * fit_al:
                res = _result(
                    ModulationType.OQPSK, 0.85, [ModulationType.QPSK, ModulationType.MSK],
                    f"4th-power carrier line; sampling Q half a symbol after I gives a clean "
                    f"QPSK constellation (fit {fit_off:.2f} vs {fit_al:.2f} aligned) → "
                    "offset QPSK.", feats)
                res.carrier_hz, res.symbol_rate_hz = carrier, refined_rate
                return res

    res = _classify_linear(samples, sample_rate, symbol_rate, cfo_hz, inband_snr, feats,
                           allow_pi4=line2_ratio < 30.0)
    if res.carrier_hz is None:
        res.carrier_hz = carrier
    if refined_rate is not None:
        res.symbol_rate_hz = refined_rate
    return res


def _classify_linear(samples: np.ndarray, sample_rate: float, symbol_rate: float,
                     cfo_hz: float, inband_snr: float,
                     feats: dict[str, float], allow_pi4: bool = True) -> ClassificationResult:
    """Cumulant test for PSK / QAM, plus the π/4-DQPSK differential test."""
    result = ClassificationResult(features=feats)
    env_cv = feats["env_cv"]
    raw: list[np.ndarray] = []
    syms = _linear_symbols(samples, sample_rate, symbol_rate, cfo_hz, raw_out=raw)
    if len(syms) < 64:
        result.evidence.append("Timing recovery produced too few symbols for classification.")
        return result

    # π/4-DQPSK alternates between two QPSK grids 45° apart, so its 4th
    # power has a *pair* of lines Rs apart instead of one; their midpoint is
    # also an exact carrier, independent of the coarse (centroid) estimate
    band = bandlimit_to_signal(samples, sample_rate)
    pair_carrier, pair_strength, pair_ratio = mpower_line_pair(band, sample_rate, symbol_rate, 4)
    feats["mpower4_pair"] = pair_strength
    if allow_pi4 and pair_strength >= 0.5 and pair_ratio >= 10.0:
        raw_pi4: list[np.ndarray] = []
        _linear_symbols(samples, sample_rate, symbol_rate, pair_carrier, raw_out=raw_pi4)
        from src.dsp.demod import differential_products

        m4d = np.mean(differential_products(raw_pi4[0]) ** 4) if raw_pi4 else 0.0
        feats["diff_m4_real"] = float(np.real(m4d))
        if np.real(m4d) < -0.4:
            result.modulation = ModulationType.PI4_DQPSK
            result.confidence = float(min(1.0, -np.real(m4d) + 0.2))
            result.candidates = [(ModulationType.PI4_DQPSK, result.confidence),
                                 (ModulationType.PSK8, 0.1), (ModulationType.QPSK, 0.05)]
            result.evidence.append(
                f"4th-power line pair {symbol_rate:,.0f} Hz apart (partner {pair_strength:.2f} "
                f"of peak) and symbol-to-symbol phase steps at odd multiples of 45° "
                f"(Re E[d⁴] = {np.real(m4d):.2f}) → π/4-DQPSK.")
            result.symbols = syms
            result.carrier_hz = float(pair_carrier)
            return result

    # Symbol-to-symbol phase steps at odd multiples of 45° give a 4th power
    # of the differential products ≈ −1 (plain QPSK/BPSK: ≈ +1)
    if raw:
        from src.dsp.demod import differential_products

        m4d = np.mean(differential_products(raw[0]) ** 4)
        feats["diff_m4_real"] = float(np.real(m4d))
        if np.real(m4d) < -0.4:
            result.modulation = ModulationType.PI4_DQPSK
            result.confidence = float(min(1.0, -np.real(m4d) + 0.2))
            result.candidates = [(ModulationType.PI4_DQPSK, result.confidence),
                                 (ModulationType.PSK8, 0.1), (ModulationType.QPSK, 0.05)]
            result.evidence.append(
                f"Symbol-to-symbol phase steps are odd multiples of 45° "
                f"(Re E[d⁴] = {np.real(m4d):.2f}) → π/4-DQPSK.")
            result.symbols = syms
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
    if absolute < 0.02:
        result.evidence.append(
            f"Cumulants (|C20|={c20:.2f}, |C40|={c40:.2f}, C42={c42:.2f}) are far from every "
            "known constellation; possibly OFDM, noise, or an unsupported scheme.")
        result.candidates = [(m, float(p)) for m, p in ranked[:2]]
        result.symbols = syms
        return result
    confidence = float(min(1.0, best_p * min(1.0, absolute / 0.5)))

    # 16- and 64-QAM cumulants differ by < 0.1 and the rings blur at
    # moderate SNR: decide by which grid the carrier-recovered symbols fit
    grid_note = ""
    if best_mod in (ModulationType.QAM16, ModulationType.QAM64):
        from src.dsp.demod import demodulate_qam

        try:
            evm16 = demodulate_qam(samples, sample_rate, symbol_rate, 16, cfo_hz).evm_percent
            evm64 = demodulate_qam(samples, sample_rate, symbol_rate, 64, cfo_hz).evm_percent
            feats.update({"evm_qam16": evm16, "evm_qam64": evm64})
            # When noise dominates, the denser 64-point grid "fits" noise
            # better: only prefer it if 16-QAM misfits beyond what noise explains
            evm_noise = 100.0 * 10 ** (-inband_snr / 20.0)
            prefer64 = evm64 < 0.6 * evm16 and evm16 > 1.5 * evm_noise
            order_mod = ModulationType.QAM64 if prefer64 else ModulationType.QAM16
            if order_mod != best_mod:
                ranked = [(order_mod, best_p)] + [(m, p) for m, p in ranked if m != order_mod]
                best_mod = order_mod
            grid_note = (f" Grid fit: EVM {evm16:.1f}% against 16-QAM, "
                         f"{evm64:.1f}% against 64-QAM.")
        except SigmaError:
            pass

    result.modulation = best_mod
    result.confidence = confidence
    result.candidates = [(m, float(p)) for m, p in ranked]
    result.evidence.append(
        f"Fluctuating envelope (CV {env_cv:.2f}) → linear modulation. "
        f"Cumulants |C20|={c20:.2f}, |C40|={c40:.2f}, C42={c42:.2f}; "
        f"{rings} amplitude ring(s).{grid_note}"
    )
    result.features = feats
    result.symbols = syms
    return result
