"""Sample-rate inference for headerless recordings.

A raw IQ file carries no sample rate, and nothing in the samples
themselves reveals the *absolute* rate: the same byte stream played at
1 MHz or 2 MHz looks identical apart from a time scale.  What *can* be
measured is the symbol rate in **cycles per sample**.  Combining that
with the assumption that the transmitter used a standard symbol rate,
or that the receiver used a standard SDR sample rate, yields a short
list of consistent candidates for the analyst to pick from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.dsp.measurements import estimate_symbol_rate_candidates

# Common over-the-air symbol rates (baud)
STANDARD_SYMBOL_RATES: tuple[float, ...] = (
    50, 75, 100, 300, 600, 1200, 2400, 4800, 9600, 19200, 38400,
    8000, 10000, 12500, 16000, 20000, 25000, 32000, 48000, 50000,
    64000, 100000, 128000, 250000, 256000, 500000, 1000000,
)

# Common SDR / recorder sample rates (Hz)
STANDARD_SAMPLE_RATES: tuple[float, ...] = (
    8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000, 192000,
    200000, 250000, 256000, 500000, 1000000, 1024000, 1200000,
    1800000, 2000000, 2048000, 2400000, 2500000, 2560000, 3000000,
    3200000, 4000000, 5000000, 6000000, 8000000, 10000000, 12000000,
    16000000, 20000000, 25000000, 30720000, 40000000, 50000000, 61440000,
)


@dataclass
class SampleRateCandidate:
    sample_rate_hz: float
    implied_symbol_rate_hz: float
    samples_per_symbol: float
    basis: str                     # why this candidate was proposed
    score: float


@dataclass
class SampleRateInference:
    symbol_rate_cycles_per_sample: float = 0.0
    confidence: float = 0.0
    candidates: list[SampleRateCandidate] = field(default_factory=list)
    note: str = ""


def infer_sample_rate_candidates(
    samples: np.ndarray,
    max_candidates: int = 10,
) -> SampleRateInference:
    """Propose plausible sample rates for a headerless recording.

    The symbol rate is estimated with a nominal sample rate of 1.0, giving
    cycles/sample.  Each standard symbol rate then implies a sample rate,
    and each standard sample rate implies a symbol rate; candidates whose
    implied value is itself "round" score highest.
    """
    out = SampleRateInference()
    if len(samples) < 4096:
        out.note = "Too few samples to estimate a symbol rate."
        return out

    # Estimate at a nominal 1 MHz so the estimator's absolute-Hz floors
    # behave, then convert to cycles per sample.
    nominal = 1e6
    cands = estimate_symbol_rate_candidates(samples, nominal)
    if not cands:
        out.note = "No symbol-rate line found; the recording may be analog or noise."
        return out

    cps, conf = cands[0][0] / nominal, cands[0][1]
    out.symbol_rate_cycles_per_sample = cps
    out.confidence = conf
    if cps <= 0 or cps >= 0.5:
        out.note = "Symbol-rate estimate is outside the usable range."
        return out

    results: list[SampleRateCandidate] = []

    # Basis A: assume a standard symbol rate → sample rate = Rs / cps
    for rs in STANDARD_SYMBOL_RATES:
        fs = rs / cps
        if fs < 1000 or fs > 200e6:
            continue
        sps = fs / rs
        # Bonus when the implied sample rate is itself a standard one
        nearest = min(STANDARD_SAMPLE_RATES, key=lambda s: abs(s - fs))
        closeness = abs(nearest - fs) / fs
        score = conf * (1.0 + (1.0 if closeness < 0.01 else 0.0)) * min(1.0, sps / 4.0)
        results.append(SampleRateCandidate(
            sample_rate_hz=float(nearest if closeness < 0.01 else fs),
            implied_symbol_rate_hz=float(rs),
            samples_per_symbol=float(sps),
            basis=f"standard symbol rate {rs:g} baud",
            score=float(score),
        ))

    # Basis B: assume a standard sample rate → symbol rate = fs · cps
    for fs in STANDARD_SAMPLE_RATES:
        rs = fs * cps
        sps = 1.0 / cps
        nearest = min(STANDARD_SYMBOL_RATES, key=lambda r: abs(r - rs))
        closeness = abs(nearest - rs) / rs
        score = conf * (1.0 + (1.0 if closeness < 0.01 else 0.0)) * 0.9 * min(1.0, sps / 4.0)
        results.append(SampleRateCandidate(
            sample_rate_hz=float(fs),
            implied_symbol_rate_hz=float(rs),
            samples_per_symbol=float(sps),
            basis=f"standard SDR rate {fs / 1e6:g} MHz",
            score=float(score),
        ))

    # De-duplicate on sample rate (keep highest score)
    best: dict[int, SampleRateCandidate] = {}
    for c in results:
        key = int(round(c.sample_rate_hz))
        if key not in best or c.score > best[key].score:
            best[key] = c
    ranked = sorted(best.values(), key=lambda c: c.score, reverse=True)
    out.candidates = ranked[:max_candidates]
    out.note = (
        f"Symbol rate ≈ {cps:.5f} cycles/sample ({1 / cps:.1f} samples/symbol). "
        "The absolute sample rate cannot be recovered from a headerless file; "
        "candidates assume either a standard symbol rate or a standard SDR sample rate."
    )
    return out
