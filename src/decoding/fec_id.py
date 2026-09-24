"""Blind identification of forward-error-correction codes.

Given a hard-decision bit stream of unknown origin, work out which FEC
(if any) produced it:

* **Convolutional codes (library search)** – every standard code
  (K = 3…9, rate 1/2 and 1/3, all generator orders, DVB/802.11 puncture
  patterns 2/3 … 7/8) is turned into its low-weight *parity checks*
  (vectors of the dual code).  A check applied to a genuine code stream
  gives syndrome bits that are 0 except where channel errors hit it, so
  the fraction of 1s is ``(1 − (1 − 2p)^w) / 2`` for channel BER *p* and
  check weight *w*, versus 0.5 for anything else.  This separates true
  and false hypotheses by tens of standard deviations even at a few
  percent BER, and also yields the code phase (where decoding must
  start), bit inversion, and a channel-BER estimate.
* **Convolutional codes (blind)** – for rate 1/n codes that are not in
  the library, the generators are recovered from the null space of
  windowed stream segments (RANSAC-style over many short segments so a
  noisy stream still has error-free samples), then validated on the
  whole stream with the same syndrome statistic.
* **Reed-Solomon** – for each candidate length *n*, field polynomial and
  bit/byte alignment, blocks are evaluated at powers of α.  An aligned,
  error-free RS codeword evaluates to exactly 0 at every generator root
  (chance: 1/256 per root), which reveals the alignment, the first
  consecutive root (fcr) and the number of parity symbols (n − k).
  The result is confirmed by actually decoding blocks.
* **Unknown linear block codes** – GF(2) rank deficiency of the stream
  arranged into rows of length L reveals the code length / period and
  rate (needs a low-error stream).

:func:`identify_fec` runs these in order and returns a single report.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.core.enums import FECType
from src.decoding.gf2 import gf2_nullspace, gf2_rank, int_to_bits, rows_to_ints
from src.decoding.reed_solomon import GF256, ReedSolomon, bits_to_bytes
from src.decoding.viterbi import STANDARD_CODES, ConvCode, viterbi_decode

ProgressCallback = Callable[[float, str], None]
CancelCheck = Callable[[], bool]

# ---------------------------------------------------------------------------
# Code library
# ---------------------------------------------------------------------------

#: Puncture masks over the interleaved mother-code output (c1, c2, c1, c2, …),
#: using the DVB-S / DVB-T / IEEE 802.11 conventions.
PUNCTURE_PATTERNS: dict[str, tuple[int, ...]] = {
    "2/3": (1, 1, 0, 1),
    "3/4": (1, 1, 0, 1, 1, 0),
    "5/6": (1, 1, 0, 1, 1, 0, 0, 1, 1, 0),
    "7/8": (1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0),
}

#: Additional well-known mother codes beyond :data:`STANDARD_CODES`.
EXTRA_CODES: dict[str, tuple[int, tuple[int, ...]]] = {
    "K5 r1/2 GSM (23,33)": (5, (0o23, 0o33)),
    "K4 r1/2 (15,17)": (4, (0o15, 0o17)),
}


def _octal(gens: tuple[int, ...]) -> str:
    return ",".join(f"{g:o}" for g in gens)


@dataclass(frozen=True)
class ConvHypothesis:
    """One candidate convolutional code (mother code + optional puncturing)."""

    name: str
    constraint_length: int
    generators: tuple[int, ...]
    puncture: tuple[int, ...] | None = None

    def code(self) -> ConvCode:
        return ConvCode(self.constraint_length, self.generators, self.puncture)

    def checks(self) -> tuple[ParityCheck, ...]:
        return conv_parity_checks(self.constraint_length, self.generators, self.puncture)

    @property
    def rate(self) -> float:
        n = len(self.generators)
        if self.puncture:
            return (len(self.puncture) / n) / sum(self.puncture)
        return 1.0 / n


def conv_library(
    include_punctured: bool = True,
    include_permutations: bool = True,
    include_rate_third: bool = True,
) -> list[ConvHypothesis]:
    """All library hypotheses (≈ 70 by default)."""
    bases = {**STANDARD_CODES, **EXTRA_CODES}
    out: list[ConvHypothesis] = []
    seen: set[tuple] = set()
    for name, (k, gens) in bases.items():
        if len(gens) > 2 and not include_rate_third:
            continue
        orders = (dict.fromkeys(itertools.permutations(gens)) if include_permutations
                  else [gens])
        for order in orders:
            label = name if order == gens else f"{name} [order {_octal(order)}]"
            variants: list[tuple[str, tuple[int, ...] | None]] = [(label, None)]
            if include_punctured and len(order) == 2:
                variants += [(f"{label} punctured {r}", p) for r, p in PUNCTURE_PATTERNS.items()]
            for vname, punct in variants:
                key = (k, order, punct)
                if key in seen:
                    continue
                seen.add(key)
                out.append(ConvHypothesis(vname, k, order, punct))
    return out


def common_conv_hypotheses() -> list[ConvHypothesis]:
    """Unpunctured rate-1/2 codes in both generator orders – the default for
    expensive searches (interleaver sweeps) where the full library is too slow."""
    return conv_library(include_punctured=False, include_rate_third=False)


# ---------------------------------------------------------------------------
# Parity checks (dual-code vectors) and the syndrome statistic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParityCheck:
    """A dual-code vector: ``Σ taps[k]·x[p+k] = 0 (mod 2)`` for every window
    start *p* on a code-period boundary (``p ≡ phase mod period``)."""

    taps: tuple[int, ...]
    period: int

    @property
    def length(self) -> int:
        return len(self.taps)

    @property
    def weight(self) -> int:
        return sum(self.taps)

    def array(self) -> np.ndarray:
        return np.asarray(self.taps, dtype=np.uint8)


def _window_generator_rows(
    k: int, gens: tuple[int, ...], pattern: tuple[int, ...], periods: int
) -> tuple[list[int], int]:
    """Rows = contribution of each input bit to a window of *periods*
    puncture periods of (punctured) code output."""
    n = len(gens)
    t_in = len(pattern) // n                     # input bits per puncture period
    steps = periods * t_in
    mask = np.tile(np.asarray(pattern, dtype=bool), periods)
    col_of = np.full(steps * n, -1, dtype=np.int64)
    kept = np.flatnonzero(mask)
    col_of[kept] = np.arange(len(kept))
    rows: list[int] = []
    for j in range(-(k - 1), steps):
        v = 0
        for d in range(k):                       # impulse at j reaches output time j+d
            t = j + d
            if not 0 <= t < steps:
                continue
            for i, g in enumerate(gens):
                if (g >> (k - 1 - d)) & 1:
                    c = col_of[t * n + i]
                    if c >= 0:
                        v ^= 1 << int(c)
        rows.append(v)
    return rows, len(kept)


def _low_weight_vectors(basis: list[int], keep: int = 8, max_enum: int = 12) -> list[int]:
    """Lowest-weight non-zero vectors in span(basis), plus the lightest
    odd-weight one (needed to detect bit inversion) if it exists."""
    if len(basis) <= max_enum:
        vecs = []
        for mask in range(1, 1 << len(basis)):
            v = 0
            for i, b in enumerate(basis):
                if (mask >> i) & 1:
                    v ^= b
            vecs.append(v)
    else:
        vecs = list(basis) + [a ^ b for a, b in itertools.combinations(basis, 2)]
    vecs = [v for v in dict.fromkeys(vecs) if v]
    vecs.sort(key=lambda v: v.bit_count())
    out = vecs[:keep]
    odd = next((v for v in vecs if v.bit_count() % 2), None)
    if odd is not None and odd not in out:
        out.append(odd)
    return out


def _trim_check(taps: np.ndarray, period: int) -> np.ndarray:
    """Drop whole leading periods of zeros (a shift by a full period is the
    same check) and all trailing zeros."""
    nz = np.flatnonzero(taps)
    if len(nz) == 0:
        return taps[:0]
    lead = (int(nz[0]) // period) * period
    return taps[lead: int(nz[-1]) + 1]


@lru_cache(maxsize=1024)
def conv_parity_checks(
    constraint_length: int,
    generators: tuple[int, ...],
    puncture: tuple[int, ...] | None = None,
    max_checks: int = 3,
) -> tuple[ParityCheck, ...]:
    """Low-weight parity checks of a (punctured) rate-1/n convolutional code.

    For an unpunctured rate-1/2 code this finds the classic check
    ``c1·g2 + c2·g1 = 0``; for punctured codes it finds the (heavier)
    checks of the equivalent periodic code.
    """
    k = constraint_length
    n = len(generators)
    pattern = puncture or (1,) * n
    if len(pattern) % n:
        raise ValueError("Puncture pattern length must be a multiple of n")
    t_in = len(pattern) // n
    q = int(sum(pattern))
    if q <= t_in:
        return ()                                # rate ≥ 1: no redundancy
    w0 = (k - 1) // (q - t_in) + 1               # smallest window with redundancy
    found: dict[tuple[int, ...], ParityCheck] = {}
    for periods in range(w0, w0 + 3):
        rows, ncols = _window_generator_rows(k, generators, pattern, periods)
        basis = gf2_nullspace(rows, ncols)
        for v in _low_weight_vectors(basis):
            taps = _trim_check(int_to_bits(v, ncols), q)
            key = tuple(int(b) for b in taps)
            if key:
                found.setdefault(key, ParityCheck(key, q))
    ordered = sorted(found.values(), key=lambda c: (c.weight, c.length))
    chosen = ordered[:max_checks]
    if not any(c.weight % 2 for c in chosen):
        odd = next((c for c in ordered if c.weight % 2), None)
        if odd is not None:
            chosen.append(odd)
    return tuple(chosen)


def detect_inversion(bits: np.ndarray, checks: tuple[ParityCheck, ...], phase: int) -> bool | None:
    """Use the lightest odd-weight check at *phase*: complementing the stream
    flips its syndrome, so a rate near 1 means the stream is inverted.
    Returns None if the code is inversion-invariant (all checks even)."""
    odd = next((c for c in checks if c.weight % 2), None)
    if odd is None:
        return None
    s = syndrome_stream(bits, odd.array())
    s = s[phase::odd.period]
    if len(s) == 0:
        return None
    return bool(np.mean(s) > 0.5)


def syndrome_stream(bits: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """``s[p] = Σ_k taps[k]·bits[p+k] mod 2`` for every window start *p*."""
    x = np.asarray(bits, dtype=np.int32).reshape(-1)
    t = np.asarray(taps, dtype=np.int32).reshape(-1)
    if len(x) < len(t) or len(t) == 0:
        return np.zeros(0, dtype=np.uint8)
    return (np.correlate(x, t, mode="valid") & 1).astype(np.uint8)


def ber_from_syndrome_rate(rate: float, weight: int) -> float:
    """Invert ``rate = (1 − (1 − 2p)^w) / 2`` for the channel BER *p*."""
    base = max(1.0 - 2.0 * rate, 1e-12)
    return float(np.clip((1.0 - base ** (1.0 / max(weight, 1))) / 2.0, 0.0, 0.5))


@dataclass
class SyndromeScore:
    phase: int                  # window-start offset (mod period) of the code
    rate: float                 # fraction of unsatisfied checks (after inversion fix)
    inverted: bool              # stream is bit-inverted (only detectable for odd checks)
    windows: int
    z: float                    # standard deviations below the random-data rate of 0.5
    ber_estimate: float


def informative_windows(bits: np.ndarray, length: int) -> np.ndarray:
    """True for windows that are not constant.  All-0 (idle, padding,
    interleaver start-up) windows satisfy every parity check trivially and
    all-1 windows every even one, so they carry no evidence of a code."""
    x = np.asarray(bits, dtype=np.int64).reshape(-1)
    if len(x) < length:
        return np.zeros(0, dtype=bool)
    c = np.concatenate([[0], np.cumsum(x)])
    ones = c[length:] - c[:-length]
    return (ones > 0) & (ones < length)


def score_check(bits: np.ndarray, check: ParityCheck) -> SyndromeScore | None:
    """Best-phase syndrome statistic of *check* on *bits*."""
    s = syndrome_stream(bits, check.array())
    q = check.period
    nw = len(s) // q
    if nw < 8:
        return None
    valid = informative_windows(bits, check.length)[: nw * q].reshape(nw, q)
    counts = valid.sum(axis=0)
    ones = (s[: nw * q].reshape(nw, q) & valid).sum(axis=0)
    odd = check.weight % 2 == 1
    best: SyndromeScore | None = None
    for phase in range(q):
        m = int(counts[phase])
        if m < 8:
            continue
        f, inv = float(ones[phase]) / m, False
        if odd and f > 0.5:
            f, inv = 1.0 - f, True
        z = (0.5 - f) * 2.0 * math.sqrt(m)
        if best is None or z > best.z:
            best = SyndromeScore(phase, f, inv, m, z, ber_from_syndrome_rate(f, check.weight))
    return best


# ---------------------------------------------------------------------------
# Convolutional identification
# ---------------------------------------------------------------------------


@dataclass
class ConvCandidate:
    """An identified convolutional code and how to decode the stream with it."""

    name: str
    code: ConvCode
    phase: int                  # skip this many bits so decoding starts on a period boundary
    inverted: bool              # invert the stream before decoding
    syndrome_rate: float
    z_score: float
    ber_estimate: float         # channel BER implied by the syndrome rate
    check_weight: int
    blind: bool = False         # recovered without the library
    viterbi_ber: float | None = None   # channel BER measured by a trial Viterbi decode

    @property
    def rate(self) -> float:
        return self.code.rate

    @property
    def puncture_text(self) -> str:
        return ",".join(str(b) for b in self.code.puncture) if self.code.puncture else ""

    def describe(self) -> str:
        parts = [f"{self.name}: K={self.code.constraint_length}, rate {self.rate:.3g}, "
                 f"generators ({_octal(self.code.generators)}) octal"]
        if self.code.puncture:
            parts.append(f"puncture [{self.puncture_text}]")
        parts.append(f"start at bit {self.phase}")
        if self.inverted:
            parts.append("stream inverted")
        parts.append(f"confidence z={self.z_score:.0f}")
        ber = self.viterbi_ber if self.viterbi_ber is not None else self.ber_estimate
        parts.append(f"channel BER ≈ {ber:.2%}")
        return ", ".join(parts)


def _candidate_from_score(name: str, code: ConvCode, check: ParityCheck,
                          sc: SyndromeScore, blind: bool = False) -> ConvCandidate:
    return ConvCandidate(name, code, sc.phase, sc.inverted, sc.rate, sc.z,
                         sc.ber_estimate, check.weight, blind)


def identify_convolutional(
    bits: np.ndarray,
    library: list[ConvHypothesis] | None = None,
    max_bits: int = 60_000,
    z_threshold: float = 6.0,
    top_n: int = 5,
    progress_cb: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> list[ConvCandidate]:
    """Test every library hypothesis; return significant ones, best first."""
    x = np.asarray(bits, dtype=np.uint8).reshape(-1)[:max_bits]
    lib = library if library is not None else conv_library()
    out: list[ConvCandidate] = []
    for i, hyp in enumerate(lib):
        if cancel_check and cancel_check():
            break
        if progress_cb and i % 8 == 0:
            progress_cb(i / max(1, len(lib)), f"Testing {hyp.name}")
        checks = hyp.checks()
        if not checks:
            continue
        sc = score_check(x, checks[0])
        if sc is None or sc.z < z_threshold:
            continue
        out.append(_candidate_from_checks(x, hyp.name, hyp.code(), checks, sc))
    out.sort(key=lambda c: c.z_score, reverse=True)
    return out[:top_n]


def _candidate_from_checks(x: np.ndarray, name: str, code: ConvCode,
                           checks: tuple[ParityCheck, ...], sc: SyndromeScore,
                           blind: bool = False) -> ConvCandidate:
    cand = _candidate_from_score(name, code, checks[0], sc, blind)
    inv = detect_inversion(x, checks, sc.phase)
    if inv is not None:
        cand.inverted = inv
    return cand


def viterbi_confirm(bits: np.ndarray, cand: ConvCandidate, n_bits: int = 4000) -> float:
    """Decode a segment with the candidate and return the fraction of
    received bits that disagree with the re-encoded decision (≈ channel BER)."""
    from src.decoding.viterbi import conv_encode

    x = np.asarray(bits, dtype=np.uint8).reshape(-1)[cand.phase: cand.phase + n_bits]
    if cand.inverted:
        x = x ^ 1
    if len(x) < 256:
        return float("nan")
    decoded = viterbi_decode(x, cand.code, terminated=False).bits
    recoded = conv_encode(decoded, cand.code, terminate=False)
    # The last decisions of an unterminated trellis are unreliable: skip them
    tail = 8 * cand.code.constraint_length * cand.code.n
    m = min(len(recoded), len(x)) - tail
    if m <= 0:
        return float("nan")
    return float(np.mean(recoded[:m] != x[:m]))


def apply_conv_candidate(bits: np.ndarray, cand: ConvCandidate) -> np.ndarray:
    """Viterbi-decode the whole stream with an identified candidate."""
    x = np.asarray(bits, dtype=np.uint8).reshape(-1)[cand.phase:]
    if cand.inverted:
        x = x ^ 1
    return viterbi_decode(x, cand.code, terminated=False).bits


# ---- blind rate-1/n search ------------------------------------------------


def _blind_rate_half(
    x: np.ndarray, max_k: int, z_threshold: float, attempts: int, rng: np.random.Generator,
    phases: tuple[int, ...] = (0, 1),
    cancel_check: CancelCheck | None = None,
) -> tuple[int, int, tuple[int, int], SyndromeScore, ParityCheck] | None:
    """Recover (phase, K, (g1, g2)) of a rate-1/2 code from its output."""
    rejected: set[tuple[int, int, int]] = set()
    for w in range(2, max_k + 1):
        width = 2 * w
        n_rows = width + 8
        span = 2 * (n_rows - 1) + width
        for phase in phases:
            if cancel_check and cancel_check():
                return None
            y = x[phase:]
            if len(y) < span + 2:
                continue
            max_start = (len(y) - span) // 2
            for _ in range(attempts):
                tau = 2 * int(rng.integers(0, max_start + 1))
                win = sliding_window_view(y[tau: tau + span], width)[::2][:n_rows]
                basis = gf2_nullspace(rows_to_ints(win), width)
                if not basis or len(basis) > 2:  # none, or degenerate (e.g. all-zero data)
                    continue
                for v in basis:
                    if (phase, w, v) in rejected:
                        continue
                    taps = int_to_bits(v, width)
                    # a minimal-span check touches both the first and last step
                    if not (taps[0] or taps[1]) or not (taps[-2] or taps[-1]):
                        rejected.add((phase, w, v))
                        continue
                    chk = ParityCheck(tuple(int(t) for t in taps), 2)
                    sc = score_check(y, chk)
                    if sc is None or sc.phase != 0 or sc.z < z_threshold:
                        rejected.add((phase, w, v))
                        continue
                    # c1 at step m pairs with bit m of g2; c2 with bit m of g1
                    g2 = sum(int(taps[2 * m]) << m for m in range(w))
                    g1 = sum(int(taps[2 * m + 1]) << m for m in range(w))
                    return phase, w, (g1, g2), sc, chk
    return None


def blind_conv_search(
    bits: np.ndarray,
    max_k: int = 9,
    rates: tuple[int, ...] = (2, 3),
    max_bits: int = 30_000,
    z_threshold: float = 6.0,
    attempts: int = 60,
    seed: int = 0,
    cancel_check: CancelCheck | None = None,
) -> list[ConvCandidate]:
    """Recover the generators of an unknown unpunctured rate-1/n code."""
    x = np.asarray(bits, dtype=np.uint8).reshape(-1)[:max_bits]
    rng = np.random.default_rng(seed)
    found: list[ConvCandidate] = []

    if 2 in rates:
        res = _blind_rate_half(x, max_k, z_threshold, attempts, rng, cancel_check=cancel_check)
        if res is not None:
            phase, k, gens, _, _ = res
            cand = _validated_candidate(x, k, gens, phase, blind=True)
            if cand is not None:
                found.append(cand)

    if 3 in rates:
        best: ConvCandidate | None = None
        for phase in range(3):
            y = x[phase:]
            m = len(y) // 3
            if m < 64:
                break
            c = y[: 3 * m].reshape(m, 3)
            pair = []
            for other in (1, 2):
                sub = c[:, [0, other]].reshape(-1)
                r = _blind_rate_half(sub, max_k, z_threshold, attempts, rng, phases=(0,),
                                     cancel_check=cancel_check)
                pair.append(r)
            if pair[0] is None or pair[1] is None:
                continue
            (_, ka, (a1, a2), _, _), (_, kb, (b1, b3), _, _) = pair
            gens3 = _combine_rate_third(a1, a2, ka, b1, b3, kb)
            if gens3 is None:
                continue
            k3, gens = gens3
            if k3 > max_k:
                continue
            cand = _validated_candidate(x, k3, gens, phase, blind=True)
            # Other phases describe the same code with a shifted (larger-K)
            # generator set; the minimal K is the canonical description.
            if cand is not None and (best is None or
                                     (k3, -cand.z_score) < (best.code.constraint_length,
                                                            -best.z_score)):
                best = cand
        if best is not None:
            found.append(best)

    # A single valid parity relation does not prove the whole code: rank by
    # how well a trial Viterbi decode explains the stream.
    confirmed: list[tuple[float, ConvCandidate]] = []
    for cand in found:
        ber = viterbi_confirm(x, cand)
        cand.viterbi_ber = ber
        if np.isfinite(ber) and ber < 0.2:
            confirmed.append((ber, cand))
    confirmed.sort(key=lambda t: (round(t[0], 3), -t[1].rate, -t[1].z_score))
    return [c for _, c in confirmed]


# ---- GF(2)[D] polynomial helpers (generator recombination) -----------------


def _to_dpoly(g: int, k: int) -> int:
    """Generator int (bit K−1 = newest input) → polynomial in D (bit d = D^d)."""
    return sum(((g >> (k - 1 - d)) & 1) << d for d in range(k))


def _from_dpoly(p: int, k: int) -> int:
    return sum(((p >> d) & 1) << (k - 1 - d) for d in range(k))


def _pmul(a: int, b: int) -> int:
    out = 0
    while b:
        if b & 1:
            out ^= a
        a <<= 1
        b >>= 1
    return out


def _pdivmod(a: int, b: int) -> tuple[int, int]:
    q = 0
    db = b.bit_length()
    while a and a.bit_length() >= db:
        s = a.bit_length() - db
        q ^= 1 << s
        a ^= b << s
    return q, a


def _pgcd(a: int, b: int) -> int:
    while b:
        a, b = b, _pdivmod(a, b)[1]
    return a


def _combine_rate_third(a1: int, a2: int, ka: int, b1: int, b3: int,
                        kb: int) -> tuple[int, tuple[int, int, int]] | None:
    """Rebuild (g1, g2, g3) from the pairwise blind results
    (a1, a2) = (g1, g2)/gcd(g1, g2) and (b1, b3) = (g1, g3)/gcd(g1, g3)."""
    pa1, pa2 = _to_dpoly(a1, ka), _to_dpoly(a2, ka)
    pb1, pb3 = _to_dpoly(b1, kb), _to_dpoly(b3, kb)
    if not (pa1 and pb1):
        return None
    g1 = _pmul(pa1, pb1)
    g1 = _pdivmod(g1, _pgcd(pa1, pb1))[0]            # lcm
    d12, r1 = _pdivmod(g1, pa1)
    d13, r2 = _pdivmod(g1, pb1)
    if r1 or r2:
        return None
    polys = (g1, _pmul(pa2, d12), _pmul(pb3, d13))
    k = max(p.bit_length() for p in polys)
    return k, tuple(_from_dpoly(p, k) for p in polys)


def _validated_candidate(x: np.ndarray, k: int, gens: tuple[int, ...], phase: int,
                         blind: bool) -> ConvCandidate | None:
    checks = conv_parity_checks(k, gens)
    if not checks:
        return None
    y = x[phase:]
    sc = score_check(y, checks[0])
    if sc is None or sc.z < 6.0:
        return None
    code = ConvCode(k, gens)
    name = f"Blind: K={k} r1/{len(gens)} ({_octal(gens)})"
    cand = _candidate_from_checks(y, name, code, checks, sc, blind)
    cand.phase = (phase + sc.phase) % len(gens)
    return cand


# ---------------------------------------------------------------------------
# Reed-Solomon identification
# ---------------------------------------------------------------------------

#: The 16 primitive polynomials of degree 8, most common first.
PRIMITIVE_POLYS_GF256: tuple[int, ...] = (
    0x11D, 0x187, 0x12B, 0x12D, 0x14D, 0x15F, 0x163, 0x165,
    0x169, 0x171, 0x18D, 0x1A9, 0x1C3, 0x1CF, 0x1E7, 0x1F5,
)

#: Common RS block lengths (bytes): full-length and well-known shortened codes.
#: Searched shortest first: back-to-back codewords of a shortened code also
#: form a valid codeword of any multiple length (x^n·c1 + c2 ≡ 0 mod g).
RS_CANDIDATE_LENGTHS: tuple[int, ...] = (32, 64, 128, 160, 204, 207, 255)

#: First-consecutive-root values probed (generator element α, i.e. prim = 1).
RS_PROBE_FCRS: tuple[int, ...] = (0, 1)

_RS_PROBE_LEN = 12            # syndromes per probe window


@dataclass
class RSCandidate:
    n: int
    k: int
    prim_poly: int
    fcr: int
    bit_offset: int             # skip this many bits so the first block starts aligned
    blocks_tested: int
    blocks_ok: int
    symbols_corrected: int

    @property
    def success_rate(self) -> float:
        return self.blocks_ok / max(1, self.blocks_tested)

    def make_codec(self) -> ReedSolomon:
        return ReedSolomon(self.n, self.k, self.prim_poly, self.fcr, 1)

    def describe(self) -> str:
        return (f"RS({self.n},{self.k}) t={(self.n - self.k) // 2}, field poly "
                f"0x{self.prim_poly:X}, fcr={self.fcr}, start at bit {self.bit_offset}; "
                f"{self.blocks_ok}/{self.blocks_tested} blocks decoded, "
                f"{self.symbols_corrected} symbol errors corrected")


def _gf_mul_vec(gf: GF256, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.broadcast_arrays(a, b)
    out = gf.exp[gf.log[a] + gf.log[b]]
    return np.where((a == 0) | (b == 0), 0, out)


def _rs_syndromes(blocks: np.ndarray, gf: GF256, roots: np.ndarray) -> np.ndarray:
    """``S[..., j] = C(α^roots[j])`` for byte blocks shaped (..., n), where
    ``C(x) = Σ c_i x^(n−1−i)`` (first byte = highest degree)."""
    n = blocks.shape[-1]
    logc = gf.log[blocks]
    zero = blocks == 0
    powers = (n - 1 - np.arange(n))
    out = np.empty(blocks.shape[:-1] + (len(roots),), dtype=np.int64)
    for j, root in enumerate(roots):
        terms = gf.exp[(logc + (int(root) * powers) % 255) % 255]
        terms[zero] = 0
        out[..., j] = np.bitwise_xor.reduce(terms, axis=-1)
    return out


def _sliding_syndromes(data: np.ndarray, n: int, nb: int, gf: GF256,
                       roots: np.ndarray) -> np.ndarray:
    """Syndromes of blocks ``data[o + m·n : o + (m+1)·n]`` for every byte
    offset o ∈ [0, n) and block m ∈ [0, nb) – shape (n, nb, len(roots)).

    Uses ``C'(x) = x·(C(x) − c_0·x^(n−1)) + c_n`` to slide by one byte in
    O(1) per root instead of re-evaluating the polynomial.
    """
    roots = np.asarray(roots, dtype=np.int64)
    starts = np.arange(nb) * n
    s = _rs_syndromes(data[: nb * n].reshape(nb, n), gf, roots)
    out = np.empty((n, nb, len(roots)), dtype=np.int64)
    out[0] = s
    lead = (roots * (n - 1)) % 255
    for o in range(1, n):
        leaving = data[starts + o - 1].astype(np.int64)
        entering = data[starts + o - 1 + n].astype(np.int64)
        t = gf.exp[(gf.log[leaving][:, None] + lead[None, :]) % 255]
        s = s ^ np.where(leaving[:, None] == 0, 0, t)
        s = np.where(s == 0, 0, gf.exp[(gf.log[s] + roots[None, :]) % 255])
        s = s ^ entering[:, None]
        out[o] = s
    return out


def _bm_lengths(seqs: np.ndarray, gf: GF256) -> np.ndarray:
    """Vectorised Berlekamp-Massey over GF(2^8).

    *seqs* is (M, ℓ); returns the linear complexity after each prefix,
    shape (M, ℓ).  For RS syndromes of a block with *e* symbol errors the
    complexity stays at *e* across the generator-root run; a random
    sequence has complexity ≈ ℓ/2.
    """
    m_rows, ell = seqs.shape
    rows = np.arange(m_rows)[:, None]
    c = np.zeros((m_rows, ell + 1), dtype=np.int64)
    c[:, 0] = 1
    bpoly = c.copy()
    length = np.zeros(m_rows, dtype=np.int64)
    shift = np.ones(m_rows, dtype=np.int64)
    bdisc = np.ones(m_rows, dtype=np.int64)
    out = np.zeros((m_rows, ell), dtype=np.int64)
    cols = np.arange(ell + 1)[None, :]
    for step in range(ell):
        d = seqs[:, step].astype(np.int64).copy()
        for i in range(1, step + 1):
            d ^= _gf_mul_vec(gf, c[:, i], seqs[:, step - i])
        nz = d != 0
        coef = np.where(nz, gf.exp[(gf.log[d] - gf.log[bdisc]) % 255], 0)
        src = cols - shift[:, None]
        shifted = np.where(src >= 0, bpoly[rows, np.clip(src, 0, ell)], 0)
        prev = c
        c = np.where(nz[:, None], c ^ _gf_mul_vec(gf, coef[:, None], shifted), c)
        upd = nz & (2 * length <= step)
        length = np.where(upd, step + 1 - length, length)
        bpoly = np.where(upd[:, None], prev, bpoly)
        bdisc = np.where(upd, d, bdisc)
        shift = np.where(upd, 1, shift + 1)
        out[:, step] = length
    return out


def _low_complexity(lengths: np.ndarray) -> np.ndarray:
    """Prefix complexities far below the random ≈ ℓ/2 (chance ≈ 256⁻⁴ each)."""
    ok = np.zeros(lengths.shape[0], dtype=bool)
    for ell, max_l in ((6, 1), (8, 2), (12, 4)):
        if lengths.shape[1] >= ell:
            ok |= lengths[:, ell - 1] <= max_l
    return ok


def identify_reed_solomon(
    bits: np.ndarray,
    lengths: tuple[int, ...] = RS_CANDIDATE_LENGTHS,
    prim_polys: tuple[int, ...] = PRIMITIVE_POLYS_GF256,
    fcrs: tuple[int, ...] = RS_PROBE_FCRS,
    search_blocks: int = 4,
    confirm_blocks: int = 32,
    progress_cb: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> RSCandidate | None:
    """Find RS block length, field, fcr, n−k and alignment in *bits*.

    Blocks may contain symbol errors: detection uses the linear complexity
    of the syndrome sequence (≤ 4 errors per block inside a 12-root probe),
    not exact zero syndromes.  Codes need ≥ 6 parity symbols.
    """
    x = np.asarray(bits, dtype=np.uint8).reshape(-1)
    packs = [bits_to_bytes(x[b:]) for b in range(8)]
    probe_roots = np.arange(max(fcrs) + _RS_PROBE_LEN)
    total = len(lengths) * len(prim_polys) * 8
    step = 0
    for n in sorted(lengths):
        for poly in prim_polys:
            gf = GF256(poly)
            for b in range(8):
                step += 1
                if cancel_check and cancel_check():
                    return None
                if progress_cb and step % 8 == 0:
                    progress_cb(step / total, f"RS search n={n}, poly 0x{poly:X}")
                data = packs[b]
                nb = min(search_blocks, len(data) // n - 1)
                if nb < 1:
                    continue
                synd = _sliding_syndromes(data, n, nb, gf, probe_roots)   # (offset, block, root)
                hits = np.zeros(n, dtype=np.int64)
                complexity = np.full(n, np.iinfo(np.int64).max)
                for f in fcrs:
                    seq = synd[:, :, f: f + _RS_PROBE_LEN].reshape(-1, _RS_PROBE_LEN)
                    lens = _bm_lengths(seq, gf)
                    good = _low_complexity(lens).reshape(n, nb).sum(axis=1)
                    # Complexity over the first 8 roots (inside the root run for
                    # any code with ≥ 8 parity symbols) ≈ errors in the block
                    cplx = (lens[:, 5] + lens[:, 7]).reshape(n, nb).sum(axis=1)
                    better = (good > hits) | ((good == hits) & (cplx < complexity))
                    hits = np.where(better, good, hits)
                    complexity = np.where(better, cplx, complexity)
                if hits.max() < 1:
                    continue
                # A cyclic code also "decodes" when misaligned by ≤ t symbols
                # (the shift looks like errors), so take the alignment with
                # the lowest syndrome complexity, i.e. the fewest errors.
                top = np.flatnonzero(hits == hits.max())
                offset = int(top[np.argmin(complexity[top])])
                cand = _rs_refine(data, n, offset, b, gf, poly, fcrs, confirm_blocks)
                if cand is not None and cand.success_rate >= 0.5:
                    return cand
    return None


def _rs_refine(data: np.ndarray, n: int, offset: int, bit_offset: int, gf: GF256,
               poly: int, fcrs: tuple[int, ...], confirm_blocks: int) -> RSCandidate | None:
    """Pin down fcr and n − k at a detected alignment, then decode to confirm."""
    nb = min(confirm_blocks, (len(data) - offset) // n)
    if nb < 1:
        return None
    blocks = data[offset: offset + nb * n].reshape(nb, n)
    max_roots = min(n - 1, 96)
    synd = _rs_syndromes(blocks, gf, np.arange(max(fcrs) + max_roots))
    trials: list[tuple[int, int]] = []
    for f in fcrs:
        lens = _bm_lengths(synd[:, f: f + max_roots], gf)
        steps = np.arange(1, max_roots + 1)[None, :]
        # Longest prefix whose complexity is still well below random
        low = 2 * lens <= steps - 2
        run = np.where(low.any(axis=1), max_roots - np.argmax(low[:, ::-1], axis=1), 0)
        detected = run[run >= 6]
        if len(detected) == 0:
            continue
        est = int(np.median(detected))
        for nsym in {est - est % 2, est - est % 2 - 2}:
            if 2 <= nsym < n:
                trials.append((f, nsym))
    best: RSCandidate | None = None
    for f, nsym in dict.fromkeys(trials):
        rs = ReedSolomon(n, n - nsym, poly, f, 1)
        ok = corrected = 0
        for blk in blocks:
            r = rs.decode(blk)
            ok += int(r.success)
            corrected += r.corrected
        cand = RSCandidate(n, n - nsym, poly, f, bit_offset + 8 * offset, nb, ok, corrected)
        if best is None or (cand.blocks_ok, n - cand.k) > (best.blocks_ok, n - best.k):
            best = cand
    return best


def apply_rs_candidate(bits: np.ndarray, cand: RSCandidate) -> tuple[np.ndarray, int, int]:
    """Decode all aligned blocks; returns (message bits, blocks_ok, blocks_total)."""
    from src.decoding.reed_solomon import rs_decode_stream

    out, results = rs_decode_stream(np.asarray(bits)[cand.bit_offset:], cand.make_codec())
    return out, sum(1 for r in results if r.success), len(results)


# ---------------------------------------------------------------------------
# Generic linear-structure (rank) detection
# ---------------------------------------------------------------------------


@dataclass
class LinearStructure:
    """Linear redundancy found by GF(2) rank analysis."""

    kind: str                   # "block" or "convolutional-like"
    period: int                 # code length n (block) or output period (conv-like)
    rate_estimate: float
    deficiency: dict[int, int] = field(default_factory=dict)   # row length → rank deficit
    n: int | None = None
    k: int | None = None
    bit_offset: int | None = None

    def describe(self) -> str:
        if self.kind == "block" and self.n:
            return (f"Linear block code n={self.n}, k≈{self.k} (rate ≈ {self.rate_estimate:.2f}), "
                    f"codewords start at bit {self.bit_offset}")
        return (f"Convolutional-like linear structure, period {self.period} bits, "
                f"rate ≈ {self.rate_estimate:.2f}")


def _min_rank(x: np.ndarray, width: int, n_rows: int, starts: list[int]) -> int:
    best = width
    for s in starts:
        seg = x[s: s + width * n_rows]
        if len(seg) < width * n_rows:
            continue
        best = min(best, gf2_rank(rows_to_ints(seg.reshape(n_rows, width))))
    return best


def detect_linear_structure(
    bits: np.ndarray,
    max_len: int = 64,
    trials: int = 6,
    seed: int = 0,
    cancel_check: CancelCheck | None = None,
) -> LinearStructure | None:
    """Rank-deficiency profile → code period and rate (low-BER streams only)."""
    x = np.asarray(bits, dtype=np.uint8).reshape(-1)
    rng = np.random.default_rng(seed)
    deficiency: dict[int, int] = {}
    for width in range(2, max_len + 1):
        if cancel_check and cancel_check():
            return None
        n_rows = width + 12
        need = width * n_rows
        if len(x) < need:
            break
        starts = [int(s) for s in rng.integers(0, len(x) - need + 1, size=trials)]
        d = width - _min_rank(x, width, n_rows, starts)
        if d > 0:
            deficiency[width] = d
    if len(deficiency) < 2:
        return None
    lengths = sorted(deficiency)
    period = 0
    for L in lengths:
        period = math.gcd(period, L)
    if period < 2:
        return None
    ls = np.array(lengths, dtype=float)
    ds = np.array([deficiency[L] for L in lengths], dtype=float)
    slope = float(np.polyfit(ls, ds, 1)[0]) if len(ls) >= 2 else ds[0] / ls[0]
    rate = float(np.clip(1.0 - slope, 0.0, 1.0))

    # Block code test: some alignment gives deficiency at a single period
    need = period * (period + 12)
    best_d, best_off = 0, 0
    if len(x) >= need + period:
        for off in range(period):
            seg_starts = [off + period * int(s) for s in
                          rng.integers(0, (len(x) - need - off) // period + 1, size=trials)]
            d = period - _min_rank(x, period, period + 12, seg_starts)
            if d > best_d:
                best_d, best_off = d, off
    if best_d > 0:
        return LinearStructure("block", period, (period - best_d) / period, deficiency,
                               n=period, k=period - best_d, bit_offset=best_off)
    return LinearStructure("convolutional-like", period, rate, deficiency)


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


@dataclass
class FECIdentification:
    fec_type: FECType
    conv: ConvCandidate | None = None
    conv_candidates: list[ConvCandidate] = field(default_factory=list)
    rs: RSCandidate | None = None
    structure: LinearStructure | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = []
        if self.fec_type == FECType.CONCATENATED and self.conv and self.rs:
            lines.append("Concatenated code: inner convolutional + outer Reed-Solomon")
        if self.conv:
            prefix = "Inner code: " if self.rs else ""
            lines.append(prefix + self.conv.describe())
        if self.rs:
            lines.append(("Outer RS (after Viterbi): " if self.conv else "") + self.rs.describe())
        if self.structure and not (self.conv or self.rs):
            lines.append(self.structure.describe())
        if not lines:
            lines.append("No FEC structure detected (uncoded, unsupported code, "
                         "still interleaved, or too many bit errors).")
        others = [c for c in self.conv_candidates if c is not self.conv]
        if others:
            lines.append("Other candidates: " + "; ".join(
                f"{c.name} (z={c.z_score:.0f})" for c in others[:3]))
        lines += self.notes
        return "\n".join(lines)


def identify_fec(
    bits: np.ndarray,
    try_blind: bool = True,
    try_rs: bool = True,
    try_structure: bool = True,
    max_bits: int = 60_000,
    rs_inner_bits: int = 24_000,
    progress_cb: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
    **_: object,
) -> FECIdentification:
    """Identify the FEC applied to *bits* (see module docstring).

    Accepts the ``progress_cb`` / ``cancel_check`` keywords supplied by the
    GUI :class:`~src.gui.workers.Worker`.
    """
    x = np.asarray(bits, dtype=np.uint8).reshape(-1)
    result = FECIdentification(FECType.UNKNOWN)
    if len(x) < 256:
        result.notes.append("Need at least 256 bits for FEC identification.")
        return result

    def prog(lo: float, hi: float) -> ProgressCallback:
        return lambda f, m: progress_cb(lo + (hi - lo) * f, m) if progress_cb else None

    cancelled = cancel_check or (lambda: False)

    # 1 · Convolutional library
    cands = identify_convolutional(x, max_bits=max_bits, progress_cb=prog(0.0, 0.3),
                                   cancel_check=cancel_check)
    # 2 · Blind rate-1/n
    if not cands and try_blind and not cancelled():
        if progress_cb:
            progress_cb(0.3, "Blind convolutional generator search")
        cands = blind_conv_search(x, cancel_check=cancel_check)
    result.conv_candidates = cands

    if cands:
        best = cands[0]
        if progress_cb:
            progress_cb(0.45, "Confirming with Viterbi")
        best.viterbi_ber = viterbi_confirm(x, best)
        result.conv = best
        result.fec_type = FECType.CONVOLUTIONAL
        if try_rs and not cancelled():
            if progress_cb:
                progress_cb(0.5, "Decoding inner code to look for an outer RS code")
            n_in = min(len(x), rs_inner_bits)
            inner = apply_conv_candidate(x[:n_in], best)
            rs = identify_reed_solomon(inner, progress_cb=prog(0.6, 1.0),
                                       cancel_check=cancel_check)
            if rs is not None:
                result.rs = rs
                result.fec_type = FECType.CONCATENATED
        return result

    # 3 · Reed-Solomon directly on the stream
    if try_rs and not cancelled():
        rs = identify_reed_solomon(x, progress_cb=prog(0.4, 0.8), cancel_check=cancel_check)
        if rs is not None:
            result.rs = rs
            result.fec_type = FECType.REED_SOLOMON
            return result

    # 4 · Any linear structure
    if try_structure and not cancelled():
        if progress_cb:
            progress_cb(0.85, "Rank analysis for unknown linear codes")
        st = detect_linear_structure(x, cancel_check=cancel_check)
        if st is not None:
            result.structure = st
            result.fec_type = FECType.UNKNOWN
            result.notes.append("Linear redundancy found but no library code matched; "
                                "parameters above are estimates.")
            return result

    result.fec_type = FECType.NONE
    return result
