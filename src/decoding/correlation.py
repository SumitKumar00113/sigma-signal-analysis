"""Bit-stream correlation: frame-period detection and sync-word search.

* :func:`bit_autocorrelation` – normalised autocorrelation of a ±1 mapped
  bit stream; periodic framing shows up as peaks at multiples of the
  frame length.
* :func:`detect_frame_period` – ranks autocorrelation peaks.
* :func:`find_sync_word` – sliding correlation against a known pattern
  (also checks the inverted pattern to survive a BPSK phase flip),
  reporting every position within a Hamming-distance tolerance.
* :func:`split_frames` – cut the stream into frames at the sync positions
  and separate header from payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sp_signal

# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------


def bit_autocorrelation(bits: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Normalised autocorrelation of bits mapped to ±1, lags 0..max_lag."""
    x = 1.0 - 2.0 * np.asarray(bits, dtype=np.float64).reshape(-1)
    n = len(x)
    if n == 0:
        return np.zeros(0)
    if max_lag is None:
        max_lag = min(n // 2, 8192)
    x = x - np.mean(x)
    full = sp_signal.fftconvolve(x, x[::-1], mode="full")
    ac = full[n - 1: n + max_lag]
    if ac[0] <= 0:
        return np.zeros(max_lag + 1)
    return ac / ac[0]


@dataclass
class PeriodCandidate:
    period: int
    strength: float          # autocorrelation value at that lag
    harmonics: int = 0       # how many multiples of the period also peak


def detect_frame_period(
    bits: np.ndarray,
    min_period: int = 8,
    max_period: int | None = None,
    top_n: int = 5,
) -> list[PeriodCandidate]:
    """Rank candidate frame periods from autocorrelation peaks."""
    ac = bit_autocorrelation(bits, max_period)
    if len(ac) <= min_period + 2:
        return []
    seg = ac.copy()
    seg[:min_period] = 0.0
    noise = np.std(seg[min_period:]) if len(seg) > min_period else 0.0
    thresh = max(4.0 * noise, 0.02)
    peaks, props = sp_signal.find_peaks(seg, height=thresh, distance=max(2, min_period // 2))
    if len(peaks) == 0:
        return []
    cands = []
    for p in peaks:
        # Count harmonics present
        h = 0
        for k in range(2, 6):
            lag = p * k
            if lag < len(ac) and ac[lag] > thresh:
                h += 1
        cands.append(PeriodCandidate(int(p), float(ac[p]), h))
    cands.sort(key=lambda c: (c.strength + 0.1 * c.harmonics), reverse=True)
    return cands[:top_n]


# ---------------------------------------------------------------------------
# Sync word search
# ---------------------------------------------------------------------------


def parse_pattern(text: str) -> np.ndarray:
    """Parse a sync word given as binary (``1010...``) or hex (``0x1ACFFC1D``)."""
    t = text.strip().replace(" ", "").replace("_", "")
    if not t:
        return np.zeros(0, dtype=np.uint8)
    if t.lower().startswith("0x"):
        h = t[2:]
        if len(h) % 2:
            h = "0" + h
        return np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8))
    if set(t) <= {"0", "1"}:
        return np.array([int(c) for c in t], dtype=np.uint8)
    raise ValueError("Pattern must be binary digits or 0x-prefixed hex")


@dataclass
class SyncMatch:
    position: int
    errors: int
    inverted: bool


@dataclass
class SyncSearchResult:
    pattern_length: int
    matches: list[SyncMatch] = field(default_factory=list)
    best_period: int | None = None
    correlation: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def positions(self) -> list[int]:
        return [m.position for m in self.matches]


def find_sync_word(
    bits: np.ndarray,
    pattern: np.ndarray | str,
    max_errors: int = 0,
    allow_inverted: bool = True,
) -> SyncSearchResult:
    """Find every occurrence of *pattern* within *max_errors* bit flips."""
    if isinstance(pattern, str):
        pattern = parse_pattern(pattern)
    p = np.asarray(pattern, dtype=np.float64).reshape(-1)
    x = np.asarray(bits, dtype=np.float64).reshape(-1)
    L = len(p)
    if L == 0 or len(x) < L:
        return SyncSearchResult(L)

    # Correlate ±1 versions: perfect match → +L, perfect inverse → −L
    xa = 1.0 - 2.0 * x
    pa = 1.0 - 2.0 * p
    corr = sp_signal.fftconvolve(xa, pa[::-1], mode="valid")
    corr = np.round(corr)
    # errors = (L − corr) / 2
    errs_norm = (L - corr) / 2.0
    errs_inv = (L + corr) / 2.0

    matches: list[SyncMatch] = []
    for pos in np.flatnonzero(errs_norm <= max_errors):
        matches.append(SyncMatch(int(pos), int(errs_norm[pos]), False))
    if allow_inverted:
        for pos in np.flatnonzero(errs_inv <= max_errors):
            matches.append(SyncMatch(int(pos), int(errs_inv[pos]), True))
    matches.sort(key=lambda m: m.position)

    period = None
    if len(matches) >= 2:
        gaps = np.diff([m.position for m in matches])
        vals, counts = np.unique(gaps, return_counts=True)
        period = int(vals[np.argmax(counts)])

    return SyncSearchResult(L, matches, period, corr / L)


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    start: int
    header: np.ndarray
    payload: np.ndarray
    inverted: bool = False


def split_frames(
    bits: np.ndarray,
    sync: SyncSearchResult,
    header_bits: int = 0,
    frame_bits: int | None = None,
    undo_inversion: bool = True,
) -> list[Frame]:
    """Cut *bits* into frames at each sync match.

    The sync word itself is included at the start of ``header``.  If
    *frame_bits* is omitted the frame runs to the next sync match (or to
    the end of the stream).
    """
    x = np.asarray(bits, dtype=np.uint8).reshape(-1)
    frames: list[Frame] = []
    pos = sync.positions
    for i, m in enumerate(sync.matches):
        start = m.position
        if frame_bits is not None:
            end = start + frame_bits
        else:
            end = pos[i + 1] if i + 1 < len(pos) else len(x)
        end = min(end, len(x))
        seg = x[start:end]
        if undo_inversion and m.inverted:
            seg = seg ^ 1
        hb = sync.pattern_length + header_bits
        frames.append(Frame(start, seg[:hb], seg[hb:], m.inverted))
    return frames


def bits_to_hex(bits: np.ndarray, group: int = 8) -> str:
    """Render bits as space-separated hex bytes (partial trailing byte dropped)."""
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    n = (len(b) // 8) * 8
    if n == 0:
        return ""
    return " ".join(f"{v:02X}" for v in np.packbits(b[:n]))


def bits_to_string(bits: np.ndarray) -> str:
    return "".join("1" if v else "0" for v in np.asarray(bits).reshape(-1))
