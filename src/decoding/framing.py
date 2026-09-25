"""Automatic frame synchronisation and header discovery.

Given a (decoded) bit stream, find where frames start and which bits form
the header, without being told the sync word:

1. **Known sync words** – a small library of published markers (CCSDS
   ASM, POCSAG, IRIG-106, DMR, P25, …) is correlated against the stream,
   allowing a few bit errors and inversion.  Short markers (MPEG-TS
   ``0x47``, the GPS preamble) are only accepted when they recur on a
   regular grid.
2. **Blind discovery** – a sync word is a bit pattern that recurs far
   more often than chance.  Every *W*-bit window (W = 32, 24, 16) is
   counted, together with its complement (a phase flip inverts it); the
   most frequent pattern is kept if its count is significant under a
   Poisson model of random data.  Low-complexity patterns (idle fill such
   as ``0000…`` or ``0101…``) are ignored.

Either way the occurrences are then aligned and described:

* the pattern is extended bit by bit while the bit stays constant across
  frames – the **constant block** (sync word plus any fixed header
  fields);
* the frame **period** is the most common spacing (``regular`` when
  nearly all frames have it);
* fields just after the constant block that count up by one per frame
  are reported as **frame counters**; the header is the constant block
  plus those counters, the rest of the frame is payload.

Every statistical decision uses a false-alarm probability (``1e-6`` by
default), so random or scrambled data yields "no framing found".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from src.decoding.correlation import (
    Frame,
    PeriodCandidate,
    SyncMatch,
    SyncSearchResult,
    bits_to_hex,
    bits_to_string,
    detect_frame_period,
    find_sync_word,
    parse_pattern,
    split_frames,
)


@dataclass(frozen=True)
class KnownSync:
    name: str
    pattern: str                # hex (0x…) or binary
    period: int | None = None   # frame length in bits, when fixed by the standard

    def bits(self) -> np.ndarray:
        return parse_pattern(self.pattern)


KNOWN_SYNC_WORDS: tuple[KnownSync, ...] = (
    KnownSync("CCSDS attached sync marker", "0x1ACFFC1D"),
    KnownSync("CCSDS 64-bit ASM (turbo/LDPC)", "0x034776C7272895B0"),
    KnownSync("CCSDS telecommand start sequence", "0xEB90"),
    KnownSync("POCSAG frame sync", "0x7CD215D8"),
    KnownSync("IRIG-106 PCM frame sync", "0xFE6B2840"),
    KnownSync("DMR base-station data sync", "0xDFF57D75DF5D"),
    KnownSync("DMR base-station voice sync", "0x755FD7DF75F7"),
    KnownSync("APCO P25 frame sync", "0x5575F5FF77FF"),
    # On-air bit order (bytes 10 B6 CA 11 22 96 12 F8, LSB first)
    KnownSync("Vaisala RS41 radiosonde header",
              "0000100001101101010100111000100001000100011010010100100000011111"),
    KnownSync("Barker-13", "1111100110101"),
    KnownSync("MPEG-TS sync byte", "0x47", 188 * 8),
    KnownSync("GPS LNAV TLM preamble", "0x8B", 300),
)

BLIND_WINDOWS = (32, 24, 16)
MIN_FRAMES = 3
CONSTANT_AGREEMENT = 0.9        # a column is "constant" if ≥90 % of frames agree
MAX_CONSTANT_BITS = 512
MIN_FRAME_BITS = 64            # shorter "frames" are repeating patterns (idle, phasing)
MAX_CONSTANT_PAYLOAD = 0.9      # payload columns constant across frames: a repeated message
COUNTER_SEARCH_BITS = 64        # look for counters this far past the constant block
COUNTER_WIDTHS = range(4, 33)
COUNTER_FRACTION = 0.5          # of frame-to-frame steps equal to +1 (chance: 2^-width)
COUNTER_FALSE_ALARM = 1e-4      # … and significant against chance over all positions/widths


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class HeaderField:
    kind: str                   # "sync", "fixed", "counter"
    start: int                  # bit offset from the frame start
    length: int
    description: str


@dataclass
class FramingResult:
    found: bool = False
    method: str = "none"                    # "known", "blind" or "none"
    sync_name: str = ""
    sync_bits: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint8))
    constant_bits: int = 0                  # sync + fixed header fields
    header_bits: int = 0                    # constant block + counters
    period: int | None = None
    regular: bool = False
    sync: SyncSearchResult | None = None
    frames: list[Frame] = field(default_factory=list)
    fields: list[HeaderField] = field(default_factory=list)
    period_candidates: list[PeriodCandidate] = field(default_factory=list)
    false_alarm: float = 1.0                # probability the sync is a chance pattern
    notes: list[str] = field(default_factory=list)

    @property
    def sync_hex(self) -> str:
        b = self.sync_bits
        if len(b) % 8 == 0 and len(b):
            return "0x" + bits_to_hex(b).replace(" ", "")
        return bits_to_string(b)

    @property
    def inverted_frames(self) -> int:
        return sum(1 for f in self.frames if f.inverted)

    def summary(self) -> str:
        if not self.found:
            lines = ["No frame sync found (random, scrambled, or unframed data)."]
            if self.period_candidates:
                p = self.period_candidates[0]
                lines.append(f"Autocorrelation peak at {p.period} bits (r={p.strength:.2f}).")
            return "\n".join(lines + self.notes)
        how = self.sync_name if self.method == "known" else "discovered blindly"
        lines = [f"Sync word {self.sync_hex} ({len(self.sync_bits)} bits, {how}); "
                 f"{len(self.frames)} frames"
                 + (f", {self.inverted_frames} inverted" if self.inverted_frames else "")
                 + (f", false-alarm probability {self.false_alarm:.1e}"
                    if self.false_alarm > 0 else "")]
        if self.period:
            lines.append(f"Frame length {self.period} bits"
                         + (" (regular)" if self.regular else " (most common spacing)"))
        lines.append(f"Header {self.header_bits} bits, payload "
                     f"{(self.period or 0) - self.header_bits} bits"
                     if self.period else f"Header {self.header_bits} bits")
        for f in self.fields:
            lines.append(f"  bits {f.start}–{f.start + f.length - 1}: {f.description}")
        return "\n".join(lines + self.notes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _match_probability(length: int, max_errors: int, inverted: bool = True) -> float:
    """Probability a random window matches a *length*-bit pattern."""
    p = sum(math.comb(length, e) for e in range(max_errors + 1)) / 2.0 ** length
    return min(1.0, p * (2 if inverted else 1))


def _default_errors(length: int) -> int:
    return 0 if length <= 16 else length // 10


def _sliding_words(x: np.ndarray, w: int) -> np.ndarray:
    """Integer value of every *w*-bit window (MSB first), w ≤ 63."""
    n = len(x) - w + 1
    v = np.zeros(n, dtype=np.int64)
    for j in range(w):
        v = (v << 1) | x[j: j + n].astype(np.int64)
    return v


def _periodic(words: np.ndarray, w: int, max_period: int) -> np.ndarray:
    out = np.zeros(len(words), dtype=bool)
    for p in range(1, max_period + 1):
        mask = (1 << (w - p)) - 1
        out |= ((words >> p) & mask) == (words & mask)
    return out


def _low_complexity(words: np.ndarray, w: int, max_period: int = 8) -> np.ndarray:
    """Windows that are idle fill (``0000…``, ``0101…``, a repeated byte)
    over at least ¾ of their length – e.g. the end of a zero run followed
    by data."""
    part = (3 * w) // 4
    return (_periodic(words, w, max_period)
            | _periodic(words >> (w - part), part, max_period)
            | _periodic(words & ((1 << part) - 1), part, max_period))


def _mode_spacing(positions: np.ndarray, min_gap: int) -> tuple[int | None, float]:
    """Most common spacing between consecutive positions and the fraction
    of frames that have it."""
    if len(positions) < 2:
        return None, 0.0
    gaps = np.diff(np.sort(positions))
    gaps = gaps[gaps >= min_gap]
    if len(gaps) == 0:
        return None, 0.0
    vals, counts = np.unique(gaps, return_counts=True)
    i = int(np.argmax(counts))
    return int(vals[i]), float(counts[i] / len(gaps))


def _grid_chain(positions: np.ndarray, period: int) -> np.ndarray:
    """Positions that have another position exactly one period before or after."""
    s = set(positions.tolist())
    return np.array([p for p in positions if p + period in s or p - period in s], dtype=np.int64)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


@dataclass
class _Occurrences:
    positions: np.ndarray       # start of the core pattern in each frame
    inverted: np.ndarray        # bool per occurrence
    core_len: int
    false_alarm: float
    name: str = ""
    method: str = "blind"
    period_hint: int | None = None
    pattern: np.ndarray | None = None


def _search_known(x: np.ndarray, library: tuple[KnownSync, ...],
                  false_alarm: float) -> _Occurrences | None:
    best: _Occurrences | None = None
    n = len(x)
    for ks in library:
        pat = ks.bits()
        L = len(pat)
        if n < 4 * L:
            continue
        errs = _default_errors(L)
        res = find_sync_word(x, pat, max_errors=errs)
        if len(res.matches) < MIN_FRAMES:
            continue
        pos = np.array(res.positions, dtype=np.int64)
        inv = np.array([m.inverted for m in res.matches])
        q = _match_probability(L, errs)
        trials = len(library)
        # Isolated matches: significant only if the pattern is long enough
        pa = float(stats.poisson.sf(len(pos) - 1, n * q)) * trials
        period = ks.period or _mode_spacing(pos, L)[0]
        if period and pa > false_alarm:
            # Short patterns: count only matches on a regular grid
            chained = _grid_chain(pos, period)
            if len(chained) >= MIN_FRAMES:
                # each chained match needed a partner at ±period by chance
                pg = float(stats.binom.sf(len(chained) - 1, len(pos), 2 * q)) * trials * (
                    1 if ks.period else n)          # any spacing could have been picked
                if pg <= false_alarm:
                    keep = np.isin(pos, chained)
                    pos, inv, pa = pos[keep], inv[keep], pg
        if pa > false_alarm:
            continue
        cand = _Occurrences(pos, inv, L, pa, ks.name, "known", ks.period, pat)
        if best is None or (len(cand.positions), -cand.false_alarm) > \
                (len(best.positions), -best.false_alarm):
            best = cand
    return best


def _search_blind(x: np.ndarray, false_alarm: float) -> _Occurrences | None:
    """Most significant recurring pattern over all window lengths (a window
    that straddles the sync word and a few chance bits recurs too, but in
    far fewer frames)."""
    n = len(x)
    best: _Occurrences | None = None
    for w in BLIND_WINDOWS:
        if n < 8 * w:
            continue
        words = _sliding_words(x, w)
        full = (1 << w) - 1
        canon = np.minimum(words, words ^ full)
        ok = ~_low_complexity(words, w)
        vals, counts = np.unique(canon[ok], return_counts=True)
        if len(counts) == 0:
            continue
        i = int(np.argmax(counts))
        c = int(counts[i])
        lam = len(words) * 2.0 / 2.0 ** w
        # Bonferroni over the 2^(w-1) canonical words and the window lengths
        logp = float(stats.poisson.logsf(c - 1, lam)) + (w - 1) * math.log(2) \
            + math.log(len(BLIND_WINDOWS))
        if c < MIN_FRAMES or logp > math.log(false_alarm):
            continue
        if best is not None and logp >= math.log(max(best.false_alarm, 1e-300)):
            continue
        idx = np.flatnonzero(ok & (canon == vals[i]))
        keep = [idx[0]]                          # drop overlaps (self-similar words)
        for p in idx[1:]:
            if p - keep[-1] >= w:
                keep.append(p)
        pos = np.asarray(keep, dtype=np.int64)
        inv = words[pos] != vals[i]
        if inv.mean() > 0.5:
            inv = ~inv
        best = _Occurrences(pos, inv, w, math.exp(max(logp, -690.0)))
    return best


# ---------------------------------------------------------------------------
# Description of the frames
# ---------------------------------------------------------------------------


def _columns(x: np.ndarray, starts: np.ndarray, inv: np.ndarray, offsets: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
    """Bits at ``start + offset`` for every frame (inversion undone) and a
    validity mask for positions outside the stream."""
    idx = starts[:, None] + offsets[None, :]
    valid = (idx >= 0) & (idx < len(x))
    vals = x[np.clip(idx, 0, len(x) - 1)] ^ inv[:, None].astype(np.uint8)
    return vals, valid


def _is_constant(vals: np.ndarray, valid: np.ndarray) -> tuple[bool, int]:
    k = int(valid.sum())
    if k < MIN_FRAMES:
        return False, 0
    ones = int((vals & valid).sum())
    agree = max(ones, k - ones) / k
    return agree >= CONSTANT_AGREEMENT, int(ones * 2 > k)


def _constant_block(x: np.ndarray, occ: _Occurrences, max_right: int,
                    extend_left: bool, core: np.ndarray | None = None) -> tuple[int, np.ndarray]:
    """Extend the core pattern while bits stay constant across frames.
    Returns (offset of the block relative to the core, block bits)."""
    left: list[int] = []
    if extend_left:
        d = -1
        while len(left) < MAX_CONSTANT_BITS // 2:
            vals, valid = _columns(x, occ.positions, occ.inverted, np.array([d]))
            const, bit = _is_constant(vals[:, 0], valid[:, 0])
            if not const:
                break
            left.append(bit)
            d -= 1
    right: list[int] = []
    d = 0
    while d < max_right:
        vals, valid = _columns(x, occ.positions, occ.inverted, np.array([d]))
        const, bit = _is_constant(vals[:, 0], valid[:, 0])
        if not const and d >= occ.core_len:
            break
        right.append(int(core[d]) if core is not None and d < len(core) else bit)
        d += 1
    bits = np.array(left[::-1] + right, dtype=np.uint8)
    return -len(left), bits


def _find_counters(frames: list[Frame], start: int, max_len: int) -> list[HeaderField]:
    """Fields (bit ranges from the frame start) that increase by one per frame."""
    if len(frames) < 4:
        return []
    rows = [np.concatenate([f.header, f.payload]) for f in frames]
    width = min(len(r) for r in rows)
    end_limit = min(width, start + max_len)
    if end_limit - start < min(COUNTER_WIDTHS):
        return []
    m = np.stack([r[:end_limit] for r in rows]).astype(np.int64)
    out: list[HeaderField] = []
    pos = start
    while pos < end_limit:
        best = None
        for w in COUNTER_WIDTHS:
            if pos + w > end_limit:
                break
            v = np.zeros(len(m), dtype=np.int64)
            for j in range(w):
                v = (v << 1) | m[:, pos + j]
            d = np.mod(np.diff(v), 1 << w)
            hits = int(np.sum(d == 1))
            frac = hits / len(d)
            trials = COUNTER_SEARCH_BITS * len(COUNTER_WIDTHS)
            p_chance = float(stats.binom.sf(hits - 1, len(d), 2.0 ** -w)) * trials
            if frac >= COUNTER_FRACTION and p_chance <= COUNTER_FALSE_ALARM:
                best = (w, int(v[0]))
            elif best is not None:
                break
        if best is None:
            pos += 1
            continue
        w, first = best
        out.append(HeaderField("counter", pos, w,
                               f"frame counter ({w} bits, +1 per frame, first value {first})"))
        pos += w
    return out


def _locate(x: np.ndarray, pattern: np.ndarray, period: int | None) -> SyncSearchResult:
    """All frame starts of *pattern* (errors and inversion allowed)."""
    L = len(pattern)
    sync = find_sync_word(x, pattern, max_errors=_default_errors(L))
    if period and L < 24 and sync.matches:
        # A short word also matches in the payload: keep the dominant grid phase
        phase = np.array(sync.positions) % period
        vals, counts = np.unique(phase, return_counts=True)
        best = vals[int(np.argmax(counts))]
        sync.matches = [m for m, ph in zip(sync.matches, phase, strict=True) if ph == best]
    matches: list[SyncMatch] = []
    for mt in sync.matches:              # overlapping matches: keep the better one
        if matches and mt.position - matches[-1].position < L:
            if mt.errors < matches[-1].errors:
                matches[-1] = mt
            continue
        matches.append(mt)
    sync.matches = matches
    return sync


def _field_value(frames: list[Frame], start: int, length: int) -> int:
    f = frames[0]
    row = np.concatenate([f.header, f.payload])[start: start + length]
    return int("".join(str(int(b)) for b in row) or "0", 2)


def _describe(x: np.ndarray, occ: _Occurrences, result: FramingResult) -> None:
    period = occ.period_hint or _mode_spacing(occ.positions, occ.core_len)[0]
    max_right = min(period or MAX_CONSTANT_BITS, MAX_CONSTANT_BITS)
    blind = occ.method == "blind"
    shift, block = _constant_block(x, occ, max_right, extend_left=blind, core=occ.pattern)
    if len(block) >= (period or 1 << 30):          # the whole frame repeats
        block = block[: max(occ.core_len, 1)]
        result.notes.append("Frames have identical content (repeated message).")
    # A known marker is the sync word and constant bits after it are fixed
    # header fields; blindly the two cannot be told apart
    sync_len = len(block) if blind else occ.core_len
    sync = _locate(x, block[:sync_len], period)
    pos = np.array(sync.positions, dtype=np.int64)
    if len(pos) < MIN_FRAMES:
        return
    period, frac = _mode_spacing(pos, sync_len)
    if occ.period_hint:
        period = occ.period_hint
        frac = float(np.mean(np.diff(pos) == period))
    sync.best_period = period
    regular = bool(period) and frac >= 0.8
    frame_bits = period if regular else None
    if period and period < MIN_FRAME_BITS:
        result.notes.append(f"A {sync_len}-bit pattern repeats every {period} bits: idle, "
                            "phasing or a repeated character, not a frame structure.")
        return
    if period:
        # The payload must carry data: mostly constant columns = a repeated message
        span = np.arange(len(block), min(period, len(block) + 512))
        if len(span):
            vals, valid = _columns(x, pos, np.array([m.inverted for m in sync.matches]), span)
            constant = sum(_is_constant(vals[:, j], valid[:, j])[0] for j in range(len(span)))
            if constant > MAX_CONSTANT_PAYLOAD * len(span):
                result.notes.append(f"Pattern repeats every {period} bits with an almost "
                                    "constant payload: a repeated message or idle sequence.")
                return

    frames = split_frames(x, sync, frame_bits=frame_bits)
    counters = _find_counters(frames, len(block), COUNTER_SEARCH_BITS)
    if counters and counters[0].start == len(block):
        # The high bits of a counter do not change in a short recording and
        # were taken as constant: give the counter the trailing zeros, up
        # to a whole number of bytes
        c = counters[0]
        floor = occ.core_len if not blind else min(len(block), 16)
        zeros = 0
        while len(block) - zeros > floor and block[len(block) - zeros - 1] == 0:
            zeros += 1
        widths = [w for w in range(8, 33, 8) if c.length <= w <= c.length + zeros]
        absorb = (max(widths) - c.length) if widths else 0
        if absorb:
            block = block[: len(block) - absorb]
            c.start -= absorb
            c.length += absorb
            if blind:
                sync_len = len(block)
                sync.pattern_length = sync_len
    sync_bits = block[:sync_len]
    fields = [HeaderField("sync", 0, sync_len,
                          ("sync word / fixed header " if blind else "sync word ")
                          + _fmt(sync_bits))]
    if len(block) > sync_len:
        fixed = block[sync_len:]
        fields.append(HeaderField("fixed", sync_len, len(fixed), f"constant field {_fmt(fixed)}"))
    for c in counters:
        c.description = (f"frame counter ({c.length} bits, +1 per frame, first value "
                         f"{_field_value(frames, c.start, c.length)})")
    fields += counters
    header = max(f.start + f.length for f in fields)
    # Re-cut so Frame.header holds the whole header
    frames = split_frames(x, sync, header_bits=header - sync_len, frame_bits=frame_bits)

    result.found = True
    result.method = occ.method
    result.sync_name = occ.name
    result.sync_bits = sync_bits
    result.constant_bits = len(block)
    result.header_bits = header
    result.period = period
    result.regular = regular
    result.sync = sync
    result.frames = frames
    result.fields = fields
    result.false_alarm = occ.false_alarm


def _fmt(bits: np.ndarray) -> str:
    if len(bits) % 8 == 0 and len(bits) >= 8:
        return "0x" + bits_to_hex(bits).replace(" ", "")
    s = bits_to_string(bits)
    return s if len(s) <= 48 else s[:48] + "…"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def analyse_framing(
    bits: np.ndarray,
    library: tuple[KnownSync, ...] = KNOWN_SYNC_WORDS,
    false_alarm: float = 1e-6,
    max_bits: int = 2_000_000,
) -> FramingResult:
    """Find frame sync, frame length and header structure in *bits*."""
    x = np.asarray(bits, dtype=np.uint8).reshape(-1)[:max_bits]
    result = FramingResult()
    if len(x) < 256:
        result.notes.append("Need at least 256 bits for frame analysis.")
        return result
    result.period_candidates = detect_frame_period(x)
    occ = _search_known(x, library, false_alarm) if library else None
    if occ is None:
        occ = _search_blind(x, false_alarm)
    if occ is not None:
        _describe(x, occ, result)
    return result
