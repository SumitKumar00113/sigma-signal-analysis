"""Blind identification of bit interleavers.

Interleaving destroys the *local* structure of an FEC code stream, and
the right de-interleaver restores it.  Everything here therefore uses the
convolutional-code parity-check statistic from :mod:`src.decoding.fec_id`
as an oracle ("does this look like a code stream?") and searches the
interleaver parameters that make it true.

How each family is found
------------------------

**Block** (R rows × C columns, written row-wise, read column-wise) and
**diagonal** interleavers: consecutive input bits sit exactly R apart in
the output, so decimating the stream by R yields runs of C consecutive
coded bits.  A stride scan over R (scored for the whole code library at
once) finds R and the code; the run length C and the block alignment
follow from where the parity checks break inside the decimated stream.

**Convolutional** (Forney, B branches, delay M): consecutive input bits
sit M·B + 1 apart, giving runs of B coded bits under decimation, so the
same stride scan finds M·B + 1 and the run length gives B.

**Short runs** (C or B shorter than the code's check span): the check is
mapped through the hypothesised interleaver into a sparse "comb" in the
received stream and evaluated directly at every position; the correct
(R, C) or (B, M) makes the comb syndromes periodically near zero.

**Pseudo-random** (seeded LCG / NumPy permutations): no local structure
survives, so seeds are brute-forced for given block sizes, with all
block alignments scored at once per seed.

All searches need the stream to carry a convolutional code that is in
the library (or supplied), because that is the only thing that reveals
the correct de-interleaving.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.core.enums import InterleaverType
from src.decoding.fec_id import (
    ConvHypothesis,
    ParityCheck,
    common_conv_hypotheses,
    conv_library,
    informative_windows,
    syndrome_stream,
)
from src.decoding.interleaving import InterleaverSpec, deinterleave

ProgressCallback = Callable[[float, str], None]
CancelCheck = Callable[[], bool]

# ---------------------------------------------------------------------------
# Code-structure oracle
# ---------------------------------------------------------------------------


@dataclass
class OracleScore:
    z: float                    # std-devs below the random syndrome rate (0.5)
    hyp: int = -1               # index of the best hypothesis
    phase: int = 0
    rate: float = 0.5
    windows: int = 0


class CodeOracle:
    """Scores a bit stream against many code hypotheses in one matrix product."""

    def __init__(self, hypotheses: Iterable[ConvHypothesis]):
        items = [(h, h.checks()[0]) for h in hypotheses if h.checks()]
        if not items:
            raise ValueError("No usable code hypotheses")
        self.hypotheses = [h for h, _ in items]
        self.checks: list[ParityCheck] = [c for _, c in items]
        self.lmax = max(c.length for c in self.checks)
        self._taps = np.zeros((self.lmax, len(self.checks)), dtype=np.float32)
        for i, c in enumerate(self.checks):
            self._taps[: c.length, i] = c.array()
        self._odd = np.array([c.weight % 2 == 1 for c in self.checks])
        self._groups: dict[int, np.ndarray] = {}
        for i, c in enumerate(self.checks):
            self._groups.setdefault(c.period, []).append(i)  # type: ignore[arg-type]
        self._groups = {q: np.asarray(v) for q, v in self._groups.items()}
        self.max_period = max(self._groups)

    def subset(self, index: int) -> CodeOracle:
        return CodeOracle([self.hypotheses[index]])

    def score(self, bits: np.ndarray) -> OracleScore:
        y = np.asarray(bits).reshape(-1)
        nwin = len(y) - self.lmax + 1
        if nwin < 16 * self.max_period:
            return OracleScore(0.0)
        if len(self.checks) == 1:
            s = syndrome_stream(y, self.checks[0].array())[:nwin].astype(np.float32)[:, None]
        else:
            win = sliding_window_view(y.astype(np.float32), self.lmax)
            s = np.mod(win @ self._taps, 2.0)
        # Constant windows (idle/zero padding) satisfy every check: ignore them
        valid = informative_windows(y, self.lmax)[:nwin].astype(np.float32)
        s = s * valid[:, None]
        best = OracleScore(-np.inf)
        for q, idx in self._groups.items():
            m = nwin // q
            counts = valid[: m * q].reshape(m, q).sum(axis=0)[:, None]        # (phase, 1)
            ones = s[: m * q][:, idx].reshape(m, q, len(idx)).sum(axis=0)     # (phase, hyp)
            rates = ones / np.maximum(counts, 1.0)
            flip = self._odd[idx][None, :] & (rates > 0.5)
            rates = np.where(flip, 1.0 - rates, rates)
            z = (0.5 - rates) * 2.0 * np.sqrt(counts)
            phase, j = np.unravel_index(int(np.argmax(z)), z.shape)
            if z[phase, j] > best.z:
                best = OracleScore(float(z[phase, j]), int(idx[j]), int(phase),
                                   float(rates[phase, j]), int(counts[phase, 0]))
        return best

    def name(self, index: int) -> str:
        return self.hypotheses[index].name if 0 <= index < len(self.hypotheses) else "?"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class InterleaverCandidate:
    spec: InterleaverSpec
    offset: int                 # skip this many bits before de-interleaving
    z_score: float              # code-structure score after de-interleaving
    code_name: str
    syndrome_rate: float
    method: str

    def describe(self) -> str:
        s = self.spec
        if s.kind in (InterleaverType.BLOCK, InterleaverType.DIAGONAL):
            params = f"{s.rows} rows × {s.cols} cols (block {s.rows * s.cols} bits)"
        elif s.kind == InterleaverType.CONVOLUTIONAL:
            params = f"{s.branches} branches, delay {s.delay}"
        elif s.kind == InterleaverType.PSEUDO_RANDOM:
            params = f"block {s.block}, seed {s.seed}, {s.generator}"
        else:
            params = ""
        return (f"{s.kind.value} {params}; skip {self.offset} bits; restores "
                f"{self.code_name} (z={self.z_score:.0f}) [{self.method}]")


@dataclass
class InterleaverIdentification:
    best: InterleaverCandidate | None
    candidates: list[InterleaverCandidate] = field(default_factory=list)
    no_interleaver_z: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def kind(self) -> InterleaverType:
        if self.best is None:
            return InterleaverType.NONE if self.no_interleaver_z > 0 else InterleaverType.UNKNOWN
        return self.best.spec.kind

    def summary(self) -> str:
        lines = []
        if self.best:
            lines.append(self.best.describe())
            for c in self.candidates[1:3]:
                lines.append(f"Alternative: {c.describe()}")
        elif self.kind == InterleaverType.NONE:
            lines.append("No interleaver: the stream already shows code structure "
                         f"(z={self.no_interleaver_z:.0f}).")
        else:
            lines.append("No interleaver identified (no known code structure could be restored).")
        lines += self.notes
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _transpose_stream(x: np.ndarray, stride: int) -> np.ndarray:
    """Concatenate the decimated sequences x[r::stride] for r = 0..stride−1."""
    m = len(x) // stride
    return x[: m * stride].reshape(m, stride).T.reshape(-1)


def _chunk_scan(seqs: list[np.ndarray], check: ParityCheck, min_len: int, max_len: int,
                cancel_check: CancelCheck | None = None) -> list[tuple[int, float, int]]:
    """Find the length ℓ of the runs of consecutive code bits.

    *seqs* are decimated sequences that share (to within one position) the
    same run boundaries, e.g. ``x[r::R]`` for several r.  For each ℓ and
    alignment β they are cut into chunks [β + kℓ, β + (k+1)ℓ); inside each
    chunk the code phase is chosen on one half of the windows and the
    syndrome rate measured on the other half (so the choice does not bias
    the score).  Returns (ℓ, score, β) with the best (lowest) first.
    """
    L, q = check.length, check.period
    streams = []
    for seq in seqs:
        s = syndrome_stream(seq, check.array()).astype(np.float32)
        # constant windows prove nothing: score them as random
        s[~informative_windows(seq, L)[: len(s)]] = 0.5
        streams.append(s)
    n_min = min(len(s) for s in streams)
    out: list[tuple[int, float, int]] = []

    def chunk_score(ell: int, beta: int) -> float:
        rounds = (ell - L + 1) // q
        nch = (n_min - beta) // ell
        if rounds < 2 or nch < 3:
            return np.inf
        total = 0.0
        for s in streams:
            a = s[beta: beta + nch * ell].reshape(nch, ell)[:, : rounds * q]
            a = a.reshape(nch, rounds, q)
            sel = a[:, 0::2, :].mean(axis=1)
            ev = a[:, 1::2, :].mean(axis=1)
            phi = np.argmin(sel, axis=1)
            total += float(ev[np.arange(nch), phi].mean())
        return total / len(streams)

    for ell in range(max(min_len, L + 2 * q), max_len + 1):
        if cancel_check and cancel_check():
            break
        if n_min // ell < 4:
            break
        step = max(1, min(L, ell // 16))
        coarse = [(chunk_score(ell, b), b) for b in range(0, ell, step)]
        sc, b0 = min(coarse)
        fine = [(chunk_score(ell, b % ell), b % ell) for b in range(b0 - step, b0 + step + 1)]
        sc, b0 = min(fine + [(sc, b0)])
        if np.isfinite(sc):
            out.append((ell, sc, b0))
    out.sort(key=lambda t: t[1])

    # Any divisor of the true run length scores as well as the run length
    # itself (its chunks never straddle a break), so extend each top result
    # to the largest multiple that is still as good.
    extended: list[tuple[int, float, int]] = []
    for ell, sc, beta in out[:3]:
        extended.append((ell, sc, beta))
        best = None
        for k in range(2, max_len // ell + 1):
            e = k * ell
            s2, b2 = min((chunk_score(e, (beta + j * ell) % e), (beta + j * ell) % e)
                         for j in range(k))
            if s2 <= sc + 0.03:
                best = (e, s2, b2)
        if best is not None:
            extended.insert(0, best)
    return extended + out[3:]


def _deinterleave_candidate(x: np.ndarray, spec: InterleaverSpec, offset: int) -> np.ndarray:
    return deinterleave(x[offset:], spec)


def _best_offset(x: np.ndarray, spec: InterleaverSpec, offsets: Iterable[int],
                 oracle: CodeOracle, seg_len: int) -> tuple[int, float]:
    best = (0, -np.inf)
    for o in dict.fromkeys(int(v) for v in offsets):
        if o < 0 or len(x) - o < seg_len // 2:
            continue
        y = _deinterleave_candidate(x[: o + seg_len], spec, o)
        z = oracle.score(y).z
        if z > best[1]:
            best = (o, z)
    return best


def _block_period(spec: InterleaverSpec) -> int:
    if spec.kind in (InterleaverType.BLOCK, InterleaverType.DIAGONAL):
        return spec.rows * spec.cols
    if spec.kind == InterleaverType.CONVOLUTIONAL:
        return spec.branches
    return spec.block


def _finalise(x: np.ndarray, spec: InterleaverSpec, offset: int, oracle: CodeOracle,
              method: str) -> InterleaverCandidate:
    sc = oracle.score(_deinterleave_candidate(x, spec, offset))
    return InterleaverCandidate(spec, offset, sc.z, oracle.name(sc.hyp), sc.rate, method)


# ---------------------------------------------------------------------------
# Stage A: stride scan (block / diagonal / convolutional with long runs)
# ---------------------------------------------------------------------------


def _stride_stage(
    x: np.ndarray, x_scan: np.ndarray, oracle: CodeOracle, max_stride: int, max_chunk: int,
    max_delay: int, z_threshold: float,
    progress_cb: ProgressCallback | None, cancel_check: CancelCheck | None,
) -> tuple[list[InterleaverCandidate], list[ConvHypothesis]]:
    """Returns candidates plus the code hypotheses that produced significant
    strides (useful to the comb search even when no candidate was built)."""
    scan: list[tuple[int, OracleScore]] = []
    for stride in range(2, max_stride + 1):
        if cancel_check and cancel_check():
            return [], []
        if progress_cb and stride % 16 == 0:
            progress_cb(0.4 * stride / max_stride, f"Stride scan {stride}/{max_stride}")
        if len(x_scan) // stride < 4 * oracle.lmax:
            break
        sc = oracle.score(_transpose_stream(x_scan, stride))
        if sc.z >= z_threshold:
            scan.append((stride, sc))
    scan.sort(key=lambda t: t[1].z, reverse=True)

    out: list[InterleaverCandidate] = []
    hyps: list[ConvHypothesis] = []
    for stride, sc in scan[:3]:
        single = oracle.subset(sc.hyp)
        if single.hypotheses[0] not in hyps:
            hyps.append(single.hypotheses[0])
        check = single.checks[0]
        seqs = [x[r0::stride] for r0 in range(min(stride, 8))]
        chunks = _chunk_scan(seqs, check, 2, min(max_chunk, len(x) // stride // 4),
                             cancel_check)[:5]
        # Scores within noise of each other: prefer the longer run (see _chunk_scan)
        chunks.sort(key=lambda t: (round(t[1] / 0.03), -t[0]))
        seen: set[int] = set()
        for ell, _, beta in chunks:
            if ell in seen or len(seen) >= 3:
                continue
            seen.add(ell)
            out += _stride_candidates(x, stride, ell, beta, single)
        out += _known_stride_short_runs(x, stride, single, max_delay, z_threshold)
    return out, hyps


def _stride_candidates(x: np.ndarray, stride: int, ell: int, beta: int,
                       single: CodeOracle) -> list[InterleaverCandidate]:
    out = []
    # Block / diagonal: R = stride, C = ell
    n_block = stride * ell
    seg = min(len(x), max(4 * n_block, 8000))
    for kind in (InterleaverType.BLOCK, InterleaverType.DIAGONAL):
        spec = InterleaverSpec(kind, rows=stride, cols=ell)
        if n_block <= 4096:
            offsets: Iterable[int] = range(n_block)
        else:
            # The run boundary at β in x[0::R] is a column-0 element of some
            # row r, i.e. block start ≡ β·R − r (mod N); for the diagonal
            # interleaver the boundary may also be the column wrap.
            offs = []
            for b in (beta - 1, beta, beta + 1):
                g = b * stride
                for r in range(stride):
                    offs.append((g - r) % n_block)
                    if kind == InterleaverType.DIAGONAL:
                        offs.append((g - (ell - r) * stride - r) % n_block)
            offsets = offs
        o, z = _best_offset(x, spec, offsets, single, seg)
        if z > 0:
            out.append(InterleaverCandidate(spec, o, z, "", 0.5, "stride scan"))
    # Convolutional: stride = M·B + 1 with B = ell
    if (stride - 1) % ell == 0 and stride > 1:
        out += _conv_offset_candidates(x, ell, (stride - 1) // ell, single, "stride scan")
    return out


def _conv_offset_candidates(x: np.ndarray, branches: int, delay: int, single: CodeOracle,
                            method: str) -> list[InterleaverCandidate]:
    spec = InterleaverSpec(InterleaverType.CONVOLUTIONAL, branches=branches, delay=delay)
    skip = (branches - 1) * delay * branches
    if skip + 4000 > len(x):
        return []
    o, z = _best_offset(x, spec, range(branches), single, min(len(x), skip + 8000))
    return [InterleaverCandidate(spec, o, z, "", 0.5, method)] if z > 0 else []


def _known_stride_short_runs(x: np.ndarray, stride: int, single: CodeOracle, max_delay: int,
                             z_threshold: float) -> list[InterleaverCandidate]:
    """Runs too short for the chunk scan, but the stride is known: try the
    convolutional factorisations M·B + 1 = stride and short-column block
    interleavers with R = stride (comb statistic)."""
    check = single.checks[0]
    L, q = check.length, check.period
    out: list[InterleaverCandidate] = []
    for branches in range(2, min(stride - 1, L + 2 * q + 2) + 1):
        if (stride - 1) % branches == 0 and (stride - 1) // branches <= max_delay:
            out += _conv_offset_candidates(x, branches, (stride - 1) // branches, single,
                                           "stride scan + factorisation")
    best = (-np.inf, 0, 0)
    for cols in range(2, L + 2 * q + 2):
        for c0 in range(min(cols, q)):
            z, j = _comb_block(x, check, stride, cols, c0)
            if z > best[0]:
                best = (z, cols, j)
    if best[0] >= z_threshold:
        spec = InterleaverSpec(InterleaverType.BLOCK, rows=stride, cols=best[1])
        n_block = stride * best[1]
        offsets = range(n_block) if n_block <= 4096 else \
            [(best[2] + d) % n_block for d in range(-2 * q, 2 * q + 1)]
        o, z = _best_offset(x, spec, offsets, single, min(len(x), max(4 * n_block, 8000)))
        if z > 0:
            out.append(InterleaverCandidate(spec, o, z, "", 0.5, "stride scan + comb"))
    return out


# ---------------------------------------------------------------------------
# Stage B: comb search for short runs (code-specific)
# ---------------------------------------------------------------------------


def _comb_syndrome(x: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    span = int(deltas.max()) + 1
    n = len(x) - span + 1
    if n <= 0:
        return np.zeros(0, dtype=np.uint8)
    s = np.zeros(n, dtype=np.uint8)
    for d in deltas:
        s ^= x[d: d + n]
    return s


def _comb_block(x: np.ndarray, check: ParityCheck, rows: int, cols: int,
                c0: int) -> tuple[float, int]:
    """z-score and residue for a block interleaver rows×cols."""
    taps = np.flatnonzero(check.array())
    k = c0 + taps
    deltas = (k % cols) * rows + k // cols
    last_row = (c0 + check.length - 1) // cols
    r_eff = rows - last_row
    n_block = rows * cols
    if r_eff < 1 or deltas.max() >= n_block * 2:
        return -np.inf, 0
    s = _comb_syndrome(x, deltas)
    nb = len(s) // n_block
    if nb < 4:
        return -np.inf, 0
    res = s[: nb * n_block].reshape(nb, n_block).mean(axis=0)
    csum = np.concatenate([[0.0], np.cumsum(np.concatenate([res, res[: r_eff]]))])
    window = (csum[r_eff: r_eff + n_block] - csum[:n_block]) / r_eff
    j = int(np.argmin(window))
    sigma = 0.5 / math.sqrt(r_eff * nb)
    z = (0.5 - float(window[j])) / sigma - math.sqrt(2 * math.log(n_block))
    return z, j


def _comb_conv(x: np.ndarray, check: ParityCheck, branches: int, delay: int,
               i0: int) -> tuple[float, int]:
    taps = np.flatnonzero(check.array())
    k = i0 + taps
    deltas = (k // branches + (k % branches) * delay) * branches + k % branches
    period = branches * check.period
    if deltas.max() > len(x) // 2:
        return -np.inf, 0
    s = _comb_syndrome(x, deltas)
    nb = len(s) // period
    if nb < 8:
        return -np.inf, 0
    res = s[: nb * period].reshape(nb, period).mean(axis=0)
    j = int(np.argmin(res))
    sigma = 0.5 / math.sqrt(nb)
    z = (0.5 - float(res[j])) / sigma - math.sqrt(2 * math.log(period))
    return z, j


def _comb_stage(
    x: np.ndarray, hypotheses: list[ConvHypothesis], oracle: CodeOracle,
    max_rows: int, max_stride: int, max_branches: int, max_delay: int, z_threshold: float,
    progress_cb: ProgressCallback | None, cancel_check: CancelCheck | None,
) -> list[InterleaverCandidate]:
    out: list[InterleaverCandidate] = []
    for hi, hyp in enumerate(hypotheses):
        checks = hyp.checks()
        if not checks:
            continue
        check = checks[0]
        single = CodeOracle([hyp])
        L, q = check.length, check.period
        if progress_cb:
            progress_cb(0.5 + 0.4 * hi / max(1, len(hypotheses)), f"Comb search: {hyp.name}")
        found: list[InterleaverCandidate] = []

        # Block interleavers whose columns are too short for the stride scan
        best_blk = (-np.inf, 0, 0, 0)
        for cols in range(2, L + 2 * q + 2):
            for rows in range(2, max_rows + 1):
                if cancel_check and cancel_check():
                    return out
                for c0 in range(min(cols, q)):
                    z, j = _comb_block(x, check, rows, cols, c0)
                    if z > best_blk[0]:
                        best_blk = (z, rows, cols, j)
        if best_blk[0] >= z_threshold:
            _, rows, cols, j = best_blk
            spec = InterleaverSpec(InterleaverType.BLOCK, rows=rows, cols=cols)
            n_block = rows * cols
            offsets = range(n_block) if n_block <= 4096 else \
                [(j + d) % n_block for d in range(-2 * q, 2 * q + 1)]
            o, z = _best_offset(x, spec, offsets, single, min(len(x), max(4 * n_block, 8000)))
            if z >= z_threshold:
                found.append(_finalise(x, spec, o, oracle, "comb search"))

        # Convolutional interleavers not already covered by the stride scan
        best_cv = (-np.inf, 0, 0, 0)
        for branches in range(2, max_branches + 1):
            for delay in range(1, max_delay + 1):
                if branches >= L + 2 * q and delay * branches + 1 <= max_stride:
                    continue
                if cancel_check and cancel_check():
                    return out
                for i0 in range(q):
                    z, j = _comb_conv(x, check, branches, delay, i0)
                    if z > best_cv[0]:
                        best_cv = (z, branches, delay, j)
        if best_cv[0] >= z_threshold:
            _, branches, delay, j = best_cv
            spec = InterleaverSpec(InterleaverType.CONVOLUTIONAL, branches=branches, delay=delay)
            skip = (branches - 1) * delay * branches
            o, z = _best_offset(x, spec, range(branches), single,
                                min(len(x), skip + 8000))
            if z >= z_threshold:
                found.append(_finalise(x, spec, o, oracle, "comb search"))

        if found:
            out += found
            break                                # code found: no need to try others
    return out


# ---------------------------------------------------------------------------
# Stage C: pseudo-random seed search
# ---------------------------------------------------------------------------

_LCG_A, _LCG_C, _MASK32 = 1664525, 1013904223, 0xFFFFFFFF


def lcg_permutations(block: int, seeds: np.ndarray) -> np.ndarray:
    """Vectorised equivalent of ``random_permutation(block, seed, "lcg")``
    for many seeds at once – shape (len(seeds), block)."""
    a_pow = np.empty(block, dtype=np.uint64)
    b_off = np.empty(block, dtype=np.uint64)
    a, b = _LCG_A, _LCG_C
    for i in range(block):
        a_pow[i], b_off[i] = a, b
        a, b = (a * _LCG_A) & _MASK32, (b * _LCG_A + _LCG_C) & _MASK32
    s = (np.asarray(seeds, dtype=np.uint64) & np.uint64(_MASK32))[:, None]
    keys = (a_pow[None, :] * s + b_off[None, :]) & np.uint64(_MASK32)
    return np.argsort(keys, axis=1, kind="stable")


def _score_all_offsets(x: np.ndarray, inv: np.ndarray, check: ParityCheck,
                       n_blocks: int) -> tuple[int, float]:
    """Best alignment of a pseudo-random de-interleaver, all offsets at once."""
    n = len(inv)
    base = (np.arange(n_blocks)[:, None] * n + inv[None, :]).reshape(-1)
    idx = np.arange(n)[:, None] + base[None, :]
    y = x[idx].astype(np.float32)                                   # (offset, bits)
    taps = check.array().astype(np.float32)
    s = np.mod(sliding_window_view(y, check.length, axis=1) @ taps, 2.0)
    q = check.period
    m = s.shape[1] // q
    rates = s[:, : m * q].reshape(n, m, q).mean(axis=1)             # (offset, phase)
    if check.weight % 2:
        rates = np.minimum(rates, 1.0 - rates)
    flat = int(np.argmin(rates))
    o = flat // q
    z = (0.5 - float(rates.flat[flat])) * 2.0 * math.sqrt(m) - math.sqrt(2 * math.log(n * q))
    return o, z


def search_pseudo_random(
    bits: np.ndarray,
    block_sizes: Iterable[int],
    hypotheses: list[ConvHypothesis] | None = None,
    seeds: Iterable[int] = range(256),
    generators: tuple[str, ...] = ("lcg", "numpy"),
    z_threshold: float = 8.0,
    progress_cb: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> list[InterleaverCandidate]:
    """Brute-force seeds of the built-in pseudo-random interleavers."""
    x = np.asarray(bits, dtype=np.uint8).reshape(-1)
    hyps = hypotheses or common_conv_hypotheses()[:2]
    seeds_arr = np.asarray(list(seeds), dtype=np.int64)
    blocks = [int(b) for b in block_sizes if 2 <= int(b) <= len(x) // 3]
    out: list[InterleaverCandidate] = []
    total = max(1, len(blocks) * len(generators) * len(seeds_arr) * len(hyps))
    done = 0
    for n in blocks:
        n_blocks = max(2, min(8, len(x) // n - 1, 4_000_000 // (n * n)))
        if n + n_blocks * n > len(x):
            continue
        for gen in generators:
            perms = (lcg_permutations(n, seeds_arr) if gen == "lcg"
                     else np.stack([np.random.default_rng(int(s)).permutation(n)
                                    for s in seeds_arr]))
            for si, perm in enumerate(perms):
                if cancel_check and cancel_check():
                    return out
                inv = np.argsort(perm)
                for hyp in hyps:
                    done += 1
                    if progress_cb and done % 32 == 0:
                        progress_cb(done / total, f"Pseudo-random block {n}, {gen} seed "
                                                  f"{int(seeds_arr[si])}")
                    o, z = _score_all_offsets(x, inv, hyp.checks()[0], n_blocks)
                    if z >= z_threshold:
                        spec = InterleaverSpec(InterleaverType.PSEUDO_RANDOM, block=n,
                                               seed=int(seeds_arr[si]), generator=gen)
                        out.append(_finalise(x, spec, o, CodeOracle([hyp]), "seed search"))
    out.sort(key=lambda c: c.z_score, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


def identify_interleaver(
    bits: np.ndarray,
    hypotheses: list[ConvHypothesis] | None = None,
    comb_hypotheses: list[ConvHypothesis] | None = None,
    max_stride: int = 256,
    max_chunk: int = 512,
    max_rows_small: int = 64,
    max_branches: int = 32,
    max_delay: int = 64,
    pseudo_random_blocks: Iterable[int] = (),
    seeds: Iterable[int] = range(256),
    max_bits: int = 40_000,
    scan_bits: int = 20_000,
    z_threshold: float = 8.0,
    progress_cb: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
    **_: object,
) -> InterleaverIdentification:
    """Identify the interleaver in front of a convolutional code (see module
    docstring).  *hypotheses* is the code library used by the stride scan;
    *comb_hypotheses* the (smaller) set used by the slower comb search."""
    x = np.asarray(bits, dtype=np.uint8).reshape(-1)[:max_bits]
    oracle = CodeOracle(hypotheses if hypotheses is not None else conv_library())
    result = InterleaverIdentification(None)
    if len(x) < 2000:
        result.notes.append("Need at least 2000 bits for interleaver identification.")
        return result

    raw = oracle.score(x)
    if raw.z >= z_threshold:
        result.no_interleaver_z = raw.z
        result.notes.append(f"Code visible without de-interleaving: {oracle.name(raw.hyp)}.")
        return result

    cands, stride_hyps = _stride_stage(x, x[:scan_bits], oracle, max_stride, max_chunk,
                                       max_delay, z_threshold, progress_cb, cancel_check)
    cands = [_finalise(x, c.spec, c.offset, oracle, c.method) for c in cands]
    cands = [c for c in cands if c.z_score >= z_threshold]
    strong = any(c.z_score >= 2 * z_threshold for c in cands)

    if not strong and not (cancel_check and cancel_check()):
        comb = comb_hypotheses
        if comb is None:
            comb = hypotheses if hypotheses is not None and len(hypotheses) <= 8 \
                else common_conv_hypotheses()
        # Codes seen by the stride scan first (e.g. punctured codes)
        comb = stride_hyps + [h for h in comb if h not in stride_hyps]
        cands += _comb_stage(x, comb, oracle, max_rows_small, max_stride, max_branches,
                             max_delay, z_threshold, progress_cb, cancel_check)

    blocks = list(pseudo_random_blocks)
    if not cands and blocks and not (cancel_check and cancel_check()):
        prh = comb_hypotheses or (hypotheses if hypotheses is not None and len(hypotheses) <= 4
                                  else None)
        cands = search_pseudo_random(x, blocks, prh, seeds, z_threshold=z_threshold,
                                     progress_cb=progress_cb, cancel_check=cancel_check)

    # Prefer the strongest restoration; among equals, the smallest structure
    cands.sort(key=lambda c: (-round(c.z_score), _block_period(c.spec)))
    result.candidates = cands
    result.best = cands[0] if cands else None
    if result.best is None:
        result.notes.append("Searched: block/diagonal/convolutional via stride scan, short "
                            "block and convolutional interleavers via comb search"
                            + (", pseudo-random seeds" if blocks else "")
                            + ". The stream must carry a library convolutional code.")
    if progress_cb:
        progress_cb(1.0, "Interleaver search complete")
    return result
