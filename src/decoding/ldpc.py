"""LDPC codes at real-world sizes: construction, encoding, decoding.

* **Sparse representation** – *H* is kept as its list of edges (row,
  column), so codes up to DVB-S2's 64 800 bits fit comfortably.
* **Construction** – from a dense 0/1 matrix, an ``.alist`` file (MacKay's
  exchange format, used for most published standard codes), or a
  quasi-cyclic exponent table (``.qc``: 802.11n/ac, 802.16e, 5G NR,
  CCSDS AR4JA …), including punctured column blocks.  See
  :mod:`src.decoding.ldpc_library` for downloading standard codes.
* **Encoding** – systematic.  If the parity part of *H* is lower
  triangular (IRA / dual-diagonal codes such as DVB-S2) parity follows by
  forward substitution in O(edges); otherwise a packed-bit GF(2)
  elimination builds the generator once (codes up to ~16 k bits).
  Rank-deficient matrices (e.g. 10GBASE-T) get their true dimension *k*.
* **Decoding** – normalised min-sum belief propagation, fully vectorised
  over edges and over a batch of codewords, with early termination.
  Punctured bits enter as erasures (LLR 0); input may be LLRs (positive ⇒
  0) or hard bits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------

#: Largest code for which the generic (dense elimination) encoder is built
MAX_DENSE_ENCODER_BITS = 16384


@dataclass
class LDPCCode:
    """A binary LDPC code given by the sparse parity-check matrix *H*."""

    n: int                              # codeword length (incl. punctured bits)
    m: int                              # parity-check rows
    rows: np.ndarray                    # edge row indices (sorted)
    cols: np.ndarray                    # edge column indices
    name: str = ""
    transmitted: np.ndarray | None = None   # bool mask over the n bits (None = all)
    _encoder: _Encoder | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        order = np.lexsort((self.cols, self.rows))
        self.rows = np.asarray(self.rows, dtype=np.int64)[order]
        self.cols = np.asarray(self.cols, dtype=np.int64)[order]
        counts = np.bincount(self.rows, minlength=self.m)
        if np.any(counts == 0):
            keep = counts > 0          # drop empty checks, they carry no information
            remap = np.cumsum(keep) - 1
            self.rows = remap[self.rows]
            self.m = int(keep.sum())
            counts = counts[keep]
        self._row_starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        self._row_degree = counts.astype(np.int64)
        self._col_degree = np.bincount(self.cols, minlength=self.n)
        if self.transmitted is not None:
            self.transmitted = np.asarray(self.transmitted, dtype=bool)
            if len(self.transmitted) != self.n:
                raise ValueError("transmitted mask must have n entries")

    # ---- basic properties -------------------------------------------------

    @property
    def num_edges(self) -> int:
        return int(len(self.rows))

    @property
    def n_transmitted(self) -> int:
        return self.n if self.transmitted is None else int(self.transmitted.sum())

    @property
    def encoder(self) -> _Encoder:
        if self._encoder is None:
            self._encoder = _build_encoder(self)
        return self._encoder

    @property
    def k(self) -> int:
        return self.encoder.k

    @property
    def info_cols(self) -> np.ndarray:
        return self.encoder.info_cols

    @property
    def _G_info_cols(self) -> np.ndarray:  # noqa: N802 – kept for backwards compatibility
        return self.info_cols

    @property
    def rate(self) -> float:
        return self.k / max(1, self.n_transmitted)

    @property
    def H(self) -> np.ndarray:  # noqa: N802 – coding-theory notation
        """Dense *H* (only sensible for small codes)."""
        h = np.zeros((self.m, self.n), dtype=np.uint8)
        h[self.rows, self.cols] = 1
        return h

    def describe(self) -> str:
        punct = f", {self.n - self.n_transmitted} punctured" if self.transmitted is not None \
            else ""
        return (f"{self.name or 'LDPC'} ({self.n_transmitted},{self.k}), rate {self.rate:.3f}, "
                f"{self.m} checks, {self.num_edges:,} edges{punct}")

    # ---- syndrome ---------------------------------------------------------

    def syndrome(self, bits: np.ndarray) -> np.ndarray:
        """Unsatisfied-check indicator for codeword(s) of *n* bits."""
        b = np.atleast_2d(np.asarray(bits, dtype=np.int64))
        s = np.add.reduceat(b[:, self.cols], self._row_starts, axis=1) & 1
        return s[0] if np.ndim(bits) == 1 else s

    def is_codeword(self, bits: np.ndarray) -> bool:
        return not np.any(self.syndrome(bits))


def ldpc_from_H(H: np.ndarray, name: str = "",  # noqa: N803
                transmitted: np.ndarray | None = None) -> LDPCCode:
    """Build a code from a dense parity-check matrix."""
    H = np.asarray(H, dtype=np.uint8)  # noqa: N806
    rows, cols = np.nonzero(H)
    return LDPCCode(H.shape[1], H.shape[0], rows, cols, name, transmitted)


def ldpc_from_edges(n: int, m: int, rows: np.ndarray, cols: np.ndarray, name: str = "",
                    transmitted: np.ndarray | None = None) -> LDPCCode:
    return LDPCCode(int(n), int(m), np.asarray(rows), np.asarray(cols), name, transmitted)


# ---------------------------------------------------------------------------
# File formats
# ---------------------------------------------------------------------------


def read_alist(path: str | Path, name: str | None = None) -> LDPCCode:
    """MacKay ``.alist``: ``n m`` / max degrees / column degrees / row
    degrees / per-column row lists / per-row column lists (1-based, 0 = pad)."""
    # Some writers add "#" comment lines (e.g. a description of the code)
    text = "\n".join(ln.split("#", 1)[0] for ln in Path(path).read_text().splitlines())
    vals = [int(t) for t in text.split()]
    n, m, max_col, max_row = vals[0], vals[1], vals[2], vals[3]
    pos = 4
    col_deg = vals[pos: pos + n]
    pos += n + m                                    # row degrees are re-derived
    # MacKay pads every column/row list with zeros to the maximum degree;
    # some writers do not.  The token count tells which.
    edges = sum(col_deg)
    padded = len(vals) >= pos + n * max_col + m * max_row
    if not padded and len(vals) < pos + 2 * edges:
        raise ValueError(f"{path}: truncated alist")
    rows, cols = [], []
    for c in range(n):
        width = max_col if padded else col_deg[c]
        entries = vals[pos: pos + width]
        pos += width
        for r in entries[: col_deg[c]]:
            if r > 0:
                rows.append(r - 1)
                cols.append(c)
    return LDPCCode(n, m, np.array(rows), np.array(cols), name or Path(path).stem)


def write_alist(code: LDPCCode, path: str | Path) -> None:
    col_lists: list[list[int]] = [[] for _ in range(code.n)]
    row_lists: list[list[int]] = [[] for _ in range(code.m)]
    for r, c in zip(code.rows.tolist(), code.cols.tolist(), strict=True):
        col_lists[c].append(r + 1)
        row_lists[r].append(c + 1)
    mc = max(len(v) for v in col_lists)
    mr = max(len(v) for v in row_lists)
    lines = [f"{code.n} {code.m}", f"{mc} {mr}",
             " ".join(str(len(v)) for v in col_lists),
             " ".join(str(len(v)) for v in row_lists)]
    lines += [" ".join(str(x) for x in v + [0] * (mc - len(v))) for v in col_lists]
    lines += [" ".join(str(x) for x in v + [0] * (mr - len(v))) for v in row_lists]
    Path(path).write_text("\n".join(lines) + "\n")


def ldpc_from_qc(exponents: np.ndarray, z: int, name: str = "",
                 transmitted_blocks: np.ndarray | None = None) -> LDPCCode:
    """Quasi-cyclic code: entry *e* ≥ 0 of the base matrix is the Z×Z
    identity cyclically shifted right by *e*; −1 is the zero block."""
    base = np.asarray(exponents, dtype=np.int64)
    mb, nb = base.shape
    rows, cols = [], []
    ar = np.arange(z)
    for i in range(mb):
        for j in range(nb):
            e = base[i, j]
            if e < 0:
                continue
            rows.append(i * z + ar)
            cols.append(j * z + (ar + e) % z)
    mask = None
    if transmitted_blocks is not None:
        mask = np.repeat(np.asarray(transmitted_blocks, dtype=bool), z)
    return LDPCCode(nb * z, mb * z, np.concatenate(rows), np.concatenate(cols), name, mask)


def read_qc(path: str | Path, name: str | None = None) -> LDPCCode:
    """AFF3CT-style ``.qc``: ``cols rows Z``, the exponent rows, and an
    optional line of per-column-block flags (1 = transmitted, 0 = punctured)."""
    lines = [ln.split("#", 1)[0].split() for ln in Path(path).read_text().splitlines()]
    lines = [ln for ln in lines if ln]
    nb, mb, z = (int(v) for v in lines[0][:3])
    base = np.array([[int(v) for v in ln] for ln in lines[1: 1 + mb]], dtype=np.int64)
    if base.shape != (mb, nb):
        raise ValueError(f"{path}: expected a {mb}×{nb} exponent matrix, got {base.shape}")
    transmitted = None
    if len(lines) > 1 + mb and len(lines[1 + mb]) == nb:
        flags = np.array([int(v) for v in lines[1 + mb]])
        if not np.all(flags == 1):
            transmitted = flags == 1
    return ldpc_from_qc(base, z, name or Path(path).stem, transmitted)


def load_ldpc(path: str | Path) -> LDPCCode:
    p = Path(path)
    if p.suffix.lower() == ".alist":
        return read_alist(p)
    if p.suffix.lower() == ".qc":
        return read_qc(p)
    raise ValueError(f"{p}: unsupported LDPC file type (use .alist or .qc)")


# ---------------------------------------------------------------------------
# Demo construction
# ---------------------------------------------------------------------------


def make_regular_ldpc(n: int, col_weight: int = 3, row_weight: int = 6,
                      seed: int | None = 0) -> np.ndarray:
    """Random (col_weight, row_weight)-regular H of size (n·wc/wr) × n."""
    if (n * col_weight) % row_weight != 0:
        raise ValueError("n·col_weight must be divisible by row_weight")
    m = n * col_weight // row_weight
    rng = np.random.default_rng(seed)
    H = np.zeros((m, n), dtype=np.uint8)  # noqa: N806
    # Gallager construction: stack col_weight permuted band matrices
    base = np.zeros((m // col_weight, n), dtype=np.uint8)
    for r in range(m // col_weight):
        base[r, r * row_weight:(r + 1) * row_weight] = 1
    for b in range(col_weight):
        perm = rng.permutation(n)
        H[b * (m // col_weight):(b + 1) * (m // col_weight)] = base[:, perm]
    return H


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


@dataclass
class _Encoder:
    k: int
    info_cols: np.ndarray
    parity_cols: np.ndarray
    kind: str                          # "triangular" or "dense"
    # triangular: per parity row, the edges into info / earlier parity bits
    tri_row_starts: np.ndarray | None = None
    # dense: parity = P · info over GF(2), P packed along the info axis
    dense_p: np.ndarray | None = None


def _lower_triangular_parity(code: LDPCCode) -> bool:
    """Is H = [A | T] with T (last m columns) lower triangular, unit diagonal?"""
    k0 = code.n - code.m
    par = code.cols >= k0
    j = code.cols[par] - k0
    r = code.rows[par]
    if np.any(j > r):
        return False
    diag = np.zeros(code.m, dtype=bool)
    diag[r[j == r]] = True
    return bool(diag.all())


def _pack_rows(rows: np.ndarray, cols: np.ndarray, m: int, n: int) -> np.ndarray:
    words = (n + 63) // 64
    packed = np.zeros((m, words), dtype=np.uint64)
    np.bitwise_or.at(packed, (rows, cols // 64),
                     np.left_shift(np.uint64(1), (cols % 64).astype(np.uint64)))
    return packed


def _build_encoder(code: LDPCCode) -> _Encoder:
    n, m = code.n, code.m
    k0 = n - m
    if _lower_triangular_parity(code):
        return _Encoder(k0, np.arange(k0), np.arange(k0, n), "triangular")

    if n > MAX_DENSE_ENCODER_BITS:
        raise ValueError(f"{code.name or 'code'}: no structured encoder and n={n} is too large "
                         "for dense elimination (decoding still works)")

    # GF(2) Gauss-Jordan on packed rows.  Pivot on the rightmost columns first
    # so that, for standard systematic codes, the parity bits come out last.
    a = _pack_rows(code.rows, code.cols, m, n)
    pivots: list[int] = []
    r = 0
    for col in range(n - 1, -1, -1):
        if r >= m:
            break
        w, b = divmod(col, 64)
        bit = np.uint64(1) << np.uint64(b)
        hits = np.flatnonzero(a[r:, w] & bit) + r
        if len(hits) == 0:
            continue
        p = hits[0]
        if p != r:
            a[[r, p]] = a[[p, r]]
        others = np.flatnonzero(a[:, w] & bit)
        others = others[others != r]
        a[others] ^= a[r]
        pivots.append(col)
        r += 1
    rank = r
    parity_cols = np.array(pivots, dtype=np.int64)
    is_parity = np.zeros(n, dtype=bool)
    is_parity[parity_cols] = True
    info_cols = np.flatnonzero(~is_parity)
    # Row i of the reduced matrix: x[parity_cols[i]] = Σ_j a[i, info_cols[j]] x[info_j]
    bits = np.unpackbits(a[:rank].view(np.uint8), axis=1, bitorder="little")[:, :n]
    p_matrix = bits[:, info_cols]
    return _Encoder(n - rank, info_cols, parity_cols, "dense",
                    dense_p=np.packbits(p_matrix, axis=1, bitorder="little"))


def ldpc_encode(info_bits: np.ndarray, code: LDPCCode) -> np.ndarray:
    """Systematic codeword (all *n* bits, including punctured ones)."""
    enc = code.encoder
    info = np.asarray(info_bits, dtype=np.uint8).reshape(-1)
    if len(info) != enc.k:
        raise ValueError(f"Expected {enc.k} information bits, got {len(info)}")
    cw = np.zeros(code.n, dtype=np.uint8)
    cw[enc.info_cols] = info
    if enc.kind == "triangular":
        k0 = code.n - code.m
        # Parity rows in order: p_i = Σ(info edges) + Σ(earlier parity edges)
        acc = np.zeros(code.m, dtype=np.uint8)
        info_edge = code.cols < k0
        np.bitwise_xor.at(acc, code.rows[info_edge], cw[code.cols[info_edge]])
        par_edge = (code.cols >= k0) & ((code.cols - k0) < code.rows)
        pr, pc = code.rows[par_edge], code.cols[par_edge] - k0
        if len(pr) and np.all(pc == pr - 1):          # dual diagonal: running XOR
            parity = np.bitwise_xor.accumulate(acc)
        else:
            parity = np.zeros(code.m, dtype=np.uint8)
            order = np.argsort(pr, kind="stable")
            pr, pc = pr[order], pc[order]
            bounds = np.searchsorted(pr, np.arange(code.m + 1))
            for i in range(code.m):
                v = acc[i]
                for j in pc[bounds[i]: bounds[i + 1]]:
                    v ^= parity[j]
                parity[i] = v
        cw[k0:] = parity
        return cw
    packed_info = np.packbits(info, bitorder="little")
    prod = np.bitwise_and(enc.dense_p, packed_info[None, :])
    cw[enc.parity_cols] = np.unpackbits(prod, axis=1).sum(axis=1) & 1
    return cw


def ldpc_transmit(codeword: np.ndarray, code: LDPCCode) -> np.ndarray:
    """Drop punctured bits."""
    cw = np.asarray(codeword)
    return cw if code.transmitted is None else cw[..., code.transmitted]


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


@dataclass
class LDPCResult:
    bits: np.ndarray          # full codeword hard decisions (n bits)
    info_bits: np.ndarray     # systematic information bits
    iterations: int
    converged: bool           # all parity checks satisfied
    unsatisfied_checks: int


@dataclass
class LDPCBatchResult:
    bits: np.ndarray          # (B, n)
    info_bits: np.ndarray     # (B, k)
    iterations: np.ndarray    # (B,)
    converged: np.ndarray     # (B,) bool
    unsatisfied_checks: np.ndarray


def _expand_llr(llr: np.ndarray, code: LDPCCode) -> np.ndarray:
    x = np.atleast_2d(np.asarray(llr, dtype=np.float64))
    if x.shape[1] == code.n:
        return x
    if code.transmitted is not None and x.shape[1] == code.n_transmitted:
        full = np.zeros((x.shape[0], code.n))
        full[:, code.transmitted] = x               # punctured bits: erasures
        return full
    raise ValueError(f"Expected {code.n} (or {code.n_transmitted} transmitted) LLRs "
                     f"per codeword, got {x.shape[1]}")


def ldpc_decode_batch(
    llr: np.ndarray,
    code: LDPCCode,
    max_iter: int = 50,
    alpha: float = 0.75,
) -> LDPCBatchResult:
    """Normalised min-sum decoding of a batch ``(B, n)`` of LLR vectors."""
    L = _expand_llr(llr, code)
    B = L.shape[0]
    cols, starts = code.cols, code._row_starts
    row_of_edge = code.rows
    E = code.num_edges
    edge_idx = np.arange(E)
    c2v = np.zeros((B, E))
    total = L.copy()
    iterations = np.zeros(B, dtype=np.int64)
    active = np.ones(B, dtype=bool)
    hard = (total < 0).astype(np.uint8)

    for it in range(1, max_iter + 1):
        idx = np.flatnonzero(active)
        if len(idx) == 0:
            break
        v2c = total[idx][:, cols] - c2v[idx]
        mag = np.abs(v2c)
        neg = v2c < 0
        # Sign: parity of negatives on the row, excluding self
        row_neg = np.add.reduceat(neg.astype(np.int64), starts, axis=1) & 1
        sign = np.where((row_neg[:, row_of_edge] ^ neg) == 1, -1.0, 1.0)
        # Magnitude: smallest on the row excluding self (min2 at the min edge)
        min1 = np.minimum.reduceat(mag, starts, axis=1)
        is_min = mag == min1[:, row_of_edge]
        first = np.minimum.reduceat(np.where(is_min, edge_idx[None, :], E), starts, axis=1)
        at_min = edge_idx[None, :] == first[:, row_of_edge]
        min2 = np.minimum.reduceat(np.where(at_min, np.inf, mag), starts, axis=1)
        excl = np.where(at_min, min2[:, row_of_edge], min1[:, row_of_edge])
        excl[~np.isfinite(excl)] = 0.0                 # degree-1 checks
        new_c2v = alpha * sign * excl
        c2v[idx] = new_c2v
        for bi, b in enumerate(idx):
            total[b] = L[b] + np.bincount(cols, weights=new_c2v[bi], minlength=code.n)
        hard[idx] = (total[idx] < 0).astype(np.uint8)
        iterations[idx] = it
        ok = ~np.any(code.syndrome(hard[idx]), axis=1)
        active[idx[ok]] = False

    synd = code.syndrome(hard)
    unsat = synd.sum(axis=1)
    info = hard[:, code.info_cols] if _has_encoder_info(code) else hard[:, : code.n - code.m]
    return LDPCBatchResult(hard, info, iterations, unsat == 0, unsat)


def _has_encoder_info(code: LDPCCode) -> bool:
    try:
        _ = code.encoder
    except ValueError:
        return False
    return True


def ldpc_decode(
    llr: np.ndarray,
    code: LDPCCode,
    max_iter: int = 50,
    alpha: float = 0.75,
) -> LDPCResult:
    """Decode one codeword.  *llr* positive ⇒ bit 0; hard bits can be passed
    as ``1 − 2·bits`` scaled by any positive constant."""
    x = np.asarray(llr, dtype=np.float64).reshape(-1)
    if len(x) not in (code.n, code.n_transmitted):
        raise ValueError(f"Expected {code.n} LLRs, got {len(x)}")
    r = ldpc_decode_batch(x[None, :], code, max_iter, alpha)
    return LDPCResult(r.bits[0], r.info_bits[0], int(r.iterations[0]), bool(r.converged[0]),
                      int(r.unsatisfied_checks[0]))


@dataclass
class LDPCStreamResult:
    info_bits: np.ndarray
    blocks: int
    converged: int
    corrected_bits: int
    mean_iterations: float


def ldpc_decode_stream(
    received: np.ndarray,
    code: LDPCCode,
    soft: bool = False,
    max_iter: int = 50,
    batch: int = 32,
    llr_scale: float = 4.0,
) -> LDPCStreamResult:
    """Cut a stream into transmitted-length blocks and decode each.

    Hard bits (``soft=False``) are mapped to LLRs ±*llr_scale*.
    """
    x = np.asarray(received, dtype=np.float64).reshape(-1)
    nt = code.n_transmitted
    nblk = len(x) // nt
    if nblk == 0:
        return LDPCStreamResult(np.zeros(0, dtype=np.uint8), 0, 0, 0, 0.0)
    blocks = x[: nblk * nt].reshape(nblk, nt)
    llr = blocks if soft else llr_scale * (1.0 - 2.0 * blocks)
    infos, conv, corr, iters = [], 0, 0, []
    for s in range(0, nblk, batch):
        r = ldpc_decode_batch(llr[s: s + batch], code, max_iter)
        infos.append(r.info_bits)
        conv += int(r.converged.sum())
        hard_in = (llr[s: s + batch] < 0).astype(np.uint8)
        corr += int(np.sum(ldpc_transmit(r.bits, code) != hard_in))
        iters.extend(r.iterations.tolist())
    return LDPCStreamResult(np.concatenate(infos).reshape(-1).astype(np.uint8), nblk, conv, corr,
                            float(np.mean(iters)))
