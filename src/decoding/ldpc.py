"""LDPC encoding and decoding (experimental).

Provides:

* :func:`make_regular_ldpc` – a random regular Gallager parity-check
  matrix for experimentation.
* :func:`ldpc_encode` – systematic encoding via GF(2) Gaussian
  elimination of *H* (works for any full-rank H, slow for large codes).
* :func:`ldpc_decode` – normalised min-sum belief propagation on LLRs.

Real-world LDPC codes (DVB-S2, CCSDS, 5G) need their specific *H*
matrices; load one as a dense or sparse 0/1 array and pass it in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def make_regular_ldpc(n: int, col_weight: int = 3, row_weight: int = 6,
                      seed: int | None = 0) -> np.ndarray:
    """Random (col_weight, row_weight)-regular H of size (n·wc/wr) × n."""
    if (n * col_weight) % row_weight != 0:
        raise ValueError("n·col_weight must be divisible by row_weight")
    m = n * col_weight // row_weight
    rng = np.random.default_rng(seed)
    H = np.zeros((m, n), dtype=np.uint8)
    # Gallager construction: stack col_weight permuted band matrices
    base = np.zeros((m // col_weight, n), dtype=np.uint8)
    for r in range(m // col_weight):
        base[r, r * row_weight:(r + 1) * row_weight] = 1
    for b in range(col_weight):
        perm = rng.permutation(n)
        H[b * (m // col_weight):(b + 1) * (m // col_weight)] = base[:, perm]
    return H


# ---------------------------------------------------------------------------
# GF(2) linear algebra
# ---------------------------------------------------------------------------


def _gf2_row_reduce(H: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Row-reduce H over GF(2).  Returns (reduced, pivot_columns)."""
    A = H.copy().astype(np.uint8)
    m, n = A.shape
    pivots: list[int] = []
    row = 0
    for col in range(n):
        if row >= m:
            break
        pivot_rows = np.flatnonzero(A[row:, col]) + row
        if len(pivot_rows) == 0:
            continue
        p = pivot_rows[0]
        if p != row:
            A[[row, p]] = A[[p, row]]
        others = np.flatnonzero(A[:, col])
        others = others[others != row]
        A[others] ^= A[row]
        pivots.append(col)
        row += 1
    return A[:row], pivots


@dataclass
class LDPCCode:
    H: np.ndarray
    _G_info_cols: np.ndarray            # columns of H that are information bits
    _G_parity_cols: np.ndarray          # columns that are parity bits
    _P: np.ndarray                      # parity = P @ info (mod 2)

    @property
    def n(self) -> int:
        return int(self.H.shape[1])

    @property
    def k(self) -> int:
        return int(len(self._G_info_cols))

    @property
    def rate(self) -> float:
        return self.k / self.n


def ldpc_from_H(H: np.ndarray) -> LDPCCode:
    """Build a systematic encoder from a parity-check matrix."""
    H = np.asarray(H, dtype=np.uint8)
    R, pivots = _gf2_row_reduce(H)
    n = H.shape[1]
    parity_cols = np.array(pivots, dtype=np.int64)
    info_cols = np.array([c for c in range(n) if c not in set(pivots)], dtype=np.int64)
    # In reduced form, each pivot row reads: x_pivot + Σ R[row, info] x_info = 0
    P = R[:, info_cols]     # parity_j = Σ P[j, i] · info_i
    return LDPCCode(H=H, _G_info_cols=info_cols, _G_parity_cols=parity_cols, _P=P)


def ldpc_encode(info_bits: np.ndarray, code: LDPCCode) -> np.ndarray:
    info = np.asarray(info_bits, dtype=np.uint8).reshape(-1)
    if len(info) != code.k:
        raise ValueError(f"Expected {code.k} information bits, got {len(info)}")
    parity = (code._P.astype(np.int64) @ info.astype(np.int64)) % 2
    cw = np.zeros(code.n, dtype=np.uint8)
    cw[code._G_info_cols] = info
    cw[code._G_parity_cols] = parity.astype(np.uint8)
    return cw


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


@dataclass
class LDPCResult:
    bits: np.ndarray          # full codeword hard decisions
    info_bits: np.ndarray     # systematic information bits
    iterations: int
    converged: bool           # all parity checks satisfied
    unsatisfied_checks: int


def ldpc_decode(
    llr: np.ndarray,
    code: LDPCCode,
    max_iter: int = 50,
    alpha: float = 0.8,
) -> LDPCResult:
    """Normalised min-sum decoding.

    *llr* are log-likelihood ratios with positive ⇒ bit 0.  Hard bits can
    be passed as ``1 - 2·bits`` scaled by any positive constant.
    """
    H = code.H
    m, n = H.shape
    L = np.asarray(llr, dtype=np.float64).reshape(-1)
    if len(L) != n:
        raise ValueError(f"Expected {n} LLRs, got {len(L)}")

    rows, cols = np.nonzero(H)
    msg_c2v = np.zeros(len(rows))            # check → variable messages, per edge

    hard = (L < 0).astype(np.uint8)
    it = 0
    for it in range(1, max_iter + 1):
        # Variable → check
        total = L.copy()
        np.add.at(total, cols, msg_c2v)
        msg_v2c = total[cols] - msg_c2v

        # Check → variable (min-sum with normalisation)
        sign = np.sign(msg_v2c)
        sign[sign == 0] = 1.0
        mag = np.abs(msg_v2c)
        new_c2v = np.zeros_like(msg_c2v)
        for r in range(m):
            idx = np.flatnonzero(rows == r)
            if len(idx) == 0:
                continue
            s = np.prod(sign[idx])
            mg = mag[idx]
            # Exclude-self min: two smallest
            order = np.argsort(mg)
            min1 = mg[order[0]]
            min2 = mg[order[1]] if len(mg) > 1 else min1
            mins = np.full(len(idx), min1)
            mins[order[0]] = min2
            new_c2v[idx] = alpha * s * sign[idx] * mins
        msg_c2v = new_c2v

        total = L.copy()
        np.add.at(total, cols, msg_c2v)
        hard = (total < 0).astype(np.uint8)
        syndrome = (H.astype(np.int64) @ hard.astype(np.int64)) % 2
        if not syndrome.any():
            return LDPCResult(hard, hard[code._G_info_cols], it, True, 0)

    syndrome = (H.astype(np.int64) @ hard.astype(np.int64)) % 2
    return LDPCResult(hard, hard[code._G_info_cols], it, False, int(syndrome.sum()))
