"""GF(2) linear algebra on bit-packed rows.

Rows are Python ``int`` bitmasks (bit *j* = column *j*), which makes row
XOR a single big-integer operation and keeps elimination fast for the
matrix sizes used in blind code identification (up to a few hundred
columns).
"""

from __future__ import annotations

import numpy as np


def bits_to_int(bits: np.ndarray) -> int:
    """Pack a 0/1 vector into an int, element *j* → bit *j*."""
    out = 0
    for j in np.flatnonzero(np.asarray(bits).reshape(-1)):
        out |= 1 << int(j)
    return out


def int_to_bits(value: int, width: int) -> np.ndarray:
    return np.array([(value >> j) & 1 for j in range(width)], dtype=np.uint8)


def rows_to_ints(matrix: np.ndarray) -> list[int]:
    """Pack each row of a 0/1 matrix into an int (column *j* → bit *j*)."""
    m = np.asarray(matrix, dtype=np.uint8)
    if m.ndim != 2:
        raise ValueError("Expected a 2-D matrix")
    if m.shape[1] <= 62:
        weights = np.left_shift(np.int64(1), np.arange(m.shape[1], dtype=np.int64))
        return [int(v) for v in m.astype(np.int64) @ weights]
    packed = np.packbits(m, axis=1, bitorder="little")
    return [int.from_bytes(row.tobytes(), "little") for row in packed]


def _reduce(rows: list[int]) -> dict[int, int]:
    """Reduced row-echelon form as {pivot_column: row}."""
    pivots: dict[int, int] = {}
    for r in rows:
        for c, pr in pivots.items():
            if (r >> c) & 1:
                r ^= pr
        if r:
            c = r.bit_length() - 1
            for c2, pr in pivots.items():
                if (pr >> c) & 1:
                    pivots[c2] = pr ^ r
            pivots[c] = r
    return pivots


def gf2_rank(rows: list[int]) -> int:
    return len(_reduce(rows))


def gf2_nullspace(rows: list[int], ncols: int) -> list[int]:
    """Basis of {v : row·v = 0 for every row}, each vector as an int."""
    pivots = _reduce(rows)
    basis = []
    for f in range(ncols):
        if f in pivots:
            continue
        v = 1 << f
        for c, pr in pivots.items():
            if (pr >> f) & 1:
                v |= 1 << c
        basis.append(v)
    return basis


def popcount(v: int) -> int:
    return v.bit_count()
