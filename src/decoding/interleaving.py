"""Interleavers and de-interleavers.

Four families, each with a forward (``interleave``) and inverse
(``deinterleave``) function so tests can round-trip:

* **Block** – write row-wise into an R×C matrix, read column-wise.
* **Convolutional** (Forney/Ramsey) – *N* branches with delays 0, M, 2M, …
* **Diagonal** – write row-wise, read along wrapped diagonals.
* **Pseudo-random** – a seeded permutation (LCG or NumPy PCG64).

All functions operate on 1-D ``uint8`` bit (or symbol) arrays.  Where
the stream length is not a multiple of the block size, the tail is
padded with zeros on interleave and truncated on de-interleave.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.enums import InterleaverType

# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------


def block_interleave(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    x = np.asarray(bits).reshape(-1)
    block = rows * cols
    pad = (-len(x)) % block
    x = np.concatenate([x, np.zeros(pad, dtype=x.dtype)])
    out = x.reshape(-1, rows, cols).transpose(0, 2, 1).reshape(-1)
    return out


def block_deinterleave(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    x = np.asarray(bits).reshape(-1)
    block = rows * cols
    n_full = len(x) // block
    x = x[: n_full * block]
    return x.reshape(-1, cols, rows).transpose(0, 2, 1).reshape(-1)


# ---------------------------------------------------------------------------
# Convolutional (Forney)
# ---------------------------------------------------------------------------


def convolutional_interleave(bits: np.ndarray, branches: int, delay: int) -> np.ndarray:
    """Branch *i* delays its symbols by ``i·delay``.  Output is the same
    length as the input plus the flush of ``(branches−1)·delay·branches``
    symbols; zeros fill the initial delays."""
    x = np.asarray(bits).reshape(-1)
    n = branches
    pad = (-len(x)) % n
    x = np.concatenate([x, np.zeros(pad, dtype=x.dtype)])
    flush = (n - 1) * delay * n
    x = np.concatenate([x, np.zeros(flush, dtype=x.dtype)])
    frames = x.reshape(-1, n)                    # each row = one symbol per branch
    out = np.zeros_like(frames)
    for i in range(n):
        d = i * delay
        if d == 0:
            out[:, i] = frames[:, i]
        else:
            out[d:, i] = frames[:-d, i]
    return out.reshape(-1)


def convolutional_deinterleave(bits: np.ndarray, branches: int, delay: int) -> np.ndarray:
    """Inverse of :func:`convolutional_interleave`: branch *i* is delayed
    by ``(branches−1−i)·delay`` so all branches line up again.  The
    leading ``(branches−1)·delay`` frames are garbage and are dropped."""
    x = np.asarray(bits).reshape(-1)
    n = branches
    pad = (-len(x)) % n
    x = np.concatenate([x, np.zeros(pad, dtype=x.dtype)])
    frames = x.reshape(-1, n)
    out = np.zeros_like(frames)
    for i in range(n):
        d = (n - 1 - i) * delay
        if d == 0:
            out[:, i] = frames[:, i]
        else:
            out[d:, i] = frames[:-d, i]
    skip = (n - 1) * delay
    return out[skip:].reshape(-1)


# ---------------------------------------------------------------------------
# Diagonal
# ---------------------------------------------------------------------------


def _diag_order(rows: int, cols: int) -> np.ndarray:
    """Read-out order for a rows×cols matrix along wrapped diagonals.

    Diagonal *d* visits (r, (d + r) mod cols) for r = 0..rows−1.
    """
    idx = np.zeros(rows * cols, dtype=np.int64)
    k = 0
    for d in range(cols):
        for r in range(rows):
            idx[k] = r * cols + (d + r) % cols
            k += 1
    return idx


def diagonal_interleave(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    x = np.asarray(bits).reshape(-1)
    block = rows * cols
    pad = (-len(x)) % block
    x = np.concatenate([x, np.zeros(pad, dtype=x.dtype)])
    order = _diag_order(rows, cols)
    return x.reshape(-1, block)[:, order].reshape(-1)


def diagonal_deinterleave(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    x = np.asarray(bits).reshape(-1)
    block = rows * cols
    n_full = len(x) // block
    x = x[: n_full * block]
    order = _diag_order(rows, cols)
    inv = np.argsort(order)
    return x.reshape(-1, block)[:, inv].reshape(-1)


# ---------------------------------------------------------------------------
# Pseudo-random
# ---------------------------------------------------------------------------


def _lcg_permutation(n: int, seed: int) -> np.ndarray:
    """Deterministic permutation from a 32-bit LCG (Numerical Recipes)."""
    keys = np.zeros(n, dtype=np.uint64)
    s = np.uint64(seed & 0xFFFFFFFF)
    a, c, m = np.uint64(1664525), np.uint64(1013904223), np.uint64(0xFFFFFFFF)
    for i in range(n):
        s = (a * s + c) & m
        keys[i] = s
    return np.argsort(keys, kind="stable")


def random_permutation(block: int, seed: int, generator: str = "lcg") -> np.ndarray:
    if generator == "lcg":
        return _lcg_permutation(block, seed)
    rng = np.random.default_rng(seed)
    return rng.permutation(block)


def pseudo_random_interleave(bits: np.ndarray, block: int, seed: int,
                             generator: str = "lcg") -> np.ndarray:
    x = np.asarray(bits).reshape(-1)
    pad = (-len(x)) % block
    x = np.concatenate([x, np.zeros(pad, dtype=x.dtype)])
    perm = random_permutation(block, seed, generator)
    return x.reshape(-1, block)[:, perm].reshape(-1)


def pseudo_random_deinterleave(bits: np.ndarray, block: int, seed: int,
                               generator: str = "lcg") -> np.ndarray:
    x = np.asarray(bits).reshape(-1)
    n_full = len(x) // block
    x = x[: n_full * block]
    perm = random_permutation(block, seed, generator)
    inv = np.argsort(perm)
    return x.reshape(-1, block)[:, inv].reshape(-1)


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------


@dataclass
class InterleaverSpec:
    """Parameters for any supported interleaver."""

    kind: InterleaverType = InterleaverType.NONE
    rows: int = 8               # block / diagonal
    cols: int = 8               # block / diagonal
    branches: int = 4           # convolutional
    delay: int = 1              # convolutional
    block: int = 64             # pseudo-random
    seed: int = 1               # pseudo-random
    generator: str = "lcg"      # pseudo-random


def deinterleave(bits: np.ndarray, spec: InterleaverSpec) -> np.ndarray:
    k = spec.kind
    if k == InterleaverType.NONE or k == InterleaverType.UNKNOWN:
        return np.asarray(bits).reshape(-1)
    if k == InterleaverType.BLOCK:
        return block_deinterleave(bits, spec.rows, spec.cols)
    if k == InterleaverType.CONVOLUTIONAL:
        return convolutional_deinterleave(bits, spec.branches, spec.delay)
    if k == InterleaverType.DIAGONAL:
        return diagonal_deinterleave(bits, spec.rows, spec.cols)
    if k == InterleaverType.PSEUDO_RANDOM:
        return pseudo_random_deinterleave(bits, spec.block, spec.seed, spec.generator)
    raise ValueError(f"Unsupported interleaver {k}")


def interleave(bits: np.ndarray, spec: InterleaverSpec) -> np.ndarray:
    k = spec.kind
    if k == InterleaverType.NONE or k == InterleaverType.UNKNOWN:
        return np.asarray(bits).reshape(-1)
    if k == InterleaverType.BLOCK:
        return block_interleave(bits, spec.rows, spec.cols)
    if k == InterleaverType.CONVOLUTIONAL:
        return convolutional_interleave(bits, spec.branches, spec.delay)
    if k == InterleaverType.DIAGONAL:
        return diagonal_interleave(bits, spec.rows, spec.cols)
    if k == InterleaverType.PSEUDO_RANDOM:
        return pseudo_random_interleave(bits, spec.block, spec.seed, spec.generator)
    raise ValueError(f"Unsupported interleaver {k}")
