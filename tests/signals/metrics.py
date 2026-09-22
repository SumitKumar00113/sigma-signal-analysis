"""Bit-error-rate helpers that tolerate demodulator ambiguities."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np


def _transforms(k: int) -> Iterator[tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    """Bit-group transforms produced by constellation phase rotations."""
    yield "identity", lambda g: g
    yield "invert", lambda g: g ^ 1
    if k == 2:
        yield "swap", lambda g: g[:, ::-1]
        yield "swap+inv", lambda g: g[:, ::-1] ^ 1
        yield "inv0", lambda g: g ^ np.array([1, 0], dtype=np.uint8)
        yield "inv1", lambda g: g ^ np.array([0, 1], dtype=np.uint8)
        yield "swap+inv0", lambda g: g[:, ::-1] ^ np.array([1, 0], dtype=np.uint8)
        yield "swap+inv1", lambda g: g[:, ::-1] ^ np.array([0, 1], dtype=np.uint8)
    if k == 4:
        swap = lambda g: np.hstack([g[:, 2:], g[:, :2]])  # noqa: E731
        yield "iq swap", swap
        yield "inv I", lambda g: g ^ np.array([1, 1, 0, 0], dtype=np.uint8)
        yield "inv Q", lambda g: g ^ np.array([0, 0, 1, 1], dtype=np.uint8)
        yield "swap, inv I", lambda g: swap(g) ^ np.array([1, 1, 0, 0], dtype=np.uint8)
        yield "swap, inv Q", lambda g: swap(g) ^ np.array([0, 0, 1, 1], dtype=np.uint8)


def ber_with_ambiguity(
    rx_bits: np.ndarray,
    tx_bits: np.ndarray,
    bits_per_symbol: int,
    max_shift_symbols: int = 400,
    min_overlap: int = 100,
) -> tuple[float, int, str]:
    """Best BER over symbol alignment shifts and constellation rotations.

    Returns ``(ber, shift_symbols, transform_name)``.
    """
    rx = np.asarray(rx_bits, dtype=np.uint8)
    tx = np.asarray(tx_bits, dtype=np.uint8)
    k = bits_per_symbol
    rx_g = rx[: (len(rx) // k) * k].reshape(-1, k)
    tx_g = tx[: (len(tx) // k) * k].reshape(-1, k)
    best = (1.0, 0, "none")
    for name, fn in _transforms(k):
        cand = fn(rx_g)
        for shift in range(-max_shift_symbols, max_shift_symbols + 1):
            a, b = (cand, tx_g[shift:]) if shift >= 0 else (cand[-shift:], tx_g)
            n = min(len(a), len(b)) - 10
            if n < min_overlap:
                continue
            errs = float(np.mean(a[:n] != b[:n]))
            if errs < best[0]:
                best = (errs, shift, name)
    return best
