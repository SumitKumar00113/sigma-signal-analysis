"""Reed-Solomon codec over GF(2^8).

Implements a general RS(n, k) code with n ≤ 255 (shortened codes are
handled by zero-padding), with configurable primitive polynomial, first
consecutive root (``fcr``) and generator element (``prim``).  The
defaults (``0x11d``, fcr = 0, prim = 1) are the common "QR-code /
DVB-T" convention; CCSDS uses ``0x187`` with fcr = 112 and prim = 11
and a dual-basis mapping that is *not* applied here.

Decoding uses syndromes → Berlekamp-Massey → Chien search → Forney,
correcting up to ⌊(n−k)/2⌋ symbol errors.  Erasure decoding is not
implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# GF(2^8) arithmetic
# ---------------------------------------------------------------------------


class GF256:
    """Galois field GF(2^8) tables for a given primitive polynomial."""

    def __init__(self, prim_poly: int = 0x11D) -> None:
        self.exp = np.zeros(512, dtype=np.int64)
        self.log = np.zeros(256, dtype=np.int64)
        x = 1
        for i in range(255):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & 0x100:
                x ^= prim_poly
        for i in range(255, 512):
            self.exp[i] = self.exp[i - 255]

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return int(self.exp[self.log[a] + self.log[b]])

    def div(self, a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError
        if a == 0:
            return 0
        return int(self.exp[(self.log[a] - self.log[b]) % 255])

    def pow(self, a: int, n: int) -> int:
        if a == 0:
            return 0 if n > 0 else 1
        return int(self.exp[(self.log[a] * n) % 255])

    def inv(self, a: int) -> int:
        return int(self.exp[(255 - self.log[a]) % 255])

    # --- polynomial helpers (coefficient lists, highest degree first) ---

    def poly_mul(self, p: list[int], q: list[int]) -> list[int]:
        out = [0] * (len(p) + len(q) - 1)
        for i, a in enumerate(p):
            if a == 0:
                continue
            for j, b in enumerate(q):
                out[i + j] ^= self.mul(a, b)
        return out

    def poly_eval(self, p: list[int], x: int) -> int:
        y = 0
        for c in p:
            y = self.mul(y, x) ^ c
        return y

    def poly_scale(self, p: list[int], s: int) -> list[int]:
        return [self.mul(c, s) for c in p]

    @staticmethod
    def poly_add(p: list[int], q: list[int]) -> list[int]:
        n = max(len(p), len(q))
        p2 = [0] * (n - len(p)) + p
        q2 = [0] * (n - len(q)) + q
        return [a ^ b for a, b in zip(p2, q2, strict=True)]


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


@dataclass
class RSResult:
    data: np.ndarray            # decoded k message symbols (bytes)
    corrected: int              # number of symbol errors corrected
    success: bool               # False if more errors than t
    message: str = ""


class ReedSolomon:
    """RS(n, k) codec.  Symbols are bytes (0..255)."""

    def __init__(
        self,
        n: int = 255,
        k: int = 223,
        prim_poly: int = 0x11D,
        fcr: int = 0,
        prim: int = 1,
    ) -> None:
        if not (0 < k < n <= 255):
            raise ValueError("Require 0 < k < n ≤ 255")
        self.n, self.k = n, k
        self.nsym = n - k
        self.fcr, self.prim = fcr, prim
        self.gf = GF256(prim_poly)
        self.gen = self._generator_poly()

    @property
    def t(self) -> int:
        """Maximum correctable symbol errors."""
        return self.nsym // 2

    def _generator_poly(self) -> list[int]:
        g = [1]
        for i in range(self.nsym):
            if self.prim != 1:
                root = self.gf.pow(2, self.prim * (i + self.fcr))
            else:
                root = self.gf.exp[i + self.fcr]
            g = self.gf.poly_mul(g, [1, root])
        return g

    # ---- encode -------------------------------------------------------

    def encode(self, data: np.ndarray | bytes) -> np.ndarray:
        """Systematic encode: returns message followed by parity."""
        msg = list(np.asarray(bytearray(data) if isinstance(data, bytes) else data, dtype=np.int64))
        if len(msg) > self.k:
            raise ValueError(f"Message longer than k={self.k}")
        msg = msg + [0] * (self.k - len(msg))     # pad short messages
        # Polynomial long division of msg·x^nsym by gen
        rem = msg + [0] * self.nsym
        for i in range(len(msg)):
            coef = rem[i]
            if coef != 0:
                for j in range(1, len(self.gen)):
                    rem[i + j] ^= self.gf.mul(self.gen[j], coef)
        parity = rem[-self.nsym:]
        return np.array(msg + parity, dtype=np.uint8)

    # ---- decode -------------------------------------------------------

    def _syndromes(self, cw: list[int]) -> list[int]:
        out = []
        for i in range(self.nsym):
            root = self.gf.exp[(self.prim * (i + self.fcr)) % 255]
            out.append(self.gf.poly_eval(cw, root))
        return out

    def _berlekamp_massey(self, synd: list[int]) -> list[int]:
        """Return the error-locator polynomial (highest degree first)."""
        gf = self.gf
        err_loc = [1]
        old_loc = [1]
        for i in range(self.nsym):
            delta = synd[i]
            for j in range(1, len(err_loc)):
                delta ^= gf.mul(err_loc[-(j + 1)], synd[i - j])
            old_loc = old_loc + [0]
            if delta != 0:
                if len(old_loc) > len(err_loc):
                    new_loc = gf.poly_scale(old_loc, delta)
                    old_loc = gf.poly_scale(err_loc, gf.inv(delta))
                    err_loc = new_loc
                err_loc = gf.poly_add(err_loc, gf.poly_scale(old_loc, delta))
        # strip leading zeros
        while len(err_loc) > 1 and err_loc[0] == 0:
            err_loc = err_loc[1:]
        return err_loc

    def _find_error_positions(self, err_loc: list[int], n: int) -> list[int]:
        """Chien search: positions (0 = first symbol) whose X^-1 is a root."""
        errs = len(err_loc) - 1
        positions = []
        for i in range(n):
            # position i corresponds to power (n-1-i); evaluate at alpha^{-(n-1-i)}
            x_inv = self.gf.exp[(255 - ((n - 1 - i) * self.prim) % 255) % 255]
            if self.gf.poly_eval(err_loc, x_inv) == 0:
                positions.append(i)
        if len(positions) != errs:
            return []
        return positions

    def _correct(self, cw: list[int], synd: list[int], positions: list[int]) -> list[int]:
        """Forney algorithm."""
        gf = self.gf
        n = len(cw)
        # Error locator Λ(x) = Π (1 + X_j·x), coefficient list highest first,
        # where X_j = (α^prim)^(n−1−p_j)
        coef_pos = [(n - 1 - p) * self.prim % 255 for p in positions]
        X = [int(gf.exp[cp]) for cp in coef_pos]
        err_loc = [1]
        for xj in X:
            err_loc = gf.poly_mul(err_loc, [xj, 1])          # xj·x + 1
        n_err = len(X)

        # Error evaluator Ω(x) = [x·S(x)·Λ(x)] mod x^(n_err+1).  The syndrome
        # polynomial gets a leading zero (S_0 term) so its degrees line up
        # with the locator's under the highest-first convention.
        synd_rev = ([0] + list(synd))[::-1]
        prod = gf.poly_mul(synd_rev, err_loc)
        err_eval = prod[len(prod) - (n_err + 1):]

        for i, p in enumerate(positions):
            xi = X[i]
            xi_inv = gf.inv(xi)
            # Formal derivative Λ'(X_i^-1) via the product form
            denom = 1
            for j in range(n_err):
                if j != i:
                    denom = gf.mul(denom, 1 ^ gf.mul(xi_inv, X[j]))
            if denom == 0:
                raise ValueError("Forney: zero denominator")
            y = gf.poly_eval(err_eval, xi_inv)
            y = gf.mul(gf.pow(xi, 1 - self.fcr), y)
            magnitude = gf.div(y, denom)
            cw[p] ^= magnitude
        return cw

    def decode(self, codeword: np.ndarray | bytes) -> RSResult:
        """Decode an n-symbol codeword (or a shortened one of length ≥ k)."""
        cw = list(np.asarray(bytearray(codeword) if isinstance(codeword, bytes) else codeword,
                             dtype=np.int64))
        if len(cw) > self.n or len(cw) <= self.nsym:
            return RSResult(np.zeros(0, dtype=np.uint8), 0, False,
                            f"Codeword length {len(cw)} invalid for RS({self.n},{self.k})")
        # Shortened code: the encoder zero-pads the *message* to k symbols,
        # so re-insert those zeros between the message and the parity.
        pad = self.n - len(cw)
        msg_len = len(cw) - self.nsym
        cw_full = cw[:msg_len] + [0] * pad + cw[msg_len:]

        def _data(word: list[int]) -> np.ndarray:
            return np.array(word[:msg_len], dtype=np.uint8)

        synd = self._syndromes(cw_full)
        if max(synd) == 0:
            return RSResult(_data(cw_full), 0, True, "No errors")

        err_loc = self._berlekamp_massey(synd)
        n_err = len(err_loc) - 1
        if n_err > self.t:
            return RSResult(_data(cw_full), 0, False,
                            f"Too many errors (locator degree {n_err} > t={self.t})")
        positions = self._find_error_positions(err_loc, self.n)
        if not positions:
            return RSResult(_data(cw_full), 0, False,
                            "Could not locate errors (uncorrectable)")
        try:
            corrected = self._correct(cw_full, synd, positions)
        except (ValueError, ZeroDivisionError) as exc:
            return RSResult(_data(cw_full), 0, False, str(exc))
        if max(self._syndromes(corrected)) != 0:
            return RSResult(_data(corrected), 0, False,
                            "Residual syndrome after correction")
        return RSResult(_data(corrected), len(positions), True,
                        f"Corrected {len(positions)} symbol error(s)")


# ---------------------------------------------------------------------------
# Bit-stream helpers
# ---------------------------------------------------------------------------


def bits_to_bytes(bits: np.ndarray) -> np.ndarray:
    """Pack MSB-first bits into bytes (truncating a partial trailing byte)."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    n = (len(bits) // 8) * 8
    return np.packbits(bits[:n])


def bytes_to_bits(data: np.ndarray) -> np.ndarray:
    return np.unpackbits(np.asarray(data, dtype=np.uint8))


def rs_decode_stream(bits: np.ndarray, rs: ReedSolomon) -> tuple[np.ndarray, list[RSResult]]:
    """Split a bit stream into consecutive n-byte codewords and decode each.

    Returns the concatenated decoded message bits and the per-block results.
    """
    data = bits_to_bytes(bits)
    n_blocks = len(data) // rs.n
    out: list[np.ndarray] = []
    results: list[RSResult] = []
    for b in range(n_blocks):
        res = rs.decode(data[b * rs.n:(b + 1) * rs.n])
        results.append(res)
        out.append(res.data)
    if not out:
        return np.zeros(0, dtype=np.uint8), results
    return bytes_to_bits(np.concatenate(out)), results
