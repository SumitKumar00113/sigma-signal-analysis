"""Asynchronous Baudot (ITA2) teleprinter decoding – RTTY.

RTTY sends 5-bit ITA2 characters asynchronously: the line rests on *mark*,
each character is a start element (space), five data elements (least
significant first; mark = 1) and a stop element of 1, 1.5 or 2 elements
(mark).  Two shift characters switch between the letters and figures
tables.

:func:`decode_baudot` works on demodulated hard bits sampled at an integer
number of samples per element – 1 for synchronous-looking signals, 2 for
the usual case where the demodulator ran on the half-element grid that the
1.5-element stop pulse creates (see
:func:`src.dsp.pipeline.element_rate_note`).  :func:`detect_baudot` tries
both polarities, both grids and every stop length, and accepts the
decoding whose stop elements are valid far more often than chance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LTRS, FIGS = 0x1F, 0x1B

# Index = code with the first data element as bit 0
_LETTERS = ("\0", "E", "\n", "A", " ", "S", "I", "U", "\r", "D", "R", "J", "N", "F", "C", "K",
            "T", "Z", "L", "W", "H", "Y", "P", "Q", "O", "B", "G", "", "M", "X", "V", "")
# ITA2 figures (positions without an international assignment use the
# common US-TTY characters)
_FIGURES = ("\0", "3", "\n", "-", " ", "'", "8", "7", "\r", "\x05", "4", "\x07", ",", "!", ":",
            "(", "5", "+", ")", "2", "#", "6", "0", "1", "9", "?", "&", "", ".", "/", "=", "")

MIN_CHARS = 20
MIN_VALID = 0.9        # stop elements valid in ≥ 90 % of characters (random data: ≈ 50 %)


@dataclass
class BaudotResult:
    text: str = ""
    characters: int = 0
    valid_fraction: float = 0.0          # characters whose stop element was mark
    samples_per_element: int = 1
    stop_elements: float = 1.5
    inverted: bool = False               # mark was demodulated as 0
    codes: list[int] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.characters >= MIN_CHARS and self.valid_fraction >= MIN_VALID

    def describe(self) -> str:
        pol = ", mark = 0" if self.inverted else ""
        return (f"ITA2 Baudot, asynchronous, 1 start + 5 data + {self.stop_elements:g} stop "
                f"elements{pol}: {self.characters} characters, {self.valid_fraction:.0%} with a "
                "valid stop element")


def codes_to_text(codes: list[int], unshift_on_space: bool = False) -> str:
    """ITA2 codes → text (letters/figures shifts applied; CR dropped)."""
    out: list[str] = []
    figs = False
    for c in codes:
        if c == LTRS:
            figs = False
            continue
        if c == FIGS:
            figs = True
            continue
        ch = (_FIGURES if figs else _LETTERS)[c]
        if ch == " " and unshift_on_space:
            figs = False
        if ch in ("\0", "\r"):
            continue
        out.append(ch)
    return "".join(out)


def decode_baudot(bits: np.ndarray, samples_per_element: int = 2, stop_elements: float = 1.5,
                  inverted: bool = False) -> BaudotResult:
    """Decode asynchronous ITA2 from hard bits (see module docstring)."""
    x = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if inverted:
        x = x ^ 1
    k = samples_per_element
    stop_len = max(1, int(round(stop_elements * k)))
    frame = 6 * k + stop_len                   # start + 5 data + stop
    codes: list[int] = []
    valid = 0
    i, n = 1, len(x)
    while i + frame <= n:
        # A start element begins at a mark → space transition
        if not (x[i - 1] == 1 and x[i] == 0):
            i += 1
            continue
        if np.any(x[i: i + k] != 0):           # start element must stay space
            i += 1
            continue
        code = 0
        for b in range(5):
            elem = x[i + k * (b + 1): i + k * (b + 2)]
            code |= int(elem.mean() >= 0.5) << b
        stop = x[i + 6 * k: i + frame]
        ok = bool(np.all(stop == 1))
        codes.append(code)
        valid += ok
        # Next character: after the stop element (or resynchronise one
        # element later on a framing error)
        i += frame if ok else k
    res = BaudotResult(codes_to_text(codes), len(codes), valid / max(1, len(codes)),
                       k, stop_elements, inverted, codes)
    return res


def detect_baudot(bits: np.ndarray, grids: tuple[int, ...] = (2, 1),
                  stops: tuple[float, ...] = (1.5, 1.0, 2.0)) -> BaudotResult:
    """Best asynchronous ITA2 decoding over polarity, grid and stop length.

    Ranked by characters × validity⁴ × stop length: validity dominates,
    and between equally valid framings the longer (stricter) stop wins – a
    1-element stop also "fits" a signal sent with 1.5."""
    best = BaudotResult()
    best_score = -1.0
    for k in grids:
        for stop in stops:
            if k == 1 and stop % 1:
                continue                        # 1.5 elements needs the half grid
            for inv in (False, True):
                r = decode_baudot(bits, k, stop, inv)
                score = r.characters * r.valid_fraction ** 4 * (stop * k)
                if score > best_score:
                    best, best_score = r, score
    return best
