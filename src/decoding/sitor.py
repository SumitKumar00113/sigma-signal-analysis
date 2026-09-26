"""SITOR-B / NAVTEX decoding (CCIR 476, FEC mode B).

NAVTEX (518 / 490 / 4209.5 kHz) sends 100 Bd FSK carrying 7-bit CCIR 476
characters.  Every valid character has exactly four *mark* elements out of
seven – a constant-ratio code that detects most errors.  In FEC mode B each
character is sent twice: in a *DX* slot and again five slots later in an
*RX* slot (DX and RX slots alternate), so a character lost in one copy is
usually recovered from the other.

:func:`detect_sitor_b` finds the character alignment and polarity (the
7-bit phase at which almost every character has four marks), the DX slot
parity (the one whose characters reappear five slots later), combines the
two copies, and decodes letters / figures to text.  The alignment is
re-checked in windows, so a bit slip mid-stream only costs a few characters.
Messages are split at ``ZCZC B1B2B3B4`` … ``NNNN`` with the header decoded
(transmitter, subject, serial number).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# CCIR 476 codes, first transmitted element = bit 0, mark = 1
_LETTERS = {
    0x47: "A", 0x72: "B", 0x1D: "C", 0x53: "D", 0x56: "E", 0x1B: "F", 0x35: "G", 0x69: "H",
    0x4D: "I", 0x17: "J", 0x1E: "K", 0x65: "L", 0x39: "M", 0x59: "N", 0x71: "O", 0x2D: "P",
    0x2E: "Q", 0x55: "R", 0x4B: "S", 0x74: "T", 0x4E: "U", 0x3C: "V", 0x27: "W", 0x3A: "X",
    0x2B: "Y", 0x63: "Z",
}
# Figures case (ITA2 pairing of letters and figures)
_FIG_OF = {"A": "-", "B": "?", "C": ":", "D": "$", "E": "3", "F": "!", "G": "&", "H": "#",
           "I": "8", "J": "'", "K": "(", "L": ")", "M": ".", "N": ",", "O": "9", "P": "0",
           "Q": "1", "R": "4", "S": "'", "T": "5", "U": "7", "V": "=", "W": "2", "X": "/",
           "Y": "6", "Z": "+"}
SPACE, LTRS, FIGS, CR, LF = 0x5C, 0x5A, 0x36, 0x78, 0x6C
ALPHA = 0x66                     # phasing 1 / idle; also inserted during messages
# Remaining valid codes: phasing 2 (RQ), idle β, blank – none prints
VALID = frozenset(c for c in range(128) if bin(c).count("1") == 4)

SUBJECTS = {
    "A": "navigational warning", "B": "meteorological warning", "C": "ice report",
    "D": "search and rescue / piracy", "E": "meteorological forecast", "F": "pilot service",
    "G": "AIS", "H": "LORAN", "J": "GNSS", "K": "other electronic navaid",
    "L": "navigational warning (additional)", "T": "test", "V": "special service",
    "W": "special service", "X": "special service", "Y": "special service",
    "Z": "no messages on hand",
}

MIN_VALID = 0.8                  # characters with four marks (random bits: ≈ 27 %)
MIN_CHARS = 50
WINDOW = 70                      # characters per alignment check


@dataclass
class NavtexMessage:
    header: str                  # B1B2B3B4, e.g. "EA35"
    text: str

    @property
    def station(self) -> str:
        return self.header[:1]

    @property
    def subject(self) -> str:
        return SUBJECTS.get(self.header[1:2], "unknown subject")

    @property
    def number(self) -> str:
        return self.header[2:]

    def describe(self) -> str:
        return f"{self.header}: station {self.station}, {self.subject}, no. {self.number}"


@dataclass
class SitorResult:
    text: str = ""
    characters: int = 0
    valid_fraction: float = 0.0          # characters with four marks
    recovered: int = 0                   # DX copies replaced by the RX copy
    lost: int = 0                        # both copies invalid
    inverted: bool = False
    messages: list[NavtexMessage] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.characters >= MIN_CHARS and self.valid_fraction >= MIN_VALID

    def describe(self) -> str:
        pol = ", mark = 0" if self.inverted else ""
        return (f"SITOR-B / NAVTEX (CCIR 476, FEC mode B{pol}): {self.characters} characters, "
                f"{self.valid_fraction:.0%} valid; {self.recovered} restored from the repeat, "
                f"{self.lost} lost; {len(self.messages)} message(s)")


def _codes(bits: np.ndarray, start: int, count: int) -> np.ndarray:
    g = bits[start: start + 7 * count].reshape(-1, 7).astype(np.int64)
    return (g << np.arange(7)).sum(axis=1)


def _validity(codes: np.ndarray) -> float:
    return float(np.mean([c in VALID for c in codes])) if len(codes) else 0.0


def _segments(bits: np.ndarray) -> list[tuple[int, int]]:
    """(first bit, characters) of runs with a stable 7-bit alignment."""
    n_chars = len(bits) // 7
    segs: list[tuple[int, int]] = []
    pos = 0
    while pos + 7 * WINDOW <= len(bits):
        best = max(range(7), key=lambda ph: _validity(_codes(bits, pos + ph, WINDOW)))
        start = pos + best
        count = 0
        while start + 7 * (count + WINDOW) <= len(bits) and \
                _validity(_codes(bits, start + 7 * count, WINDOW)) >= 0.5:
            count += WINDOW
        if count == 0:
            pos += 7 * WINDOW // 2
            continue
        # extend to the last valid character of the run
        while start + 7 * (count + 1) <= len(bits) and \
                _codes(bits, start + 7 * count, 1)[0] in VALID:
            count += 1
        segs.append((start, count))
        pos = start + 7 * count
    return segs or [(0, n_chars)]


def _combine(codes: np.ndarray) -> tuple[list[int], int, int]:
    """DX/RX time diversity: the DX character, else its RX repeat 5 slots on."""
    parity = max((0, 1), key=lambda p: np.mean(codes[p:len(codes) - 5:2] ==
                                               codes[p + 5::2][:len(codes[p:len(codes) - 5:2])])
                  if len(codes) > 7 else 0.0)
    out, recovered, lost = [], 0, 0
    for i in range(parity, len(codes) - 5, 2):
        dx, rx = int(codes[i]), int(codes[i + 5])
        if dx in VALID:
            out.append(dx)
        elif rx in VALID:
            out.append(rx)
            recovered += 1
        else:
            out.append(-1)
            lost += 1
    return out, recovered, lost


def codes_to_text(codes: list[int]) -> str:
    out: list[str] = []
    figs = False
    for c in codes:
        if c == LTRS:
            figs = False
        elif c == FIGS:
            figs = True
        elif c == SPACE:
            out.append(" ")
        elif c == LF:
            out.append("\n")
        elif c in _LETTERS:
            ch = _LETTERS[c]
            out.append(_FIG_OF[ch] if figs else ch)
        elif c == -1:
            out.append("~")                       # both copies lost
        # CR, phasing and idle signals do not print
    return "".join(out)


_MSG = re.compile(r"ZCZC ?([A-Z][A-Z]\s*[0-9~]{2})(.*?)(?:NNNN|$)", re.S)


def split_messages(text: str) -> list[NavtexMessage]:
    return [NavtexMessage(m.group(1).replace(" ", ""), m.group(2).strip("\n "))
            for m in _MSG.finditer(text)]


def decode_sitor_b(bits: np.ndarray, inverted: bool = False) -> SitorResult:
    x = np.asarray(bits, dtype=np.uint8).reshape(-1) ^ (1 if inverted else 0)
    all_codes, recovered, lost, valid, total = [], 0, 0, 0, 0
    for start, count in _segments(x):
        codes = _codes(x, start, count)
        valid += sum(int(c) in VALID for c in codes)
        total += len(codes)
        chars, r, lo = _combine(codes)
        all_codes += chars
        recovered += r
        lost += lo
    text = codes_to_text(all_codes)
    return SitorResult(text, len(all_codes), valid / max(1, total), recovered, lost, inverted,
                       split_messages(text))


def detect_sitor_b(bits: np.ndarray) -> SitorResult:
    """Decode with the polarity whose characters are (almost all) valid."""
    x = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if len(x) < 7 * WINDOW:
        return SitorResult()
    probe = x[: 7 * 400]
    inv = max((False, True), key=lambda i: max(
        _validity(_codes(probe ^ int(i), ph, (len(probe) - ph) // 7)) for ph in range(7)))
    return decode_sitor_b(x, inv)
