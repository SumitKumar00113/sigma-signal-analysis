"""NOAA APT (Automatic Picture Transmission) image decoding.

The NOAA POES satellites (NOAA-15/18/19, 137 MHz) frequency-modulate a
2400 Hz subcarrier that is amplitude-modulated by the image: 2 lines per
second, 4160 words per second, 2080 words per line.  Each line carries two
channels::

    sync A (39) | space A (47) | image A (909) | telemetry A (45)
    sync B (39) | space B (47) | image B (909) | telemetry B (45)

Sync A is seven cycles of a 1040 Hz square wave, sync B seven pulses at
832 Hz.

:func:`decode_apt` takes complex baseband samples, FM-demodulates them,
recovers the subcarrier envelope at 5 samples per word, finds every line
start by correlating with sync A (each line is searched near where the
previous one predicts, so a slightly wrong receiver clock is followed),
and returns the image as 8-bit grey levels.  :func:`write_png` saves it
without any imaging library.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal

WORDS_PER_LINE = 2080
WORD_RATE = 4160.0
SUBCARRIER_HZ = 2400.0
OVERSAMPLE = 5                                  # envelope samples per word
ENV_RATE = WORD_RATE * OVERSAMPLE               # 20 800 Hz
CHANNEL_WORDS = 1040
IMAGE_A = slice(86, 86 + 909)
IMAGE_B = slice(CHANNEL_WORDS + 86, CHANNEL_WORDS + 86 + 909)

# Sync A in words: 4 low, 7 × (2 high, 2 low), 7 low
_SYNC_A = np.array([-1] * 4 + [1, 1, -1, -1] * 7 + [-1] * 7, dtype=np.float64)
MIN_SYNC_SCORE = 0.5            # normalised correlation of a found line start
MIN_GOOD_LINES = 0.6            # share of lines whose sync was found


@dataclass
class APTImage:
    image: np.ndarray            # (lines, 2080) uint8, both channels side by side
    lines: int
    sync_quality: float          # share of lines with a clear sync A
    line_rate_hz: float          # measured (nominal 2.0)
    fm_deviation_hz: float

    @property
    def found(self) -> bool:
        return self.lines >= 10 and self.sync_quality >= MIN_GOOD_LINES

    @property
    def channel_a(self) -> np.ndarray:
        return self.image[:, IMAGE_A]

    @property
    def channel_b(self) -> np.ndarray:
        return self.image[:, IMAGE_B]

    def describe(self) -> str:
        return (f"NOAA APT: {self.lines} lines ({self.lines / 2:.0f} s), sync A found on "
                f"{self.sync_quality:.0%} of lines, line rate {self.line_rate_hz:.4f} Hz, "
                f"FM deviation ≈ {self.fm_deviation_hz / 1e3:.1f} kHz")


def fm_audio(samples: np.ndarray, sample_rate: float, bandwidth_hz: float = 40_000.0,
             audio_rate: float = ENV_RATE) -> tuple[np.ndarray, float]:
    """FM discriminator output resampled to *audio_rate*; also returns the
    peak deviation (99th percentile of |f|)."""
    x = np.asarray(samples, dtype=np.complex128)
    if bandwidth_hz < 0.95 * sample_rate:
        taps = sp_signal.firwin(129, bandwidth_hz / 2, fs=sample_rate)
        x = sp_signal.lfilter(taps, 1.0, x)
    f = np.angle(x[1:] * np.conj(x[:-1])) * sample_rate / (2 * np.pi)
    f -= np.median(f)                                   # residual carrier / Doppler
    dev = float(np.percentile(np.abs(f), 99))
    ratio = Fraction(audio_rate / sample_rate).limit_denominator(2000)
    audio = sp_signal.resample_poly(f, ratio.numerator, ratio.denominator)
    return audio, dev


def subcarrier_envelope(audio: np.ndarray, rate: float = ENV_RATE) -> np.ndarray:
    """Amplitude of the 2400 Hz subcarrier (image brightness)."""
    t = np.arange(len(audio)) / rate
    base = audio * np.exp(-2j * np.pi * SUBCARRIER_HZ * t)
    taps = sp_signal.firwin(101, 2080.0, fs=rate)          # word-rate video bandwidth
    return np.abs(sp_signal.filtfilt(taps, 1.0, base))


def _sync_correlation(words: np.ndarray) -> np.ndarray:
    """Normalised correlation of *words* with sync A at every offset."""
    w = words - sp_signal.convolve(words, np.ones(39) / 39, mode="same")
    num = sp_signal.correlate(w, _SYNC_A, mode="valid")
    energy = np.sqrt(sp_signal.convolve(w ** 2, np.ones(39), mode="valid") * 39)
    return num / np.maximum(energy, 1e-12)


def decode_apt(samples: np.ndarray, sample_rate: float,
               max_seconds: float | None = None) -> APTImage:
    """Decode an APT image from complex baseband samples (see module docstring)."""
    x = samples if max_seconds is None else samples[: int(max_seconds * sample_rate)]
    audio, dev = fm_audio(x, sample_rate)
    env = subcarrier_envelope(audio)
    spl = WORDS_PER_LINE * OVERSAMPLE                     # envelope samples per line
    if len(env) < 3 * spl:
        return APTImage(np.zeros((0, WORDS_PER_LINE), np.uint8), 0, 0.0, 0.0, dev)
    # Sync search at word resolution on each of the 5 sub-sample phases
    corr = np.full(len(env), -1.0)
    for ph in range(OVERSAMPLE):
        c = _sync_correlation(env[ph::OVERSAMPLE])
        corr[ph: ph + OVERSAMPLE * len(c): OVERSAMPLE] = c
    # First line: best sync in the first 2 lines; then track line by line
    start = int(np.argmax(corr[: 2 * spl]))
    starts, good = [], 0
    pos = float(start)
    window = 20 * OVERSAMPLE
    period = float(spl)
    while pos + spl <= len(env):
        lo, hi = int(pos) - window, int(pos) + window + 1
        if lo >= 0 and hi <= len(corr):
            k = lo + int(np.argmax(corr[lo:hi]))
            if corr[k] >= MIN_SYNC_SCORE:
                good += 1
                if starts:
                    # follow the receiver clock: blend the measured spacing
                    period = 0.9 * period + 0.1 * (k - starts[-1])
                pos = float(k)
        starts.append(int(round(pos)))
        pos += period
    starts = [s for s in starts if s + spl <= len(env)]
    lines = np.stack([env[s: s + spl: OVERSAMPLE][:WORDS_PER_LINE] for s in starts])
    img = _to_grey(lines)
    rate = ENV_RATE / (np.median(np.diff(starts)) if len(starts) > 1 else spl)
    return APTImage(img, len(starts), good / max(1, len(starts)), float(rate), dev)


def _to_grey(lines: np.ndarray) -> np.ndarray:
    """Contrast stretch between the 1st and 99th percentile of the image areas."""
    area = np.concatenate([lines[:, IMAGE_A].ravel(), lines[:, IMAGE_B].ravel()])
    lo, hi = np.percentile(area, [1, 99])
    g = np.clip((lines - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    return (g * 255 + 0.5).astype(np.uint8)


def looks_like_apt(tone_hz: float, fm_deviation_hz: float | None = None) -> bool:
    """FM whose discriminator output is a ≈ 2400 Hz tone (the classifier's
    audio-tone evidence) – worth an APT decoding attempt."""
    return abs(tone_hz - SUBCARRIER_HZ) <= 60.0 and (
        fm_deviation_hz is None or fm_deviation_hz >= 3000.0)


def write_png(path: str | Path, image: np.ndarray) -> Path:
    """Save an 8-bit greyscale image as PNG (no imaging library needed)."""
    img = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    h, w = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    p = Path(path)
    p.write_bytes(png)
    return p
