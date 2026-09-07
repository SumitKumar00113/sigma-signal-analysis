"""Generate synthetic test fixture files (WAV, raw IQ, SigMF)."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

FIXTURES_DIR = Path(__file__).parent


def generate_tone_wav(
    path: Path | None = None,
    sample_rate: int = 48000,
    duration: float = 0.5,
    frequency: float = 1000.0,
    channels: int = 1,
    bits: int = 16,
) -> Path:
    """Generate a WAV file containing a simple sine tone."""
    if path is None:
        path = FIXTURES_DIR / "tone.wav"
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float64) / sample_rate
    signal = (0.8 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    if bits == 16:
        data = (signal * 32767).astype(np.int16)
    elif bits == 32:
        data = (signal * 2147483647).astype(np.int32)
    else:
        data = (signal * 127).astype(np.int8)

    if channels == 2:
        stereo = np.column_stack([data, data])
        data = stereo.flatten()

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(bits // 8)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())

    return path


def generate_stereo_iq_wav(
    path: Path | None = None,
    sample_rate: int = 48000,
    duration: float = 0.5,
    frequency: float = 5000.0,
) -> Path:
    """Generate a stereo WAV with I on left, Q on right (complex sinusoid)."""
    if path is None:
        path = FIXTURES_DIR / "iq_stereo.wav"
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float64) / sample_rate
    iq = 0.8 * np.exp(1j * 2 * np.pi * frequency * t)

    i_ch = (iq.real * 32767).astype(np.int16)
    q_ch = (iq.imag * 32767).astype(np.int16)
    stereo = np.column_stack([i_ch, q_ch]).flatten()

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())

    return path


def generate_raw_iq_cf32(
    path: Path | None = None,
    sample_rate: float = 2.4e6,
    duration: float = 0.01,
    frequency: float = 100e3,
) -> Path:
    """Generate a raw IQ file in cf32_le format."""
    if path is None:
        path = FIXTURES_DIR / "test.cf32"
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float64) / sample_rate
    iq = (0.8 * np.exp(1j * 2 * np.pi * frequency * t)).astype(np.complex64)
    # Interleaved I,Q float32
    interleaved = np.empty(2 * n, dtype=np.float32)
    interleaved[0::2] = iq.real
    interleaved[1::2] = iq.imag
    interleaved.tofile(str(path))
    return path


def generate_raw_iq_cu8(
    path: Path | None = None,
    sample_rate: float = 2.4e6,
    duration: float = 0.01,
    frequency: float = 100e3,
) -> Path:
    """Generate a raw IQ file in cu8 format (RTL-SDR style)."""
    if path is None:
        path = FIXTURES_DIR / "test.cu8"
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float64) / sample_rate
    iq = 0.8 * np.exp(1j * 2 * np.pi * frequency * t)
    # CU8: unsigned 8-bit, centred at 127.5
    i_u8 = ((iq.real * 127) + 127.5).clip(0, 255).astype(np.uint8)
    q_u8 = ((iq.imag * 127) + 127.5).clip(0, 255).astype(np.uint8)
    interleaved = np.empty(2 * n, dtype=np.uint8)
    interleaved[0::2] = i_u8
    interleaved[1::2] = q_u8
    interleaved.tofile(str(path))
    return path


def generate_sigmf_pair(
    base_path: Path | None = None,
    sample_rate: float = 2.4e6,
    center_freq: float = 145.5e6,
    duration: float = 0.01,
    frequency: float = 100e3,
) -> tuple[Path, Path]:
    """Generate a SigMF meta/data pair."""
    if base_path is None:
        base_path = FIXTURES_DIR / "test"
    meta_path = base_path.with_suffix(".sigmf-meta")
    data_path = base_path.with_suffix(".sigmf-data")

    # Data: cf32_le
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float64) / sample_rate
    iq = (0.8 * np.exp(1j * 2 * np.pi * frequency * t)).astype(np.complex64)
    interleaved = np.empty(2 * n, dtype=np.float32)
    interleaved[0::2] = iq.real
    interleaved[1::2] = iq.imag
    interleaved.tofile(str(data_path))

    # Metadata
    meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": sample_rate,
            "core:version": "1.2.6",
            "core:description": "Test signal",
            "core:num_channels": 1,
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": center_freq,
            }
        ],
        "annotations": [],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return meta_path, data_path


if __name__ == "__main__":
    print("Generating test fixtures…")
    generate_tone_wav()
    generate_stereo_iq_wav()
    generate_raw_iq_cf32()
    generate_raw_iq_cu8()
    generate_sigmf_pair()
    print(f"Done. Fixtures in {FIXTURES_DIR}")
