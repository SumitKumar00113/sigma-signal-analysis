"""WAV file reader.

Supports PCM 8/16/24/32-bit, IEEE float32, mono, stereo (I/Q), and
multi-channel WAV files.  Interpretation mode controls how channels map
to complex samples (PRD §6.1):

* ``STEREO_IQ`` – left = I, right = Q (optionally swapped).
* ``REAL`` / ``DUAL_CHANNEL`` / ``CUSTOM`` – one real channel, converted to
  its analytic signal (see :mod:`src.ingestion.real_to_complex`) so the
  spectrum is one-sided and signals keep their true audio frequency.
* ``DISCRIMINATOR`` – FM-discriminator audio, re-modulated to a complex
  FM signal whose instantaneous frequency follows the audio.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from src.core.enums import (
    FileFormat,
    ParameterStatus,
    SampleDatatype,
    WavInterpretation,
)
from src.core.models import FileValidationReport, RecordingMetadata
from src.ingestion.base import FileReader
from src.ingestion.real_to_complex import (
    InterpretationSuggestion,
    analytic_from_padded,
    discriminator_phase,
    hilbert_margin,
    padded_slice,
    suggest_wav_interpretation,
)
from src.ingestion.validator import compute_checksum, validate_file_basics, validate_samples


class WavReader(FileReader):
    """Read and validate standard WAV recordings."""

    def __init__(
        self,
        path: str | Path,
        interpretation: WavInterpretation = WavInterpretation.REAL,
        channel: int = 0,
        swap_iq: bool = False,
    ) -> None:
        super().__init__(path)
        self.interpretation = interpretation
        self.channel = channel          # real channel analysed for REAL / DUAL_CHANNEL / …
        self.swap_iq = swap_iq          # STEREO_IQ: left = Q, right = I
        self._disc_phase: np.ndarray | None = None
        self._metadata: RecordingMetadata | None = None
        self._raw_rate: int = 0
        self._raw_data: np.ndarray | None = None
        self._n_channels: int = 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy-load the WAV header (and data on first access)."""
        if self._raw_data is not None:
            return
        rate, data = wavfile.read(str(self.path))
        self._raw_rate = int(rate)
        if data.ndim == 1:
            self._n_channels = 1
            data = data.reshape(-1, 1)
        else:
            self._n_channels = data.shape[1]
        self._raw_data = data

    def _to_float(self, data: np.ndarray) -> np.ndarray:
        """Convert any integer PCM data to float32 in [-1, 1]."""
        if np.issubdtype(data.dtype, np.floating):
            return data.astype(np.float32)
        info = np.iinfo(data.dtype)
        return data.astype(np.float32) / max(abs(info.min), abs(info.max))

    # ------------------------------------------------------------------
    # FileReader interface
    # ------------------------------------------------------------------

    def validate(self) -> FileValidationReport:
        report = validate_file_basics(self.path)
        if not report.is_valid:
            return report

        try:
            with wave.open(str(self.path), "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                _ = wf.getframerate()  # consumed but not needed here
                n_frames = wf.getnframes()

            report.header_valid = True

            # Check sample alignment
            data_bytes = report.file_size_bytes - 44  # rough WAV header size
            expected_bytes = n_frames * n_channels * sampwidth
            if data_bytes < expected_bytes:
                report.truncated = True
                report.warnings.append("File appears truncated.")

            report.expected_sample_count = n_frames
            report.actual_sample_count = n_frames

        except (wave.Error, struct.error, EOFError) as exc:
            report.header_valid = False
            report.is_valid = False
            report.errors.append(f"Invalid WAV header: {exc}")
            return report

        # Numeric checks on a small preview
        try:
            self._ensure_loaded()
            assert self._raw_data is not None
            n_preview = min(100_000, len(self._raw_data))
            report = validate_samples(self.read_samples(0, n_preview), report)
        except Exception as exc:
            report.warnings.append(f"Could not run numeric checks: {exc}")

        report.sha256 = compute_checksum(self.path)
        return report

    def read_metadata(self) -> RecordingMetadata:
        self._ensure_loaded()
        assert self._raw_data is not None

        total = self._raw_data.shape[0]
        rate = self._raw_rate

        # Determine datatype label
        dt = self._raw_data.dtype
        if np.issubdtype(dt, np.floating):
            sample_dt = SampleDatatype.RF32_LE
        elif dt == np.int16:
            sample_dt = SampleDatatype.RI16_LE
        elif dt == np.int32:
            sample_dt = SampleDatatype.RI32_LE
        elif dt == np.uint8:
            sample_dt = SampleDatatype.RU8
        else:
            sample_dt = SampleDatatype.RI16_LE  # fallback

        self._metadata = RecordingMetadata(
            source_path=str(self.path),
            source_format=FileFormat.WAV,
            sample_datatype=sample_dt,
            sample_rate_hz=float(rate),
            sample_rate_status=ParameterStatus.PROVIDED,
            num_channels=self._n_channels,
            sample_count=total,
            duration_seconds=total / rate if rate > 0 else 0.0,
            wav_interpretation=self.interpretation,
            metadata_status="partial",
        )

        # Apply user hints
        for field_name in ("center_frequency_hz", "sample_rate_hz"):
            hint = self._get_hint(field_name)
            if hint is not None:
                setattr(self._metadata, field_name, float(hint))  # type: ignore[arg-type]

        return self._metadata

    def read_samples(self, start: int = 0, count: int = -1) -> np.ndarray:
        self._ensure_loaded()
        assert self._raw_data is not None

        total = self._raw_data.shape[0]
        if count < 0:
            count = total - start
        end = min(start + count, total)
        start = max(0, start)
        if end <= start:
            return np.zeros(0, dtype=np.complex64)

        if self.interpretation == WavInterpretation.STEREO_IQ and self._n_channels >= 2:
            chunk = self._to_float(self._raw_data[start:end])
            i, q = (1, 0) if self.swap_iq else (0, 1)
            return (chunk[:, i] + 1j * chunk[:, q]).astype(np.complex64)

        ch = self._real_channel()
        if self.interpretation == WavInterpretation.DISCRIMINATOR:
            if self._disc_phase is None:
                self._disc_phase = discriminator_phase(self._to_float(self._raw_data[:, ch]))
            return np.exp(1j * self._disc_phase[start:end]).astype(np.complex64)

        # Real signal → analytic signal (one-sided spectrum), with enough
        # context around the chunk that chunked reads match a full read
        margin = hilbert_margin()
        column = self._raw_data[:, ch]
        padded = self._to_float(padded_slice(column, start, end, margin))
        return analytic_from_padded(padded)

    def _real_channel(self) -> int:
        if not 0 <= self.channel < self._n_channels:
            raise ValueError(f"Channel {self.channel} not in file ({self._n_channels} channels)")
        return self.channel

    def suggest_interpretation(self) -> InterpretationSuggestion:
        """Heuristic guess of how this file's channels should be read."""
        self._ensure_loaded()
        assert self._raw_data is not None
        preview = self._to_float(self._raw_data[: min(200_000, len(self._raw_data))])
        return suggest_wav_interpretation(preview, float(self._raw_rate))

    def total_samples(self) -> int:
        self._ensure_loaded()
        assert self._raw_data is not None
        return int(self._raw_data.shape[0])
