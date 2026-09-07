"""WAV file reader.

Supports PCM 8/16/24/32-bit, IEEE float32, mono, stereo (I/Q), and
multi-channel WAV files.  Interpretation mode controls how channels map
to complex samples (PRD §6.1).
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
from src.ingestion.validator import compute_checksum, validate_file_basics, validate_samples


class WavReader(FileReader):
    """Read and validate standard WAV recordings."""

    def __init__(
        self,
        path: str | Path,
        interpretation: WavInterpretation = WavInterpretation.REAL,
    ) -> None:
        super().__init__(path)
        self.interpretation = interpretation
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
            preview = self._to_float(self._raw_data[:min(100_000, len(self._raw_data))])
            preview_complex = preview[:, 0] + 0j
            report = validate_samples(preview_complex, report)
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
        chunk = self._to_float(self._raw_data[start:end])

        if self.interpretation == WavInterpretation.STEREO_IQ and self._n_channels >= 2:
            # Left = I, Right = Q
            return (chunk[:, 0] + 1j * chunk[:, 1]).astype(np.complex64)
        else:
            # Mono / real
            return (chunk[:, 0] + 0j).astype(np.complex64)

    def total_samples(self) -> int:
        self._ensure_loaded()
        assert self._raw_data is not None
        return int(self._raw_data.shape[0])
