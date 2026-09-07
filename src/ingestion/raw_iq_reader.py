"""Raw IQ file reader.

Supports all binary sample formats listed in the PRD (cf32, ci16, cu8, etc.)
with configurable endianness, IQ ordering, byte offset, and memory-mapped
large-file access.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.enums import (
    FileFormat,
    IQOrder,
    ParameterStatus,
    SampleDatatype,
)
from src.core.models import FileValidationReport, RecordingMetadata
from src.ingestion.base import FileReader
from src.ingestion.normalizer import bytes_per_sample, normalize_samples
from src.ingestion.validator import compute_checksum, validate_file_basics, validate_samples


class RawIQReader(FileReader):
    """Read raw IQ binary files with user-specified metadata.

    Because raw IQ files carry no header, the caller MUST supply at minimum:
    * ``sample_datatype``
    * ``sample_rate_hz``
    via ``set_metadata_hints()`` before calling :meth:`read_metadata` or
    :meth:`read_samples`.
    """

    def __init__(
        self,
        path: str | Path,
        datatype: SampleDatatype = SampleDatatype.CF32_LE,
        sample_rate_hz: float = 0.0,
        iq_order: IQOrder = IQOrder.IQ,
        byte_offset: int = 0,
    ) -> None:
        super().__init__(path)
        self.datatype = datatype
        self.sample_rate_hz = sample_rate_hz
        self.iq_order = iq_order
        self.byte_offset = byte_offset

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _data_bytes(self) -> int:
        """Return number of payload bytes (file size minus byte offset)."""
        return max(0, self.path.stat().st_size - self.byte_offset)

    # ------------------------------------------------------------------
    # FileReader interface
    # ------------------------------------------------------------------

    def validate(self) -> FileValidationReport:
        report = validate_file_basics(self.path)
        if not report.is_valid:
            return report

        bps = bytes_per_sample(self.datatype)
        data_size = self._data_bytes()

        # Sample alignment
        if data_size % bps != 0:
            report.sample_aligned = False
            report.warnings.append(
                f"File payload ({data_size} bytes) is not aligned to "
                f"sample size ({bps} bytes)."
            )

        total = data_size // bps
        report.expected_sample_count = total
        report.actual_sample_count = total

        # Numeric checks on a preview
        try:
            preview_count = min(100_000, total)
            preview = self.read_samples(0, preview_count)
            report = validate_samples(preview, report)
        except Exception as exc:
            report.warnings.append(f"Could not run numeric checks: {exc}")

        report.sha256 = compute_checksum(self.path)
        return report

    def read_metadata(self) -> RecordingMetadata:
        bps = bytes_per_sample(self.datatype)
        data_size = self._data_bytes()
        total = data_size // bps
        duration = total / self.sample_rate_hz if self.sample_rate_hz > 0 else 0.0

        center_freq = float(self._get_hint("center_frequency_hz", 0.0) or 0.0)

        return RecordingMetadata(
            source_path=str(self.path),
            source_format=FileFormat.RAW_IQ,
            sample_datatype=self.datatype,
            sample_rate_hz=self.sample_rate_hz,
            sample_rate_status=(
                ParameterStatus.PROVIDED if self.sample_rate_hz > 0
                else ParameterStatus.UNKNOWN
            ),
            center_frequency_hz=center_freq,
            center_frequency_status=(
                ParameterStatus.PROVIDED if center_freq > 0
                else ParameterStatus.UNKNOWN
            ),
            iq_order=self.iq_order,
            num_channels=1,
            sample_count=total,
            duration_seconds=duration,
            byte_offset=self.byte_offset,
            metadata_status="partial",
        )

    def read_samples(self, start: int = 0, count: int = -1) -> np.ndarray:
        bps = bytes_per_sample(self.datatype)
        data_size = self._data_bytes()
        total = data_size // bps

        if count < 0:
            count = total - start
        end = min(start + count, total)
        if end <= start:
            return np.array([], dtype=np.complex64)

        # Read raw bytes using memory mapping for large files
        byte_start = self.byte_offset + start * bps
        byte_count = (end - start) * bps

        mm = np.memmap(
            str(self.path),
            dtype=np.uint8,
            mode="r",
            offset=byte_start,
            shape=(byte_count,),
        )
        raw_bytes = bytes(mm)
        del mm

        return normalize_samples(
            raw_bytes,
            self.datatype,
            iq_order=self.iq_order,
        )

    def total_samples(self) -> int:
        bps = bytes_per_sample(self.datatype)
        return self._data_bytes() // bps
