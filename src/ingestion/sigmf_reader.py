"""SigMF recording reader.

Reads ``.sigmf-meta`` (JSON) and ``.sigmf-data`` file pairs, as well as
``.sigmf-archive`` (tar) bundles per the SigMF v1.2.6 specification.
Extension namespaces are preserved rather than discarded.
"""

from __future__ import annotations

import json
import tarfile
import tempfile
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

# Mapping from SigMF core:datatype strings to our enum
_SIGMF_DTYPE_MAP: dict[str, SampleDatatype] = {
    "cf32_le": SampleDatatype.CF32_LE,
    "cf32_be": SampleDatatype.CF32_BE,
    "cf64_le": SampleDatatype.CF64_LE,
    "ci8":     SampleDatatype.CI8,
    "ci16_le": SampleDatatype.CI16_LE,
    "ci16_be": SampleDatatype.CI16_BE,
    "ci32_le": SampleDatatype.CI32_LE,
    "cu8":     SampleDatatype.CU8,
    "cu16_le": SampleDatatype.CU16_LE,
    "rf32_le": SampleDatatype.RF32_LE,
    "rf64_le": SampleDatatype.RF64_LE,
    "ri8":     SampleDatatype.RI8,
    "ri16_le": SampleDatatype.RI16_LE,
    "ri32_le": SampleDatatype.RI32_LE,
    "ru8":     SampleDatatype.RU8,
    "ru16_le": SampleDatatype.RU16_LE,
}


class SigMFReader(FileReader):
    """Read SigMF recordings (meta + data pairs or archive bundles)."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        self._meta_path: Path | None = None
        self._data_path: Path | None = None
        self._meta: dict | None = None
        self._temp_dir: tempfile.TemporaryDirectory | None = None  # type: ignore[type-arg]
        self._resolve_paths()

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve_paths(self) -> None:
        """Determine the meta and data file paths from the given path."""
        p = self.path

        if p.suffix == ".sigmf-meta":
            self._meta_path = p
            self._data_path = p.with_suffix(".sigmf-data")
        elif p.suffix == ".sigmf-data":
            self._data_path = p
            self._meta_path = p.with_suffix(".sigmf-meta")
        elif p.suffix == ".sigmf-archive":
            self._extract_archive(p)
        else:
            # Try to guess by looking for companion files
            meta_candidate = p.with_suffix(".sigmf-meta")
            data_candidate = p.with_suffix(".sigmf-data")
            if meta_candidate.exists():
                self._meta_path = meta_candidate
            if data_candidate.exists():
                self._data_path = data_candidate

    def _extract_archive(self, archive_path: Path) -> None:
        """Extract a .sigmf-archive tar to a temp directory."""
        self._temp_dir = tempfile.TemporaryDirectory(prefix="sigma_sigmf_")
        with tarfile.open(str(archive_path), "r") as tar:
            tar.extractall(self._temp_dir.name, filter="data")
        # Find the meta and data files inside
        tmp = Path(self._temp_dir.name)
        for f in tmp.rglob("*.sigmf-meta"):
            self._meta_path = f
            self._data_path = f.with_suffix(".sigmf-data")
            break

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_meta(self) -> dict:
        """Lazy-load the SigMF JSON metadata."""
        if self._meta is not None:
            return self._meta
        if self._meta_path is None or not self._meta_path.exists():
            raise FileNotFoundError(
                f"SigMF metadata file not found for {self.path}"
            )
        with open(self._meta_path, encoding="utf-8") as f:
            self._meta = json.load(f)
        return self._meta

    @property
    def _global(self) -> dict:
        return self._ensure_meta().get("global", {})

    @property
    def _captures(self) -> list[dict]:
        return self._ensure_meta().get("captures", [])

    @property
    def _annotations(self) -> list[dict]:
        return self._ensure_meta().get("annotations", [])

    def _datatype(self) -> SampleDatatype:
        dt_str = self._global.get("core:datatype", "cf32_le")
        return _SIGMF_DTYPE_MAP.get(dt_str, SampleDatatype.CF32_LE)

    # ------------------------------------------------------------------
    # FileReader interface
    # ------------------------------------------------------------------

    def validate(self) -> FileValidationReport:
        # Validate the data file
        if self._data_path is None:
            report = FileValidationReport(path=str(self.path))
            report.is_valid = False
            report.errors.append("SigMF data file not found.")
            return report

        report = validate_file_basics(self._data_path)
        if not report.is_valid:
            return report

        # Check meta file
        if self._meta_path is None or not self._meta_path.exists():
            report.warnings.append("SigMF metadata file not found.")
        else:
            try:
                self._ensure_meta()
                report.header_valid = True
            except (json.JSONDecodeError, KeyError) as exc:
                report.header_valid = False
                report.warnings.append(f"Invalid SigMF metadata: {exc}")

        # Sample alignment
        dt = self._datatype()
        bps = bytes_per_sample(dt)
        total = report.file_size_bytes // bps
        report.expected_sample_count = total
        report.actual_sample_count = total
        if report.file_size_bytes % bps != 0:
            report.sample_aligned = False
            report.warnings.append("Data file not aligned to sample size.")

        # Preview check
        try:
            preview = self.read_samples(0, min(100_000, total))
            report = validate_samples(preview, report)
        except Exception as exc:
            report.warnings.append(f"Numeric check failed: {exc}")

        report.sha256 = compute_checksum(self._data_path)
        return report

    def read_metadata(self) -> RecordingMetadata:
        self._ensure_meta()
        g = self._global
        dt = self._datatype()
        bps = bytes_per_sample(dt)

        sr = float(g.get("core:sample_rate", 0))
        cf = 0.0
        if self._captures:
            cf = float(self._captures[0].get("core:frequency", 0))

        data_size = self._data_path.stat().st_size if self._data_path else 0
        total = data_size // bps

        iq_order_str = g.get("rfanalyzer:iq_order", "IQ")
        try:
            iq_order = IQOrder(iq_order_str)
        except ValueError:
            iq_order = IQOrder.IQ

        return RecordingMetadata(
            source_path=str(self._data_path or self.path),
            source_format=FileFormat.SIGMF,
            sample_datatype=dt,
            sample_rate_hz=sr,
            sample_rate_status=(
                ParameterStatus.PROVIDED if sr > 0 else ParameterStatus.UNKNOWN
            ),
            center_frequency_hz=cf,
            center_frequency_status=(
                ParameterStatus.PROVIDED if cf > 0 else ParameterStatus.UNKNOWN
            ),
            iq_order=iq_order,
            num_channels=int(g.get("core:num_channels", 1)),
            sample_count=total,
            duration_seconds=total / sr if sr > 0 else 0.0,
            metadata_status="complete" if sr > 0 else "partial",
            notes=g.get("core:description", ""),
        )

    def read_samples(self, start: int = 0, count: int = -1) -> np.ndarray:
        if self._data_path is None or not self._data_path.exists():
            return np.array([], dtype=np.complex64)

        dt = self._datatype()
        bps = bytes_per_sample(dt)
        file_size = self._data_path.stat().st_size
        total = file_size // bps

        if count < 0:
            count = total - start
        end = min(start + count, total)
        if end <= start:
            return np.array([], dtype=np.complex64)

        byte_start = start * bps
        byte_count = (end - start) * bps

        mm = np.memmap(
            str(self._data_path),
            dtype=np.uint8,
            mode="r",
            offset=byte_start,
            shape=(byte_count,),
        )
        raw_bytes = bytes(mm)
        del mm

        iq_order_str = self._global.get("rfanalyzer:iq_order", "IQ")
        try:
            iq_order = IQOrder(iq_order_str)
        except ValueError:
            iq_order = IQOrder.IQ

        return normalize_samples(raw_bytes, dt, iq_order=iq_order)

    def total_samples(self) -> int:
        if self._data_path is None or not self._data_path.exists():
            return 0
        dt = self._datatype()
        bps = bytes_per_sample(dt)
        return self._data_path.stat().st_size // bps

    def get_raw_metadata(self) -> dict:
        """Return the full SigMF metadata dict (including extensions)."""
        return self._ensure_meta()

    def __del__(self) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
