"""File validation engine.

Performs all integrity and quality checks defined in PRD §6.2 without
modifying the source file.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from src.core.models import FileValidationReport


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Compute SHA-256 hex digest of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def validate_file_basics(path: str | Path) -> FileValidationReport:
    """Check existence, permissions, and size."""
    path = Path(path)
    report = FileValidationReport(path=str(path))

    if not path.exists():
        report.exists = False
        report.is_valid = False
        report.errors.append("File does not exist.")
        return report
    report.exists = True

    if not os.access(path, os.R_OK):
        report.readable = False
        report.is_valid = False
        report.errors.append("File is not readable (permission denied).")
        return report
    report.readable = True

    report.file_size_bytes = path.stat().st_size
    if report.file_size_bytes == 0:
        report.is_valid = False
        report.errors.append("File is empty (0 bytes).")
        return report

    return report


def validate_samples(
    samples: np.ndarray,
    report: FileValidationReport,
    *,
    clipping_threshold: float = 0.99,
) -> FileValidationReport:
    """Run numeric quality checks on a sample array.

    Parameters
    ----------
    samples:
        1-D complex64 array of samples to inspect.
    report:
        An existing :class:`FileValidationReport` to extend in-place.
    clipping_threshold:
        Fraction of max representable amplitude above which a sample is
        considered clipped.
    """
    if samples.size == 0:
        report.warnings.append("Sample array is empty — no numeric checks run.")
        return report

    real = samples.real.astype(np.float32)
    imag = samples.imag.astype(np.float32)

    # NaN / Inf
    report.nan_count = int(np.isnan(real).sum() + np.isnan(imag).sum())
    report.inf_count = int(np.isinf(real).sum() + np.isinf(imag).sum())
    if report.nan_count > 0:
        report.warnings.append(f"Found {report.nan_count} NaN values.")
    if report.inf_count > 0:
        report.warnings.append(f"Found {report.inf_count} Inf values.")

    # DC offset
    report.dc_offset_i = float(np.mean(real))
    report.dc_offset_q = float(np.mean(imag))
    dc_mag = np.sqrt(report.dc_offset_i**2 + report.dc_offset_q**2)
    rms = float(np.sqrt(np.mean(np.abs(samples) ** 2)))
    if rms > 0 and dc_mag / rms > 0.05:
        report.warnings.append(
            f"Significant DC offset detected (I={report.dc_offset_i:.4f}, "
            f"Q={report.dc_offset_q:.4f}, RMS={rms:.4f})."
        )

    # Clipping
    peak = float(np.max(np.abs(samples)))
    if peak > 0:
        clipped = int(np.sum(np.abs(samples) > clipping_threshold * peak))
        report.clipping_percentage = 100.0 * clipped / samples.size
        if report.clipping_percentage > 1.0:
            report.warnings.append(
                f"Clipping detected: {report.clipping_percentage:.1f}% of samples."
            )

    # IQ imbalance (power ratio)
    power_i = float(np.mean(real**2))
    power_q = float(np.mean(imag**2))
    if power_q > 0:
        imbalance = 10 * np.log10(power_i / power_q) if power_i > 0 else 0.0
        report.iq_imbalance_db = float(imbalance)
        if abs(imbalance) > 1.0:
            report.warnings.append(
                f"IQ gain imbalance: {imbalance:.2f} dB."
            )

    return report


def compute_checksum(path: str | Path) -> str:
    """Return the SHA-256 hex digest of the file."""
    return _sha256(Path(path))
