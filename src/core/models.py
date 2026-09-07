"""Pydantic data models for the Sigma Signal Analysis platform.

These models form the canonical in-memory representation for projects,
recordings, detected signal regions, analysis results, and processing jobs.
They map directly to the data model defined in PRD §21.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from src.core.enums import (
    ConfidenceLevel,
    DetectionMethod,
    FECType,
    FileFormat,
    InterleaverType,
    IQOrder,
    JobStatus,
    ModulationType,
    ParameterStatus,
    SampleDatatype,
    WavInterpretation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------

class Project(BaseModel):
    """Top-level container grouping recordings and analysis results."""

    project_id: str = Field(default_factory=_uuid)
    name: str = "Untitled Project"
    description: str = ""
    owner: str = "analyst"
    schema_version: str = "1.0.0"
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Recording metadata
# ---------------------------------------------------------------------------

class RecordingMetadata(BaseModel):
    """Metadata describing a single recording file.

    Some fields are *required* for correct processing (e.g. ``sample_rate_hz``),
    while others are informational.  Fields whose provenance is uncertain
    carry an explicit :class:`ParameterStatus`.
    """

    recording_id: str = Field(default_factory=_uuid)
    project_id: str = ""
    source_path: str = ""
    source_format: FileFormat = FileFormat.RAW_IQ

    # Checksum (SHA-256 hex) — filled after validation
    sha256: str = ""

    # Sample description
    sample_datatype: SampleDatatype = SampleDatatype.CF32_LE
    sample_rate_hz: float = 0.0
    sample_rate_status: ParameterStatus = ParameterStatus.UNKNOWN
    center_frequency_hz: float = 0.0
    center_frequency_status: ParameterStatus = ParameterStatus.UNKNOWN
    iq_order: IQOrder = IQOrder.IQ
    num_channels: int = 1
    sample_count: int = 0
    duration_seconds: float = 0.0
    amplitude_scale: float = 1.0
    byte_offset: int = 0

    # WAV-specific
    wav_interpretation: WavInterpretation = WavInterpretation.REAL

    # Optional provenance
    sensor: str = ""
    location: str = ""
    time_origin: str = ""
    notes: str = ""

    metadata_status: str = "incomplete"
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------

class FileValidationReport(BaseModel):
    """Result of running the file-integrity and quality checks."""

    path: str = ""
    exists: bool = False
    readable: bool = False
    file_size_bytes: int = 0
    header_valid: bool = True
    truncated: bool = False
    sample_aligned: bool = True
    expected_sample_count: int = 0
    actual_sample_count: int = 0
    nan_count: int = 0
    inf_count: int = 0
    clipping_percentage: float = 0.0
    dc_offset_i: float = 0.0
    dc_offset_q: float = 0.0
    iq_imbalance_db: float = 0.0
    sha256: str = ""
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    is_valid: bool = True


# ---------------------------------------------------------------------------
# Signal region
# ---------------------------------------------------------------------------

class SignalRegion(BaseModel):
    """A candidate signal detected inside a recording."""

    region_id: str = Field(default_factory=_uuid)
    recording_id: str = ""
    start_sample: int = 0
    end_sample: int = 0
    start_time_sec: float = 0.0
    end_time_sec: float = 0.0
    center_frequency_hz: float = 0.0
    bandwidth_hz: float = 0.0
    snr_db: float = 0.0
    detection_method: DetectionMethod = DetectionMethod.MANUAL
    confidence: float = 0.0
    annotation_source: str = "manual"
    label: str = ""


# ---------------------------------------------------------------------------
# Parameter estimate
# ---------------------------------------------------------------------------

class ParameterEstimate(BaseModel):
    """A single estimated parameter with provenance and alternatives."""

    parameter: str = ""
    value: Any = None
    status: ParameterStatus = ParameterStatus.UNKNOWN
    confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    alternatives: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Analysis result
# ---------------------------------------------------------------------------

class AnalysisResult(BaseModel):
    """Outcome of running the analysis pipeline on a signal region."""

    analysis_id: str = Field(default_factory=_uuid)
    region_id: str = ""
    algorithm_version: str = "0.1.0"

    # Estimated parameters
    modulation: ModulationType = ModulationType.UNKNOWN
    modulation_confidence: float = 0.0
    symbol_rate_hz: float = 0.0
    symbol_rate_confidence: float = 0.0
    fec_type: FECType = FECType.UNKNOWN
    fec_confidence: float = 0.0
    interleaver_type: InterleaverType = InterleaverType.UNKNOWN
    interleaver_confidence: float = 0.0

    # Modulation candidates (ranked)
    modulation_candidates: list[dict[str, Any]] = Field(default_factory=list)

    # Detailed parameter estimates
    parameters: list[ParameterEstimate] = Field(default_factory=list)

    # Artefact paths (plots, decoded bits, etc.)
    artifacts: list[str] = Field(default_factory=list)

    # Overall quality
    overall_confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    warnings: list[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Processing job
# ---------------------------------------------------------------------------

class ProcessingJob(BaseModel):
    """Tracks a background processing task."""

    job_id: str = Field(default_factory=_uuid)
    job_type: str = "analysis"
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0  # 0.0 – 1.0

    submitted_at: datetime = Field(default_factory=_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    cpu_seconds: float = 0.0
    memory_peak_bytes: int = 0
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Hypothesis (for the Input Characterization Wizard)
# ---------------------------------------------------------------------------

class InputHypothesis(BaseModel):
    """A candidate interpretation of a raw recording."""

    hypothesis_id: str = Field(default_factory=_uuid)
    label: str = "Hypothesis"
    sample_rate_hz: float = 0.0
    sample_datatype: SampleDatatype = SampleDatatype.CF32_LE
    iq_order: IQOrder = IQOrder.IQ
    center_frequency_hz: float = 0.0
    byte_offset: int = 0
    score: float = 0.0
    evidence: list[str] = Field(default_factory=list)
