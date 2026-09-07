"""Unit tests for Pydantic data models."""

from datetime import datetime, timezone

from src.core.enums import (
    ConfidenceLevel,
    FileFormat,
    JobStatus,
    ModulationType,
    SampleDatatype,
)
from src.core.models import (
    AnalysisResult,
    FileValidationReport,
    InputHypothesis,
    ParameterEstimate,
    ProcessingJob,
    Project,
    RecordingMetadata,
    SignalRegion,
)


class TestProject:
    def test_defaults(self):
        p = Project()
        assert p.name == "Untitled Project"
        assert p.schema_version == "1.0.0"
        assert p.project_id  # non-empty UUID

    def test_custom_values(self):
        p = Project(name="Test", owner="alice")
        assert p.name == "Test"
        assert p.owner == "alice"


class TestRecordingMetadata:
    def test_defaults(self):
        m = RecordingMetadata()
        assert m.sample_rate_hz == 0.0
        assert m.source_format == FileFormat.RAW_IQ
        assert m.num_channels == 1

    def test_duration_calc(self):
        m = RecordingMetadata(
            sample_rate_hz=48000,
            sample_count=96000,
            duration_seconds=2.0,
        )
        assert m.duration_seconds == 2.0

    def test_serialization_roundtrip(self):
        m = RecordingMetadata(
            source_path="/tmp/test.iq",
            sample_rate_hz=2.4e6,
            center_frequency_hz=145.5e6,
            sample_datatype=SampleDatatype.CI16_LE,
        )
        data = m.model_dump(mode="json")
        m2 = RecordingMetadata.model_validate(data)
        assert m2.sample_rate_hz == m.sample_rate_hz
        assert m2.sample_datatype == m.sample_datatype


class TestFileValidationReport:
    def test_default_valid(self):
        r = FileValidationReport()
        assert r.is_valid is True
        assert len(r.warnings) == 0

    def test_with_warnings(self):
        r = FileValidationReport(
            warnings=["DC offset detected"],
            clipping_percentage=2.5,
        )
        assert len(r.warnings) == 1
        assert r.clipping_percentage == 2.5


class TestSignalRegion:
    def test_creation(self):
        sr = SignalRegion(
            start_sample=1000,
            end_sample=5000,
            snr_db=15.0,
            confidence=0.9,
        )
        assert sr.end_sample - sr.start_sample == 4000
        assert sr.confidence == 0.9


class TestAnalysisResult:
    def test_defaults(self):
        ar = AnalysisResult()
        assert ar.modulation == ModulationType.UNKNOWN
        assert ar.overall_confidence == ConfidenceLevel.UNKNOWN

    def test_with_candidates(self):
        ar = AnalysisResult(
            modulation=ModulationType.QPSK,
            modulation_confidence=0.85,
            modulation_candidates=[
                {"label": "QPSK", "confidence": 0.85},
                {"label": "BPSK", "confidence": 0.60},
            ],
        )
        assert len(ar.modulation_candidates) == 2


class TestProcessingJob:
    def test_lifecycle(self):
        j = ProcessingJob()
        assert j.status == JobStatus.QUEUED
        j.status = JobStatus.RUNNING
        j.progress = 0.5
        assert j.progress == 0.5


class TestParameterEstimate:
    def test_creation(self):
        pe = ParameterEstimate(
            parameter="sample_rate_hz",
            value=2400000,
            confidence=1.0,
            evidence=["sigmf.global.sample_rate"],
        )
        assert pe.value == 2400000
        assert len(pe.evidence) == 1


class TestInputHypothesis:
    def test_creation(self):
        h = InputHypothesis(
            label="Hypothesis A",
            sample_rate_hz=2.4e6,
            sample_datatype=SampleDatatype.CI16_LE,
        )
        assert h.label == "Hypothesis A"
        assert h.score == 0.0
