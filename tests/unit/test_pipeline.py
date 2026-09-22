"""Unit tests for the analysis pipeline."""

import numpy as np
import pytest

from src.core.enums import ConfidenceLevel, FileFormat, SampleDatatype
from src.core.exceptions import ProcessingError
from src.core.models import RecordingMetadata
from src.dsp.pipeline import AnalysisPipeline, PipelineConfig


def _tone(freq: float, sr: float = 10000.0, n: int = 16384) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    return (np.exp(1j * 2 * np.pi * freq * t)).astype(np.complex64)


class TestAnalysisPipeline:
    @pytest.fixture
    def meta(self) -> RecordingMetadata:
        return RecordingMetadata(
            source_path="test.iq",
            source_format=FileFormat.RAW_IQ,
            sample_rate_hz=10000.0,
            sample_datatype=SampleDatatype.CF32_LE,
        )

    def test_pipeline_on_synthetic_tone(self, meta: RecordingMetadata):
        # Generate a strong tone at 1000 Hz
        sig = _tone(1000.0, sr=meta.sample_rate_hz)

        pipeline = AnalysisPipeline()

        # Track progress calls
        progress_calls = []
        pipeline.on_progress = lambda f, m: progress_calls.append((f, m))

        result = pipeline.run(sig, meta)

        # Verify progress was emitted
        assert len(progress_calls) > 0
        assert progress_calls[-1][0] == 1.0

        # Verify results
        assert result.samples_processed == len(sig)
        assert result.processing_time_ms > 0

        # Should have found a region
        assert len(result.regions) == 1

        # Validation should be clean (pure tone, no clipping if normalized)
        assert result.validation.nan_count == 0

        # Analysis should have caught the offset
        assert any("offset" in w.lower() and "1,000" in w for w in result.analysis.warnings)

    def test_pipeline_on_noise(self, meta: RecordingMetadata):
        # Generate pure noise
        noise = (np.random.randn(16384) + 1j * np.random.randn(16384)).astype(np.complex64)

        pipeline = AnalysisPipeline()
        result = pipeline.run(noise, meta)

        # Should find no regions
        assert len(result.regions) == 0
        assert result.analysis.overall_confidence == ConfidenceLevel.UNKNOWN

    def test_pipeline_empty_samples(self, meta: RecordingMetadata):
        pipeline = AnalysisPipeline()

        with pytest.raises(ProcessingError):
            pipeline.run(np.array([], dtype=np.complex64), meta)

    def test_pipeline_config_disables_features(self, meta: RecordingMetadata):
        sig = _tone(1000.0, sr=meta.sample_rate_hz)

        config = PipelineConfig(
            detect_regions=False,
            measure_snr=False,
            measure_freq_offset=False,
            measure_symbol_rate=False
        )
        pipeline = AnalysisPipeline(config=config)
        result = pipeline.run(sig, meta)

        assert len(result.regions) == 0
        assert len(result.analysis.warnings) == 0
        assert result.analysis.symbol_rate_hz == 0.0


class TestFullPipeline:
    """Classification and demodulation stages."""

    def test_bpsk_end_to_end(self):
        from src.core.enums import ModulationType
        from tests.signals.generators import generate_bpsk
        np.random.seed(0)
        sig, bits = generate_bpsk(4000, 10e3, 200e3, 12.0, 800.0)
        meta = RecordingMetadata(sample_rate_hz=200e3, source_format=FileFormat.RAW_IQ)
        res = AnalysisPipeline().run(sig, meta)
        assert res.analysis.modulation == ModulationType.BPSK
        assert abs(res.analysis.symbol_rate_hz - 10e3) < 50
        assert abs(res.snr_db - 12.0) < 1.5
        assert abs(res.frequency_offset_hz - 800.0) < 60
        assert res.demod is not None and res.demod.num_bits > 3000
        assert res.analysis.overall_confidence in (
            ConfidenceLevel.CONFIRMED, ConfidenceLevel.PROBABLE)
        assert not res.stage_errors

    def test_overrides(self):
        from src.core.enums import ModulationType
        from tests.signals.generators import generate_qpsk
        np.random.seed(0)
        sig, _ = generate_qpsk(3000, 12.5e3, 250e3, 12.0)
        meta = RecordingMetadata(sample_rate_hz=250e3, source_format=FileFormat.RAW_IQ)
        cfg = PipelineConfig(modulation_override=ModulationType.BPSK, symbol_rate_override=12.5e3)
        res = AnalysisPipeline(cfg).run(sig, meta)
        assert res.analysis.modulation == ModulationType.BPSK
        assert res.analysis.symbol_rate_hz == 12.5e3
        assert res.classification is None
        assert res.demod is not None and res.demod.bits_per_symbol == 1

    def test_stages_can_be_disabled(self):
        from tests.signals.generators import generate_qpsk
        np.random.seed(0)
        sig, _ = generate_qpsk(2000, 12.5e3, 250e3, 12.0)
        meta = RecordingMetadata(sample_rate_hz=250e3, source_format=FileFormat.RAW_IQ)
        res = AnalysisPipeline(PipelineConfig(classify=False, demodulate=False)).run(sig, meta)
        assert res.classification is None and res.demod is None
        assert res.analysis.symbol_rate_hz > 0
