"""Unit tests for the export engine."""

import json
from pathlib import Path

from src.core.enums import ConfidenceLevel, FileFormat, ModulationType, SampleDatatype
from src.core.models import AnalysisResult, RecordingMetadata
from src.reporting.exporter import export_analysis_json, export_html_report, export_metadata_json


class TestExporter:
    def test_export_metadata_json(self, tmp_path: Path):
        meta = RecordingMetadata(
            source_path="/fake/path.iq",
            source_format=FileFormat.RAW_IQ,
            sample_rate_hz=1e6,
            sample_datatype=SampleDatatype.CF32_LE,
        )
        out_path = tmp_path / "meta.json"
        export_metadata_json(meta, out_path)

        assert out_path.exists()
        loaded = json.loads(out_path.read_text())
        assert loaded["source_path"] == "/fake/path.iq"
        assert loaded["sample_rate_hz"] == 1000000.0
        assert loaded["source_format"] == "raw_iq"

    def test_export_analysis_json(self, tmp_path: Path):
        result = AnalysisResult(
            modulation=ModulationType.QPSK,
            modulation_confidence=0.9,
            overall_confidence=ConfidenceLevel.CONFIRMED,
            symbol_rate_hz=250000.0,
        )
        out_path = tmp_path / "analysis.json"
        export_analysis_json(result, out_path)

        assert out_path.exists()
        loaded = json.loads(out_path.read_text())
        assert loaded["modulation"] == "QPSK"
        assert loaded["symbol_rate_hz"] == 250000.0
        assert loaded["overall_confidence"] == "Confirmed"

    def test_export_html_report_basic(self, tmp_path: Path):
        meta = RecordingMetadata(source_path="test.wav", sample_rate_hz=48000)
        out_path = tmp_path / "report.html"

        export_html_report(meta, path=out_path)

        assert out_path.exists()
        html = out_path.read_text()
        assert "Sigma Signal Analysis Report" in html
        assert "test.wav" in html
        assert "48,000 Hz" in html

    def test_export_html_report_full(self, tmp_path: Path):
        meta = RecordingMetadata(source_path="test.wav", sample_rate_hz=48000)
        result = AnalysisResult(
            modulation=ModulationType.BPSK,
            modulation_confidence=0.95,
            warnings=["Low SNR detected"],
        )
        out_path = tmp_path / "report_full.html"

        export_html_report(meta, result=result, path=out_path, plot_paths=["plot1.png"])

        assert out_path.exists()
        html = out_path.read_text()
        assert "<h2>Classification</h2>" in html
        assert "BPSK" in html
        assert "95%" in html
        assert "Low SNR detected" in html
        assert "plot1.png" in html


class TestFullReport:
    """Reports built from a real pipeline run plus the decoding chain."""

    @staticmethod
    def _coded_qpsk():
        import numpy as np

        from src.decoding.auto_decode import auto_decode
        from src.decoding.correlation import parse_pattern
        from src.decoding.viterbi import ConvCode, conv_encode
        from src.dsp import synth
        from src.dsp.pipeline import AnalysisPipeline

        rng = np.random.default_rng(6)
        np.random.seed(6)
        asm = parse_pattern("0x1ACFFC1D")
        data = np.concatenate([np.concatenate([asm, rng.integers(0, 2, 992, dtype=np.uint8)])
                               for _ in range(12)])
        sig = synth.modulate_bits(conv_encode(data, ConvCode(7, (0o171, 0o133)),
                                              terminate=False),
                                  "qpsk", 4800, 48000, snr_db=9, freq_offset_hz=700)
        meta = RecordingMetadata(source_path="<coded & framed>.iq", sample_rate_hz=48000,
                                 notes="<b>not markup</b>")
        result = AnalysisPipeline().run(sig, meta)
        chain = auto_decode(result.demod.bits, result.demod.bits_per_symbol,
                            result.demod.modulation, ldpc_codes=[], search_interleaver=False)
        return meta, result, chain

    def test_html_has_every_stage(self, tmp_path: Path):
        meta, result, chain = self._coded_qpsk()
        out = tmp_path / "r.html"
        export_html_report(meta, path=out, pipeline=result, decode=chain)
        html = out.read_text()
        for heading in ("Recording", "File integrity", "Signal detection", "Classification",
                        "Signal parameters", "Demodulation", "Decoding"):
            assert f"<h2>{heading}</h2>" in html
        assert "0x1ACFFC1D" in html and "Viterbi" in html
        assert "Rules vs learned model" in html and "<svg" in html
        from src import __version__
        assert f"Sigma v{__version__}" in html

    def test_html_escapes_text_and_bounds_confidence(self, tmp_path: Path):
        meta, result, _ = self._coded_qpsk()
        result.analysis.symbol_rate_confidence = 1.56     # unbounded estimator score
        out = tmp_path / "r.html"
        export_html_report(meta, path=out, pipeline=result)
        html = out.read_text()
        assert "<b>not markup</b>" not in html and "&lt;b&gt;not markup&lt;/b&gt;" in html
        assert "156%" not in html
        assert "decoding chain was not run" in html

    def test_json_includes_decoding(self, tmp_path: Path):
        from src.reporting.exporter import export_pipeline_json

        meta, result, chain = self._coded_qpsk()
        out = tmp_path / "r.json"
        export_pipeline_json(result, out, decode=chain)
        data = json.loads(out.read_text())
        assert data["classifier_source"] == result.classifier_source
        dec = data["decoding"]
        assert dec["framing"]["found"] and dec["framing"]["sync"] == "0x1ACFFC1D"
        assert dec["fec"]["type"] == "convolutional" and dec["final_bits"] > 0
        assert data["bursts"] is not None
