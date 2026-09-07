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
        assert "Analysis Results" in html
        assert "BPSK" in html
        assert "95%" in html
        assert "Low SNR detected" in html
        assert "plot1.png" in html
