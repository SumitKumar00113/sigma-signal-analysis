"""Export engine for analysis results.

Supports JSON metadata export, HTML reports, and PNG plot snapshots.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.models import AnalysisResult, RecordingMetadata

if TYPE_CHECKING:
    from src.dsp.pipeline import PipelineResult


def export_metadata_json(meta: RecordingMetadata, path: str | Path) -> None:
    """Write recording metadata as pretty-printed JSON."""
    data = meta.model_dump(mode="json")
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def export_analysis_json(result: AnalysisResult, path: str | Path) -> None:
    """Write analysis result as pretty-printed JSON."""
    data = result.model_dump(mode="json")
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def pipeline_result_to_dict(result: PipelineResult, max_bits: int = 1_000_000) -> dict[str, Any]:
    """Serialise a full pipeline result (metadata, analysis, regions, bits)."""
    d = result.demod
    out: dict[str, Any] = {
        "schema_version": "1.1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "metadata": result.metadata.model_dump(mode="json"),
        "validation": result.validation.model_dump(mode="json"),
        "analysis": result.analysis.model_dump(mode="json"),
        "regions": [r.model_dump(mode="json") for r in result.regions],
        "measurements": {
            "snr_db": result.snr_db,
            "snr_inband_db": result.snr_inband_db,
            "occupied_bandwidth_hz": result.occupied_bandwidth_hz,
            "frequency_offset_hz": result.frequency_offset_hz,
            "symbol_rate_candidates": [
                {"rate_hz": r, "confidence": c} for r, c in result.symbol_rate_candidates
            ],
        },
        "classification": None,
        "demodulation": None,
        "stage_errors": dict(result.stage_errors),
        "processing_time_ms": result.processing_time_ms,
        "samples_processed": result.samples_processed,
    }
    if result.classification is not None:
        c = result.classification
        out["classification"] = {
            "modulation": c.modulation.value,
            "confidence": c.confidence,
            "candidates": [{"modulation": m.value, "probability": p} for m, p in c.candidates],
            "features": {k: float(v) for k, v in c.features.items()},
            "evidence": list(c.evidence),
        }
    if d is not None:
        bits = d.bits[:max_bits]
        n = (len(bits) // 8) * 8
        out["demodulation"] = {
            "modulation": d.modulation.value,
            "symbol_rate_hz": d.symbol_rate_hz,
            "samples_per_symbol": d.samples_per_symbol,
            "bits_per_symbol": d.bits_per_symbol,
            "num_symbols": d.num_symbols,
            "num_bits": d.num_bits,
            "evm_percent": d.evm_percent,
            "coarse_cfo_hz": d.coarse_cfo_hz,
            "residual_cfo_hz": d.residual_cfo_hz,
            "fsk_levels_hz": list(d.fsk_levels_hz),
            "warnings": list(d.warnings),
            "bits_hex": np.packbits(bits[:n]).tobytes().hex() if n else "",
            "bits_truncated": len(d.bits) > max_bits,
        }
    return out


def export_pipeline_json(result: PipelineResult, path: str | Path) -> None:
    """Write a complete pipeline result as JSON."""
    data = pipeline_result_to_dict(result)
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def export_html_report(
    meta: RecordingMetadata,
    result: AnalysisResult | None = None,
    path: str | Path = "report.html",
    plot_paths: list[str] | None = None,
    pipeline: PipelineResult | None = None,
) -> None:
    """Generate an HTML analysis report.

    Pass *pipeline* to include measurements, classifier evidence, and a
    preview of the recovered bit stream.
    """
    now = datetime.now(UTC).isoformat()
    if pipeline is not None and result is None:
        result = pipeline.analysis

    plots_html = ""
    if plot_paths:
        for p in plot_paths:
            plots_html += f'<img src="{p}" style="max-width:100%;margin:8px 0;">\n'

    result_html = ""
    if result:
        result_html = f"""
        <h2>Analysis Results</h2>
        <table>
            <tr><td>Modulation</td><td>{result.modulation.value}</td>
                <td>Confidence: {result.modulation_confidence:.0%}</td></tr>
            <tr><td>Symbol Rate</td><td>{result.symbol_rate_hz:,.0f} Hz</td>
                <td>Confidence: {result.symbol_rate_confidence:.0%}</td></tr>
            <tr><td>FEC</td><td>{result.fec_type.value}</td>
                <td>Confidence: {result.fec_confidence:.0%}</td></tr>
            <tr><td>Overall</td><td colspan="2">{result.overall_confidence.value}</td></tr>
        </table>
        """
        if result.modulation_candidates:
            result_html += "<h3>Modulation candidates</h3><ul>"
            for c in result.modulation_candidates[:5]:
                result_html += f"<li>{c['modulation']}: {c['probability']:.0%}</li>"
            result_html += "</ul>"
        if result.warnings:
            result_html += "<h3>Warnings</h3><ul>"
            for w in result.warnings:
                result_html += f"<li>{w}</li>"
            result_html += "</ul>"

    if pipeline is not None:
        result_html += f"""
        <h2>Measurements</h2>
        <table>
            <tr><td>SNR (full band)</td><td>{pipeline.snr_db:.1f} dB</td></tr>
            <tr><td>SNR (in-band)</td><td>{pipeline.snr_inband_db:.1f} dB</td></tr>
            <tr><td>Carrier offset</td><td>{pipeline.frequency_offset_hz:+,.1f} Hz</td></tr>
            <tr><td>Occupied bandwidth</td><td>{pipeline.occupied_bandwidth_hz:,.0f} Hz</td></tr>
            <tr><td>Regions detected</td><td>{len(pipeline.regions)}</td></tr>
            <tr><td>Processing time</td><td>{pipeline.processing_time_ms:,.0f} ms</td></tr>
        </table>
        """
        if pipeline.classification and pipeline.classification.evidence:
            result_html += "<h3>Classifier evidence</h3><p class='meta'>"
            result_html += " ".join(pipeline.classification.evidence) + "</p>"
        d = pipeline.demod
        if d is not None:
            n = min(len(d.bits), 2048)
            preview = np.packbits(d.bits[: (n // 8) * 8]).tobytes().hex(" ").upper()
            result_html += f"""
            <h2>Demodulation</h2>
            <table>
                <tr><td>Symbols</td><td>{d.num_symbols:,}</td></tr>
                <tr><td>Bits</td><td>{d.num_bits:,} ({d.bits_per_symbol} per symbol)</td></tr>
                <tr><td>EVM</td><td>{d.evm_percent:.1f} %</td></tr>
                <tr><td>Residual CFO</td><td>{d.residual_cfo_hz:+.1f} Hz</td></tr>
            </table>
            <h3>Bit stream (first {n:,} bits, hex)</h3>
            <pre style="white-space:pre-wrap;word-break:break-all;color:#94a3b8;">{preview}</pre>
            """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sigma Signal Analysis Report</title>
<style>
    body {{
        font-family: 'Inter', 'Segoe UI', sans-serif;
        background: #0f1420; color: #e2e8f0;
        max-width: 900px; margin: 0 auto; padding: 24px;
    }}
    h1 {{ color: #4ea8f5; border-bottom: 2px solid #1e293b; padding-bottom: 8px; }}
    h2 {{ color: #7c5cfc; margin-top: 24px; }}
    table {{
        width: 100%; border-collapse: collapse; margin: 12px 0;
    }}
    td {{
        padding: 8px 12px; border: 1px solid #1e293b;
    }}
    tr:nth-child(even) {{ background: #151c2c; }}
    .meta {{ color: #94a3b8; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>⚡ Sigma Signal Analysis Report</h1>
<p class="meta">Generated: {now} | Platform v0.1.0</p>

<h2>Recording Information</h2>
<table>
    <tr><td>File</td><td>{meta.source_path}</td></tr>
    <tr><td>Format</td><td>{meta.source_format.value}</td></tr>
    <tr><td>Sample Rate</td><td>{meta.sample_rate_hz:,.0f} Hz</td></tr>
    <tr><td>Center Frequency</td><td>{meta.center_frequency_hz:,.0f} Hz</td></tr>
    <tr><td>Duration</td><td>{meta.duration_seconds:.3f} s</td></tr>
    <tr><td>Samples</td><td>{meta.sample_count:,}</td></tr>
    <tr><td>Data Type</td><td>{meta.sample_datatype.value}</td></tr>
    <tr><td>Channels</td><td>{meta.num_channels}</td></tr>
    <tr><td>IQ Order</td><td>{meta.iq_order.value}</td></tr>
</table>

{result_html}

{plots_html}

</body>
</html>"""

    Path(path).write_text(html, encoding="utf-8")
