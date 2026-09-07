"""Export engine for analysis results.

Supports JSON metadata export, HTML reports, and PNG plot snapshots.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.core.models import AnalysisResult, RecordingMetadata


def export_metadata_json(meta: RecordingMetadata, path: str | Path) -> None:
    """Write recording metadata as pretty-printed JSON."""
    data = meta.model_dump(mode="json")
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def export_analysis_json(result: AnalysisResult, path: str | Path) -> None:
    """Write analysis result as pretty-printed JSON."""
    data = result.model_dump(mode="json")
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def export_html_report(
    meta: RecordingMetadata,
    result: AnalysisResult | None = None,
    path: str | Path = "report.html",
    plot_paths: list[str] | None = None,
) -> None:
    """Generate a basic HTML analysis report."""
    now = datetime.now(UTC).isoformat()

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
        if result.warnings:
            result_html += "<h3>Warnings</h3><ul>"
            for w in result.warnings:
                result_html += f"<li>{w}</li>"
            result_html += "</ul>"

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
