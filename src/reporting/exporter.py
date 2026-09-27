"""Export engine for analysis results.

Supports JSON export (metadata, analysis, or the full pipeline result plus
the automatic decoding chain) and a self-contained HTML report.
"""

from __future__ import annotations

import base64
import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.core.models import AnalysisResult, RecordingMetadata

if TYPE_CHECKING:
    from src.decoding.auto_decode import AutoDecodeResult
    from src.dsp.pipeline import PipelineResult

SCHEMA_VERSION = "1.2.0"
TEXT_PREVIEW_CHARS = 20_000
FRAME_PREVIEW = 5


def _app_version() -> str:
    from src import __version__

    return __version__


def export_metadata_json(meta: RecordingMetadata, path: str | Path) -> None:
    """Write recording metadata as pretty-printed JSON."""
    data = meta.model_dump(mode="json")
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def export_analysis_json(result: AnalysisResult, path: str | Path) -> None:
    """Write analysis result as pretty-printed JSON."""
    data = result.model_dump(mode="json")
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _bits_hex(bits: np.ndarray, max_bits: int) -> str:
    b = np.asarray(bits, dtype=np.uint8)[:max_bits]
    n = (len(b) // 8) * 8
    return np.packbits(b[:n]).tobytes().hex() if n else ""


def decode_chain_to_dict(chain: AutoDecodeResult, max_bits: int = 1_000_000) -> dict[str, Any]:
    """Serialise the automatic decoding chain (mapping → interleaver → FEC → framing → text)."""
    f = chain.framing
    final = chain.final_bits
    out: dict[str, Any] = {
        "steps": [{"step": s.name, "status": s.status, "detail": s.detail} for s in chain.steps],
        "mapping": ({"name": chain.mapping.name, "z_score": chain.mapping.z_score,
                     "identity_z": chain.mapping.identity_z} if chain.mapping else None),
        "interleaver": ({"type": chain.interleaver.kind.value,
                         "summary": chain.interleaver.summary()} if chain.interleaver else None),
        "fec": ({"type": chain.fec.fec_type.value, "summary": chain.fec.summary()}
                if chain.fec else None),
        "decode_ok": chain.decode_ok,
        "final_stage": chain.final_stage,
        "final_bits": int(len(final)),
        "final_bits_hex": _bits_hex(final, max_bits),
        "final_bits_truncated": len(final) > max_bits,
        "framing": {"found": False},
        "text": None,
        "elapsed_s": chain.elapsed_s,
        "cancelled": chain.cancelled,
    }
    if f.found:
        out["framing"] = {
            "found": True, "method": f.method, "stage": chain.framing_stage,
            "sync": f.sync_hex, "sync_name": f.sync_name, "frame_bits": f.period,
            "regular": f.regular, "frames": len(f.frames), "header_bits": f.header_bits,
            "false_alarm": f.false_alarm, "notes": list(f.notes),
            "fields": [{"kind": h.kind, "start": h.start, "length": h.length,
                        "description": h.description} for h in f.fields],
        }
    if chain.text is not None:
        t = chain.text
        out["text"] = {
            "format": t.describe(), "characters": t.characters, "content": t.text,
            "messages": [{"header": m.header, "station": m.station, "subject": m.subject,
                          "number": m.number, "text": m.text}
                         for m in getattr(t, "messages", [])],
        }
    return out


def pipeline_result_to_dict(result: PipelineResult, max_bits: int = 1_000_000,
                            decode: AutoDecodeResult | None = None) -> dict[str, Any]:
    """Serialise a full pipeline result (metadata, analysis, regions, bits) and,
    when given, the automatic decoding chain run on it."""
    d = result.demod
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "app_version": _app_version(),
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
        "classifier_source": result.classifier_source,
        "classification": None,
        "model_prediction": None,
        "bursts": None,
        "demodulation": None,
        "apt": None,
        "decoding": decode_chain_to_dict(decode, max_bits) if decode is not None else None,
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
    mp = result.model_prediction
    if mp is not None:
        out["model_prediction"] = {
            "modulation": mp.modulation.value, "probability": mp.probability,
            "ranked": [{"modulation": m.value, "probability": p} for m, p in mp.ranked[:5]],
        }
    bd = result.burst_detection
    if bd is not None:
        out["bursts"] = {
            "continuous": bd.continuous, "duty_cycle": bd.duty_cycle,
            "noise_floor_db": bd.noise_floor_db,
            "primary": result.primary_burst,
            "analysed": [{
                "index": b.index, "modulation": b.modulation.value,
                "confidence": b.result.analysis.modulation_confidence,
                "symbol_rate_hz": b.result.analysis.symbol_rate_hz,
                "start_time_sec": b.region.start_time_sec, "end_time_sec": b.region.end_time_sec,
                "center_frequency_hz": b.region.center_frequency_hz,
                "bandwidth_hz": b.region.bandwidth_hz, "snr_db": b.region.snr_db,
            } for b in result.bursts],
        }
    if d is not None:
        bits = d.bits[:max_bits]
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
            "timing_lock": d.timing_lock,
            "carrier_lock": d.carrier_lock,
            "fsk_levels_hz": list(d.fsk_levels_hz),
            "warnings": list(d.warnings),
            "bits_hex": _bits_hex(bits, max_bits),
            "bits_truncated": len(d.bits) > max_bits,
            "carrier_hz": d.carrier_hz,
            "audio_seconds": (len(d.audio) / d.audio_rate_hz
                              if d.audio is not None and d.audio_rate_hz > 0 else 0.0),
            "audio_rate_hz": d.audio_rate_hz,
        }
    if result.apt is not None:
        a = result.apt
        out["apt"] = {"lines": a.lines, "sync_quality": a.sync_quality,
                      "line_rate_hz": a.line_rate_hz, "fm_deviation_hz": a.fm_deviation_hz,
                      "description": a.describe()}
    return out


def export_audio_wav(result: PipelineResult, path: str | Path) -> float:
    """Write the demodulated audio of an analog signal as 16-bit PCM WAV.

    Returns the duration in seconds.  Raises ``ValueError`` if the result
    carries no audio (digital modulation or no demodulation).
    """
    from scipy.io import wavfile

    d = result.demod
    if d is None or d.audio is None or d.audio_rate_hz <= 0:
        raise ValueError("No demodulated audio in this result.")
    audio = np.asarray(d.audio, dtype=np.float64)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    pcm = (audio / peak * 0.9 * 32767).astype(np.int16) if peak > 0 else \
        np.zeros(len(audio), dtype=np.int16)
    wavfile.write(str(path), int(round(d.audio_rate_hz)), pcm)
    return len(audio) / d.audio_rate_hz


def export_pipeline_json(result: PipelineResult, path: str | Path,
                         decode: AutoDecodeResult | None = None) -> None:
    """Write a complete pipeline result (and decoding chain, if given) as JSON."""
    data = pipeline_result_to_dict(result, decode=decode)
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _e(value: Any) -> str:
    return html.escape(str(value))


def _pct(x: float) -> str:
    """Confidence as a percentage; some estimators return unbounded scores."""
    return f"{min(max(float(x), 0.0), 1.0):.0%}"


def _hz(x: float, signed: bool = False) -> str:
    x = float(x)
    if abs(x) >= 1e6:
        s = f"{x / 1e6:{'+' if signed else ''},.6f} MHz"
    elif abs(x) >= 1e3:
        s = f"{x / 1e3:{'+' if signed else ''},.3f} kHz"
    else:
        s = f"{x:{'+' if signed else ''},.1f} Hz"
    return s


def _table(rows: list[tuple[Any, ...]], head: tuple[str, ...] | None = None) -> str:
    """Rows of already-escaped cells."""
    h = ("<tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr>") if head else ""
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table>{h}{body}</table>"


def _list(items: list[str], cls: str = "") -> str:
    if not items:
        return ""
    c = f' class="{cls}"' if cls else ""
    return f"<ul{c}>" + "".join(f"<li>{_e(i)}</li>" for i in items) + "</ul>"


def _status(ok: bool, good: str = "OK", bad: str = "Problem") -> str:
    return f'<span class="{"ok" if ok else "bad"}">{good if ok else bad}</span>'


def _constellation_svg(symbols: np.ndarray, size: int = 320, max_points: int = 3000) -> str:
    s = np.asarray(symbols).reshape(-1)
    s = s[np.isfinite(s)]
    if len(s) == 0 or not np.iscomplexobj(s):
        return ""
    if len(s) > max_points:
        s = s[np.linspace(0, len(s) - 1, max_points).astype(int)]
    lim = float(np.percentile(np.abs(np.concatenate([s.real, s.imag])), 99.5)) * 1.15 or 1.0
    half = size / 2
    xs = half + s.real / lim * (half - 8)
    ys = half - s.imag / lim * (half - 8)
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.3"/>'
                   for x, y in zip(xs, ys, strict=True))
    return (f'<svg class="plot" width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
            f'role="img" aria-label="Constellation diagram">'
            f'<rect width="{size}" height="{size}" fill="#0b1020"/>'
            f'<line x1="0" y1="{half}" x2="{size}" y2="{half}" stroke="#1e293b"/>'
            f'<line x1="{half}" y1="0" x2="{half}" y2="{size}" stroke="#1e293b"/>'
            f'<g fill="#38bdf8" fill-opacity="0.55">{dots}</g></svg>')


def _section(title: str, body: str, anchor: str) -> str:
    return f'<section id="{anchor}"><h2>{_e(title)}</h2>{body}</section>' if body else ""


def _recording_html(meta: RecordingMetadata, sha256: str) -> str:
    rows = [
        ("File", _e(meta.source_path or "—")),
        ("Format", _e(meta.source_format.value)),
        ("SHA-256", f"<code>{_e(sha256 or meta.sha256 or 'not computed')}</code>"),
        ("Sample rate", f"{meta.sample_rate_hz:,.0f} Hz "
                        f'<span class="meta">({_e(meta.sample_rate_status.value)})</span>'),
        ("Centre frequency",
         f"{_e(_hz(meta.center_frequency_hz))} "
         f'<span class="meta">({_e(meta.center_frequency_status.value)})</span>'),
        ("Duration", f"{meta.duration_seconds:.3f} s"),
        ("Samples", f"{meta.sample_count:,}"),
        ("Data type", _e(meta.sample_datatype.value)),
        ("Channels", str(meta.num_channels)),
        ("IQ order", _e(meta.iq_order.value)),
    ]
    if meta.byte_offset:
        rows.append(("Byte offset", f"{meta.byte_offset:,}"))
    if meta.source_path.lower().endswith(".wav"):
        rows.append(("WAV interpretation", _e(meta.wav_interpretation.value)
                     + (" (I/Q swapped)" if meta.wav_swap_iq else "")))
    for label, v in (("Sensor", meta.sensor), ("Location", meta.location),
                     ("Time origin", meta.time_origin), ("Notes", meta.notes)):
        if v:
            rows.append((label, _e(v)))
    return _table(rows)


def _validation_html(pipeline: PipelineResult) -> str:
    v = pipeline.validation
    rows = [
        ("Overall", _status(v.is_valid, "Valid", "Invalid")),
        ("File size", f"{v.file_size_bytes:,} bytes" if v.file_size_bytes else "—"),
        ("Header", _status(v.header_valid, "Valid", "Invalid")),
        ("Truncated", _status(not v.truncated, "No", "Yes")),
        ("Sample aligned", _status(v.sample_aligned, "Yes", "No")),
    ]
    if v.expected_sample_count or v.actual_sample_count:
        rows.append(("Samples (expected / read)",
                     f"{v.expected_sample_count:,} / {v.actual_sample_count:,}"))
    rows += [
        ("NaN / Inf samples", _status(v.nan_count + v.inf_count == 0,
                                      f"{v.nan_count} / {v.inf_count}",
                                      f"{v.nan_count} / {v.inf_count}")),
        ("Clipping", _status(v.clipping_percentage < 1.0, f"{v.clipping_percentage:.2f} %",
                             f"{v.clipping_percentage:.2f} %")),
        ("DC offset (I / Q)", f"{v.dc_offset_i:+.4f} / {v.dc_offset_q:+.4f}"),
        ("IQ imbalance", f"{v.iq_imbalance_db:.2f} dB"),
    ]
    out = _table(rows)
    if v.errors:
        out += "<h3>Errors</h3>" + _list(v.errors, "bad")
    if v.warnings:
        out += "<h3>Warnings</h3>" + _list(v.warnings)
    return out


def _detection_html(pipeline: PipelineResult) -> str:
    out = ""
    bd = pipeline.burst_detection
    if bd is not None:
        kind = ("continuous signal" if bd.continuous
                else f"{len(bd.bursts)} burst(s)" if bd.bursts else "no bursts found")
        out += _table([("Activity", _e(kind)), ("Duty cycle", f"{bd.duty_cycle:.0%}"),
                       ("Noise floor", f"{bd.noise_floor_db:.1f} dB")])
    if pipeline.bursts:
        rows = []
        for b in pipeline.bursts:
            r, a = b.region, b.result.analysis
            primary = " ★" if pipeline.primary_burst is not None and \
                pipeline.bursts[pipeline.primary_burst] is b else ""
            rows.append((f"{b.index + 1}{primary}",
                         f"{r.start_time_sec:.4f} – {r.end_time_sec:.4f} s",
                         _e(_hz(r.center_frequency_hz)), _e(_hz(r.bandwidth_hz)),
                         f"{r.snr_db:.1f} dB", _e(a.modulation.value),
                         _pct(a.modulation_confidence),
                         f"{a.symbol_rate_hz:,.1f}" if a.symbol_rate_hz > 0 else "—"))
        out += _table(rows, ("#", "Time", "Centre", "Bandwidth", "SNR", "Modulation",
                             "Confidence", "Symbol rate (Bd)"))
        out += '<p class="meta">★ the burst the rest of this report describes.</p>'
    elif pipeline.regions:
        rows = [(str(i + 1), f"{r.start_time_sec:.4f} – {r.end_time_sec:.4f} s",
                 _e(_hz(r.center_frequency_hz)), _e(_hz(r.bandwidth_hz)), f"{r.snr_db:.1f} dB",
                 _e(r.detection_method.value))
                for i, r in enumerate(pipeline.regions)]
        out += _table(rows, ("#", "Time", "Centre", "Bandwidth", "SNR", "Method"))
    return out


def _classification_html(result: AnalysisResult, pipeline: PipelineResult | None) -> str:
    rows = [
        ("Modulation", f"<b>{_e(result.modulation.value)}</b>"),
        ("Confidence", _pct(result.modulation_confidence)),
    ]
    if pipeline is not None:
        rows.append(("Decided by", _e(pipeline.classifier_source)))
    rows.append(("Overall confidence", _e(result.overall_confidence.value)))
    out = _table(rows)
    if result.modulation_candidates:
        out += "<h3>Ranked candidates</h3>" + _table(
            [(_e(c["modulation"]), _pct(c["probability"]))
             for c in result.modulation_candidates[:6]], ("Modulation", "Probability"))
    if pipeline is not None:
        c, mp = pipeline.classification, pipeline.model_prediction
        if c is not None and mp is not None:
            out += "<h3>Rules vs learned model</h3>" + _table([
                ("Rule-based", _e(c.modulation.value), _pct(c.confidence)),
                ("Learned model", _e(mp.modulation.value), _pct(mp.probability)),
            ], ("Classifier", "Modulation", "Confidence"))
        if c is not None and c.evidence:
            out += "<h3>Evidence</h3>" + _list(c.evidence)
        if c is not None and c.features:
            feats = "".join(f"<tr><td>{_e(k)}</td><td>{float(v):.4g}</td></tr>"
                            for k, v in sorted(c.features.items()))
            out += (f"<details><summary>Signal features ({len(c.features)})</summary>"
                    f"<table>{feats}</table></details>")
    return out


def _is_analog(pipeline: PipelineResult | None) -> bool:
    return pipeline is not None and pipeline.demod is not None and pipeline.demod.audio is not None


def _parameters_html(result: AnalysisResult, pipeline: PipelineResult | None,
                     chain: AutoDecodeResult | None) -> str:
    analog = _is_analog(pipeline)
    rows: list[tuple[Any, ...]] = []
    if not analog:
        rows.append(("Symbol rate",
                     f"{result.symbol_rate_hz:,.1f} Bd" if result.symbol_rate_hz > 0 else "—",
                     _pct(result.symbol_rate_confidence) if result.symbol_rate_hz > 0 else "—"))
    if chain is not None:
        if chain.fec is not None:
            rows.append(("FEC", _e(chain.fec.fec_type.value), ""))
        if chain.interleaver is not None:
            rows.append(("Interleaver", _e(chain.interleaver.kind.value), ""))
    if pipeline is not None:
        rows += [
            ("Carrier offset", _e(_hz(pipeline.frequency_offset_hz, signed=True)), ""),
            ("SNR (full band)", f"{pipeline.snr_db:.1f} dB", ""),
            ("SNR (in-band)", f"{pipeline.snr_inband_db:.1f} dB", ""),
            ("Occupied bandwidth", _e(_hz(pipeline.occupied_bandwidth_hz)), ""),
        ]
    out = _table(rows, ("Parameter", "Value", "Confidence"))
    if result.parameters:
        prow = []
        for p in result.parameters:
            val = f"{p.value:,.1f}" if isinstance(p.value, float) else _e(p.value)
            alts = ", ".join(
                f"{a.get('value', 0):,.1f}" if isinstance(a.get("value"), float)
                else _e(a.get("value")) for a in p.alternatives[:3])
            prow.append((_e(p.parameter), val, _e(p.status.value), _pct(p.confidence),
                         _e("; ".join(p.evidence)), alts or "—"))
        out += "<h3>Estimate provenance</h3>" + _table(
            prow, ("Parameter", "Value", "Status", "Confidence", "Evidence", "Alternatives"))
    if pipeline is not None and not analog and len(pipeline.symbol_rate_candidates) > 1:
        out += "<h3>Symbol-rate candidates</h3>" + _table(
            [(f"{r:,.1f} Bd", _pct(c)) for r, c in pipeline.symbol_rate_candidates[:5]],
            ("Rate", "Confidence"))
    return out


def _demod_html(pipeline: PipelineResult) -> str:
    d = pipeline.demod
    if d is None:
        return '<p class="meta">Nothing was demodulated.</p>'
    rows = [("Modulation", _e(d.modulation.value))]
    if d.audio is not None:
        secs = len(d.audio) / d.audio_rate_hz if d.audio_rate_hz > 0 else 0.0
        rows += [("Output", "Analog audio"),
                 ("Audio", f"{secs:.2f} s at {d.audio_rate_hz:,.0f} Hz")]
        if d.carrier_hz is not None:
            rows.append(("Carrier used", _e(_hz(d.carrier_hz, signed=True))))
    else:
        rows += [
            ("Symbol rate", f"{d.symbol_rate_hz:,.2f} Bd"),
            ("Samples per symbol", f"{d.samples_per_symbol:.2f}"),
            ("Symbols", f"{d.num_symbols:,}"),
            ("Bits", f"{d.num_bits:,} ({d.bits_per_symbol} per symbol)"),
            ("EVM", f"{d.evm_percent:.1f} %"),
            ("Coarse / residual CFO", f"{d.coarse_cfo_hz:+,.1f} Hz / {d.residual_cfo_hz:+,.1f} Hz"),
            ("Timing / carrier lock error", f"{d.timing_lock:.3f} / {d.carrier_lock:.3f}"),
        ]
        if d.fsk_levels_hz:
            rows.append(("FSK tones", _e(", ".join(_hz(f, signed=True) for f in d.fsk_levels_hz))))
    out = _table(rows)
    if d.warnings:
        out += "<h3>Demodulator notes</h3>" + _list(d.warnings)
    if d.audio is None:
        # FSK symbols are tone decisions, not points in the I/Q plane
        svg = "" if d.fsk_levels_hz or "FSK" in d.modulation.value else \
            _constellation_svg(d.symbols)
        if svg:
            out += f"<h3>Constellation</h3>{svg}"
        if len(d.bits) >= 8:
            n = min(len(d.bits), 2048) // 8 * 8
            preview = np.packbits(np.asarray(d.bits[:n], dtype=np.uint8)).tobytes().hex(" ")
            out += (f"<h3>Demodulated bits (first {n:,} of {d.num_bits:,}, hex)</h3>"
                    f"<pre>{preview.upper()}</pre>")
    if pipeline.apt is not None:
        from src.decoding.apt import png_bytes

        a = pipeline.apt
        img = base64.b64encode(png_bytes(a.image)).decode("ascii")
        out += (f"<h3>NOAA APT image</h3><p>{_e(a.describe())}</p>"
                f'<img alt="Decoded APT image, channel A | channel B" '
                f'src="data:image/png;base64,{img}">')
    return out


def _decoding_html(chain: AutoDecodeResult | None, pipeline: PipelineResult) -> str:
    d = pipeline.demod
    if chain is None:
        if d is not None and d.audio is None and d.num_bits:
            return ('<p class="meta">The automatic decoding chain was not run for this '
                    "result. Use Auto-Analyse (or <code>python -m src.auto_analyse</code>) "
                    "to include bit mapping, interleaver, FEC, framing and text.</p>")
        return ""
    icon = {"found": "ok", "failed": "bad"}
    out = _table([(f'<span class="{icon.get(s.status, "meta")}">{_e(s.icon)}</span>',
                   _e(s.name), _e(s.status), _e(s.detail)) for s in chain.steps],
                 ("", "Step", "Status", "Detail"))
    if chain.cancelled:
        out += '<p class="bad">Decoding was cancelled; the results below are partial.</p>'
    summary = [("Elapsed", f"{chain.elapsed_s:.1f} s")]
    if chain.mapping is not None:
        z = f'<span class="meta">(z={chain.mapping.z_score:.1f})</span>'
        summary.append(("Bit mapping", f"{_e(chain.mapping.name)} {z}"))
    if chain.decode_ok is not None:
        summary.append(("FEC decode", _status(chain.decode_ok, "Consistent", "Errors remain")))
    final = chain.final_bits
    summary.append(("Output", f"{len(final):,} bits after stage "
                              f"<code>{_e(chain.final_stage or 'raw')}</code>"))
    out += _table(summary)
    if chain.interleaver is not None:
        out += f"<h3>Interleaver</h3><pre>{_e(chain.interleaver.summary())}</pre>"
    if chain.fec is not None:
        out += f"<h3>Forward error correction</h3><pre>{_e(chain.fec.summary())}</pre>"

    f = chain.framing
    if f.found:
        out += "<h3>Framing</h3>" + _table([
            ("Sync word", f"<code>{_e(f.sync_hex)}</code>"
                          + (f" ({_e(f.sync_name)})" if f.sync_name else "")),
            ("Method", _e(f.method)),
            ("Found in stage", f"<code>{_e(chain.framing_stage)}</code>"),
            ("Frame length", f"{f.period:,} bits" if f.period else "variable"),
            ("Regular spacing", "Yes" if f.regular else "No"),
            ("Frames", f"{len(f.frames):,}"),
            ("Header", f"{f.header_bits} bits"),
            ("False-alarm probability", f"{f.false_alarm:.2e}"),
        ])
        if f.fields:
            out += "<h4>Header fields</h4>" + _table(
                [(_e(h.kind), f"{h.start}–{h.start + h.length - 1}", str(h.length),
                  _e(h.description)) for h in f.fields],
                ("Kind", "Bits", "Length", "Description"))
        if f.frames:
            from src.decoding.correlation import bits_to_hex

            out += f"<h4>First {min(FRAME_PREVIEW, len(f.frames))} frames</h4>" + _table(
                [(f"{fr.start:,}", f"<code>{_e(bits_to_hex(fr.header))}</code>",
                  f"<code>{_e(bits_to_hex(fr.payload[:256]))}</code>"
                  + (" …" if len(fr.payload) > 256 else ""))
                 for fr in f.frames[:FRAME_PREVIEW]],
                ("Start bit", "Header", "Payload"))
        if f.notes:
            out += _list(f.notes, "meta")

    t = chain.text
    if t is not None:
        out += f"<h3>Decoded text</h3><p>{_e(t.describe())}</p>"
        msgs = getattr(t, "messages", [])
        if msgs:
            out += _table([(_e(m.header), _e(m.station), _e(m.subject), _e(m.number))
                           for m in msgs], ("Header", "Station", "Subject", "No."))
        body = t.text[:TEXT_PREVIEW_CHARS]
        more = (f'<p class="meta">Showing the first {TEXT_PREVIEW_CHARS:,} of '
                f"{len(t.text):,} characters.</p>") if len(t.text) > TEXT_PREVIEW_CHARS else ""
        out += f"<pre>{_e(body)}</pre>{more}"
    elif len(final) >= 8 and chain.final_stage:
        n = min(len(final), 2048) // 8 * 8
        preview = np.packbits(np.asarray(final[:n], dtype=np.uint8)).tobytes().hex(" ")
        out += (f"<h3>Decoded bits (first {n:,} of {len(final):,}, hex)</h3>"
                f"<pre>{preview.upper()}</pre>")
    return out


def _summary_html(meta: RecordingMetadata, result: AnalysisResult | None,
                  pipeline: PipelineResult | None, chain: AutoDecodeResult | None) -> str:
    if result is None:
        return ""
    cards = [("Modulation", result.modulation.value, _pct(result.modulation_confidence))]
    if result.symbol_rate_hz > 0 and not _is_analog(pipeline):
        cards.append(("Symbol rate", f"{result.symbol_rate_hz:,.1f} Bd",
                      _pct(result.symbol_rate_confidence)))
    if pipeline is not None:
        cards.append(("SNR", f"{pipeline.snr_db:.1f} dB",
                      f"in-band {pipeline.snr_inband_db:.1f} dB"))
        cards.append(("Carrier offset", _hz(pipeline.frequency_offset_hz, signed=True),
                      f"BW {_hz(pipeline.occupied_bandwidth_hz)}"))
        d = pipeline.demod
        if pipeline.apt is not None:
            cards.append(("Decoded", "NOAA APT image", f"{pipeline.apt.lines} lines"))
        elif d is not None and d.audio is not None:
            cards.append(("Decoded", "Analog audio", f"{len(d.audio) / d.audio_rate_hz:.1f} s"
                          if d.audio_rate_hz > 0 else ""))
        elif chain is not None:
            if chain.text is not None:
                cards.append(("Decoded", "Text", f"{chain.text.characters:,} characters"))
            elif chain.framing.found:
                cards.append(("Decoded", f"{len(chain.framing.frames):,} frames",
                              f"sync {chain.framing.sync_hex}"))
            if chain.fec is not None:
                cards.append(("FEC", chain.fec.fec_type.value, ""))
        elif d is not None:
            cards.append(("Demodulated", f"{d.num_bits:,} bits", f"EVM {d.evm_percent:.1f} %"))
    return '<div class="cards">' + "".join(
        f'<div class="card"><div class="k">{_e(k)}</div><div class="v">{_e(v)}</div>'
        f'<div class="s">{_e(s)}</div></div>' for k, v, s in cards) + "</div>"


_CSS = """
body { font-family: -apple-system, 'Inter', 'Segoe UI', sans-serif; background: #0f1420;
       color: #e2e8f0; max-width: 1000px; margin: 0 auto; padding: 24px; line-height: 1.45; }
h1 { color: #4ea8f5; border-bottom: 2px solid #1e293b; padding-bottom: 8px; margin-bottom: 4px; }
h2 { color: #7c9cfc; margin-top: 32px; border-bottom: 1px solid #1e293b; padding-bottom: 4px; }
h3 { color: #cbd5e1; margin: 18px 0 6px; font-size: 1.02em; }
h4 { color: #94a3b8; margin: 12px 0 4px; }
table { width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 0.93em; }
th { text-align: left; color: #94a3b8; font-weight: 600; }
td, th { padding: 6px 10px; border: 1px solid #1e293b; vertical-align: top; }
td:first-child { white-space: nowrap; }
tr:nth-child(even) td { background: #151c2c; }
code, pre { font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 0.88em; }
pre { white-space: pre-wrap; word-break: break-all; background: #0b1020; color: #cbd5e1;
      padding: 10px; border: 1px solid #1e293b; border-radius: 6px; max-height: 480px;
      overflow: auto; }
.meta { color: #94a3b8; font-size: 0.9em; }
.ok { color: #34d399; } .bad { color: #f87171; }
.cards { display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 0; }
.card { flex: 1 1 150px; background: #151c2c; border: 1px solid #1e293b; border-radius: 8px;
        padding: 10px 12px; }
.card .k { color: #94a3b8; font-size: 0.8em; text-transform: uppercase; letter-spacing: .05em; }
.card .v { font-size: 1.2em; font-weight: 600; margin-top: 2px; }
.card .s { color: #94a3b8; font-size: 0.85em; }
nav { font-size: 0.9em; margin: 8px 0 0; } nav a { color: #7c9cfc; margin-right: 12px; }
img { image-rendering: pixelated; }
img, .plot { max-width: 100%; border: 1px solid #1e293b; border-radius: 6px; margin: 6px 0; }
details summary { cursor: pointer; color: #94a3b8; margin: 8px 0; }
@media print { body { background: #fff; color: #000; } td, th, pre { border-color: #ccc; }
  tr:nth-child(even) td, pre, .card { background: #f5f5f5; color: #000; } nav { display: none; } }
"""


def export_html_report(
    meta: RecordingMetadata,
    result: AnalysisResult | None = None,
    path: str | Path = "report.html",
    plot_paths: list[str] | None = None,
    pipeline: PipelineResult | None = None,
    decode: AutoDecodeResult | None = None,
) -> None:
    """Generate a self-contained HTML analysis report.

    Pass *pipeline* to include file integrity, detected bursts, classifier
    evidence, estimate provenance and demodulation; pass *decode* (the
    automatic decoding chain run on it) to add interleaver, FEC, framing
    and decoded text.
    """
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    if pipeline is not None and result is None:
        result = pipeline.analysis
    sha = pipeline.validation.sha256 if pipeline is not None else ""

    sections = [
        ("Recording", _recording_html(meta, sha), "recording"),
    ]
    if pipeline is not None:
        sections.append(("File integrity", _validation_html(pipeline), "integrity"))
        sections.append(("Signal detection", _detection_html(pipeline), "detection"))
    if result is not None:
        sections.append(("Classification", _classification_html(result, pipeline),
                         "classification"))
        sections.append(("Signal parameters", _parameters_html(result, pipeline, decode),
                         "parameters"))
    if pipeline is not None:
        sections.append(("Demodulation", _demod_html(pipeline), "demodulation"))
        sections.append(("Decoding", _decoding_html(decode, pipeline), "decoding"))
    notes = list(result.warnings) if result is not None else []
    if pipeline is not None:
        notes += [f"Stage '{k}' failed: {v}" for k, v in pipeline.stage_errors.items()]
    sections.append(("Notes and warnings", _list(notes), "notes"))
    if plot_paths:
        sections.append(("Plots", "".join(f'<img alt="plot" src="{_e(p)}">' for p in plot_paths),
                         "plots"))

    rendered = [(t, b, a) for t, b, a in sections if b]
    nav = "".join(f'<a href="#{a}">{_e(t)}</a>' for t, _, a in rendered)
    body = "\n".join(_section(t, b, a) for t, b, a in rendered)
    footer = ""
    if pipeline is not None:
        footer = (f"Processed {pipeline.samples_processed:,} samples in "
                  f"{pipeline.processing_time_ms:,.0f} ms")

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sigma Signal Analysis Report — {_e(Path(meta.source_path).name or "recording")}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Sigma Signal Analysis Report</h1>
<p class="meta">{_e(Path(meta.source_path).name or "Recording")} · Generated {now} ·
Sigma v{_e(_app_version())}</p>
<nav>{nav}</nav>
{_summary_html(meta, result, pipeline, decode)}
{body}
<p class="meta" style="margin-top:32px">{footer}</p>
</body>
</html>"""

    Path(path).write_text(doc, encoding="utf-8")
