"""Command-line one-click analysis: recording → signal parameters → decoded frames.

::

    python -m src.auto_analyse capture.sigmf-meta
    python -m src.auto_analyse capture.wav
    python -m src.auto_analyse capture.iq --fs 250000 --dtype cf32_le
    python -m src.auto_analyse capture.iq --fs 250000 --save-bits out.bin --json report.json

Runs the analysis pipeline (sample-rate check, burst detection,
modulation classification, symbol rate, carrier, SNR, demodulation) and
then the automatic decoding chain of :mod:`src.decoding.auto_decode`
(bit mapping → interleaver → FEC → framing), and prints one report.

WAV files are read the way :meth:`WavReader.suggest_interpretation`
proposes unless ``--wav-mode`` is given.  Raw ``.iq`` files carry no
sample rate: pass ``--fs``; without it the most plausible rate from
:func:`~src.dsp.rate_inference.infer_sample_rate_candidates` is used and
the alternatives are listed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.core.enums import SampleDatatype, WavInterpretation
from src.core.models import RecordingMetadata

RAW_SUFFIXES = {".iq", ".raw", ".bin", ".cfile", ".dat", ".cf32", ".cs16", ".cs8", ".cu8"}


def load_recording(path: Path, fs: float | None = None, dtype: str = "cf32_le",
                   wav_mode: str = "auto", max_samples: int = 10_000_000,
                   log=print) -> tuple[np.ndarray, RecordingMetadata]:
    """Read samples and metadata from a WAV, SigMF or raw IQ file."""
    from src.ingestion.raw_iq_reader import RawIQReader
    from src.ingestion.sigmf_reader import SigMFReader
    from src.ingestion.wav_reader import WavReader

    name = path.name.lower()
    if name.endswith((".sigmf-meta", ".sigmf-data", ".sigmf")):
        reader = SigMFReader(path)
    elif name.endswith(".wav"):
        if wav_mode == "auto":
            sug = WavReader(path).suggest_interpretation()
            interp = sug.interpretation
            log(f"WAV read as {interp.value} ({sug.confidence:.0%}: {sug.reason})")
        else:
            interp = WavInterpretation(wav_mode)
        reader = WavReader(path, interpretation=interp)
    else:
        dt = SampleDatatype(dtype)
        rate = fs
        if not rate:
            from src.dsp.rate_inference import infer_sample_rate_candidates

            probe = RawIQReader(path, datatype=dt, sample_rate_hz=1.0)
            inf = infer_sample_rate_candidates(probe.read_samples(0, min(500_000,
                                                                          probe.total_samples())))
            if not inf.candidates:
                raise SystemExit("Raw file without --fs and no sample rate could be inferred: "
                                 + inf.note)
            rate = inf.candidates[0].sample_rate_hz
            log(f"No --fs given; assuming {rate:,.0f} Hz. Other candidates: "
                + ", ".join(f"{c.sample_rate_hz:,.0f}" for c in inf.candidates[1:6]))
        reader = RawIQReader(path, datatype=dt, sample_rate_hz=rate)
    meta = reader.read_metadata()
    if fs:
        meta.sample_rate_hz = fs
    samples = reader.read_samples(0, min(max_samples, reader.total_samples()))
    return samples, meta


def _report(result, chain) -> dict:
    a = result.analysis
    out: dict = {
        "modulation": a.modulation.value,
        "modulation_confidence": round(float(a.modulation_confidence), 3),
        "classifier": result.classifier_source,
        "sample_rate_hz": result.metadata.sample_rate_hz,
        "symbol_rate_hz": round(float(a.symbol_rate_hz), 2),
        "carrier_offset_hz": round(float(result.frequency_offset_hz), 2),
        "snr_db": round(float(result.snr_db), 2),
        "occupied_bandwidth_hz": round(float(result.occupied_bandwidth_hz), 1),
        "bursts": len(result.bursts),
    }
    d = result.demod
    if d is not None:
        out["demod"] = {"symbols": int(d.num_symbols), "bits": int(d.num_bits),
                        "evm_percent": round(float(d.evm_percent), 2),
                        "audio": d.audio is not None}
    if chain is not None:
        f = chain.framing
        out["decoding"] = {
            "steps": [{"step": s.name, "status": s.status, "detail": s.detail}
                      for s in chain.steps],
            "final_stage": chain.final_stage,
            "final_bits": int(len(chain.final_bits)),
            "framing": {
                "found": f.found, "stage": chain.framing_stage, "sync": f.sync_hex,
                "sync_name": f.sync_name, "frame_bits": f.period, "regular": f.regular,
                "frames": len(f.frames), "header_bits": f.header_bits,
                "fields": [{"kind": h.kind, "start": h.start, "length": h.length,
                            "description": h.description} for h in f.fields],
            } if f.found else {"found": False},
            "elapsed_s": round(chain.elapsed_s, 2),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path)
    ap.add_argument("--fs", type=float, default=None, help="sample rate in Hz (raw IQ files)")
    ap.add_argument("--dtype", default="cf32_le",
                    choices=[d.value for d in SampleDatatype], help="raw IQ sample format")
    ap.add_argument("--wav-mode", default="auto",
                    choices=["auto"] + [w.value for w in WavInterpretation])
    ap.add_argument("--no-decode", action="store_true", help="stop after demodulation")
    ap.add_argument("--no-interleaver", action="store_true",
                    help="skip the (slow) interleaver search")
    ap.add_argument("--ldpc", action="store_true",
                    help="also try the installed standard LDPC codes (~/.sigma/ldpc)")
    ap.add_argument("--save-bits", type=Path, help="write the final bit stream (packed bytes)")
    ap.add_argument("--json", type=Path, help="write the report as JSON")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    log = (lambda *_: None) if args.quiet else print
    if not args.file.exists():
        ap.error(f"{args.file} not found")

    from src.dsp.pipeline import AnalysisPipeline

    samples, meta = load_recording(args.file, args.fs, args.dtype, args.wav_mode, log=log)
    log(f"{args.file.name}: {len(samples):,} samples at {meta.sample_rate_hz:,.0f} Hz")
    result = AnalysisPipeline().run(samples, meta)
    a = result.analysis
    log(f"Modulation   {a.modulation.value} ({a.modulation_confidence:.0%}, "
        f"{result.classifier_source})")
    if a.symbol_rate_hz > 0:
        log(f"Symbol rate  {a.symbol_rate_hz:,.1f} baud")
    log(f"Carrier      {result.frequency_offset_hz:+,.1f} Hz   SNR {result.snr_db:.1f} dB   "
        f"occupied BW {result.occupied_bandwidth_hz:,.0f} Hz")
    if result.bursts:
        log(f"Bursts       {len(result.bursts)} (report is for burst "
            f"{result.bursts[result.primary_burst].index + 1})")
    for w in a.warnings:
        log(f"  note: {w}")

    chain = None
    d = result.demod
    if d is None:
        log("Nothing demodulated.")
    elif d.audio is not None:
        log(f"Analog signal: {len(d.audio) / d.audio_rate_hz:.1f} s of audio demodulated.")
    else:
        log(f"Demodulated  {d.num_symbols:,} symbols → {d.num_bits:,} bits, "
            f"EVM {d.evm_percent:.1f}%")
        if not args.no_decode:
            from src.decoding.auto_decode import auto_decode, expected_ber_from_evm

            bits, n_bursts = AnalysisPipeline.train_bits(result)
            if n_bursts > 1:
                log(f"Decoding {n_bursts} bursts of this signal together ({len(bits):,} bits)")
            log("Decoding chain:")
            chain = auto_decode(bits, d.bits_per_symbol, d.modulation,
                                ldpc_codes=None if args.ldpc else [],
                                search_interleaver=not args.no_interleaver,
                                expected_ber=expected_ber_from_evm(d.evm_percent,
                                                                   d.bits_per_symbol))
            for line in chain.summary().splitlines():
                log(f"  {line}")
            for f in chain.framing.fields:
                log(f"    bits {f.start}–{f.start + f.length - 1}: {f.description}")
            for fr in chain.framing.frames[:3]:
                from src.decoding.correlation import bits_to_hex

                log(f"    frame @ bit {fr.start}: header {bits_to_hex(fr.header)} | "
                    f"payload {bits_to_hex(fr.payload[:128])} …")

    if args.save_bits:
        bits = chain.final_bits if chain is not None else (d.bits if d is not None else [])
        bits = np.asarray(bits, dtype=np.uint8)
        args.save_bits.write_bytes(np.packbits(bits[: len(bits) // 8 * 8]).tobytes())
        log(f"Saved {len(bits):,} bits to {args.save_bits}")
    if args.json:
        args.json.write_text(json.dumps(_report(result, chain), indent=2), encoding="utf-8")
        log(f"Report written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
