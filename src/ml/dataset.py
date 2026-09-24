"""Training data for the learned modulation classifier.

Two sources, both turned into :func:`~src.ml.features.extract_features`
vectors exactly the way the analysis pipeline would see them (same
preprocessing, and the pipeline's *estimated* symbol rate / carrier rather
than the ground truth):

* **Synthetic** – every supported modulation with randomised symbol rate
  (600–9600 Bd), 8–24 samples/symbol, carrier offset (±10 % of fs), SNR,
  pulse roll-off, modulation parameters, and optional phase noise.
* **Recordings** – a CSV manifest of labelled ``.wav`` / raw ``.iq`` /
  SigMF files, cut into fixed-length windows::

      path,modulation,sample_rate,datatype,wav_interpretation
      captures/hf_psk31.wav,BPSK,,,real
      captures/burst.cf32,QPSK,2400000,cf32_le,

  ``modulation`` is a :class:`~src.core.enums.ModulationType` value or name
  (e.g. ``8-PSK`` or ``PSK8``); relative paths are resolved against the
  manifest's folder; the optional columns override what the file header
  says.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.enums import ModulationType as M
from src.core.enums import SampleDatatype, WavInterpretation
from src.dsp import synth
from src.dsp.measurements import estimate_frequency_offset, estimate_symbol_rate_candidates
from src.dsp.preprocessing import normalize_peak, remove_dc_mean
from src.ml.features import N_FEATURES, extract_features

#: Classes the model predicts.  DBPSK looks exactly like BPSK on the air and
#: GFSK like GMSK, so they share a label; UNKNOWN covers noise and pure tones.
CLASSES: tuple[M, ...] = (
    M.BPSK, M.QPSK, M.PSK8, M.QAM16, M.QAM64, M.OQPSK, M.PI4_DQPSK,
    M.FSK2, M.FSK4, M.MSK, M.GMSK, M.OOK, M.ASK,
    M.AM, M.FM, M.SSB_USB, M.SSB_LSB, M.UNKNOWN,
)

LABEL_ALIASES: dict[M, M] = {M.DPSK: M.BPSK, M.GFSK: M.GMSK}


def canonical_label(mod: M) -> M:
    return LABEL_ALIASES.get(mod, mod)


def parse_modulation(text: str) -> M:
    t = text.strip()
    for m in M:
        if t in (m.value, m.name) or t.lower() in (m.value.lower(), m.name.lower()):
            return m
    raise ValueError(f"Unknown modulation label {text!r}")


# ---------------------------------------------------------------------------
# Synthetic examples
# ---------------------------------------------------------------------------


@dataclass
class Example:
    samples: np.ndarray
    sample_rate: float
    label: M
    snr_db: float


def _digital_params(rng: np.random.Generator) -> tuple[float, float, int, float]:
    rs = float(np.exp(rng.uniform(np.log(600), np.log(9600))))
    sps = int(rng.integers(8, 25))
    fs = rs * sps
    n = int(rng.integers(1200, 2500))
    f0 = float(rng.uniform(-0.1, 0.1) * fs)
    return rs, fs, n, f0


def synthetic_example(label: M, snr_db: float, rng: np.random.Generator) -> Example:
    """One random signal of the given class (uses and advances NumPy's
    global RNG as well, which the generators draw from)."""
    np.random.seed(int(rng.integers(1 << 31)))
    beta = float(rng.uniform(0.2, 0.5))
    if label in (M.AM, M.FM, M.SSB_USB, M.SSB_LSB, M.UNKNOWN):
        fs = float(rng.choice([16000, 24000, 32000, 48000, 96000]))
        n = int(fs * rng.uniform(0.6, 1.0))
        f0 = float(rng.uniform(-0.1, 0.1) * fs)
        if label == M.AM:
            x, _ = synth.generate_am(n, fs, float(rng.uniform(0.3, 0.95)), None, snr_db, f0)
        elif label == M.FM:
            dev = float(rng.uniform(800, min(8000, 0.12 * fs)))
            x, _ = synth.generate_fm(n, fs, dev, None, snr_db, f0)
        elif label in (M.SSB_USB, M.SSB_LSB):
            side = "usb" if label == M.SSB_USB else "lsb"
            x, _ = synth.generate_ssb(n, fs, side, None, snr_db, f0)
        elif rng.random() < 0.5:                                  # noise only
            x = ((np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)).astype(np.complex64)
        else:                                                     # unmodulated carrier
            tone = np.exp(2j * np.pi * f0 * np.arange(n) / fs)
            x = synth.add_awgn(tone.astype(np.complex64), snr_db)
        return Example(x, fs, label, snr_db)

    rs, fs, n, f0 = _digital_params(rng)
    if label == M.BPSK:
        gen = synth.generate_dbpsk if rng.random() < 0.3 else synth.generate_bpsk
        x = gen(n, rs, fs, snr_db, f0)[0]
    elif label == M.QPSK:
        x = synth.generate_mpsk(n, rs, fs, 4, snr_db, f0, beta)[0]
    elif label == M.PSK8:
        x = synth.generate_mpsk(n, rs, fs, 8, snr_db, f0, beta)[0]
    elif label == M.QAM16:
        x = synth.generate_qam(n, rs, fs, 16, snr_db, f0, beta)[0]
    elif label == M.QAM64:
        x = synth.generate_qam(n, rs, fs, 64, snr_db, f0, beta)[0]
    elif label == M.OQPSK:
        x = synth.generate_oqpsk(n, rs, fs, snr_db, f0)[0]
    elif label == M.PI4_DQPSK:
        x = synth.generate_pi4_dqpsk(n, rs, fs, snr_db, f0)[0]
    elif label == M.FSK2:
        h = float(rng.choice([rng.uniform(0.3, 0.4), rng.uniform(0.65, 3.0)]))
        x = synth.generate_mfsk(n, rs, fs, 2, h, snr_db, f0)[0]
    elif label == M.FSK4:
        x = synth.generate_mfsk(n, rs, fs, 4, float(rng.uniform(0.5, 2.0)), snr_db, f0)[0]
    elif label == M.MSK:
        x = synth.generate_msk(n, rs, fs, None, snr_db, f0)[0]
    elif label == M.GMSK:
        x = synth.generate_msk(n, rs, fs, float(rng.uniform(0.25, 0.5)), snr_db, f0)[0]
    elif label == M.OOK:
        x = synth.generate_ask(n, rs, fs, (0.0, 1.0), snr_db, f0)[0]
    elif label == M.ASK:
        lo = float(rng.uniform(0.2, 0.6))
        levels = (lo, 1.0) if rng.random() < 0.5 else tuple(np.linspace(lo, 1.0, 4))
        x = synth.generate_ask(n, rs, fs, levels, snr_db, f0)[0]
    else:
        raise ValueError(f"No generator for {label.value}")
    if rng.random() < 0.25:
        x = synth.add_phase_noise(x, float(rng.uniform(0.05, 0.5)))
    return Example(x, fs, label, snr_db)


def features_for(samples: np.ndarray, sample_rate: float) -> np.ndarray:
    """Features exactly as the pipeline computes them for classification."""
    x = normalize_peak(remove_dc_mean(np.asarray(samples, dtype=np.complex64)))
    cands = estimate_symbol_rate_candidates(x, sample_rate)
    rate, conf = cands[0] if cands else (0.0, 0.0)
    cfo = estimate_frequency_offset(x, sample_rate)
    return extract_features(x, sample_rate, rate, conf, cfo)


def _synthetic_task(args: tuple[int, int, float, float]) -> tuple[np.ndarray, int, float] | None:
    class_idx, seed, snr_lo, snr_hi = args
    rng = np.random.default_rng(seed)
    snr = float(rng.uniform(snr_lo, snr_hi))
    ex = synthetic_example(CLASSES[class_idx], snr, rng)
    try:
        return features_for(ex.samples, ex.sample_rate), class_idx, snr
    except Exception:  # noqa: BLE001 – skip (and count) rather than abort a long build
        return None


def build_synthetic(
    per_class: int,
    snr_range: tuple[float, float] = (0.0, 20.0),
    seed: int = 0,
    workers: int = 1,
    progress_cb: Callable[[float, str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(X, y, snr)`` with *per_class* examples of every class."""
    rng = np.random.default_rng(seed)
    tasks = [(ci, int(rng.integers(1 << 62)), *snr_range)
             for _ in range(per_class) for ci in range(len(CLASSES))]
    results: list[tuple[np.ndarray, int, float] | None] = []
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, r in enumerate(pool.map(_synthetic_task, tasks, chunksize=8)):
                results.append(r)
                if progress_cb and i % 50 == 0:
                    progress_cb(i / len(tasks), f"{i}/{len(tasks)} synthetic examples")
    else:
        for i, t in enumerate(tasks):
            results.append(_synthetic_task(t))
            if progress_cb and i % 50 == 0:
                progress_cb(i / len(tasks), f"{i}/{len(tasks)} synthetic examples")
    failed = sum(r is None for r in results)
    if failed:
        import warnings

        warnings.warn(f"{failed} synthetic example(s) failed feature extraction and were "
                      "skipped", RuntimeWarning, stacklevel=2)
    results = [r for r in results if r is not None]
    x = np.stack([r[0] for r in results]) if results else np.zeros((0, N_FEATURES))
    labels = np.array([r[1] for r in results])
    return x.astype(np.float32), labels, np.array([r[2] for r in results])


# ---------------------------------------------------------------------------
# Labelled recordings
# ---------------------------------------------------------------------------


@dataclass
class ManifestEntry:
    path: Path
    label: M
    sample_rate: float | None = None
    datatype: SampleDatatype | None = None
    wav_interpretation: WavInterpretation | None = None


def read_manifest(path: str | Path) -> list[ManifestEntry]:
    base = Path(path).resolve().parent
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row.get("path") or not row.get("modulation"):
                continue
            p = Path(row["path"].strip())
            sr = (row.get("sample_rate") or "").strip()
            dt = (row.get("datatype") or "").strip()
            wi = (row.get("wav_interpretation") or "").strip()
            out.append(ManifestEntry(
                p if p.is_absolute() else base / p,
                parse_modulation(row["modulation"]),
                float(sr) if sr else None,
                SampleDatatype(dt) if dt else None,
                WavInterpretation(wi) if wi else None,
            ))
    return out


def _open(entry: ManifestEntry):  # noqa: ANN202 – returns a FileReader
    from src.ingestion.raw_iq_reader import RawIQReader
    from src.ingestion.sigmf_reader import SigMFReader
    from src.ingestion.wav_reader import WavReader

    suffix = entry.path.suffix.lower()
    if suffix == ".wav":
        return WavReader(entry.path, entry.wav_interpretation or WavInterpretation.REAL)
    if suffix in (".sigmf-meta", ".sigmf-data", ".sigmf"):
        return SigMFReader(entry.path)
    if entry.sample_rate is None:
        raise ValueError(f"{entry.path}: raw IQ needs a sample_rate in the manifest")
    return RawIQReader(entry.path, datatype=entry.datatype or SampleDatatype.CF32_LE,
                       sample_rate_hz=entry.sample_rate)


def recording_windows(entry: ManifestEntry, window_seconds: float = 0.5,
                      max_windows: int = 20) -> Iterable[tuple[np.ndarray, float]]:
    """Yield ``(samples, sample_rate)`` windows of a labelled recording."""
    reader = _open(entry)
    meta = reader.read_metadata()
    fs = entry.sample_rate or meta.sample_rate_hz
    if not fs or fs <= 0:
        raise ValueError(f"{entry.path}: unknown sample rate")
    n = max(8192, int(window_seconds * fs))
    total = reader.total_samples()
    starts = list(range(0, max(total - n, 0) + 1, n))[:max_windows] or [0]
    for s in starts:
        x = reader.read_samples(s, n)
        if len(x) >= 4096:
            yield x, float(fs)


def build_from_manifest(
    manifest: str | Path,
    window_seconds: float = 0.5,
    max_windows: int = 20,
    progress_cb: Callable[[float, str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Features and class indices for every window of every manifest file."""
    entries = read_manifest(manifest)
    feats, labels = [], []
    for i, e in enumerate(entries):
        label = canonical_label(e.label)
        if label not in CLASSES:
            raise ValueError(f"{e.path}: {e.label.value} is not a trainable class")
        if progress_cb:
            progress_cb(i / max(1, len(entries)), f"Reading {e.path.name}")
        for x, fs in recording_windows(e, window_seconds, max_windows):
            feats.append(features_for(x, fs))
            labels.append(CLASSES.index(label))
    if not feats:
        return np.zeros((0, N_FEATURES), dtype=np.float32), np.zeros(0, dtype=int)
    return np.stack(feats).astype(np.float32), np.array(labels)
