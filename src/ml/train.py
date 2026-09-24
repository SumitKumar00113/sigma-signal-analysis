"""Train the learned modulation classifier.

Examples::

    # synthetic data only (≈ 3 min on 8 cores), install for this user
    python -m src.ml.train --per-class 600 --workers 8

    # add labelled recordings (see src/ml/dataset.py for the CSV format)
    python -m src.ml.train --manifest captures/labels.csv --recording-weight 3

The model is written to ``~/.sigma/models/modulation_classifier.joblib``
unless ``--out`` is given; the analysis pipeline picks it up automatically.
A held-out split is used to report accuracy per class and per SNR band.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ml.dataset import CLASSES, build_from_manifest, build_synthetic  # noqa: E402
from src.ml.model import USER_MODEL, ModulationModel, ml_available, train_model  # noqa: E402


def _report(model, x: np.ndarray, y: np.ndarray, snr: np.ndarray | None) -> float:
    pred = np.argmax(model.predict_proba(x), axis=1)
    acc = float(np.mean(pred == y))
    print(f"\nHeld-out accuracy: {acc:.1%} over {len(y)} examples")
    print(f"{'class':11s} {'acc':>6s}  most confused with")
    for ci, c in enumerate(CLASSES):
        m = y == ci
        if not np.any(m):
            continue
        wrong = pred[m][pred[m] != ci]
        conf = ""
        if len(wrong):
            vals, counts = np.unique(wrong, return_counts=True)
            conf = f"{CLASSES[vals[np.argmax(counts)]].value} ({counts.max()})"
        print(f"{c.value:11s} {np.mean(pred[m] == ci):6.1%}  {conf}")
    if snr is not None and len(snr):
        print("\nAccuracy by SNR (full-band):")
        for lo in range(int(np.floor(snr.min())), int(np.ceil(snr.max())), 4):
            m = (snr >= lo) & (snr < lo + 4)
            if np.any(m):
                print(f"  {lo:3d}–{lo + 4:<3d} dB  {np.mean(pred[m] == y[m]):6.1%}  (n={m.sum()})")
    return acc


@dataclass
class TrainingResult:
    model: ModulationModel
    path: Path
    holdout_accuracy: float
    recording_accuracy: float | None
    n_examples: int
    seconds: float


def run_training(
    per_class: int = 600,
    snr_range: tuple[float, float] = (0.0, 20.0),
    manifest: Path | None = None,
    recording_weight: float = 2.0,
    window_seconds: float = 0.5,
    workers: int = 4,
    seed: int = 0,
    holdout: float = 0.2,
    out: Path = USER_MODEL,
    progress_cb: Callable[[float, str], None] | None = None,
    verbose: bool = False,
) -> TrainingResult:
    """Build the data set, measure held-out accuracy, fit on everything, save."""
    t0 = time.time()

    def prog(lo: float, hi: float) -> Callable[[float, str], None] | None:
        if progress_cb is None:
            return None
        return lambda f, m: progress_cb(lo + (hi - lo) * f, m)

    x, y, snr = build_synthetic(per_class, snr_range, seed, workers, prog(0.0, 0.8))
    w = np.ones(len(y))
    source = np.zeros(len(y), dtype=int)
    if manifest:
        xr, yr = build_from_manifest(manifest, window_seconds, progress_cb=prog(0.8, 0.9))
        if verbose:
            print(f"Loaded {len(yr)} windows from {manifest}")
        x = np.concatenate([x, xr])
        y = np.concatenate([y, yr])
        snr = np.concatenate([snr, np.full(len(yr), np.nan)])
        w = np.concatenate([w, np.full(len(yr), recording_weight)])
        source = np.concatenate([source, np.ones(len(yr), dtype=int)])
    if verbose:
        print(f"Features ready in {time.time() - t0:.0f} s")
    if progress_cb:
        progress_cb(0.9, "Fitting the model")

    weighted = bool(np.any(w != 1))

    def fit(mask: np.ndarray, info: dict | None = None) -> ModulationModel:
        m = train_model(x[mask], y[mask], CLASSES, seed, info)
        if weighted:
            m.estimator.fit(x[mask], y[mask], sample_weight=w[mask])
        return m

    rng = np.random.default_rng(seed)
    test = rng.random(len(y)) < holdout
    trial = fit(~test)
    syn = test & (source == 0)
    if verbose:
        acc = _report(trial, x[syn], y[syn], snr[syn])
    else:
        acc = float(np.mean(np.argmax(trial.predict_proba(x[syn]), 1) == y[syn])) \
            if np.any(syn) else float("nan")
    rec_acc = None
    rec = test & (source == 1)
    if np.any(rec):
        rec_acc = float(np.mean(np.argmax(trial.predict_proba(x[rec]), 1) == y[rec]))
        if verbose:
            print(f"Held-out accuracy on recordings: {rec_acc:.1%} (n={rec.sum()})")

    final = fit(np.ones(len(y), dtype=bool), info={
        "per_class": per_class, "snr_range": list(snr_range),
        "manifest": str(manifest) if manifest else None,
        "n_examples": int(len(y)), "holdout_accuracy": acc,
        "recording_accuracy": rec_acc,
    })
    path = final.save(out)
    if progress_cb:
        progress_cb(1.0, "Model saved")
    return TrainingResult(final, path, acc, rec_acc, int(len(y)), time.time() - t0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--per-class", type=int, default=600, help="synthetic examples per class")
    ap.add_argument("--snr", type=float, nargs=2, default=(0.0, 20.0), metavar=("LO", "HI"))
    ap.add_argument("--manifest", type=Path, help="CSV of labelled recordings")
    ap.add_argument("--recording-weight", type=float, default=2.0,
                    help="sample weight of recording windows relative to synthetic ones")
    ap.add_argument("--window", type=float, default=0.5, help="recording window, seconds")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=USER_MODEL)
    args = ap.parse_args(argv)

    if not ml_available():
        print("scikit-learn is required: pip install -e '.[ml]'", file=sys.stderr)
        return 2
    print(f"Generating {args.per_class} synthetic examples × {len(CLASSES)} classes "
          f"(SNR {args.snr[0]:g}–{args.snr[1]:g} dB, {args.workers} workers)…")
    res = run_training(args.per_class, tuple(args.snr), args.manifest, args.recording_weight,
                       args.window, args.workers, args.seed, args.holdout, args.out,
                       verbose=True)
    print(f"\nSaved model to {res.path}  ({res.path.stat().st_size / 1024:.0f} kB, "
          f"{res.seconds:.0f} s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
