"""Evaluate the demodulators on synthetic signals (known bits) and on the
ground-truth test recordings.  Not a unit test; a development harness.

Run:  python scripts/eval_demod.py [path/to/sigma_test_signals]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.enums import ModulationType  # noqa: E402
from src.dsp.demod import demodulate  # noqa: E402
from src.ingestion.sigmf_reader import SigMFReader  # noqa: E402
from tests.signals.generators import (  # noqa: E402
    generate_2fsk,
    generate_bpsk,
    generate_qam16,
    generate_qpsk,
)
from tests.signals.metrics import ber_with_ambiguity  # noqa: E402


def synthetic() -> None:
    np.random.seed(7)
    cases = [
        ("BPSK 15dB foff=1200", ModulationType.BPSK,
         lambda: generate_bpsk(4000, 10e3, 200e3, 15.0, 1200.0), 200e3, 10e3),
        ("BPSK 3dB foff=2000", ModulationType.BPSK,
         lambda: generate_bpsk(4000, 10e3, 200e3, 3.0, 2000.0), 200e3, 10e3),
        ("QPSK 12dB", ModulationType.QPSK,
         lambda: generate_qpsk(4000, 12.5e3, 250e3, 12.0), 250e3, 12.5e3),
        ("QPSK 6dB", ModulationType.QPSK,
         lambda: generate_qpsk(4000, 12.5e3, 250e3, 6.0), 250e3, 12.5e3),
        ("16QAM 20dB foff=500", ModulationType.QAM16,
         lambda: generate_qam16(4000, 10e3, 200e3, 20.0, 500.0), 200e3, 10e3),
        ("16QAM 14dB", ModulationType.QAM16,
         lambda: generate_qam16(4000, 10e3, 200e3, 14.0), 200e3, 10e3),
        ("2FSK 14dB", ModulationType.FSK2,
         lambda: generate_2fsk(4000, 8e3, 2.4e6, 20e3, 14.0), 2.4e6, 8e3),
        ("2FSK 6dB", ModulationType.FSK2,
         lambda: generate_2fsk(4000, 8e3, 2.4e6, 20e3, 6.0), 2.4e6, 8e3),
    ]
    print("=== synthetic (known bits) ===")
    for label, mod, gen, fs, rs in cases:
        sig, bits = gen()
        # Add a fractional timing offset by dropping a few samples
        sig = sig[7:]
        try:
            r = demodulate(sig, fs, mod, rs)
            ber, shift, xf = ber_with_ambiguity(r.bits, bits, r.bits_per_symbol)
            print(f"{label:24s} syms={r.num_symbols:5d} evm={r.evm_percent:5.1f}% "
                  f"cfo_res={r.residual_cfo_hz:7.1f}Hz tlock={r.timing_lock:.3f} "
                  f"clock={r.carrier_lock:.3f}  BER={ber:.4f} (shift {shift}, {xf})")
        except Exception as exc:  # noqa: BLE001
            print(f"{label:24s} FAILED: {exc}")


def recordings(folder: Path) -> None:
    gt_path = folder / "_ground_truth.json"
    if not gt_path.exists():
        print(f"(no ground truth at {folder}; skipping recordings)")
        return
    gt = json.loads(gt_path.read_text())
    mod_map = {"BPSK": ModulationType.BPSK, "QPSK": ModulationType.QPSK,
               "2FSK": ModulationType.FSK2, "16QAM": ModulationType.QAM16}
    print("\n=== ground-truth recordings ===")
    for g in gt:
        r = SigMFReader(folder / f"{g['name']}.sigmf-meta")
        meta = r.read_metadata()
        s = r.read_samples()
        mod = mod_map[g["modulation"]]
        try:
            d = demodulate(s, meta.sample_rate_hz, mod, float(g["symbol_rate_hz"]))
            print(f"{g['name']:22s} {mod.value:5s} syms={d.num_symbols:5d} bits={d.num_bits:6d} "
                  f"evm={d.evm_percent:5.1f}% coarse_cfo={d.coarse_cfo_hz:7.1f} "
                  f"(truth {g['freq_offset_hz']}) res_cfo={d.residual_cfo_hz:6.1f} "
                  f"fsk_levels={[round(x) for x in d.fsk_levels_hz]}")
        except Exception as exc:  # noqa: BLE001
            print(f"{g['name']:22s} FAILED: {exc}")


if __name__ == "__main__":
    synthetic()
    default = Path.home() / "Downloads" / "sigma_test_signals"
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    recordings(folder)
