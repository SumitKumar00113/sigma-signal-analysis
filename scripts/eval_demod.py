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


def ber_with_ambiguity(rx_bits: np.ndarray, tx_bits: np.ndarray, bits_per_symbol: int,
                       max_shift_symbols: int = 400) -> tuple[float, int, str]:
    """Best BER over symbol alignment shifts and constellation rotations.

    Rotation is handled by trying every XOR mask / bit permutation that a
    phase rotation of the Gray-coded constellation produces.  For the
    demo we brute-force over a small candidate set.
    """
    rx = rx_bits.astype(np.uint8)
    tx = tx_bits.astype(np.uint8)
    best = (1.0, 0, "none")

    # Candidate transforms on each symbol's bit group
    def transforms(k: int):
        yield "identity", lambda g: g
        yield "invert", lambda g: g ^ 1
        if k == 2:
            yield "swap", lambda g: g[:, ::-1]
            yield "swap+inv", lambda g: g[:, ::-1] ^ 1
            yield "inv0", lambda g: g ^ np.array([1, 0], dtype=np.uint8)
            yield "inv1", lambda g: g ^ np.array([0, 1], dtype=np.uint8)
            yield "swap+inv0", lambda g: g[:, ::-1] ^ np.array([1, 0], dtype=np.uint8)
            yield "swap+inv1", lambda g: g[:, ::-1] ^ np.array([0, 1], dtype=np.uint8)
        if k == 4:
            # 90° rotations of 16-QAM: swap I/Q halves, invert one half
            yield "iq swap", lambda g: np.hstack([g[:, 2:], g[:, :2]])
            yield "inv I", lambda g: g ^ np.array([1, 1, 0, 0], dtype=np.uint8)
            yield "inv Q", lambda g: g ^ np.array([0, 0, 1, 1], dtype=np.uint8)
            yield "swap, inv I", lambda g: np.hstack([g[:, 2:], g[:, :2]]) ^ np.array([1, 1, 0, 0], dtype=np.uint8)
            yield "swap, inv Q", lambda g: np.hstack([g[:, 2:], g[:, :2]]) ^ np.array([0, 0, 1, 1], dtype=np.uint8)

    n_rx_sym = len(rx) // bits_per_symbol
    rx_g = rx[: n_rx_sym * bits_per_symbol].reshape(-1, bits_per_symbol)
    n_tx_sym = len(tx) // bits_per_symbol
    tx_g = tx[: n_tx_sym * bits_per_symbol].reshape(-1, bits_per_symbol)

    for name, fn in transforms(bits_per_symbol):
        cand = fn(rx_g)
        for shift in range(-max_shift_symbols, max_shift_symbols + 1):
            if shift >= 0:
                a, b = cand, tx_g[shift:]
            else:
                a, b = cand[-shift:], tx_g
            n = min(len(a), len(b)) - 10
            if n < 100:
                continue
            a, b = a[:n], b[:n]
            errs = np.mean(a != b)
            if errs < best[0]:
                best = (float(errs), shift, name)
    return best


def synthetic() -> None:
    np.random.seed(7)
    cases = [
        ("BPSK 15dB foff=1200", ModulationType.BPSK, lambda: generate_bpsk(4000, 10e3, 200e3, 15.0, 1200.0), 200e3, 10e3),
        ("BPSK 3dB foff=2000", ModulationType.BPSK, lambda: generate_bpsk(4000, 10e3, 200e3, 3.0, 2000.0), 200e3, 10e3),
        ("QPSK 12dB", ModulationType.QPSK, lambda: generate_qpsk(4000, 12.5e3, 250e3, 12.0), 250e3, 12.5e3),
        ("QPSK 6dB", ModulationType.QPSK, lambda: generate_qpsk(4000, 12.5e3, 250e3, 6.0), 250e3, 12.5e3),
        ("16QAM 20dB foff=500", ModulationType.QAM16, lambda: generate_qam16(4000, 10e3, 200e3, 20.0, 500.0), 200e3, 10e3),
        ("16QAM 14dB", ModulationType.QAM16, lambda: generate_qam16(4000, 10e3, 200e3, 14.0), 200e3, 10e3),
        ("2FSK 14dB", ModulationType.FSK2, lambda: generate_2fsk(4000, 8e3, 2.4e6, 20e3, 14.0), 2.4e6, 8e3),
        ("2FSK 6dB", ModulationType.FSK2, lambda: generate_2fsk(4000, 8e3, 2.4e6, 20e3, 6.0), 2.4e6, 8e3),
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
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "sigma_test_signals"
    recordings(folder)
