"""Evaluate the one-click analysis on real off-air recordings.

    python scripts/eval_offair.py "/path/to/off-air captures"

Each known capture (matched by file-name prefix) has a ground truth that
was established from the signal itself (tone spacing, element length,
standard parameters of the service):

========== ================================================= =================
capture    signal                                            checked
========== ================================================= =================
navtex     NAVTEX (SITOR-B), 2-FSK 100 Bd, 170 Hz shift      modulation, rate
rtty       DWD RTTY, 2-FSK 50 Bd, 450 Hz shift, async         modulation, rate,
                                                              Baudot text
no84       NO-84 packets: FM carrying audio tones             modulation
radiosonde Vaisala RS41, 2-FSK 4800 Bd, one frame per second  modulation, rate,
                                                              RS41 header
sstv       SSTV on narrow-band FM                             modulation
apt        NOAA-18 APT: FM with a 2400 Hz AM subcarrier       modulation, image
SDRSharp…  NOAA-15 SARP-3: residual-carrier PM, 2400 bps,     (not supported:
           ±35 kHz Doppler                                    reported only)
========== ================================================= =================

RTTY is demodulated on its 100 Bd half-element grid; the analysis notes
the 50 Bd element rate, and either is accepted.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.auto_analyse import load_recording  # noqa: E402
from src.core.enums import ModulationType as M  # noqa: E402
from src.decoding.auto_decode import auto_decode, expected_ber_from_evm  # noqa: E402
from src.dsp.pipeline import AnalysisPipeline  # noqa: E402

TRUTH = {
    "navtex": dict(mod={M.FSK2}, rates=(100.0,), sync=None),
    "rtty": dict(mod={M.FSK2}, rates=(50.0, 100.0), sync=None, text=True),
    "no84": dict(mod={M.FM}, rates=(), sync=None),
    "radiosonde": dict(mod={M.FSK2, M.GFSK}, rates=(4800.0,),
                       sync="Vaisala RS41 radiosonde header"),
    "sstv": dict(mod={M.FM}, rates=(), sync=None),
    "apt": dict(mod={M.FM}, rates=(), sync=None, image=True),
    "SDRSharp": dict(mod=None, rates=(), sync=None),
}


def _truth(path: Path) -> tuple[str, dict] | None:
    for key, t in TRUTH.items():
        if path.name.startswith(key):
            return key, t
    return None


def evaluate(path: Path) -> dict:
    key, t = _truth(path)
    t0 = time.perf_counter()
    samples, meta = load_recording(path, log=lambda *_: None)
    res = AnalysisPipeline().run(samples, meta)
    a, d = res.analysis, res.demod
    row = {"capture": key, "modulation": a.modulation.value, "rate": a.symbol_rate_hz,
           "framing": "", "ok": None}
    if d is not None and d.audio is None:
        bits, _ = AnalysisPipeline.train_bits(res)
        chain = auto_decode(bits, d.bits_per_symbol, d.modulation, ldpc_codes=[],
                            search_interleaver=False,
                            expected_ber=expected_ber_from_evm(d.evm_percent, d.bits_per_symbol))
        f = chain.framing
        row["framing"] = (f"{f.sync_name or 'blind'} ×{len(f.frames)}" if f.found else "none")
        if chain.text is not None:
            row["framing"] = f"Baudot text, {chain.text.characters:,} characters"
        row["fec"] = chain.step("FEC").status
    else:
        row["fec"] = "—"
    if res.apt is not None:
        row["framing"] = f"APT image, {res.apt.lines} lines ({res.apt.sync_quality:.0%} sync)"
    if t["mod"] is not None:
        ok = a.modulation in t["mod"]
        if t["rates"]:
            ok &= any(abs(a.symbol_rate_hz - r) <= 0.01 * r for r in t["rates"])
        if t["sync"]:
            ok &= t["sync"] in row["framing"]
        if t.get("image"):
            ok &= res.apt is not None and res.apt.sync_quality >= 0.9
        if t.get("text"):
            ok &= row["framing"].startswith("Baudot text")
        ok &= row["fec"] in ("none", "skipped", "—")      # none carries a library code
        row["ok"] = bool(ok)
    row["seconds"] = time.perf_counter() - t0
    return row


def main(argv: list[str]) -> int:
    folder = Path(argv[1]) if len(argv) > 1 else Path(".")
    files = sorted(p for p in folder.rglob("*.wav") if _truth(p))
    if not files:
        print(f"No known captures under {folder}")
        return 1
    print(f"{'capture':11s} {'modulation':11s} {'rate (Bd)':>10s} {'FEC':6s} "
          f"{'framing':34s} {'result':7s} {'s':>5s}")
    passed = total = 0
    for p in files:
        r = evaluate(p)
        verdict = {True: "PASS", False: "FAIL", None: "n/a"}[r["ok"]]
        if r["ok"] is not None:
            total += 1
            passed += r["ok"]
        print(f"{r['capture']:11s} {r['modulation']:11s} {r['rate']:10,.1f} {r['fec']:6s} "
              f"{r['framing']:34s} {verdict:7s} {r['seconds']:5.1f}")
    print(f"\n{passed}/{total} captures correct")
    return 0 if passed == total else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
