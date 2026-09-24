"""One-click decoding chain: bits → de-interleave → FEC decode → framing.

:func:`auto_decode` takes the hard bits of a demodulator and runs, without
analyst input:

1. **Bit mapping.**  A QPSK/16-QAM carrier recovered with a 90° or 180°
   phase error swaps and inverts the bits of each symbol (OQPSK can also
   pair each I with the wrong Q).  Every such mapping is scored against
   the convolutional-code library; one that shows code structure the raw
   stream does not is adopted.
2. **FEC on the stream as received.**  If a convolutional, Reed-Solomon
   or LDPC code is already visible, there is no interleaver to undo.
3. **Interleaver.**  Otherwise the interleaver search
   (:func:`~src.decoding.interleaver_id.identify_interleaver`) looks for
   the permutation that restores a code, and FEC identification is run
   again on the de-interleaved stream.
4. **Decoding** of the whole stream with the identified code(s):
   Viterbi, Reed-Solomon, Viterbi + RS (concatenated) or LDPC.
5. **Framing** (:func:`~src.decoding.framing.analyse_framing`): sync word,
   frame length and header fields on the most processed stream that shows
   framing.

Each step records what it found and why, so the analyst can check (and
redo by hand in the Decoding panel) any decision.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from src.core.enums import FECType, InterleaverType, ModulationType
from src.decoding.fec_id import (
    FECIdentification,
    apply_rs_candidate,
    conv_library,
    identify_fec,
)
from src.decoding.framing import FramingResult, analyse_framing
from src.decoding.interleaver_id import CodeOracle, InterleaverIdentification, identify_interleaver
from src.decoding.interleaving import deinterleave
from src.decoding.ldpc import ldpc_decode_stream
from src.decoding.viterbi import viterbi_decode

ProgressCallback = Callable[[float, str], None]
CancelCheck = Callable[[], bool]

MAPPING_Z_THRESHOLD = 8.0
MAPPING_SCAN_BITS = 20_000
STAGE_ORDER = ("raw", "deinterleaved", "decoded")


# ---------------------------------------------------------------------------
# Bit-mapping ambiguity
# ---------------------------------------------------------------------------


def _pairs(bits: np.ndarray, k: int) -> np.ndarray:
    return bits[: (len(bits) // k) * k].reshape(-1, k)


def mapping_transforms(bits_per_symbol: int, offset_rails: bool = False
                       ) -> list[tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    """Bit transforms produced by the demodulator's phase ambiguity.

    The identity comes first.  Pure inversion is left out: every later
    step detects an inverted stream by itself.
    """
    k = bits_per_symbol
    out: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = [("identity", lambda b: b)]
    if k == 2:
        masks = {"": (0, 0), "invert bit 1": (1, 0), "invert bit 2": (0, 1)}
        for swap in (False, True):
            for mname, m in masks.items():
                if not swap and not mname:
                    continue
                name = ", ".join(p for p in ("swap bits" if swap else "", mname) if p)
                mask = np.array(m, dtype=np.uint8)
                out.append((name, lambda b, s=swap, mk=mask:
                            ((_pairs(b, 2)[:, ::-1] if s else _pairs(b, 2)) ^ mk).reshape(-1)))
        if offset_rails:
            base = list(out)
            for name, fn in base:
                for shift in (1, -1):
                    out.append((f"{name}, re-pair rails {shift:+d}",
                                lambda b, f=fn, sh=shift: f(_repair_rails(b, sh))))
    elif k == 4:
        # Gray 16-QAM: negating an axis flips its first bit; 90° swaps the pairs
        for swap in (False, True):
            for ni in (0, 1):
                for nq in (0, 1):
                    if not (swap or ni or nq):
                        continue
                    mask = np.array([ni, 0, nq, 0], dtype=np.uint8)
                    name = ", ".join(p for p in ("swap I/Q" if swap else "",
                                                 "negate I" if ni else "",
                                                 "negate Q" if nq else "") if p)
                    out.append((name, lambda b, s=swap, mk=mask: (
                        (np.hstack([_pairs(b, 4)[:, 2:], _pairs(b, 4)[:, :2]]) if s
                         else _pairs(b, 4)) ^ mk).reshape(-1)))
    return out


def _repair_rails(bits: np.ndarray, shift: int) -> np.ndarray:
    g = _pairs(bits, 2)
    a, b = g[:, 0], g[:, 1]
    if shift > 0:
        a, b = a[:-shift], b[shift:]
    else:
        a, b = a[-shift:], b[:shift]
    return np.stack([a, b], axis=1).reshape(-1)


@dataclass
class MappingChoice:
    name: str
    z_score: float              # code-structure score with this mapping
    identity_z: float


def resolve_mapping(bits: np.ndarray, bits_per_symbol: int, offset_rails: bool = False,
                    oracle: CodeOracle | None = None, cancel_check: CancelCheck | None = None
                    ) -> tuple[np.ndarray, MappingChoice | None]:
    """Pick the bit mapping under which a convolutional code is visible."""
    transforms = mapping_transforms(bits_per_symbol, offset_rails)
    if len(transforms) == 1:
        return bits, None
    oracle = oracle or CodeOracle(conv_library())
    scan = bits[:MAPPING_SCAN_BITS]
    scores = []
    for name, fn in transforms:
        if cancel_check and cancel_check():
            break
        scores.append((oracle.score(fn(scan)).z, name, fn))
    if not scores:
        return bits, None
    identity_z = scores[0][0]
    z, name, fn = max(scores, key=lambda s: s[0])
    choice = MappingChoice(name, z, identity_z)
    if name == "identity" or z < MAPPING_Z_THRESHOLD or z < identity_z + MAPPING_Z_THRESHOLD:
        choice.name = "identity"
        choice.z_score = identity_z
        return bits, choice
    return fn(bits).astype(np.uint8), choice


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ChainStep:
    name: str
    status: str                 # "found", "none", "failed", "skipped"
    detail: str

    @property
    def icon(self) -> str:
        return {"found": "✓", "none": "–", "failed": "✗", "skipped": "·"}.get(self.status, "?")


@dataclass
class AutoDecodeResult:
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    steps: list[ChainStep] = field(default_factory=list)
    mapping: MappingChoice | None = None
    interleaver: InterleaverIdentification | None = None
    fec: FECIdentification | None = None
    decode_ok: bool | None = None
    framing: FramingResult = field(default_factory=FramingResult)
    framing_stage: str = ""
    elapsed_s: float = 0.0
    cancelled: bool = False

    @property
    def final_stage(self) -> str:
        for k in reversed(STAGE_ORDER):
            if k in self.stages:
                return k
        return ""

    @property
    def final_bits(self) -> np.ndarray:
        return self.stages.get(self.final_stage, np.zeros(0, dtype=np.uint8))

    def step(self, name: str) -> ChainStep | None:
        return next((s for s in self.steps if s.name == name), None)

    def summary(self) -> str:
        lines = [f"{s.icon} {s.name}: {s.detail}" for s in self.steps]
        tail = f"Done in {self.elapsed_s:.1f} s"
        if self.cancelled:
            tail = "Cancelled after " + f"{self.elapsed_s:.1f} s"
        return "\n".join(lines + [tail])


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------


def _has_code(fec: FECIdentification | None) -> bool:
    return fec is not None and any(c is not None for c in (fec.conv, fec.rs, fec.ldpc))


def _decode(bits: np.ndarray, fec: FECIdentification) -> tuple[np.ndarray, str, bool | None]:
    """Decode the whole stream with an identified code."""
    if fec.ldpc is not None:
        c = fec.ldpc
        x = bits[c.offset:]
        if c.inverted:
            x = x ^ 1
        r = ldpc_decode_stream(x, c.code)
        return (r.info_bits, f"LDPC: {r.converged}/{r.blocks} blocks converged, "
                f"{r.corrected_bits:,} channel bits corrected",
                (r.converged == r.blocks) if r.blocks else None)
    if fec.conv is not None:
        cv = fec.conv
        x = bits[cv.phase:]
        if cv.inverted:
            x = x ^ 1
        vr = viterbi_decode(x, cv.code, terminated=False)
        msg = (f"Viterbi: {len(x):,} → {len(vr.bits):,} bits, {vr.estimated_errors:,} "
               f"channel errors corrected ({vr.estimated_errors / max(1, len(x)):.2%})")
        ok: bool | None = True
        out = vr.bits
        if fec.rs is not None:
            out, good, total = apply_rs_candidate(vr.bits, fec.rs)
            msg += f"; RS({fec.rs.n},{fec.rs.k}): {good}/{total} blocks OK"
            ok = good == total if total else None
        return out, msg, ok
    if fec.rs is not None:
        out, good, total = apply_rs_candidate(bits, fec.rs)
        return (out, f"RS({fec.rs.n},{fec.rs.k}): {good}/{total} blocks OK",
                good == total if total else None)
    raise ValueError("no code to decode with")


def auto_decode(
    bits: np.ndarray,
    bits_per_symbol: int = 1,
    modulation: ModulationType | None = None,
    ldpc_codes: list | None = None,
    pseudo_random_blocks: tuple[int, ...] = (),
    search_interleaver: bool = True,
    progress_cb: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
    **_: object,
) -> AutoDecodeResult:
    """Run the full decoding chain on demodulated *bits* (see module docstring).

    *ldpc_codes* defaults to the installed standard codes.  Accepts the
    ``progress_cb`` / ``cancel_check`` keywords of the GUI worker.
    """
    t0 = time.perf_counter()
    res = AutoDecodeResult()
    x = np.asarray(bits, dtype=np.uint8).reshape(-1)
    res.stages["raw"] = x
    cancelled = cancel_check or (lambda: False)

    def prog(lo: float, hi: float) -> ProgressCallback | None:
        if progress_cb is None:
            return None
        return lambda f, m: progress_cb(lo + (hi - lo) * f, m)

    def report(frac: float, msg: str) -> None:
        if progress_cb:
            progress_cb(frac, msg)

    def finish() -> AutoDecodeResult:
        res.cancelled = cancelled()
        res.elapsed_s = time.perf_counter() - t0
        report(1.0, "Auto-decode complete")
        return res

    if len(x) < 256:
        res.steps.append(ChainStep("Input", "failed", f"only {len(x)} bits; need ≥ 256"))
        return finish()

    # 1 · Bit mapping
    report(0.0, "Resolving the bit mapping")
    offset_rails = modulation == ModulationType.OQPSK
    x, choice = resolve_mapping(x, bits_per_symbol, offset_rails, cancel_check=cancel_check)
    res.mapping = choice
    if choice is None:
        res.steps.append(ChainStep("Bit mapping", "skipped",
                                   f"{bits_per_symbol} bit(s)/symbol: no mapping ambiguity "
                                   "beyond inversion (handled later)"))
    elif choice.name != "identity":
        res.stages["raw"] = x
        res.steps.append(ChainStep("Bit mapping", "found",
                                   f"{choice.name} (code structure z={choice.z_score:.0f} vs "
                                   f"{choice.identity_z:.0f} as demodulated)"))
    else:
        res.steps.append(ChainStep("Bit mapping", "none", "as demodulated"))

    # 2 · FEC on the stream as received
    if cancelled():
        return finish()
    codes = ldpc_codes
    if codes is None:
        from src.decoding.ldpc_library import load_installed

        codes = load_installed()
    fec = identify_fec(x, ldpc_codes=codes, progress_cb=prog(0.05, 0.35),
                       cancel_check=cancel_check)
    stream = x

    # 3 · Interleaver
    if _has_code(fec):
        res.steps.append(ChainStep("Interleaver", "none",
                                   "code visible without de-interleaving"))
    elif not search_interleaver:
        res.steps.append(ChainStep("Interleaver", "skipped", "search disabled"))
    elif not cancelled():
        report(0.35, "Searching for an interleaver")
        il = identify_interleaver(x, pseudo_random_blocks=pseudo_random_blocks,
                                  progress_cb=prog(0.35, 0.75), cancel_check=cancel_check)
        res.interleaver = il
        if il.best is not None:
            b = il.best
            stream = deinterleave(x[b.offset:], b.spec).astype(np.uint8)
            res.stages["deinterleaved"] = stream
            res.steps.append(ChainStep("Interleaver", "found", b.describe()))
            fec = identify_fec(stream, ldpc_codes=codes, progress_cb=prog(0.75, 0.85),
                               cancel_check=cancel_check)
        else:
            detail = ("none found (no library convolutional code restored)"
                      if il.kind != InterleaverType.NONE else "none")
            res.steps.append(ChainStep("Interleaver", "none", detail))
    res.fec = fec

    # 4 · FEC decoding
    if cancelled():
        return finish()
    if _has_code(fec):
        head = fec.summary().splitlines()[0]
        res.steps.append(ChainStep("FEC", "found", head))
        report(0.85, "Decoding")
        try:
            out, msg, ok = _decode(stream, fec)
            res.stages["decoded"] = out.astype(np.uint8)
            res.decode_ok = ok
            res.steps.append(ChainStep("Decode", "found" if ok is not False else "failed", msg))
        except Exception as exc:  # noqa: BLE001 – report, keep the chain going
            res.decode_ok = False
            res.steps.append(ChainStep("Decode", "failed", f"{type(exc).__name__}: {exc}"))
    else:
        detail = fec.summary().splitlines()[0] if fec else "not run"
        if fec is not None and fec.fec_type == FECType.NONE:
            detail = "no FEC detected (uncoded, unsupported code or too many errors)"
        res.steps.append(ChainStep("FEC", "none", detail))

    # 5 · Framing: the most processed stream that shows it
    if cancelled():
        return finish()
    report(0.92, "Searching for frame sync and header")
    tried = []
    for stage in reversed(STAGE_ORDER):
        if stage not in res.stages:
            continue
        fr = analyse_framing(res.stages[stage])
        tried.append(stage)
        if fr.found:
            res.framing, res.framing_stage = fr, stage
            break
        if not tried[1:]:
            res.framing = fr
    if res.framing.found:
        f = res.framing
        what = f.sync_name or "blind discovery"
        res.steps.append(ChainStep(
            "Framing", "found",
            f"{len(f.frames)} frames on the {res.framing_stage} stream, sync "
            f"{f.sync_hex} ({what}), "
            + (f"length {f.period} bits, " if f.period else "")
            + f"header {f.header_bits} bits"))
    else:
        res.steps.append(ChainStep("Framing", "none",
                                   "no sync word found in " + ", ".join(tried) + " bits"))
    return finish()
