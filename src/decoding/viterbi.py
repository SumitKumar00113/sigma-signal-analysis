"""Convolutional encoding and Viterbi decoding.

Supports any rate-1/n feed-forward convolutional code described by its
constraint length *K* and *n* generator polynomials (given in octal, the
usual textbook convention, e.g. ``(0o171, 0o133)`` for the NASA/CCSDS
K=7 rate-1/2 code).  Puncturing to rate 2/3, 3/4, … is supported via a
puncture pattern.

The decoder accepts hard bits (0/1) or soft values (real numbers where
positive means "0" and negative means "1", i.e. sign-antipodal LLRs).
State-metric updates are vectorised across all 2^(K-1) states so the
Python loop is only over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Standard codes
# ---------------------------------------------------------------------------

STANDARD_CODES: dict[str, tuple[int, tuple[int, ...]]] = {
    # name: (K, generators in octal)
    "K7 r1/2 (NASA/CCSDS 171,133)": (7, (0o171, 0o133)),
    "K7 r1/3 (171,133,165)": (7, (0o171, 0o133, 0o165)),
    "K5 r1/2 (23,35)": (5, (0o23, 0o35)),
    "K3 r1/2 (7,5)": (3, (0o7, 0o5)),
    "K9 r1/2 (561,753)": (9, (0o561, 0o753)),
    "K9 r1/3 (557,663,711)": (9, (0o557, 0o663, 0o711)),
}


@dataclass
class ConvCode:
    """A rate-1/n feed-forward convolutional code."""

    constraint_length: int
    generators: tuple[int, ...]              # octal-specified polynomials as ints
    puncture: tuple[int, ...] | None = None  # e.g. (1,1,0,1) for rate 3/4 from 1/2

    # Derived tables
    _next_state: np.ndarray = field(init=False, repr=False)
    _outputs: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        k = self.constraint_length
        n_states = 1 << (k - 1)
        self._next_state = np.zeros((n_states, 2), dtype=np.int64)
        self._outputs = np.zeros((n_states, 2, self.n), dtype=np.uint8)
        for s in range(n_states):
            for bit in (0, 1):
                reg = (bit << (k - 1)) | s          # newest bit is MSB
                self._next_state[s, bit] = reg >> 1
                for gi, g in enumerate(self.generators):
                    self._outputs[s, bit, gi] = bin(reg & g).count("1") & 1

    @property
    def n(self) -> int:
        return len(self.generators)

    @property
    def num_states(self) -> int:
        return 1 << (self.constraint_length - 1)

    @property
    def rate(self) -> float:
        if self.puncture:
            return (len(self.puncture) / self.n) / sum(self.puncture)
        return 1.0 / self.n

    @classmethod
    def standard(cls, name: str) -> ConvCode:
        k, gens = STANDARD_CODES[name]
        return cls(k, gens)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def conv_encode(bits: np.ndarray, code: ConvCode, terminate: bool = True) -> np.ndarray:
    """Encode *bits* (0/1) and return the coded bit stream.

    With *terminate* the encoder is flushed with K−1 zeros so the decoder
    can end in the zero state.
    """
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if terminate:
        bits = np.concatenate([bits, np.zeros(code.constraint_length - 1, dtype=np.uint8)])
    out = np.zeros((len(bits), code.n), dtype=np.uint8)
    state = 0
    for i, b in enumerate(bits):
        out[i] = code._outputs[state, b]
        state = int(code._next_state[state, b])
    coded = out.reshape(-1)
    if code.puncture:
        mask = np.resize(np.asarray(code.puncture, dtype=bool), len(coded))
        coded = coded[mask]
    return coded


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


@dataclass
class ViterbiResult:
    bits: np.ndarray
    path_metric: float
    normalized_metric: float           # metric per coded symbol; near 0 = clean
    estimated_errors: int              # hard-decision disagreements along the path


def viterbi_decode(
    received: np.ndarray,
    code: ConvCode,
    soft: bool = False,
    terminated: bool = True,
) -> ViterbiResult:
    """Viterbi-decode *received*.

    Parameters
    ----------
    received :
        Hard bits (0/1) or, with ``soft=True``, real soft values where
        positive ⇒ 0 and negative ⇒ 1 (magnitude = reliability).
    terminated :
        If True, the decoder assumes the encoder was flushed to state 0
        and strips the K−1 tail bits.
    """
    r = np.asarray(received, dtype=np.float64).reshape(-1)
    n = code.n

    # Un-puncture: insert zero-reliability erasures where bits were removed
    received_mask: np.ndarray | None = None      # True where a bit was actually received
    if code.puncture:
        pat = np.asarray(code.puncture, dtype=bool)
        per_period = int(pat.sum())
        periods = int(np.ceil(len(r) / per_period))
        full = np.zeros(periods * len(pat))
        full_mask = np.resize(pat, len(full))
        full[full_mask] = np.pad(r, (0, full_mask.sum() - len(r)))
        r = full
        received_mask = full_mask.copy()
        received_mask[np.flatnonzero(full_mask)[len(received):]] = False   # padding
        # Erased positions carry 0 reliability in soft mode; in hard mode
        # we mark them with 0.5 so both hypotheses cost the same
        if not soft:
            r[~full_mask] = 0.5

    n_steps = len(r) // n
    r = r[: n_steps * n].reshape(n_steps, n)

    if soft:
        # Branch metric: correlation with antipodal expected outputs
        expected = 1.0 - 2.0 * code._outputs.astype(np.float64)   # 0→+1, 1→−1
        def branch_cost(t: int) -> np.ndarray:
            return -np.einsum("sbn,n->sb", expected, r[t])  # lower = better
    else:
        expected_bits = code._outputs.astype(np.float64)
        def branch_cost(t: int) -> np.ndarray:
            return np.sum(np.abs(expected_bits - r[t][None, None, :]), axis=2)

    n_states = code.num_states
    inf = np.inf
    metrics = np.full(n_states, inf)
    metrics[0] = 0.0
    survivors = np.zeros((n_steps, n_states), dtype=np.int64)   # predecessor state
    survivor_bits = np.zeros((n_steps, n_states), dtype=np.uint8)

    next_state = code._next_state
    for t in range(n_steps):
        bc = branch_cost(t)                                       # [state, bit]
        cand = metrics[:, None] + bc                              # [from_state, bit]
        new_metrics = np.full(n_states, inf)
        for bit in (0, 1):
            dest = next_state[:, bit]
            vals = cand[:, bit]
            # For each destination pick the best incoming
            order = np.argsort(vals)
            dest_sorted = dest[order]
            _, first_idx = np.unique(dest_sorted, return_index=True)
            best_from = order[first_idx]
            best_dest = dest_sorted[first_idx]
            better = vals[best_from] < new_metrics[best_dest]
            new_metrics[best_dest[better]] = vals[best_from][better]
            survivors[t, best_dest[better]] = best_from[better]
            survivor_bits[t, best_dest[better]] = bit
        # Normalise to prevent overflow on long streams
        metrics = new_metrics - np.min(new_metrics)

    # Trace back
    state = 0 if terminated else int(np.argmin(metrics))
    final_metric = float(metrics[state])
    decoded = np.zeros(n_steps, dtype=np.uint8)
    for t in range(n_steps - 1, -1, -1):
        decoded[t] = survivor_bits[t, state]
        state = int(survivors[t, state])

    # Count hard disagreements along the chosen path (re-encode)
    re_coded = conv_encode(decoded, ConvCode(code.constraint_length, code.generators),
                           terminate=False)
    if soft:
        hard_rx = (r.reshape(-1) < 0).astype(np.uint8)
    else:
        hard_rx = np.round(r.reshape(-1)).astype(np.uint8)
    m = min(len(re_coded), len(hard_rx))
    disagree = re_coded[:m] != hard_rx[:m]
    if received_mask is not None:
        # Punctured (erased) positions carry no information – don't count them
        disagree &= received_mask[:m]
    est_err = int(np.sum(disagree))

    if terminated and len(decoded) >= code.constraint_length - 1:
        decoded = decoded[: len(decoded) - (code.constraint_length - 1)]

    return ViterbiResult(
        bits=decoded,
        path_metric=final_metric,
        normalized_metric=final_metric / max(1, n_steps * n),
        estimated_errors=est_err,
    )
