"""Tests for blind interleaver identification (src/decoding/interleaver_id.py)."""

import numpy as np
import pytest

from src.core.enums import InterleaverType as T
from src.decoding import interleaver_id as iid
from src.decoding.fec_id import ConvHypothesis, common_conv_hypotheses
from src.decoding.interleaving import InterleaverSpec, deinterleave, interleave, random_permutation
from src.decoding.viterbi import ConvCode, conv_encode

K7 = ConvCode(7, (0o171, 0o133))
FAST = common_conv_hypotheses()      # unpunctured rate-1/2 codes: quick stride scans


@pytest.fixture
def rng():
    return np.random.default_rng(21)


def _stream(rng, spec, code=K7, n=16000, skip=0, ber=0.0):
    coded = conv_encode(rng.integers(0, 2, n).astype(np.uint8), code, terminate=False)
    x = interleave(coded, spec)[skip:]
    return coded, x ^ (rng.random(len(x)) < ber).astype(np.uint8)


def _restores(coded, x, cand):
    """De-interleaving with the candidate yields a contiguous run of the coded stream."""
    y = deinterleave(x[cand.offset:], cand.spec)
    key = y[1000:1128]
    for p in range(len(coded) - 128):
        if np.array_equal(coded[p: p + 128], key):
            start = p - 1000
            m = min(len(y), len(coded) - start) - 2000
            return np.array_equal(y[1000:m], coded[start + 1000: start + m])
    return False


class TestPrimitives:
    def test_lcg_vectorised_matches_reference(self):
        seeds = np.array([0, 1, 5, 12345, 2**31 + 7])
        perms = iid.lcg_permutations(97, seeds)
        for s, p in zip(seeds, perms, strict=True):
            assert np.array_equal(p, random_permutation(97, int(s)))

    def test_oracle_prefers_true_code(self, rng):
        coded, _ = _stream(rng, InterleaverSpec(T.NONE))
        oracle = iid.CodeOracle(FAST)
        sc = oracle.score(coded)
        assert sc.z > 50
        assert oracle.hypotheses[sc.hyp].generators == (0o171, 0o133)
        assert oracle.score(rng.integers(0, 2, 20000).astype(np.uint8)).z < 8


class TestIdentify:
    @pytest.mark.parametrize("spec,skip", [
        (InterleaverSpec(T.BLOCK, rows=12, cols=40), 301),
        (InterleaverSpec(T.BLOCK, rows=32, cols=64), 77),
        (InterleaverSpec(T.DIAGONAL, rows=16, cols=30), 77),
        (InterleaverSpec(T.CONVOLUTIONAL, branches=20, delay=3), 0),
        (InterleaverSpec(T.BLOCK, rows=8, cols=8), 3),             # short columns: comb search
        (InterleaverSpec(T.BLOCK, rows=24, cols=6), 11),
        (InterleaverSpec(T.CONVOLUTIONAL, branches=6, delay=5), 2),
        (InterleaverSpec(T.CONVOLUTIONAL, branches=12, delay=17), 5),
    ])
    def test_exact_recovery(self, rng, spec, skip):
        coded, x = _stream(rng, spec, skip=skip)
        res = iid.identify_interleaver(x, hypotheses=FAST)
        best = res.best
        assert best is not None, res.summary()
        assert best.spec.kind == spec.kind
        assert (best.spec.rows, best.spec.cols, best.spec.branches, best.spec.delay) == \
            (spec.rows, spec.cols, spec.branches, spec.delay)
        assert _restores(coded, x, best)
        assert "171,133" in best.code_name

    def test_noisy_full_library(self, rng):
        spec = InterleaverSpec(T.DIAGONAL, rows=16, cols=30)
        _, x = _stream(rng, spec, skip=40, ber=0.03)
        best = iid.identify_interleaver(x).best
        assert best is not None
        assert (best.spec.kind, best.spec.rows, best.spec.cols) == (T.DIAGONAL, 16, 30)
        assert best.offset == 480 - 40

    def test_punctured_code_supplied(self, rng):
        code = ConvCode(7, (0o171, 0o133), (1, 1, 0, 1, 1, 0))
        spec = InterleaverSpec(T.CONVOLUTIONAL, branches=20, delay=3)
        coded, x = _stream(rng, spec, code=code, ber=0.01)
        hyp = ConvHypothesis("K7 3/4", 7, (0o171, 0o133), (1, 1, 0, 1, 1, 0))
        best = iid.identify_interleaver(x, hypotheses=FAST, comb_hypotheses=[hyp]).best
        assert best is not None
        assert (best.spec.kind, best.spec.branches, best.spec.delay) == (T.CONVOLUTIONAL, 20, 3)

    def test_no_interleaver(self, rng):
        coded, _ = _stream(rng, InterleaverSpec(T.NONE))
        res = iid.identify_interleaver(coded, hypotheses=FAST)
        assert res.best is None and res.kind == T.NONE
        assert "No interleaver" in res.summary()

    def test_random_data(self, rng):
        res = iid.identify_interleaver(rng.integers(0, 2, 20000).astype(np.uint8),
                                     hypotheses=FAST, comb_hypotheses=FAST[:2])
        assert res.best is None and res.kind == T.UNKNOWN

    def test_too_short(self):
        res = iid.identify_interleaver(np.zeros(100, dtype=np.uint8))
        assert res.best is None and res.notes


class TestPseudoRandom:
    @pytest.mark.parametrize("gen,block,seed,skip", [
        ("lcg", 256, 42, 100),
        ("numpy", 128, 9, 0),
    ])
    def test_seed_search(self, rng, gen, block, seed, skip):
        spec = InterleaverSpec(T.PSEUDO_RANDOM, block=block, seed=seed, generator=gen)
        coded, x = _stream(rng, spec, skip=skip, ber=0.01)
        cands = iid.search_pseudo_random(x, [block], FAST[:1], seeds=range(64), generators=(gen,))
        assert cands
        best = cands[0]
        assert (best.spec.seed, best.spec.generator) == (seed, gen)
        assert best.offset == (-skip) % block

    def test_via_identify(self, rng):
        spec = InterleaverSpec(T.PSEUDO_RANDOM, block=64, seed=5, generator="lcg")
        coded, x = _stream(rng, spec, skip=10)
        res = iid.identify_interleaver(x, hypotheses=FAST, comb_hypotheses=FAST[:1],
                                     pseudo_random_blocks=[64], seeds=range(16))
        assert res.best is not None and res.best.spec.seed == 5
        assert _restores(coded, x, res.best)
