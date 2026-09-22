"""Round-trip tests for all interleaver families."""

import numpy as np
import pytest

from src.core.enums import InterleaverType
from src.decoding import interleaving as il
from src.decoding.interleaving import InterleaverSpec, deinterleave, interleave


@pytest.fixture
def bits():
    np.random.seed(2)
    return np.random.randint(0, 2, 1000).astype(np.uint8)


def _roundtrip_ok(orig: np.ndarray, de: np.ndarray, tolerance: int = 0) -> bool:
    n = min(len(orig), len(de))
    return n >= len(orig) - tolerance and np.array_equal(orig[:n], de[:n])


class TestBlock:
    def test_roundtrip(self, bits):
        assert _roundtrip_ok(bits, il.block_deinterleave(il.block_interleave(bits, 8, 12), 8, 12))

    def test_permutes(self, bits):
        out = il.block_interleave(bits, 8, 12)
        assert not np.array_equal(out[:1000], bits)

    def test_known_small(self):
        x = np.arange(6)
        assert il.block_interleave(x, 2, 3).tolist() == [0, 3, 1, 4, 2, 5]


class TestConvolutional:
    def test_roundtrip(self, bits):
        out = il.convolutional_interleave(bits, 4, 3)
        de = il.convolutional_deinterleave(out, 4, 3)
        assert _roundtrip_ok(bits, de)

    def test_spreads_burst(self, bits):
        out = il.convolutional_interleave(bits, 4, 5)
        out[100:108] ^= 1  # burst of 8
        de = il.convolutional_deinterleave(out, 4, 5)
        errs = np.flatnonzero(de[:1000] != bits)
        assert len(errs) == 8
        # Errors are spread apart, not contiguous
        assert np.max(np.diff(errs)) > 4


class TestDiagonal:
    def test_roundtrip(self, bits):
        out = il.diagonal_interleave(bits, 6, 10)
        assert _roundtrip_ok(bits, il.diagonal_deinterleave(out, 6, 10))

    def test_differs_from_block(self, bits):
        assert not np.array_equal(il.diagonal_interleave(bits, 6, 10)[:60],
                                  il.block_interleave(bits, 6, 10)[:60])


class TestPseudoRandom:
    @pytest.mark.parametrize("gen", ["lcg", "numpy"])
    def test_roundtrip(self, bits, gen):
        out = il.pseudo_random_interleave(bits, 64, 42, gen)
        assert _roundtrip_ok(bits, il.pseudo_random_deinterleave(out, 64, 42, gen))

    def test_seed_matters(self, bits):
        a = il.pseudo_random_interleave(bits, 64, 1)
        b = il.pseudo_random_interleave(bits, 64, 2)
        assert not np.array_equal(a, b)

    def test_permutation_is_bijection(self):
        p = il.random_permutation(128, 9)
        assert sorted(p.tolist()) == list(range(128))


class TestUnified:
    @pytest.mark.parametrize("spec", [
        InterleaverSpec(InterleaverType.NONE),
        InterleaverSpec(InterleaverType.BLOCK, rows=4, cols=16),
        InterleaverSpec(InterleaverType.CONVOLUTIONAL, branches=3, delay=2),
        InterleaverSpec(InterleaverType.DIAGONAL, rows=5, cols=7),
        InterleaverSpec(InterleaverType.PSEUDO_RANDOM, block=32, seed=7),
    ])
    def test_roundtrip(self, bits, spec):
        assert _roundtrip_ok(bits, deinterleave(interleave(bits, spec), spec))
