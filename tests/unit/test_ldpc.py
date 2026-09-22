"""Tests for the experimental LDPC codec."""

import numpy as np
import pytest

from src.decoding.ldpc import ldpc_decode, ldpc_encode, ldpc_from_H, make_regular_ldpc


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(5)


def test_regular_construction():
    H = make_regular_ldpc(96, 3, 6)  # noqa: N806
    assert H.shape == (48, 96)
    assert np.all(H.sum(axis=0) == 3)
    assert np.all(H.sum(axis=1) == 6)


def test_encode_satisfies_parity():
    code = ldpc_from_H(make_regular_ldpc(96, 3, 6))
    info = np.random.randint(0, 2, code.k).astype(np.uint8)
    cw = ldpc_encode(info, code)
    assert not ((code.H.astype(int) @ cw.astype(int)) % 2).any()
    assert np.array_equal(cw[code._G_info_cols], info)


@pytest.mark.parametrize("sigma", [0.3, 0.6, 0.8])
def test_decode_awgn(sigma):
    code = ldpc_from_H(make_regular_ldpc(96, 3, 6))
    info = np.random.randint(0, 2, code.k).astype(np.uint8)
    cw = ldpc_encode(info, code)
    llr = ((1 - 2 * cw.astype(float)) + sigma * np.random.randn(code.n)) * 2 / sigma**2
    r = ldpc_decode(llr, code)
    assert r.converged
    assert np.array_equal(r.info_bits, info)


def test_bad_length():
    code = ldpc_from_H(make_regular_ldpc(96, 3, 6))
    with pytest.raises(ValueError):
        ldpc_decode(np.zeros(10), code)
    with pytest.raises(ValueError):
        ldpc_encode(np.zeros(3, dtype=np.uint8), code)
