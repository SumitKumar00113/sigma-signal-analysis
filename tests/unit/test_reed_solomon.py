"""Tests for the Reed-Solomon codec."""

import numpy as np
import pytest

from src.decoding.reed_solomon import (
    GF256,
    ReedSolomon,
    bits_to_bytes,
    bytes_to_bits,
    rs_decode_stream,
)


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(11)


class TestGF:
    def test_mul_inverse(self):
        gf = GF256()
        for a in range(1, 256):
            assert gf.mul(a, gf.inv(a)) == 1

    def test_poly_eval(self):
        gf = GF256()
        # p(x) = x + 1 at x=1 → 0 ; at x=0 → 1
        assert gf.poly_eval([1, 1], 1) == 0
        assert gf.poly_eval([1, 1], 0) == 1


@pytest.mark.parametrize("n,k", [(255, 223), (15, 11), (63, 55), (204, 188)])
class TestRS:
    def test_encode_systematic(self, n, k):
        rs = ReedSolomon(n, k)
        msg = np.random.randint(0, 256, k).astype(np.uint8)
        cw = rs.encode(msg)
        assert len(cw) == n
        assert np.array_equal(cw[:k], msg)
        assert max(rs._syndromes(list(cw.astype(int)))) == 0

    def test_corrects_up_to_t(self, n, k):
        rs = ReedSolomon(n, k)
        msg = np.random.randint(0, 256, k).astype(np.uint8)
        cw = rs.encode(msg)
        for ne in (1, rs.t // 2, rs.t):
            c = cw.copy()
            pos = np.random.choice(n, ne, replace=False)
            c[pos] ^= np.random.randint(1, 256, ne).astype(np.uint8)
            res = rs.decode(c)
            assert res.success
            assert res.corrected == ne
            assert np.array_equal(res.data, msg)

    def test_detects_beyond_t(self, n, k):
        rs = ReedSolomon(n, k)
        cw = rs.encode(np.random.randint(0, 256, k).astype(np.uint8))
        c = cw.copy()
        pos = np.random.choice(n, rs.t + 1, replace=False)
        c[pos] ^= np.random.randint(1, 256, rs.t + 1).astype(np.uint8)
        assert not rs.decode(c).success


class TestVariants:
    def test_ccsds_field_parameters(self):
        rs = ReedSolomon(255, 223, prim_poly=0x187, fcr=112, prim=11)
        msg = np.random.randint(0, 256, 223).astype(np.uint8)
        c = rs.encode(msg)
        pos = np.random.choice(255, 10, replace=False)
        c[pos] ^= np.random.randint(1, 256, 10).astype(np.uint8)
        res = rs.decode(c)
        assert res.success and np.array_equal(res.data, msg)

    def test_shortened_code(self):
        rs = ReedSolomon(255, 223)
        msg = np.random.randint(0, 256, 100).astype(np.uint8)
        cw = rs.encode(msg)
        short = np.concatenate([cw[:100], cw[223:]])
        short[5] ^= 0x55
        short[50] ^= 0x01
        short[120] ^= 0x80
        res = rs.decode(short)
        assert res.success and res.corrected == 3
        assert np.array_equal(res.data, msg)

    def test_stream_decode(self):
        rs = ReedSolomon(15, 11)
        msgs = [np.random.randint(0, 256, 11).astype(np.uint8) for _ in range(4)]
        bits = bytes_to_bits(np.concatenate([rs.encode(m) for m in msgs]))
        out, results = rs_decode_stream(bits, rs)
        assert len(results) == 4 and all(r.success for r in results)
        assert np.array_equal(bits_to_bytes(out), np.concatenate(msgs))

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            ReedSolomon(10, 12)
