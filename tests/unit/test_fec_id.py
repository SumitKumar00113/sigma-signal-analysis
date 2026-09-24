"""Tests for blind FEC identification (src/decoding/fec_id.py)."""

import numpy as np
import pytest

from src.core.enums import FECType
from src.decoding import fec_id as fid
from src.decoding.gf2 import gf2_nullspace, gf2_rank, rows_to_ints
from src.decoding.reed_solomon import ReedSolomon, bytes_to_bits
from src.decoding.viterbi import ConvCode, conv_encode


@pytest.fixture
def rng():
    return np.random.default_rng(11)


def _noisy(bits, ber, rng):
    return bits ^ (rng.random(len(bits)) < ber).astype(np.uint8)


def _encoded(rng, code, n=12000, skip=0, ber=0.0):
    u = rng.integers(0, 2, n).astype(np.uint8)
    c = conv_encode(u, code, terminate=False)
    return u, _noisy(c[skip:], ber, rng)


def _data_ber(decoded, u, margin=200):
    """Locate *decoded* inside the transmitted data *u* and return the BER."""
    key = decoded[margin: margin + 64]
    for p in range(len(u) - 64):
        if np.array_equal(u[p: p + 64], key):
            start = p - margin
            m = min(len(decoded), len(u) - start) - margin
            return float(np.mean(decoded[margin:m] != u[start + margin: start + m]))
    return 1.0


# ---------------------------------------------------------------------------
# GF(2) helpers
# ---------------------------------------------------------------------------


class TestGF2:
    def test_rank_and_nullspace(self):
        m = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0]], dtype=np.uint8)
        rows = rows_to_ints(m)
        assert gf2_rank(rows) == 2
        ns = gf2_nullspace(rows, 4)
        assert len(ns) == 2
        for v in ns:
            for r in rows:
                assert (r & v).bit_count() % 2 == 0

    def test_wide_rows(self):
        rng = np.random.default_rng(0)
        m = rng.integers(0, 2, (10, 100)).astype(np.uint8)
        m[9] = m[0] ^ m[1]
        assert gf2_rank(rows_to_ints(m)) == 9


# ---------------------------------------------------------------------------
# Parity checks
# ---------------------------------------------------------------------------


class TestParityChecks:
    @pytest.mark.parametrize("k,gens,punct", [
        (7, (0o171, 0o133), None),
        (7, (0o171, 0o133), (1, 1, 0, 1, 1, 0)),
        (5, (0o23, 0o35), (1, 1, 0, 1)),
        (9, (0o557, 0o663, 0o711), None),
        (7, (0o133, 0o171), (1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0)),
    ])
    def test_checks_annihilate_codewords(self, rng, k, gens, punct):
        code = ConvCode(k, gens, punct)
        c = conv_encode(rng.integers(0, 2, 3000).astype(np.uint8), code, terminate=False)
        checks = fid.conv_parity_checks(k, gens, punct)
        assert checks
        for chk in checks:
            s = fid.syndrome_stream(c, chk.array())
            assert not np.any(s[:: chk.period][: len(s) // chk.period - 1])

    def test_k7_classic_check(self):
        chk = fid.conv_parity_checks(7, (0o171, 0o133))[0]
        assert chk.period == 2 and chk.length == 14 and chk.weight == 10

    def test_odd_check_kept_for_inversion(self):
        # 663 has even weight, so (557,663,711)'s complement is not a codeword
        checks = fid.conv_parity_checks(9, (0o557, 0o663, 0o711))
        assert any(c.weight % 2 for c in checks)

    def test_ber_inversion_formula(self):
        p, w = 0.03, 10
        rate = (1 - (1 - 2 * p) ** w) / 2
        assert fid.ber_from_syndrome_rate(rate, w) == pytest.approx(p, rel=1e-6)

    def test_constant_windows_ignored(self):
        chk = fid.conv_parity_checks(7, (0o171, 0o133))[0]
        assert fid.score_check(np.zeros(5000, dtype=np.uint8), chk) is None


# ---------------------------------------------------------------------------
# Convolutional identification
# ---------------------------------------------------------------------------


class TestConvolutionalLibrary:
    def test_library_contents(self):
        lib = fid.conv_library()
        assert len(lib) > 50
        names = {h.name for h in lib}
        assert any("punctured 3/4" in n for n in names)
        assert any("order 133,171" in n for n in names)

    @pytest.mark.parametrize("k,gens,punct,skip,ber", [
        (7, (0o171, 0o133), None, 3, 0.05),
        (7, (0o133, 0o171), (1, 1, 0, 1, 1, 0), 3, 0.01),
        (5, (0o23, 0o35), (1, 1, 0, 1), 2, 0.02),
        (9, (0o557, 0o663, 0o711), None, 1, 0.04),
        (7, (0o171, 0o133), (1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0), 5, 0.003),
        (5, (0o23, 0o33), None, 0, 0.02),
    ])
    def test_identify_and_decode(self, rng, k, gens, punct, skip, ber):
        code = ConvCode(k, gens, punct)
        u, x = _encoded(rng, code, skip=skip, ber=ber)
        cands = fid.identify_convolutional(x)
        assert cands, "code not identified"
        best = cands[0]
        assert best.code.constraint_length == k
        assert best.code.generators == gens
        assert best.code.puncture == punct
        assert best.ber_estimate == pytest.approx(ber, abs=0.01)
        assert fid.viterbi_confirm(x, best) == pytest.approx(ber, abs=0.01)
        dec = fid.apply_conv_candidate(x, best)
        assert _data_ber(dec, u) < max(0.01, ber / 4)

    def test_phase_matches_offset(self, rng):
        code = ConvCode(7, (0o171, 0o133), (1, 1, 0, 1, 1, 0))
        for skip in range(4):
            _, x = _encoded(rng, code, n=6000, skip=skip)
            best = fid.identify_convolutional(x)[0]
            assert best.phase == (-skip) % 4

    def test_inversion_detected(self, rng):
        code = ConvCode(9, (0o557, 0o663, 0o711))
        u, x = _encoded(rng, code, ber=0.02)
        best = fid.identify_convolutional(x ^ 1)[0]
        assert best.inverted
        assert _data_ber(fid.apply_conv_candidate(x ^ 1, best), u) < 0.01

    def test_random_data_rejected(self, rng):
        assert fid.identify_convolutional(rng.integers(0, 2, 40000).astype(np.uint8)) == []


class TestBlindConvolutional:
    @pytest.mark.parametrize("k,gens,ber", [
        (6, (0o65, 0o57), 0.0),
        (8, (0o247, 0o371), 0.03),
        (7, (0o171, 0o133), 0.02),
    ])
    def test_rate_half(self, rng, k, gens, ber):
        u, x = _encoded(rng, ConvCode(k, gens), skip=1, ber=ber)
        found = fid.blind_conv_search(x)
        assert found
        best = found[0]
        assert best.blind
        assert best.code.constraint_length == k and best.code.generators == gens
        assert _data_ber(fid.apply_conv_candidate(x, best), u) < 0.01

    @pytest.mark.parametrize("skip", [0, 1, 2])
    def test_rate_third_with_common_factor(self, rng, skip):
        # g1 and g2 share the factor (x²+x+1): exercises the lcm recombination
        u, x = _encoded(rng, ConvCode(5, (0o25, 0o33, 0o37)), skip=skip, ber=0.01)
        best = fid.blind_conv_search(x)[0]
        assert best.code.generators == (0o25, 0o33, 0o37)
        assert best.phase == (-skip) % 3
        assert _data_ber(fid.apply_conv_candidate(x, best), u) < 0.01

    def test_random_rejected(self, rng):
        assert fid.blind_conv_search(rng.integers(0, 2, 20000).astype(np.uint8)) == []


# ---------------------------------------------------------------------------
# Reed-Solomon identification
# ---------------------------------------------------------------------------


def _rs_stream(rng, n, k, poly, fcr, blocks=12, sym_err=0, prefix=1237):
    rs = ReedSolomon(n, k, poly, fcr, 1)
    msgs, words = [], []
    for _ in range(blocks):
        m = rng.integers(0, 256, k).astype(np.uint8)
        cw = rs.encode(m).copy()
        for p in rng.choice(n, sym_err, replace=False):
            cw[p] ^= rng.integers(1, 256)
        msgs.append(m)
        words.append(cw)
    bits = bytes_to_bits(np.concatenate(words))
    return np.concatenate([rng.integers(0, 2, prefix).astype(np.uint8), bits]), msgs


class TestReedSolomon:
    @pytest.mark.parametrize("n,k,poly,fcr,err", [
        (255, 223, 0x11D, 0, 0),
        (204, 188, 0x11D, 0, 3),
        (255, 239, 0x187, 1, 1),
        (255, 247, 0x11D, 0, 1),
    ])
    def test_identify(self, rng, n, k, poly, fcr, err):
        bits, msgs = _rs_stream(rng, n, k, poly, fcr, sym_err=err)
        cand = fid.identify_reed_solomon(bits, lengths=(n,), prim_polys=(0x11D, 0x187))
        assert cand is not None
        assert (cand.n, cand.k, cand.prim_poly, cand.fcr) == (n, k, poly, fcr)
        assert cand.bit_offset == 1237              # exact, despite RS being cyclic
        assert cand.success_rate == 1.0
        out, ok, total = fid.apply_rs_candidate(bits, cand)
        assert ok == total == 12
        assert np.array_equal(out[: 8 * k], bytes_to_bits(msgs[0]))

    def test_short_code_not_mistaken_for_multiple(self, rng):
        bits, _ = _rs_stream(rng, 64, 48, 0x169, 1, blocks=16, sym_err=2, prefix=13)
        cand = fid.identify_reed_solomon(bits, lengths=(128, 64), prim_polys=(0x169,))
        assert cand is not None and (cand.n, cand.k) == (64, 48)

    def test_random_rejected(self, rng):
        x = rng.integers(0, 2, 40000).astype(np.uint8)
        assert fid.identify_reed_solomon(x, lengths=(255, 204)) is None


# ---------------------------------------------------------------------------
# Generic linear structure
# ---------------------------------------------------------------------------


class TestLinearStructure:
    def test_hamming_7_4(self, rng):
        g = np.array([[1, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 1],
                      [0, 0, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8)
        cw = (rng.integers(0, 2, (3000, 4)).astype(np.uint8) @ g) % 2
        bits = np.concatenate([np.zeros(3, dtype=np.uint8), cw.reshape(-1)])
        st = fid.detect_linear_structure(bits)
        assert st is not None and st.kind == "block"
        assert (st.n, st.k, st.bit_offset) == (7, 4, 3)

    def test_convolutional_like(self, rng):
        _, x = _encoded(rng, ConvCode(7, (0o171, 0o133)))
        st = fid.detect_linear_structure(x)
        assert st is not None and st.kind == "convolutional-like"
        assert st.period == 2 and st.rate_estimate == pytest.approx(0.5, abs=0.05)

    def test_random_none(self, rng):
        assert fid.detect_linear_structure(rng.integers(0, 2, 40000).astype(np.uint8)) is None


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


class TestIdentifyFEC:
    def test_concatenated(self, rng):
        rs = ReedSolomon(204, 188)
        outer = bytes_to_bits(np.concatenate(
            [rs.encode(rng.integers(0, 256, 188).astype(np.uint8)) for _ in range(10)]))
        outer = np.concatenate([rng.integers(0, 2, 333).astype(np.uint8), outer])
        coded = conv_encode(outer, ConvCode(7, (0o171, 0o133), (1, 1, 0, 1)), terminate=False)
        coded = _noisy(coded[5:], 0.02, rng)
        progress = []
        res = fid.identify_fec(coded, progress_cb=lambda f, m: progress.append(f))
        assert res.fec_type == FECType.CONCATENATED
        assert res.conv.code.puncture == (1, 1, 0, 1)
        assert (res.rs.n, res.rs.k) == (204, 188)
        assert res.rs.success_rate == 1.0
        assert progress and "Concatenated" in res.summary()

    def test_plain_convolutional(self, rng):
        _, x = _encoded(rng, ConvCode(7, (0o171, 0o133)), ber=0.01)
        res = fid.identify_fec(x, try_rs=False)
        assert res.fec_type == FECType.CONVOLUTIONAL
        assert res.conv.viterbi_ber == pytest.approx(0.01, abs=0.005)

    def test_block_structure_reported(self, rng):
        g = np.array([[1, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 1],
                      [0, 0, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8)
        cw = (rng.integers(0, 2, (3000, 4)).astype(np.uint8) @ g) % 2
        res = fid.identify_fec(cw.reshape(-1), try_rs=False, try_blind=False)
        assert res.fec_type == FECType.UNKNOWN
        assert res.structure is not None and res.structure.n == 7

    def test_uncoded(self, rng):
        res = fid.identify_fec(rng.integers(0, 2, 20000).astype(np.uint8), try_rs=False)
        assert res.fec_type == FECType.NONE
        assert "No FEC structure" in res.summary()

    def test_too_short(self):
        res = fid.identify_fec(np.zeros(10, dtype=np.uint8))
        assert res.fec_type == FECType.UNKNOWN and res.notes
