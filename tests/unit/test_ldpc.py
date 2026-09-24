"""Tests for the LDPC codec, file formats, identification and code library."""

import numpy as np
import pytest

from src.core.enums import FECType
from src.decoding import ldpc_library as lib
from src.decoding.fec_id import identify_fec, identify_ldpc
from src.decoding.ldpc import (
    LDPCCode,
    ldpc_decode,
    ldpc_decode_batch,
    ldpc_decode_stream,
    ldpc_encode,
    ldpc_from_edges,
    ldpc_from_H,
    ldpc_from_qc,
    ldpc_transmit,
    load_ldpc,
    make_regular_ldpc,
    read_alist,
    read_qc,
    write_alist,
)


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(5)


def ira_code(k: int = 1000, m: int = 1000, col_weight: int = 3, seed: int = 1) -> LDPCCode:
    """DVB-S2-style IRA code: random info part, dual-diagonal parity part."""
    rng = np.random.default_rng(seed)
    rows = [rng.choice(m, col_weight, replace=False) for _ in range(k)]
    r_info = np.concatenate(rows)
    c_info = np.repeat(np.arange(k), col_weight)
    r_par = np.concatenate([np.arange(m), np.arange(1, m)])
    c_par = np.concatenate([k + np.arange(m), k + np.arange(m - 1)])
    return ldpc_from_edges(k + m, m, np.concatenate([r_info, r_par]),
                           np.concatenate([c_info, c_par]), name="IRA test")


def qc_base(seed: int = 3) -> np.ndarray:
    """4×12 base matrix, dual-diagonal parity blocks, one punctured column."""
    rng = np.random.default_rng(seed)
    base = np.where(rng.random((4, 8)) < 0.6, rng.integers(0, 16, (4, 8)), -1)
    par = -np.ones((4, 4), dtype=int)
    for i in range(4):
        par[i, i] = 0
        if i:
            par[i, i - 1] = 0
    base = np.hstack([base, par])
    base[:, 0] = rng.integers(0, 16, 4)            # punctured column: connect everywhere
    return base


def _awgn_llr(cw, rate, ebn0_db, rng):
    sigma = np.sqrt(1 / (2 * rate * 10 ** (ebn0_db / 10)))
    y = (1 - 2.0 * cw) + sigma * rng.standard_normal(len(cw))
    return 2 * y / sigma**2


# ---------------------------------------------------------------------------
# Legacy API
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Construction and encoders
# ---------------------------------------------------------------------------


class TestEncoders:
    def test_ira_uses_triangular_encoder(self):
        code = ira_code()
        assert code.encoder.kind == "triangular" and code.k == 1000
        info = np.random.randint(0, 2, code.k).astype(np.uint8)
        cw = ldpc_encode(info, code)
        assert code.is_codeword(cw) and np.array_equal(cw[:1000], info)

    def test_generic_lower_triangular(self):
        # Parity part lower triangular but not just bidiagonal
        code = ira_code(k=200, m=200)
        rows = np.concatenate([code.rows, np.arange(5, 200)])
        cols = np.concatenate([code.cols, 200 + np.arange(0, 195)])
        tri = ldpc_from_edges(400, 200, rows, cols)
        assert tri.encoder.kind == "triangular"
        cw = ldpc_encode(np.random.randint(0, 2, 200).astype(np.uint8), tri)
        assert tri.is_codeword(cw)

    def test_rank_deficient_dimension(self):
        h = make_regular_ldpc(96, 3, 6)
        h = np.vstack([h, h[0] ^ h[1]])                # redundant check
        code = ldpc_from_H(h)
        # GF(2) rank of the original 48 checks, plus a dependent row
        base = ldpc_from_H(h[:48])
        assert code.m == 49 and code.k == base.k
        cw = ldpc_encode(np.random.randint(0, 2, code.k).astype(np.uint8), code)
        assert code.is_codeword(cw)

    def test_qc_expansion(self):
        base = np.array([[0, 2, -1], [1, -1, 0]])
        code = ldpc_from_qc(base, 4)
        h = code.H
        assert h.shape == (8, 12)
        assert h[0, 0] == 1 and h[0, 4 + 2] == 1 and h[1, 4 + 3] == 1   # shift by 2
        assert not h[0:4, 8:12].any()                                  # −1 → zero block

    def test_empty_rows_dropped(self):
        h = make_regular_ldpc(96, 3, 6)
        code = ldpc_from_H(np.vstack([h, np.zeros((1, 96), dtype=np.uint8)]))
        assert code.m == 48


# ---------------------------------------------------------------------------
# File formats
# ---------------------------------------------------------------------------


class TestFormats:
    def test_alist_roundtrip(self, tmp_path):
        code = ira_code(200, 100)
        write_alist(code, tmp_path / "c.alist")
        back = read_alist(tmp_path / "c.alist")
        assert np.array_equal(back.H, code.H)

    def test_alist_unpadded_and_comments(self, tmp_path):
        h = np.array([[1, 1, 0, 1], [0, 1, 1, 1]], dtype=np.uint8)
        text = ("# a comment line\n4 2\n2 3\n1 2 1 2\n3 3\n"
                "1\n1 2\n2\n1 2\n"                      # column lists, not padded
                "1 2 4\n2 3 4\n")
        (tmp_path / "u.alist").write_text(text)
        assert np.array_equal(read_alist(tmp_path / "u.alist").H, h)

    def test_qc_file_with_puncturing(self, tmp_path):
        base = qc_base()
        lines = [f"{base.shape[1]} {base.shape[0]} 16", ""]
        lines += [" ".join(str(v) for v in row) for row in base]
        lines += ["", " ".join(["0"] + ["1"] * (base.shape[1] - 1))]
        (tmp_path / "c.qc").write_text("\n".join(lines))
        code = read_qc(tmp_path / "c.qc")
        assert code.n == 12 * 16 and code.n_transmitted == 11 * 16
        assert not code.transmitted[:16].any()
        assert load_ldpc(tmp_path / "c.qc").n == code.n
        with pytest.raises(ValueError):
            load_ldpc(tmp_path / "c.txt")


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


class TestDecoding:
    def test_ira_awgn_batch(self):
        code = ira_code()
        rng = np.random.default_rng(0)
        infos = rng.integers(0, 2, (6, code.k)).astype(np.uint8)
        cws = np.stack([ldpc_encode(i, code) for i in infos])
        # 3.5 dB: this random test code has a small error floor at 3 dB
        # (the standard codes decode cleanly there)
        llr = np.stack([_awgn_llr(c, code.rate, 3.5, rng) for c in cws])
        r = ldpc_decode_batch(llr, code)
        assert r.converged.all()
        assert np.array_equal(r.info_bits, infos)
        assert r.iterations.max() < 50

    def test_punctured_qc_decodes(self):
        code = ldpc_from_qc(qc_base(), 16, transmitted_blocks=np.r_[0, np.ones(11)])
        rng = np.random.default_rng(1)
        info = rng.integers(0, 2, code.k).astype(np.uint8)
        tx = ldpc_transmit(ldpc_encode(info, code), code)
        r = ldpc_decode(_awgn_llr(tx, code.rate, 5.0, rng), code)   # transmitted-length input
        assert r.converged and np.array_equal(r.info_bits, info)

    def test_stream_hard_bits(self):
        code = ira_code()
        rng = np.random.default_rng(2)
        infos = [rng.integers(0, 2, code.k).astype(np.uint8) for _ in range(4)]
        stream = np.concatenate([ldpc_encode(i, code) for i in infos])
        flips = rng.choice(len(stream), 40, replace=False)
        stream[flips] ^= 1
        r = ldpc_decode_stream(stream, code)
        assert r.blocks == 4 and r.converged == 4 and r.corrected_bits == 40
        assert np.array_equal(r.info_bits, np.concatenate(infos))

    def test_syndrome(self):
        code = ira_code(200, 100)
        cw = ldpc_encode(np.random.randint(0, 2, 200).astype(np.uint8), code)
        assert code.is_codeword(cw)
        cw[0] ^= 1
        assert code.syndrome(cw).sum() == 3             # info column weight 3


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------


def _stream(code, nblk, ber, skip, rng, invert=False):
    blocks = [ldpc_transmit(ldpc_encode(rng.integers(0, 2, code.k).astype(np.uint8), code),
                            code) for _ in range(nblk)]
    x = np.concatenate(blocks)[skip:]
    x = x ^ (rng.random(len(x)) < ber).astype(np.uint8)
    return x ^ 1 if invert else x


class TestIdentification:
    def test_finds_code_and_alignment(self):
        rng = np.random.default_rng(4)
        codes = [ira_code(seed=1), ira_code(seed=2), ldpc_from_H(make_regular_ldpc(96, 3, 6))]
        x = _stream(codes[1], 6, 0.02, 777, rng)
        cand = identify_ldpc(x, codes)
        assert cand is not None and cand.code is codes[1]
        assert cand.offset == 2000 - 777

    def test_inversion(self):
        rng = np.random.default_rng(5)
        code = ira_code()                    # parity rows have odd weight (3 + 2)
        cand = identify_ldpc(_stream(code, 6, 0.01, 10, rng, invert=True), [code])
        assert cand is not None and cand.inverted

    def test_random_rejected(self):
        rng = np.random.default_rng(6)
        assert identify_ldpc(rng.integers(0, 2, 20000).astype(np.uint8), [ira_code()]) is None

    def test_identify_fec_reports_ldpc(self):
        rng = np.random.default_rng(7)
        code = ira_code()
        res = identify_fec(_stream(code, 6, 0.01, 333, rng), ldpc_codes=[code],
                           try_rs=False, try_structure=False, try_blind=False)
        assert res.fec_type == FECType.LDPC and res.ldpc.offset == 2000 - 333
        assert "LDPC" in res.summary()


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------


class TestLibrary:
    def test_catalogue_lookup(self):
        e = lib.find_entry("DVB-S2 r1/2 (64800)")
        assert e.filename == "DVB-S2_64800x32400.alist"
        assert lib.find_entry("wifi_540_648").name.startswith("Wi-Fi")
        with pytest.raises(KeyError):
            lib.find_entry("nope")

    def test_fetch_validates_and_installs(self, tmp_path):
        src = tmp_path / "remote"
        (src / "sub").mkdir(parents=True)
        write_alist(ira_code(200, 100), src / "sub" / "good.alist")
        (src / "bad.alist").write_text("not an alist")
        base = src.as_uri() + "/"
        store = tmp_path / "store"
        good = lib.CatalogueEntry("Good", "sub/good.alist", "test")
        path = lib.fetch(good, store, base_url=base)
        assert path.is_file() and lib.fetch(good, store, base_url=base) == path
        with pytest.raises(ValueError):
            lib.fetch(lib.CatalogueEntry("Bad", "bad.alist", "test"), store, base_url=base)
        assert [p.name for p in lib.installed(store)] == ["good.alist"]
        codes = lib.load_installed(store)
        assert len(codes) == 1 and codes[0].n == 300

    def test_cli_list(self, capsys):
        assert lib.main(["list"]) == 0
        assert "DVB-S2 r1/2 (64800)" in capsys.readouterr().out
