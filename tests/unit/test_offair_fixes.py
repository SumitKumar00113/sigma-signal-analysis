"""Regression tests for problems found on real off-air recordings
(NAVTEX, RTTY, NOAA APT, SSTV, NO-84 packets, RS41 radiosonde)."""

import numpy as np
import pytest

from src.core.enums import ModulationType as M
from src.core.models import RecordingMetadata
from src.decoding.auto_decode import auto_decode, expected_ber_from_evm
from src.decoding.correlation import parse_pattern
from src.decoding.fec_id import identify_fec
from src.decoding.framing import analyse_framing
from src.dsp import synth
from src.dsp.bursts import detect_bursts
from src.dsp.classification import (
    AUDIO_TONE_FRACTION,
    classify_modulation,
    discriminator_tonality,
)
from src.dsp.measurements import _prefer_fundamentals, estimate_symbol_rate_candidates
from src.dsp.pipeline import AnalysisPipeline, PipelineConfig, element_rate_note

RS41 = "0000100001101101010100111000100001000100011010010100100000011111"


def _noise(n, power, seed=0):
    rng = np.random.default_rng(seed)
    return ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) * np.sqrt(power / 2)
            ).astype(np.complex64)


class TestSymbolRate:
    def test_fundamental_beats_stronger_harmonic(self):
        """Long NAVTEX recording: the 3rd harmonic was the strongest line."""
        lines = [(300.0, 0.78), (200.0, 0.6), (100.0, 0.45), (400.0, 0.27), (500.0, 0.16)]
        assert _prefer_fundamentals(lines)[0][0] == 100.0

    def test_weak_subharmonic_is_not_promoted(self):
        lines = [(2400.0, 1.0), (1200.0, 0.1)]
        assert _prefer_fundamentals(lines)[0][0] == 2400.0

    def test_50_baud_fsk_is_reachable(self):
        np.random.seed(0)
        x = synth.modulate_bits(np.random.randint(0, 2, 1500), "2fsk", 50, 2000, snr_db=20,
                                deviation_hz=225)
        rates = [r for r, _ in estimate_symbol_rate_candidates(x, 2000)]
        assert any(abs(r - 50) < 1 for r in rates[:2])


class TestBursts:
    def test_dominant_signal_with_impulsive_noise_is_continuous(self):
        """HF atmospherics around a transmission must not make it intermittent."""
        np.random.seed(1)
        fs = 2000.0
        sig = synth.modulate_bits(np.random.randint(0, 2, 4000), "2fsk", 100, fs,
                                  deviation_hz=85)
        x = _noise(len(sig) + 4000, 0.002, 1)
        x[2000:2000 + len(sig)] += sig
        rng = np.random.default_rng(2)
        for s in rng.integers(0, len(x) - 200, 12):          # short noise crashes
            x[s:s + 150] += _noise(150, 0.02, int(s))
        det = detect_bursts(x, fs)
        assert det.continuous and not det.intermittent

    def test_burst_train_bits_are_joined(self):
        np.random.seed(3)
        fs = 48000.0
        x = _noise(int(3.2 * fs), 0.01, 3)
        for k in range(3):
            b = synth.modulate_bits(np.random.randint(0, 2, 1200), "2fsk", 4800, fs,
                                    deviation_hz=2400, freq_offset_hz=8000)
            s = int((0.2 + k) * fs)
            x[s:s + len(b)] += b
        res = AnalysisPipeline().run(x, RecordingMetadata(sample_rate_hz=fs))
        assert len(res.bursts) == 3
        bits, n = AnalysisPipeline.train_bits(res)
        assert n == 3 and len(bits) > 3 * 1000


class TestFalseStructure:
    def test_partial_structure_is_not_a_convolutional_code(self):
        """Asynchronous RTTY sampled on its half-element grid satisfies some
        parity checks of a punctured code, leaving many "errors"."""
        rng = np.random.default_rng(4)
        half = []
        for _ in range(3000):
            half += [0, 0] + [int(b) for b in np.repeat(rng.integers(0, 2, 5), 2)] + [1, 1, 1]
        stream = np.array(half, np.uint8)
        res = auto_decode(stream, 1, M.FSK2, ldpc_codes=[], search_interleaver=False,
                          expected_ber=expected_ber_from_evm(5.0, 1))
        assert res.step("FEC").status in ("none", "skipped")
        assert "decoded" not in res.stages

    def test_viterbi_check_rejects_partial_structure(self):
        rng = np.random.default_rng(5)
        stream = np.repeat(rng.integers(0, 2, 15000).astype(np.uint8), 2)
        fec = identify_fec(stream, try_rs=False, try_structure=False, ldpc_codes=[])
        assert fec.conv is None or (fec.conv.viterbi_ber or 0) <= 0.12

    def test_short_repeating_pattern_is_not_a_frame(self):
        rng = np.random.default_rng(6)
        idle = np.tile(parse_pattern("0x67E19F"), 400)[: 28 * 300]
        x = np.concatenate([rng.integers(0, 2, 500, dtype=np.uint8), idle])
        r = analyse_framing(x)
        assert not r.found and any("not a frame" in n or "repeat" in n for n in r.notes)

    def test_counter_needs_enough_frames(self):
        rng = np.random.default_rng(7)
        sync = parse_pattern("0x1ACFFC1D")
        frames = [np.concatenate([sync, np.array([(i >> (4 - j)) & 1 for j in range(5)],
                                                 np.uint8), rng.integers(0, 2, 300, np.uint8)])
                  for i in range(4)]
        r = analyse_framing(np.concatenate([rng.integers(0, 2, 50, np.uint8)] + frames))
        assert r.found and not any(f.kind == "counter" for f in r.fields)


class TestKnownSync:
    def test_rs41_header(self):
        rng = np.random.default_rng(8)
        hdr = parse_pattern(RS41)
        frames = [np.concatenate([hdr, rng.integers(0, 2, 2640, np.uint8)]) for _ in range(6)]
        r = analyse_framing(np.concatenate([rng.integers(0, 2, 140, np.uint8)] + frames))
        assert r.found and r.sync_name == "Vaisala RS41 radiosonde header"
        assert r.sync_hex == "0x086D53884469481F" and len(r.frames) == 6


def _fm(message, fs, dev):
    return np.exp(2j * np.pi * dev * np.cumsum(message) / fs).astype(np.complex64)


class TestFMWithAudio:
    @pytest.mark.parametrize("kind", ["apt", "afsk", "sstv"])
    def test_fm_carrying_audio_is_fm(self, kind):
        np.random.seed(9)
        fs = 48000.0
        t = np.arange(int(1.5 * fs)) / fs
        if kind == "apt":           # 2400 Hz AM subcarrier (image lines)
            m = (0.5 + 0.4 * np.sign(np.sin(2 * np.pi * 2 * t))) * np.sin(2 * np.pi * 2400 * t)
            dev = 12000.0
        elif kind == "afsk":        # Bell-202 tones, 1200 Bd
            bits = np.random.randint(0, 2, int(1.5 * 1200))
            f = np.where(np.repeat(bits, int(fs / 1200))[: len(t)] > 0, 1200.0, 2200.0)
            m = np.sin(2 * np.pi * np.cumsum(f) / fs)
            dev = 3000.0
        else:                       # SSTV-like: 1200 Hz sync, 1500–2300 Hz lines
            f = np.full(len(t), 1900.0)
            f[(t % 0.2) < 0.005] = 1200.0
            f[(t % 0.2) > 0.1] = 1500 + 800 * ((t[(t % 0.2) > 0.1] % 0.2) - 0.1) / 0.1
            m = np.sin(2 * np.pi * np.cumsum(f) / fs)
            dev = 5000.0
        x = synth.add_awgn(_fm(m, fs, dev), 20)
        assert discriminator_tonality(x, fs)[0] >= AUDIO_TONE_FRACTION
        res = AnalysisPipeline(PipelineConfig(analyze_bursts=False)).run(
            x, RecordingMetadata(sample_rate_hz=fs))
        assert res.analysis.modulation == M.FM
        assert res.demod is not None and res.demod.audio is not None

    @pytest.mark.parametrize("bt", [None, 0.3])
    def test_msk_gmsk_are_not_tonal(self, bt):
        np.random.seed(10)
        x, _ = synth.generate_msk(3000, 1200, 24000, bt, 15)
        assert discriminator_tonality(x, 24000)[0] < 0.1
        assert classify_modulation(x, 24000, 1200).modulation != M.FM


class TestElementRate:
    def test_async_rtty_note(self):
        """50 Bd Baudot with 1.5-element stop pulses sampled on a 100 Bd grid."""
        rng = np.random.default_rng(11)
        half = []
        for _ in range(600):
            half += [0, 0] + [int(b) for b in np.repeat(rng.integers(0, 2, 5), 2)] + [1, 1, 1]
        note = element_rate_note(np.array(half, np.uint8), 100.0)
        assert note is not None and "50.00 Bd" in note

    def test_random_bits_no_note(self):
        rng = np.random.default_rng(12)
        assert element_rate_note(rng.integers(0, 2, 5000), 100.0) is None
