"""End-to-end demodulator tests with known transmitted bits."""

import numpy as np
import pytest

from src.core.enums import ModulationType
from src.core.exceptions import DemodulationError, InsufficientSamplesError
from src.dsp.demod import (
    demodulate,
    differential_decode_bits,
    psk_symbols_to_bits,
    qam_symbols_to_bits,
)
from tests.signals.generators import generate_2fsk, generate_bpsk, generate_qam16, generate_qpsk
from tests.signals.metrics import ber_with_ambiguity


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(42)


class TestBitMapping:
    def test_bpsk_mapping(self):
        bits = psk_symbols_to_bits(np.array([1 + 0j, -1 + 0j, 1 + 0j]), 2)
        assert bits.tolist() == [0, 1, 0]

    def test_qpsk_gray_adjacent_points_differ_by_one_bit(self):
        pts = np.exp(1j * (np.arange(4) * np.pi / 2))
        bits = psk_symbols_to_bits(pts, 4).reshape(-1, 2)
        for i in range(4):
            assert np.sum(bits[i] != bits[(i + 1) % 4]) == 1

    def test_qam16_bits_shape_and_gray(self):
        levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        pts = np.array([complex(i, q) for i in levels for q in levels])
        bits = qam_symbols_to_bits(pts, 16)
        assert bits.shape == (64,)
        # Distinct symbols map to distinct 4-bit words
        words = {tuple(w) for w in bits.reshape(-1, 4)}
        assert len(words) == 16

    def test_differential_decode(self):
        b = np.array([1, 1, 0, 1, 1], dtype=np.uint8)
        assert differential_decode_bits(b).tolist() == [1, 0, 1, 1, 0]


class TestDemodulators:
    @pytest.mark.parametrize("snr,cfo", [(15.0, 1200.0), (3.0, 2000.0)])
    def test_bpsk(self, snr, cfo):
        sig, bits = generate_bpsk(4000, 10e3, 200e3, snr, cfo)
        r = demodulate(sig[7:], 200e3, ModulationType.BPSK, 10e3)
        ber, _, _ = ber_with_ambiguity(r.bits, bits, 1)
        assert ber < 0.01
        assert abs(r.coarse_cfo_hz - cfo) < 60
        assert r.evm_percent < 30

    @pytest.mark.parametrize("snr", [12.0, 6.0])
    def test_qpsk(self, snr):
        sig, bits = generate_qpsk(4000, 12.5e3, 250e3, snr)
        r = demodulate(sig[3:], 250e3, ModulationType.QPSK, 12.5e3)
        ber, _, _ = ber_with_ambiguity(r.bits, bits, 2)
        assert ber < 0.01
        assert r.bits_per_symbol == 2

    def test_qam16(self):
        sig, bits = generate_qam16(4000, 10e3, 200e3, 20.0, 500.0)
        r = demodulate(sig[5:], 200e3, ModulationType.QAM16, 10e3)
        ber, _, _ = ber_with_ambiguity(r.bits, bits, 4)
        assert ber < 0.01
        assert r.bits_per_symbol == 4

    @pytest.mark.parametrize("snr", [14.0, 6.0])
    def test_fsk2(self, snr):
        sig, bits = generate_2fsk(4000, 8e3, 2.4e6, 20e3, snr)
        r = demodulate(sig[11:], 2.4e6, ModulationType.FSK2, 8e3)
        ber, _, _ = ber_with_ambiguity(r.bits, bits, 1)
        assert ber < 0.01
        assert len(r.fsk_levels_hz) == 2
        # Tones at roughly ±20 kHz
        assert abs(abs(r.fsk_levels_hz[1] - r.fsk_levels_hz[0]) - 40e3) < 8e3

    def test_symbols_unit_power(self):
        sig, _ = generate_qpsk(2000, 12.5e3, 250e3, 15.0)
        r = demodulate(sig, 250e3, ModulationType.QPSK, 12.5e3)
        assert np.isclose(np.mean(np.abs(r.symbols) ** 2), 1.0, atol=0.15)


class TestErrors:
    def test_unsupported_modulation(self):
        sig, _ = generate_bpsk(500, 10e3, 200e3)
        with pytest.raises(DemodulationError):
            demodulate(sig, 200e3, ModulationType.OFDM, 10e3)

    def test_too_few_samples(self):
        sig, _ = generate_bpsk(20, 10e3, 200e3)
        with pytest.raises(InsufficientSamplesError):
            demodulate(sig, 200e3, ModulationType.BPSK, 10e3)

    def test_bad_symbol_rate(self):
        sig, _ = generate_bpsk(500, 10e3, 200e3)
        with pytest.raises(DemodulationError):
            demodulate(sig, 200e3, ModulationType.BPSK, 0.0)

    def test_undersampled(self):
        sig, _ = generate_bpsk(500, 10e3, 200e3)
        with pytest.raises(DemodulationError):
            demodulate(sig, 200e3, ModulationType.BPSK, 180e3)
