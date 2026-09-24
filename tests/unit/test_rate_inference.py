"""Tests for headerless sample-rate inference."""

import numpy as np

from src.dsp.rate_inference import infer_sample_rate_candidates
from tests.signals.generators import generate_bpsk


def test_infers_cycles_per_sample_and_lists_truth():
    np.random.seed(4)
    sig, _ = generate_bpsk(4000, 10e3, 200e3, 15.0)
    inf = infer_sample_rate_candidates(sig)
    assert abs(inf.symbol_rate_cycles_per_sample - 0.05) < 0.001
    rates = [c.sample_rate_hz for c in inf.candidates]
    assert any(abs(r - 200e3) < 1 for r in rates)
    assert all(abs(c.samples_per_symbol - 20) < 0.5 for c in inf.candidates)
    assert "cannot be recovered" in inf.note


def test_too_short():
    inf = infer_sample_rate_candidates(np.zeros(100, dtype=np.complex64))
    assert not inf.candidates and inf.note


def test_noise_has_no_candidates():
    np.random.seed(1)
    noise = (np.random.randn(50000) + 1j * np.random.randn(50000)).astype(np.complex64)
    inf = infer_sample_rate_candidates(noise)
    assert inf.confidence < 0.5


def test_qpsk_with_iq_imbalance():
    """IQ imbalance puts an image at −offset; its beat with the signal (2·offset)
    is the strongest envelope line.  The rate is checked against the
    constellation, as in the pipeline."""
    from src.dsp import synth

    np.random.seed(7)
    sig, _ = synth.generate_mpsk(4000, 12.5e3, 250e3, 4, 12.0, -800.0)
    ph = np.radians(10)
    sig = sig.real + 1j * 1.2 * (sig.imag * np.cos(ph) + sig.real * np.sin(ph))
    inf = infer_sample_rate_candidates(sig.astype(np.complex64))
    assert abs(inf.symbol_rate_cycles_per_sample - 0.05) < 0.001
    assert any(abs(c.sample_rate_hz - 250e3) < 1 for c in inf.candidates)
