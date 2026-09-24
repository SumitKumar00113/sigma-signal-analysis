"""Tests for real → complex conversion of WAV recordings."""

import wave
from pathlib import Path

import numpy as np
import pytest

from src.core.enums import WavInterpretation
from src.core.models import RecordingMetadata
from src.dsp.measurements import estimate_frequency_offset, estimate_snr_inband, estimate_snr_psd
from src.dsp.pipeline import AnalysisPipeline
from src.ingestion import real_to_complex as rc
from src.ingestion.wav_reader import WavReader
from tests.signals.generators import generate_2fsk, generate_bpsk, generate_qpsk
from tests.signals.metrics import ber_with_ambiguity

FS = 48000


def _write_wav(path: Path, data: np.ndarray, fs: int = FS) -> Path:
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    data = data / np.max(np.abs(data)) * 0.8
    with wave.open(str(path), "wb") as w:
        w.setnchannels(data.shape[1])
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes((data * 32767).astype("<i2").tobytes())
    return path


def _audio_modem(kind: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Real passband audio of a modem signal plus its bits."""
    np.random.seed(3)
    t = np.arange
    if kind == "bpsk":
        sig, bits = generate_bpsk(3000, 1200, FS, 15, 1800.0)
        k = 1
    elif kind == "qpsk":
        sig, bits = generate_qpsk(3000, 2400, FS, 15)
        sig = sig * np.exp(2j * np.pi * 1800 * t(len(sig)) / FS)
        k = 2
    else:
        sig, bits = generate_2fsk(1500, 300, FS, 100, 20)
        sig = sig * np.exp(2j * np.pi * 1700 * t(len(sig)) / FS)
        k = 1
    return np.real(sig), bits, k


class TestHilbert:
    def test_fir_properties(self):
        h = rc.hilbert_fir()
        assert len(h) % 2 == 1
        assert np.allclose(h, -h[::-1])             # antisymmetric (type III)
        offsets = np.arange(len(h)) - (len(h) - 1) // 2
        assert np.all(h[offsets % 2 == 0] == 0)     # even offsets from the centre are zero
        with pytest.raises(ValueError):
            rc.hilbert_fir(254)

    @pytest.mark.parametrize("f", [0.02, 0.1, 0.25, 0.4, 0.48])
    def test_image_rejection_and_flat_gain(self, f):
        n = 1 << 14
        z = rc.analytic_signal(np.cos(2 * np.pi * f * np.arange(n)))[1000:-1000]
        spec = np.abs(np.fft.fft(z * np.hanning(len(z))))
        freqs = np.fft.fftfreq(len(z))
        pos = spec[np.argmin(np.abs(freqs - f))]
        neg = spec[np.argmin(np.abs(freqs + f))]
        assert 20 * np.log10(pos / neg) > 60
        assert np.mean(np.abs(z)) == pytest.approx(1.0, abs=0.005)

    def test_chunked_equals_full(self):
        x = np.random.default_rng(0).standard_normal(10_000)
        full = rc.analytic_signal(x)
        d = rc.hilbert_margin()
        parts = [rc.analytic_from_padded(rc.padded_slice(x, s, min(s + 777, len(x)), d))
                 for s in range(0, len(x), 777)]
        assert np.array_equal(np.concatenate(parts), full)

    def test_padded_slice_edges(self):
        x = np.arange(10)
        assert rc.padded_slice(x, 0, 3, 2).tolist() == [0, 0, 0, 1, 2, 3, 4]
        assert rc.padded_slice(x, 8, 10, 2).tolist() == [6, 7, 8, 9, 0, 0]


class TestDiscriminator:
    def test_remodulated_frequency_follows_audio(self):
        audio = np.repeat([1.0, -1.0, 1.0, -1.0], 2000)
        z = rc.fm_remodulate(audio)
        inst = np.diff(np.unwrap(np.angle(z))) / (2 * np.pi)
        assert np.allclose(np.abs(z), 1.0)
        assert inst[500] == pytest.approx(rc.DISCRIMINATOR_PEAK_DEVIATION, rel=1e-3)
        assert inst[2500] == pytest.approx(-rc.DISCRIMINATOR_PEAK_DEVIATION, rel=1e-3)

    def test_silence(self):
        assert np.all(rc.discriminator_phase(np.zeros(100)) == 0)


class TestSuggestion:
    def test_mono(self):
        s = rc.suggest_wav_interpretation(np.random.default_rng(0).standard_normal(5000))
        assert s.interpretation == WavInterpretation.REAL

    def test_duplicated_mono(self):
        x = np.random.default_rng(0).standard_normal(5000)
        s = rc.suggest_wav_interpretation(np.column_stack([x, x]))
        assert s.interpretation == WavInterpretation.REAL and "identical" in s.reason

    def test_silent_second_channel(self):
        x = np.random.default_rng(0).standard_normal(5000)
        s = rc.suggest_wav_interpretation(np.column_stack([x, np.zeros_like(x)]))
        assert s.interpretation == WavInterpretation.REAL

    def test_iq_offset_tone_signal(self):
        z, _ = generate_bpsk(2000, 1200, FS, 20, 3000.0)
        s = rc.suggest_wav_interpretation(np.column_stack([z.real, z.imag]), FS)
        assert s.interpretation == WavInterpretation.STEREO_IQ

    def test_iq_baseband_noise_like(self):
        z, _ = generate_qpsk(2000, 4800, FS, 20)
        s = rc.suggest_wav_interpretation(np.column_stack([z.real, z.imag]), FS)
        assert s.interpretation == WavInterpretation.STEREO_IQ

    def test_dual_sensor(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal(20000)
        b = 0.9 * a + 0.3 * rng.standard_normal(20000)
        s = rc.suggest_wav_interpretation(np.column_stack([a, b]), FS)
        assert s.interpretation == WavInterpretation.DUAL_CHANNEL


class TestWavModes:
    def test_chunked_reads_match(self, tmp_path):
        audio, _, _ = _audio_modem("bpsk")
        r = WavReader(_write_wav(tmp_path / "m.wav", audio))
        full = r.read_samples()
        parts = [r.read_samples(s, 5000) for s in range(0, r.total_samples(), 5000)]
        assert np.array_equal(np.concatenate(parts), full)

    def test_channel_selection_and_swap(self, tmp_path):
        z, _ = generate_bpsk(1000, 1200, FS, 20, 3000.0)
        path = _write_wav(tmp_path / "s.wav", np.column_stack([z.real, z.imag]))
        iq = WavReader(path, WavInterpretation.STEREO_IQ).read_samples()
        qi = WavReader(path, WavInterpretation.STEREO_IQ, swap_iq=True).read_samples()
        assert np.allclose(qi, iq.imag + 1j * iq.real)
        right = WavReader(path, WavInterpretation.DUAL_CHANNEL, channel=1).read_samples()
        assert np.allclose(right.real, iq.imag, atol=1e-6)
        with pytest.raises(ValueError):
            WavReader(path, WavInterpretation.REAL, channel=2).read_samples()

    def test_discriminator_mode(self, tmp_path):
        audio = np.repeat(np.tile([1.0, -1.0], 20), 400)
        r = WavReader(_write_wav(tmp_path / "d.wav", audio), WavInterpretation.DISCRIMINATOR)
        z = r.read_samples()
        assert np.allclose(np.abs(z), 1.0, atol=1e-5)
        assert np.array_equal(r.read_samples(1000, 500), z[1000:1500])

    def test_validation_no_false_iq_imbalance(self, tmp_path):
        audio, _, _ = _audio_modem("bpsk")
        report = WavReader(_write_wav(tmp_path / "m.wav", audio)).validate()
        assert abs(report.iq_imbalance_db) < 0.5
        assert not any("imbalance" in w for w in report.warnings)

    def test_reader_suggestion(self, tmp_path):
        audio, _, _ = _audio_modem("bpsk")
        r = WavReader(_write_wav(tmp_path / "m.wav", audio))
        assert r.suggest_interpretation().interpretation == WavInterpretation.REAL


class TestOneSidedMeasurements:
    """Noise-floor estimates must ignore the empty half of an analytic signal."""

    def test_snr_bandwidth_offset(self):
        np.random.seed(0)
        z, _ = generate_bpsk(4000, 1200, FS, None, 1800.0)
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(len(z)) * np.sqrt(np.mean(np.abs(z) ** 2) / 2 / 10 ** 1.5)
        a = rc.analytic_signal(np.real(z) + noise)            # real recording, 15 dB
        snr_in, bw = estimate_snr_inband(a, FS)
        assert 1200 < bw < 2500                                # ≈ 1.35 × 1200 Bd
        assert estimate_frequency_offset(a, FS) == pytest.approx(1800, abs=40)
        assert estimate_snr_psd(a, FS) == pytest.approx(15, abs=1.5)
        assert snr_in > 15


class TestEndToEnd:
    @pytest.mark.parametrize("kind,mod,rate", [
        ("bpsk", "BPSK", 1200), ("qpsk", "QPSK", 2400), ("fsk", "2-FSK", 300),
    ])
    def test_mono_audio_modem_decodes(self, tmp_path, kind, mod, rate):
        audio, bits, k = _audio_modem(kind)
        reader = WavReader(_write_wav(tmp_path / f"{kind}.wav", audio))
        res = AnalysisPipeline().run(reader.read_samples(), reader.read_metadata())
        assert res.classification.modulation.value == mod
        assert res.analysis.symbol_rate_hz == pytest.approx(rate, rel=0.01)
        assert ber_with_ambiguity(res.demod.bits, bits, k)[0] < 1e-3


def test_fsk_demod_with_audio_carrier():
    """Regression: FSK used to alias when resampled below its carrier."""
    from src.dsp.demod import demodulate_fsk

    np.random.seed(5)
    sig, bits = generate_2fsk(1500, 300, FS, 85, 20)
    sig = (sig * np.exp(2j * np.pi * 2125 * np.arange(len(sig)) / FS)).astype(np.complex64)
    res = demodulate_fsk(sig, FS, 300)
    assert ber_with_ambiguity(res.bits, bits, 1)[0] == 0.0
    assert res.fsk_levels_hz[0] == pytest.approx(2040, abs=30)
    assert res.fsk_levels_hz[1] == pytest.approx(2210, abs=30)
    assert RecordingMetadata  # imported for API parity with other tests
