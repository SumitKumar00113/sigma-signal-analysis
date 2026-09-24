"""Tests for burst detection, extraction and per-burst analysis."""

import numpy as np
import pytest

from src.core.enums import DetectionMethod
from src.core.enums import ModulationType as M
from src.core.models import RecordingMetadata
from src.dsp import synth
from src.dsp.bursts import BurstConfig, detect_bursts, extract_burst
from src.dsp.measurements import estimate_frequency_offset, mpower_carrier
from src.dsp.pipeline import AnalysisPipeline, PipelineConfig
from tests.signals.metrics import ber_with_ambiguity

FS = 48000.0


def _noise(n, power, seed=0):
    rng = np.random.default_rng(seed)
    return ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) * np.sqrt(power / 2)
            ).astype(np.complex64)


def _three_bursts(noise_power=0.05):
    np.random.seed(0)
    x = _noise(int(2 * FS), noise_power)
    truth = []
    for t0, dur, f0 in [(0.1, 0.3, 3000), (0.7, 0.2, 3000), (1.3, 0.5, -6000)]:
        s, n = int(t0 * FS), int(dur * FS)
        b, _ = synth.generate_bpsk(n // 20, 2400, FS, None, f0)
        x[s:s + len(b)] += b.astype(np.complex64)
        truth.append((s, s + len(b), f0))
    return x, truth


class TestDetection:
    @pytest.mark.parametrize("noise_power", [0.05, 0.5])
    def test_three_bursts(self, noise_power):
        x, truth = _three_bursts(noise_power)
        det = detect_bursts(x, FS)
        assert len(det.bursts) == 3 and det.intermittent and not det.continuous
        for b, (s, e, f0) in zip(det.bursts, truth, strict=True):
            assert abs(b.start_sample - s) < 0.001 * FS          # within 1 ms
            assert abs(b.end_sample - e) < 0.001 * FS
            assert abs(b.center_hz - f0) < 400
            assert 2500 < b.bandwidth_hz < 5000                   # 2400 Bd, roll-off 0.35
        assert 0.4 < det.duty_cycle < 0.65

    def test_continuous_signal(self):
        np.random.seed(1)
        x, _ = synth.generate_2fsk(4000, 1200, FS, 600, 10)
        det = detect_bursts(x, FS)
        assert len(det.bursts) == 1 and det.continuous and not det.intermittent

    def test_noise_only(self):
        false = sum(len(detect_bursts(_noise(int(FS), 1.0, seed), FS).bursts)
                    for seed in range(10))
        assert false <= 1

    def test_simultaneous_signals_separated(self):
        np.random.seed(2)
        x = _noise(int(2 * FS), 0.05)
        o, _ = synth.generate_ask(1000, 1000, FS, (0, 1), None, 2000)   # keyed on/off
        x[4800:4800 + len(o)] += o
        q, _ = synth.generate_mpsk(3000, 2400, FS, 4, None, -8000)
        x[: len(q)] += q
        det = detect_bursts(x, FS)
        assert len(det.bursts) == 2                    # OOK "off" symbols bridged
        centres = sorted(b.center_hz for b in det.bursts)
        assert centres[0] == pytest.approx(-8000, abs=500)
        assert centres[1] == pytest.approx(2000, abs=500)

    def test_fsk_tones_are_one_burst(self):
        np.random.seed(3)
        x = _noise(int(FS), 0.05)
        f, _ = synth.generate_mfsk(150, 300, FS, 2, 3.0, None, 0)       # tones 900 Hz apart
        x[10000:10000 + len(f)] += f
        assert len(detect_bursts(x, FS).bursts) == 1

    def test_wide_fast_fsk_is_one_burst(self):
        """Tones far apart (beyond the frequency-merge distance) alternate
        faster than a spectrogram frame: merged by anti-correlated envelopes."""
        np.random.seed(2)
        fs = 96000.0
        sig, _ = synth.generate_2fsk(3000, 9600, fs, 9600 * 1.5)
        x = synth._finish(sig, fs, 8, 600)
        assert len(detect_bursts(x, fs).bursts) == 1
        res = AnalysisPipeline().run(x, RecordingMetadata(sample_rate_hz=fs))
        assert res.analysis.modulation == M.FSK2

    def test_independent_keyed_carriers_stay_separate(self):
        np.random.seed(3)
        fs = 96000.0
        x = _noise(int(fs), 0.0025, 3)
        a, _ = synth.generate_ask(800, 1000, fs, (0, 1), None, -30000)
        b, _ = synth.generate_ask(800, 1000, fs, (0, 1), None, 30000)
        x[1000:1000 + len(a)] += a
        x[1000:1000 + len(b)] += b
        assert len(detect_bursts(x, fs).bursts) == 2

    def test_min_duration(self):
        x, _ = _three_bursts()
        assert len(detect_bursts(x, FS, BurstConfig(min_duration_s=0.25)).bursts) == 2

    def test_region_conversion(self):
        x, _ = _three_bursts()
        b = detect_bursts(x, FS).bursts[0]
        r = b.to_region(FS, center_frequency_hz=1e6, recording_id="rec")
        assert r.detection_method == DetectionMethod.BURST
        assert r.center_frequency_hz == pytest.approx(1e6 + b.center_hz)
        assert r.start_time_sec == pytest.approx(b.start_sample / FS)


class TestExtraction:
    def test_moves_to_baseband_and_decimates(self):
        x, _ = _three_bursts()
        b = detect_bursts(x, FS).bursts[2]                        # at −6 kHz
        ex = extract_burst(x, FS, b)
        assert ex.sample_rate < FS / 1.5
        assert ex.offset_hz == pytest.approx(b.center_hz)
        assert abs(estimate_frequency_offset(ex.samples, ex.sample_rate)) < 300
        assert len(ex.samples) == pytest.approx(b.num_samples * ex.sample_rate / FS, rel=0.05)


class TestPipeline:
    def test_mixed_bursts_analysed_individually(self):
        np.random.seed(1)
        fs = 96000.0
        n_tot = int(2.5 * fs)
        x = _noise(n_tot, 0.02, seed=1)
        plan = [
            (M.BPSK, 0.10, 2400, 8000,
             lambda n, rs, f0: synth.generate_bpsk(n, rs, fs, None, f0), 1),
            (M.FSK2, 0.60, 1200, -15000,
             lambda n, rs, f0: synth.generate_mfsk(n, rs, fs, 2, 1.0, None, f0), 1),
            (M.QPSK, 0.80, 4800, 20000,
             lambda n, rs, f0: synth.generate_mpsk(n, rs, fs, 4, None, f0), 2),
            (M.OOK, 1.60, 1000, -30000,
             lambda n, rs, f0: synth.generate_ask(n, rs, fs, (0, 1), None, f0), 1),
            (M.QAM16, 1.80, 4800, 5000,
             lambda n, rs, f0: synth.generate_qam16(n, rs, fs, None, f0), 4),
        ]
        truth = []
        for mod, t0, rs, f0, gen, k in plan:
            sig, bits = gen(int(0.5 * rs), rs, f0)
            s = int(t0 * fs)
            x[s:s + len(sig)] += sig[: n_tot - s].astype(np.complex64)
            truth.append((mod, t0, f0, bits, k, rs))
        res = AnalysisPipeline().run(x, RecordingMetadata(sample_rate_hz=fs))
        assert len(res.bursts) == 5 and len(res.regions) == 5
        for b in res.bursts:
            mod, t0, f0, bits, k, rs = min(truth,
                                           key=lambda t: abs(t[1] - b.region.start_time_sec))
            assert b.region.start_time_sec == pytest.approx(t0, abs=0.002)
            assert b.region.center_frequency_hz == pytest.approx(f0, abs=1000)
            assert b.modulation == mod and b.region.label == mod.value
            assert ber_with_ambiguity(b.result.demod.bits, bits, k)[0] < 0.01
            # carrier back in the recording's frame (FSK: centre between tones)
            assert b.result.frequency_offset_hz + b.offset_hz == pytest.approx(f0, abs=0.1 * rs)
        # Main result = one of the bursts, in the recording's frequency frame
        assert res.primary_burst is not None
        primary = res.bursts[res.primary_burst]
        assert res.analysis.modulation == primary.modulation
        assert "burst(s)/signal(s) detected" in res.analysis.warnings[0]
        # Switching the shown burst
        other = next(b for b in res.bursts if b is not primary)
        AnalysisPipeline.adopt_burst(res, other)
        assert res.analysis.modulation == other.modulation
        assert res.bursts[res.primary_burst] is other

    def test_continuous_signal_uses_normal_path(self):
        np.random.seed(4)
        x, _ = synth.generate_bpsk(3000, 2400, FS, 15, 1500)
        res = AnalysisPipeline().run(x, RecordingMetadata(sample_rate_hz=FS))
        assert res.bursts == [] and res.primary_burst is None
        assert len(res.regions) == 1 and res.analysis.modulation == M.BPSK

    def test_burst_analysis_can_be_disabled(self):
        x, _ = _three_bursts()
        res = AnalysisPipeline(PipelineConfig(analyze_bursts=False)).run(
            x, RecordingMetadata(sample_rate_hz=FS))
        assert len(res.regions) == 3 and res.bursts == []


class TestRegressions:
    @pytest.mark.parametrize("f0", [20000.0, -35000.0])
    def test_mpower_carrier_alias(self, f0):
        """4·f_c wraps around the sample rate for large offsets."""
        np.random.seed(5)
        fs = 96000.0
        q, bits = synth.generate_mpsk(2880, 4800, fs, 4, None, f0)
        q = q + _noise(len(q), 0.02, 5)
        assert mpower_carrier(q, fs, 4)[0] == pytest.approx(f0, abs=50)
        res = AnalysisPipeline().run(q, RecordingMetadata(sample_rate_hz=fs))
        assert res.analysis.modulation == M.QPSK
        assert ber_with_ambiguity(res.demod.bits, bits, 2)[0] == 0.0

    def test_qam16_rotation_metric(self):
        """A 90° rotation of Gray 16-QAM swaps I/Q pairs and flips one bit."""
        np.random.seed(6)
        _, bits = synth.generate_qam16(500, 4800, FS)
        g = bits.reshape(-1, 4)
        rotated = np.hstack([g[:, 2:] ^ np.array([1, 0]), g[:, :2]]).reshape(-1)
        assert ber_with_ambiguity(rotated, bits, 4)[0] == 0.0
