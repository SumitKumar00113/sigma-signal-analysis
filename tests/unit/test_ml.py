"""Tests for the learned modulation classifier (src/ml)."""

import wave

import numpy as np
import pytest

pytest.importorskip("sklearn")

from src.core.enums import ModulationType as M  # noqa: E402
from src.core.models import RecordingMetadata  # noqa: E402
from src.dsp import synth  # noqa: E402
from src.dsp.classification import ClassificationResult  # noqa: E402
from src.dsp.pipeline import AnalysisPipeline, PipelineConfig  # noqa: E402
from src.ml import dataset as ds  # noqa: E402
from src.ml.features import FEATURE_NAMES, N_FEATURES, extract_features  # noqa: E402
from src.ml.hybrid import combine  # noqa: E402
from src.ml.model import (  # noqa: E402
    ModulationModel,
    Prediction,
    default_model_path,
    load_default_model,
    ml_available,
    train_model,
)


@pytest.fixture(scope="module")
def tiny_model():
    x, y, _ = ds.build_synthetic(per_class=6, snr_range=(10, 20), seed=1, workers=1)
    return train_model(x, y, ds.CLASSES, seed=1)


def _write_wav(path, x, fs):
    x = np.real(np.asarray(x))
    x = x / np.max(np.abs(x)) * 0.8
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes((x * 32767).astype("<i2").tobytes())


class TestFeatures:
    def test_shape_and_finite(self):
        np.random.seed(0)
        x, _ = synth.generate_bpsk(2000, 2400, 48000, 15, 1500)
        f = ds.features_for(x, 48000)
        assert f.shape == (N_FEATURES,) and len(FEATURE_NAMES) == N_FEATURES
        assert np.all(np.isfinite(f))

    def test_features_discriminate(self):
        np.random.seed(0)
        am, _ = synth.generate_am(48000, 48000, 0.6, None, 20, 1500)
        bpsk, _ = synth.generate_bpsk(2000, 2400, 48000, 20, 1500)
        i = FEATURE_NAMES.index("carrier_fraction")
        assert ds.features_for(am, 48000)[i] > 0.5 > ds.features_for(bpsk, 48000)[i]

    def test_short_input(self):
        assert np.all(extract_features(np.zeros(10, np.complex64), 1e3, 0, 0, 0) == 0)


class TestDataset:
    @pytest.mark.parametrize("label", ds.CLASSES)
    def test_every_class_generates(self, label):
        ex = ds.synthetic_example(label, 10.0, np.random.default_rng(3))
        assert ex.label == label and len(ex.samples) > 4000 and ex.sample_rate > 0

    def test_labels(self):
        assert ds.parse_modulation("8-PSK") == M.PSK8
        assert ds.parse_modulation("psk8") == M.PSK8
        assert ds.canonical_label(M.DPSK) == M.BPSK
        with pytest.raises(ValueError):
            ds.parse_modulation("no-such-thing")

    def test_manifest(self, tmp_path):
        np.random.seed(2)
        fs = 48000
        _write_wav(tmp_path / "bpsk.wav", synth.generate_bpsk(4000, 1200, fs, 20, 1800)[0], fs)
        _write_wav(tmp_path / "am.wav", synth.generate_am(fs * 2, fs, 0.7, None, 20, 3000)[0], fs)
        np.random.seed(3)
        iq, _ = synth.generate_qam(6000, 4800, fs, 16, 20, 2000)
        iq.astype(np.complex64).tofile(tmp_path / "qam.cf32")
        (tmp_path / "labels.csv").write_text(
            "path,modulation,sample_rate,datatype,wav_interpretation\n"
            "bpsk.wav,BPSK,,,real\n"
            "am.wav,AM,,,\n"
            "qam.cf32,16-QAM,48000,cf32_le,\n", encoding="utf-8")
        x, y = ds.build_from_manifest(tmp_path / "labels.csv", window_seconds=0.5)
        assert x.shape[1] == N_FEATURES and len(y) == len(x) >= 5
        assert set(ds.CLASSES[i] for i in y) == {M.BPSK, M.AM, M.QAM16}

    def test_raw_iq_needs_rate(self, tmp_path):
        np.zeros(100, np.complex64).tofile(tmp_path / "a.cf32")
        (tmp_path / "m.csv").write_text("path,modulation\na.cf32,BPSK\n", encoding="utf-8")
        with pytest.raises(ValueError, match="sample_rate"):
            ds.build_from_manifest(tmp_path / "m.csv")


class TestModel:
    def test_predict_and_roundtrip(self, tiny_model, tmp_path):
        np.random.seed(4)
        x, _ = synth.generate_mfsk(2000, 2400, 48000, 2, 1.0, 20, 1000)
        f = ds.features_for(x, 48000)
        p = tiny_model.predict_proba(f)
        assert p.shape == (1, len(ds.CLASSES)) and np.isclose(p.sum(), 1.0)
        path = tiny_model.save(tmp_path / "m.joblib")
        loaded = ModulationModel.load(path)
        assert np.allclose(loaded.predict_proba(f), p)
        assert loaded.classes == ds.CLASSES and "sklearn_version" in loaded.info

    def test_learns_easy_classes(self, tiny_model):
        rng = np.random.default_rng(99)
        hits = 0
        for label in (M.AM, M.FM, M.BPSK, M.FSK2, M.SSB_USB):
            ex = ds.synthetic_example(label, 18.0, rng)
            hits += tiny_model.predict(ds.features_for(ex.samples, ex.sample_rate)).modulation \
                == label
        assert hits >= 4

    def test_rejects_other_feature_set(self, tiny_model, tmp_path):
        import joblib

        path = tiny_model.save(tmp_path / "m.joblib")
        data = joblib.load(path)
        data["feature_names"] = ["x"] * 3
        joblib.dump(data, path)
        with pytest.raises(ValueError, match="different feature set"):
            ModulationModel.load(path)

    def test_default_lookup(self, tiny_model, tmp_path, monkeypatch):
        path = tiny_model.save(tmp_path / "env.joblib")
        monkeypatch.setenv("SIGMA_MODEL", str(path))
        assert ml_available() and default_model_path() == path
        assert load_default_model() is not None

    def test_broken_model_never_breaks_analysis(self, tmp_path, monkeypatch):
        bad = tmp_path / "bad.joblib"
        bad.write_bytes(b"not a model")
        monkeypatch.setenv("SIGMA_MODEL", str(bad))
        assert load_default_model() is None


def _rule(mod, conf=0.8, evidence=("rules",)):
    return ClassificationResult(mod, conf, [(mod, conf)], {}, list(evidence))


def _pred(ranked):
    return Prediction(ranked[0][0], ranked[0][1], ranked)


class TestHybrid:
    def test_rules_mode_ignores_model(self):
        d = combine(_rule(M.QPSK), _pred([(M.PSK8, 0.99), (M.QPSK, 0.01)]), "rules")
        assert d.modulation == M.QPSK and d.source == "rules"

    def test_ml_mode(self):
        d = combine(_rule(M.QPSK), _pred([(M.PSK8, 0.6), (M.QPSK, 0.4)]), "ml")
        assert d.modulation == M.PSK8 and d.source == "model"

    def test_agreement_raises_confidence(self):
        d = combine(_rule(M.QPSK, 0.5), _pred([(M.QPSK, 0.95)]), "hybrid")
        assert d.modulation == M.QPSK and d.confidence == 0.95 and d.source == "rules+model"

    def test_aliases_count_as_agreement(self):
        d = combine(_rule(M.DPSK), _pred([(M.BPSK, 0.9)]), "hybrid")
        assert d.source == "rules+model"

    def test_unknown_taken_over(self):
        d = combine(_rule(M.UNKNOWN, 0.0), _pred([(M.GMSK, 0.8), (M.FM, 0.2)]), "hybrid")
        assert d.modulation == M.GMSK and d.source == "model"

    def test_unmodulated_carrier_kept(self):
        rule = _rule(M.UNKNOWN, 0.0, ["Unmodulated carrier: 99% of the power…"])
        d = combine(rule, _pred([(M.AM, 0.9)]), "hybrid")
        assert d.modulation == M.UNKNOWN

    def test_confident_override(self):
        d = combine(_rule(M.FM), _pred([(M.GMSK, 0.9), (M.FM, 0.02)]), "hybrid")
        assert d.modulation == M.GMSK and "overrides" in d.evidence[0]

    def test_weak_disagreement_keeps_rules(self):
        d = combine(_rule(M.QAM16), _pred([(M.QAM64, 0.7), (M.QAM16, 0.3)]), "hybrid")
        assert d.modulation == M.QAM16 and "kept the rule-based answer" in d.evidence[-1]

    def test_bad_mode(self):
        with pytest.raises(ValueError):
            combine(_rule(M.QPSK), None, "magic")


class TestPipelineIntegration:
    def _signal(self):
        np.random.seed(8)
        return synth.generate_am(48000, 48000.0, 0.6, None, 15, 1500)[0]

    def test_rules_mode(self, tiny_model):
        cfg = PipelineConfig(classifier_mode="rules", model=tiny_model)
        res = AnalysisPipeline(cfg).run(self._signal(), RecordingMetadata(sample_rate_hz=48000))
        assert res.model_prediction is None and res.classifier_source == "rules"

    def test_ml_mode(self, tiny_model):
        cfg = PipelineConfig(classifier_mode="ml", model=tiny_model)
        res = AnalysisPipeline(cfg).run(self._signal(), RecordingMetadata(sample_rate_hz=48000))
        assert res.model_prediction is not None and res.classifier_source == "model"
        assert res.analysis.modulation == res.model_prediction.modulation
        assert any("Decided by: model" in e for p in res.analysis.parameters
                   if p.parameter == "modulation" for e in p.evidence)

    def test_hybrid_mode(self, tiny_model):
        cfg = PipelineConfig(classifier_mode="hybrid", model=tiny_model)
        res = AnalysisPipeline(cfg).run(self._signal(), RecordingMetadata(sample_rate_hz=48000))
        assert res.analysis.modulation == M.AM
        assert res.classifier_source in ("rules", "rules+model")


def test_run_training_end_to_end(tmp_path):
    from src.ml.train import run_training

    np.random.seed(2)
    _write_wav(tmp_path / "fm.wav", synth.generate_fm(96000, 48000, 3000, None, 20, 2000)[0],
               48000)
    (tmp_path / "labels.csv").write_text("path,modulation\nfm.wav,FM\n", encoding="utf-8")
    progress = []
    res = run_training(per_class=4, manifest=tmp_path / "labels.csv", workers=1,
                       out=tmp_path / "model.joblib",
                       progress_cb=lambda f, m: progress.append(f))
    assert res.path.is_file() and res.n_examples == 4 * len(ds.CLASSES) + 4
    assert progress and progress[-1] == 1.0
    assert ModulationModel.load(res.path).info["manifest"].endswith("labels.csv")
