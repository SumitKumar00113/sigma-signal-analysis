"""Analysis pipeline orchestrator.

Coordinates validation, preprocessing, detection, measurement,
modulation classification, and demodulation:

    validate → preprocess → detect regions → measure (SNR, CFO, symbol
    rate) → classify modulation → demodulate → bits

Each stage is optional via :class:`PipelineConfig`, and each stage's
output is kept on the :class:`PipelineResult` so the GUI can show the
evidence behind every number.
"""

from __future__ import annotations

import contextlib
import dataclasses
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.core.enums import ConfidenceLevel, ModulationType, ParameterStatus
from src.core.exceptions import ProcessingError, SigmaError
from src.core.models import (
    AnalysisResult,
    FileValidationReport,
    ParameterEstimate,
    RecordingMetadata,
    SignalRegion,
)
from src.dsp.bursts import Burst, BurstConfig, BurstDetection, detect_bursts, extract_burst
from src.dsp.classification import ClassificationResult, classify_modulation
from src.dsp.demod import ANALOG_MODULATIONS, DemodResult, demodulate
from src.dsp.detection import DetectionConfig
from src.dsp.measurements import (
    estimate_frequency_offset,
    estimate_snr_inband,
    estimate_snr_psd,
    estimate_symbol_rate_candidates,
)
from src.dsp.preprocessing import normalize_peak, remove_dc_mean
from src.dsp.spectral import SpectralAnalysis, analyze_spectrum
from src.ingestion.validator import validate_samples

_FSK_FAMILY = {ModulationType.FSK2, ModulationType.FSK4, ModulationType.MSK,
               ModulationType.GMSK, ModulationType.GFSK}
_ASK_FAMILY = {ModulationType.ASK, ModulationType.OOK}


def _rate_for_family(x: np.ndarray, fs: float, mod: ModulationType, current: float) -> float:
    """Best symbol rate for a known modulation family.

    Used when the learned model, not the rules, chose the modulation: the
    generic estimate may have locked onto something else.  Candidates from
    the family's own estimator (plus the current value) are scored by the
    structure the family shows at its symbol centres.
    """
    from src.dsp.demod import envelope_symbols, fit_levels, fsk_front_end, oqpsk_rails, qpsk_fit

    if mod in _FSK_FAMILY:
        method = "instfreq"
    elif mod in _ASK_FAMILY:
        method = "transitions"
    else:
        method = "auto"
    rates = [current] + [r for r, _ in estimate_symbol_rate_candidates(x, fs, method)[:3]]
    best, best_score = current, -np.inf
    for r in dict.fromkeys(round(v, 1) for v in rates if v > 0):
        try:
            if mod in _FSK_FAMILY:
                score = fit_levels(fsk_front_end(x, fs, r).freq_symbols_hz).separation
            elif mod in _ASK_FAMILY:
                score = fit_levels(envelope_symbols(x, fs, r)[0]).separation
            else:
                off, aligned, *_ = oqpsk_rails(x, fs, r)
                score = max(qpsk_fit(off), qpsk_fit(aligned))
        except SigmaError:
            continue
        if score > best_score:
            best, best_score = r, score
    return float(best)


def element_rate_note(bits: np.ndarray, symbol_rate: float,
                      max_single: float = 0.03) -> str | None:
    """Warn when no element is shorter than two symbols.

    Random data has about half its runs one symbol long.  If almost none
    are, the transitions sit on a grid finer than the signalling element –
    typically asynchronous teleprinter (RTTY/Baudot) whose 1.5-element stop
    pulse puts transitions at half-element positions – and the modulation
    rate is half the grid rate that was measured.
    """
    b = np.asarray(bits).reshape(-1)
    if len(b) < 2000 or symbol_rate <= 0:
        return None
    edges = np.flatnonzero(np.diff(b.astype(np.int8)) != 0)
    runs = np.diff(edges)
    if len(runs) < 200:
        return None
    single = float(np.mean(runs == 1))
    if single > max_single:
        return None
    return (f"Only {single:.1%} of the runs are one symbol long (random data: ≈ 50 %): the "
            f"signalling element is 2 symbols, i.e. the modulation rate is about "
            f"{symbol_rate / 2:,.2f} Bd – typical of asynchronous RTTY with 1.5-element stop "
            f"pulses. The bits were sampled on the {symbol_rate:,.1f} Bd half-element grid.")


@dataclass
class PipelineConfig:
    """Configuration for the analysis pipeline."""

    remove_dc: bool = True
    normalize: bool = True
    detect_regions: bool = True
    measure_snr: bool = True
    measure_freq_offset: bool = True
    measure_symbol_rate: bool = True
    classify: bool = True
    demodulate: bool = True

    # Overrides: when set, skip the estimate and use the analyst's value
    modulation_override: ModulationType | None = None
    symbol_rate_override: float | None = None

    # Cap on samples used for the heavy stages (classification/demod)
    max_demod_samples: int = 2_000_000

    # Modulation classifier: "rules" (explainable decision tree), "ml"
    # (learned model only) or "hybrid" (rules, corrected by the model where
    # it is much surer).  The model is loaded from the default locations
    # (see src/ml/model.py) unless given; without one, rules are used.
    classifier_mode: str = "hybrid"
    model: Any = None

    detection_config: DetectionConfig = field(default_factory=DetectionConfig)

    # Bursts / several signals: detected on the spectrogram; if the
    # recording is intermittent each burst is extracted (own band, own
    # time span) and analysed on its own, the strongest giving the main result
    burst_config: BurstConfig = field(default_factory=BurstConfig)
    analyze_bursts: bool = True
    max_burst_analyses: int = 16


@dataclass
class BurstAnalysis:
    """One detected burst and the full analysis of it on its own."""

    index: int                  # position in ``PipelineResult.regions``
    burst: Burst
    region: SignalRegion
    result: PipelineResult      # analysis of the extracted burst
    offset_hz: float            # the burst was moved down by this much
    start_sample: int           # of the extracted segment, in the recording

    @property
    def modulation(self) -> ModulationType:
        return self.result.analysis.modulation


@dataclass
class PipelineResult:
    """Wrapper for all outputs from a pipeline run."""

    metadata: RecordingMetadata
    validation: FileValidationReport
    regions: list[SignalRegion]
    analysis: AnalysisResult

    processing_time_ms: float
    samples_processed: int

    spectral: SpectralAnalysis | None = None
    snr_db: float = 0.0
    snr_inband_db: float = 0.0
    occupied_bandwidth_hz: float = 0.0
    frequency_offset_hz: float = 0.0
    symbol_rate_candidates: list[tuple[float, float]] = field(default_factory=list)
    classification: ClassificationResult | None = None
    model_prediction: Any = None           # src.ml.model.Prediction, when a model ran
    classifier_source: str = "rules"       # which classifier made the final call
    demod: DemodResult | None = None
    stage_errors: dict[str, str] = field(default_factory=dict)
    burst_detection: BurstDetection | None = None
    bursts: list[BurstAnalysis] = field(default_factory=list)
    primary_burst: int | None = None        # index into ``bursts`` shown as the main result


class AnalysisPipeline:
    """Orchestrates signal analysis from raw samples to bits."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

        # Callback for progress updates: (progress_fraction, status_message)
        self.on_progress: Callable[[float, str], None] | None = None

    # ------------------------------------------------------------------

    def run(
        self,
        samples: np.ndarray,
        metadata: RecordingMetadata,
    ) -> PipelineResult:
        """Execute the pipeline on a chunk of samples."""
        start_time = time.perf_counter()
        cfg = self.config

        if len(samples) == 0:
            raise ProcessingError("Cannot process empty sample array.")
        fs = float(metadata.sample_rate_hz) if metadata.sample_rate_hz > 0 else 1.0

        self._emit_progress(0.0, "Validating samples...")
        validation = validate_samples(samples, FileValidationReport())

        # 1. Preprocessing
        self._emit_progress(0.05, "Preprocessing...")
        processed = samples
        if cfg.remove_dc:
            processed = remove_dc_mean(processed)
        if cfg.normalize:
            processed = normalize_peak(processed)

        analysis = AnalysisResult(
            modulation=ModulationType.UNKNOWN,
            overall_confidence=ConfidenceLevel.UNKNOWN,
        )
        result = PipelineResult(
            metadata=metadata,
            validation=validation,
            regions=[],
            analysis=analysis,
            processing_time_ms=0.0,
            samples_processed=len(samples),
        )

        # 2. Spectral overview + region detection
        self._emit_progress(0.15, "Spectral analysis...")
        try:
            result.spectral = analyze_spectrum(processed, fs)
            result.occupied_bandwidth_hz = result.spectral.occupied_bandwidth_hz
        except Exception as exc:  # noqa: BLE001
            result.stage_errors["spectral"] = str(exc)

        if cfg.detect_regions:
            self._emit_progress(0.25, "Detecting signal regions...")
            try:
                det = detect_bursts(processed, fs, cfg.burst_config)
                result.burst_detection = det
                result.regions = [b.to_region(fs, metadata.center_frequency_hz,
                                              metadata.recording_id) for b in det.bursts]
            except Exception as exc:  # noqa: BLE001
                result.stage_errors["detection"] = str(exc)
            det = result.burst_detection
            if cfg.analyze_bursts and det is not None and det.intermittent:
                self._analyze_bursts(processed, metadata, fs, result)
                analysis.overall_confidence = self._overall_confidence(result)
                self._emit_progress(1.0, "Analysis complete.")
                result.processing_time_ms = (time.perf_counter() - start_time) * 1000.0
                return result

        # 3. Measurements
        self._emit_progress(0.35, "Measuring signal parameters...")
        if cfg.measure_snr:
            try:
                result.snr_db = estimate_snr_psd(processed, fs)
                result.snr_inband_db, bw = estimate_snr_inband(processed, fs)
                if bw > 0:
                    result.occupied_bandwidth_hz = bw
                analysis.parameters.append(ParameterEstimate(
                    parameter="snr_db", value=round(result.snr_db, 2),
                    status=ParameterStatus.INFERRED, confidence=0.8,
                    evidence=["PSD noise-floor method (full band)"],
                ))
                for r in result.regions:
                    r.snr_db = result.snr_inband_db
                    r.bandwidth_hz = result.occupied_bandwidth_hz
            except Exception as exc:  # noqa: BLE001
                result.stage_errors["snr"] = str(exc)

        if cfg.measure_freq_offset:
            try:
                result.frequency_offset_hz = estimate_frequency_offset(processed, fs)
                analysis.parameters.append(ParameterEstimate(
                    parameter="carrier_offset_hz", value=round(result.frequency_offset_hz, 1),
                    status=ParameterStatus.INFERRED, confidence=0.8,
                    evidence=["Power-weighted spectral centroid"],
                ))
                if abs(result.frequency_offset_hz) > 1.0:
                    analysis.warnings.append(
                        f"Estimated carrier offset: {result.frequency_offset_hz:,.0f} Hz"
                    )
            except Exception as exc:  # noqa: BLE001
                result.stage_errors["cfo"] = str(exc)

        symbol_rate = cfg.symbol_rate_override or 0.0
        if cfg.measure_symbol_rate and not cfg.symbol_rate_override:
            self._emit_progress(0.45, "Estimating symbol rate...")
            try:
                result.symbol_rate_candidates = estimate_symbol_rate_candidates(processed, fs)
                if result.symbol_rate_candidates:
                    symbol_rate, conf = result.symbol_rate_candidates[0]
                    analysis.symbol_rate_hz = symbol_rate
                    analysis.symbol_rate_confidence = conf
                    analysis.parameters.append(ParameterEstimate(
                        parameter="symbol_rate_hz", value=round(symbol_rate, 1),
                        status=ParameterStatus.INFERRED, confidence=conf,
                        evidence=["Envelope and instantaneous-frequency line spectra"],
                        alternatives=[{"value": r, "confidence": c}
                                      for r, c in result.symbol_rate_candidates[1:]],
                    ))
            except Exception as exc:  # noqa: BLE001
                result.stage_errors["symbol_rate"] = str(exc)
        elif cfg.symbol_rate_override:
            analysis.symbol_rate_hz = symbol_rate
            analysis.symbol_rate_confidence = 1.0
            analysis.parameters.append(ParameterEstimate(
                parameter="symbol_rate_hz", value=symbol_rate,
                status=ParameterStatus.PROVIDED, confidence=1.0,
                evidence=["Analyst override"],
            ))

        # Heavy stages work on a capped slice
        heavy = processed[: cfg.max_demod_samples]

        # Without a detected region or a usable SNR there is nothing to
        # classify; skip the heavy stages rather than guess on noise.
        checks_ran = cfg.detect_regions or cfg.measure_snr
        signal_present = (
            not checks_ran
            or bool(result.regions)
            or result.snr_db >= 3.0
            or bool(cfg.modulation_override)
        )
        if not signal_present:
            analysis.warnings.append("No signal detected above the noise floor.")

        # 4. Modulation classification
        modulation = cfg.modulation_override or ModulationType.UNKNOWN
        if cfg.classify and signal_present and not cfg.modulation_override:
            self._emit_progress(0.55, "Classifying modulation...")
            try:
                modulation, symbol_rate = self._classify(heavy, fs, symbol_rate, result)
            except Exception as exc:  # noqa: BLE001
                result.stage_errors["classification"] = str(exc)
        elif cfg.modulation_override:
            analysis.modulation = modulation
            analysis.modulation_confidence = 1.0
            analysis.parameters.append(ParameterEstimate(
                parameter="modulation", value=modulation.value,
                status=ParameterStatus.PROVIDED, confidence=1.0,
                evidence=["Analyst override"],
            ))

        # 5. Demodulation (analog modulations need no symbol rate)
        analog = modulation in ANALOG_MODULATIONS
        if cfg.demodulate and modulation != ModulationType.UNKNOWN and (symbol_rate > 0 or analog):
            self._emit_progress(0.7, f"Demodulating {modulation.value}...")
            try:
                result.demod = demodulate(
                    heavy, fs, modulation, symbol_rate,
                    coarse_cfo_hz=result.frequency_offset_hz,
                )
                analysis.warnings.extend(result.demod.warnings)
                note = element_rate_note(result.demod.bits, symbol_rate)
                if note and result.demod.bits_per_symbol == 1:
                    analysis.warnings.append(note)
                if analog and result.demod.audio is not None:
                    d = result.demod
                    analysis.parameters.append(ParameterEstimate(
                        parameter="audio_seconds",
                        value=round(len(d.audio) / max(d.audio_rate_hz, 1.0), 2),
                        status=ParameterStatus.INFERRED, confidence=0.9,
                        evidence=[f"{modulation.value} demodulated to audio at "
                                  f"{d.audio_rate_hz:,.0f} Hz"],
                    ))
                else:
                    analysis.parameters.append(ParameterEstimate(
                        parameter="evm_percent", value=round(result.demod.evm_percent, 1),
                        status=ParameterStatus.INFERRED, confidence=0.9,
                        evidence=[f"{result.demod.num_symbols:,} symbols, "
                                  f"{result.demod.num_bits:,} bits recovered"],
                    ))
            except SigmaError as exc:
                result.stage_errors["demodulation"] = str(exc)
            except Exception as exc:  # noqa: BLE001
                result.stage_errors["demodulation"] = f"{type(exc).__name__}: {exc}"

        # 6. Overall confidence
        analysis.overall_confidence = self._overall_confidence(result)
        self._emit_progress(1.0, "Analysis complete.")

        result.processing_time_ms = (time.perf_counter() - start_time) * 1000.0
        return result

    # ------------------------------------------------------------------

    def _analyze_bursts(self, x: np.ndarray, metadata: RecordingMetadata, fs: float,
                        result: PipelineResult) -> None:
        """Analyse each detected burst on its own; the strongest becomes the
        main result."""
        cfg = self.config
        det = result.burst_detection
        assert det is not None
        strength = [b.num_samples * 10 ** (b.snr_db / 10) for b in det.bursts]
        order = sorted(range(len(det.bursts)), key=lambda i: strength[i], reverse=True)
        order = order[: cfg.max_burst_analyses]
        # The extracted burst sits at 0 Hz: DC removal would delete a carrier
        # (OOK, AM); receiver DC was already removed from the whole recording
        sub_cfg = dataclasses.replace(cfg, detect_regions=False, analyze_bursts=False,
                                      remove_dc=False)
        for rank, i in enumerate(order):
            b = det.bursts[i]
            self._emit_progress(0.3 + 0.65 * rank / len(order),
                                f"Analysing burst {rank + 1}/{len(order)} "
                                f"({b.duration(fs) * 1e3:,.0f} ms at {b.center_hz:+,.0f} Hz)")
            ex = extract_burst(x, fs, b)
            if len(ex.samples) < 1024:
                continue
            sub_meta = metadata.model_copy(update={
                "sample_rate_hz": ex.sample_rate,
                "center_frequency_hz": metadata.center_frequency_hz + ex.offset_hz,
                "sample_count": len(ex.samples),
            })
            try:
                sub = AnalysisPipeline(sub_cfg).run(ex.samples, sub_meta)
            except SigmaError as exc:
                result.stage_errors[f"burst {i + 1}"] = str(exc)
                continue
            region = result.regions[i]
            region.label = sub.analysis.modulation.value
            result.bursts.append(BurstAnalysis(i, b, region, sub, ex.offset_hz, ex.start_sample))
        if not result.bursts:
            return
        # Main result: the strongest burst that was analysed
        primary = result.bursts[0]
        result.bursts.sort(key=lambda a: a.index)
        result.primary_burst = result.bursts.index(primary)
        self.adopt_burst(result, primary)
        n = len(det.bursts)
        kinds = sorted({a.modulation.value for a in result.bursts})
        result.analysis.warnings.insert(
            0, f"{n} burst(s)/signal(s) detected (active {det.duty_cycle:.0%} of the time; "
               f"{', '.join(kinds)}). Showing burst {primary.index + 1} "
               f"({primary.region.start_time_sec:.3f}–{primary.region.end_time_sec:.3f} s, "
               f"{primary.offset_hz:+,.0f} Hz); pick a region to see another.")

    @staticmethod
    def burst_train(result: PipelineResult, burst: BurstAnalysis | None = None
                    ) -> list[BurstAnalysis]:
        """Bursts from the same transmitter as *burst* (default: the shown
        one), in time order: same modulation, carrier within a quarter of
        its bandwidth, symbol rate within 1 %.  A radiosonde or a packet
        radio sends one frame per burst; their bits belong together."""
        if not result.bursts:
            return []
        ref = burst or result.bursts[result.primary_burst or 0]
        if ref.result.demod is None:
            return [ref]
        rs = ref.result.analysis.symbol_rate_hz
        tol_hz = max(ref.burst.bandwidth_hz / 4, 1.0)

        def same(b: BurstAnalysis) -> bool:
            d = b.result.demod
            return (d is not None and d.audio is None and b.modulation == ref.modulation
                    and abs(b.burst.center_hz - ref.burst.center_hz) <= tol_hz
                    and abs(b.result.analysis.symbol_rate_hz - rs) <= 0.01 * max(rs, 1.0))
        return sorted([b for b in result.bursts if b is ref or same(b)],
                      key=lambda b: b.start_sample)

    @staticmethod
    def train_bits(result: PipelineResult) -> tuple[np.ndarray, int]:
        """Demodulated bits of the shown signal – all bursts of its train
        joined in time order – and the number of bursts joined."""
        d = result.demod
        if d is None:
            return np.zeros(0, dtype=np.uint8), 0
        train = AnalysisPipeline.burst_train(result) if result.bursts else []
        if len(train) <= 1:
            return d.bits, 1
        return np.concatenate([b.result.demod.bits for b in train]).astype(np.uint8), len(train)

    @staticmethod
    def adopt_burst(result: PipelineResult, burst: BurstAnalysis) -> None:
        """Make *burst*'s analysis the main result (frequencies back in the
        recording's frame).  Regions, bursts and the spectral overview stay."""
        sub = burst.result
        result.analysis = sub.analysis.model_copy(deep=True)
        result.classification = sub.classification
        result.model_prediction = sub.model_prediction
        result.classifier_source = sub.classifier_source
        result.demod = sub.demod
        result.snr_db = sub.snr_db
        result.snr_inband_db = sub.snr_inband_db
        result.occupied_bandwidth_hz = sub.occupied_bandwidth_hz
        result.symbol_rate_candidates = sub.symbol_rate_candidates
        result.frequency_offset_hz = sub.frequency_offset_hz + burst.offset_hz
        for stage, err in sub.stage_errors.items():
            result.stage_errors[f"burst {burst.index + 1} {stage}"] = err
        if burst in result.bursts:
            result.primary_burst = result.bursts.index(burst)

    def _classify(self, heavy: np.ndarray, fs: float, symbol_rate: float,
                  result: PipelineResult) -> tuple[ModulationType, float]:
        """Rule-based classification, optionally combined with the learned
        model.  Returns the modulation and the (possibly revised) symbol rate."""
        cfg = self.config
        analysis = result.analysis
        cfo_in = result.frequency_offset_hz if cfg.measure_freq_offset else None
        rate_conf = 1.0 if cfg.symbol_rate_override else analysis.symbol_rate_confidence
        # Analog modulations have no symbol rate, so classify even without
        # one and let the classifier weigh its confidence
        rule = classify_modulation(
            heavy, fs, symbol_rate, snr_db=result.snr_db, cfo_hz=cfo_in,
            symbol_rate_confidence=rate_conf, rate_candidates=result.symbol_rate_candidates,
        )
        result.classification = rule

        decision_mod, confidence = rule.modulation, rule.confidence
        candidates, evidence = list(rule.candidates), list(rule.evidence)
        source = "rules"
        if cfg.classifier_mode != "rules":
            model = cfg.model
            if model is None:
                from src.ml.model import load_default_model

                model = load_default_model()
            if model is not None:
                from src.ml.features import extract_features
                from src.ml.hybrid import combine

                cfo = cfo_in if cfo_in is not None else estimate_frequency_offset(heavy, fs)
                pred = model.predict(extract_features(heavy, fs, symbol_rate, rate_conf, cfo))
                result.model_prediction = pred
                d = combine(rule, pred, cfg.classifier_mode)
                decision_mod, confidence, candidates, source = (d.modulation, d.confidence,
                                                                d.candidates, d.source)
                evidence = (d.evidence + evidence) if d.source == "model" else evidence + d.evidence
        result.classifier_source = source
        if source == "model" and decision_mod not in ANALOG_MODULATIONS \
                and decision_mod != ModulationType.UNKNOWN and not cfg.symbol_rate_override:
            new_rate = _rate_for_family(heavy, fs, decision_mod, symbol_rate)
            if new_rate and abs(new_rate - symbol_rate) > 0.01 * max(symbol_rate, 1.0):
                analysis.warnings.append(
                    f"Symbol rate re-estimated for {decision_mod.value}: {symbol_rate:,.1f} → "
                    f"{new_rate:,.1f} baud.")
                symbol_rate = new_rate
                analysis.symbol_rate_hz = new_rate

        # The rules' refinements (exact carrier, verified rate) belong to the
        # rules' answer; keep them only if that is the final answer
        if decision_mod == rule.modulation:
            if rule.symbol_rate_hz and not cfg.symbol_rate_override:
                analysis.warnings.append(
                    f"Symbol rate revised {symbol_rate:,.1f} → {rule.symbol_rate_hz:,.1f} baud "
                    "(cleaner constellation).")
                symbol_rate = rule.symbol_rate_hz
                analysis.symbol_rate_hz = symbol_rate
            if rule.carrier_hz is not None:
                result.frequency_offset_hz = float(rule.carrier_hz)

        analysis.modulation = decision_mod
        analysis.modulation_confidence = float(confidence)
        analysis.modulation_candidates = [
            {"modulation": m.value, "probability": round(float(p), 3)} for m, p in candidates
        ]
        analysis.parameters.append(ParameterEstimate(
            parameter="modulation", value=decision_mod.value,
            status=ParameterStatus.INFERRED, confidence=float(confidence),
            evidence=evidence + [f"Decided by: {source}"],
        ))
        return decision_mod, symbol_rate

    @staticmethod
    def _overall_confidence(result: PipelineResult) -> ConfidenceLevel:
        a = result.analysis
        if a.modulation == ModulationType.UNKNOWN:
            return ConfidenceLevel.UNKNOWN if result.snr_db < 3 else ConfidenceLevel.UNSUPPORTED
        demod_ok = result.demod is not None and result.demod.evm_percent < 30.0
        if a.modulation_confidence >= 0.8 and demod_ok and result.snr_db >= 8:
            return ConfidenceLevel.CONFIRMED
        if a.modulation_confidence >= 0.6 and (demod_ok or result.snr_db >= 6):
            return ConfidenceLevel.PROBABLE
        return ConfidenceLevel.POSSIBLE

    def _emit_progress(self, fraction: float, message: str) -> None:
        if self.on_progress:
            with contextlib.suppress(Exception):
                self.on_progress(fraction, message)
