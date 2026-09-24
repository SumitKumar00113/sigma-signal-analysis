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
import time
from collections.abc import Callable
from dataclasses import dataclass, field

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
from src.dsp.classification import ClassificationResult, classify_modulation
from src.dsp.demod import ANALOG_MODULATIONS, DemodResult, demodulate
from src.dsp.detection import DetectionConfig, detect_signal_regions
from src.dsp.measurements import (
    estimate_frequency_offset,
    estimate_snr_inband,
    estimate_snr_psd,
    estimate_symbol_rate_candidates,
)
from src.dsp.preprocessing import normalize_peak, remove_dc_mean
from src.dsp.spectral import SpectralAnalysis, analyze_spectrum
from src.ingestion.validator import validate_samples


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

    detection_config: DetectionConfig = field(default_factory=DetectionConfig)


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
    demod: DemodResult | None = None
    stage_errors: dict[str, str] = field(default_factory=dict)


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
                result.regions = detect_signal_regions(processed, fs, cfg.detection_config)
                for r in result.regions:
                    r.recording_id = metadata.recording_id
                    r.start_time_sec = r.start_sample / fs
                    r.end_time_sec = r.end_sample / fs
                    r.center_frequency_hz = metadata.center_frequency_hz
            except Exception as exc:  # noqa: BLE001
                result.stage_errors["detection"] = str(exc)

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
                # Analog modulations have no symbol rate, so classify even
                # without one and let the classifier weigh its confidence
                result.classification = classify_modulation(
                    heavy, fs, symbol_rate,
                    snr_db=result.snr_db,
                    cfo_hz=result.frequency_offset_hz if cfg.measure_freq_offset else None,
                    symbol_rate_confidence=(1.0 if cfg.symbol_rate_override
                                            else analysis.symbol_rate_confidence),
                    rate_candidates=result.symbol_rate_candidates,
                )
                modulation = result.classification.modulation
                c = result.classification
                if c.symbol_rate_hz and not cfg.symbol_rate_override:
                    analysis.warnings.append(
                        f"Symbol rate revised {symbol_rate:,.1f} → {c.symbol_rate_hz:,.1f} baud "
                        "(cleaner constellation).")
                    symbol_rate = c.symbol_rate_hz
                    analysis.symbol_rate_hz = symbol_rate
                if c.carrier_hz is not None:
                    result.frequency_offset_hz = float(c.carrier_hz)
                analysis.modulation = modulation
                analysis.modulation_confidence = result.classification.confidence
                analysis.modulation_candidates = [
                    {"modulation": m.value, "probability": round(p, 3)}
                    for m, p in result.classification.candidates
                ]
                analysis.parameters.append(ParameterEstimate(
                    parameter="modulation", value=modulation.value,
                    status=ParameterStatus.INFERRED,
                    confidence=result.classification.confidence,
                    evidence=list(result.classification.evidence),
                ))
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
