"""Analysis pipeline orchestrator.

Coordinates ingestion, validation, preprocessing, detection, measurement,
and reporting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.core.enums import ConfidenceLevel, ModulationType
from src.core.exceptions import ProcessingError
from src.core.models import AnalysisResult, FileValidationReport, RecordingMetadata, SignalRegion
from src.dsp.detection import DetectionConfig, detect_signal_regions
from src.dsp.measurements import estimate_frequency_offset, estimate_snr, estimate_symbol_rate_candidates
from src.dsp.preprocessing import remove_dc_mean, normalize_peak
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


class AnalysisPipeline:
    """Orchestrates signal analysis from raw samples to final result."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        
        # Callback for progress updates: (progress_fraction, status_message)
        self.on_progress: Callable[[float, str], None] | None = None

    def run(
        self,
        samples: np.ndarray,
        metadata: RecordingMetadata,
    ) -> PipelineResult:
        """Execute the pipeline on a chunk of samples."""
        start_time = time.perf_counter()
        
        if len(samples) == 0:
            raise ProcessingError("Cannot process empty sample array.")

        self._emit_progress(0.0, "Validating samples...")
        validation = validate_samples(samples, FileValidationReport())
        
        # 1. Preprocessing
        self._emit_progress(0.1, "Preprocessing...")
        processed = samples
        if self.config.remove_dc:
            processed = remove_dc_mean(processed)
        if self.config.normalize:
            processed = normalize_peak(processed)
            
        # 2. Region Detection
        self._emit_progress(0.3, "Detecting signal regions...")
        regions: list[SignalRegion] = []
        if self.config.detect_regions:
            regions = detect_signal_regions(
                processed, metadata.sample_rate_hz, self.config.detection_config
            )
            
        # 3. Measurements
        self._emit_progress(0.6, "Measuring signal parameters...")
        analysis = AnalysisResult(
            modulation=ModulationType.UNKNOWN,
            overall_confidence=ConfidenceLevel.UNKNOWN,
        )
        
        if regions:
            # For now, just analyze the strongest region or whole chunk if one region
            target_samples = processed
            
            if self.config.measure_snr:
                snr = estimate_snr(target_samples)
                if snr > 5.0:
                    analysis.overall_confidence = ConfidenceLevel.POSSIBLE
                    
            if self.config.measure_freq_offset:
                offset = estimate_frequency_offset(target_samples, metadata.sample_rate_hz)
                if abs(offset) > 1.0:
                    analysis.warnings.append(f"Estimated carrier offset: {offset:,.0f} Hz")
                    
            if self.config.measure_symbol_rate:
                candidates = estimate_symbol_rate_candidates(target_samples, metadata.sample_rate_hz)
                if candidates:
                    best_rate, conf = candidates[0]
                    analysis.symbol_rate_hz = best_rate
                    analysis.symbol_rate_confidence = conf
                    
        self._emit_progress(1.0, "Analysis complete.")
        
        elapsed = (time.perf_counter() - start_time) * 1000.0
        
        return PipelineResult(
            metadata=metadata,
            validation=validation,
            regions=regions,
            analysis=analysis,
            processing_time_ms=elapsed,
            samples_processed=len(samples),
        )

    def _emit_progress(self, fraction: float, message: str) -> None:
        if self.on_progress:
            try:
                self.on_progress(fraction, message)
            except Exception:
                pass
