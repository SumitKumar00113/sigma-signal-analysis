"""Enumerations for the Sigma Signal Analysis platform.

Covers sample data types, file formats, modulation families, confidence levels,
parameter provenance status, and job lifecycle states.
"""

from __future__ import annotations

from enum import StrEnum

# ---------------------------------------------------------------------------
# Sample data types (SigMF-aligned naming where applicable)
# ---------------------------------------------------------------------------

class SampleDatatype(StrEnum):
    """Binary sample format codes.

    Naming follows the SigMF convention:
      <complex/real><type><width>_<endian>
    e.g. ``cf32_le`` = complex float-32 little-endian.
    """

    CF32_LE = "cf32_le"
    CF32_BE = "cf32_be"
    CF64_LE = "cf64_le"
    CI8 = "ci8"
    CI16_LE = "ci16_le"
    CI16_BE = "ci16_be"
    CI32_LE = "ci32_le"
    CU8 = "cu8"
    CU16_LE = "cu16_le"
    RF32_LE = "rf32_le"   # real float32
    RF64_LE = "rf64_le"   # real float64
    RI8 = "ri8"
    RI16_LE = "ri16_le"
    RI32_LE = "ri32_le"
    RU8 = "ru8"
    RU16_LE = "ru16_le"


class IQOrder(StrEnum):
    """Interleaved I/Q sample ordering."""

    IQ = "IQ"
    QI = "QI"


class FileFormat(StrEnum):
    """Supported recording file formats."""

    WAV = "wav"
    RAW_IQ = "raw_iq"
    SIGMF = "sigmf"


class WavInterpretation(StrEnum):
    """How a WAV recording should be interpreted."""

    REAL = "real"                    # mono real-valued signal
    STEREO_IQ = "stereo_iq"         # left = I, right = Q
    DISCRIMINATOR = "discriminator" # audio discriminator output
    DUAL_CHANNEL = "dual_channel"   # dual-channel sensor data
    CUSTOM = "custom"               # user-defined channel mapping


# ---------------------------------------------------------------------------
# Modulation taxonomy
# ---------------------------------------------------------------------------

class ModulationType(StrEnum):
    """Supported and recognizable modulation types."""

    # Analog
    AM = "AM"
    FM = "FM"
    PM = "PM"

    # Amplitude-shift keying
    ASK = "ASK"
    OOK = "OOK"

    # Frequency-shift keying
    FSK2 = "2-FSK"
    FSK4 = "4-FSK"
    GFSK = "GFSK"
    MSK = "MSK"
    GMSK = "GMSK"
    CPFSK = "CPFSK"

    # Phase-shift keying
    BPSK = "BPSK"
    QPSK = "QPSK"
    OQPSK = "OQPSK"
    PSK8 = "8-PSK"
    PSK16 = "16-PSK"
    DPSK = "DPSK"

    # Quadrature amplitude modulation
    QAM16 = "16-QAM"
    QAM64 = "64-QAM"
    QAM256 = "256-QAM"

    # Other digital
    APSK = "APSK"
    OFDM = "OFDM"

    # Catch-all
    UNKNOWN = "Unknown"


# ---------------------------------------------------------------------------
# Confidence & provenance
# ---------------------------------------------------------------------------

class ConfidenceLevel(StrEnum):
    """Analyst-facing certainty label for an automated result."""

    CONFIRMED = "Confirmed"
    PROBABLE = "Probable"
    POSSIBLE = "Possible"
    UNSUPPORTED = "Unsupported"
    UNKNOWN = "Unknown"


class ParameterStatus(StrEnum):
    """How a parameter value was obtained."""

    PROVIDED = "provided"       # from file metadata or user entry
    VALIDATED = "validated"     # provided *and* cross-checked
    INFERRED = "inferred"      # estimated from signal features
    ASSUMED = "assumed"         # default assumption, not verified
    UNKNOWN = "unknown"         # could not be determined


# ---------------------------------------------------------------------------
# Processing lifecycle
# ---------------------------------------------------------------------------

class JobStatus(StrEnum):
    """State of a background processing job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# FEC families
# ---------------------------------------------------------------------------

class FECType(StrEnum):
    """Forward error-correction family identifier."""

    NONE = "none"
    CONVOLUTIONAL = "convolutional"
    REED_SOLOMON = "reed_solomon"
    LDPC = "ldpc"
    CONCATENATED = "concatenated"
    TURBO = "turbo"
    BCH = "bch"
    UNKNOWN = "unknown"


class InterleaverType(StrEnum):
    """De-interleaving algorithm family."""

    NONE = "none"
    BLOCK = "block"
    CONVOLUTIONAL = "convolutional"
    DIAGONAL = "diagonal"
    PSEUDO_RANDOM = "pseudo_random"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------

class DetectionMethod(StrEnum):
    """Algorithm used to detect a candidate signal region."""

    ENERGY = "energy"
    THRESHOLD = "threshold"
    SPECTRAL_PEAK = "spectral_peak"
    SPECTRAL_KURTOSIS = "spectral_kurtosis"
    CYCLOSTATIONARY = "cyclostationary"
    MATCHED_FILTER = "matched_filter"
    BURST = "burst"
    CHANGE_POINT = "change_point"
    HYBRID = "energy_spectral_hybrid"
    MANUAL = "manual"
