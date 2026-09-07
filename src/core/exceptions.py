"""Custom exception hierarchy for the Sigma Signal Analysis platform.

Every exception carries enough context to produce a user-facing diagnostic
without exposing internal stack details.
"""

from __future__ import annotations


class SigmaError(Exception):
    """Root exception for the entire application."""


# ---------------------------------------------------------------------------
# File / ingestion errors
# ---------------------------------------------------------------------------

class FileError(SigmaError):
    """Base class for file-related errors."""


class FileNotFoundSigmaError(FileError):
    """Recording file does not exist or is inaccessible."""


class FileFormatError(FileError):
    """File header or structure does not match the expected format."""


class FileTruncatedError(FileError):
    """File appears to be cut short (sample count doesn't match header)."""


class FilePermissionSigmaError(FileError):
    """Insufficient permissions to read the recording file."""


class UnsupportedFormatError(FileError):
    """The sample format, encoding, or file type is not supported."""


# ---------------------------------------------------------------------------
# Metadata errors
# ---------------------------------------------------------------------------

class MetadataError(SigmaError):
    """Base class for metadata-related errors."""


class MissingMetadataError(MetadataError):
    """A required metadata field is missing and cannot be inferred."""


class InvalidMetadataError(MetadataError):
    """A metadata field has an invalid or contradictory value."""


# ---------------------------------------------------------------------------
# Processing / DSP errors
# ---------------------------------------------------------------------------

class ProcessingError(SigmaError):
    """Base class for signal processing errors."""


class PipelineConfigError(ProcessingError):
    """Pipeline or profile configuration is invalid."""


class InsufficientSamplesError(ProcessingError):
    """Not enough samples to perform the requested operation."""


class SynchronizationError(ProcessingError):
    """Carrier, timing, or frame synchronization failed."""


class DemodulationError(ProcessingError):
    """Demodulator could not produce output for the given signal."""


class DecodingError(ProcessingError):
    """FEC or de-interleaving decoder could not converge."""


# ---------------------------------------------------------------------------
# ML errors
# ---------------------------------------------------------------------------

class ModelError(SigmaError):
    """Base class for ML-related errors."""


class ModelNotFoundError(ModelError):
    """Requested model file does not exist."""


class ModelInferenceError(ModelError):
    """Model inference failed (bad input shape, runtime error, etc.)."""


# ---------------------------------------------------------------------------
# Export / report errors
# ---------------------------------------------------------------------------

class ExportError(SigmaError):
    """Base class for report and export errors."""


class ExportFormatError(ExportError):
    """Requested export format is unsupported or misconfigured."""
