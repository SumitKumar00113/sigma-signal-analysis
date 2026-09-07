"""Abstract base class for all file readers.

Every concrete reader (WAV, raw IQ, SigMF) must implement this interface so
the rest of the application can treat any recording uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from src.core.models import FileValidationReport, RecordingMetadata


class FileReader(ABC):
    """Read and validate a recording file.

    Subclasses MUST NOT modify the source file under any circumstances.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @abstractmethod
    def validate(self) -> FileValidationReport:
        """Run integrity and quality checks on the file.

        Returns a :class:`FileValidationReport` summarising the result.
        """

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @abstractmethod
    def read_metadata(self) -> RecordingMetadata:
        """Extract or infer recording metadata from the file.

        For raw IQ files the caller must supply partial metadata via
        ``set_metadata_hints()`` before calling this method.
        """

    def set_metadata_hints(self, **kwargs: object) -> None:
        """Supply user-provided metadata hints (sample rate, datatype, …).

        The default implementation stores hints as instance attributes.
        Subclasses may override to perform early validation.
        """
        for key, value in kwargs.items():
            setattr(self, f"_hint_{key}", value)

    def _get_hint(self, key: str, default: object = None) -> object:
        return getattr(self, f"_hint_{key}", default)

    # ------------------------------------------------------------------
    # Sample I/O
    # ------------------------------------------------------------------

    @abstractmethod
    def read_samples(
        self,
        start: int = 0,
        count: int = -1,
    ) -> np.ndarray:
        """Read *count* complex-64 samples starting at *start*.

        Parameters
        ----------
        start:
            Zero-based sample index (not byte offset).
        count:
            Number of complex samples to read.  ``-1`` means *all remaining*.

        Returns
        -------
        numpy.ndarray
            1-D array of ``complex64``.
        """

    @abstractmethod
    def total_samples(self) -> int:
        """Return the total number of complex samples in the file."""

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def read_chunk(self, chunk_index: int, chunk_size: int) -> np.ndarray:
        """Read a fixed-size chunk by index.

        Equivalent to ``read_samples(chunk_index * chunk_size, chunk_size)``.
        """
        return self.read_samples(start=chunk_index * chunk_size, count=chunk_size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.path}')"
