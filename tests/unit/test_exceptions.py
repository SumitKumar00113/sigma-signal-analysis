"""Unit tests for the custom exception hierarchy."""

import pytest

from src.core.exceptions import (
    ExportError,
    FileFormatError,
    FileNotFoundSigmaError,
    FilePermissionSigmaError,
    FileTruncatedError,
    MetadataError,
    ModelError,
    ProcessingError,
    SigmaError,
    UnsupportedFormatError,
)


class TestExceptions:
    def test_base_hierarchy(self):
        # All custom exceptions must inherit from SigmaError
        assert issubclass(FileNotFoundSigmaError, SigmaError)
        assert issubclass(ProcessingError, SigmaError)
        assert issubclass(MetadataError, SigmaError)
        assert issubclass(ModelError, SigmaError)
        assert issubclass(ExportError, SigmaError)

    def test_file_error_hierarchy(self):
        # Specific file errors should be catchable as generic file errors (which are SigmaErrors)
        assert issubclass(FileFormatError, SigmaError)
        assert issubclass(FileTruncatedError, SigmaError)
        assert issubclass(FilePermissionSigmaError, SigmaError)
        assert issubclass(UnsupportedFormatError, SigmaError)

    def test_raise_and_catch(self):
        with pytest.raises(SigmaError) as exc_info:
            raise FileNotFoundSigmaError("Missing file.iq")
        
        assert "Missing file.iq" in str(exc_info.value)
        assert type(exc_info.value) is FileNotFoundSigmaError
