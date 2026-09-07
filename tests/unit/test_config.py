"""Unit tests for configuration loading and management."""

import os
from pathlib import Path
from unittest import mock

import pytest

from src.core.config import AppConfig


class TestAppConfig:
    def test_default_construction(self):
        cfg = AppConfig()
        assert cfg.app_name == "Sigma Signal Analysis"
        assert cfg.display.dark_mode is True
        assert cfg.processing.max_workers == 4
        assert cfg.paths.config_dir == Path.home() / ".sigma"

    @mock.patch.dict(
        os.environ,
        {
            "SIGMA_PROCESSING_MAX_WORKERS": "16",
            "SIGMA_ENABLE_GPU": "true",
            "SIGMA_DARK_MODE": "0",
        },
    )
    def test_load_from_env(self, tmp_path: Path):
        # We need to mock Path.home() so the dirs are created in tmp_path
        with mock.patch("src.core.config.Path.home", return_value=tmp_path):
            cfg = AppConfig.load()
            
            assert cfg.processing.max_workers == 16
            assert cfg.processing.enable_gpu is True
            assert cfg.display.dark_mode is False
            
            # Verify directories were created
            assert cfg.paths.config_dir.exists()
            assert cfg.paths.log_dir.exists()
            assert cfg.paths.models_dir.exists()

    @mock.patch.dict(os.environ, {"SIGMA_ENABLE_GPU": "yes", "SIGMA_DARK_MODE": "false"})
    def test_env_boolean_parsing(self, tmp_path: Path):
        with mock.patch("src.core.config.Path.home", return_value=tmp_path):
            cfg = AppConfig.load()
            assert cfg.processing.enable_gpu is True
            assert cfg.display.dark_mode is False
