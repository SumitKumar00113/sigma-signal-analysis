"""Application-level configuration for the Sigma Signal Analysis platform.

Settings are loaded from a TOML file (``~/.sigma/config.toml`` by default) and
can be overridden by environment variables prefixed with ``SIGMA_``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_CONFIG_DIR = Path.home() / ".sigma"


@dataclass
class DisplayConfig:
    """Visualisation defaults."""

    fft_size: int = 4096
    window: str = "hann"
    overlap: float = 0.5
    colormap: str = "viridis"
    dynamic_range_db: float = 80.0
    dark_mode: bool = True
    font_size: int = 12
    high_contrast: bool = False


@dataclass
class ProcessingConfig:
    """DSP and analysis defaults."""

    max_workers: int = 4
    chunk_size_samples: int = 1_048_576  # ~1 M samples per chunk
    max_file_size_bytes: int = 100 * 1024**3  # 100 GB
    enable_gpu: bool = False
    deterministic: bool = False  # for reproducible testing


@dataclass
class PathConfig:
    """Default filesystem paths."""

    config_dir: Path = field(default_factory=lambda: _DEFAULT_CONFIG_DIR)
    profiles_dir: Path = field(default_factory=lambda: _DEFAULT_CONFIG_DIR / "profiles")
    models_dir: Path = field(default_factory=lambda: _DEFAULT_CONFIG_DIR / "models")
    cache_dir: Path = field(default_factory=lambda: _DEFAULT_CONFIG_DIR / "cache")
    log_dir: Path = field(default_factory=lambda: _DEFAULT_CONFIG_DIR / "logs")


@dataclass
class AppConfig:
    """Root configuration container.

    Instantiate with ``AppConfig.load()`` to read from disk and env, or
    directly for in-memory / test usage.
    """

    display: DisplayConfig = field(default_factory=DisplayConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    # Application metadata
    app_name: str = "Sigma Signal Analysis"
    app_version: str = "0.1.0"
    schema_version: str = "1.0.0"

    @classmethod
    def load(cls, config_path: Path | None = None) -> AppConfig:
        """Load configuration from TOML, falling back to defaults.

        Environment variables like ``SIGMA_PROCESSING_MAX_WORKERS=8`` override
        file settings.
        """
        cfg = cls()

        # Override selected fields from env
        env_workers = os.environ.get("SIGMA_PROCESSING_MAX_WORKERS")
        if env_workers is not None:
            cfg.processing.max_workers = int(env_workers)

        env_gpu = os.environ.get("SIGMA_ENABLE_GPU")
        if env_gpu is not None:
            cfg.processing.enable_gpu = env_gpu.lower() in ("1", "true", "yes")

        env_dark = os.environ.get("SIGMA_DARK_MODE")
        if env_dark is not None:
            cfg.display.dark_mode = env_dark.lower() in ("1", "true", "yes")

        # Ensure dirs exist
        for d in (cfg.paths.config_dir, cfg.paths.profiles_dir,
                  cfg.paths.models_dir, cfg.paths.cache_dir, cfg.paths.log_dir):
            d.mkdir(parents=True, exist_ok=True)

        return cfg
