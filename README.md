# Sigma Signal Analysis

![CI](https://github.com/SumitKumar00113/sigma-signal-analysis/actions/workflows/ci.yml/badge.svg)

**Sigma Signal Analysis** is a powerful desktop software platform for RF signal inspection, classical DSP analysis, and advanced machine learning demodulation/classification. Designed for RF engineers, researchers, and signal analysts, this tool handles robust ingestion and visualization of complex baseband recordings.

> **Status:** Phase 1 (Product and Engineering Baseline) completed. Phase 2 (ML & Demodulation) in development.

## Architecture

```
src/
├── core/           # Pydantic models, enums, config, exceptions
├── ingestion/      # FileReader subclasses (WAV, SigMF, Raw IQ), normalizer, validator
├── dsp/            # Spectral analysis, preprocessing, detection, measurements, pipeline
├── gui/            # PySide6 main window, viewers (time, spectrum, waterfall, constellation)
└── reporting/      # JSON and HTML export engines
```

## Core Capabilities

- **Universal Signal Ingestion**: Supports `.wav` (Mono/Stereo IQ), Raw `.iq` (with customizable data types such as `cf32_le`, `ci16_le`, `cu8`), and `.sigmf-data` / `.sigmf-meta` format recordings.
- **Robust Normalization & Validation**: Automatically converts raw integer/float bytes to uniform `complex64` arrays. Detects signal impairments like clipping, NaNs/Infs, DC offsets, and IQ gain imbalances.
- **Advanced DSP Engine**: Powered by `NumPy` and `SciPy`, provides Welch's method PSD, spectrograms, noise-floor estimation, spectral peak detection, signal region detection, SNR estimation, and occupied bandwidth measurement.
- **Interactive GUI**: Built on **PySide6** and **PyQtGraph**. Dark-themed UI with `QThreadPool` worker architecture to prevent UI freezing during intensive DSP tasks.
- **Analysis Pipeline**: Orchestrated processing chain (validate → preprocess → detect → measure → report) with full provenance tracking.
- **Synthetic Signal Generation**: Includes built-in generators for BPSK, QPSK, and 2-FSK signals, along with realistic channel impairment models (AWGN, frequency offsets, IQ imbalances, phase noise) for robust testing and validation.

## Getting Started

### Prerequisites

- Python **3.11+**
- Supported OS: Windows 11, Ubuntu 22.04/24.04 LTS, macOS

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SumitKumar00113/sigma-signal-analysis.git
   cd sigma-signal-analysis
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
   ```

3. Install dependencies:
   ```bash
   # Install with GUI and developer dependencies
   pip install -e '.[gui,dev]'
   ```
   *(Note: The `[ml]` feature group includes PyTorch and Scikit-Learn for Phase 2.)*

### Running the Application

```bash
# Via entry point (after pip install)
sigma

# Or directly
python src/app.py
```

### Running the Test Suite

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Lint check
ruff check src/ tests/
```

## Test Coverage

| Module | Coverage |
|---|---|
| `src/core/` | 90%+ |
| `src/ingestion/` | 85%+ |
| `src/dsp/` | 80%+ |
| `src/reporting/` | 80%+ |

## Upcoming Features (Phase 2 & 3)

- Integration of PyTorch/TorchAudio for blind modulation classification
- Interactive constellation analysis and phase extraction
- Automatic de-interleaving and timing recovery
- Batch processing services and CLI API
- FSK/PSK/QAM demodulation with carrier and timing recovery

## License

This project is licensed under the MIT License.
