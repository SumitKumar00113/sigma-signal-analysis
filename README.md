# Sigma Signal Analysis

![CI](https://github.com/SumitKumar00113/sigma-signal-analysis/actions/workflows/ci.yml/badge.svg)

**Sigma Signal Analysis** is a powerful desktop software platform for RF signal inspection, classical DSP analysis, and advanced machine learning demodulation/classification. Designed for RF engineers, researchers, and signal analysts, this tool handles robust ingestion and visualization of complex baseband recordings.

> **Status:** Phase 2 complete — automated parameter estimation, modulation classification, demodulation (PSK/QAM/FSK), FEC decoding, de-interleaving, and bit-stream correlation are implemented and wired into the GUI.

## Architecture

```
src/
├── core/           # Pydantic models, enums, config, exceptions
├── ingestion/      # FileReader subclasses (WAV, SigMF, Raw IQ), normalizer, validator
├── dsp/
│   ├── spectral.py        # Welch PSD, spectrogram, peaks, noise floor, occupied BW
│   ├── preprocessing.py   # DC removal, normalisation, frequency translation, FIR filters
│   ├── detection.py       # Spectral-peak region detection
│   ├── measurements.py    # SNR (PSD), carrier offset, symbol rate (envelope + inst-freq)
│   ├── sync.py            # RRC matched filter, Gardner timing, M-power CFO, Costas loop
│   ├── classification.py  # Feature/cumulant modulation classifier
│   ├── demod.py           # PSK / QAM / FSK demodulators → symbols + bits
│   ├── rate_inference.py  # Sample-rate candidates for headerless files
│   └── pipeline.py        # validate → detect → measure → classify → demodulate
├── decoding/
│   ├── viterbi.py         # Convolutional codes (K=3..9 presets, puncturing, soft/hard)
│   ├── reed_solomon.py    # RS(n,k) over GF(2^8), configurable field, shortened codes
│   ├── ldpc.py            # Regular LDPC, min-sum decoder (experimental)
│   ├── interleaving.py    # Block, convolutional, diagonal, pseudo-random
│   └── correlation.py     # Autocorrelation, sync-word search, framing
├── gui/            # PySide6 main window, viewers, results dock, decoding workbench
└── reporting/      # JSON and HTML export (metadata, analysis, recovered bits)
```

## Core Capabilities

- **Universal Signal Ingestion**: Supports `.wav` (Mono/Stereo IQ), Raw `.iq` (with customizable data types such as `cf32_le`, `ci16_le`, `cu8`), and `.sigmf-data` / `.sigmf-meta` format recordings.
- **Robust Normalization & Validation**: Automatically converts raw integer/float bytes to uniform `complex64` arrays. Detects signal impairments like clipping, NaNs/Infs, DC offsets, and IQ gain imbalances.
- **Advanced DSP Engine**: Powered by `NumPy` and `SciPy`, provides Welch's method PSD, spectrograms, noise-floor estimation, spectral peak detection, signal region detection, SNR estimation, and occupied bandwidth measurement.
- **Interactive GUI**: Built on **PySide6** and **PyQtGraph**. Dark-themed UI with `QThreadPool` worker architecture to prevent UI freezing during intensive DSP tasks.
- **Analysis Pipeline**: Orchestrated processing chain (validate → preprocess → detect → measure → report) with full provenance tracking.
- **Parameter Estimation**: PSD-based SNR (full-band and in-band), carrier-offset centroid, symbol rate from envelope and instantaneous-frequency line spectra (PSK/QAM and FSK), occupied bandwidth. For headerless raw IQ, a sample-rate inference tool lists candidates consistent with standard symbol or SDR rates.
- **Modulation Classification**: Explainable two-stage classifier — envelope/instantaneous-frequency test for FSK (2/4-level), noise-corrected fourth-order cumulants for BPSK, QPSK, 8-PSK, 16-QAM, 64-QAM. Ranked candidates and evidence are shown in the GUI.
- **Demodulation**: RRC matched filter → Gardner timing recovery → M-th power carrier estimate → Costas phase tracking → Gray de-mapping for M-PSK and square QAM; band-limited frequency discriminator with k-means level slicing for M-FSK. Produces symbols (constellation view), hard bits, EVM.
- **Decoding Workbench**: De-interleave (block, convolutional, diagonal, pseudo-random), FEC decode (Viterbi with standard/custom polynomials and puncturing, Reed-Solomon with configurable field parameters, concatenated RS+conv, experimental LDPC), then bit-stream correlation: autocorrelation for frame period, sync-word search with error tolerance and inversion detection, header/payload framing, and bit export.
- **Synthetic Signal Generation**: Built-in generators for BPSK, QPSK, 16-QAM, and 2-FSK signals, with channel impairment models (AWGN, frequency offsets, IQ imbalances, phase noise) for testing and validation.

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

## Verified Performance

Measured with `python scripts/eval_demod.py` on synthetic signals with known bits (see `tests/signals/`):

| Signal | SNR | Classified | Bit error rate |
|---|---|---|---|
| BPSK, 1.2 kHz offset | 15 dB | BPSK | 0 |
| BPSK, 2 kHz offset | 3 dB | BPSK | 0 |
| QPSK | 12 dB / 6 dB | QPSK | 0 |
| 16-QAM | 20 dB / 14 dB | 16-QAM | 6e-4 / 5e-3 |
| 2-FSK | 14 dB / 6 dB | 2-FSK | 0 |

SNR estimates are within 0.3 dB and symbol-rate estimates within 1 Hz of ground truth on the bundled test recordings. Phase ambiguity inherent to M-PSK/QAM (rotations of the constellation) is reported as a warning; use differential decoding or a known sync word to resolve it.

## Limitations & Next Steps

- Region detection is frequency-only; bursty signals are treated as continuous.
- LDPC supports only a regular demo code; real systems need their specific parity-check matrix.
- The absolute sample rate of a headerless file cannot be recovered; only consistent candidates are offered.
- Future: deep-learning classifier for low-SNR / exotic modulations, burst segmentation, batch CLI.

## License

This project is licensed under the MIT License.
