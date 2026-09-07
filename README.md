# Sigma Signal Analysis

![Sigma Signal Analysis Logo/Banner](https://via.placeholder.com/1200x300?text=Sigma+Signal+Analysis)

**Sigma Signal Analysis** is a powerful desktop software platform for RF signal inspection, classical DSP analysis, and advanced machine learning demodulation/classification. Designed for RF engineers, researchers, and signal analysts, this tool handles robust ingestion and visualization of complex baseband recordings.

> **Status:** Phase 1 (Product and Engineering Baseline) completed. Phase 2 (ML & Demodulation) in development.

## Core Capabilities

- **Universal Signal Ingestion**: Supports `.wav` (Mono/Stereo IQ), Raw `.iq` (with customizable data types such as `cf32_le`, `ci16_le`, `cu8`), and `.sigmf-data` / `.sigmf-meta` format recordings.
- **Robust Normalization & Validation**: Automatically converts and normalizes raw integer/float bytes to uniform `complex64` arrays. Detects signal impairments like clipping, NaNs/Infs, DC offsets, and IQ gain imbalances.
- **Advanced DSP Engine**: Powered by `NumPy` and `SciPy`, provides Welch's method PSD calculations, high-resolution Spectrograms, automatic noise-floor estimation, and spectral peak detection.
- **Interactive GUI**: Built on the modern **PySide6** and **PyQtGraph** frameworks. Offers a responsive, dark-themed UI with `QThreadPool` worker architecture to prevent UI freezing during intense DSP tasks.

## Architecture

The project follows a layered architecture to separate concerns and improve maintainability:

- **`src/core/`**: Pydantic models mapping domain concepts (Validation Reports, Metadata, Results) and Python enums for strongly-typed states.
- **`src/ingestion/`**: Modular `FileReader` subclasses for WAV, SigMF, and Raw IQ formats. A central normalizer ensures every DSP block works with `np.complex64` data.
- **`src/dsp/`**: Pure math modules handling FFTs, overlaps, and spectral estimates.
- **`src/gui/`**: PySide6 panels, wizards, tabs, and `pyqtgraph` visualizers (Time, Spectrum, Waterfall).
- **`src/reporting/`**: Engines for exporting analytical results and visualizations to JSON and HTML.

## Getting Started

### Prerequisites

Ensure you have Python 3.10+ installed. Supported OS: Windows 11 and Ubuntu 22.04/24.04 LTS (macOS is also supported).

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
   # Install the GUI and Developer dependencies
   pip install -e '.[gui,dev]'
   ```
   *(Note: The `[ml]` feature group includes PyTorch and Scikit-Learn for Phase 2.)*

### Running the Application

Launch the desktop interface from the command line:

```bash
python src/app.py
```

### Running the Test Suite

The project enforces strict quality control through `pytest`. To run the 54+ unit tests checking file ingestion, numerical normalization, and mathematical stability:

```bash
pytest tests/
```

## Upcoming Features (Phase 2 & 3)
- Integration of PyTorch/TorchAudio for blind modulation classification.
- Interactive constellation analysis and phase extraction.
- Automatic de-interleaving and timing recovery.
- Batch processing services and CLI API.

## License

This project is licensed under the MIT License.
