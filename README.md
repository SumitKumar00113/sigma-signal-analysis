# Sigma Signal Analysis

![CI](https://github.com/SumitKumar00113/sigma-signal-analysis/actions/workflows/ci.yml/badge.svg)

**Sigma Signal Analysis** is a powerful desktop software platform for RF signal inspection, classical DSP analysis, and advanced machine learning demodulation/classification. Designed for RF engineers, researchers, and signal analysts, this tool handles robust ingestion and visualization of complex baseband recordings.

> **Status:** Phase 3. Implemented and wired into the GUI:
> - automated parameter estimation;
> - classification of 18 modulation types (rules plus a learned model, hybrid);
> - demodulation of PSK/QAM/FSK/MSK/GMSK/ASK/OQPSK/π/4-DQPSK, and AM/FM/SSB to audio;
> - correct handling of real (mono) WAV recordings;
> - **blind identification of FEC codes and interleavers**;
> - FEC decoding, de-interleaving and bit-stream correlation.

## Architecture

```
src/
├── core/           # Pydantic models, enums, config, exceptions
├── ingestion/      # FileReader subclasses (WAV, SigMF, Raw IQ), normalizer, validator,
│                   #   real→complex conversion (Hilbert / FM re-modulation)
├── dsp/
│   ├── spectral.py        # Welch PSD, spectrogram, peaks, noise floor, occupied BW
│   ├── preprocessing.py   # DC removal, normalisation, frequency translation, FIR filters
│   ├── detection.py       # Spectral-peak region detection (legacy)
│   ├── bursts.py          # Time–frequency burst detection and extraction
│   ├── measurements.py    # SNR (PSD), carrier offset, symbol rate (envelope + inst-freq)
│   ├── sync.py            # RRC matched filter, Gardner timing, M-power CFO, Costas loop
│   ├── classification.py  # Explainable modulation classifier (18 types)
│   ├── demod.py           # PSK/QAM/FSK/MSK/GMSK/ASK/OQPSK/DPSK demodulators → bits
│   ├── analog.py          # AM / FM / SSB demodulators → audio
│   ├── rate_inference.py  # Sample-rate candidates for headerless files
│   └── pipeline.py        # validate → detect → measure → classify → demodulate
├── decoding/
│   ├── viterbi.py         # Convolutional codes (K=3..9 presets, puncturing, soft/hard)
│   ├── reed_solomon.py    # RS(n,k) over GF(2^8), configurable field, shortened codes
│   ├── ldpc.py            # Sparse LDPC codec: .alist/.qc, IRA/dense encoders, min-sum
│   ├── ldpc_library.py    # Standard-code catalogue, download on request
│   ├── interleaving.py    # Block, convolutional, diagonal, pseudo-random
│   ├── fec_id.py          # Blind FEC identification (conv library/blind, RS, rank)
│   ├── interleaver_id.py  # Blind interleaver identification (type, size, alignment)
│   ├── gf2.py             # GF(2) rank / null space on bit-packed rows
│   └── correlation.py     # Autocorrelation, sync-word search, framing
├── gui/            # PySide6 main window, viewers, results dock, decoding workbench
├── ml/             # Learned classifier: features, training data, model, hybrid, trainer
└── reporting/      # JSON and HTML export (metadata, analysis, recovered bits)
```

## Core Capabilities

- **Universal Signal Ingestion**: Supports `.wav` (Mono/Stereo IQ), Raw `.iq` (with customizable data types such as `cf32_le`, `ci16_le`, `cu8`), and `.sigmf-data` / `.sigmf-meta` format recordings.
- **Real-signal WAV handling**: mono (real) audio, e.g. HF receiver output or a modem channel, is converted to its analytic signal with a 255-tap Hilbert FIR. Image rejection is > 80 dB and chunked reads are bit-exact. The spectrum is one-sided, so a 1800 Hz audio carrier appears at +1800 Hz instead of as a mirrored pair. Discriminator audio is FM re-modulated. Stereo files get an automatic I/Q vs. dual-channel vs. duplicated-mono suggestion in the input wizard, plus channel selection and I/Q swap. Noise-floor estimates ignore "dead" spectrum regions (the empty half of an analytic signal, receiver stop-bands), so SNR, bandwidth and carrier estimates remain correct for real recordings.
- **Robust Normalization & Validation**: Automatically converts raw integer/float bytes to uniform `complex64` arrays. Detects signal impairments like clipping, NaNs/Infs, DC offsets, and IQ gain imbalances.
- **Advanced DSP Engine**: Powered by `NumPy` and `SciPy`, provides Welch's method PSD, spectrograms, noise-floor estimation, spectral peak detection, signal region detection, SNR estimation, and occupied bandwidth measurement.
- **Interactive GUI**: Built on **PySide6** and **PyQtGraph**. Dark-themed UI with `QThreadPool` worker architecture to prevent UI freezing during intensive DSP tasks.
- **Analysis Pipeline**: Orchestrated processing chain (validate → preprocess → detect → measure → report) with full provenance tracking.
- **Parameter Estimation**: PSD-based SNR (full-band and in-band), carrier-offset centroid, symbol rate from envelope and instantaneous-frequency line spectra (PSK/QAM and FSK), occupied bandwidth. For headerless raw IQ, a sample-rate inference tool lists candidates consistent with standard symbol or SDR rates.
- **Modulation Classification**: an explainable decision tree covering 18 types.
  - **Digital:** BPSK, QPSK, 8-PSK, 16-QAM, 64-QAM, OQPSK, π/4-DQPSK, 2-FSK, 4-FSK, MSK, GMSK/GFSK, OOK, 2/4-ASK.
  - **Analog:** AM, FM, SSB (USB/LSB), plus unmodulated-carrier detection.
  - **How it decides:** by where the message lives (envelope vs. frequency) and whether a carrier line is present, then confirms with symbol-centre structure. That means discrete levels for FSK/ASK, noise-corrected cumulants plus a 16/64-QAM grid fit for PSK/QAM, the half-symbol rail offset for OQPSK, and the asymmetric 4th-power line pair for π/4-DQPSK. Analog signals are the ones with no symbol structure; SSB is recognised by its lop-sided band.
  - The QPSK family's carrier comes exactly from the M-th power lines, and the symbol rate is verified by the constellation it produces. Ranked candidates and the evidence are shown in the GUI.
- **Demodulation**:
  - **PSK/QAM:** RRC matched filter → Gardner timing → M-th power carrier estimate → Costas loop → Gray de-mapping.
  - **FSK/MSK/GMSK:** discriminator with level slicing.
  - **OOK/ASK:** envelope detector with automatic 2/4-level detection.
  - **OQPSK:** 4th-power carrier, I-rail timing, Q sampled half a symbol later.
  - **DBPSK and π/4-DQPSK:** differential detection, with no phase ambiguity.
  - Digital demodulators produce symbols (constellation view), hard bits and EVM.
  - **AM, FM, SSB:** envelope, discriminator and product detectors produce audio at 8 kHz, with the modulation index, peak deviation or inferred SSB carrier reported. Save it via *File → Save Demodulated Audio…*.
- **Decoding Workbench**: De-interleave (block, convolutional, diagonal, pseudo-random), FEC decode (Viterbi with standard/custom polynomials and puncturing, Reed-Solomon with configurable field parameters, concatenated RS+conv, LDPC), then bit-stream correlation: autocorrelation for frame period, sync-word search with error tolerance and inversion detection, header/payload framing, and bit export.
- **Blind FEC Identification** (🔍 *Auto-detect* in the decoding workbench): convolutional codes are identified from the parity checks of their dual code. The library covers K = 3…9, rate 1/2 and 1/3, every generator order, and the DVB/802.11 puncture patterns 2/3–7/8. Detection works at several percent BER and reports the code phase, bit inversion and channel BER. Unknown rate-1/n codes have their generators recovered blindly. Reed-Solomon codes are identified by n, k, field polynomial, first root and exact alignment, even when every block contains symbol errors. Concatenated RS + convolutional chains are found by decoding the inner code first. Unknown binary block codes (e.g. Hamming) are reported by length and rate.
- **LDPC at real sizes**:
  - Sparse parity-check matrices loaded from `.alist` or quasi-cyclic `.qc` files, including punctured and rank-deficient codes.
  - Linear-time encoding for IRA / dual-diagonal codes (DVB-S2); packed-bit elimination for everything else.
  - Batched, vectorised normalised min-sum decoding. A 64 800-bit DVB-S2 frame decodes in ≈ 0.2 s.
  - A catalogue of standard codes is downloaded on request into `~/.sigma/ldpc/` from the decoding panel or with `python -m src.decoding.ldpc_library fetch …`. It covers DVB-S2 (5 rates), Wi-Fi 802.11n, WiMAX, WRAN, 10GBASE-T, CCSDS, CCSDS AR4JA, and 5G NR BG1/BG2.
  - Verified end to end on those matrices: encoded words satisfy H·c = 0, and decode error-free at their normal operating points (e.g. DVB-S2 r1/2 at Eb/N0 = 1.5 dB, 10GBASE-T at 4 dB).
  - Blind identification tries every alignment against the installed codes, including inverted streams.
- **Burst and multi-signal handling**:
  - Signals that come and go, or share the recording at different frequencies, are detected on the spectrogram: robust noise floor, then hysteresis thresholding (seed + extent), then connected components. FSK tones and keyed-carrier gaps are merged, and edges are refined to sample resolution (≈ 1 ms).
  - Each burst is cut out, moved to 0 Hz, filtered to its own band (so simultaneous signals are separated), decimated, and analysed in full.
  - The strongest burst drives the main result. All bursts are outlined on the waterfall and listed in the navigator; click one to see its own constellation, bits and parameters.
  - Example: a 2.5 s recording with BPSK, 2-FSK, QPSK, OOK and 16-QAM bursts (some overlapping in time) is analysed in ≈ 2 s. Every burst is found, correctly classified and decoded error-free or nearly so.
- **Blind Interleaver Identification**: block, diagonal and convolutional interleavers are found by a stride scan over the whole code library. Short-column block and short-branch convolutional interleavers use a comb search. Pseudo-random (LCG/NumPy) interleavers use a seed search. Each result gives the dimensions and the exact bit alignment, verified by restoring the code structure.
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
   # Install with GUI, learned classifier and developer dependencies
   pip install -e '.[gui,ml,dev]'
   ```
   *(`[ml]` adds scikit-learn for the learned modulation classifier; without it the rule-based classifier is used.)*

### Training the Learned Classifier

A trained model ships with the application (`src/ml/models/`). To retrain, or to add your own labelled recordings, use either:

- **GUI:** *Analysis → Classifier → Train Classifier…*
- **Command line:**

```bash
# synthetic data only (≈ 10 min on 8 cores)
python -m src.ml.train --per-class 1000 --workers 8

# plus labelled recordings: CSV with path,modulation[,sample_rate,datatype,wav_interpretation]
python -m src.ml.train --manifest captures/labels.csv --recording-weight 3
```

The model is saved to `~/.sigma/models/` and used automatically. *Analysis → Classifier* selects one of three modes:

- **Hybrid** (default): explainable rules, corrected by the model where it is much surer.
- **Rules only.**
- **Learned model only.**

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

**Classification over all 18 types.** Measured with `python scripts/eval_classifier.py 20 15 8 4`, running the full pipeline exactly as the GUI does. Each type gets 20 randomised trials: symbol rates 1.2–9.6 kBd, 10–20 samples/symbol, carrier offsets up to ±10 % of the sample rate. SNR is measured over the full sample band.

| SNR | Accuracy | Notes |
|---|---|---|
| 15 dB | 100 % for every type | BER 0 wherever bits are checked (4-ASK 2e-3); AM/FM audio correlation 0.92 / 0.96 |
| 8 dB | 90–100 % for every type | |
| 4 dB | 100 % for BPSK, QPSK, 8-PSK, 4-FSK, OOK, π/4-DQPSK, DBPSK, FM, SSB | 64-QAM 45 % (mostly read as 16-QAM), 4-ASK 45 % (read as AM); 2-FSK, MSK, GMSK, OQPSK 70–80 % |

DBPSK is reported as BPSK: differential encoding is a property of the data, not the signal. It decodes correctly with the DPSK demodulator or differential decoding.

**Learned classifier (hybrid mode, the default).** A gradient-boosted model on 38 rate-normalised features, trained on 18,000 synthetic signals (0–20 dB). Its held-out accuracy is 98.9 %. The table below uses signals generated independently of the training data (same harness as above, 20 per type), with all three classifiers run on the same signals:

| SNR | Rules only | Learned model only | Hybrid |
|---|---|---|---|
| 4 dB | 86 % | 99 % | **99 %** |
| 8 dB | 99 % | 100 % | **100 %** |
| 15 dB | 100 % | 100 % | **100 %** |

At 4 dB the model fixes the rules' weak cases: 64-QAM 60 → 95 %, 4-ASK 30 → 100 %, GMSK 45 → 100 %, MSK 60 → 100 %. In hybrid mode the rules' explainable evidence is kept, and every override by the model is stated in the result.

## Limitations & Next Steps

- Bursts must stand out from the noise on the spectrogram (≈ 5 dB after averaging a few frames); very short (< 4 ms) or very weak bursts are not separated. Separate signals closer than ≈ 2 % of the sample rate in frequency and overlapping in time are treated as one. Very wide FSK (tones more than ≈ 40 % of the sample rate apart) can be split into pieces.
- Standard LDPC matrices are not bundled (the published source carries no licence); they are downloaded on request.
- 5G NR rate matching (bit selection, filler bits) is not modelled: the base-graph code is used as is.
- LDPC codes in which *every* check involves a punctured bit (CCSDS AR4JA) decode normally but cannot be identified blindly.
- Interleaver identification needs a convolutional code inside the interleaver. Punctured codes whose parity checks are longer than the interleaver runs are only found when the code is selected under FEC first. Diagonal interleavers need columns longer than the code's check span. Pseudo-random identification searches seeds 0…N−1 for the block sizes you give it.
- Reed-Solomon identification assumes GF(2⁸), generator α (fcr 0 or 1), ≥ 6 parity symbols and no CCSDS dual-basis mapping. Rank-based block-code detection needs a near error-free stream.
- The absolute sample rate of a headerless file cannot be recovered; only consistent candidates are offered.
- Modulations outside the 18 types (OFDM, APSK, CPM variants, DSB-SC) are reported as unknown. The learned model has only seen synthetic signals unless you retrain it with labelled recordings. With rules only, accuracy drops for 64-QAM and 4-ASK below ≈ 6 dB.
- At very low SNR the classification can be right while demodulation fails: 4-ASK at 4 dB is often sliced as 2 levels, and the symbol rate of weak GMSK/FSK may not be recoverable. Enter the symbol rate manually in that case.
- The SSB carrier is suppressed, so it is inferred as 300 Hz beyond the band edge. A tuning error shifts the pitch of the recovered audio but it stays intelligible.
- OQPSK and M-PSK/QAM decisions carry the usual carrier-phase ambiguities (reported as warnings); OQPSK also has a one-symbol I/Q pairing ambiguity.
- Future: deep-learning classifier for low-SNR / exotic modulations, burst segmentation, batch CLI.

## License

This project is licensed under the MIT License.
