# Sigma Signal Analysis

![CI](https://github.com/SumitKumar00113/sigma-signal-analysis/actions/workflows/ci.yml/badge.svg)

**Sigma Signal Analysis** is a powerful desktop software platform for RF signal inspection, classical DSP analysis, and advanced machine learning demodulation/classification. Designed for RF engineers, researchers, and signal analysts, this tool handles robust ingestion and visualization of complex baseband recordings.

> **Status:** Phase 3. Implemented and wired into the GUI:
> - automated parameter estimation;
> - classification of 18 modulation types (rules plus a learned model, hybrid);
> - demodulation of PSK/QAM/FSK/MSK/GMSK/ASK/OQPSK/π/4-DQPSK, and AM/FM/SSB to audio;
> - correct handling of real (mono) WAV recordings;
> - **blind identification of FEC codes and interleavers**;
> - FEC decoding, de-interleaving and bit-stream correlation;
> - **one-click Auto-Analyse**: recording → parameters → demodulation → interleaver → FEC → decoded frames with sync word and header fields found blindly (GUI and `python -m src.auto_analyse`).

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
│   ├── correlation.py     # Autocorrelation, sync-word search, framing
│   ├── framing.py         # Automatic sync-word / frame / header discovery
│   ├── baudot.py          # Asynchronous ITA2 / Baudot (RTTY) → text
│   └── auto_decode.py     # One-click chain: mapping → interleaver → FEC → framing
├── gui/            # PySide6 main window, viewers, results dock, decoding workbench
├── auto_analyse.py # Command-line one-click analysis of a recording
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
- **One-click Auto-Analyse** (⚡ in the toolbar, *Analysis → Auto-Analyse*, Ctrl+Shift+R; or ⚡ *Auto-decode* in the decoding workbench for bits already demodulated). The chain runs in the background (cancellable) and fills in every workbench control, so each decision can be checked, changed and re-applied by hand:
  1. The recording is analysed and demodulated as usual.
  2. **Bit mapping**: the demodulator's QPSK/16-QAM phase ambiguity (swapped and inverted bits) and the OQPSK I/Q pairing are resolved by testing which mapping reveals a convolutional code.
  3. **FEC on the stream as received**: if a code is already visible there is no interleaver. Otherwise the **interleaver** is identified and FEC identification is repeated on the de-interleaved stream.
  4. **Decoding** of the whole stream: Viterbi, Reed-Solomon, Viterbi + RS or LDPC.
  - **Teleprinter text**: asynchronous Baudot/ITA2 (RTTY) is recognised by its start/stop elements, right after the bit mapping. The polarity, the stop length (1, 1.5 or 2 elements) and the sampling grid are found automatically, and the letters/figures shifts are applied. The text appears in the workbench (*View → text*) and in the CLI (`--text-out file.txt`). No FEC or frame search is run on such a signal.
  5. **Framing** on the most processed stream that shows it. Known sync words are tried first (CCSDS ASM and its 64-bit variant, CCSDS telecommand, POCSAG, IRIG-106, DMR, P25, Barker-13, MPEG-TS, GPS). Otherwise the sync word is **discovered blindly** as the bit pattern that recurs far more often than chance (Poisson test, false-alarm probability 10⁻⁶; idle fill is ignored). Around it, bits that stay constant from frame to frame are reported as fixed header fields, and fields that count up by one per frame as frame counters. The result is the frame length and the header/payload split, and every frame is listed with inversion undone.
  - Example: CCSDS-framed data with a 16-bit frame counter, K = 7 rate 1/2 coded, block-interleaved 16 × 64, QPSK at 10 dB. One click gives the interleaver (16 × 64 and its alignment), the code, decoding (0.01 % channel bit errors corrected), then the ASM, the 1024-bit frames and the counter field.
  - Command line: `python -m src.auto_analyse capture.sigmf-meta` (or `.wav`, or `.iq --fs 250000 --dtype cf32_le`). Options: `--json report.json` writes the whole report; `--save-bits out.bin` saves the final bit stream; `--no-interleaver` skips the slow interleaver search; `--ldpc` tries the installed standard LDPC codes.
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

# One-click analysis from the command line (or: sigma-auto …)
python -m src.auto_analyse capture.wav --json report.json
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
| 4 dB | 88 % | 98 % | **99 %** |
| 8 dB | 95 % | 100 % | **99 %** |
| 15 dB | 100 % | 100 % | **100 %** |

At 4 dB the model fixes the rules' weak cases: 64-QAM 60 → 95 %, 4-ASK 30 → 100 %, GMSK 45 → 100 %, MSK 60 → 100 %. In hybrid mode the rules' explainable evidence is kept, and every override by the model is stated in the result.

### Real off-air recordings

`python scripts/eval_offair.py <folder>` runs the one-click analysis on a set of genuine SDR# baseband recordings (16-bit stereo I/Q WAV). The recordings are not in the repository. Each result is checked against the true parameters of the service, which were measured from the signal itself:

| Recording | What it is | Result |
|---|---|---|
| NAVTEX, 518 kHz (8 min) | SITOR-B, 2-FSK 100 Bd, 170 Hz shift | 2-FSK, 100.0 Bd, EVM 9.7 %; no false FEC or framing |
| DWD RTTY (7 min) | 2-FSK 50 Bd, 450 Hz shift, asynchronous | 2-FSK on the 100 Bd half-element grid, with a note that the element rate is 50 Bd. **Decoded to text**: 2,658 characters, 99.8 % with valid stop elements. It is the DWD Hamburg (DDH47) Baltic Sea forecast, "SEEWETTERBERICHT FUER DIE OSTSEE … 28.02.23" |
| RS41 radiosonde, 403 MHz (8 min) | GFSK 4800 Bd, one frame per second | 2-FSK 4798.9 Bd; 15 bursts decoded together; **RS41 header found in every frame with 0 bit errors** |
| NOAA-18 APT, 137.9 MHz | FM with a 2400 Hz AM subcarrier | FM, peak deviation ≈ 15 kHz, audio for an APT decoder |
| SSTV, 145.8 MHz | narrow-band FM carrying SSTV tones | FM, audio for an SSTV decoder |
| NO-84 packets, 145.8 MHz | FM bursts carrying audio tones | all 11 bursts FM, deviation ≈ 8.9 kHz |
| NOAA-15 SARP-3, 1544.5 MHz (15 min) | residual-carrier PM, 2400 bps, ±35 kHz Doppler | **not supported** (see limitations) |

These recordings exposed problems that synthetic tests had not. Each is now fixed and covered by a regression test in `tests/unit/test_offair_fixes.py`:
- **Symbol-rate harmonics.** On long recordings a harmonic (3 × or 5 × the rate) can be the strongest line. Candidates are now ranked by their harmonic comb, so the fundamental wins. The lowest rate searched is now 30 Hz instead of 100 Hz, so 45–50 Bd teleprinter modes are reachable.
- **Impulsive HF noise** around a continuous transmission split it into dozens of "bursts". A signal present ≥ 90 % of the time and holding ≥ 90 % of the burst energy is now treated as continuous.
- **False FEC.** Repetition, time diversity and half-element sampling satisfy some parity checks of a convolutional code by chance. A candidate is now rejected when a trial Viterbi decode leaves > 12 % channel errors, or leaves far more errors than the demodulator's EVM allows.
- **False framing.** Idle and phasing patterns repeat. Frames must now be ≥ 64 bits long with a varying payload, idle patterns with periods up to 8 bits are ignored, and frame counters must pass a significance test.
- **FM carrying audio** (APT subcarrier, SSTV, AFSK) was classified as FSK/PSK: a sampled tone looks like "2 levels". The FM discriminator output is now checked first. Audio has narrow spectral lines (≥ 10 % of its power; digital modes ≤ 4 %) and a continuous distribution (keyed FSK sits on two levels ≥ 80 % of the time). Such a signal is FM, and the learned model cannot overrule this: it never saw such signals in training.
- **One frame per burst** (radiosondes, packet radio): bursts from the same transmitter (same modulation, carrier and rate) are now decoded together. The RS41 header was added to the known sync words.

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
- Auto-decode looks for the code families listed above. Scrambled payloads (CCSDS/DVB randomisers) are not descrambled. An outer RS code behind a byte interleaver (e.g. the DVB-S Forney interleaver between RS and the convolutional code) is not found automatically. When nothing is present, the searches for an interleaver and an RS code take ≈ 10–15 s on 40 000 bits.
- Blind sync discovery needs at least 3 frames and a sync word of ≥ 16 bits; shorter markers are only found from the known-sync library. Blindly, the sync word and the constant header bits that follow it cannot be told apart, and the polarity of a wholly inverted stream is unknown. Counters are reported with their constant leading zeros rounded to whole bytes.
- Satellite downlinks with strong Doppler drift (hundreds of Hz/s, e.g. NOAA SARP at L-band) are not tracked, and residual-carrier PM (split-phase) has no demodulator yet. Such recordings are misclassified.
- Of the protocol layers above framing, only asynchronous Baudot (RTTY) text is decoded. Not decoded: SITOR-B (NAVTEX) text, AX.25/APRS inside AFSK audio, RS41 descrambling, and APT/SSTV images. The analysis stops at the demodulated bits, the frames or the audio.
- Only the first 10 million samples of a recording are analysed (40 s at 250 kHz).
- Future: deep-learning classifier for low-SNR / exotic modulations.

## License

This project is licensed under the MIT License.
