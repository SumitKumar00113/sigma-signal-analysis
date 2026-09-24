"""Time–frequency burst detection and burst extraction.

Signals that come and go (HF/VHF bursts, TDMA slots, keyed transmitters)
and signals that share a recording at different frequencies must be found
and analysed separately: analysing the whole file dilutes every estimate
with the noise between bursts and mixes simultaneous signals together.

Detection works on the spectrogram:

1. power smoothed over a few bins/frames, compared with a robust noise
   floor (the lower of each bin's own low percentile over time and the
   whole spectrogram's, so continuous signals are still seen as signal);
2. cells more than *threshold* dB above it, morphologically closed in time
   (bridges keying gaps such as OOK "off" symbols);
3. hysteresis: a connected component is kept only if it contains a *seed*
   – a cell that also stands out on a much more heavily averaged
   spectrogram, where noise essentially never reaches the seed threshold –
   so noise specks vanish while weak but sizeable bursts survive;
4. components merged when they overlap in time and either sit close in
   frequency or are *complementary* – one on exactly when the other is off,
   as the tones of a wide FSK are (two independent keyed carriers are not);
5. start/end refined at sample resolution on the burst's band-limited
   envelope.

:func:`extract_burst` then cuts a burst out, moves it to 0 Hz, filters it
to its own band (removing neighbours) and decimates it, ready for the
normal analysis chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy import ndimage
from scipy import signal as sp_signal

from src.core.enums import DetectionMethod
from src.core.models import SignalRegion
from src.dsp.preprocessing import translate_frequency
from src.dsp.spectral import live_bins


@dataclass
class BurstConfig:
    fft_size: int | None = None          # None: ≈ 4 ms time resolution
    threshold_db: float = 6.0            # extent of a burst
    seed_threshold_db: float = 5.0       # on the heavily averaged spectrogram (≈ 8σ)
    min_duration_s: float = 0.004
    min_bandwidth_hz: float = 0.0
    merge_time_s: float = 0.02           # gaps shorter than this are bridged
    merge_freq_hz: float | None = None   # None: 2 % of the sample rate (≥ 3 bins)
    max_bursts: int = 64
    continuous_fraction: float = 0.9     # active this much of the time → continuous
    refine_edges: bool = True


@dataclass
class Burst:
    start_sample: int
    end_sample: int
    center_hz: float                     # relative to the recording's centre (0 Hz)
    bandwidth_hz: float
    snr_db: float
    confidence: float

    @property
    def num_samples(self) -> int:
        return self.end_sample - self.start_sample

    def duration(self, sample_rate: float) -> float:
        return self.num_samples / sample_rate

    def to_region(self, sample_rate: float, center_frequency_hz: float = 0.0,
                  recording_id: str = "") -> SignalRegion:
        return SignalRegion(
            recording_id=recording_id,
            start_sample=int(self.start_sample), end_sample=int(self.end_sample),
            start_time_sec=self.start_sample / sample_rate,
            end_time_sec=self.end_sample / sample_rate,
            center_frequency_hz=center_frequency_hz + self.center_hz,
            bandwidth_hz=self.bandwidth_hz, snr_db=self.snr_db,
            detection_method=DetectionMethod.BURST, confidence=self.confidence,
            annotation_source="auto",
        )


@dataclass
class BurstDetection:
    bursts: list[Burst] = field(default_factory=list)
    duty_cycle: float = 0.0              # fraction of the recording with any burst active
    continuous: bool = False             # one signal present (almost) all the time
    noise_floor_db: float = 0.0
    fft_size: int = 0

    @property
    def intermittent(self) -> bool:
        """Bursty, or several signals: analyse the bursts one by one."""
        return bool(self.bursts) and (not self.continuous or len(self.bursts) > 1)


def _auto_fft(sample_rate: float) -> int:
    target = sample_rate * 0.004
    return int(np.clip(2 ** round(np.log2(max(target, 1.0))), 64, 4096))


def detect_bursts(samples: np.ndarray, sample_rate: float,
                  config: BurstConfig | None = None) -> BurstDetection:
    cfg = config or BurstConfig()
    x = np.asarray(samples, dtype=np.complex64).reshape(-1)
    nfft = cfg.fft_size or _auto_fft(sample_rate)
    if len(x) < 4 * nfft:
        return BurstDetection(fft_size=nfft)
    hop = nfft // 2
    freqs, times, sxx = sp_signal.spectrogram(
        x, fs=sample_rate, window="hann", nperseg=nfft, noverlap=nfft - hop,
        return_onesided=False, detrend=False, mode="psd")
    freqs = np.fft.fftshift(freqs)
    sxx = np.fft.fftshift(sxx, axes=0).astype(np.float64)     # (freq, time)
    n_bins, n_frames = sxx.shape
    bin_hz = sample_rate / nfft
    frame_s = hop / sample_rate

    smooth = ndimage.uniform_filter(sxx, size=(3, 3), mode="nearest")
    sdb = 10 * np.log10(smooth + 1e-300)
    live = live_bins(sdb.reshape(-1)).reshape(sdb.shape)
    floor_global = np.percentile(sdb[live], 20) if live.any() else float(np.min(sdb))
    floor_bin = np.minimum(np.percentile(sdb, 20, axis=1), floor_global)
    mask = sdb > floor_bin[:, None] + cfg.threshold_db
    mask &= live

    # Seeds: 5 bins × 9 frames averaging (≈ 90 degrees of freedom) makes
    # noise-only cells cluster tightly around the floor
    heavy = ndimage.uniform_filter(sxx, size=(5, 9), mode="nearest")
    hdb = 10 * np.log10(heavy + 1e-300)
    h_floor = np.minimum(np.percentile(hdb, 20, axis=1),
                         np.percentile(hdb[live], 20) if live.any() else np.min(hdb))
    seeds = (hdb > h_floor[:, None] + cfg.seed_threshold_db) & live

    # Bridge short gaps in time; keep only components containing a seed
    gap_frames = max(1, int(round(cfg.merge_time_s / frame_s)))
    mask = ndimage.binary_closing(mask, structure=np.ones((1, gap_frames + 1), bool))
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    if n == 0:
        return BurstDetection(noise_floor_db=float(floor_global), fft_size=nfft)
    seeded = np.unique(labels[seeds & (labels > 0)])
    keep = np.zeros(n + 1, dtype=bool)
    keep[seeded] = True
    labels = np.where(keep[labels], labels, 0)
    slices = ndimage.find_objects(labels)

    # (f0, f1, t0, t1) boxes, merged when time-overlapping and close in frequency
    merge_bins = max(3, int(round((cfg.merge_freq_hz or 0.02 * sample_rate) / bin_hz)))
    boxes = [[s[0].start, s[0].stop, s[1].start, s[1].stop] for s in slices if s is not None]
    max_complement_bins = n_bins          # anti-correlation is the real test

    def complementary(a: list[int], b: list[int]) -> bool:
        """Anti-correlated band envelopes at sample resolution (fast FSK
        alternates faster than a spectrogram frame)."""
        t0, t1 = max(a[2], b[2]), min(a[3], b[3])
        if t1 - t0 < 4 or max(a[0], b[0]) - min(a[1], b[1]) > max_complement_bins:
            return False
        if (t1 - t0) < 0.5 * min(a[3] - a[2], b[3] - b[2]):
            return False
        s0 = int(t0 * hop)
        s1 = int(min(len(x), s0 + min((t1 - t0) * hop, int(sample_rate))))   # ≤ 1 s
        seg = x[s0:s1]
        smooth_len = max(4, nfft // 32)
        envs = []
        for box in (a, b):
            centre = float(freqs[box[0]:box[1]].mean())
            width = float((box[1] - box[0]) * bin_hz)
            envs.append(_band_envelope(seg, sample_rate, centre, width, smooth_len))
        ea, eb = envs
        if np.std(ea) == 0 or np.std(eb) == 0:
            return False
        return float(np.corrcoef(ea, eb)[0, 1]) < -0.3

    merged = True
    while merged:
        merged = False
        out: list[list[int]] = []
        for b in sorted(boxes, key=lambda v: (v[2], v[0])):
            for o in out:
                t_overlap = b[2] <= o[3] + gap_frames and o[2] <= b[3] + gap_frames
                f_close = b[0] <= o[1] + merge_bins and o[0] <= b[1] + merge_bins
                if t_overlap and (f_close or complementary(o, b)):
                    o[0], o[1] = min(o[0], b[0]), max(o[1], b[1])
                    o[2], o[3] = min(o[2], b[2]), max(o[3], b[3])
                    merged = True
                    break
            else:
                out.append(list(b))
        boxes = out

    min_frames = max(2, int(np.ceil(cfg.min_duration_s / frame_s)))
    min_bins = max(1, int(np.ceil(cfg.min_bandwidth_hz / bin_hz)))
    bursts: list[Burst] = []
    active = np.zeros(n_frames, dtype=bool)
    for f0, f1, t0, t1 in boxes:
        if t1 - t0 < min_frames or f1 - f0 < min_bins:
            continue
        region = smooth[f0:f1, t0:t1]
        noise = 10 ** (floor_bin[f0:f1] / 10)
        snr = 10 * np.log10(max(np.mean(region) / np.mean(noise) - 1.0, 1e-3))
        start = int(t0 * hop)
        end = int(min(len(x), (t1 - 1) * hop + nfft))
        center = float(freqs[f0:f1].mean())
        bw = float((f1 - f0) * bin_hz)
        if cfg.refine_edges:
            start, end = _refine_edges(x, sample_rate, start, end, center, bw, nfft)
        if end - start < min_frames * hop:
            continue
        active[t0:t1] = True
        conf = float(np.clip(0.4 + snr / 30.0, 0.1, 0.99))
        bursts.append(Burst(start, end, center, bw, float(snr), conf))

    # Spectral skirts of a strong burst (weak, inside its time span, just
    # outside its band) are part of it, not separate signals
    def is_skirt(w: Burst, s: Burst) -> bool:
        gap = abs(w.center_hz - s.center_hz) - (w.bandwidth_hz + s.bandwidth_hz) / 2
        return (w is not s and s.snr_db - w.snr_db >= 6.0 and gap <= 0.1 * sample_rate
                and w.start_sample >= s.start_sample - hop and w.end_sample <= s.end_sample + hop)
    bursts = [w for w in bursts if not any(is_skirt(w, s) for s in bursts)]
    bursts.sort(key=lambda b: (b.start_sample, b.center_hz))
    bursts = bursts[: cfg.max_bursts]
    duty = float(active.mean()) if n_frames else 0.0
    continuous = len(bursts) == 1 and \
        bursts[0].num_samples >= cfg.continuous_fraction * len(x)
    return BurstDetection(bursts, duty, continuous, float(floor_global), nfft)


def _band_envelope(x: np.ndarray, sample_rate: float, center: float, bw: float,
                   smooth_len: int) -> np.ndarray:
    base = translate_frequency(x, sample_rate, -center)
    cutoff = float(np.clip(0.75 * bw + 2 * sample_rate / smooth_len, 0.01 * sample_rate,
                           0.49 * sample_rate))
    taps = sp_signal.firwin(129, cutoff / (sample_rate / 2))
    if len(base) > 3 * len(taps):
        base = sp_signal.filtfilt(taps, 1.0, base)
    power = np.abs(base) ** 2
    return np.convolve(power, np.ones(smooth_len) / smooth_len, mode="same")


def _refine_edges(x: np.ndarray, sample_rate: float, start: int, end: int, center: float,
                  bw: float, nfft: int) -> tuple[int, int]:
    """Move the coarse (frame-resolution) edges to where the band-limited
    envelope stays above half the burst level (−3 dB, or the geometric mean
    of burst and noise level if that is higher)."""
    guard = 2 * nfft
    smooth_len = max(8, nfft // 8)
    lo, hi = max(0, start - guard), min(len(x), end + guard)
    env = _band_envelope(x[lo:hi], sample_rate, center, bw, smooth_len)
    s, e = start - lo, end - lo
    inner = env[s + nfft // 2: max(s + nfft // 2 + 1, e - nfft // 2)]
    outer = np.concatenate([env[: max(0, s - nfft // 2)], env[min(len(env), e + nfft // 2):]])
    if len(inner) < 4 or len(outer) < 4:
        return start, end
    on, off = float(np.median(inner)), float(np.median(outer))
    if on <= 2 * off:
        return start, end
    thr = max(0.5 * on, np.sqrt(on * off))
    # "Stays above": a run of smooth_len samples, so noise spikes do not count
    above = (env > thr).astype(np.int64)
    run = np.convolve(above, np.ones(smooth_len, dtype=np.int64), mode="valid") == smooth_len
    if not run.any():
        return start, end
    idx = np.flatnonzero(run)
    first = int(idx[0])
    last = int(idx[-1]) + smooth_len
    if first >= e or last <= s:                    # nothing inside the coarse burst
        return start, end
    return lo + first, lo + last


@dataclass
class ExtractedBurst:
    samples: np.ndarray
    sample_rate: float
    offset_hz: float                     # frequency the burst was moved down by
    start_sample: int                    # in the original recording


def extract_burst(samples: np.ndarray, sample_rate: float, burst: Burst,
                  guard_s: float = 0.002, oversample: float = 6.0) -> ExtractedBurst:
    """Burst at 0 Hz, filtered to its own band and decimated to ≈
    *oversample* × its bandwidth."""
    x = np.asarray(samples, dtype=np.complex64).reshape(-1)
    guard = int(guard_s * sample_rate)
    s0, s1 = max(0, burst.start_sample - guard), min(len(x), burst.end_sample + guard)
    seg = translate_frequency(x[s0:s1], sample_rate, -burst.center_hz)
    bw = max(burst.bandwidth_hz, 4 * sample_rate / 1024)
    cutoff = min(0.75 * bw + 0.02 * sample_rate, 0.49 * sample_rate)
    taps = sp_signal.firwin(255, cutoff / (sample_rate / 2))
    if len(seg) > 3 * len(taps):
        seg = sp_signal.filtfilt(taps, 1.0, seg)
    fs_out = sample_rate
    target = oversample * bw
    if sample_rate > 1.5 * target:
        frac = Fraction(target / sample_rate).limit_denominator(64)
        if frac.numerator < frac.denominator:
            seg = sp_signal.resample_poly(seg, frac.numerator, frac.denominator)
            fs_out = sample_rate * frac.numerator / frac.denominator
    return ExtractedBurst(np.asarray(seg, dtype=np.complex64), float(fs_out),
                          float(burst.center_hz), int(s0))
