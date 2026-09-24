"""Synthetic signal generators for testing."""

from __future__ import annotations

import numpy as np


def generate_bpsk(
    num_symbols: int,
    symbol_rate: float,
    sample_rate: float,
    snr_db: float | None = None,
    freq_offset_hz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a BPSK signal.

    Returns
    -------
    (samples, bits)
    """
    bits = np.random.randint(0, 2, num_symbols)
    symbols = 2 * bits - 1  # Map to -1, +1

    sps = int(sample_rate / symbol_rate)

    # Upsample
    upsampled = np.zeros(num_symbols * sps)
    upsampled[::sps] = symbols

    # RRC Pulse shaping
    num_taps = 6 * sps + 1
    t = np.arange(num_taps) - (num_taps - 1) // 2
    # Simple raised cosine approximation
    beta = 0.35
    ts = sps
    rc = np.sinc(t / ts) * np.cos(np.pi * beta * t / ts) / (1 - (2 * beta * t / ts)**2 + 1e-10)

    sig = np.convolve(upsampled, rc, mode='same').astype(np.complex64)

    # Apply frequency offset
    if freq_offset_hz != 0.0:
        t_sec = np.arange(len(sig)) / sample_rate
        mixer = np.exp(1j * 2 * np.pi * freq_offset_hz * t_sec)
        sig = (sig * mixer).astype(np.complex64)

    if snr_db is not None:
        from .channels import add_awgn
        sig = add_awgn(sig, snr_db)

    return sig, bits


def generate_2fsk(
    num_symbols: int,
    symbol_rate: float,
    sample_rate: float,
    deviation_hz: float,
    snr_db: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a continuous-phase 2-FSK signal."""
    bits = np.random.randint(0, 2, num_symbols)
    symbols = 2 * bits - 1  # Map to -1, +1

    sps = int(sample_rate / symbol_rate)

    # NRZ shaping (rect pulse)
    upsampled = np.repeat(symbols, sps)

    # Integrate to phase (CPFSK)
    h = 2 * deviation_hz / symbol_rate  # modulation index
    phase_diff = upsampled * (np.pi * h / sps)
    phase = np.cumsum(phase_diff)

    sig = np.exp(1j * phase).astype(np.complex64)

    if snr_db is not None:
        from .channels import add_awgn
        sig = add_awgn(sig, snr_db)

    return sig, bits


def _rrc(sps: int, beta: float = 0.35, span: int = 6) -> np.ndarray:
    num_taps = span * sps + 1
    t = np.arange(num_taps) - (num_taps - 1) // 2
    rc = np.sinc(t / sps) * np.cos(np.pi * beta * t / sps) / (1 - (2 * beta * t / sps)**2 + 1e-10)
    return rc


def _gray(n: int) -> int:
    return n ^ (n >> 1)


def generate_qam16(
    num_symbols: int,
    symbol_rate: float,
    sample_rate: float,
    snr_db: float | None = None,
    freq_offset_hz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a Gray-coded 16-QAM signal (I bits then Q bits per symbol).

    Level mapping per axis: Gray(index) where levels = [-3, -1, 1, 3].
    Returns (samples, bits).
    """
    levels = np.array([-3.0, -1.0, 1.0, 3.0])
    gray_to_index = {_gray(i): i for i in range(4)}
    bits = np.random.randint(0, 2, num_symbols * 4)
    b = bits.reshape(-1, 4)
    i_code = b[:, 0] * 2 + b[:, 1]
    q_code = b[:, 2] * 2 + b[:, 3]
    i_lvl = levels[[gray_to_index[c] for c in i_code]]
    q_lvl = levels[[gray_to_index[c] for c in q_code]]
    symbols = (i_lvl + 1j * q_lvl) / np.sqrt(10.0)

    sps = int(sample_rate / symbol_rate)
    upsampled = np.zeros(num_symbols * sps, dtype=np.complex128)
    upsampled[::sps] = symbols
    sig = np.convolve(upsampled, _rrc(sps), mode='same').astype(np.complex64)

    if freq_offset_hz != 0.0:
        t_sec = np.arange(len(sig)) / sample_rate
        sig = (sig * np.exp(1j * 2 * np.pi * freq_offset_hz * t_sec)).astype(np.complex64)

    if snr_db is not None:
        from .channels import add_awgn
        sig = add_awgn(sig, snr_db)

    return sig, bits


def generate_qpsk(
    num_symbols: int,
    symbol_rate: float,
    sample_rate: float,
    snr_db: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a QPSK signal."""
    bits = np.random.randint(0, 2, num_symbols * 2)
    # Map to complex symbols: 00 -> 1+j, 01 -> -1+j, 10 -> 1-j, 11 -> -1-j
    # Scale by 1/sqrt(2) for unit power
    i_syms = 2 * bits[0::2] - 1
    q_syms = 2 * bits[1::2] - 1
    symbols = (i_syms + 1j * q_syms) / np.sqrt(2)

    sps = int(sample_rate / symbol_rate)

    # Upsample
    upsampled = np.zeros(num_symbols * sps, dtype=np.complex128)
    upsampled[::sps] = symbols

    # RRC Pulse shaping
    num_taps = 6 * sps + 1
    t = np.arange(num_taps) - (num_taps - 1) // 2
    beta = 0.35
    rc = np.sinc(t / sps) * np.cos(np.pi * beta * t / sps) / (1 - (2 * beta * t / sps)**2 + 1e-10)

    sig = np.convolve(upsampled, rc, mode='same').astype(np.complex64)

    if snr_db is not None:
        from .channels import add_awgn
        sig = add_awgn(sig, snr_db)

    return sig, bits


# ---------------------------------------------------------------------------
# Additional digital and analog modulations
# ---------------------------------------------------------------------------


def _finish(sig: np.ndarray, sample_rate: float, snr_db: float | None,
            freq_offset_hz: float) -> np.ndarray:
    if freq_offset_hz:
        sig = sig * np.exp(2j * np.pi * freq_offset_hz * np.arange(len(sig)) / sample_rate)
    sig = sig.astype(np.complex64)
    if snr_db is not None:
        from .channels import add_awgn
        sig = add_awgn(sig, snr_db)
    return sig


def _shape(symbols: np.ndarray, sps: int, beta: float = 0.35) -> np.ndarray:
    up = np.zeros(len(symbols) * sps, dtype=np.complex128)
    up[::sps] = symbols
    return np.convolve(up, _rrc(sps, beta), mode="same")


def generate_ask(
    num_symbols: int,
    symbol_rate: float,
    sample_rate: float,
    levels: tuple[float, ...] = (0.0, 1.0),
    snr_db: float | None = None,
    freq_offset_hz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Unipolar M-ASK (OOK when the lowest level is 0), Gray-coded,
    raised-cosine shaped.  Returns (samples, bits)."""
    m = len(levels)
    k = int(np.log2(m))
    bits = np.random.randint(0, 2, num_symbols * k)
    codes = bits.reshape(-1, k) @ (1 << np.arange(k - 1, -1, -1)) if k else np.zeros(num_symbols)
    gray_to_index = {_gray(i): i for i in range(m)}
    amp = np.asarray(levels, dtype=float)[[gray_to_index[int(c)] for c in codes]]
    sps = int(sample_rate / symbol_rate)
    return _finish(_shape(amp, sps), sample_rate, snr_db, freq_offset_hz), bits


def generate_msk(
    num_symbols: int,
    symbol_rate: float,
    sample_rate: float,
    bt: float | None = None,
    snr_db: float | None = None,
    freq_offset_hz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """MSK (h = 0.5), or GMSK when a Gaussian bandwidth-time product *bt*
    is given.  Bit 1 → +deviation.  Returns (samples, bits)."""
    bits = np.random.randint(0, 2, num_symbols)
    sps = int(sample_rate / symbol_rate)
    freq = np.repeat(2.0 * bits - 1.0, sps)
    if bt is not None:
        t = np.arange(-2 * sps, 2 * sps + 1) / sps
        sigma = np.sqrt(np.log(2)) / (2 * np.pi * bt)
        g = np.exp(-t ** 2 / (2 * sigma ** 2))
        freq = np.convolve(freq, g / g.sum(), mode="same")
    phase = np.cumsum(freq) * (np.pi * 0.5 / sps)
    return _finish(np.exp(1j * phase), sample_rate, snr_db, freq_offset_hz), bits


def generate_oqpsk(
    num_symbols: int,
    symbol_rate: float,
    sample_rate: float,
    snr_db: float | None = None,
    freq_offset_hz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Offset QPSK: the Q rail is delayed by half a symbol.  Returns (samples, bits)
    with bits ordered I, Q per symbol."""
    bits = np.random.randint(0, 2, num_symbols * 2)
    sps = int(sample_rate / symbol_rate)
    i = _shape(2.0 * bits[0::2] - 1.0, sps).real
    q = _shape(2.0 * bits[1::2] - 1.0, sps).real
    q = np.concatenate([np.zeros(sps // 2), q[: len(q) - sps // 2]])
    return _finish((i + 1j * q) / np.sqrt(2), sample_rate, snr_db, freq_offset_hz), bits


def generate_pi4_dqpsk(
    num_symbols: int,
    symbol_rate: float,
    sample_rate: float,
    snr_db: float | None = None,
    freq_offset_hz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """π/4-DQPSK (TETRA / IS-54 style): dibits 00, 01, 11, 10 → phase steps
    +π/4, +3π/4, −3π/4, −π/4.  Returns (samples, bits)."""
    bits = np.random.randint(0, 2, num_symbols * 2)
    steps = {(0, 0): np.pi / 4, (0, 1): 3 * np.pi / 4, (1, 1): -3 * np.pi / 4,
             (1, 0): -np.pi / 4}
    dphi = np.array([steps[(int(a), int(b))] for a, b in zip(bits[0::2], bits[1::2], strict=True)])
    symbols = np.exp(1j * np.cumsum(dphi))
    sps = int(sample_rate / symbol_rate)
    return _finish(_shape(symbols, sps), sample_rate, snr_db, freq_offset_hz), bits


def generate_dbpsk(
    num_symbols: int,
    symbol_rate: float,
    sample_rate: float,
    snr_db: float | None = None,
    freq_offset_hz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Differentially encoded BPSK: bit 1 flips the phase.  Returns (samples, bits)."""
    bits = np.random.randint(0, 2, num_symbols)
    symbols = np.where(np.cumsum(bits) % 2 == 0, 1.0, -1.0)
    sps = int(sample_rate / symbol_rate)
    return _finish(_shape(symbols, sps), sample_rate, snr_db, freq_offset_hz), bits


def speech_like(num_samples: int, sample_rate: float, seed: int | None = None) -> np.ndarray:
    """Band-limited (300–3000 Hz) noise with a speech-like spectral tilt and
    a 3–6 Hz syllabic envelope; unit RMS.  A stand-in for voice."""
    rng = np.random.default_rng(seed)
    n = num_samples
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sample_rate)
    shape = np.where((f > 300) & (f < 3000), 1.0 / np.maximum(f, 500) * 500, 0.0)
    x = np.fft.irfft(spec * shape, n)
    t = np.arange(n) / sample_rate
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t) * np.sin(2 * np.pi * 0.7 * t + 1.0)
    x = x * env
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12)


def generate_am(
    num_samples: int,
    sample_rate: float,
    mod_index: float = 0.5,
    message: np.ndarray | None = None,
    snr_db: float | None = None,
    freq_offset_hz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Double-sideband AM with carrier.  Returns (samples, message)."""
    msg = speech_like(num_samples, sample_rate) if message is None else message
    msg = msg / (np.max(np.abs(msg)) + 1e-12)
    sig = (1.0 + mod_index * msg).astype(np.complex128)
    return _finish(sig, sample_rate, snr_db, freq_offset_hz), msg


def generate_fm(
    num_samples: int,
    sample_rate: float,
    deviation_hz: float = 3000.0,
    message: np.ndarray | None = None,
    snr_db: float | None = None,
    freq_offset_hz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Analog FM (peak deviation *deviation_hz*).  Returns (samples, message)."""
    msg = speech_like(num_samples, sample_rate) if message is None else message
    msg = msg / (np.max(np.abs(msg)) + 1e-12)
    phase = 2 * np.pi * deviation_hz * np.cumsum(msg) / sample_rate
    return _finish(np.exp(1j * phase), sample_rate, snr_db, freq_offset_hz), msg


def generate_ssb(
    num_samples: int,
    sample_rate: float,
    sideband: str = "usb",
    message: np.ndarray | None = None,
    snr_db: float | None = None,
    carrier_hz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Suppressed-carrier single sideband.  The (suppressed) carrier sits at
    *carrier_hz*; USB puts the audio above it, LSB below.  Returns
    (samples, message)."""
    from scipy.signal import hilbert

    msg = speech_like(num_samples, sample_rate) if message is None else message
    analytic = hilbert(msg)
    sig = analytic if sideband.lower() == "usb" else np.conj(analytic)
    return _finish(sig, sample_rate, snr_db, carrier_hz), msg
