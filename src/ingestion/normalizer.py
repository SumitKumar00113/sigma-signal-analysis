"""Format normalizer — convert any supported input into complex64 samples.

This module handles the type conversions, scaling, IQ re-ordering, and
channel extraction needed to present a uniform ``complex64`` view to the
rest of the DSP pipeline (PRD §6.3).
"""

from __future__ import annotations

import numpy as np

from src.core.enums import IQOrder, SampleDatatype

# ---------------------------------------------------------------------------
# Dtype mapping
# ---------------------------------------------------------------------------

# Maps SampleDatatype → (numpy dtype for ONE component, is_complex, is_unsigned)
_DTYPE_MAP: dict[SampleDatatype, tuple[np.dtype, bool, bool]] = {
    # Complex float
    SampleDatatype.CF32_LE: (np.dtype("<f4"), True, False),
    SampleDatatype.CF32_BE: (np.dtype(">f4"), True, False),
    SampleDatatype.CF64_LE: (np.dtype("<f8"), True, False),
    # Complex int
    SampleDatatype.CI8:     (np.dtype("i1"), True, False),
    SampleDatatype.CI16_LE: (np.dtype("<i2"), True, False),
    SampleDatatype.CI16_BE: (np.dtype(">i2"), True, False),
    SampleDatatype.CI32_LE: (np.dtype("<i4"), True, False),
    SampleDatatype.CU8:     (np.dtype("u1"), True, True),
    SampleDatatype.CU16_LE: (np.dtype("<u2"), True, True),
    # Real float
    SampleDatatype.RF32_LE: (np.dtype("<f4"), False, False),
    SampleDatatype.RF64_LE: (np.dtype("<f8"), False, False),
    # Real int
    SampleDatatype.RI8:     (np.dtype("i1"), False, False),
    SampleDatatype.RI16_LE: (np.dtype("<i2"), False, False),
    SampleDatatype.RI32_LE: (np.dtype("<i4"), False, False),
    SampleDatatype.RU8:     (np.dtype("u1"), False, True),
    SampleDatatype.RU16_LE: (np.dtype("<u2"), False, True),
}


def bytes_per_sample(datatype: SampleDatatype) -> int:
    """Return the number of bytes for one sample (one complex or one real)."""
    dt, is_complex, _ = _DTYPE_MAP[datatype]
    return dt.itemsize * (2 if is_complex else 1)


def normalize_samples(
    raw: np.ndarray | bytes | memoryview,
    datatype: SampleDatatype,
    iq_order: IQOrder = IQOrder.IQ,
    amplitude_scale: float = 1.0,
) -> np.ndarray:
    """Convert raw sample data to a 1-D ``complex64`` array.

    Parameters
    ----------
    raw:
        Raw bytes, memoryview, or numpy array.
    datatype:
        The :class:`SampleDatatype` describing the binary format.
    iq_order:
        Whether samples are interleaved as I,Q or Q,I.
    amplitude_scale:
        Multiplicative scaling applied after conversion.

    Returns
    -------
    numpy.ndarray
        1-D ``complex64`` array.
    """
    dt, is_complex, is_unsigned = _DTYPE_MAP[datatype]

    # Ensure we have a numpy array of the component dtype
    if isinstance(raw, (bytes, memoryview)):
        arr = np.frombuffer(raw, dtype=dt)
    elif raw.dtype != dt:
        arr = raw.view(dt)
    else:
        arr = raw

    if is_complex:
        # Interleaved: I0,Q0,I1,Q1,…
        if arr.size % 2 != 0:
            arr = arr[: arr.size - 1]  # drop trailing incomplete sample
        iq = arr.reshape(-1, 2).astype(np.float32)

        # Unsigned integers → centre at zero
        if is_unsigned:
            max_val = float(np.iinfo(dt).max)
            iq = iq - (max_val + 1) / 2.0

        # IQ ordering
        if iq_order == IQOrder.QI:
            iq = iq[:, ::-1].copy()

        samples = iq[:, 0] + 1j * iq[:, 1]
    else:
        # Real-only → imaginary = 0
        samples = arr.astype(np.float32)
        if is_unsigned:
            max_val = float(np.iinfo(dt).max)
            samples = samples - (max_val + 1) / 2.0
        samples = samples + 0j

    result = samples.astype(np.complex64)

    if amplitude_scale != 1.0:
        result *= amplitude_scale

    return result
