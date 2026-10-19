"""Rainflow counting, damage-equivalent load and band substitution (manuscript Sections 3.6, 4.3).

The damage ratio of two histories is ``r_D = r_DEL^m``, with ``r_DEL`` the ratio of their
damage-equivalent loads. Band substitution replaces one frequency band of a reference history by
the same band of a reconstruction and repeats the broadband rainflow count; the band ratios are
not additive, because rainflow cycles combine content from all bands.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dmifem.elements.base import Array


def reversals(series: Array) -> Array:
    """Return the sequence of local extrema (peaks and valleys) of a series.

    Repeated values (plateaus) are collapsed before the extrema are extracted.
    """
    x = np.asarray(series, dtype=float)
    if x.size < 3:
        return x.copy()
    x = x[np.concatenate([[True], np.diff(x) != 0.0])]
    if x.size < 3:
        return x.copy()
    d = np.diff(x)
    keep = np.ones(x.size, dtype=bool)
    keep[1:-1] = np.sign(d[:-1]) != np.sign(d[1:])
    return x[keep]


def rainflow(series: Array) -> tuple[Array, Array, Array]:
    """Three-point rainflow counting (ASTM E1049 procedure).

    Returns the cycle ranges, means and counts (``1.0`` for full cycles, ``0.5`` for the
    residual half cycles).
    """
    points = list(reversals(series))
    ranges, means, counts = [], [], []
    stack: list[float] = []
    for value in points:
        stack.append(value)
        while len(stack) >= 3:
            x = abs(stack[-1] - stack[-2])
            y = abs(stack[-2] - stack[-3])
            if x < y:
                break
            if len(stack) == 3:
                ranges.append(y)
                means.append(0.5 * (stack[0] + stack[1]))
                counts.append(0.5)
                stack.pop(0)
            else:
                ranges.append(y)
                means.append(0.5 * (stack[-2] + stack[-3]))
                counts.append(1.0)
                last = stack.pop()
                stack.pop()
                stack.pop()
                stack.append(last)
    for i in range(len(stack) - 1):
        ranges.append(abs(stack[i + 1] - stack[i]))
        means.append(0.5 * (stack[i + 1] + stack[i]))
        counts.append(0.5)
    return np.array(ranges), np.array(means), np.array(counts)


def damage_equivalent_load(ranges: Array, counts: Array, slope: float, n_ref: float) -> float:
    """Damage-equivalent load ``(sum n_i dS_i^m / N_ref)^(1/m)`` of Eq. (del)."""
    ranges = np.asarray(ranges, dtype=float)
    counts = np.asarray(counts, dtype=float)
    return float((np.sum(counts * ranges**slope) / n_ref) ** (1.0 / slope))


def del_of_series(series: Array, slope: float, n_ref: float) -> float:
    """Rainflow count a history and return its damage-equivalent load."""
    ranges, _, counts = rainflow(series)
    return damage_equivalent_load(ranges, counts, slope, n_ref)


def damage_ratio(del_ratio: float | Array, slope: float) -> float | Array:
    """Damage ratio ``r_D = r_DEL^m`` for a ratio of damage-equivalent loads."""
    return np.asarray(del_ratio, dtype=float) ** slope if np.ndim(del_ratio) else float(del_ratio) ** slope


def band_substitution(
    reference: Array,
    reconstruction: Array,
    fs: float,
    bands: Sequence[tuple[float, float]],
    slope: float,
    n_ref: float = 1.0,
) -> tuple[Array, float]:
    """Damage ratio when one band of ``reference`` is replaced by the band of ``reconstruction``.

    Bands are half-open ``[f_low, f_high)`` in the frequency unit of ``fs`` (use ``np.inf`` to
    include the Nyquist bin). Returns the damage ratio per band and the ratio of the full
    reconstruction, ``(DEL_rec / DEL_ref)^m``, which equals the substitution of all bands when the
    bands cover the spectrum.
    """
    ref = np.asarray(reference, dtype=float).reshape(-1)
    rec = np.asarray(reconstruction, dtype=float).reshape(-1)
    if ref.size != rec.size:
        raise ValueError("reference and reconstruction must have the same length")
    del_ref = del_of_series(ref, slope, n_ref)
    if del_ref <= 0.0:
        raise ValueError("the reference history has no cycles")
    freq = np.fft.rfftfreq(ref.size, 1.0 / fs)
    spec_ref = np.fft.rfft(ref)
    spec_rec = np.fft.rfft(rec)
    ratios = []
    for low, high in bands:
        mask = (freq >= low) & (freq < high)
        mixed = np.where(mask, spec_rec, spec_ref)
        series = np.fft.irfft(mixed, ref.size)
        ratios.append(float(del_of_series(series, slope, n_ref) / del_ref) ** slope)
    total = float(del_of_series(rec, slope, n_ref) / del_ref) ** slope
    return np.array(ratios), total
