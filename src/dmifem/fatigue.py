"""Rainflow counting and damage-equivalent load (manuscript Section 3.6, Eq. del)."""

from __future__ import annotations

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
