"""Union grid of asynchronous sampling sets (manuscript Section 2.3).

The union of the sampling instants of all sensor types defines the estimation grid. At each
instant the types sampled there contribute rows; the others contribute none.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dmifem.elements.base import Array
from dmifem.sensors import SensorType


@dataclass(frozen=True)
class UnionGrid:
    """Sorted union of sampling instants with, per instant, the sample index of each type.

    Attributes
    ----------
    times:
        Instants ``t_1 < ... < t_N``.
    samples:
        For each type an integer array of length ``N`` holding the index of the sample of that
        type at the instant, or ``-1`` when the type is not sampled there.
    """

    times: Array
    samples: dict[SensorType, Array]

    @property
    def n(self) -> int:
        return self.times.size

    def intervals(self) -> Array:
        """Sampling intervals ``dt_k = t_{k+1} - t_k`` (length ``N - 1``)."""
        return np.diff(self.times)

    def active_types(self, k: int) -> list[SensorType]:
        return [stype for stype, idx in self.samples.items() if idx[k] >= 0]


def build_union_grid(sample_times: dict[SensorType, Array], tol: float = 1e-9) -> UnionGrid:
    """Merge the sampling sets; instants closer than ``tol`` are identified."""
    all_times = np.concatenate([np.asarray(t, dtype=float) for t in sample_times.values()])
    merged = np.sort(all_times)
    keep = np.ones(merged.size, dtype=bool)
    keep[1:] = np.diff(merged) > tol
    times = merged[keep]
    samples: dict[SensorType, Array] = {}
    for stype, t in sample_times.items():
        t = np.asarray(t, dtype=float)
        idx = np.full(times.size, -1, dtype=int)
        pos = np.searchsorted(times, t)
        pos = np.clip(pos, 0, times.size - 1)
        left = np.clip(pos - 1, 0, times.size - 1)
        choose_left = np.abs(times[left] - t) < np.abs(times[pos] - t)
        pos = np.where(choose_left, left, pos)
        idx[pos] = np.arange(t.size)
        samples[SensorType(stype)] = idx
    return UnionGrid(times, samples)
