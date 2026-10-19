"""Dimensionless coordinates (manuscript Section 3.3, Eq. scaling).

With a reference length ``ell`` and a reference time ``T = 1 / (2 pi f_ref)``, the free degrees of
freedom are ``u_bar = S^{-1} u`` with ``S = diag(ell I_t, I_r)`` (translations divided by ``ell``,
rotations unchanged), time is ``t_bar = t / T`` and the state holds
``[u_bar, T du_bar/dt, T^2 d2u_bar/dt2]``. The prior (:mod:`dmifem.prior`) multiplies curvature and
twist by ``ell`` and divides the quadrature weights by the total measure of the mesh, and the
estimator divides every channel by its nominal noise standard deviation. A change of units
multiplies the data by known factors and leaves every dimensionless quantity unchanged.

Default rules: ``ell`` is the outer radius of the tubular sections when they are given, otherwise
the largest fibre offset of the strain gauges; ``f_ref`` is the dominant frequency of the
accelerometer spectrum, or of the strain spectrum when no accelerometer is installed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import welch

from dmifem.elements.base import Array
from dmifem.mesh import DOFS_PER_NODE, Mesh
from dmifem.sensors import SensorData, SensorType, StrainGauge

logger = logging.getLogger(__name__)

TRANSLATION_DOFS = (0, 1, 2)
_TIME_SOURCES = (
    SensorType.ACCELERATION,
    SensorType.STRAIN,
    SensorType.DISPLACEMENT,
    SensorType.INCLINATION,
)
_MIN_SAMPLES = 8


@dataclass(frozen=True)
class Scaling:
    """Reference length ``ell`` and reference time ``T`` of Eq. (scaling), in the units of the data."""

    length: float
    time: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.length) and self.length > 0.0):
            raise ValueError("reference length must be positive")
        if not (np.isfinite(self.time) and self.time > 0.0):
            raise ValueError("reference time must be positive")

    @classmethod
    def from_frequency(cls, length: float, f_ref: float) -> Scaling:
        """Scaling with ``T = 1 / (2 pi f_ref)``."""
        if not f_ref > 0.0:
            raise ValueError("reference frequency must be positive")
        return cls(float(length), 1.0 / (2.0 * np.pi * float(f_ref)))

    @property
    def reference_frequency(self) -> float:
        """``f_ref = 1 / (2 pi T)``."""
        return 1.0 / (2.0 * np.pi * self.time)


def dof_scale(mesh: Mesh, length: float) -> Array:
    """Diagonal of ``S`` over the free degrees of freedom: ``ell`` on translations, 1 on rotations."""
    local = mesh.free_dofs % DOFS_PER_NODE
    return np.where(np.isin(local, TRANSLATION_DOFS), float(length), 1.0)


def default_length(
    gauges: Sequence[StrainGauge] = (),
    sections: Sequence[Any] | None = None,
    fallback: float = 1.0,
) -> float:
    """Reference length: outer radius of the sections, else the largest gauge offset, else ``fallback``.

    The estimator passes the length of the structure as ``fallback``, so the reference length is a
    length of the mesh in every case and the estimates do not depend on the units.

    ``sections`` holds one object per element with an ``outer_radius`` attribute (``None`` when
    unknown), such as :class:`dmifem.prior.BeamSection`. When the sections at the gauges carry an
    outer radius, the largest of them is used; otherwise the largest radius of all sections.
    """
    if sections is not None:
        at_gauges = [getattr(sections[g.element], "outer_radius", None) for g in gauges]
        if at_gauges and all(r is not None and r > 0.0 for r in at_gauges):
            return float(max(at_gauges))
        radii = [getattr(s, "outer_radius", None) for s in sections]
        radii = [r for r in radii if r is not None and r > 0.0]
        if radii:
            return float(max(radii))
    offsets = [float(np.hypot(g.y, g.z)) for g in gauges]
    if offsets and max(offsets) > 0.0:
        return max(offsets)
    if not fallback > 0.0:
        raise ValueError("the fallback reference length must be positive")
    logger.info("no section radius or gauge offset: reference length %.4g from the fallback", fallback)
    return float(fallback)


def dominant_frequency(times: Array, values: Array) -> float:
    """Peak of the channel-averaged Welch spectrum, each channel normalized to unit power.

    The zero-frequency bin is excluded; missing samples are set to the channel mean.
    """
    t = np.asarray(times, dtype=float).reshape(-1)
    if t.size < _MIN_SAMPLES:
        raise ValueError(f"at least {_MIN_SAMPLES} samples are needed for the reference frequency")
    v = np.asarray(values, dtype=float).reshape(t.size, -1)
    finite = np.isfinite(v)
    counts = np.maximum(finite.sum(axis=0), 1)
    means = np.where(finite, v, 0.0).sum(axis=0) / counts
    v = np.where(finite, v - means, 0.0)
    dt = float(np.median(np.diff(t)))
    f, p = welch(v, fs=1.0 / dt, nperseg=int(min(256, t.size)), axis=0)
    power = p.sum(axis=0)
    active = power > 0.0
    if not np.any(active):
        raise ValueError("the channels have no variance")
    spectrum = (p[:, active] / power[active]).mean(axis=1)
    spectrum[f <= 0.0] = 0.0
    return float(f[int(np.argmax(spectrum))])


def default_time_scale(data: SensorData) -> float:
    """``T = 1 / (2 pi f_ref)`` from the accelerometer stream, else strain, displacement, inclination.

    Streams that are too short or carry no variance (all samples missing or constant) are skipped.
    """
    for stype in _TIME_SOURCES:
        if stype not in data.streams:
            continue
        t, v = data.streams[stype]
        if t.size < _MIN_SAMPLES:
            continue
        try:
            f_ref = dominant_frequency(t, v)
        except ValueError:
            logger.info("%s stream has no variance; trying the next source for the reference time", stype.value)
            continue
        if f_ref > 0.0:
            return 1.0 / (2.0 * np.pi * f_ref)
    raise ValueError("no stream sets the reference time; pass a Scaling to the estimator")
