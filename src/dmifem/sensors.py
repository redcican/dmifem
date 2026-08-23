"""Sensor descriptions and measured data (manuscript Section 2.3 and Table 3).

Every sensor is attached to an element at a natural coordinate ``xi``. Sensor types are
sampled on their own time grids; :class:`SensorData` holds, per type, the sampling instants
and the channel values in the order in which the sensors are listed in the estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from dmifem.elements.base import Array


class SensorType(str, Enum):
    """The four admitted sensor types."""

    STRAIN = "strain"
    ACCELERATION = "acceleration"
    INCLINATION = "inclination"
    DISPLACEMENT = "displacement"


@dataclass(frozen=True)
class StrainGauge:
    """Surface strain gauge or fibre point at section offsets ``(y, z)`` of a beam element."""

    element: int
    xi: float
    y: float
    z: float
    sigma: float


@dataclass(frozen=True)
class Accelerometer:
    """Accelerometer measuring along ``direction`` (structural frame, unit vector).

    ``orientation`` is the matrix ``O_i`` relating the sensor frame to the structural frame
    (identity by default); ``gravity`` is the gravity-induced offset vector in the structural
    frame (zero by default), projected on the sensing direction as the known offset ``b_i^a``.
    """

    element: int
    xi: float
    direction: tuple[float, float, float]
    sigma: float
    orientation: tuple[tuple[float, float, float], ...] | None = None
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def unit_direction(self) -> Array:
        d = np.asarray(self.direction, dtype=float)
        return d / np.linalg.norm(d)

    def orientation_matrix(self) -> Array:
        if self.orientation is None:
            return np.eye(3)
        return np.asarray(self.orientation, dtype=float).reshape(3, 3)


@dataclass(frozen=True)
class Inclinometer:
    """Inclinometer reading the rotation about ``direction``."""

    element: int
    xi: float
    direction: tuple[float, float, float]
    sigma: float

    def unit_direction(self) -> Array:
        d = np.asarray(self.direction, dtype=float)
        return d / np.linalg.norm(d)


@dataclass(frozen=True)
class DisplacementSensor:
    """Displacement transducer reading the translation along ``direction``."""

    element: int
    xi: float
    direction: tuple[float, float, float]
    sigma: float

    def unit_direction(self) -> Array:
        d = np.asarray(self.direction, dtype=float)
        return d / np.linalg.norm(d)


@dataclass
class SensorData:
    """Measured data per sensor type: ``{type: (times, values)}`` with ``values`` of shape
    ``(n_samples, n_channels)``. Missing samples are encoded as ``nan`` and are deleted from
    the update at that instant."""

    streams: dict[SensorType, tuple[Array, Array]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cleaned: dict[SensorType, tuple[Array, Array]] = {}
        for stype, (times, values) in self.streams.items():
            t = np.asarray(times, dtype=float).reshape(-1)
            v = np.asarray(values, dtype=float).reshape(t.size, -1)
            if np.any(np.diff(t) <= 0):
                raise ValueError(f"sampling instants of {stype.value} must increase strictly")
            cleaned[SensorType(stype)] = (t, v)
        self.streams = cleaned

    def types(self) -> list[SensorType]:
        return list(self.streams)

    def window(self, t_start: float, t_end: float) -> SensorData:
        """Restrict every stream to ``t_start <= t <= t_end``."""
        out: dict[SensorType, tuple[Array, Array]] = {}
        for stype, (t, v) in self.streams.items():
            mask = (t >= t_start) & (t <= t_end)
            out[stype] = (t[mask], v[mask])
        return SensorData(out)
