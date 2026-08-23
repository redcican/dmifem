"""Observation operators per sensor type (manuscript Section 3.2, Eqs. strainrow–incdisp).

All operators act on the free displacement vector ``u`` (or on the acceleration ``u_dd`` for
accelerometers) and are expressed in the structural frame. A strain gauge reads the fibre
strain of Eq. (strainrow); an accelerometer reads ``d^T O N(xi) u_dd + d^T O a_g``, with an
optional first-order tilt coupling ``d^T O [a_g]_x N_theta(xi) u`` on the displacement block
(Eq. tilt); an inclinometer reads ``d^T N_theta(xi) u`` and a displacement sensor
``d^T N(xi) u`` (Eq. incdisp).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dmifem.elements.base import Array
from dmifem.mesh import Mesh
from dmifem.sensors import Accelerometer, DisplacementSensor, Inclinometer, StrainGauge


@dataclass(frozen=True)
class ObservationOperator:
    """Rows of one sensor type on the displacement or acceleration block.

    Attributes
    ----------
    rows:
        Operator on the block named by ``block``, shape ``(m_s, n)``.
    tilt_rows:
        Rows on the displacement block from the tilt coupling (accelerometers only), or
        ``None``.
    offsets:
        Known offsets ``b`` per channel, shape ``(m_s,)``.
    sigma:
        Nominal noise standard deviation per channel, shape ``(m_s,)``.
    block:
        ``"displacement"`` or ``"acceleration"``.
    """

    rows: Array
    offsets: Array
    sigma: Array
    block: str
    tilt_rows: Array | None = None

    @property
    def n_channels(self) -> int:
        return self.rows.shape[0]


def _global_translation(mesh: Mesh, element: int, xi: float) -> Array:
    elem = mesh.elements[element]
    return elem.local_frame().T @ elem.translation_interpolation(xi) @ mesh.gather(element)


def _global_rotation(mesh: Mesh, element: int, xi: float) -> Array:
    elem = mesh.elements[element]
    return elem.local_frame().T @ elem.rotation_interpolation(xi) @ mesh.gather(element)


def _cross_matrix(v: Array) -> Array:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def strain_operator(mesh: Mesh, gauges: Sequence[StrainGauge]) -> ObservationOperator:
    """Operator ``H_eps`` of the strain gauges (one row per gauge)."""
    rows = np.zeros((len(gauges), mesh.n_free))
    for i, g in enumerate(gauges):
        elem = mesh.elements[g.element]
        rows[i] = elem.gauge_row(g.xi, g.y, g.z) @ mesh.gather(g.element)
    sigma = np.array([g.sigma for g in gauges], dtype=float)
    return ObservationOperator(rows, np.zeros(len(gauges)), sigma, "displacement")


def acceleration_operator(
    mesh: Mesh, sensors: Sequence[Accelerometer], tilt_coupling: bool = False
) -> ObservationOperator:
    """Operator ``H_a`` of the accelerometers with gravity offsets and optional tilt rows."""
    m = len(sensors)
    rows = np.zeros((m, mesh.n_free))
    tilt = np.zeros((m, mesh.n_free))
    offsets = np.zeros(m)
    for i, s in enumerate(sensors):
        d_o = s.unit_direction() @ s.orientation_matrix()
        rows[i] = d_o @ _global_translation(mesh, s.element, s.xi)
        gravity = np.asarray(s.gravity, dtype=float)
        offsets[i] = float(d_o @ gravity)
        if tilt_coupling:
            tilt[i] = d_o @ _cross_matrix(gravity) @ _global_rotation(mesh, s.element, s.xi)
    sigma = np.array([s.sigma for s in sensors], dtype=float)
    return ObservationOperator(rows, offsets, sigma, "acceleration", tilt if tilt_coupling else None)


def inclination_operator(mesh: Mesh, sensors: Sequence[Inclinometer]) -> ObservationOperator:
    """Operator ``H_theta`` of the inclinometers."""
    rows = np.zeros((len(sensors), mesh.n_free))
    for i, s in enumerate(sensors):
        rows[i] = s.unit_direction() @ _global_rotation(mesh, s.element, s.xi)
    sigma = np.array([s.sigma for s in sensors], dtype=float)
    return ObservationOperator(rows, np.zeros(len(sensors)), sigma, "displacement")


def displacement_operator(
    mesh: Mesh, sensors: Sequence[DisplacementSensor]
) -> ObservationOperator:
    """Operator ``H_d`` of the displacement sensors."""
    rows = np.zeros((len(sensors), mesh.n_free))
    for i, s in enumerate(sensors):
        rows[i] = s.unit_direction() @ _global_translation(mesh, s.element, s.xi)
    sigma = np.array([s.sigma for s in sensors], dtype=float)
    return ObservationOperator(rows, np.zeros(len(sensors)), sigma, "displacement")
