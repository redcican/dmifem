"""Two-node inverse beam element with axial, bending and torsional section strains.

Transverse displacements use cubic Hermite interpolation, the axial displacement and the twist
use linear interpolation. Local degrees of freedom per node are ``[u_x, u_y, u_z, t_x, t_y,
t_z]`` with ``t_z = du_y/dx`` and ``t_y = -du_z/dx``. The transverse-shear section strains of
the Euler–Bernoulli kinematics vanish identically; their rows are kept (as zeros) so that the
operator layout matches the shear-deformable elements of the manuscript.
"""

from __future__ import annotations

import numpy as np

from dmifem.elements.base import Array, InverseElement, SectionStrainOperators

_UY_DOFS = (1, 5, 7, 11)  # u_y1, t_z1, u_y2, t_z2
_UZ_DOFS = (2, 4, 8, 10)  # u_z1, t_y1, u_z2, t_y2
_UZ_SIGNS = np.array([1.0, -1.0, 1.0, -1.0])
_GAUSS_POINTS = np.array([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)])


def _hermite(s: float, length: float) -> tuple[Array, Array, Array]:
    """Return the cubic Hermite functions and their first and second derivatives in ``s``."""
    h = np.array(
        [
            1.0 - 3.0 * s**2 + 2.0 * s**3,
            length * (s - 2.0 * s**2 + s**3),
            3.0 * s**2 - 2.0 * s**3,
            length * (-(s**2) + s**3),
        ]
    )
    dh = np.array(
        [
            -6.0 * s + 6.0 * s**2,
            length * (1.0 - 4.0 * s + 3.0 * s**2),
            6.0 * s - 6.0 * s**2,
            length * (-2.0 * s + 3.0 * s**2),
        ]
    )
    ddh = np.array(
        [
            -6.0 + 12.0 * s,
            length * (-4.0 + 6.0 * s),
            6.0 - 12.0 * s,
            length * (-2.0 + 6.0 * s),
        ]
    )
    return h, dh, ddh


class InverseBeam(InverseElement):
    """Inverse beam element between two nodes.

    Parameters
    ----------
    coords:
        Global coordinates of the two nodes, shape ``(2, 3)``.
    reference:
        Vector defining the local ``y`` axis (projected orthogonally to the element axis).
        Defaults to the global ``z`` axis, or to the global ``y`` axis for elements parallel to
        ``z``.
    """

    n_nodes = 2

    def __init__(self, coords: Array, reference: Array | None = None) -> None:
        coords = np.asarray(coords, dtype=float).reshape(2, 3)
        axis = coords[1] - coords[0]
        self.length = float(np.linalg.norm(axis))
        if self.length <= 0.0:
            raise ValueError("beam element has zero length")
        e_x = axis / self.length
        ref = np.array([0.0, 0.0, 1.0]) if reference is None else np.asarray(reference, float)
        if abs(np.dot(ref, e_x)) > 0.99 * np.linalg.norm(ref):
            ref = np.array([0.0, 1.0, 0.0])
        e_y = ref - np.dot(ref, e_x) * e_x
        e_y /= np.linalg.norm(e_y)
        e_z = np.cross(e_x, e_y)
        self.coords = coords
        self._frame = np.vstack([e_x, e_y, e_z])

    # ---- interface -------------------------------------------------------------------------
    def local_frame(self) -> Array:
        return self._frame.copy()

    def quadrature(self) -> tuple[Array, Array]:
        return _GAUSS_POINTS.copy(), np.full(2, 0.5 * self.length)

    def section_strain_operators(self, xi: float) -> SectionStrainOperators:
        length = self.length
        _, _, ddh = _hermite(xi, length)
        membrane = np.zeros((1, 12))
        membrane[0, 0], membrane[0, 6] = -1.0 / length, 1.0 / length
        bending = np.zeros((3, 12))
        bending[0, list(_UZ_DOFS)] = _UZ_SIGNS * ddh / length**2  # curvature kappa_y = u_z''
        bending[1, list(_UY_DOFS)] = ddh / length**2  # curvature kappa_z = u_y''
        bending[2, 3], bending[2, 9] = -1.0 / length, 1.0 / length  # twist rate
        shear = np.zeros((2, 12))
        return SectionStrainOperators(membrane=membrane, bending=bending, shear=shear)

    def translation_interpolation(self, xi: float) -> Array:
        h, _, _ = _hermite(xi, self.length)
        n = np.zeros((3, 12))
        n[0, 0], n[0, 6] = 1.0 - xi, xi
        n[1, list(_UY_DOFS)] = h
        n[2, list(_UZ_DOFS)] = _UZ_SIGNS * h
        return n

    def rotation_interpolation(self, xi: float) -> Array:
        _, dh, _ = _hermite(xi, self.length)
        n = np.zeros((3, 12))
        n[0, 3], n[0, 9] = 1.0 - xi, xi
        n[1, list(_UZ_DOFS)] = -_UZ_SIGNS * dh / self.length  # t_y = -u_z'
        n[2, list(_UY_DOFS)] = dh / self.length  # t_z = u_y'
        return n

    def gauge_row(self, xi: float, *position: float) -> Array:
        """Surface strain of a fibre at local section offsets ``(y, z)``: ``e - y k_z - z k_y``."""
        if len(position) != 2:
            raise ValueError("a beam gauge position is (y, z)")
        y, z = position
        ops = self.section_strain_operators(xi)
        return ops.membrane[0] - z * ops.bending[0] - y * ops.bending[1]

    def point(self, xi: float) -> Array:
        """Global coordinates of the point at natural coordinate ``xi``."""
        return (1.0 - xi) * self.coords[0] + xi * self.coords[1]
