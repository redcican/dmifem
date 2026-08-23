"""Abstract interface of inverse elements (manuscript Sections 2.1 and 3.2).

An inverse element interpolates the displacement field from nodal degrees of freedom and
provides the section-strain operators ``B_m`` (membrane/axial), ``B_b`` (bending/twist) and
``B_s`` (transverse shear) of Eq. (sectionstrains), the interpolation of translations and
rotations used by the observation operators, and the quadrature rule of the element norms.
All quantities are expressed in the local element frame; :meth:`transformation` maps the
global nodal degrees of freedom of the element to the local ones.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import block_diag

Array = NDArray[np.float64]


@dataclass(frozen=True)
class SectionStrainOperators:
    """Section-strain operators of one element at one point (local frame).

    Attributes
    ----------
    membrane:
        Operator of the membrane (axial) section strains, shape ``(n_m, n_dofs)``.
    bending:
        Operator of the bending section strains (curvatures and twist), shape ``(n_b, n_dofs)``.
    shear:
        Operator of the transverse-shear section strains, shape ``(n_s, n_dofs)``.
    """

    membrane: Array
    bending: Array
    shear: Array

    def stacked(self) -> Array:
        """Return the three operators stacked row-wise."""
        return np.vstack([self.membrane, self.bending, self.shear])


class InverseElement(ABC):
    """Base class of inverse elements.

    Subclasses define the interpolation and the section strains of one element type. The
    parameter ``xi`` is a natural coordinate along or over the element in ``[0, 1]`` (beam) or
    in the reference domain of the element (shell).
    """

    n_nodes: int
    dofs_per_node: int = 6

    @property
    def n_dofs(self) -> int:
        """Number of local degrees of freedom of the element."""
        return self.n_nodes * self.dofs_per_node

    @abstractmethod
    def section_strain_operators(self, xi: float) -> SectionStrainOperators:
        """Return ``B_m``, ``B_b`` and ``B_s`` at the natural coordinate ``xi``."""

    @abstractmethod
    def translation_interpolation(self, xi: float) -> Array:
        """Return the ``(3, n_dofs)`` operator giving the local translations at ``xi``."""

    @abstractmethod
    def rotation_interpolation(self, xi: float) -> Array:
        """Return the ``(3, n_dofs)`` operator giving the local rotations at ``xi``."""

    @abstractmethod
    def quadrature(self) -> tuple[Array, Array]:
        """Return the quadrature points (natural coordinates) and weights of the element norms."""

    @abstractmethod
    def local_frame(self) -> Array:
        """Return the ``(3, 3)`` matrix whose rows are the local axes in global coordinates."""

    @abstractmethod
    def gauge_row(self, xi: float, *position: float) -> Array:
        """Return the surface-strain row of a gauge at ``xi`` and the given section position.

        For a beam the position is ``(y, z)`` in the local section frame; the row combines the
        axial strain and the curvatures as in Eq. (strainrow) of the manuscript.
        """

    def transformation(self) -> Array:
        """Return the block-diagonal rotation mapping global nodal dofs to local ones."""
        rotation = self.local_frame()
        return block_diag(*([rotation] * (2 * self.n_nodes)))
