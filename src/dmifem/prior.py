"""Spatial prior and static iFEM solve (manuscript Sections 2.1, 3.3 and Proposition 3.1).

The unit-weight section-strain operator ``L`` stacks, for every element and quadrature point,
the rows ``sqrt(w_q) B(xi_q) T_e`` so that ``K1 = L^T L`` is the unit-weight iFEM matrix of
Eq. (prior) with the element norms evaluated by quadrature. With the classical weights
``w_m, w_b, w_s`` applied per row, ``K = L^T W L`` and ``f = L^T W y`` reproduce Eq. (assembly),
and :func:`static_ifem` returns the posterior of Proposition 3.1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dmifem.elements.base import Array
from dmifem.mesh import Mesh


@dataclass(frozen=True)
class PriorOperator:
    """Unit-weight section-strain operator and its row structure.

    Attributes
    ----------
    rows:
        ``L`` of shape ``(n_rows, n)``, quadrature weights included as ``sqrt(w_q)``.
    kind:
        Kind of each row: ``0`` membrane, ``1`` bending, ``2`` shear.
    element:
        Element index of each row.
    """

    rows: Array
    kind: Array
    element: Array

    @property
    def precision(self) -> Array:
        """Unit-weight iFEM matrix ``K1 = L^T L``."""
        return self.rows.T @ self.rows


def assemble_prior(mesh: Mesh) -> PriorOperator:
    """Assemble ``L`` over all elements and quadrature points."""
    blocks, kinds, elems = [], [], []
    for e, elem in enumerate(mesh.elements):
        gather = mesh.gather(e)
        points, weights = elem.quadrature()
        for xi, w in zip(points, weights, strict=True):
            ops = elem.section_strain_operators(xi)
            for kind, op in enumerate((ops.membrane, ops.bending, ops.shear)):
                if op.shape[0] == 0:
                    continue
                blocks.append(np.sqrt(w) * op @ gather)
                kinds.extend([kind] * op.shape[0])
                elems.extend([e] * op.shape[0])
    rows = np.vstack(blocks)
    return PriorOperator(rows, np.array(kinds, dtype=int), np.array(elems, dtype=int))


def classical_weights(prior: PriorOperator, w_m: float, w_b: float, w_s: float) -> Array:
    """Per-row weights of the classical functional, Eq. (functional), for the rows of ``L``."""
    table = np.array([w_m, w_b, w_s], dtype=float)
    return table[prior.kind]


def static_ifem(rows: Array, weights: Array, measurements: Array) -> tuple[Array, Array]:
    """Static iFEM solution and covariance, Eq. (static): ``u = K^{-1} f``, ``P = K^{-1}``.

    Parameters
    ----------
    rows:
        Section-strain rows ``H_eps`` (quadrature weights included), shape ``(n_rows, n)``.
    weights:
        Inverse noise variances per row (the classical weights), shape ``(n_rows,)``.
    measurements:
        Measured section strains per row, shape ``(n_rows,)``.
    """
    precision = rows.T @ (weights[:, None] * rows)
    rhs = rows.T @ (weights * measurements)
    covariance = np.linalg.inv(precision)
    return covariance @ rhs, covariance
