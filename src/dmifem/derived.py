"""Derived fields from the posterior (manuscript Section 3.6, Eq. derivedpost).

Strain at any point is linear in the reduced displacement coordinates, so its posterior mean
and variance follow from the reduced moments; the covariance of a full field adds the prior
covariance projected on the orthogonal complement of the basis.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dmifem.elements.base import Array
from dmifem.mesh import Mesh


def strain_rows(mesh: Mesh, points: Sequence[tuple[int, float, float, float]]) -> Array:
    """Fibre-strain rows at ``(element, xi, y, z)`` points, shape ``(n_points, n)``."""
    rows = np.zeros((len(points), mesh.n_free))
    for i, (element, xi, y, z) in enumerate(points):
        rows[i] = mesh.elements[element].gauge_row(xi, y, z) @ mesh.gather(element)
    return rows


def complement_covariance(basis: Array, precision_prior: Array, lam: float) -> Array:
    """Prior covariance ``(lambda K1)^{-1}`` projected on the orthogonal complement of ``Phi``."""
    n = basis.shape[0]
    projector = np.eye(n) - basis @ basis.T
    prior_cov = np.linalg.pinv(lam * precision_prior)
    return projector @ prior_cov @ projector


def posterior_linear(
    rows: Array, basis: Array, mean_q: Array, cov_qq: Array, complement: Array | None = None
) -> tuple[Array, Array]:
    """Posterior mean and standard deviation of linear functionals ``rows @ u``.

    Parameters
    ----------
    rows:
        Linear functionals on the free displacement vector, ``(n_points, n)``.
    basis:
        Reduced basis ``Phi``.
    mean_q, cov_qq:
        Reduced displacement mean ``(N, r)`` and covariance ``(N, r, r)``.
    complement:
        Optional complement covariance added to every instant.
    """
    reduced = rows @ basis
    mean = mean_q @ reduced.T
    var = np.einsum("pi,kij,pj->kp", reduced, cov_qq, reduced)
    if complement is not None:
        var = var + np.einsum("pi,ij,pj->p", rows, complement, rows)[None, :]
    return mean, np.sqrt(np.clip(var, 0.0, None))


def stress_from_strain(strain: Array, modulus: float) -> Array:
    """Uniaxial stress ``sigma = E eps``, the only place where material data enter."""
    return modulus * strain
