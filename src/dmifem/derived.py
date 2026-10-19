"""Derived fields from the posterior (manuscript Section 3.6, Eqs. fieldcov, vr and derivedpost).

Every derived quantity is linear in the field. For a functional ``c^T u`` of a state block the
posterior mean is ``c^T Phi q_hat`` and the variance is

    ``c^T Phi P Phi^T c + [Pi_k]_ii c^T Sigma_perp c``,  ``Sigma_perp = K1^{-1} - Phi Phi^T``,

the complement covariance of Proposition reduced scaled by the covariance ``Pi_k`` of the scalar
chain (entry ``ii`` = 1, 2, 3 for the displacement, velocity and acceleration blocks). The
measurable fraction of Corollary unobservable is ``mf = 1 - c^T Sigma_perp c / c^T K1^{-1} c``.

All functionals act on the dimensionless coordinates of Eq. (scaling). With a
:class:`dmifem.prior.SpatialPrior` the products are formed in the prior-whitened coordinates
``w = R u`` (``K1 = R^T R``): a functional becomes ``z = R^{-T} c``, the basis ``U = R Phi`` has
orthonormal columns, ``c^T Phi = z^T U`` and ``c^T Sigma_perp c = |z - U U^T z|^2``, free of the
cancellation that ``c^T K1^{-1} c - |Phi^T c|^2`` suffers when ``mf`` is close to one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np

from dmifem.basis import ReducedBasis
from dmifem.elements.base import Array
from dmifem.mesh import Mesh
from dmifem.prior import SpatialPrior, prior_solve

BasisLike = Union[ReducedBasis, Array]  # noqa: UP007 (runtime alias)


def strain_rows(mesh: Mesh, points: Sequence[tuple[int, float, float, float]]) -> Array:
    """Fibre-strain rows at ``(element, xi, y, z)`` points on the free dofs, ``(n_points, n)``."""
    rows = np.zeros((len(points), mesh.n_free))
    for i, (element, xi, y, z) in enumerate(points):
        rows[i] = mesh.elements[element].gauge_row(xi, y, z) @ mesh.gather(element)
    return rows


def _basis_arrays(basis: BasisLike, prior: SpatialPrior | Array | None) -> tuple[Array, Array | None]:
    """``Phi`` and, for a :class:`SpatialPrior`, ``U = R Phi``."""
    if isinstance(basis, ReducedBasis):
        phi, u = basis.vectors, basis.whitened
    else:
        phi, u = np.asarray(basis, dtype=float), None
    if u is None and isinstance(prior, SpatialPrior):
        u = prior.factor @ phi
    if not isinstance(prior, SpatialPrior):
        u = None
    return phi, u


def reduced_rows(rows: Array, basis: BasisLike, prior: SpatialPrior | Array | None = None) -> Array:
    """``rows @ Phi``, formed as ``(R^{-T} rows^T)^T U`` for a :class:`SpatialPrior`."""
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    phi, u = _basis_arrays(basis, prior)
    if u is not None:
        return prior.whiten(rows).T @ u  # type: ignore[union-attr]
    return rows @ phi


def _prior_and_complement(rows: Array, basis: BasisLike, prior: SpatialPrior | Array) -> tuple[Array, Array]:
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    phi, u = _basis_arrays(basis, prior)
    if u is not None:
        z = prior.whiten(rows)  # type: ignore[union-attr]
        residual = z - u @ (u.T @ z)
        return np.sum(z**2, axis=0), np.sum(residual**2, axis=0)
    total = np.einsum("pi,ip->p", rows, prior_solve(prior, rows.T))
    projected = np.sum((rows @ phi) ** 2, axis=1)
    return total, np.clip(total - projected, 0.0, None)


def complement_covariance(basis: BasisLike, prior: SpatialPrior | Array) -> Array:
    """``Sigma_perp = K1^{-1} - Phi Phi^T`` for a K1-orthonormal basis ``Phi``.

    ``prior`` is a :class:`dmifem.prior.SpatialPrior` or the covariance ``K1^{-1}`` as a matrix.
    """
    phi, u = _basis_arrays(basis, prior)
    if u is not None:
        projector = np.eye(u.shape[0]) - u @ u.T
        half = prior.color(projector)  # type: ignore[union-attr]  # R^{-1} (I - U U^T)
        out = prior.color(half.T)  # type: ignore[union-attr]  # R^{-1} (I - U U^T) R^{-T}
    else:
        out = np.asarray(prior, dtype=float) - phi @ phi.T
    return 0.5 * (out + out.T)


def complement_variance(rows: Array, basis: BasisLike, prior: SpatialPrior | Array) -> Array:
    """``c^T Sigma_perp c`` for every row ``c``, shape ``(n_points,)``."""
    return _prior_and_complement(rows, basis, prior)[1]


def complement_output_covariance(rows: Array, basis: BasisLike, prior: SpatialPrior | Array) -> Array:
    """``B Sigma_perp B^T`` for the rows ``B``, shape ``(n_points, n_points)``."""
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    phi, u = _basis_arrays(basis, prior)
    if u is not None:
        z = prior.whiten(rows)  # type: ignore[union-attr]
        residual = z - u @ (u.T @ z)
        return residual.T @ residual
    reduced = rows @ phi
    out = rows @ prior_solve(prior, rows.T) - reduced @ reduced.T
    return 0.5 * (out + out.T)


def _informed(basis: BasisLike) -> BasisLike:
    """The leading ``n_informed`` columns of an enriched basis (enrichment carries no sensor content)."""
    if not isinstance(basis, ReducedBasis) or basis.n_informed >= basis.r:
        return basis
    n = basis.n_informed
    whitened = None if basis.whitened is None else basis.whitened[:, :n]
    return ReducedBasis(basis.vectors[:, :n], basis.eigenvalues, n, whitened)


def measurable_fraction(rows: Array, basis: BasisLike, prior: SpatialPrior | Array) -> Array:
    """Measurable fraction ``mf(c)`` of Eq. (vr) for every row (``nan`` for zero rows).

    Only the informed columns of the basis enter: enrichment vectors leave ``mf`` unchanged.
    """
    total, comp = _prior_and_complement(rows, _informed(basis), prior)
    safe = np.where(total > 0.0, total, 1.0)
    return np.where(total > 0.0, 1.0 - comp / safe, np.nan)


def posterior_reduced(
    reduced: Array,
    mean_q: Array,
    cov_qq: Array,
    complement: Array | None = None,
    chain_variance: Array | None = None,
) -> tuple[Array, Array]:
    """Posterior mean and standard deviation from reduced rows ``rows @ Phi``, Eq. (derivedpost).

    Parameters
    ----------
    reduced:
        Reduced functionals ``(n_points, r)``.
    mean_q, cov_qq:
        Reduced mean ``(N, r)`` and covariance ``(N, r, r)`` of the state block.
    complement:
        Complement variances ``c^T Sigma_perp c`` per row, ``(n_points,)``, or ``None``.
    chain_variance:
        ``[Pi_k]_ii`` per instant, ``(N,)``; one when omitted.
    """
    mean = mean_q @ reduced.T
    var = np.einsum("pi,kij,pj->kp", reduced, cov_qq, reduced)
    if complement is not None:
        comp = np.asarray(complement, dtype=float).reshape(-1)
        scale = (
            np.ones(var.shape[0])
            if chain_variance is None
            else np.asarray(chain_variance, dtype=float).reshape(-1)
        )
        var = var + scale[:, None] * comp[None, :]
    return mean, np.sqrt(np.clip(var, 0.0, None))


def posterior_linear(
    rows: Array,
    basis: BasisLike,
    mean_q: Array,
    cov_qq: Array,
    complement: Array | None = None,
    chain_variance: Array | None = None,
    prior: SpatialPrior | Array | None = None,
) -> tuple[Array, Array]:
    """Posterior mean and standard deviation of linear functionals ``rows @ u``, Eq. (derivedpost).

    ``rows`` act on the dimensionless free dofs; see :func:`posterior_reduced` for the other
    arguments. With ``prior`` the reduced rows are formed in the whitened coordinates.
    """
    return posterior_reduced(reduced_rows(rows, basis, prior), mean_q, cov_qq, complement, chain_variance)


def stress_from_strain(strain: Array, modulus: float) -> Array:
    """Uniaxial stress ``sigma = E eps``, the only place where material data enter."""
    return modulus * strain
