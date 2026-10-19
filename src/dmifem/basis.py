"""Reduced kinematic basis and observability (manuscript Section 3.3, Eqs. influence, basis, gramian).

The range of the influence operator ``U = K_lambda^{-1} H^T R^{-1/2}`` is ``K1^{-1} W``, with ``W`` the
span of the sensor rows, for every prior scale, noise level and choice of units. The basis is the
``K1``-orthonormal basis of this range (Eq. basis): with the eigendecomposition
``R^{-1/2} H K1^{-1} H^T R^{-1/2} = V Lambda V^T``,

    ``Phi = K1^{-1} H^T R^{-1/2} V_r Lambda_r^{-1/2}``,  ``Phi^T K1 Phi = I_r``,

where the ``r`` retained eigenvalues lie above ``tau Lambda_1``. With ``tau`` between the eigenvalues
of the independent sensor rows and the round-off level, ``r = dim W``. The noise levels and the
time scale order the eigenvalues and leave the span unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from dmifem.elements.base import Array
from dmifem.prior import SpatialPrior, prior_solve

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReducedBasis:
    """Basis ``Phi`` (``n x r``) and the eigenvalues that selected it.

    Attributes
    ----------
    vectors:
        ``Phi``; ``K1``-orthonormal for :func:`k_orthonormal_basis`.
    eigenvalues:
        Eigenvalues of the whitened Gram matrix of the sensor rows, in descending order (all
        channels, retained or not).
    n_informed:
        Number of leading vectors that span ``K1^{-1} W``; the remaining ones are enrichment.
    whitened:
        ``R Phi`` with orthonormal columns, the basis in the prior-whitened coordinates
        ``w = R u`` (``K1 = R^T R``), when the basis was formed from a
        :class:`dmifem.prior.SpatialPrior`; ``None`` otherwise.
    """

    vectors: Array
    eigenvalues: Array
    n_informed: int
    whitened: Array | None = None

    @property
    def r(self) -> int:
        return self.vectors.shape[1]

    @property
    def normalized_eigenvalues(self) -> Array:
        """Eigenvalues divided by the largest one (negative round-off set to zero)."""
        return np.clip(self.eigenvalues / self.eigenvalues[0], 0.0, None)

    @property
    def singular_values(self) -> Array:
        """Normalized singular values of the whitened sensor rows in the ``K1^{-1}`` metric."""
        return np.sqrt(self.normalized_eigenvalues)


def influence_operator(rows: Array, sigma: Array, precision_prior: Array, lam: float) -> Array:
    """Whitened influence operator ``U`` of Eq. (influence).

    Parameters
    ----------
    rows:
        Stacked observation rows ``H`` of all channels on the displacement basis, ``(m, n)``.
    sigma:
        Noise standard deviation of every channel, ``(m,)``.
    precision_prior:
        Prior precision ``K1``.
    lam:
        Prior scale ``lambda``.
    """
    whitened = rows / sigma[:, None]
    k_lambda = whitened.T @ whitened + lam * precision_prior
    # least squares (SVD) rather than a direct solve: K_lambda spans many decades when the prior
    # scale is small relative to the data precision
    return np.linalg.lstsq(k_lambda, whitened.T, rcond=None)[0]


def k_orthonormal_basis(
    rows: Array, sigma: Array, prior: SpatialPrior | Array, tau: float = 1e-10
) -> ReducedBasis:
    """``K1``-orthonormal basis of ``K1^{-1} W``, Eq. (basis).

    Parameters
    ----------
    rows:
        Stacked rows of all channels on the field, ``(m, n)`` (tilt rows included when active).
    sigma:
        Noise standard deviation of every row, ``(m,)``; ones for rows that are already whitened.
    prior:
        :class:`dmifem.prior.SpatialPrior`, or the covariance ``K1^{-1}`` as a matrix.
    tau:
        Eigenvalue truncation level relative to the largest eigenvalue.
    """
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    scaled = rows / sigma[:, None]
    if isinstance(prior, SpatialPrior):
        y = prior.whiten(scaled)  # R^{-T} H^T R^{-1/2}: the rows in w = R u
        gram = y.T @ y
    else:
        k_rows = prior_solve(prior, scaled.T)
        gram = scaled @ k_rows
    vals, vecs = np.linalg.eigh(0.5 * (gram + gram.T))
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    if not vals[0] > 0.0:
        raise ValueError("the sensor rows do not observe the field")
    keep = vals >= tau * vals[0]
    roundoff = vals.size * np.finfo(float).eps * vals[0]
    n_dropped = int(np.sum(~keep & (vals > roundoff)))
    if n_dropped:
        # these directions carry sensor information, so the reduction of Proposition reduced is no
        # longer exact for them; lower tau below their eigenvalues to keep r = dim W
        logger.warning("tau = %g drops %d directions above round-off; the reduction is not exact", tau, n_dropped)
    coef = vecs[:, keep] / np.sqrt(vals[keep])
    if isinstance(prior, SpatialPrior):
        u = y @ coef  # orthonormal columns
        return ReducedBasis(prior.color(u), vals, int(np.sum(keep)), u)
    return ReducedBasis(k_rows @ coef, vals, int(np.sum(keep)))


def _greedy_complement(u: Array, candidates: Array, n_extra: int, tol: float = 1e-5) -> Array:
    """Orthonormalize ``candidates`` in order against ``u`` and each other, keeping ``n_extra``.

    A candidate is kept when its residual norm exceeds ``tol`` (candidates have unit norm), so
    round-off of directions already in the span is never normalized into a basis vector.
    """
    q = u
    kept: list[Array] = []
    for j in range(candidates.shape[1]):
        v = candidates[:, j]
        for _ in range(2):  # repeated Gram-Schmidt
            v = v - q @ (q.T @ v)
        norm = float(np.linalg.norm(v))
        if norm > tol:
            v = v / norm
            kept.append(v)
            q = np.column_stack([q, v])
            if len(kept) == n_extra:
                break
    return np.column_stack(kept) if kept else np.zeros((u.shape[0], 0))


def enrich_basis(basis: ReducedBasis, prior: SpatialPrior | Array, n_extra: int) -> ReducedBasis:
    """Append the ``n_extra`` smoothest admissible fields, ``K1``-orthonormalized against ``Phi``.

    The candidates are the eigenvectors of ``K1`` in increasing order of eigenvalue; ``n_extra`` is
    capped by the dimension left in the field. The added
    directions receive no information from any sensor, so the field covariance of Eq. (fieldcov)
    is unchanged once the complement is formed with the enlarged basis. ``prior`` is a
    :class:`dmifem.prior.SpatialPrior` (computed in ``w = R u``) or the precision ``K1``.
    """
    n_extra = min(int(n_extra), basis.vectors.shape[0] - basis.r)
    if n_extra <= 0:
        return basis
    if isinstance(prior, SpatialPrior):
        # K1 = R^T R = B S^2 B^T with R = A S B^T: the K1-normalized eigenvectors of K1 with the
        # smallest eigenvalues are the left singular vectors of R with the smallest singular values;
        # all of them are candidates, smoothest first, so the requested number is reached whenever
        # the field has room for it
        left, _, _ = np.linalg.svd(prior.factor)
        candidates = left[:, ::-1]
        u = basis.whitened if basis.whitened is not None else prior.factor @ basis.vectors
        extra_w = _greedy_complement(u, candidates, n_extra)
        if extra_w.shape[1] == 0:
            return basis
        return ReducedBasis(
            np.hstack([basis.vectors, prior.color(extra_w)]),
            basis.eigenvalues,
            basis.n_informed,
            np.hstack([u, extra_w]),
        )
    k1 = np.asarray(prior, dtype=float)
    _, candidates = eigh(k1, subset_by_index=[0, n_extra - 1])
    phi = basis.vectors
    residual = candidates - phi @ (phi.T @ (k1 @ candidates))
    gram = residual.T @ k1 @ residual
    vals, vecs = np.linalg.eigh(0.5 * (gram + gram.T))
    reference = float(np.max(np.diag(candidates.T @ k1 @ candidates)))
    keep = vals > 1e-10 * reference
    if not np.any(keep):
        return basis
    extra = residual @ (vecs[:, keep] / np.sqrt(vals[keep]))
    return ReducedBasis(np.hstack([phi, extra]), basis.eigenvalues, basis.n_informed)


def observability_gramian(transitions: list[Array], rows: list[Array], sigma: list[Array]) -> Array:
    """Observability Gramian of Eq. (gramian) over a window of samples.

    Parameters
    ----------
    transitions:
        Transition matrices ``F_k, ..., F_{k+w-2}`` between consecutive samples of the window.
    rows:
        Reduced observation matrices ``H_{k+i}`` (``m_i x 3r``) of the window samples.
    sigma:
        Noise standard deviations of the rows at each sample.
    """
    dim = rows[0].shape[1]
    gramian = np.zeros((dim, dim))
    psi = np.eye(dim)
    for i, (h, s) in enumerate(zip(rows, sigma, strict=True)):
        hw = h / s[:, None]
        gramian += psi.T @ hw.T @ hw @ psi
        if i < len(transitions):
            psi = transitions[i] @ psi
    return gramian
