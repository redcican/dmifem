"""Reduced kinematic basis and observability (manuscript Section 3.3).

The whitened influence operator ``U = K_lambda^{-1} H^T R^{-1/2}`` has one column per channel,
the displacement field produced by a reading of one noise standard deviation; its thin SVD
truncated at ``sigma_i >= tau sigma_1`` gives the orthonormal basis ``Phi``. The construction is
invariant to the physical units of the channels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from dmifem.elements.base import Array


@dataclass(frozen=True)
class ReducedBasis:
    """Orthonormal basis ``Phi`` (``n x r``) and the normalized singular values of ``U``."""

    vectors: Array
    singular_values: Array

    @property
    def r(self) -> int:
        return self.vectors.shape[1]


def influence_operator(rows: Array, sigma: Array, precision_prior: Array, lam: float) -> Array:
    """Whitened influence operator ``U`` of Eq. (influence).

    Parameters
    ----------
    rows:
        Stacked observation rows ``H`` of all channels on the displacement basis, ``(m, n)``.
    sigma:
        Noise standard deviation of every channel, ``(m,)``.
    precision_prior:
        Unit-weight prior precision ``K1``.
    lam:
        Prior scale ``lambda``.
    """
    whitened = rows / sigma[:, None]
    k_lambda = whitened.T @ whitened + lam * precision_prior
    # least squares (SVD) rather than a direct solve: K_lambda spans many decades when the prior
    # scale is small relative to the data precision
    return np.linalg.lstsq(k_lambda, whitened.T, rcond=None)[0]


def reduced_basis(influence: Array, tau: float) -> ReducedBasis:
    """Thin SVD of ``U`` truncated at ``tau`` times the largest singular value."""
    vectors, values, _ = np.linalg.svd(influence, full_matrices=False)
    normalized = values / values[0]
    r = int(np.sum(normalized >= tau))
    if r == 0:
        raise ValueError("truncation level leaves no basis vector")
    return ReducedBasis(vectors[:, :r], normalized)


def enrich_basis(basis: ReducedBasis, precision_prior: Array, n_extra: int) -> ReducedBasis:
    """Append the ``n_extra`` smoothest admissible fields (smallest eigenvectors of ``K1``)."""
    if n_extra <= 0:
        return basis
    _, vectors = eigh(precision_prior, subset_by_index=[0, n_extra - 1])
    residual = vectors - basis.vectors @ (basis.vectors.T @ vectors)
    q, _ = np.linalg.qr(residual)
    return ReducedBasis(np.hstack([basis.vectors, q]), basis.singular_values)


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
