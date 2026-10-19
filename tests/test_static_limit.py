"""Numerical check of Proposition 3.1: the static limit is the classical iFEM solution.

The prior rows are those of Eq. (prior) at unit reference length: one row per component and element
for the axial strain and the twist, two per bending plane, and no transverse-shear row on
Euler-Bernoulli elements. With the root clamped the rows have full column rank.
"""

import numpy as np

from dmifem.prior import assemble_prior, classical_weights, static_ifem


def test_static_limit_recovers_field_and_covariance(mesh):
    prior = assemble_prior(mesh)
    rng = np.random.default_rng(3)
    u_true = 1e-3 * rng.standard_normal(mesh.n_free)
    # section strains of every element at the quadrature points (noise-free)
    y = prior.rows @ u_true
    weights = classical_weights(prior, w_m=1.0, w_b=1.0, w_s=1e-4)
    u_hat, cov = static_ifem(prior.rows, weights, y)
    np.testing.assert_allclose(u_hat, u_true, rtol=1e-8, atol=1e-12)
    k_matrix = prior.rows.T @ (weights[:, None] * prior.rows)
    np.testing.assert_allclose(cov @ k_matrix, np.eye(mesh.n_free), atol=1e-8)


def test_unit_weight_precision_is_symmetric_positive(mesh):
    prior = assemble_prior(mesh)
    k1 = prior.precision
    np.testing.assert_allclose(k1, k1.T)
    # K1 = rows^T rows is positive definite when the rows have full column rank
    singular = np.linalg.svd(prior.rows, compute_uv=False)
    assert singular.size == mesh.n_free
    assert singular.min() > 1e-10 * singular.max()
