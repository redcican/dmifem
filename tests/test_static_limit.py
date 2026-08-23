"""Numerical check of Proposition 3.1: the static limit is the classical iFEM solution."""

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
    k1 = assemble_prior(mesh).precision
    np.testing.assert_allclose(k1, k1.T)
    assert np.linalg.eigvalsh(k1).min() > 0.0
