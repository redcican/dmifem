"""Whitened influence operator and reduced basis."""

import numpy as np

from dmifem.basis import enrich_basis, influence_operator, observability_gramian, reduced_basis
from dmifem.observation import acceleration_operator, strain_operator
from dmifem.prior import assemble_prior


def _rows_and_sigma(mesh, gauges, accelerometers):
    ops = [strain_operator(mesh, gauges), acceleration_operator(mesh, accelerometers)]
    return np.vstack([o.rows for o in ops]), np.concatenate([o.sigma for o in ops])


def test_basis_is_orthonormal_and_invariant_to_channel_units(mesh, gauges, accelerometers):
    rows, sigma = _rows_and_sigma(mesh, gauges, accelerometers)
    k1 = assemble_prior(mesh).precision
    basis = reduced_basis(influence_operator(rows, sigma, k1, 1e-6), tau=1e-8)
    np.testing.assert_allclose(basis.vectors.T @ basis.vectors, np.eye(basis.r), atol=1e-10)
    # express the accelerometer channels in another unit: rows and noise scale together
    scale = np.ones_like(sigma)
    scale[len(gauges):] = 1000.0
    basis_scaled = reduced_basis(influence_operator(rows * scale[:, None], sigma * scale, k1, 1e-6), tau=1e-8)
    np.testing.assert_allclose(basis_scaled.singular_values, basis.singular_values, rtol=1e-9)
    assert basis_scaled.r == basis.r


def test_truncation_and_enrichment(mesh, gauges, accelerometers):
    rows, sigma = _rows_and_sigma(mesh, gauges, accelerometers)
    k1 = assemble_prior(mesh).precision
    influence = influence_operator(rows, sigma, k1, 1e-6)
    coarse = reduced_basis(influence, tau=1e-2)
    fine = reduced_basis(influence, tau=1e-12)
    assert coarse.r <= fine.r <= rows.shape[0]
    enriched = enrich_basis(coarse, k1, 2)
    assert enriched.r == coarse.r + 2
    np.testing.assert_allclose(enriched.vectors.T @ enriched.vectors, np.eye(enriched.r), atol=1e-10)


def test_gramian_is_symmetric_positive_semidefinite():
    rng = np.random.default_rng(0)
    dim = 6
    f = np.eye(dim) + 0.01 * rng.standard_normal((dim, dim))
    rows = [rng.standard_normal((2, dim)) for _ in range(4)]
    sigma = [np.ones(2) for _ in range(4)]
    gramian = observability_gramian([f] * 3, rows, sigma)
    np.testing.assert_allclose(gramian, gramian.T)
    assert np.linalg.eigvalsh(gramian).min() >= -1e-10
