"""K1-orthonormal basis of Eq. (basis), measurable fraction of Eq. (vr) and the Gramian."""

from __future__ import annotations

import numpy as np

from dmifem import DynamicMultiSensorIFEM, Scaling
from dmifem.basis import enrich_basis, k_orthonormal_basis, observability_gramian
from dmifem.derived import (
    complement_covariance,
    complement_variance,
    measurable_fraction,
    reduced_rows,
    strain_rows,
)

HELD_OUT = (4, 0.5, 0.15, 0.0)


def _estimator(mesh, gauges, accelerometers, **kwargs):
    est = DynamicMultiSensorIFEM(
        mesh, gauges=gauges, accelerometers=accelerometers, scaling=Scaling(0.15, 0.05), **kwargs
    )
    est.form_basis()
    return est


def test_basis_is_k1_orthonormal_and_spans_the_sensor_rows(mesh, gauges, accelerometers):
    # checked in the prior-whitened coordinates w = R u: Phi^T K1 Phi = U^T U with U = R Phi, and
    # K1^{-1} H^T in span(Phi) is R^{-T} H^T in span(U); forming K1 itself loses about 13 digits
    est = _estimator(mesh, gauges, accelerometers)
    prior, basis = est.spatial_prior, est.basis
    u = basis.whitened
    assert u is not None
    assert basis.r == basis.n_informed == len(gauges) + len(accelerometers)
    np.testing.assert_allclose(u.T @ u, np.eye(basis.r), atol=1e-10)
    np.testing.assert_allclose(prior.factor @ basis.vectors, u, atol=1e-8 * np.abs(u).max())
    target = prior.whiten(est.sensor_rows())  # R^{-T} H^T
    residual = target - u @ (u.T @ target)
    assert np.abs(residual).max() <= 1e-8 * np.abs(target).max()


def test_span_is_invariant_to_noise_levels_and_units(mesh, gauges, accelerometers):
    est = _estimator(mesh, gauges, accelerometers)
    rows = est.sensor_rows()
    ones = np.ones(rows.shape[0])
    base = k_orthonormal_basis(rows, ones, est.spatial_prior)
    # another unit for one channel type: rows and noise scale together
    unit = ones.copy()
    unit[len(gauges) :] = 1e3
    rescaled = k_orthonormal_basis(rows * unit[:, None], unit, est.spatial_prior)
    np.testing.assert_allclose(rescaled.normalized_eigenvalues, base.normalized_eigenvalues, rtol=1e-9, atol=1e-15)
    # another noise level orders the eigenvalues differently and leaves the span unchanged
    noisy = ones.copy()
    noisy[len(gauges) :] = 10.0
    other = k_orthonormal_basis(rows, noisy, est.spatial_prior)
    assert other.r == base.r
    proj_base = base.whitened @ base.whitened.T
    proj_other = other.whitened @ other.whitened.T
    np.testing.assert_allclose(proj_other, proj_base, rtol=0.0, atol=1e-8)


def test_measurable_fraction_and_complement(mesh, gauges, accelerometers):
    est = _estimator(mesh, gauges, accelerometers)
    basis, prior = est.basis, est.spatial_prior
    sensor = est.sensor_rows()
    mf_sensor = measurable_fraction(sensor, basis, prior)
    np.testing.assert_allclose(mf_sensor, 1.0, atol=1e-8)
    held = est.dimensionless_rows(strain_rows(mesh, [HELD_OUT]))
    mf_held = est.measurable_fraction([HELD_OUT])
    np.testing.assert_allclose(mf_held, measurable_fraction(held, basis, prior))
    assert 0.0 < mf_held[0] < 1.0
    # the complement variance from Sigma_perp agrees with the whitened residual
    sigma_perp = complement_covariance(basis, prior)
    np.testing.assert_allclose(
        complement_variance(held, basis, prior),
        np.einsum("pi,ij,pj->p", held, sigma_perp, held),
        rtol=1e-6,
    )
    assert np.linalg.eigvalsh(sigma_perp).min() >= -1e-10 * np.abs(sigma_perp).max()
    # Sigma_perp annihilates the sensor rows: no sensor sees the complement
    leak = sensor @ sigma_perp
    assert np.abs(leak).max() <= 1e-6 * np.abs(sensor).max() * np.abs(sigma_perp).max()
    uncorrelated = _estimator(mesh, gauges, accelerometers, correlated_prior=False)
    assert uncorrelated.measurable_fraction([HELD_OUT])[0] < mf_held[0]


def test_enrichment_is_k1_orthonormal_and_leaves_the_field_covariance(mesh, gauges, accelerometers):
    est = _estimator(mesh, gauges, accelerometers)
    prior = est.spatial_prior
    enriched = enrich_basis(est.basis, prior, 2)
    assert enriched.r == est.basis.r + 2
    assert enriched.n_informed == est.basis.n_informed
    u = enriched.whitened
    assert u is not None
    np.testing.assert_allclose(u.T @ u, np.eye(enriched.r), atol=1e-10)
    np.testing.assert_allclose(u[:, : est.basis.r], est.basis.whitened, atol=0.0)
    # the leading vectors are kept and the added ones are smooth fields with no sensor content
    held = est.dimensionless_rows(strain_rows(mesh, [HELD_OUT]))
    total = complement_variance(held, est.basis, prior)
    enriched_total = complement_variance(held, enriched, prior) + np.sum(
        reduced_rows(held, enriched, prior)[:, est.basis.r :] ** 2, axis=1
    )
    np.testing.assert_allclose(enriched_total, total, rtol=1e-8)
    np.testing.assert_allclose(reduced_rows(est.sensor_rows(), enriched, prior)[:, est.basis.r :], 0.0,
                               atol=1e-8 * np.abs(est.sensor_rows()).max())
    # enrichment adds no sensor information: the measurable fraction is unchanged
    np.testing.assert_allclose(
        measurable_fraction(held, enriched, prior), measurable_fraction(held, est.basis, prior), rtol=1e-12
    )


def test_enrichment_stops_at_the_dimension_of_the_field(mesh, gauges, accelerometers):
    est = _estimator(mesh, gauges, accelerometers)
    n = mesh.n_free
    full = enrich_basis(est.basis, est.spatial_prior, 10 * n)
    assert full.r == n
    np.testing.assert_allclose(full.whitened.T @ full.whitened, np.eye(n), atol=1e-9)
    # a complete basis has no room left: further enrichment returns it unchanged
    assert enrich_basis(full, est.spatial_prior, 3) is full


def test_gramian_is_symmetric_positive_semidefinite():
    rng = np.random.default_rng(0)
    dim = 6
    f = np.eye(dim) + 0.01 * rng.standard_normal((dim, dim))
    rows = [rng.standard_normal((2, dim)) for _ in range(4)]
    sigma = [np.ones(2) for _ in range(4)]
    gramian = observability_gramian([f] * 3, rows, sigma)
    np.testing.assert_allclose(gramian, gramian.T)
    assert np.linalg.eigvalsh(gramian).min() >= -1e-10
