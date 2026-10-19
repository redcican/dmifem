"""Joint smoothing samples of the strain: reduced trajectories plus complement trajectories.

The sampled histories must reproduce the pointwise posterior of Eq. (fieldcov) returned by
``strain_at``; without the complement they reproduce the reduced posterior alone.
"""

from __future__ import annotations

import numpy as np

from dmifem import DynamicMultiSensorIFEM, Scaling
from dmifem.derived import complement_output_covariance, posterior_reduced, reduced_rows, strain_rows

HELD_OUT = (4, 0.5, 0.15, 0.0)
SECOND = (7, 0.5, 0.15, 0.0)
N_SAMPLES = 400


def _fitted(mesh, gauges, accelerometers, data):
    window = data.window(0.0, 4.0)
    est = DynamicMultiSensorIFEM(
        mesh, gauges=gauges, accelerometers=accelerometers, gate_alpha=0.0, scaling=Scaling(0.15, 0.05)
    )
    est.prepare(window)
    est.fit(window, theta=est.initial_hyperparameters(window))
    return est


def test_samples_reproduce_the_field_posterior(mesh, gauges, accelerometers, data):
    est = _fitted(mesh, gauges, accelerometers, data)
    mean, std = est.strain_at([HELD_OUT])
    draws = est.sample_strain_trajectories([HELD_OUT], N_SAMPLES, np.random.default_rng(0))
    assert draws.shape == (N_SAMPLES, mean.shape[0], 1)
    # standard error of a sample std is about 1 / sqrt(2 n) = 3.5 %; rtol 0.2 is far outside it
    np.testing.assert_allclose(draws.std(axis=0), std, rtol=0.2)
    np.testing.assert_allclose(draws.mean(axis=0), mean, rtol=0.0, atol=0.3 * std.max())


def test_reduced_samples_reproduce_the_reduced_posterior(mesh, gauges, accelerometers, data):
    est = _fitted(mesh, gauges, accelerometers, data)
    rows_bar = est.dimensionless_rows(strain_rows(mesh, [HELD_OUT]))
    mean_q, cov_qq = est.reduced_block(0, smoothed=True)
    mean, std = posterior_reduced(reduced_rows(rows_bar, est.basis, est.spatial_prior), mean_q, cov_qq)
    draws = est.sample_strain_trajectories(
        [HELD_OUT], N_SAMPLES, np.random.default_rng(1), include_complement=False
    )
    np.testing.assert_allclose(draws.std(axis=0), std, rtol=0.2)
    np.testing.assert_allclose(draws.mean(axis=0), mean, rtol=0.0, atol=0.3 * std.max())
    # the complement adds variance at a held-out fibre (mf < 1)
    _, std_field = est.strain_at([HELD_OUT])
    assert np.all(std_field >= std * (1.0 - 1e-12))
    assert np.mean(std_field) > np.mean(std)


def test_samples_carry_cross_point_and_cross_time_covariance(mesh, gauges, accelerometers, data):
    """Rainflow counting needs the joint law: check covariances across points and across time."""
    est = _fitted(mesh, gauges, accelerometers, data)
    points = [HELD_OUT, SECOND]
    draws = est.sample_strain_trajectories(points, N_SAMPLES, np.random.default_rng(3))
    rows_bar = est.dimensionless_rows(strain_rows(mesh, points))
    reduced = reduced_rows(rows_bar, est.basis, est.spatial_prior)
    comp = complement_output_covariance(rows_bar, est.basis, est.spatial_prior)
    r = est.basis.r
    smoother, chain = est.smoother_result, est.chain_smoothed
    _, std = est.strain_at(points)
    centred = draws - draws.mean(axis=0)
    # across points at every instant: reduced cross term plus the complement scaled by the chain
    cross = np.einsum("i,kij,j->k", reduced[0], smoother.cov[:, :r, :r], reduced[1])
    cross = cross + chain.cov[:, 0, 0] * comp[0, 1]
    sample = np.mean(centred[:, :, 0] * centred[:, :, 1], axis=0)
    np.testing.assert_allclose(sample, cross, rtol=0.0, atol=0.3 * np.max(std[:, 0] * std[:, 1]))
    # across time at one point: Cov(x_k, x_{k+L}) = G_k ... G_{k+L-1} P_{k+L|N}
    k, lag = 50, 2
    gain = np.eye(smoother.gains.shape[1])
    chain_gain = np.eye(3)
    for j in range(k, k + lag):
        gain = gain @ smoother.gains[j]
        chain_gain = chain_gain @ chain.gains[j]
    state = (gain @ smoother.cov[k + lag])[:r, :r]
    expected = reduced[0] @ state @ reduced[0] + (chain_gain @ chain.cov[k + lag])[0, 0] * comp[0, 0]
    sample_lag = np.mean(centred[:, k, 0] * centred[:, k + lag, 0])
    scale = std[k, 0] * std[k + lag, 0]
    assert abs(expected) > 0.2 * scale
    assert abs(sample_lag - expected) <= 0.25 * scale
