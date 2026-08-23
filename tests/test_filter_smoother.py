"""Kalman filter, prior information update, gate and smoother on a scalar chain."""

import numpy as np

from dmifem.filter import ReducedObservation, prior_information_update, prior_square_root, run_filter
from dmifem.grid import build_union_grid
from dmifem.kinematics import ConstantAccelerationModel
from dmifem.metrics import coverage
from dmifem.sensors import SensorData, SensorType
from dmifem.smoother import rts_smoother, sample_trajectories


def _simulate(rng, n=400, dt=0.02, q=5.0, sigma=0.05):
    model = ConstantAccelerationModel(np.array([[q]]))
    f, qmat = model.transition(dt), model.process_noise(dt)
    chol = np.linalg.cholesky(qmat + 1e-14 * np.eye(3))
    states = np.zeros((n, 3))
    x = np.array([0.0, 0.0, 0.0])
    for k in range(n):
        states[k] = x
        x = f @ x + chol @ rng.standard_normal(3)
    times = dt * np.arange(n)
    y = states[:, :1] + sigma * rng.standard_normal((n, 1))
    return model, times, states, y


def _blocks(sigma):
    rows = np.array([[1.0, 0.0, 0.0]])
    return {SensorType.DISPLACEMENT: ReducedObservation(rows, np.zeros(1), np.array([sigma**2]))}


def test_filter_and_smoother_track_the_chain():
    rng = np.random.default_rng(7)
    model, times, states, y = _simulate(rng)
    data = SensorData({SensorType.DISPLACEMENT: (times, y)})
    grid = build_union_grid({SensorType.DISPLACEMENT: times})
    cov0 = np.diag([1.0, 10.0, 100.0])
    result = run_filter(model, grid, _blocks(0.05), data, np.eye(1), lam=0.0, mean0=np.zeros(3), cov0=cov0, gate_alpha=0.0)
    smoothed = rts_smoother(result, model)
    err_f = np.sqrt(np.mean((result.mean_filt[50:, 0] - states[50:, 0]) ** 2))
    err_s = np.sqrt(np.mean((smoothed.mean[50:, 0] - states[50:, 0]) ** 2))
    assert err_s < err_f < 0.05
    cov = coverage(states[50:, 0], smoothed.mean[50:, 0], np.sqrt(smoothed.cov[50:, 0, 0]))
    assert 0.85 <= cov <= 1.0
    assert np.isfinite(result.log_likelihood)


def test_gate_rejects_an_outlier_and_nan_is_skipped():
    rng = np.random.default_rng(8)
    model, times, states, y = _simulate(rng)
    y[200, 0] += 50.0  # gross outlier
    y[210, 0] = np.nan  # missing sample
    data = SensorData({SensorType.DISPLACEMENT: (times, y)})
    grid = build_union_grid({SensorType.DISPLACEMENT: times})
    cov0 = np.diag([1.0, 10.0, 100.0])
    result = run_filter(model, grid, _blocks(0.05), data, np.eye(1), lam=0.0, mean0=np.zeros(3), cov0=cov0, gate_alpha=1e-3)
    assert result.n_rejected >= 1
    assert abs(result.mean_filt[200, 0] - states[200, 0]) < 1.0
    assert np.all(np.isfinite(result.mean_filt))


def test_prior_update_shrinks_toward_zero_and_is_rate_invariant():
    mean = np.array([1.0, 0.0, 0.0])
    cov = np.eye(3)
    core = prior_square_root(np.array([[4.0]]))
    m1, c1 = prior_information_update(mean, cov, core, strength=2.0, r=1)
    assert 0.0 < m1[0] < mean[0]
    assert c1[0, 0] < cov[0, 0]
    np.testing.assert_allclose(c1[1:, 1:], np.eye(2))
    # two half-strength updates equal one full-strength update (same total precision)
    m_a, c_a = prior_information_update(mean, cov, core, strength=1.0, r=1)
    m_b, c_b = prior_information_update(m_a, c_a, core, strength=1.0, r=1)
    np.testing.assert_allclose(m_b, m1)
    np.testing.assert_allclose(c_b, c1)


def test_backward_sampling_matches_smoother_moments():
    rng = np.random.default_rng(9)
    model, times, states, y = _simulate(rng, n=200)
    data = SensorData({SensorType.DISPLACEMENT: (times, y)})
    grid = build_union_grid({SensorType.DISPLACEMENT: times})
    cov0 = np.diag([1.0, 10.0, 100.0])
    result = run_filter(model, grid, _blocks(0.05), data, np.eye(1), lam=0.0, mean0=np.zeros(3), cov0=cov0, gate_alpha=0.0)
    smoothed = rts_smoother(result, model)
    draws = sample_trajectories(result, smoothed, model, 300, np.random.default_rng(10))
    sample_mean = draws[:, :, 0].mean(axis=0)
    sample_std = draws[:, :, 0].std(axis=0)
    np.testing.assert_allclose(sample_mean[20:], smoothed.mean[20:, 0], atol=0.02)
    np.testing.assert_allclose(sample_std[20:], np.sqrt(smoothed.cov[20:, 0, 0]), rtol=0.3)
