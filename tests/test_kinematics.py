"""Exact discretization of the white-noise-jerk model and the scalar chain of Proposition reduced."""

import numpy as np
import pytest
from scipy.integrate import quad

from dmifem.chain import (
    chain_filter,
    chain_smoother,
    pseudo_strengths,
    sample_chain,
    stationary_chain_covariance,
)
from dmifem.kinematics import (
    ConstantAccelerationModel,
    covariance_root,
    process_noise_core,
    transition_core,
)


def test_transition_is_a_semigroup():
    a, b = 0.3, 0.7
    np.testing.assert_allclose(transition_core(a) @ transition_core(b), transition_core(a + b))


def test_process_noise_matches_numerical_integral():
    dt = 0.4
    g = np.array([0.0, 0.0, 1.0])

    def integrand(s, i, j):
        f = transition_core(s) @ g
        return f[i] * f[j]

    expected = np.array([[quad(integrand, 0.0, dt, args=(i, j))[0] for j in range(3)] for i in range(3)])
    np.testing.assert_allclose(process_noise_core(dt), expected, rtol=1e-10)


def test_kronecker_structure():
    model = ConstantAccelerationModel(2.0 * np.eye(3))
    f, q = model.transition(0.1), model.process_noise(0.1)
    assert f.shape == (9, 9) and q.shape == (9, 9)
    np.testing.assert_allclose(f[:3, 3:6], 0.1 * np.eye(3))
    np.testing.assert_allclose(q[6:, 6:], 0.2 * np.eye(3))
    np.testing.assert_allclose(q, q.T)
    assert np.all(np.linalg.eigvalsh(q) >= -1e-12)


def test_covariance_root_reproduces_semidefinite_matrices():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5, 3))
    cov = a @ a.T  # rank 3
    root = covariance_root(cov)
    np.testing.assert_allclose(root @ root.T, cov, atol=1e-12 * np.abs(cov).max())


def test_pseudo_strengths_use_the_preceding_interval():
    dts = np.array([0.1, 0.2, 0.3])
    np.testing.assert_allclose(pseudo_strengths(2.0, dts, 4), [0.2, 0.2, 0.4, 0.6])
    with pytest.raises(ValueError):
        pseudo_strengths(2.0, dts, 3)


def test_stationary_covariance_is_a_fixed_point_and_constant_on_a_uniform_grid():
    q, lam, dt = 2.0, 0.3, 0.05
    p1 = stationary_chain_covariance(q, lam, dt)
    assert np.linalg.eigvalsh(p1).min() > 0.0
    result = chain_filter(q, lam, np.full(40, dt), p1)
    np.testing.assert_allclose(result.cov_pred, np.broadcast_to(p1, result.cov_pred.shape), rtol=1e-8,
                               atol=1e-10 * np.abs(p1).max())
    with pytest.raises(ValueError):
        stationary_chain_covariance(q, 0.0, dt)


def test_chain_density_matches_the_batch_gaussian():
    q, lam = 1.5, 0.8
    dts = np.array([0.2, 0.1, 0.3])
    cov0 = np.diag([0.5, 1.0, 2.0])
    result = chain_filter(q, lam, dts, cov0)
    n = dts.size + 1
    # prior covariances of the chain states without any update, then Cov(pi_i[0], pi_j[0])
    prior = [cov0]
    for dt in dts:
        f = transition_core(dt)
        prior.append(f @ prior[-1] @ f.T + q * process_noise_core(dt))
    cov_z = np.zeros((n, n))
    for i in range(n):
        cross = prior[i]
        for j in range(i, n):
            cov_z[i, j] = cov_z[j, i] = cross[0, 0]
            if j < n - 1:
                cross = transition_core(dts[j]) @ cross
    cov_z += np.diag(1.0 / pseudo_strengths(lam, dts, n))
    _, log_det = np.linalg.slogdet(cov_z)
    expected = -0.5 * (log_det + n * np.log(2.0 * np.pi))
    assert result.log_density == pytest.approx(expected, rel=1e-12)


def test_chain_smoother_reduces_variance_and_samples_match_it():
    q, lam = 2.0, 0.3
    dts = np.full(80, 0.05)
    p1 = stationary_chain_covariance(q, lam, 0.05)
    result = chain_filter(q, lam, dts, p1)
    smoothed = chain_smoother(result, q, dts)
    filt_var = result.cov_filt[:, 0, 0]
    smooth_var = smoothed.cov[:, 0, 0]
    assert np.all(smooth_var <= filt_var * (1.0 + 1e-9))
    assert smooth_var[-1] == pytest.approx(filt_var[-1])
    paths = sample_chain(result, smoothed, 4000, np.random.default_rng(1))
    assert paths.shape == (4000, dts.size + 1, 3)
    np.testing.assert_allclose(paths[:, :, 0].var(axis=0), smooth_var, rtol=0.15)
    np.testing.assert_allclose(paths[:, :, 0].mean(axis=0), 0.0, atol=0.1 * np.sqrt(smooth_var.max()))


def test_chain_samples_carry_the_smoothed_cross_time_covariance():
    q, lam, dt = 2.0, 0.3, 0.05
    dts = np.full(80, dt)
    result = chain_filter(q, lam, dts, stationary_chain_covariance(q, lam, dt))
    smoothed = chain_smoother(result, q, dts)
    paths = sample_chain(result, smoothed, 4000, np.random.default_rng(2))
    k, lag = 30, 10
    # Cov(pi_k, pi_{k+L} | z) = G_k ... G_{k+L-1} Pi_{k+L|N}
    gain = np.eye(3)
    for j in range(k, k + lag):
        gain = gain @ smoothed.gains[j]
    expected = (gain @ smoothed.cov[k + lag])[0, 0]
    sample = np.mean(paths[:, k, 0] * paths[:, k + lag, 0])
    scale = np.sqrt(smoothed.cov[k, 0, 0] * smoothed.cov[k + lag, 0, 0])
    assert abs(expected) > 0.2 * scale  # the lag is short enough to leave a clear correlation
    assert abs(sample - expected) <= 0.1 * scale
