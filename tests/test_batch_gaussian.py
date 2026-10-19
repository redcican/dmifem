"""Batch Gaussian check of the recursion, the reduction and the exact likelihood (Eq. exactll).

A model with five degrees of freedom, three instants, two displacement-type rows and one
acceleration row is small enough to condition the joint Gaussian of all states, pseudo-measurements
and channels directly.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from dmifem.basis import k_orthonormal_basis
from dmifem.chain import chain_filter, chain_smoother, stationary_chain_covariance
from dmifem.derived import complement_variance
from dmifem.filter import ReducedObservation, prior_square_root, run_filter
from dmifem.grid import build_union_grid
from dmifem.kinematics import ConstantAccelerationModel, process_noise_core, transition_core
from dmifem.sensors import SensorData, SensorType
from dmifem.smoother import rts_smoother

LOG_2PI = np.log(2.0 * np.pi)
N_DOF, N_STEPS, DT, Q, LAM = 5, 3, 0.2, 2.0, 0.7
DISP, ACC = SensorType.DISPLACEMENT, SensorType.ACCELERATION
D = 3 * N_DOF
NX, NZ, NY = N_STEPS * D, N_STEPS * N_DOF, 3 * N_STEPS
TOL = 1e-9


def _problem():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((N_DOF, N_DOF))
    cov = a @ a.T / N_DOF + 0.3 * np.eye(N_DOF)  # K1^{-1}
    rows_d = rng.standard_normal((2, N_DOF))
    rows_a = rng.standard_normal((1, N_DOF))
    sigma = np.sqrt(np.array([0.3, 0.2, 0.5]))
    y = rng.standard_normal((N_STEPS, 3))
    times = DT * np.arange(N_STEPS)
    data = SensorData({DISP: (times, y[:, :2]), ACC: (times, y[:, 2:])})
    grid = build_union_grid({DISP: times, ACC: times})
    return cov, rows_d, rows_a, sigma, y, data, grid


def _blocks(rows_d, rows_a, sigma, basis):
    r = basis.shape[1]
    h_d = np.zeros((2, 3 * r))
    h_d[:, :r] = rows_d @ basis
    h_a = np.zeros((1, 3 * r))
    h_a[:, 2 * r :] = rows_a @ basis
    return {
        DISP: ReducedObservation(h_d, np.zeros(2), sigma[:2] ** 2),
        ACC: ReducedObservation(h_a, np.zeros(1), sigma[2:] ** 2),
    }


def _condition(mu, cov, idx, value):
    """Condition N(mu, cov) on x[idx] = value: mean and covariance of the rest, log density."""
    rest = np.setdiff1d(np.arange(mu.size), idx)
    factor = cho_factor(cov[np.ix_(idx, idx)])
    d = value - mu[idx]
    gain = cho_solve(factor, cov[np.ix_(idx, rest)]).T
    mean = mu[rest] + gain @ d
    cond = cov[np.ix_(rest, rest)] - gain @ cov[np.ix_(idx, rest)]
    logp = -0.5 * (2.0 * np.sum(np.log(np.diag(factor[0]))) + d @ cho_solve(factor, d) + d.size * LOG_2PI)
    return mean, cond, logp


def _joint(cov, rows_d, rows_a, sigma):
    """Joint covariance of [x_1..x_N, pseudo-measurements, channels] and the initial factor."""
    n = N_DOF
    f = np.kron(transition_core(DT), np.eye(n))
    q_mat = Q * np.kron(process_noise_core(DT), cov)
    p1 = stationary_chain_covariance(Q, LAM, DT)
    marginals = [np.kron(p1, cov)]
    for _ in range(1, N_STEPS):
        marginals.append(f @ marginals[-1] @ f.T + q_mat)
    s_x = np.zeros((NX, NX))
    for i in range(N_STEPS):
        for j in range(N_STEPS):
            if j >= i:
                blk = marginals[i] @ np.linalg.matrix_power(f, j - i).T
            else:
                blk = np.linalg.matrix_power(f, i - j) @ marginals[j]
            s_x[i * D : (i + 1) * D, j * D : (j + 1) * D] = blk
    h_z = np.zeros((n, D))
    h_z[:, :n] = prior_square_root(np.linalg.inv(cov))
    h_y = np.zeros((3, D))
    h_y[:2, :n] = rows_d
    h_y[2:, 2 * n :] = rows_a
    eye = np.eye(N_STEPS)
    hz, hy = np.kron(eye, h_z), np.kron(eye, h_y)
    rz = np.kron(eye, np.eye(n) / (LAM * DT))
    ry = np.kron(eye, np.diag(sigma**2))
    joint = np.block(
        [
            [s_x, s_x @ hz.T, s_x @ hy.T],
            [hz @ s_x, hz @ s_x @ hz.T + rz, hz @ s_x @ hy.T],
            [hy @ s_x, hy @ s_x @ hz.T, hy @ s_x @ hy.T + ry],
        ]
    )
    return joint, p1


def _batch_up_to(joint, y, k):
    """Batch posterior of the states given pseudo-measurements and channels of instants 0..k."""
    iz = np.arange(NX, NX + N_DOF * (k + 1))
    iy = np.arange(NX + NZ, NX + NZ + 3 * (k + 1))
    values = np.concatenate([np.zeros(iz.size), y[: k + 1].reshape(-1)])
    mean, cond, _ = _condition(np.zeros(joint.shape[0]), joint, np.concatenate([iz, iy]), values)
    return mean[:NX], cond[:NX, :NX]  # the states are the leading unobserved entries


def _batch_likelihood(joint, y):
    iz = np.arange(NX, NX + NZ)
    iy = np.arange(NX + NZ, NX + NZ + NY)
    mu = np.zeros(joint.shape[0])
    _, _, log_zy = _condition(mu, joint, np.concatenate([iz, iy]), np.concatenate([np.zeros(NZ), y.reshape(-1)]))
    _, _, log_z = _condition(mu, joint, iz, np.zeros(NZ))
    return log_zy - log_z


def test_full_recursion_matches_the_batch_posterior_and_likelihood():
    cov, rows_d, rows_a, sigma, y, data, grid = _problem()
    joint, p1 = _joint(cov, rows_d, rows_a, sigma)
    model = ConstantAccelerationModel(Q * cov)
    full = run_filter(
        model, grid, _blocks(rows_d, rows_a, sigma, np.eye(N_DOF)), data, np.linalg.inv(cov), LAM,
        np.zeros(D), np.kron(p1, cov), gate_alpha=0.0,
    )
    exact = _batch_likelihood(joint, y)
    np.testing.assert_allclose(full.log_likelihood, exact, rtol=0.0, atol=TOL)
    # the physical-channel sum is a different quantity
    assert abs(full.log_likelihood_physical - exact) > 1e-6
    for k in range(N_STEPS):
        mean_k, cov_k = _batch_up_to(joint, y, k)
        sel = slice(k * D, (k + 1) * D)
        np.testing.assert_allclose(full.mean_filt[k], mean_k[sel], atol=TOL)
        np.testing.assert_allclose(full.cov_filt[k], cov_k[sel, sel], atol=TOL)
    smoothed = rts_smoother(full, model)
    mean_all, cov_all = _batch_up_to(joint, y, N_STEPS - 1)
    for k in range(N_STEPS):
        sel = slice(k * D, (k + 1) * D)
        np.testing.assert_allclose(smoothed.mean[k], mean_all[sel], atol=TOL)
        np.testing.assert_allclose(smoothed.cov[k], cov_all[sel, sel], atol=TOL)


def test_reduced_model_with_complement_matches_the_batch():
    cov, rows_d, rows_a, sigma, y, data, grid = _problem()
    joint, p1 = _joint(cov, rows_d, rows_a, sigma)
    basis = k_orthonormal_basis(np.vstack([rows_d, rows_a]), sigma, cov, tau=1e-10)
    r, phi = basis.r, basis.vectors
    assert r == 3
    np.testing.assert_allclose(phi.T @ np.linalg.solve(cov, phi), np.eye(r), atol=1e-10)
    dts = grid.intervals()
    chain = chain_filter(Q, LAM, dts, p1, grid.n)
    model = ConstantAccelerationModel(Q * np.eye(r))
    args = (model, grid, _blocks(rows_d, rows_a, sigma, phi), data, None, LAM, np.zeros(3 * r), np.kron(p1, np.eye(r)))
    reduced = run_filter(*args, gate_alpha=0.0, prior_log_density=r * chain.log_density)
    generic = run_filter(*args, gate_alpha=0.0)
    exact = _batch_likelihood(joint, y)
    np.testing.assert_allclose(reduced.log_likelihood, exact, rtol=0.0, atol=TOL)
    np.testing.assert_allclose(generic.log_prior_density, r * chain.log_density, rtol=1e-12)
    smoothed = rts_smoother(reduced, model)
    chain_s = chain_smoother(chain, Q, dts)
    c = np.random.default_rng(4).standard_normal(N_DOF)
    comp = complement_variance(c[None, :], phi, cov)[0]
    a = c @ phi
    mean_all, cov_all = _batch_up_to(joint, y, N_STEPS - 1)
    for k in range(N_STEPS):
        mean_k, cov_k = _batch_up_to(joint, y, k)
        u = slice(k * D, k * D + N_DOF)  # displacement block of x_k
        for (m_red, p_red, pi), (m_full, p_full) in (
            ((reduced.mean_filt[k], reduced.cov_filt[k], chain.cov_filt[k]), (mean_k[u], cov_k[u, u])),
            ((smoothed.mean[k], smoothed.cov[k], chain_s.cov[k]), (mean_all[u], cov_all[u, u])),
        ):
            np.testing.assert_allclose(a @ m_red[:r], c @ m_full, rtol=1e-9, atol=1e-12)
            var_red = a @ p_red[:r, :r] @ a + pi[0, 0] * comp
            np.testing.assert_allclose(var_red, c @ p_full @ c, rtol=1e-9)
