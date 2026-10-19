"""Full nodal model against the reduced model with the complement (Proposition reduced).

The structured model on all ``3n`` states of the cantilever must return the strain moments of the
reduced model plus the complement, and the same exact likelihood. The full model is written in the
prior-whitened coordinates ``w = R u_bar`` (``K1 = R^T R``), where the process covariance is
``q I_n``, the pseudo-measurement has precision ``lambda dt I_n`` and the initial covariance is
``P_bar_1 (x) I_n``. In ``u_bar`` the same model carries ``K1`` and ``K1^{-1}``, whose condition
number on the cantilever is of order ``1e13``; in ``w`` only ``R`` enters, through ``u_bar = R^{-1} w``.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

from dmifem import DynamicMultiSensorIFEM, Scaling
from dmifem.chain import stationary_chain_covariance
from dmifem.derived import strain_rows
from dmifem.filter import run_filter
from dmifem.kinematics import ConstantAccelerationModel
from dmifem.smoother import rts_smoother

HELD_OUT = (4, 0.5, 0.15, 0.0)
GAUGE = (3, 0.5, 0.15, 0.0)


def test_reduced_model_reproduces_the_full_nodal_model(mesh, gauges, accelerometers, data):
    window = data.window(0.0, 1.2)
    est = DynamicMultiSensorIFEM(
        mesh, gauges=gauges, accelerometers=accelerometers, gate_alpha=0.0, scaling=Scaling(0.15, 0.05)
    )
    est.prepare(window)
    theta = est.initial_hyperparameters(window)
    est.fit(window, theta=theta)
    n = mesh.n_free
    factor = est.spatial_prior.factor
    to_field = solve_triangular(factor, np.eye(n), lower=False)  # u_bar = R^{-1} w
    grid = est.union_grid(window)
    p1 = stationary_chain_covariance(theta.q, theta.lam, float(np.median(grid.intervals())))
    model = ConstantAccelerationModel(theta.q * np.eye(n))
    # the prior density p(Y^lambda = 0) is left to the covariance-only recursion of run_filter, an
    # independent check of the shortcut r * chain.log_density used by the reduced model
    full = run_filter(
        model,
        grid,
        est.observation_blocks(theta, vectors=to_field),
        window,
        None,
        theta.lam,
        np.zeros(3 * n),
        np.kron(p1, np.eye(n)),
        gate_alpha=0.0,
    )
    full_s = rts_smoother(full, model)
    points = [HELD_OUT, GAUGE]
    z = est.spatial_prior.whiten(est.dimensionless_rows(strain_rows(mesh, points)))  # (n, 2)
    for smoothed, (mean_x, cov_x) in (
        (False, (full.mean_filt, full.cov_filt)),
        (True, (full_s.mean, full_s.cov)),
    ):
        mean, std = est.strain_at(points, smoothed=smoothed)
        mean_full = mean_x[:, :n] @ z
        var_full = np.einsum("ip,kij,jp->kp", z, cov_x[:, :n, :n], z)
        std_full = np.sqrt(np.clip(var_full, 0.0, None))
        scale = np.abs(mean_full).max()
        np.testing.assert_allclose(mean, mean_full, rtol=0.0, atol=1e-6 * scale)
        np.testing.assert_allclose(std, std_full, rtol=1e-5)
    # the exact likelihoods agree; their parts do not (r against n copies of the chain density)
    np.testing.assert_allclose(est.filter_result.log_likelihood, full.log_likelihood, rtol=1e-8)
    np.testing.assert_allclose(full.log_prior_density, n * est.chain_result.log_density, rtol=1e-8)
    np.testing.assert_allclose(
        est.filter_result.log_prior_density, est.basis.r * est.chain_result.log_density, rtol=1e-12
    )


def test_isotropic_model_differs_from_the_full_nodal_model(mesh, gauges, accelerometers, data):
    """The isotropic Galerkin model is an approximation: its strain moments move off the full model."""
    window = data.window(0.0, 1.2)
    common = dict(gauges=gauges, accelerometers=accelerometers, gate_alpha=0.0, scaling=Scaling(0.15, 0.05))
    structured = DynamicMultiSensorIFEM(mesh, **common)
    isotropic = DynamicMultiSensorIFEM(mesh, process_model="isotropic", **common)
    structured.prepare(window)
    theta = structured.initial_hyperparameters(window)
    structured.fit(window, theta=theta)
    isotropic.fit(window, theta=theta)
    mean_s, std_s = structured.strain_at([HELD_OUT])
    mean_i, std_i = isotropic.strain_at([HELD_OUT])
    assert np.all(np.isfinite(mean_i)) and np.all(std_i > 0.0)
    assert not np.allclose(std_i, std_s, rtol=1e-3)
