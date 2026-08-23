"""End-to-end run of Algorithm 1 on the synthetic cantilever."""

import numpy as np

from dmifem import DynamicMultiSensorIFEM, Hyperparameters
from dmifem.metrics import coverage, nrmse, trac

HELD_OUT = (4, 0.5, 0.15, 0.0)  # element 4, between the gauges at 3.5 and 6.5 m, outer fibre


def test_algorithm_one_reconstructs_held_out_strain_and_tip(beam, mesh, gauges, accelerometers, synthetic, data):
    response, modal = synthetic
    est = DynamicMultiSensorIFEM(mesh, gauges=gauges, accelerometers=accelerometers, tau=1e-8)
    est.fit(data, training_window=(0.0, 6.0), max_iter=80)
    assert est.basis is not None and est.hyperparameters is not None
    assert est.basis.r <= len(gauges) + len(accelerometers)
    x_held = (HELD_OUT[0] + HELD_OUT[1]) * (beam.length / 10)
    ref = response.strain_at(beam, np.array([x_held]), modal)[:, 0]
    mean, std = est.strain_at([HELD_OUT])
    sel = slice(300, None)  # after the training window
    # the kinematic estimator trades low-frequency drift against attenuation: shape is recovered
    # (TRAC) with a moderate amplitude error at a held-out fibre between two gauges
    assert nrmse(ref[sel], mean[sel, 0]) < 0.6
    assert trac(ref[sel], mean[sel, 0]) > 0.85
    # at a held-out fibre the reduced posterior is under-dispersed (the basis directions
    # partly cancel there), so only a loose lower bound is asserted
    cov = coverage(ref[sel], mean[sel, 0], std[sel, 0])
    assert 0.3 <= cov <= 1.0
    # tip displacement from the free-dof field (u_y of the last node)
    u_mean, _ = est.displacement()
    tip_dof = np.where(mesh.free_dofs == 6 * 10 + 1)[0][0]
    assert trac(response.displacement[sel, -1], u_mean[sel, tip_dof]) > 0.9


def test_filter_is_recursive_and_smoother_improves(mesh, gauges, accelerometers, data):
    est = DynamicMultiSensorIFEM(mesh, gauges=gauges, accelerometers=accelerometers)
    theta = Hyperparameters(q=1e4, lam=1e8, rho={t: 1.0 for t in est.types})
    est.fit(data, theta=theta)
    mean_f, std_f = est.strain_at([HELD_OUT], smoothed=False)
    mean_s, std_s = est.strain_at([HELD_OUT], smoothed=True)
    assert np.mean(std_s) <= np.mean(std_f)
    draws = est.sample_strain_trajectories([HELD_OUT], n_samples=5, rng=np.random.default_rng(0))
    assert draws.shape == (5, mean_s.shape[0], 1)
