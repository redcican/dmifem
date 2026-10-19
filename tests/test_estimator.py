"""End-to-end run of Algorithm 1 on the synthetic cantilever."""

import numpy as np
import pytest

from dmifem import DynamicMultiSensorIFEM, Scaling
from dmifem.metrics import coverage, nrmse, trac

HELD_OUT = (4, 0.5, 0.15, 0.0)  # element 4, between the gauges at 3.5 and 6.5 m, outer fibre


def test_algorithm_one_reconstructs_held_out_strain_and_tip(beam, mesh, gauges, accelerometers, synthetic, data):
    response, modal = synthetic
    est = DynamicMultiSensorIFEM(mesh, gauges=gauges, accelerometers=accelerometers)
    est.fit(data, training_window=(0.0, 6.0), max_iter=80)
    assert est.basis is not None and est.hyperparameters is not None
    # r = dim W: four gauges and two accelerometers with independent rows
    assert est.basis.r == est.basis.n_informed == len(gauges) + len(accelerometers)
    # reference length from the gauge offsets, reference time from the accelerometer spectrum
    assert est.length == pytest.approx(0.15)
    assert 0.0 < est.scaling.reference_frequency < 25.0
    np.testing.assert_allclose(est.times, response.times, rtol=0.0, atol=1e-9)
    mf = est.measurable_fraction([HELD_OUT])
    assert 0.0 < mf[0] < 1.0
    x_held = (HELD_OUT[0] + HELD_OUT[1]) * (beam.length / 10)
    ref = response.strain_at(beam, np.array([x_held]), modal)[:, 0]
    mean, std = est.strain_at([HELD_OUT])
    sel = slice(300, None)  # after the training window
    # shape is recovered (TRAC) with a moderate amplitude error at a held-out fibre between two
    # gauges, where part of the strain lies in the complement that no sensor sees
    assert nrmse(ref[sel], mean[sel, 0]) < 0.6
    assert trac(ref[sel], mean[sel, 0]) > 0.85
    # the standard deviation includes the complement of Eq. (fieldcov)
    cov = coverage(ref[sel], mean[sel, 0], std[sel, 0])
    assert 0.3 <= cov <= 1.0
    # tip displacement from the free-dof field (u_y of the last node), in the units of the data
    u_mean, _ = est.displacement()
    tip_dof = np.where(mesh.free_dofs == 6 * 10 + 1)[0][0]
    assert trac(response.displacement[sel, -1], u_mean[sel, tip_dof]) > 0.9


def test_filter_is_recursive_and_smoother_improves(mesh, gauges, accelerometers, data):
    est = DynamicMultiSensorIFEM(mesh, gauges=gauges, accelerometers=accelerometers, scaling=Scaling(0.15, 0.05))
    theta = est.initial_hyperparameters(data)
    est.fit(data, theta=theta)
    mean_f, std_f = est.strain_at([HELD_OUT], smoothed=False)
    mean_s, std_s = est.strain_at([HELD_OUT], smoothed=True)
    assert np.all(std_s <= std_f * (1.0 + 1e-9))
    draws = est.sample_strain_trajectories([HELD_OUT], n_samples=5, rng=np.random.default_rng(0))
    assert draws.shape == (5, mean_s.shape[0], 1)
    reduced_only = est.sample_strain_trajectories(
        [HELD_OUT], n_samples=5, rng=np.random.default_rng(0), include_complement=False
    )
    assert reduced_only.shape == draws.shape


def test_exact_and_physical_objectives_differ(mesh, gauges, accelerometers, data):
    window = data.window(0.0, 2.0)
    est = DynamicMultiSensorIFEM(mesh, gauges=gauges, accelerometers=accelerometers, scaling=Scaling(0.15, 0.05))
    theta = est.initial_hyperparameters(window)
    exact = est.negative_log_likelihood(window, theta, "exact")
    physical = est.negative_log_likelihood(window, theta, "physical")
    assert np.isfinite(exact) and np.isfinite(physical)
    assert exact != pytest.approx(physical, rel=1e-6)
    with pytest.raises(ValueError):
        est.negative_log_likelihood(window, theta, "other")


def test_invalid_process_model_is_rejected(mesh, gauges):
    with pytest.raises(ValueError):
        DynamicMultiSensorIFEM(mesh, gauges=gauges, process_model="modal")


def test_prepare_keeps_the_reference_time_and_discards_old_results(mesh, gauges, accelerometers, data):
    est = DynamicMultiSensorIFEM(mesh, gauges=gauges, accelerometers=accelerometers)
    window = data.window(0.0, 2.0)
    est.fit(window, theta=est.initial_hyperparameters(window))
    time_scale = est.time_scale
    est.prepare(data.window(10.0, 20.0))
    assert est.time_scale == time_scale
    assert est.filter_result is None and est.smoother_result is None and est.chain_result is None
    with pytest.raises(RuntimeError):
        est.strain_at([HELD_OUT])
