"""Change of units (Eq. scaling): the same records in (m, s) and (mm, ms) give the same estimates.

The reference length and time follow the default rules in each unit system, and the
dimensionless hyperparameters are shared, so the estimates transformed back must agree to
round-off. The cantilever mixes axial and bending rows, so the check is not trivial.
"""

from __future__ import annotations

import numpy as np
import pytest

from dmifem import Accelerometer, DynamicMultiSensorIFEM, Mesh, SensorData, SensorType, StrainGauge
from dmifem.scaling import default_length, default_time_scale

N_EL, LENGTH, RADIUS = 10, 10.0, 0.15
POINTS_SI = [(4, 0.5, RADIUS, 0.0), (8, 0.25, RADIUS, 0.0)]


def _case(unit_length: float, unit_time: float, data_si: SensorData):
    """Estimator and data with lengths in ``unit_length`` metres and times in ``unit_time`` seconds."""
    x = np.linspace(0.0, LENGTH, N_EL + 1) / unit_length
    mesh = Mesh(
        np.column_stack([x, np.zeros_like(x), np.zeros_like(x)]),
        [(i, i + 1) for i in range(N_EL)],
        reference=(0.0, 1.0, 0.0),
    )
    mesh.fix(0)
    acc_factor = unit_time**2 / unit_length
    gauges = [StrainGauge(element=e, xi=0.5, y=RADIUS / unit_length, z=0.0, sigma=2e-6) for e in (0, 3, 6, 9)]
    accels = [
        Accelerometer(element=e, xi=1.0, direction=(0.0, 1.0, 0.0), sigma=0.02 * acc_factor) for e in (4, 9)
    ]
    t_s, eps = data_si.streams[SensorType.STRAIN]
    t_a, acc = data_si.streams[SensorType.ACCELERATION]
    data = SensorData(
        {
            SensorType.STRAIN: (t_s / unit_time, eps),
            SensorType.ACCELERATION: (t_a / unit_time, acc * acc_factor),
        }
    )
    est = DynamicMultiSensorIFEM(mesh, gauges=gauges, accelerometers=accels, gate_alpha=0.0)
    points = [(e, xi, y / unit_length, z / unit_length) for e, xi, y, z in POINTS_SI]
    return est, data, points


def test_estimates_are_invariant_to_the_units(data):
    window = data.window(0.0, 3.0)
    si, data_si, points_si = _case(1.0, 1.0, window)
    mm, data_mm, points_mm = _case(1e-3, 1e-3, window)
    si.prepare(data_si)
    mm.prepare(data_mm)
    assert si.scaling.length == pytest.approx(RADIUS)
    assert mm.scaling.length == pytest.approx(1e3 * si.scaling.length, rel=1e-12)
    assert mm.scaling.time == pytest.approx(1e3 * si.scaling.time, rel=1e-9)
    np.testing.assert_allclose(mm.basis.normalized_eigenvalues, si.basis.normalized_eigenvalues, rtol=1e-8, atol=1e-14)
    theta = si.initial_hyperparameters(data_si)
    theta_mm = mm.initial_hyperparameters(data_mm)
    assert theta_mm.q == pytest.approx(theta.q, rel=1e-7)
    assert theta_mm.lam == pytest.approx(theta.lam, rel=1e-7)
    si.fit(data_si, theta=theta)
    mm.fit(data_mm, theta=theta)
    np.testing.assert_allclose(mm.times * 1e-3, si.times, rtol=1e-12, atol=1e-12)
    for smoothed in (False, True):
        m_si, s_si = si.strain_at(points_si, smoothed=smoothed)
        m_mm, s_mm = mm.strain_at(points_mm, smoothed=smoothed)
        np.testing.assert_allclose(m_mm, m_si, rtol=0.0, atol=1e-7 * np.abs(m_si).max())
        np.testing.assert_allclose(s_mm, s_si, rtol=1e-7)
    factor = mm.dof_scale / si.dof_scale  # 1e3 on translations, 1 on rotations
    u_si, su_si = si.displacement()
    u_mm, su_mm = mm.displacement()
    np.testing.assert_allclose(u_mm / factor, u_si, rtol=0.0, atol=1e-7 * np.abs(u_si).max())
    np.testing.assert_allclose(su_mm / factor, su_si, rtol=1e-7)
    a_si, _ = si.acceleration()
    a_mm, _ = mm.acceleration()
    factor_a = factor * (si.scaling.time / mm.scaling.time) ** 2
    np.testing.assert_allclose(a_mm / factor_a, a_si, rtol=0.0, atol=1e-7 * np.abs(a_si).max())
    np.testing.assert_allclose(mm.filter_result.log_likelihood, si.filter_result.log_likelihood, rtol=1e-9)


def test_reference_time_skips_a_stream_without_variance():
    t = np.arange(400) / 50.0
    strain = np.sin(2.0 * np.pi * 2.0 * t)[:, None]
    data = SensorData(
        {SensorType.ACCELERATION: (t, np.full((t.size, 2), np.nan)), SensorType.STRAIN: (t, strain)}
    )
    f_ref = 1.0 / (2.0 * np.pi * default_time_scale(data))
    assert f_ref == pytest.approx(2.0, abs=50.0 / 256)


def test_reference_length_falls_back_to_the_given_length():
    assert default_length((), None, fallback=10.0) == 10.0
    with pytest.raises(ValueError):
        default_length((), None, fallback=0.0)


def test_accelerometer_only_estimates_are_invariant_to_the_length_unit(data):
    """Without gauge offsets the reference length is the length of the structure, in any unit."""
    window = data.window(0.0, 2.0)
    t_a, acc = window.streams[SensorType.ACCELERATION]
    out = []
    for unit in (1.0, 1e-3):  # metres, millimetres; seconds in both
        x = np.linspace(0.0, LENGTH, N_EL + 1) / unit
        mesh = Mesh(np.column_stack([x, np.zeros_like(x), np.zeros_like(x)]),
                    [(i, i + 1) for i in range(N_EL)], reference=(0.0, 1.0, 0.0))
        mesh.fix(0)
        accels = [Accelerometer(element=e, xi=1.0, direction=(0.0, 1.0, 0.0), sigma=0.02 / unit) for e in (4, 9)]
        est = DynamicMultiSensorIFEM(mesh, accelerometers=accels, gate_alpha=0.0)
        est_data = SensorData({SensorType.ACCELERATION: (t_a, acc / unit)})
        est.prepare(est_data)
        assert est.length == pytest.approx(LENGTH / unit)
        est.fit(est_data, theta=est.initial_hyperparameters(est_data))
        to_metres = np.where(est.dof_scale == est.length, unit, 1.0)  # translations only
        u, su = est.displacement()
        out.append((u * to_metres, su * to_metres))
    np.testing.assert_allclose(out[1][0], out[0][0], rtol=0.0, atol=1e-7 * np.abs(out[0][0]).max())
    np.testing.assert_allclose(out[1][1], out[0][1], rtol=1e-6)
