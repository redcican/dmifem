"""Exact discretization of the white-noise-jerk model."""

import numpy as np
from scipy.integrate import quad

from dmifem.kinematics import ConstantAccelerationModel, process_noise_core, transition_core


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
