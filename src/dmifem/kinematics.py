"""Kinematic state-space model (manuscript Section 3.1, Eqs. ct–Q).

The state stacks the reduced displacement, velocity and acceleration coordinates in the
dimensionless time of Eq. (scaling). Between samples the acceleration is locally constant with
white-noise jerk of spatial covariance ``Sigma_j``; the exact discretization over an interval
``dt`` is ``F = T(dt) (x) I_r`` and ``Q = M(dt) (x) Sigma_j`` with the Kronecker product. With the
structured covariance ``Sigma_j = q K1^{-1}`` of Eq. (sigmaj) and the K1-orthonormal basis, the
reduced jerk covariance is ``q I_r``.
"""

from __future__ import annotations

import numpy as np

from dmifem.elements.base import Array


def covariance_root(cov: Array) -> Array:
    """Factor ``A`` with ``A A^T = cov`` from the eigendecomposition (negative round-off clipped)."""
    sym = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    vals, vecs = np.linalg.eigh(sym)
    return vecs * np.sqrt(np.clip(vals, 0.0, None))[None, :]


def transition_core(dt: float) -> Array:
    """Scalar transition matrix ``T(dt)`` of the constant-acceleration chain."""
    return np.array([[1.0, dt, 0.5 * dt**2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])


def process_noise_core(dt: float) -> Array:
    """Scalar process-noise matrix ``M(dt)`` of white-noise jerk with unit intensity."""
    return np.array(
        [
            [dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
            [dt**4 / 8.0, dt**3 / 3.0, dt**2 / 2.0],
            [dt**3 / 6.0, dt**2 / 2.0, dt],
        ]
    )


class ConstantAccelerationModel:
    """White-noise-jerk model on ``r`` coordinates.

    Parameters
    ----------
    jerk_covariance:
        Spatial covariance ``Sigma_j`` of the jerk, shape ``(r, r)`` (``q I`` in the isotropic
        case).
    """

    def __init__(self, jerk_covariance: Array) -> None:
        sigma = np.asarray(jerk_covariance, dtype=float)
        if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
            raise ValueError("jerk covariance must be square")
        self.jerk_covariance = sigma
        self.r = sigma.shape[0]

    @property
    def dim(self) -> int:
        """State dimension ``3r``."""
        return 3 * self.r

    def transition(self, dt: float) -> Array:
        """``F(dt)`` of Eq. (dt)."""
        return np.kron(transition_core(dt), np.eye(self.r))

    def process_noise(self, dt: float) -> Array:
        """``Q(dt)`` of Eq. (Q)."""
        return np.kron(process_noise_core(dt), self.jerk_covariance)

    @staticmethod
    def block(state: Array, index: int, r: int) -> Array:
        """Return block ``index`` (0 displacement, 1 velocity, 2 acceleration) of a state."""
        return state[..., index * r : (index + 1) * r]
