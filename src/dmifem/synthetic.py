"""Synthetic reference data: modal simulation of an Euler–Bernoulli cantilever.

Used by the tests and the examples to produce full-field displacements, rotations,
accelerations and surface strains of a clamped beam under a random tip load. The forward
model is independent of the inverse mesh (analytical mode shapes, exact modal integration).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from dmifem.elements.base import Array

_BETA_L = np.array([1.8751041, 4.6940911, 7.8547574, 10.9955407, 14.1371684])
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2 renamed trapz


@dataclass(frozen=True)
class CantileverBeam:
    """Uniform cantilever: length, bending stiffness ``EI``, mass per length, fibre offset."""

    length: float
    ei: float
    mass_per_length: float
    fibre_offset: float
    n_modes: int = 3

    def beta(self) -> Array:
        return _BETA_L[: self.n_modes] / self.length

    def omega(self) -> Array:
        return self.beta() ** 2 * np.sqrt(self.ei / self.mass_per_length)

    def modes(self, x: Array) -> tuple[Array, Array, Array]:
        """Mode shapes and their first two spatial derivatives, shape ``(n_modes, len(x))``."""
        x = np.asarray(x, dtype=float)
        beta = self.beta()
        bl = beta * self.length
        sigma = (np.cosh(bl) + np.cos(bl)) / (np.sinh(bl) + np.sin(bl))
        bx = beta[:, None] * x[None, :]
        s = sigma[:, None]
        phi = np.cosh(bx) - np.cos(bx) - s * (np.sinh(bx) - np.sin(bx))
        dphi = beta[:, None] * (np.sinh(bx) + np.sin(bx) - s * (np.cosh(bx) - np.cos(bx)))
        ddphi = beta[:, None] ** 2 * (np.cosh(bx) + np.cos(bx) - s * (np.sinh(bx) + np.sin(bx)))
        return phi, dphi, ddphi

    def modal_mass(self) -> Array:
        x = np.linspace(0.0, self.length, 2001)
        phi, _, _ = self.modes(x)
        return self.mass_per_length * _trapezoid(phi**2, x, axis=1)


@dataclass(frozen=True)
class CantileverResponse:
    """Nodal response histories of the forward model."""

    times: Array
    x_nodes: Array
    displacement: Array  # u_y (N, n_nodes)
    rotation: Array  # t_z = du_y/dx (N, n_nodes)
    acceleration: Array  # u_y_ddot (N, n_nodes)
    curvature: Array  # d2u_y/dx2 (N, n_nodes)

    def surface_strain(self, fibre_offset: float) -> Array:
        """Fibre strain ``-y kappa_z`` at the nodes for a fibre at offset ``y``."""
        return -fibre_offset * self.curvature

    def strain_at(self, beam: CantileverBeam, x: Array, modal: Array) -> Array:
        """Fibre strain at arbitrary positions from the modal coordinates ``(N, n_modes)``."""
        _, _, ddphi = beam.modes(np.atleast_1d(x))
        return -beam.fibre_offset * modal @ ddphi


def simulate_cantilever(
    beam: CantileverBeam,
    x_nodes: Array,
    fs: float,
    duration: float,
    rng: np.random.Generator,
    damping_ratio: float = 0.01,
    force_std: float = 1.0,
    force_cutoff: float | None = None,
) -> tuple[CantileverResponse, Array]:
    """Simulate the beam under a random tip force with exact modal integration.

    Returns the nodal response and the modal coordinates ``(N, n_modes)``.
    """
    dt = 1.0 / fs
    n = int(round(duration * fs)) + 1
    times = np.arange(n) * dt
    omega = beam.omega()
    mass = beam.modal_mass()
    phi_tip, _, _ = beam.modes(np.array([beam.length]))
    force = force_std * rng.standard_normal(n)
    if force_cutoff is not None:
        from scipy.signal import butter, filtfilt

        b, a = butter(2, force_cutoff / (0.5 * fs))
        force = filtfilt(b, a, force)
    modal = np.zeros((n, beam.n_modes))
    modal_acc = np.zeros((n, beam.n_modes))
    for i, (w, m) in enumerate(zip(omega, mass, strict=True)):
        a_c = np.array([[0.0, 1.0], [-(w**2), -2.0 * damping_ratio * w]])
        b_c = np.array([0.0, phi_tip[i, 0] / m])
        big = np.zeros((3, 3))
        big[:2, :2], big[:2, 2] = a_c, b_c
        big_d = expm(big * dt)
        a_d, b_d = big_d[:2, :2], big_d[:2, 2]
        state = np.zeros(2)
        for k in range(n):
            modal[k, i] = state[0]
            modal_acc[k, i] = a_c[1] @ state + b_c[1] * force[k]
            state = a_d @ state + b_d * force[k]
    phi, dphi, ddphi = beam.modes(x_nodes)
    response = CantileverResponse(
        times=times,
        x_nodes=np.asarray(x_nodes, dtype=float),
        displacement=modal @ phi,
        rotation=modal @ dphi,
        acceleration=modal_acc @ phi,
        curvature=modal @ ddphi,
    )
    return response, modal
