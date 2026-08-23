"""Shared fixtures: a clamped cantilever mesh and a synthetic data set."""

from __future__ import annotations

import numpy as np
import pytest

from dmifem import Accelerometer, Mesh, SensorData, SensorType, StrainGauge
from dmifem.synthetic import CantileverBeam, simulate_cantilever

N_ELEMENTS = 10
LENGTH = 10.0


@pytest.fixture(scope="session")
def beam() -> CantileverBeam:
    # steel tube, outer radius 0.15 m, wall 8 mm
    radius, wall = 0.15, 0.008
    area = 2.0 * np.pi * radius * wall
    inertia = np.pi * radius**3 * wall
    return CantileverBeam(length=LENGTH, ei=210e9 * inertia, mass_per_length=7850.0 * area,
                          fibre_offset=radius, n_modes=3)


@pytest.fixture(scope="session")
def mesh() -> Mesh:
    x = np.linspace(0.0, LENGTH, N_ELEMENTS + 1)
    nodes = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    connectivity = [(i, i + 1) for i in range(N_ELEMENTS)]
    m = Mesh(nodes, connectivity, reference=(0.0, 1.0, 0.0))  # local y = global Y, the bending plane
    m.fix(0)
    return m


@pytest.fixture(scope="session")
def gauges() -> list[StrainGauge]:
    return [StrainGauge(element=e, xi=0.5, y=0.15, z=0.0, sigma=2e-6) for e in (0, 3, 6, 9)]


@pytest.fixture(scope="session")
def accelerometers() -> list[Accelerometer]:
    return [Accelerometer(element=e, xi=1.0, direction=(0.0, 1.0, 0.0), sigma=0.02) for e in (4, 9)]


@pytest.fixture(scope="session")
def synthetic(beam, mesh):
    rng = np.random.default_rng(0)
    fs, duration = 50.0, 20.0
    response, modal = simulate_cantilever(beam, mesh.nodes[:, 0], fs, duration, rng,
                                          force_std=2e3, force_cutoff=3.0)
    return response, modal


@pytest.fixture(scope="session")
def data(beam, synthetic, gauges, accelerometers) -> SensorData:
    response, modal = synthetic
    rng = np.random.default_rng(1)
    x_g = np.array([(g.element + g.xi) * (LENGTH / N_ELEMENTS) for g in gauges])
    strain = response.strain_at(beam, x_g, modal)
    strain += 2e-6 * rng.standard_normal(strain.shape)
    acc = np.column_stack([response.acceleration[:, a.element + 1] for a in accelerometers])
    acc += 0.02 * rng.standard_normal(acc.shape)
    return SensorData({SensorType.STRAIN: (response.times, strain),
                       SensorType.ACCELERATION: (response.times, acc)})
