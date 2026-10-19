"""Synthetic cantilever: strain gauges and accelerometers, held-out strain and tip displacement.

Run with ``python examples/cantilever_demo.py``. The forward model is a modal simulation of a
steel tube cantilever under a random tip load; the inverse mesh has ten beam elements, four
strain gauges and two accelerometers. The script reports the measurable fraction and the metrics
of Section 4.3 at a held-out strain position and at the tip, the damage-equivalent load ratio with
its credible interval, the damage ratio ``r_D = r_DEL^m`` and the band substitution of Section 4.3.
"""

from __future__ import annotations

import logging

import numpy as np

from dmifem import Accelerometer, DynamicMultiSensorIFEM, Mesh, SensorData, SensorType, StrainGauge
from dmifem.fatigue import band_substitution, damage_ratio, del_of_series
from dmifem.metrics import coverage, crps_gaussian, nrmse, trac
from dmifem.synthetic import CantileverBeam, simulate_cantilever

LENGTH, N_ELEMENTS = 10.0, 10
RADIUS, WALL = 0.15, 0.008
FS, DURATION = 50.0, 30.0
SLOPE, N_REF = 3.0, 1e3
BANDS = ((0.0, 5.0), (5.0, np.inf))  # first bending mode (about 3 Hz) and the higher modes


def build_case(seed: int = 0):
    area, inertia = 2.0 * np.pi * RADIUS * WALL, np.pi * RADIUS**3 * WALL
    beam = CantileverBeam(LENGTH, 210e9 * inertia, 7850.0 * area, RADIUS, n_modes=3)
    x = np.linspace(0.0, LENGTH, N_ELEMENTS + 1)
    mesh = Mesh(np.column_stack([x, np.zeros_like(x), np.zeros_like(x)]), [(i, i + 1) for i in range(N_ELEMENTS)],
                reference=(0.0, 1.0, 0.0))  # local y = global Y, the bending plane
    mesh.fix(0)
    gauges = [StrainGauge(element=e, xi=0.5, y=RADIUS, z=0.0, sigma=2e-6) for e in (0, 3, 6, 9)]
    accels = [Accelerometer(element=e, xi=1.0, direction=(0.0, 1.0, 0.0), sigma=0.02) for e in (4, 9)]
    rng = np.random.default_rng(seed)
    response, modal = simulate_cantilever(beam, x, FS, DURATION, rng, force_std=2e3, force_cutoff=3.0)
    x_g = np.array([(g.element + g.xi) * LENGTH / N_ELEMENTS for g in gauges])
    strain = response.strain_at(beam, x_g, modal) + 2e-6 * rng.standard_normal((response.times.size, len(gauges)))
    acc = np.column_stack([response.acceleration[:, a.element + 1] for a in accels])
    acc += 0.02 * rng.standard_normal(acc.shape)
    data = SensorData({SensorType.STRAIN: (response.times, strain), SensorType.ACCELERATION: (response.times, acc)})
    return beam, mesh, gauges, accels, response, modal, data


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    beam, mesh, gauges, accels, response, modal, data = build_case()
    est = DynamicMultiSensorIFEM(mesh, gauges=gauges, accelerometers=accels)
    est.fit(data, training_window=(0.0, 6.0), max_iter=120)
    held_out = (4, 0.5, RADIUS, 0.0)  # between the gauges at 3.5 and 6.5 m
    x_held = (held_out[0] + held_out[1]) * LENGTH / N_ELEMENTS
    mf = est.measurable_fraction([held_out])[0]
    print(f"reference length {est.scaling.length:.3g} m, reference frequency {est.scaling.reference_frequency:.3g} Hz")
    print(f"measurable fraction of the held-out strain: {mf:.3f}")
    ref = response.strain_at(beam, np.array([x_held]), modal)[:, 0]
    sel = response.times > 6.0
    for label, smoothed in (("filter", False), ("smoother", True)):
        mean, std = est.strain_at([held_out], smoothed=smoothed)
        print(
            f"{label:8s} held-out strain at x = {x_held:.1f} m: NRMSE {100 * nrmse(ref[sel], mean[sel, 0]):.1f} %, "
            f"TRAC {trac(ref[sel], mean[sel, 0]):.3f}, coverage {coverage(ref[sel], mean[sel, 0], std[sel, 0]):.2f}, "
            f"CRPS {crps_gaussian(ref[sel], mean[sel, 0], std[sel, 0]):.3f}"
        )
    u_mean, _ = est.displacement()
    tip_dof = np.where(mesh.free_dofs == 6 * N_ELEMENTS + 1)[0][0]
    print(f"tip displacement: NRMSE {100 * nrmse(response.displacement[sel, -1], u_mean[sel, tip_dof]):.1f} %, "
          f"TRAC {trac(response.displacement[sel, -1], u_mean[sel, tip_dof]):.3f}")
    mean_s, _ = est.strain_at([held_out])
    del_ref = del_of_series(ref[sel], SLOPE, N_REF)
    ratio = del_of_series(mean_s[sel, 0], SLOPE, N_REF) / del_ref
    # joint smoothing samples: reduced trajectories plus complement trajectories (Eq. fieldcov)
    draws = est.sample_strain_trajectories([held_out], n_samples=50, rng=np.random.default_rng(1))
    dels = np.array([del_of_series(d[sel, 0], SLOPE, N_REF) for d in draws]) / del_ref
    print(f"damage-equivalent load ratio (m = {SLOPE:g}): {ratio:.3f}, 95 percent interval "
          f"[{np.percentile(dels, 2.5):.3f}, {np.percentile(dels, 97.5):.3f}], "
          f"damage ratio {damage_ratio(ratio, SLOPE):.3f}")
    band_ratios, total = band_substitution(ref[sel], mean_s[sel, 0], FS, BANDS, SLOPE, N_REF)
    for (low, high), r_d in zip(BANDS, band_ratios, strict=True):
        print(f"damage ratio with the band [{low:g}, {high:g}) Hz substituted: {r_d:.3f}")
    print(f"damage ratio of the full reconstruction: {total:.3f} (bands do not add in rainflow counting)")
    print(f"basis size r = {est.basis.r}, hyperparameters {est.hyperparameters}")


if __name__ == "__main__":
    main()
