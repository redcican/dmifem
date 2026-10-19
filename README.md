# dmifem — dynamic multi-sensor inverse finite element method

Reference implementation of the estimator described in the manuscript *Inverse finite element
fusion of strain and acceleration reconstructs operating structures with quantified uncertainty*.
The strain functional of the inverse finite element method (iFEM) is used as the observation model
of a kinematic state-space estimator whose state holds the displacement, velocity and acceleration
fields. Strain gauges, accelerometers, inclinometers and displacement sensors enter through
kinematic observation operators; a Kalman filter and a Rauch–Tung–Striebel smoother return
full-field histories with posterior uncertainty. No load, mass, stiffness or material model is
required: the model holds the geometry, the boundary conditions and the sensor metadata, and works
in dimensionless coordinates set by a reference length and a reference time.

## Layout

| Module | Content (manuscript section) |
| --- | --- |
| `elements/` | inverse element interface and the inverse beam element (2.1, 3.2) |
| `mesh.py` | nodes, elements, boundary conditions, gather/transformation matrices |
| `sensors.py` | sensor descriptions, sampling sets and measured data |
| `observation.py` | observation operators per sensor type, gravity offsets, tilt coupling (3.2) |
| `scaling.py` | reference length and time, dimensionless dofs, default rules (3.3, Eq. scaling) |
| `prior.py` | section-strain operator `L`, Matérn-3/2 correlation along the members, free-end conditioning, section factors, `K1 = R^T R` through its factor, static iFEM solve (3.3, Eq. prior, Prop. 3.1) |
| `basis.py` | `K1`-orthonormal basis of `K1^{-1} W`, enrichment, observability Gramian (3.3, Eq. basis) |
| `kinematics.py` | exact discretisation of the white-noise-jerk model (3.1) |
| `chain.py` | scalar kinematic chain: stationary covariance, filter, smoother and samples (Prop. reduced, Eq. exactll) |
| `grid.py` | union grid of asynchronous sampling sets (2.3) |
| `filter.py` | Kalman filter with the pseudo-measurement of precision `lambda dt`, per-channel gate and the exact likelihood (3.4, Eq. exactll) |
| `smoother.py` | Rauch–Tung–Striebel smoother and vectorized backward sampling (3.4, 3.6) |
| `hyperparameters.py` | derivative-free estimation of `q`, `lambda`, `rho` by the exact marginal likelihood (3.5) |
| `derived.py` | posterior strain and stress at arbitrary points, complement covariance, measurable fraction (3.6, Eqs. fieldcov, vr) |
| `fatigue.py` | rainflow counting, damage-equivalent load, damage ratio and band substitution (3.6, 4.3) |
| `metrics.py` | NRMSE, TRAC, band-limited spectral error, coverage, CRPS (4.3) |
| `estimator.py` | `DynamicMultiSensorIFEM`, the orchestration of Algorithm 1 |

## Installation

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from dmifem import DynamicMultiSensorIFEM, Mesh, StrainGauge, Accelerometer, SensorData, SensorType

mesh = Mesh(nodes, connectivity, reference=(0, 1, 0))  # beam mesh, local y axis = global Y
mesh.fix(node=0)                          # clamped root
est = DynamicMultiSensorIFEM(mesh, gauges=[StrainGauge(element=2, xi=0.5, y=0.1, z=0.0, sigma=2e-6)],
                             accelerometers=[Accelerometer(element=9, xi=1.0, direction=(0, 1, 0), sigma=0.02)])
data = SensorData({SensorType.STRAIN: (t_strain, eps), SensorType.ACCELERATION: (t_acc, acc)})
est.fit(data)                              # scaling, basis, hyperparameters, filter, smoother
mean, std = est.strain_at([(5, 0.5, 0.1, 0.0)])  # std includes the directions no sensor sees
mf = est.measurable_fraction([(5, 0.5, 0.1, 0.0)])
```

`examples/cantilever_demo.py` runs a complete synthetic case. `pytest` runs the test suite, which
includes the static limit of Proposition 3.1, the batch Gaussian check of the recursions and the
exact likelihood, the full nodal model against the reduced model with its complement, and the
invariance of the estimates to the units.

## Behaviour worth knowing

- **Scaling.** Translations are divided by the reference length `ell` (the section outer radius
  when sections are given, otherwise the largest gauge offset, otherwise the length of the
  structure) and time by `T = 1 / (2 pi f_ref)`,
  with `f_ref` the Welch peak of the accelerometer spectrum (the strain spectrum when there are no
  accelerometers). Channels are whitened by their nominal noise, so the hyperparameters are
  dimensionless and the estimates do not depend on the units of the data. `T` is set by the first
  `prepare` or `fit` and kept afterwards; pass `scaling=Scaling(ell, T)` to fix both.
- **Process model.** The default `process_model="structured"` takes the jerk covariance `q K1^{-1}`;
  with the `K1`-orthonormal basis the reduction is exact and the reduced process covariance is `q I`.
  `"isotropic"` keeps `q I` on the field, a Galerkin approximation retained for comparison; it has
  no independent complement, so its outputs carry the reduced posterior alone.
- **Prior.** The section resultants follow a Matérn-3/2 correlation along the members with length
  `ell_c` (the length of the structure by default, not estimated), conditioned on vanishing bending
  moments at free ends. `correlated_prior=False` gives the unit-weight `K1` of the classical method.
  The kernel of the graph distance is positive definite on unbranched members; on branched or
  closed frames a shorter `correlation_length` or the uncorrelated prior may be needed.
- **Complement.** Field directions that no sensor sees keep their prior covariance, scaled by the
  covariance of the scalar chain. `strain_at`, `displacement` and the samples include them;
  `measurable_fraction` reports the share of the prior variance of a quantity that the sensors can
  reduce.
- **Likelihood.** Hyperparameters maximize the exact marginal likelihood `log p(y | z = 0)` of
  Eq. (exactll): the innovation terms of the pseudo-measurements and the physical channels, minus
  the density of the zero pseudo-measurements, which for the structured model is `r` copies of the
  scalar chain. `objective="physical"` keeps the sum over the physical innovations for comparison.
  The innovation gate is inactive while hyperparameters are estimated.
- **Prior scale.** The pseudo-measurement has precision `lambda * dt` per sampling interval. The
  search starts from the classical small weight of sensorless elements (`1e-4` times the mean
  precision per interval of the displacement-type channels), with `q = 1 / lambda`, and is bounded
  to three decades in `q`, two in `lambda` and one in the noise scales.
- **Numerics.** `K1` is held through its triangular factor `R`, and products with `K1` and `K1^{-1}`
  are formed in the whitened coordinates `w = R u`; the condition number of `K1` itself reaches
  `1e13` on a ten-element cantilever.
- **Filter versus smoother.** Displacement along accelerometer-informed directions comes from
  double integration; the real-time filter lags and drifts, the smoother removes most of it.
  Expect TRAC above 0.9 with 20–50 % NRMSE on such quantities, in line with the manuscript.
- **Interpolation versus extrapolation.** Strain between instrumented elements is recovered
  well; beyond the last gauge the prior extrapolates, the measurable fraction drops and the
  posterior standard deviation grows with the distance from the nearest sensor (Section 6.4 of the
  manuscript).
- **Frames.** Gauge offsets `(y, z)` refer to the local element frame, whose `y` axis follows the
  `reference` vector passed to `Mesh`; make it the bending plane of interest.
- **Performance.** The recursion pins BLAS to one thread (small dense matrices); a 1 500-sample
  record with `r = 6` runs in about a second on one core.
