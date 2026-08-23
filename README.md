# dmifem — dynamic multi-sensor inverse finite element method

Reference implementation of the estimator described in the manuscript *Inverse finite element
fusion of strain and acceleration reconstructs operating structures with calibrated uncertainty*.
The strain functional of the inverse finite element method (iFEM) is used as the observation model
of a kinematic state-space estimator whose state holds the displacement, velocity and acceleration
fields. Strain gauges, accelerometers, inclinometers and displacement sensors enter through
kinematic observation operators; a Kalman filter and a Rauch–Tung–Striebel smoother return
full-field histories with posterior uncertainty. No load, mass, stiffness or material model is
required.

## Layout

| Module | Content (manuscript section) |
| --- | --- |
| `elements/` | inverse element interface and the inverse beam element (2.1, 3.2) |
| `mesh.py` | nodes, elements, boundary conditions, gather/transformation matrices |
| `sensors.py` | sensor descriptions, sampling sets and measured data |
| `observation.py` | observation operators per sensor type, gravity offsets, tilt coupling (3.2) |
| `prior.py` | unit-weight section-strain operator `L`, prior precision `K1`, static iFEM solve (3.3, Prop. 3.1) |
| `basis.py` | whitened influence operator, reduced kinematic basis, enrichment, observability Gramian (3.3) |
| `kinematics.py` | exact discretisation of the white-noise-jerk model (3.1) |
| `grid.py` | union grid of asynchronous sampling sets (2.3) |
| `filter.py` | Kalman filter with rate-invariant prior in information form and per-channel gate (3.4) |
| `smoother.py` | Rauch–Tung–Striebel smoother and backward sampling (3.4, 3.6) |
| `hyperparameters.py` | marginal likelihood and derivative-free estimation of `q`, `lambda`, `rho` (3.5) |
| `derived.py` | posterior strain and stress at arbitrary points, complement covariance (3.6) |
| `fatigue.py` | rainflow counting and damage-equivalent load (3.6) |
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
est.fit(data)                              # basis, hyperparameters, filter, smoother
mean, std = est.strain_at([(5, 0.5, 0.1, 0.0)])
```

`examples/cantilever_demo.py` runs a complete synthetic case; `pytest` runs the test suite, which
includes the numerical check of Proposition 3.1 (static limit) and the unit invariance of the
reduced basis.

## Behaviour worth knowing

- **Prior scale.** The prior pseudo-measurement has precision `lambda * dt` per sampling interval.
  The search starts from the classical small weight of sensorless elements (`1e-4` times the mean
  strain-row precision per interval) and is bounded to two decades around it: the marginal
  likelihood is nearly flat in `lambda` whenever the accelerometer-informed directions have little
  strain sensitivity, and an unbounded search drifts towards over-shrinkage.
- **Likelihood.** The prior is part of the state model, not of the data: the marginal likelihood
  uses the physical channels only (pseudo-rows in the likelihood make it improper in `lambda`),
  and the innovation gate is inactive while hyperparameters are estimated (rejected channels would
  otherwise drop out of the likelihood and reward an over-confident model).
- **Filter versus smoother.** Displacement along accelerometer-informed directions comes from
  double integration; the real-time filter lags and drifts, the smoother removes most of it.
  Expect TRAC above 0.9 with 20–50 % NRMSE on such quantities, in line with the manuscript.
- **Interpolation versus extrapolation.** Strain between instrumented elements is recovered
  well; beyond the last gauge the minimum-section-strain prior extrapolates with the error
  growing with the distance from the nearest sensor (Section 6.4 of the manuscript).
- **Frames.** Gauge offsets `(y, z)` refer to the local element frame, whose `y` axis follows the
  `reference` vector passed to `Mesh`; make it the bending plane of interest.
- **Performance.** The recursion pins BLAS to one thread (small dense matrices); a 1 500-sample
  record with `r = 6` runs in about a second on one core.
