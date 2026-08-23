"""Orchestration of Algorithm 1: offline assembly, hyperparameters, filter and smoother.

:class:`DynamicMultiSensorIFEM` assembles the observation operators and the prior from the
mesh and the sensors, forms the whitened influence operator and the reduced basis, estimates
the hyperparameters on a training window, runs the filter over the union grid and, optionally,
the smoother, and exposes the reconstructed fields with their uncertainty.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from dmifem.basis import ReducedBasis, enrich_basis, influence_operator, reduced_basis
from dmifem.derived import complement_covariance, posterior_linear, strain_rows
from dmifem.elements.base import Array
from dmifem.filter import FilterResult, ReducedObservation, run_filter
from dmifem.grid import UnionGrid, build_union_grid
from dmifem.hyperparameters import Hyperparameters, estimate_hyperparameters
from dmifem.kinematics import ConstantAccelerationModel
from dmifem.mesh import Mesh
from dmifem.observation import (
    ObservationOperator,
    acceleration_operator,
    displacement_operator,
    inclination_operator,
    strain_operator,
)
from dmifem.prior import PriorOperator, assemble_prior
from dmifem.sensors import (
    Accelerometer,
    DisplacementSensor,
    Inclinometer,
    SensorData,
    SensorType,
    StrainGauge,
)
from dmifem.smoother import SmootherResult, rts_smoother, sample_trajectories

logger = logging.getLogger(__name__)


class DynamicMultiSensorIFEM:
    """Dynamic multi-sensor inverse finite element estimator.

    Parameters
    ----------
    mesh:
        Inverse mesh with boundary conditions applied.
    gauges, accelerometers, inclinometers, displacement_sensors:
        Sensors of the four types.
    tau:
        Truncation level of the reduced basis.
    n_enrichment:
        Number of smoothest admissible fields appended to the basis (``0`` by default).
    tilt_coupling:
        Whether the first-order gravity tilt term of the accelerometers is retained.
    gate_alpha:
        Level of the per-channel innovation gate (``0`` disables it).
    structured_jerk:
        Use ``Sigma_j = q K_lambda^{-1}`` instead of the isotropic ``q I``.
    """

    def __init__(
        self,
        mesh: Mesh,
        gauges: Sequence[StrainGauge] = (),
        accelerometers: Sequence[Accelerometer] = (),
        inclinometers: Sequence[Inclinometer] = (),
        displacement_sensors: Sequence[DisplacementSensor] = (),
        tau: float = 1e-5,
        n_enrichment: int = 0,
        tilt_coupling: bool = False,
        gate_alpha: float = 1e-3,
        structured_jerk: bool = False,
    ) -> None:
        self.mesh = mesh
        self.tau = tau
        self.n_enrichment = n_enrichment
        self.gate_alpha = gate_alpha
        self.structured_jerk = structured_jerk
        self.operators: dict[SensorType, ObservationOperator] = {}
        if gauges:
            self.operators[SensorType.STRAIN] = strain_operator(mesh, gauges)
        if accelerometers:
            self.operators[SensorType.ACCELERATION] = acceleration_operator(
                mesh, accelerometers, tilt_coupling
            )
        if inclinometers:
            self.operators[SensorType.INCLINATION] = inclination_operator(mesh, inclinometers)
        if displacement_sensors:
            self.operators[SensorType.DISPLACEMENT] = displacement_operator(
                mesh, displacement_sensors
            )
        if not self.operators:
            raise ValueError("at least one sensor is required")
        self.prior: PriorOperator = assemble_prior(mesh)
        self.precision_prior: Array = self.prior.precision
        self.basis: ReducedBasis | None = None
        self.hyperparameters: Hyperparameters | None = None
        self.grid: UnionGrid | None = None
        self.model: ConstantAccelerationModel | None = None
        self.filter_result: FilterResult | None = None
        self.smoother_result: SmootherResult | None = None

    # ---- offline stage ---------------------------------------------------------------------
    @property
    def types(self) -> list[SensorType]:
        return list(self.operators)

    def _stacked_rows(self) -> tuple[Array, Array]:
        rows = np.vstack([op.rows for op in self.operators.values()])
        sigma = np.concatenate([op.sigma for op in self.operators.values()])
        return rows, sigma

    def form_basis(self, lam: float) -> ReducedBasis:
        """Form the whitened influence operator and the reduced basis for a prior scale."""
        rows, sigma = self._stacked_rows()
        influence = influence_operator(rows, sigma, self.precision_prior, lam)
        basis = reduced_basis(influence, self.tau)
        basis = enrich_basis(basis, self.precision_prior, self.n_enrichment)
        self.basis = basis
        logger.info("reduced basis with r = %d (tau = %g)", basis.r, self.tau)
        return basis

    def _reduced_blocks(self, theta: Hyperparameters) -> dict[SensorType, ReducedObservation]:
        assert self.basis is not None
        phi = self.basis.vectors
        r = phi.shape[1]
        blocks: dict[SensorType, ReducedObservation] = {}
        for stype, op in self.operators.items():
            rows = np.zeros((op.n_channels, 3 * r))
            if op.block == "acceleration":
                rows[:, 2 * r :] = op.rows @ phi
                if op.tilt_rows is not None:
                    rows[:, :r] = op.tilt_rows @ phi
            else:
                rows[:, :r] = op.rows @ phi
            variance = theta.rho.get(stype, 1.0) * op.sigma**2
            blocks[stype] = ReducedObservation(rows, op.offsets, variance)
        return blocks

    def _model(self, theta: Hyperparameters) -> ConstantAccelerationModel:
        assert self.basis is not None
        phi = self.basis.vectors
        if self.structured_jerk:
            rows, sigma = self._stacked_rows()
            whitened = rows / sigma[:, None]
            k_lambda = whitened.T @ whitened + theta.lam * self.precision_prior
            sigma_j = theta.q * phi.T @ np.linalg.solve(k_lambda, phi)
        else:
            sigma_j = theta.q * np.eye(phi.shape[1])
        return ConstantAccelerationModel(sigma_j)

    def _initial_moments(self, theta: Hyperparameters, sigma_v: float, sigma_a: float) -> tuple[Array, Array]:
        assert self.basis is not None
        phi = self.basis.vectors
        r = phi.shape[1]
        core = phi.T @ self.precision_prior @ phi
        cov = np.zeros((3 * r, 3 * r))
        cov[:r, :r] = np.linalg.inv(theta.lam * core)
        cov[r : 2 * r, r : 2 * r] = sigma_v**2 * np.eye(r)
        cov[2 * r :, 2 * r :] = sigma_a**2 * np.eye(r)
        return np.zeros(3 * r), cov

    def _run(self, data: SensorData, theta: Hyperparameters, sigma_v: float, sigma_a: float) -> FilterResult:
        assert self.basis is not None
        grid = build_union_grid({t: data.streams[t][0] for t in data.types() if t in self.operators})
        model = self._model(theta)
        core = self.basis.vectors.T @ self.precision_prior @ self.basis.vectors
        mean0, cov0 = self._initial_moments(theta, sigma_v, sigma_a)
        result = run_filter(
            model, grid, self._reduced_blocks(theta), data, core, theta.lam, mean0, cov0, self.gate_alpha
        )
        self.grid, self.model = grid, model
        return result

    def negative_log_likelihood(self, data: SensorData, theta: Hyperparameters, sigma_v: float = 1.0, sigma_a: float = 10.0) -> float:
        """Negative log marginal likelihood of ``theta`` on ``data`` (Eq. loglik).

        The innovation gate is disabled while the likelihood is evaluated: rejected channels
        would otherwise drop out of the likelihood and reward an over-confident model.
        """
        gate, self.gate_alpha = self.gate_alpha, 0.0
        try:
            return -self._run(data, theta, sigma_v, sigma_a).log_likelihood
        finally:
            self.gate_alpha = gate

    def initial_hyperparameters(self, data: SensorData, sensorless_weight: float = 1e-4) -> Hyperparameters:
        """Heuristic starting point of the hyperparameter search.

        The prior scale follows the classical small weight of sensorless elements: the
        pseudo-measurement precision per sampling interval, ``lambda * dt``, is set to
        ``sensorless_weight`` times the mean precision of the strain rows (or of all rows when
        no strain gauge is present). The jerk intensity is set from the accelerometer variance
        per sampling interval; noise scales start at one.
        """
        dts = [float(np.median(np.diff(t))) for t, _ in data.streams.values()]
        dt = float(np.median(dts))
        op = self.operators.get(SensorType.STRAIN) or next(iter(self.operators.values()))
        lam0 = sensorless_weight * float(np.mean(1.0 / op.sigma**2)) / dt
        q0 = 1.0
        if SensorType.ACCELERATION in data.streams:
            t, a = data.streams[SensorType.ACCELERATION]
            q0 = float(np.nanvar(a)) / float(np.median(np.diff(t)))
        return Hyperparameters(q=q0, lam=lam0, rho={t: 1.0 for t in self.types})

    # ---- Algorithm 1 -----------------------------------------------------------------------
    def fit(
        self,
        data: SensorData,
        theta: Hyperparameters | None = None,
        training_window: tuple[float, float] | None = None,
        smooth: bool = True,
        sigma_v: float = 1.0,
        sigma_a: float = 10.0,
        max_iter: int = 200,
    ) -> DynamicMultiSensorIFEM:
        """Run Algorithm 1 on ``data``.

        Parameters
        ----------
        data:
            Measured streams.
        theta:
            Hyperparameters; estimated on ``training_window`` (or the whole record) if ``None``.
        training_window:
            ``(t_start, t_end)`` of the window used for the marginal likelihood.
        smooth:
            Whether to run the backward pass.
        sigma_v, sigma_a:
            Initial standard deviations of the velocity and acceleration blocks.
        max_iter:
            Iteration budget of the hyperparameter search.
        """
        initial = theta if theta is not None else self.initial_hyperparameters(data)
        self.form_basis(initial.lam)
        if theta is None:
            window = data if training_window is None else data.window(*training_window)
            theta = estimate_hyperparameters(
                lambda th: self.negative_log_likelihood(window, th, sigma_v, sigma_a),
                initial,
                self.types,
                max_iter=max_iter,
                q_span=1e3,
                lam_span=1e2,
                rho_span=1e1,
            )
            self.form_basis(theta.lam)
            logger.info("hyperparameters: q = %.3g, lambda = %.3g, rho = %s", theta.q, theta.lam, theta.rho)
        self.hyperparameters = theta
        self.filter_result = self._run(data, theta, sigma_v, sigma_a)
        self.smoother_result = rts_smoother(self.filter_result, self.model) if smooth else None
        return self

    # ---- outputs ---------------------------------------------------------------------------
    def _moments(self, smoothed: bool) -> tuple[Array, Array]:
        if smoothed:
            if self.smoother_result is None:
                raise RuntimeError("smoother has not been run")
            return self.smoother_result.mean, self.smoother_result.cov
        if self.filter_result is None:
            raise RuntimeError("filter has not been run")
        return self.filter_result.mean_filt, self.filter_result.cov_filt

    def reduced_block(self, block: int, smoothed: bool = True) -> tuple[Array, Array]:
        """Reduced mean ``(N, r)`` and covariance ``(N, r, r)`` of a state block."""
        assert self.basis is not None
        r = self.basis.r
        mean, cov = self._moments(smoothed)
        sl = slice(block * r, (block + 1) * r)
        return mean[:, sl], cov[:, sl, sl]

    def displacement(self, smoothed: bool = True) -> tuple[Array, Array]:
        """Free-dof displacement mean ``(N, n)`` and standard deviation ``(N, n)``."""
        assert self.basis is not None and self.hyperparameters is not None
        mean_q, cov_qq = self.reduced_block(0, smoothed)
        rows = np.eye(self.mesh.n_free)
        comp = complement_covariance(self.basis.vectors, self.precision_prior, self.hyperparameters.lam)
        return posterior_linear(rows, self.basis.vectors, mean_q, cov_qq, comp)

    def acceleration(self, smoothed: bool = True) -> tuple[Array, Array]:
        """Free-dof acceleration mean and standard deviation."""
        assert self.basis is not None
        mean_q, cov_qq = self.reduced_block(2, smoothed)
        return posterior_linear(np.eye(self.mesh.n_free), self.basis.vectors, mean_q, cov_qq)

    def strain_at(
        self, points: Sequence[tuple[int, float, float, float]], smoothed: bool = True
    ) -> tuple[Array, Array]:
        """Posterior fibre strain at ``(element, xi, y, z)`` points: mean and std ``(N, n_pts)``."""
        assert self.basis is not None and self.hyperparameters is not None
        rows = strain_rows(self.mesh, points)
        mean_q, cov_qq = self.reduced_block(0, smoothed)
        comp = complement_covariance(self.basis.vectors, self.precision_prior, self.hyperparameters.lam)
        return posterior_linear(rows, self.basis.vectors, mean_q, cov_qq, comp)

    def sample_strain_trajectories(
        self, points: Sequence[tuple[int, float, float, float]], n_samples: int, rng: np.random.Generator
    ) -> Array:
        """Sample strain histories ``(n_samples, N, n_pts)`` from the joint smoothing distribution."""
        if self.filter_result is None or self.smoother_result is None or self.model is None:
            raise RuntimeError("filter and smoother must be run first")
        assert self.basis is not None
        draws = sample_trajectories(self.filter_result, self.smoother_result, self.model, n_samples, rng)
        reduced = strain_rows(self.mesh, points) @ self.basis.vectors
        return draws[:, :, : self.basis.r] @ reduced.T
