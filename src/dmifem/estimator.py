"""Orchestration of Algorithm 1: offline assembly, hyperparameters, filter and smoother.

:class:`DynamicMultiSensorIFEM` works in the dimensionless coordinates of Eq. (scaling). It
assembles the observation operators and the prior of Eq. (prior) from the mesh and the sensors,
forms the K1-orthonormal basis of Eq. (basis), estimates the hyperparameters on a training window
by maximizing the exact marginal likelihood of Eq. (exactll), runs the filter over the union grid
together with the scalar chain of Proposition reduced and, optionally, both backward passes. The
reconstructed fields are returned in the units of the data, with the covariance of Eq. (fieldcov):
the reduced posterior plus the complement directions that no sensor sees.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from dmifem.basis import ReducedBasis, enrich_basis, k_orthonormal_basis
from dmifem.chain import (
    ChainFilterResult,
    ChainSmootherResult,
    chain_filter,
    chain_smoother,
    sample_chain,
    stationary_chain_covariance,
)
from dmifem.derived import (
    complement_output_covariance,
    complement_variance,
    posterior_reduced,
    reduced_rows,
    strain_rows,
)
from dmifem.derived import measurable_fraction as _measurable_fraction
from dmifem.elements.base import Array
from dmifem.filter import FilterResult, ReducedObservation, run_filter
from dmifem.grid import UnionGrid, build_union_grid
from dmifem.hyperparameters import Hyperparameters, estimate_hyperparameters
from dmifem.kinematics import ConstantAccelerationModel, covariance_root
from dmifem.mesh import Mesh
from dmifem.observation import (
    ObservationOperator,
    acceleration_operator,
    displacement_operator,
    inclination_operator,
    strain_operator,
)
from dmifem.prior import (
    BeamSection,
    PriorOperator,
    SpatialPrior,
    assemble_prior,
    spatial_prior,
    structure_length,
)
from dmifem.scaling import Scaling, default_length, default_time_scale, dof_scale
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

PROCESS_MODELS = ("structured", "isotropic")
OBJECTIVES = ("exact", "physical")
DISPLACEMENT_BLOCK, VELOCITY_BLOCK, ACCELERATION_BLOCK = 0, 1, 2

Point = tuple[int, float, float, float]

__all__ = ["DynamicMultiSensorIFEM", "Hyperparameters"]


class DynamicMultiSensorIFEM:
    """Dynamic multi-sensor inverse finite element estimator.

    Parameters
    ----------
    mesh:
        Inverse mesh with boundary conditions applied.
    gauges, accelerometers, inclinometers, displacement_sensors:
        Sensors of the four types.
    tau:
        Eigenvalue truncation level of Eq. (basis), between the eigenvalues of the independent
        sensor rows and the round-off level.
    n_enrichment:
        Number of smoothest admissible fields appended to the basis (``0`` by default); they leave
        Eq. (fieldcov) unchanged.
    tilt_coupling:
        Whether the first-order gravity tilt term of the accelerometers is retained.
    gate_alpha:
        Level of the per-channel innovation gate (``0`` disables it).
    process_model:
        ``"structured"`` for ``Sigma_j = q K1^{-1}`` (Eq. sigmaj; the reduction is exact) or
        ``"isotropic"`` for ``Sigma_j = q I`` (a Galerkin approximation, kept for comparison). The
        isotropic model has no independent complement, so its outputs and samples carry the
        reduced posterior alone.
    correlated_prior:
        Matern-3/2 prior of Eq. (prior); ``False`` gives uncorrelated section strains.
    correlation_length:
        ``ell_c`` in the length unit of the mesh; the length of the structure by default. It is not
        estimated.
    sections:
        One :class:`dmifem.prior.BeamSection` per element for the section factors ``G``; unit
        factors (a prior on the section strains) when omitted.
    free_ends:
        Nodes at which the bending moments vanish; free and pinned ends are detected when omitted,
        and ``()`` disables the conditioning.
    scaling:
        Reference length and time of Eq. (scaling). By default the length follows
        :func:`dmifem.scaling.default_length` (the length of the structure when neither sections
        nor gauge offsets give one) and the time is set from the data of the first call to
        :meth:`prepare` or :meth:`fit`; it is kept afterwards, so hyperparameters stay bound to
        one scaling. Construct a new estimator to change it.
    """

    def __init__(
        self,
        mesh: Mesh,
        gauges: Sequence[StrainGauge] = (),
        accelerometers: Sequence[Accelerometer] = (),
        inclinometers: Sequence[Inclinometer] = (),
        displacement_sensors: Sequence[DisplacementSensor] = (),
        tau: float = 1e-10,
        n_enrichment: int = 0,
        tilt_coupling: bool = False,
        gate_alpha: float = 1e-3,
        process_model: str = "structured",
        correlated_prior: bool = True,
        correlation_length: float | None = None,
        sections: Sequence[BeamSection] | None = None,
        free_ends: Sequence[int] | None = None,
        scaling: Scaling | None = None,
    ) -> None:
        if process_model not in PROCESS_MODELS:
            raise ValueError(f"process_model must be one of {PROCESS_MODELS}")
        self.mesh = mesh
        self.tau = tau
        self.n_enrichment = n_enrichment
        self.gate_alpha = gate_alpha
        self.process_model = process_model
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
        self.length = (
            scaling.length
            if scaling is not None
            else default_length(gauges, sections, fallback=structure_length(mesh))
        )
        self.scaling: Scaling | None = scaling
        self.dof_scale: Array = dof_scale(mesh, self.length)
        self.prior_operator: PriorOperator = assemble_prior(mesh, length=self.length, sections=sections)
        self.spatial_prior: SpatialPrior = spatial_prior(
            mesh,
            self.prior_operator,
            correlated=correlated_prior,
            correlation_length=correlation_length,
            free_ends=free_ends,
        )
        self.basis: ReducedBasis | None = None
        self.hyperparameters: Hyperparameters | None = None
        self.grid: UnionGrid | None = None
        self.model: ConstantAccelerationModel | None = None
        self.filter_result: FilterResult | None = None
        self.smoother_result: SmootherResult | None = None
        self.chain_result: ChainFilterResult | None = None
        self.chain_smoothed: ChainSmootherResult | None = None

    # ---- coordinates -----------------------------------------------------------------------
    @property
    def types(self) -> list[SensorType]:
        return list(self.operators)

    @property
    def precision_prior(self) -> Array:
        """Prior precision ``K1`` on the dimensionless free dofs."""
        return self.spatial_prior.precision

    @property
    def time_scale(self) -> float:
        """Reference time ``T`` in the time unit of the data."""
        if self.scaling is None:
            raise RuntimeError("the reference time is set by prepare() or fit()")
        return self.scaling.time

    def _basis(self) -> ReducedBasis:
        if self.basis is None:
            raise RuntimeError("the basis is formed by prepare() or fit()")
        return self.basis

    def _reduce(self, rows_bar: Array) -> Array:
        """``rows_bar @ Phi`` for dimensionless rows, formed in the prior-whitened coordinates."""
        return reduced_rows(rows_bar, self._basis(), self.spatial_prior)

    def _own_streams(self, data: SensorData) -> SensorData:
        return SensorData({t: data.streams[t] for t in data.types() if t in self.operators})

    def _whitened(self, op: ObservationOperator) -> tuple[Array, Array | None]:
        """Whitened rows of one sensor type on its dimensionless block, and its tilt rows."""
        factor = self.time_scale**2 if op.block == "acceleration" else 1.0
        rows = op.rows * self.dof_scale[None, :] / (factor * op.sigma[:, None])
        tilt = None
        if op.tilt_rows is not None:
            tilt = op.tilt_rows * self.dof_scale[None, :] / op.sigma[:, None]
        return rows, tilt

    def dimensionless_rows(self, rows: Array, block: int = DISPLACEMENT_BLOCK) -> Array:
        """Functionals on state block ``block`` for physical rows acting on ``d^b u / dt^b``."""
        return np.atleast_2d(np.asarray(rows, dtype=float)) * self.dof_scale[None, :] / self.time_scale**block

    def sensor_rows(self) -> Array:
        """Whitened dimensionless rows of all channels on the field, tilt rows included (span ``W``)."""
        stacked = []
        for op in self.operators.values():
            rows, tilt = self._whitened(op)
            stacked.append(rows)
            if tilt is not None:
                stacked.append(tilt)
        return np.vstack(stacked)

    def union_grid(self, data: SensorData) -> UnionGrid:
        """Union grid of the streams of the estimator's sensor types, in dimensionless time."""
        t_ref = self.time_scale
        return build_union_grid(
            {t: data.streams[t][0] / t_ref for t in data.types() if t in self.operators}
        )

    # ---- offline stage ---------------------------------------------------------------------
    def form_basis(self) -> ReducedBasis:
        """Form the K1-orthonormal basis of Eq. (basis), enriched if requested."""
        rows = self.sensor_rows()
        basis = k_orthonormal_basis(rows, np.ones(rows.shape[0]), self.spatial_prior, self.tau)
        basis = enrich_basis(basis, self.spatial_prior, self.n_enrichment)
        self.basis = basis
        logger.info("basis with r = %d (%d informed, tau = %g)", basis.r, basis.n_informed, self.tau)
        return basis

    def prepare(self, data: SensorData) -> ReducedBasis:
        """Set the reference time from ``data`` if it is not set yet, and form the basis.

        Results of an earlier :meth:`fit` are discarded.
        """
        if self.scaling is None:
            self.scaling = Scaling(self.length, default_time_scale(self._own_streams(data)))
            logger.info("reference time T = %.4g (f_ref = %.4g)", self.scaling.time, self.scaling.reference_frequency)
        self._clear_results()
        return self.form_basis()

    def _clear_results(self) -> None:
        self.grid = None
        self.model = None
        self.filter_result = None
        self.smoother_result = None
        self.chain_result = None
        self.chain_smoothed = None

    def observation_blocks(
        self, theta: Hyperparameters, vectors: Array | None = None
    ) -> dict[SensorType, ReducedObservation]:
        """Whitened observation blocks on the state built from ``vectors`` (``Phi`` by default).

        With ``vectors`` the field is ``u_bar = vectors @ coordinates``; the full nodal model uses
        the identity, or ``R^{-1}`` for the prior-whitened coordinates.
        """
        if vectors is None:
            r = self._basis().r
            project = self._reduce
        else:
            basis_vectors = np.asarray(vectors, dtype=float)
            r = basis_vectors.shape[1]

            def project(rows_bar: Array) -> Array:
                return rows_bar @ basis_vectors

        blocks: dict[SensorType, ReducedObservation] = {}
        for stype, op in self.operators.items():
            rows_w, tilt_w = self._whitened(op)
            rows = np.zeros((op.n_channels, 3 * r))
            if op.block == "acceleration":
                rows[:, 2 * r :] = project(rows_w)
                if tilt_w is not None:
                    rows[:, :r] = project(tilt_w)
            else:
                rows[:, :r] = project(rows_w)
            variance = theta.rho.get(stype, 1.0) * np.ones(op.n_channels)
            blocks[stype] = ReducedObservation(rows, op.offsets / op.sigma, variance, scale=op.sigma)
        return blocks

    def _model(self, theta: Hyperparameters) -> ConstantAccelerationModel:
        basis = self._basis()
        if self.process_model == "structured":
            return ConstantAccelerationModel(theta.q * np.eye(basis.r))
        u = basis.whitened if basis.whitened is not None else self.spatial_prior.factor @ basis.vectors
        k1_phi = self.spatial_prior.factor.T @ u  # K1 Phi = R^T R Phi
        return ConstantAccelerationModel(theta.q * (k1_phi.T @ k1_phi))

    def _run(
        self, data: SensorData, theta: Hyperparameters, gate_alpha: float
    ) -> tuple[UnionGrid, ConstantAccelerationModel, ChainFilterResult, FilterResult]:
        r = self._basis().r
        grid = self.union_grid(data)
        dts = grid.intervals()
        dt_ref = float(np.median(dts)) if dts.size else 1.0
        model = self._model(theta)
        p1 = stationary_chain_covariance(theta.q, theta.lam, dt_ref)
        chain = chain_filter(theta.q, theta.lam, dts, p1, grid.n)
        prior_density = r * chain.log_density if self.process_model == "structured" else None
        result = run_filter(
            model,
            grid,
            self.observation_blocks(theta),
            data,
            None,
            theta.lam,
            np.zeros(3 * r),
            np.kron(p1, np.eye(r)),
            gate_alpha,
            prior_log_density=prior_density,
        )
        return grid, model, chain, result

    def negative_log_likelihood(
        self, data: SensorData, theta: Hyperparameters, objective: str = "exact"
    ) -> float:
        """Negative log marginal likelihood of ``theta`` on ``data``.

        ``objective="exact"`` is Eq. (exactll); ``"physical"`` is the sum over the physical
        innovations alone. The innovation gate is inactive: rejected channels would otherwise drop
        out of the likelihood and reward an over-confident model.
        """
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        _, _, _, result = self._run(data, theta, gate_alpha=0.0)
        if objective == "exact":
            return -result.log_likelihood
        return -result.log_likelihood_physical

    def initial_hyperparameters(self, data: SensorData, sensorless_weight: float = 1e-4) -> Hyperparameters:
        """Starting point of the hyperparameter search, in dimensionless units.

        The prior scale follows the classical small weight of sensorless elements: ``lambda dt``
        is ``sensorless_weight`` times the mean precision per sample that the displacement-type
        channels (the accelerometers when there are none) supply to a reduced coordinate. The jerk
        intensity ``q = 1 / lambda`` places the corner of the prior spectrum,
        ``(q lambda)^{1/6}``, at ``f_ref``. Noise scales start at one.
        """
        if self.basis is None:
            self.prepare(data)
        r = self._basis().r
        dts = self.union_grid(data).intervals()
        dt = float(np.median(dts)) if dts.size else 1.0
        info = np.zeros((r, r))
        for op in self.operators.values():
            if op.block == "displacement":
                reduced = self._reduce(self._whitened(op)[0])
                info += reduced.T @ reduced
        if not np.any(info):
            for op in self.operators.values():
                reduced = self._reduce(self._whitened(op)[0])
                info += reduced.T @ reduced
        lam0 = sensorless_weight * float(np.trace(info)) / r / dt
        return Hyperparameters(q=1.0 / lam0, lam=lam0, rho={t: 1.0 for t in self.types})

    # ---- Algorithm 1 -----------------------------------------------------------------------
    def fit(
        self,
        data: SensorData,
        theta: Hyperparameters | None = None,
        training_window: tuple[float, float] | None = None,
        smooth: bool = True,
        max_iter: int = 200,
        objective: str = "exact",
    ) -> DynamicMultiSensorIFEM:
        """Run Algorithm 1 on ``data``.

        Parameters
        ----------
        data:
            Measured streams.
        theta:
            Dimensionless hyperparameters; estimated on ``training_window`` (or the whole record)
            when ``None``.
        training_window:
            ``(t_start, t_end)`` of the window used for the marginal likelihood, in data time.
        smooth:
            Whether to run the backward passes.
        max_iter:
            Iteration budget of the hyperparameter search.
        objective:
            Likelihood maximized by the search (see :meth:`negative_log_likelihood`).
        """
        self.prepare(data)
        if theta is None:
            window = data if training_window is None else data.window(*training_window)
            initial = self.initial_hyperparameters(window)
            theta = estimate_hyperparameters(
                lambda th: self.negative_log_likelihood(window, th, objective),
                initial,
                self.types,
                max_iter=max_iter,
                q_span=1e3,
                lam_span=1e2,
                rho_span=1e1,
            )
            logger.info("hyperparameters: q = %.3g, lambda = %.3g, rho = %s", theta.q, theta.lam, theta.rho)
        self.hyperparameters = theta
        grid, model, chain, result = self._run(data, theta, self.gate_alpha)
        self.grid, self.model, self.chain_result, self.filter_result = grid, model, chain, result
        if smooth:
            self.smoother_result = rts_smoother(result, model)
            self.chain_smoothed = chain_smoother(chain, theta.q, grid.intervals())
        else:
            self.smoother_result = None
            self.chain_smoothed = None
        return self

    # ---- outputs ---------------------------------------------------------------------------
    @property
    def times(self) -> Array:
        """Instants of the union grid in data time."""
        if self.grid is None:
            raise RuntimeError("fit() has not been run")
        return self.grid.times * self.time_scale

    def _moments(self, smoothed: bool) -> tuple[Array, Array]:
        if smoothed:
            if self.smoother_result is None:
                raise RuntimeError("smoother has not been run")
            return self.smoother_result.mean, self.smoother_result.cov
        if self.filter_result is None:
            raise RuntimeError("filter has not been run")
        return self.filter_result.mean_filt, self.filter_result.cov_filt

    def _chain_covariance(self, smoothed: bool) -> Array:
        if smoothed:
            if self.chain_smoothed is None:
                raise RuntimeError("smoother has not been run")
            return self.chain_smoothed.cov
        if self.chain_result is None:
            raise RuntimeError("filter has not been run")
        return self.chain_result.cov_filt

    def reduced_block(self, block: int, smoothed: bool = True) -> tuple[Array, Array]:
        """Reduced mean ``(N, r)`` and covariance ``(N, r, r)`` of a state block."""
        r = self._basis().r
        mean, cov = self._moments(smoothed)
        sl = slice(block * r, (block + 1) * r)
        return mean[:, sl], cov[:, sl, sl]

    def linear_output(
        self, rows: Array, block: int = DISPLACEMENT_BLOCK, smoothed: bool = True
    ) -> tuple[Array, Array]:
        """Posterior mean and standard deviation ``(N, n_rows)`` of ``rows @ d^b u / dt^b``.

        ``rows`` act on the physical free dofs; for the structured model the variance includes the
        complement of Eq. (fieldcov).
        """
        rows_bar = self.dimensionless_rows(rows, block)
        mean_q, cov_qq = self.reduced_block(block, smoothed)
        if self.process_model != "structured":
            return posterior_reduced(self._reduce(rows_bar), mean_q, cov_qq)
        comp = complement_variance(rows_bar, self._basis(), self.spatial_prior)
        chain_var = self._chain_covariance(smoothed)[:, block, block]
        return posterior_reduced(self._reduce(rows_bar), mean_q, cov_qq, comp, chain_var)

    def displacement(self, smoothed: bool = True) -> tuple[Array, Array]:
        """Free-dof displacement mean ``(N, n)`` and standard deviation ``(N, n)``."""
        return self.linear_output(np.eye(self.mesh.n_free), DISPLACEMENT_BLOCK, smoothed)

    def velocity(self, smoothed: bool = True) -> tuple[Array, Array]:
        """Free-dof velocity mean and standard deviation."""
        return self.linear_output(np.eye(self.mesh.n_free), VELOCITY_BLOCK, smoothed)

    def acceleration(self, smoothed: bool = True) -> tuple[Array, Array]:
        """Free-dof acceleration mean and standard deviation."""
        return self.linear_output(np.eye(self.mesh.n_free), ACCELERATION_BLOCK, smoothed)

    def strain_at(self, points: Sequence[Point], smoothed: bool = True) -> tuple[Array, Array]:
        """Posterior fibre strain at ``(element, xi, y, z)`` points: mean and std ``(N, n_pts)``."""
        return self.linear_output(strain_rows(self.mesh, points), DISPLACEMENT_BLOCK, smoothed)

    def measurable_fraction(self, points: Sequence[Point]) -> Array:
        """Measurable fraction of Eq. (vr) of the fibre strain at each point."""
        rows_bar = self.dimensionless_rows(strain_rows(self.mesh, points))
        return _measurable_fraction(rows_bar, self._basis(), self.spatial_prior)

    def sample_strain_trajectories(
        self,
        points: Sequence[Point],
        n_samples: int,
        rng: np.random.Generator,
        include_complement: bool = True,
    ) -> Array:
        """Sample strain histories ``(n_samples, N, n_pts)`` from the joint smoothing distribution.

        Reduced trajectories come from backward sampling; each is completed by an independent
        complement trajectory ``(B Sigma_perp B^T)^{1/2} pi`` with ``pi`` smoothing samples of the
        scalar chain, which reproduces Eq. (fieldcov) across space and time. The complement is
        added for the structured model only.
        """
        if (
            self.filter_result is None
            or self.smoother_result is None
            or self.model is None
            or self.chain_result is None
            or self.chain_smoothed is None
        ):
            raise RuntimeError("filter and smoother must be run first")
        basis = self._basis()
        rows_bar = self.dimensionless_rows(strain_rows(self.mesh, points))
        reduced = self._reduce(rows_bar)
        draws = sample_trajectories(self.filter_result, self.smoother_result, self.model, n_samples, rng)
        out = draws[:, :, : basis.r] @ reduced.T
        if include_complement and self.process_model == "structured":
            cov_out = complement_output_covariance(rows_bar, basis, self.spatial_prior)
            root = covariance_root(cov_out)
            n_pts = rows_bar.shape[0]
            paths = sample_chain(self.chain_result, self.chain_smoothed, n_samples * n_pts, rng)
            paths = paths[:, :, 0].reshape(n_samples, n_pts, -1)
            out = out + np.einsum("ij,sjk->ski", root, paths)
        return out
