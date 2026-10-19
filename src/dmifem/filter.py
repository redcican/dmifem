"""Kalman filter of the reduced model (manuscript Section 3.4, Eqs. joint, kf, gate and exactll).

At every instant the pseudo-measurement of Eq. (pseudo) is processed first and the physical
channels second; the order changes only the factorization of the likelihood. With the
K1-orthonormal basis the reduced pseudo-measurement is ``0 = q_k + eta_k`` with
``eta_k ~ N(0, I / (lambda dt_k))`` (identity core); a general precision core ``G`` enters through a
square root ``C`` with ``C^T C = G``. Both updates are in Joseph form.

The exact log marginal likelihood of Eq. (exactll), ``log p(Y | Y^lambda = 0)``, is the sum of the
pseudo-measurement and physical innovation terms minus ``log p(Y^lambda = 0)``, the density of the
pseudo-measurements under the prior alone. For the structured model the latter is ``r`` times that
of the scalar chain (:mod:`dmifem.chain`), passed in by the caller; otherwise it is computed by a
covariance-only recursion alongside the filter. The physical-channel sum alone is returned as well.
Physical channels pass a per-channel innovation gate; channels with ``nan`` values are deleted
from the update.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2

try:  # small dense matrices: multi-threaded BLAS is counter-productive
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover
    from contextlib import nullcontext

    def threadpool_limits(*_args, **_kwargs):  # type: ignore[misc]
        return nullcontext()

from dmifem.chain import LOG_2PI, pseudo_strengths
from dmifem.elements.base import Array
from dmifem.grid import UnionGrid
from dmifem.kinematics import ConstantAccelerationModel
from dmifem.sensors import SensorData, SensorType


@dataclass(frozen=True)
class ReducedObservation:
    """Reduced observation block of one sensor type.

    Attributes
    ----------
    rows:
        Rows on the full reduced state ``(m_s, 3r)``.
    offsets:
        Known offsets ``b`` per channel, in the units of the rows.
    variance:
        Noise variance per channel (``rho_s`` for whitened channels).
    scale:
        Divisor applied to the measured values before the update (the nominal noise standard
        deviation for whitened channels), or ``None``.
    """

    rows: Array
    offsets: Array
    variance: Array
    scale: Array | None = None


@dataclass
class FilterResult:
    """Filtered and predicted moments on the union grid with the likelihood terms.

    ``log_likelihood`` is Eq. (exactll); ``log_likelihood_physical`` is the sum over the physical
    innovations alone, ``log_likelihood_pseudo`` the sum over the pseudo-measurement innovations and
    ``log_prior_density`` the density ``log p(Y^lambda = 0)`` subtracted from their sum.
    """

    times: Array
    mean_pred: Array
    cov_pred: Array
    mean_filt: Array
    cov_filt: Array
    log_likelihood: float
    n_rejected: int
    log_likelihood_physical: float = 0.0
    log_likelihood_pseudo: float = 0.0
    log_prior_density: float = 0.0


def prior_square_root(precision_core: Array) -> Array:
    """Square root ``C`` of a precision core ``G`` with ``C^T C = G`` (rows of the prior)."""
    vals, vecs = np.linalg.eigh(precision_core)
    vals = np.clip(vals, 0.0, None)
    return np.sqrt(vals)[:, None] * vecs.T


def _pseudo_rows(prior_rows: Array | None, r: int, dim: int) -> Array:
    if prior_rows is None:
        h = np.zeros((r, dim))
        h[:, :r] = np.eye(r)
    else:
        h = np.zeros((prior_rows.shape[0], dim))
        h[:, : prior_rows.shape[1]] = prior_rows
    return h


def pseudo_measurement_update(
    mean: Array, cov: Array, strength: float, r: int, prior_rows: Array | None = None
) -> tuple[Array, Array, float]:
    """Condition on the zero pseudo-measurement ``0 = C q + eta`` with precision ``strength``.

    ``C`` is the identity (K1-orthonormal basis) when ``prior_rows`` is ``None``. Returns the updated
    mean and covariance and ``log N(nu; 0, S)`` of the pseudo-measurement innovation.
    """
    if strength <= 0.0:
        return mean, cov, 0.0
    h = _pseudo_rows(prior_rows, r, mean.size)
    var = 1.0 / strength
    innovation = -(h @ mean)
    s_mat = h @ cov @ h.T + var * np.eye(h.shape[0])
    factor = cho_factor(0.5 * (s_mat + s_mat.T))
    gain = cho_solve(factor, h @ cov).T
    mean_new = mean + gain @ innovation
    i_kh = np.eye(mean.size) - gain @ h
    cov_new = i_kh @ cov @ i_kh.T + var * (gain @ gain.T)
    log_det = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
    quad = float(innovation @ cho_solve(factor, innovation))
    loglik = -0.5 * (log_det + quad + h.shape[0] * LOG_2PI)
    return mean_new, 0.5 * (cov_new + cov_new.T), loglik


def prior_information_update(
    mean: Array, cov: Array, prior_rows: Array, strength: float, r: int
) -> tuple[Array, Array]:
    """Pseudo-measurement update with explicit rows ``C`` (the likelihood term is discarded).

    Adds ``strength * C^T C`` to the precision of the displacement block and shrinks the mean.
    """
    mean_new, cov_new, _ = pseudo_measurement_update(mean, cov, strength, r, prior_rows)
    return mean_new, cov_new


def _stack_rows(
    grid: UnionGrid, k: int, blocks: dict[SensorType, ReducedObservation], data: SensorData
) -> tuple[Array, Array, Array, Array]:
    rows, values, offsets, variances = [], [], [], []
    for stype, block in blocks.items():
        idx = grid.samples.get(stype)
        if idx is None or idx[k] < 0:
            continue
        y = data.streams[stype][1][idx[k]]
        valid = np.isfinite(y)
        if not np.any(valid):
            continue
        y_valid = y[valid] if block.scale is None else y[valid] / block.scale[valid]
        rows.append(block.rows[valid])
        values.append(y_valid)
        offsets.append(block.offsets[valid])
        variances.append(block.variance[valid])
    if not rows:
        dim = next(iter(blocks.values())).rows.shape[1]
        return np.zeros((0, dim)), np.zeros(0), np.zeros(0), np.zeros(0)
    return np.vstack(rows), np.concatenate(values), np.concatenate(offsets), np.concatenate(variances)


def _joseph_update(
    mean: Array, cov: Array, h: Array, y: Array, b: Array, var: Array, gate_alpha: float
) -> tuple[Array, Array, float, int]:
    """Per-channel gate followed by the Joseph-form update; returns the likelihood term."""
    innovation = y - b - h @ mean
    s = h @ cov @ h.T + np.diag(var)
    keep = np.ones(y.size, dtype=bool)
    if gate_alpha > 0.0 and y.size > 0:
        threshold = chi2.ppf(1.0 - gate_alpha, df=1)
        keep = innovation**2 / np.diag(s) <= threshold
    n_rejected = int(np.sum(~keep))
    if not np.any(keep):
        return mean, cov, 0.0, n_rejected
    h, innovation, var = h[keep], innovation[keep], var[keep]
    s = h @ cov @ h.T + np.diag(var)
    gain = np.linalg.solve(s, h @ cov).T
    mean_new = mean + gain @ innovation
    i_kh = np.eye(mean.size) - gain @ h
    cov_new = i_kh @ cov @ i_kh.T + gain @ np.diag(var) @ gain.T
    cov_new = 0.5 * (cov_new + cov_new.T)
    _, log_det = np.linalg.slogdet(s)
    quad = float(innovation @ np.linalg.solve(s, innovation))
    loglik = -0.5 * (log_det + quad + innovation.size * LOG_2PI)
    return mean_new, cov_new, loglik, n_rejected


def _transition(
    model: ConstantAccelerationModel, dt: float, cache: dict[float, tuple[Array, Array]]
) -> tuple[Array, Array]:
    key = float(dt)
    if key not in cache:
        cache[key] = (model.transition(dt), model.process_noise(dt))
    return cache[key]


def run_filter(
    model: ConstantAccelerationModel,
    grid: UnionGrid,
    blocks: dict[SensorType, ReducedObservation],
    data: SensorData,
    precision_core: Array | None,
    lam: float,
    mean0: Array,
    cov0: Array,
    gate_alpha: float = 1e-3,
    prior_log_density: float | None = None,
) -> FilterResult:
    """Run the recursion of Eq. (kf) over the union grid.

    Parameters
    ----------
    model:
        Kinematic model on ``r`` coordinates.
    grid:
        Union grid of the data streams (dimensionless instants).
    blocks:
        Observation blocks per sensor type.
    data:
        Measured streams.
    precision_core:
        Precision core ``G`` of the pseudo-measurement on the displacement block, or ``None`` for
        the identity of the K1-orthonormal basis.
    lam:
        Prior scale ``lambda`` (precision per unit time); ``0`` removes the pseudo-measurement.
    mean0, cov0:
        Initial mean and covariance at ``t_1`` (before any update).
    gate_alpha:
        Level of the per-channel innovation gate; ``0`` disables the gate.
    prior_log_density:
        ``log p(Y^lambda = 0)`` when known (``r`` times the scalar chain for the structured model);
        computed by a covariance-only recursion when ``None``.
    """
    n, dim, r = grid.n, model.dim, model.r
    mean_pred = np.zeros((n, dim))
    cov_pred = np.zeros((n, dim, dim))
    mean_filt = np.zeros((n, dim))
    cov_filt = np.zeros((n, dim, dim))
    dts = grid.intervals()
    strengths = pseudo_strengths(lam, dts, n)
    prior_rows = None if precision_core is None else prior_square_root(np.asarray(precision_core, dtype=float))
    track_prior = prior_log_density is None and lam > 0.0
    ll_physical = ll_pseudo = log_prior = 0.0
    rejected = 0
    mean, cov = np.asarray(mean0, dtype=float).copy(), np.asarray(cov0, dtype=float).copy()
    # prior-only recursion for log p(Y^lambda = 0): same model and pseudo-measurements, no data
    cov_z = cov.copy() if track_prior else None
    mean_z = mean.copy()
    cache: dict[float, tuple[Array, Array]] = {}
    with threadpool_limits(limits=1):
        for k in range(n):
            if k > 0:
                f, q_mat = _transition(model, dts[k - 1], cache)
                mean = f @ mean
                cov = f @ cov @ f.T + q_mat
                cov = 0.5 * (cov + cov.T)
                if cov_z is not None:
                    mean_z = f @ mean_z
                    cov_z = f @ cov_z @ f.T + q_mat
                    cov_z = 0.5 * (cov_z + cov_z.T)
            mean_pred[k], cov_pred[k] = mean, cov
            mean, cov, ll_z = pseudo_measurement_update(mean, cov, strengths[k], r, prior_rows)
            ll_pseudo += ll_z
            if cov_z is not None:
                mean_z, cov_z, ll_prior = pseudo_measurement_update(mean_z, cov_z, strengths[k], r, prior_rows)
                log_prior += ll_prior
            h, y, b, var = _stack_rows(grid, k, blocks, data)
            if y.size:
                mean, cov, ll_obs, n_rej = _joseph_update(mean, cov, h, y, b, var, gate_alpha)
                ll_physical += ll_obs
                rejected += n_rej
            mean_filt[k], cov_filt[k] = mean, cov
    if prior_log_density is not None:
        log_prior = float(prior_log_density)
    exact = ll_pseudo + ll_physical - log_prior
    return FilterResult(
        grid.times,
        mean_pred,
        cov_pred,
        mean_filt,
        cov_filt,
        float(exact),
        rejected,
        float(ll_physical),
        float(ll_pseudo),
        float(log_prior),
    )
