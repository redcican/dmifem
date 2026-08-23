"""Kalman filter of the reduced model (manuscript Section 3.4, Eqs. kf and gate).

The prior pseudo-measurement enters every update as a Joseph-form update of ``r`` rows (a square
root of ``Phi^T K1 Phi``) with the rate-invariant precision ``lambda dt_k``, which is equivalent to
adding ``lambda dt_k Phi^T K1 Phi`` to the precision of the displacement block. The prior is part of the state model, not of the data:
the marginal likelihood is the prediction-error decomposition over the physical channels only,
in which ``lambda`` enters through their predictive covariance. Physical channels are updated in
Joseph form after a per-channel innovation gate; channels with ``nan`` values are deleted from
the update.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

try:  # small dense matrices: multi-threaded BLAS is counter-productive
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover
    from contextlib import nullcontext

    def threadpool_limits(*_args, **_kwargs):  # type: ignore[misc]
        return nullcontext()

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
        Known offsets ``b`` per channel.
    variance:
        Noise variance per channel (nominal variance times ``rho_s``).
    """

    rows: Array
    offsets: Array
    variance: Array


@dataclass
class FilterResult:
    """Filtered and predicted moments on the union grid, with the log marginal likelihood."""

    times: Array
    mean_pred: Array
    cov_pred: Array
    mean_filt: Array
    cov_filt: Array
    log_likelihood: float
    n_rejected: int


def prior_square_root(precision_core: Array) -> Array:
    """Square root ``C`` of ``G = Phi^T K1 Phi`` with ``C^T C = G`` (rows of the prior)."""
    vals, vecs = np.linalg.eigh(precision_core)
    vals = np.clip(vals, 0.0, None)
    return (np.sqrt(vals)[:, None] * vecs.T)


def prior_information_update(
    mean: Array, cov: Array, prior_rows: Array, strength: float, r: int
) -> tuple[Array, Array]:
    """Condition on the zero pseudo-measurement ``0 = C q + eta`` with precision ``strength``.

    ``prior_rows`` is a square root ``C`` of ``Phi^T K1 Phi``; the update is the Joseph-form
    measurement update with noise covariance ``I / strength``, which adds ``strength * C^T C``
    to the precision of the displacement block and shrinks the mean accordingly.
    """
    if strength <= 0.0:
        return mean, cov
    dim = mean.size
    h = np.zeros((prior_rows.shape[0], dim))
    h[:, :r] = prior_rows
    var = np.full(prior_rows.shape[0], 1.0 / strength)
    s_mat = h @ cov @ h.T + np.diag(var)
    gain = np.linalg.solve(s_mat, h @ cov).T
    mean_new = mean - gain @ (h @ mean)
    i_kh = np.eye(dim) - gain @ h
    cov_new = i_kh @ cov @ i_kh.T + gain @ np.diag(var) @ gain.T
    return mean_new, 0.5 * (cov_new + cov_new.T)


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
        rows.append(block.rows[valid])
        values.append(y[valid])
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
    sign, log_det = np.linalg.slogdet(s)
    quad = float(innovation @ np.linalg.solve(s, innovation))
    loglik = -0.5 * (log_det + quad + innovation.size * np.log(2.0 * np.pi))
    return mean_new, cov_new, loglik, n_rejected


def run_filter(
    model: ConstantAccelerationModel,
    grid: UnionGrid,
    blocks: dict[SensorType, ReducedObservation],
    data: SensorData,
    precision_core: Array,
    lam: float,
    mean0: Array,
    cov0: Array,
    gate_alpha: float = 1e-3,
) -> FilterResult:
    """Run the recursion of Eq. (kf) over the union grid.

    Parameters
    ----------
    model:
        Reduced kinematic model (``r`` coordinates).
    grid:
        Union grid of the data streams.
    blocks:
        Reduced observation blocks per sensor type.
    data:
        Measured streams.
    precision_core:
        ``Phi^T K1 Phi`` of the prior.
    lam:
        Prior scale ``lambda`` (precision per unit time).
    mean0, cov0:
        Initial mean and covariance at ``t_1`` (before any update).
    gate_alpha:
        Level of the per-channel innovation gate; ``0`` disables the gate.
    """
    n, dim, r = grid.n, model.dim, model.r
    mean_pred = np.zeros((n, dim))
    cov_pred = np.zeros((n, dim, dim))
    mean_filt = np.zeros((n, dim))
    cov_filt = np.zeros((n, dim, dim))
    dts = grid.intervals()
    loglik, rejected = 0.0, 0
    prior_rows = prior_square_root(precision_core)
    mean, cov = np.asarray(mean0, dtype=float).copy(), np.asarray(cov0, dtype=float).copy()
    with threadpool_limits(limits=1):
        for k in range(n):
            if k > 0:
                f = model.transition(dts[k - 1])
                mean = f @ mean
                cov = f @ cov @ f.T + model.process_noise(dts[k - 1])
            mean_pred[k], cov_pred[k] = mean, cov
            strength = lam * (dts[k - 1] if k > 0 else dts[0])
            mean, cov = prior_information_update(mean, cov, prior_rows, strength, r)
            h, y, b, var = _stack_rows(grid, k, blocks, data)
            if y.size:
                mean, cov, ll_obs, n_rej = _joseph_update(mean, cov, h, y, b, var, gate_alpha)
                loglik += ll_obs
                rejected += n_rej
            mean_filt[k], cov_filt[k] = mean, cov
    return FilterResult(grid.times, mean_pred, cov_pred, mean_filt, cov_filt, loglik, rejected)
