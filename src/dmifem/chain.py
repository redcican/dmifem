"""Scalar kinematic chain observed by its pseudo-measurement alone (Proposition reduced, Eq. exactll).

The directions of the field that no sensor sees, and the prior-only density in the exact marginal
likelihood, both reduce to copies of one three-state chain: white-noise jerk of intensity ``q``,
observed at every instant by ``0 = pi_k + eta_k`` with ``eta_k ~ N(0, 1 / (lambda dt_k))``. Its means
are zero, so only covariances are propagated: ``Pi_{k|k}`` and ``Pi_{k|N}`` scale the complement
covariance of Eq. (fieldcov), and the log density of the zero pseudo-measurements enters
Eq. (exactll) ``r`` times. The recursion starts from ``P_bar_1``, the stationary predicted covariance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are

from dmifem.elements.base import Array
from dmifem.kinematics import covariance_root, process_noise_core, transition_core

LOG_2PI = float(np.log(2.0 * np.pi))
_H = np.array([1.0, 0.0, 0.0])


def pseudo_strengths(lam: float, dts: Array, n: int) -> Array:
    """Precision ``lambda dt_k`` of the pseudo-measurement at each of the ``n`` instants.

    The interval before an instant is used, and the first interval at the first instant.
    """
    dts = np.asarray(dts, dtype=float).reshape(-1)
    if n <= 0:
        return np.zeros(0)
    if dts.size != n - 1:
        raise ValueError("expected n - 1 intervals")
    first = dts[0] if dts.size else 1.0
    return float(lam) * np.concatenate([[first], dts])


def _riccati_step(p: Array, f: Array, qm: Array, r: float) -> Array:
    s = p[0, 0] + r
    filtered = p - np.outer(p[:, 0], p[0, :]) / s
    nxt = f @ filtered @ f.T + qm
    return 0.5 * (nxt + nxt.T)


def _is_fixed_point(p: Array, f: Array, qm: Array, r: float, rtol: float) -> bool:
    if not np.all(np.isfinite(p)) or np.min(np.linalg.eigvalsh(p)) <= 0.0:
        return False
    return float(np.max(np.abs(_riccati_step(p, f, qm, r) - p))) <= rtol * float(np.max(np.abs(p)))


def stationary_chain_covariance(
    q: float, lam: float, dt: float, rtol: float = 1e-10, max_iter: int = 200_000
) -> Array:
    """Stationary predicted covariance ``P_bar_1`` of the chain at a constant interval ``dt``.

    Solved as a discrete algebraic Riccati equation, with a fixed-point iteration as fallback.
    """
    if not (q > 0.0 and lam > 0.0 and dt > 0.0):
        raise ValueError("q, lambda and dt must be positive")
    f = transition_core(dt)
    qm = q * process_noise_core(dt)
    r = 1.0 / (lam * dt)
    try:
        p = solve_discrete_are(f.T, _H[:, None], qm, np.array([[r]]))
        p = 0.5 * (p + p.T)
    except (np.linalg.LinAlgError, ValueError):
        p = np.eye(3)
    if _is_fixed_point(p, f, qm, r, rtol):
        return p
    if not np.all(np.isfinite(p)):
        p = np.eye(3)
    for _ in range(max_iter):
        nxt = _riccati_step(p, f, qm, r)
        done = float(np.max(np.abs(nxt - p))) <= rtol * float(np.max(np.abs(nxt)))
        p = nxt
        if done:
            break
    return p


@dataclass(frozen=True)
class ChainFilterResult:
    """Predicted and filtered ``3 x 3`` covariances, innovation variances and ``log p(z = 0)``."""

    cov_pred: Array
    cov_filt: Array
    innovation_variance: Array
    log_density: float

    @property
    def n(self) -> int:
        return self.cov_filt.shape[0]


@dataclass(frozen=True)
class ChainSmootherResult:
    """Smoothed covariances ``Pi_{k|N}`` and the smoother gains of Eq. (rts)."""

    cov: Array
    gains: Array


def _chain_step(q: float, dt: float, cache: dict[float, tuple[Array, Array]]) -> tuple[Array, Array]:
    key = float(dt)
    if key not in cache:
        cache[key] = (transition_core(dt), q * process_noise_core(dt))
    return cache[key]


def chain_filter(q: float, lam: float, dts: Array, cov0: Array, n: int | None = None) -> ChainFilterResult:
    """Run the chain over ``n`` instants separated by ``dts`` from the covariance ``cov0`` at ``t_1``.

    The pseudo-measurement update is in Joseph form, as in :func:`dmifem.filter.run_filter`.
    """
    dts = np.asarray(dts, dtype=float).reshape(-1)
    n = dts.size + 1 if n is None else int(n)
    strengths = pseudo_strengths(lam, dts, n)
    pred = np.zeros((n, 3, 3))
    filt = np.zeros((n, 3, 3))
    s_all = np.zeros(n)
    p = np.asarray(cov0, dtype=float).copy()
    log_density = 0.0
    cache: dict[float, tuple[Array, Array]] = {}
    for k in range(n):
        if k > 0:
            f, qm = _chain_step(q, dts[k - 1], cache)
            p = f @ p @ f.T + qm
            p = 0.5 * (p + p.T)
        pred[k] = p
        if strengths[k] > 0.0:
            var = 1.0 / strengths[k]
            s = p[0, 0] + var
            gain = p[:, 0] / s
            i_kh = np.eye(3) - np.outer(gain, _H)
            p = i_kh @ p @ i_kh.T + var * np.outer(gain, gain)
            p = 0.5 * (p + p.T)
            s_all[k] = s
            log_density += -0.5 * (np.log(s) + LOG_2PI)
        filt[k] = p
    return ChainFilterResult(pred, filt, s_all, float(log_density))


def chain_smoother(result: ChainFilterResult, q: float, dts: Array) -> ChainSmootherResult:
    """Backward pass of Eq. (rts) for the chain: ``Pi_{k|N}``."""
    dts = np.asarray(dts, dtype=float).reshape(-1)
    n = result.n
    cov = result.cov_filt.copy()
    gains = np.zeros((n, 3, 3))
    cache: dict[float, tuple[Array, Array]] = {}
    for k in range(n - 2, -1, -1):
        f, _ = _chain_step(q, dts[k], cache)
        gain = np.linalg.solve(result.cov_pred[k + 1], f @ result.cov_filt[k]).T
        gains[k] = gain
        cov[k] = result.cov_filt[k] + gain @ (cov[k + 1] - result.cov_pred[k + 1]) @ gain.T
        cov[k] = 0.5 * (cov[k] + cov[k].T)
    return ChainSmootherResult(cov, gains)


def sample_chain(
    result: ChainFilterResult, smoothed: ChainSmootherResult, n_paths: int, rng: np.random.Generator
) -> Array:
    """Draw ``n_paths`` independent chain trajectories from the joint smoothing distribution.

    Returns an array of shape ``(n_paths, N, 3)``; entry ``0`` of the last axis is the
    displacement-type entry ``pi_k`` used for the complement trajectories.
    """
    n = result.n
    draws = np.zeros((n_paths, n, 3))
    root = covariance_root(result.cov_filt[n - 1])
    draws[:, n - 1] = rng.standard_normal((n_paths, 3)) @ root.T
    for k in range(n - 2, -1, -1):
        gain = smoothed.gains[k]
        cond = result.cov_filt[k] - gain @ result.cov_pred[k + 1] @ gain.T
        root = covariance_root(cond)
        draws[:, k] = draws[:, k + 1] @ gain.T + rng.standard_normal((n_paths, 3)) @ root.T
    return draws
