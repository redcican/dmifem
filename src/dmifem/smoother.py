"""Rauch–Tung–Striebel smoother and backward sampling (manuscript Sections 3.4 and 3.6).

The backward sampler draws reduced trajectories from the joint smoothing distribution; the
complement trajectories of Section 3.6 come from :func:`dmifem.chain.sample_chain`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover
    from contextlib import nullcontext

    def threadpool_limits(*_args, **_kwargs):  # type: ignore[misc]
        return nullcontext()

from dmifem.elements.base import Array
from dmifem.filter import FilterResult
from dmifem.kinematics import ConstantAccelerationModel, covariance_root


@dataclass
class SmootherResult:
    """Smoothed moments and the smoother gains ``C_k`` of Eq. (rts)."""

    times: Array
    mean: Array
    cov: Array
    gains: Array


def rts_smoother(result: FilterResult, model: ConstantAccelerationModel) -> SmootherResult:
    """Backward pass of Eq. (rts) over the filtered and predicted moments."""
    n, dim = result.mean_filt.shape
    mean = result.mean_filt.copy()
    cov = result.cov_filt.copy()
    gains = np.zeros((n, dim, dim))
    dts = np.diff(result.times)
    with threadpool_limits(limits=1):
        for k in range(n - 2, -1, -1):
            f = model.transition(dts[k])
            pred_cov = result.cov_pred[k + 1]
            gain = np.linalg.solve(pred_cov, f @ result.cov_filt[k]).T
            gains[k] = gain
            mean[k] = result.mean_filt[k] + gain @ (mean[k + 1] - result.mean_pred[k + 1])
            cov[k] = result.cov_filt[k] + gain @ (cov[k + 1] - pred_cov) @ gain.T
            cov[k] = 0.5 * (cov[k] + cov[k].T)
    return SmootherResult(result.times, mean, cov, gains)


def sample_trajectories(
    result: FilterResult,
    smoothed: SmootherResult,
    model: ConstantAccelerationModel,
    n_samples: int,
    rng: np.random.Generator,
) -> Array:
    """Draw state trajectories from the joint smoothing distribution by backward sampling.

    The conditional covariance at each instant is factored once and shared by all draws.
    Returns an array of shape ``(n_samples, N, 3r)``.
    """
    n, dim = result.mean_filt.shape
    draws = np.zeros((n_samples, n, dim))
    root = covariance_root(result.cov_filt[n - 1])
    draws[:, n - 1] = result.mean_filt[n - 1] + rng.standard_normal((n_samples, dim)) @ root.T
    for k in range(n - 2, -1, -1):
        gain = smoothed.gains[k]
        cond_cov = result.cov_filt[k] - gain @ result.cov_pred[k + 1] @ gain.T
        root = covariance_root(cond_cov)
        cond_mean = result.mean_filt[k] + (draws[:, k + 1] - result.mean_pred[k + 1]) @ gain.T
        draws[:, k] = cond_mean + rng.standard_normal((n_samples, dim)) @ root.T
    return draws
