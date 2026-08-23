"""Marginal-likelihood estimation of the hyperparameters (manuscript Section 3.5).

``theta = {q, lambda, rho_s}`` is estimated on a training window by maximizing the log
marginal likelihood that the filter provides through the prediction-error decomposition, over
the physical channels and the prior rows. The search is derivative-free in ``log theta``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from dmifem.sensors import SensorType


@dataclass(frozen=True)
class Hyperparameters:
    """Jerk intensity ``q``, prior scale ``lambda`` and noise scales ``rho`` per sensor type."""

    q: float
    lam: float
    rho: dict[SensorType, float] = field(default_factory=dict)

    def to_log_vector(self, types: list[SensorType]) -> np.ndarray:
        return np.log(np.array([self.q, self.lam] + [self.rho.get(t, 1.0) for t in types]))

    @classmethod
    def from_log_vector(cls, vec: np.ndarray, types: list[SensorType]) -> Hyperparameters:
        values = np.exp(np.asarray(vec, dtype=float))
        rho = {t: float(v) for t, v in zip(types, values[2:], strict=True)}
        return cls(q=float(values[0]), lam=float(values[1]), rho=rho)


def estimate_hyperparameters(
    objective: Callable[[Hyperparameters], float],
    initial: Hyperparameters,
    types: list[SensorType],
    max_iter: int = 200,
    q_span: float = 1e6,
    lam_span: float = 1e4,
    rho_span: float = 1e2,
) -> Hyperparameters:
    """Minimize ``objective`` (the negative log marginal likelihood) over ``log theta``.

    The search is bounded to ``initial / span`` ... ``initial * span`` for each quantity; the
    marginal likelihood is nearly flat in ``lambda`` when every retained direction is
    informed by the sensors, and the bound keeps the search from drifting along that
    direction.

    Parameters
    ----------
    objective:
        Callable returning the negative log marginal likelihood of a hyperparameter set.
    initial:
        Starting point.
    types:
        Sensor types whose noise scales are estimated (in a fixed order).
    max_iter:
        Iteration budget of the Nelder–Mead search.
    q_span, lam_span, rho_span:
        Multiplicative half-widths of the search box around ``initial``.
    """
    x0 = initial.to_log_vector(types)
    half_width = np.log(np.array([q_span, lam_span] + [rho_span] * len(types)))
    bounds = list(zip(x0 - half_width, x0 + half_width, strict=True))

    def fun(x: np.ndarray) -> float:
        value = objective(Hyperparameters.from_log_vector(x, types))
        return float(value) if np.isfinite(value) else 1e300

    res = minimize(
        fun,
        x0,
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-4},
    )
    return Hyperparameters.from_log_vector(res.x, types)
