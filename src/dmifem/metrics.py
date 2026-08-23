"""Evaluation metrics (manuscript Section 4.3, Eqs. nrmse–crps)."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.stats import norm

from dmifem.elements.base import Array


def nrmse(reference: Array, reconstruction: Array) -> float:
    """Normalized root-mean-square error, Eq. (nrmse)."""
    y = np.asarray(reference, dtype=float)
    yhat = np.asarray(reconstruction, dtype=float)
    return float(np.sqrt(np.sum((yhat - y) ** 2)) / np.sqrt(np.sum((y - y.mean()) ** 2)))


def trac(reference: Array, reconstruction: Array) -> float:
    """Time-response assurance criterion, Eq. (nrmse)."""
    y = np.asarray(reference, dtype=float)
    yhat = np.asarray(reconstruction, dtype=float)
    return float((yhat @ y) ** 2 / ((yhat @ yhat) * (y @ y)))


def band_spectral_error(
    reference: Array, reconstruction: Array, fs: float, band: tuple[float, float], nperseg: int | None = None
) -> float:
    """Mean absolute error in dB of the Welch spectra over a frequency band, Eq. (spec)."""
    f, s_ref = welch(reference, fs=fs, nperseg=nperseg)
    _, s_rec = welch(reconstruction, fs=fs, nperseg=nperseg)
    mask = (f >= band[0]) & (f <= band[1])
    if not np.any(mask):
        raise ValueError("band contains no frequency bin")
    return float(np.mean(np.abs(10.0 * np.log10(s_rec[mask] / s_ref[mask]))))


def coverage(reference: Array, mean: Array, std: Array, level: float = 0.95) -> float:
    """Empirical coverage of the central ``level`` interval, Eq. (crps)."""
    z = norm.ppf(0.5 * (1.0 + level))
    inside = np.abs(np.asarray(reference) - np.asarray(mean)) <= z * np.asarray(std)
    return float(np.mean(inside))


def crps_gaussian(reference: Array, mean: Array, std: Array, normalize: bool = True) -> float:
    """Continuous ranked probability score of a Gaussian predictive distribution, Eq. (crps)."""
    y = np.asarray(reference, dtype=float)
    mu = np.asarray(mean, dtype=float)
    s = np.asarray(std, dtype=float)
    z = (y - mu) / s
    score = s * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    value = float(np.mean(score))
    return value / float(np.std(y)) if normalize else value
