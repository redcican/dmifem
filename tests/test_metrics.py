"""Accuracy and calibration metrics."""

import numpy as np

from dmifem.metrics import band_spectral_error, coverage, crps_gaussian, nrmse, trac


def test_perfect_reconstruction():
    y = np.sin(np.linspace(0.0, 20.0, 500))
    assert nrmse(y, y) == 0.0
    assert np.isclose(trac(y, y), 1.0)
    assert np.isclose(band_spectral_error(y, y, fs=25.0, band=(0.05, 1.0)), 0.0)


def test_scaled_reconstruction_keeps_trac():
    y = np.sin(np.linspace(0.0, 20.0, 500))
    assert np.isclose(trac(y, 2.0 * y), 1.0)
    assert nrmse(y, 2.0 * y) > 0.9


def test_coverage_and_crps_of_calibrated_gaussian():
    rng = np.random.default_rng(0)
    mean = np.zeros(20000)
    std = np.full(20000, 2.0)
    y = mean + std * rng.standard_normal(20000)
    assert abs(coverage(y, mean, std) - 0.95) < 0.01
    expected = 2.0 / np.sqrt(np.pi)  # E[CRPS] of a calibrated Gaussian is sigma / sqrt(pi)
    assert np.isclose(crps_gaussian(y, mean, std, normalize=False), expected, rtol=0.03)
    assert crps_gaussian(y, mean, 0.5 * std) > crps_gaussian(y, mean, std)
