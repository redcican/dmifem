"""Rainflow counting, damage-equivalent load, damage ratio and band substitution."""

import numpy as np
import pytest

from dmifem.fatigue import (
    band_substitution,
    damage_equivalent_load,
    damage_ratio,
    del_of_series,
    rainflow,
    reversals,
)


def test_reversals_keep_extrema_only():
    x = np.array([0.0, 1.0, 2.0, 1.0, 0.0, -1.0, 0.5, 0.5, 2.0])
    np.testing.assert_allclose(reversals(x), [0.0, 2.0, -1.0, 2.0])


def test_rainflow_astm_example():
    # ASTM E1049 example history: one full cycle of range 4, half cycles 3, 4, 8, 9, 8, 6
    series = np.array([-2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0])
    ranges, _, counts = rainflow(series)
    assert np.isclose(counts.sum(), 4.0)
    np.testing.assert_allclose(sorted(ranges[counts == 1.0]), [4.0])
    np.testing.assert_allclose(sorted(ranges[counts == 0.5]), [3.0, 4.0, 6.0, 8.0, 8.0, 9.0])


def test_sine_gives_full_cycles_of_range_two_amplitudes():
    t = np.linspace(0.0, 10.0, 4001)
    series = 3.0 * np.sin(2.0 * np.pi * t)
    ranges, _, counts = rainflow(series)
    assert np.isclose(counts.sum(), 10.0, atol=0.6)
    np.testing.assert_allclose(ranges.max(), 6.0, rtol=1e-3)
    assert np.isclose(del_of_series(series, slope=3.0, n_ref=10.0), 6.0, rtol=0.05)


def test_damage_equivalent_load_formula():
    ranges, counts = np.array([2.0, 4.0]), np.array([8.0, 1.0])
    expected = ((8.0 * 8.0 + 1.0 * 64.0) / 4.0) ** (1.0 / 3.0)
    assert np.isclose(damage_equivalent_load(ranges, counts, 3.0, 4.0), expected)


def test_damage_ratio_is_the_del_ratio_to_the_slope():
    assert damage_ratio(0.95, 3.0) == pytest.approx(0.857375, rel=1e-12)
    np.testing.assert_allclose(damage_ratio(np.array([1.0, 0.5]), 4.0), [1.0, 0.0625])


def test_band_substitution_isolates_the_band_of_the_deficit():
    fs = 20.0
    t = np.arange(int(200 * fs)) / fs  # 200 s: 0.1 Hz and 2 Hz fall on frequency bins
    low = np.sin(2.0 * np.pi * 0.1 * t)
    high = 0.5 * np.sin(2.0 * np.pi * 2.0 * t)
    reference = low + high
    reconstruction = low + 0.8 * high  # deficit in the upper band only
    ratios, total = band_substitution(reference, reconstruction, fs, [(0.0, 1.0), (1.0, np.inf)], slope=3.0)
    assert ratios.shape == (2,)
    assert ratios[0] == pytest.approx(1.0, abs=1e-9)
    # the upper band carries the whole deficit: substituting it gives the full reconstruction
    assert ratios[1] == pytest.approx(total, rel=1e-9)
    assert total < 1.0
    with pytest.raises(ValueError):
        band_substitution(reference, reconstruction[:-1], fs, [(0.0, 1.0)], slope=3.0)
