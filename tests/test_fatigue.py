"""Rainflow counting and damage-equivalent load."""

import numpy as np

from dmifem.fatigue import damage_equivalent_load, del_of_series, rainflow, reversals


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
