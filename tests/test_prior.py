"""Correlated prior of Eq. (prior): row structure, uncorrelated limit, free ends, section factors."""

from __future__ import annotations

import numpy as np
import pytest

from dmifem import Mesh
from dmifem.prior import (
    BENDING,
    MEMBRANE,
    SHEAR,
    BeamSection,
    along_member_distances,
    assemble_prior,
    natural_boundary_nodes,
    spatial_prior,
    structure_length,
)
from dmifem.scaling import dof_scale

RADIUS = 0.15


def _bending_z(prior):
    return (prior.kind == BENDING) & (prior.component == 1)


def test_rows_are_merged_per_component_and_zero_rows_dropped(mesh):
    prior = assemble_prior(mesh, length=RADIUS)
    # per beam element: one axial row, two rows per bending plane, one twist row, no shear row
    assert prior.n_points == 6 * len(mesh.elements)
    assert not np.any(prior.kind == SHEAR)
    np.testing.assert_allclose(prior.xi[prior.kind == MEMBRANE], 0.5)
    for kind, comp in set(zip(prior.kind.tolist(), prior.component.tolist(), strict=True)):
        mask = (prior.kind == kind) & (prior.component == comp)
        assert prior.weights[mask].sum() == pytest.approx(1.0)


def test_uncorrelated_limit_is_the_unit_weight_matrix(mesh):
    prior = assemble_prior(mesh, length=RADIUS)
    scale = dof_scale(mesh, RADIUS)
    total = sum(float(np.sum(elem.quadrature()[1])) for elem in mesh.elements)
    expected = np.zeros((mesh.n_free, mesh.n_free))
    for e, elem in enumerate(mesh.elements):
        gather = mesh.gather(e) * scale[None, :]
        for xi, w in zip(*elem.quadrature(), strict=True):
            ops = elem.section_strain_operators(xi)
            for op, factor in ((ops.membrane, 1.0), (ops.bending, RADIUS), (ops.shear, 1.0)):
                rows = factor * op @ gather
                expected += (w / total) * rows.T @ rows
    atol = 1e-12 * np.abs(expected).max()
    np.testing.assert_allclose(prior.precision, expected, rtol=1e-10, atol=atol)
    uncorrelated = spatial_prior(mesh, prior, correlated=False)
    assert uncorrelated.correlation_length is None
    np.testing.assert_allclose(uncorrelated.precision, expected, rtol=1e-9, atol=1e-10 * np.abs(expected).max())


def test_correlated_precision_is_positive_definite_and_solves(mesh):
    prior = spatial_prior(mesh, assemble_prior(mesh, length=RADIUS))
    k1 = prior.precision
    np.testing.assert_allclose(k1, k1.T, rtol=1e-12, atol=1e-12 * np.abs(k1).max())
    # K1 = R^T R is positive definite when R is nonsingular; cond(K1) = cond(R)^2 is of order 1e13
    # here, so the test reads the singular values of R and a backward error, not eigvalsh(K1)
    singular = np.linalg.svd(prior.factor, compute_uv=False)
    assert singular.min() > 1e-10 * singular.max()
    rhs = np.random.default_rng(0).standard_normal((mesh.n_free, 3))
    x = prior.solve(rhs)
    backward = np.linalg.norm(k1 @ x - rhs) / (np.linalg.norm(k1, 2) * np.linalg.norm(x) + np.linalg.norm(rhs))
    assert backward < 1e-12
    w = np.random.default_rng(1).standard_normal((mesh.n_free, 2))
    np.testing.assert_allclose(prior.factor @ prior.color(w), w, atol=1e-9 * np.abs(w).max())
    np.testing.assert_allclose(prior.whiten(w.T).T @ prior.factor, w.T, atol=1e-9 * np.abs(w).max())
    gamma = prior.correlation
    np.testing.assert_allclose(gamma, gamma.T)
    assert np.linalg.eigvalsh(gamma).min() > 0.0


def test_unconstrained_mesh_violates_the_rank_condition():
    x = np.linspace(0.0, 1.0, 3)
    free = Mesh(np.column_stack([x, np.zeros_like(x), np.zeros_like(x)]), [(0, 1), (1, 2)])
    with pytest.raises(ValueError):
        spatial_prior(free, assemble_prior(free))


def test_correlation_length_defaults_to_the_structure_length(mesh):
    assert structure_length(mesh) == pytest.approx(10.0)
    assert spatial_prior(mesh, assemble_prior(mesh, length=RADIUS)).correlation_length == pytest.approx(10.0)


def test_along_member_distance_follows_the_members():
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    frame = Mesh(nodes, [(0, 1), (1, 2)], reference=(0.0, 0.0, 1.0))
    # around the corner: 0.25 + 0.25, not the straight-line distance 0.354
    assert along_member_distances(frame, [0], [0.75], [1], [0.25])[0, 0] == pytest.approx(0.5)
    assert along_member_distances(frame, [0], [0.2], [0], [0.7])[0, 0] == pytest.approx(0.5)
    assert along_member_distances(frame, [0], [0.0], [1], [1.0])[0, 0] == pytest.approx(2.0)


def test_free_end_conditioning_removes_the_tip_moment(mesh):
    prior = assemble_prior(mesh, length=RADIUS)
    assert natural_boundary_nodes(mesh) == [len(mesh.elements)]
    variance = np.diag(spatial_prior(mesh, prior).correlation)
    tip = _bending_z(prior) & (prior.element == len(mesh.elements) - 1) & (prior.xi > 0.5)
    root = _bending_z(prior) & (prior.element == 0) & (prior.xi < 0.5)
    assert variance[tip][0] < 0.01
    assert variance[root][0] > 0.5
    axial = prior.kind == MEMBRANE
    np.testing.assert_allclose(variance[axial], variance[axial][0])  # axial force not conditioned
    unconditioned = np.diag(spatial_prior(mesh, prior, free_ends=()).correlation)
    assert unconditioned[tip][0] > 0.9


def test_section_factors_scale_the_strain_prior(mesh):
    tube, thick = BeamSection.tube(RADIUS, 0.008), BeamSection.tube(RADIUS, 0.016)
    n_el = len(mesh.elements)
    uniform = assemble_prior(mesh, length=RADIUS, sections=[tube] * n_el)
    varied = assemble_prior(mesh, length=RADIUS, sections=[tube] * (n_el - 1) + [thick])
    ratio = np.diag(spatial_prior(mesh, varied).correlation) / np.diag(spatial_prior(mesh, uniform).correlation)
    bend = _bending_z(varied)
    last = ratio[bend & (varied.element == n_el - 1)]
    first = ratio[bend & (varied.element == 0)]
    # strain = moment / I: the stiffer section carries the same moment with (I_tube / I_thick)^2 variance
    np.testing.assert_allclose(last / first[0], (tube.iz / thick.iz) ** 2, rtol=1e-10)
