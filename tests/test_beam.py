"""Inverse beam element: interpolation, section strains and gauge row."""

import numpy as np

from dmifem.elements.beam import InverseBeam


def _beam(length=2.0):
    return InverseBeam(np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0]]), reference=[0.0, 1.0, 0.0])


def test_rigid_body_motion_has_no_section_strain():
    beam = _beam()
    dofs = np.zeros(12)
    dofs[[1, 7]] = 0.3  # uniform translation in y
    dofs[[5, 11]] = 0.01  # rotation about z with the matching transverse offset
    dofs[7] += 0.01 * beam.length
    for xi in (0.0, 0.37, 1.0):
        ops = beam.section_strain_operators(xi)
        np.testing.assert_allclose(ops.stacked() @ dofs, 0.0, atol=1e-12)


def test_quadratic_deflection_gives_unit_curvature():
    beam = _beam(length=3.0)
    dofs = np.zeros(12)
    # u_y = x^2 / 2  -> t_z = x, kappa_z = 1
    dofs[7], dofs[11] = 0.5 * beam.length**2, beam.length
    for xi in (0.2, 0.8):
        ops = beam.section_strain_operators(xi)
        np.testing.assert_allclose(ops.bending[1] @ dofs, 1.0, rtol=1e-12)
        np.testing.assert_allclose(ops.bending[0] @ dofs, 0.0, atol=1e-12)


def test_interpolation_recovers_nodal_values():
    beam = _beam()
    dofs = np.arange(12, dtype=float)
    np.testing.assert_allclose(beam.translation_interpolation(0.0) @ dofs, dofs[:3])
    np.testing.assert_allclose(beam.translation_interpolation(1.0) @ dofs, dofs[6:9])
    np.testing.assert_allclose(beam.rotation_interpolation(0.0) @ dofs, dofs[3:6])
    np.testing.assert_allclose(beam.rotation_interpolation(1.0) @ dofs, dofs[9:12])


def test_gauge_row_sign_and_axial_term():
    beam = _beam()
    dofs = np.zeros(12)
    dofs[6] = 1e-3 * beam.length  # axial strain 1e-3
    dofs[7], dofs[11] = 0.5 * beam.length**2, beam.length  # unit curvature kappa_z
    row = beam.gauge_row(0.5, 0.1, 0.0)
    np.testing.assert_allclose(row @ dofs, 1e-3 - 0.1, rtol=1e-10)


def test_local_frame_is_orthonormal():
    beam = InverseBeam(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    frame = beam.local_frame()
    np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(frame), 1.0)
