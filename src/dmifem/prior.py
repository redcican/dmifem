"""Spatial prior and static iFEM solve (manuscript Sections 2.1 and 3.3, Eq. prior, Proposition 3.1).

:func:`assemble_prior` stacks the dimensionless section-strain operators ``L`` of all elements at
their quadrature points: translations are divided by the reference length ``ell``, curvature and
twist are multiplied by it, and the quadrature weights are divided by the total measure of the
mesh. Rows of one component that coincide at several quadrature points of an element, such as the
constant axial strain and twist of the beam, are merged into one row with the summed weight (one
quadrature point for these strains); identically zero rows, such as the transverse shear of the
Euler-Bernoulli beam, are dropped.

:func:`spatial_prior` forms the prior of Eq. (prior). The unit-modulus stress resultants
``n = G L u`` have a Matern-3/2 correlation along the members, one independent field per
resultant component, conditioned on the resultants that vanish by the natural boundary conditions
(the bending moments at free and pinned ends), with a nugget of ``1e-6`` times the mean variance;
the section-strain correlation is ``Gamma = G^{-1} Gamma_n G^{-1}`` and ``K1 = L^T Gamma^{-1} L``.
With ``correlated=False``, ``Gamma = W^{-1}`` and ``K1`` is the unit-weight iFEM matrix of
Eq. (assembly), the limit of uncorrelated section strains. A uniform scale of ``K1`` is absorbed by
the prior scale ``lambda``. :func:`static_ifem` returns the posterior of Proposition 3.1.

Along-member distances are implemented for two-node (beam) elements.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from dmifem.elements.base import Array
from dmifem.mesh import DOFS_PER_NODE, Mesh
from dmifem.scaling import dof_scale

MEMBRANE, BENDING, SHEAR = 0, 1, 2
BENDING_MOMENTS: tuple[tuple[int, int], ...] = ((BENDING, 0), (BENDING, 1))
ROTATION_DOFS = (3, 4, 5)
_MATERN_CAP = 745.0  # exp(-745) underflows to the smallest subnormal; keeps inf distances finite


@dataclass(frozen=True)
class BeamSection:
    """Section factors of a beam element.

    Attributes
    ----------
    area:
        Cross-sectional area (membrane and shear resultants).
    iy, iz:
        Second moments of area about the local ``y`` and ``z`` axes (bending resultants).
    torsion:
        Torsion constant (twist resultant).
    outer_radius:
        Outer radius of a tubular section, used for the default reference length.
    """

    area: float
    iy: float
    iz: float
    torsion: float
    outer_radius: float | None = None

    @classmethod
    def tube(cls, outer_radius: float, wall: float) -> BeamSection:
        """Circular tube with outer radius ``outer_radius`` and wall thickness ``wall``."""
        if not 0.0 < wall <= outer_radius:
            raise ValueError("wall thickness must lie in (0, outer radius]")
        inner = outer_radius - wall
        area = np.pi * (outer_radius**2 - inner**2)
        inertia = 0.25 * np.pi * (outer_radius**4 - inner**4)
        return cls(float(area), float(inertia), float(inertia), float(2.0 * inertia), float(outer_radius))

    def factor(self, kind: int, row: int) -> float:
        """Section factor of the resultant of a section-strain row (membrane and shear: area)."""
        if kind == BENDING:
            return (self.iy, self.iz, self.torsion)[row]
        return self.area


@dataclass(frozen=True)
class PriorOperator:
    """Dimensionless section-strain operator ``L`` and its row structure.

    Attributes
    ----------
    operator:
        ``L`` of shape ``(m, n)`` on the dimensionless free degrees of freedom, without weights.
    weights:
        Quadrature weight of each row divided by the total measure of the mesh.
    kind:
        Kind of each row: ``0`` membrane, ``1`` bending (curvatures and twist), ``2`` shear.
    component:
        Row index within its kind (for the beam: bending ``0`` about ``y``, ``1`` about ``z``,
        ``2`` twist); kind and component together label the resultant component.
    element:
        Element index of each row.
    xi:
        Natural coordinate of the point of each row (weighted mean for merged rows).
    section_factor:
        Diagonal of ``G``, relative to the largest factor of the same component.
    """

    operator: Array
    weights: Array
    kind: Array
    component: Array
    element: Array
    xi: Array
    section_factor: Array

    @property
    def n_points(self) -> int:
        return self.operator.shape[0]

    @property
    def rows(self) -> Array:
        """Unit-weight rows ``W^{1/2} L``."""
        return np.sqrt(self.weights)[:, None] * self.operator

    @property
    def precision(self) -> Array:
        """Unit-weight iFEM matrix ``L^T W L`` (uncorrelated section strains)."""
        rows = self.rows
        return rows.T @ rows


def _same_row(a: Array, b: Array, tol: float) -> bool:
    peak = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return float(np.max(np.abs(a - b))) <= tol * peak


def assemble_prior(
    mesh: Mesh,
    length: float = 1.0,
    sections: Sequence[BeamSection] | None = None,
    merge_tol: float = 1e-12,
) -> PriorOperator:
    """Assemble the dimensionless ``L`` over all elements and quadrature points.

    Parameters
    ----------
    mesh:
        Inverse mesh with boundary conditions applied.
    length:
        Reference length ``ell`` of Eq. (scaling).
    sections:
        One section per element for the factors ``G``; unit factors when omitted.
    merge_tol:
        Relative tolerance under which rows of one component of an element are merged.
    """
    if sections is not None and len(sections) != len(mesh.elements):
        raise ValueError("one section per element is required")
    scale = dof_scale(mesh, length)
    total = float(sum(np.sum(elem.quadrature()[1]) for elem in mesh.elements))
    rows: list[Array] = []
    kinds: list[int] = []
    comps: list[int] = []
    elems: list[int] = []
    xis: list[float] = []
    weights: list[float] = []
    factors: list[float] = []
    for e, elem in enumerate(mesh.elements):
        gather = mesh.gather(e) * scale[None, :]
        points, point_weights = elem.quadrature()
        groups: dict[tuple[int, int], list[tuple[Array, list[float], list[float]]]] = {}
        for xi, w in zip(points, point_weights, strict=True):
            ops = elem.section_strain_operators(xi)
            for kind, op in enumerate((ops.membrane, ops.bending, ops.shear)):
                factor = length if kind == BENDING else 1.0
                for j in range(op.shape[0]):
                    row = factor * (op[j] @ gather)
                    if not np.any(row):
                        continue
                    bucket = groups.setdefault((kind, j), [])
                    for stored, xs, ws in bucket:
                        if _same_row(stored, row, merge_tol):
                            xs.append(float(xi))
                            ws.append(float(w))
                            break
                    else:
                        bucket.append((row, [float(xi)], [float(w)]))
        for kind, j in sorted(groups):
            for row, xs, ws in groups[(kind, j)]:
                rows.append(row)
                kinds.append(kind)
                comps.append(j)
                elems.append(e)
                xis.append(float(np.average(xs, weights=ws)))
                weights.append(float(np.sum(ws)) / total)
                factors.append(1.0 if sections is None else float(sections[e].factor(kind, j)))
    if not rows:
        raise ValueError("the mesh has no section-strain rows")
    kind_arr = np.array(kinds, dtype=int)
    comp_arr = np.array(comps, dtype=int)
    fac = np.array(factors, dtype=float)
    if np.any(fac <= 0.0):
        raise ValueError("section factors must be positive")
    for kind, comp in set(zip(kinds, comps, strict=True)):
        mask = (kind_arr == kind) & (comp_arr == comp)
        fac[mask] = fac[mask] / fac[mask].max()
    return PriorOperator(
        operator=np.vstack(rows),
        weights=np.array(weights),
        kind=kind_arr,
        component=comp_arr,
        element=np.array(elems, dtype=int),
        xi=np.array(xis),
        section_factor=fac,
    )


# ---- along-member distances ------------------------------------------------------------------
def _line_elements(mesh: Mesh) -> tuple[Array, Array]:
    """End nodes ``(n_el, 2)`` and lengths ``(n_el,)`` of two-node elements."""
    ends, lengths = [], []
    for conn in mesh.connectivity:
        if len(conn) != 2:
            raise NotImplementedError("along-member distances are implemented for two-node elements")
        a, b = conn
        ends.append((a, b))
        lengths.append(float(np.linalg.norm(mesh.nodes[b] - mesh.nodes[a])))
    return np.array(ends, dtype=int), np.array(lengths)


def node_distances(mesh: Mesh) -> Array:
    """Shortest along-member distances between all nodes (``inf`` between disconnected parts)."""
    ends, lengths = _line_elements(mesh)
    best: dict[tuple[int, int], float] = {}
    for (a, b), length in zip(ends, lengths, strict=True):
        key = (int(min(a, b)), int(max(a, b)))
        best[key] = min(best.get(key, np.inf), float(length))
    rows = [k[0] for k in best]
    cols = [k[1] for k in best]
    graph = csr_matrix((list(best.values()), (rows, cols)), shape=(mesh.n_nodes, mesh.n_nodes))
    return shortest_path(graph, directed=False)


def structure_length(mesh: Mesh) -> float:
    """Largest along-member distance between two nodes, the default correlation length."""
    dist = node_distances(mesh)
    return float(np.max(dist[np.isfinite(dist)]))


def along_member_distances(
    mesh: Mesh,
    element_a: Array,
    xi_a: Array,
    element_b: Array | None = None,
    xi_b: Array | None = None,
    node_dist: Array | None = None,
) -> Array:
    """Along-member distances between points ``(element, xi)`` of two sets (one set if ``b`` omitted)."""
    ends, lengths = _line_elements(mesh)
    d_nodes = node_distances(mesh) if node_dist is None else node_dist
    ea = np.asarray(element_a, dtype=int).reshape(-1)
    xa = np.asarray(xi_a, dtype=float).reshape(-1)
    eb = ea if element_b is None else np.asarray(element_b, dtype=int).reshape(-1)
    xb = xa if xi_b is None else np.asarray(xi_b, dtype=float).reshape(-1)
    off_a = np.column_stack([xa * lengths[ea], (1.0 - xa) * lengths[ea]])
    off_b = np.column_stack([xb * lengths[eb], (1.0 - xb) * lengths[eb]])
    dist = np.full((ea.size, eb.size), np.inf)
    for p in (0, 1):
        for q in (0, 1):
            via = d_nodes[ends[ea, p][:, None], ends[eb, q][None, :]]
            dist = np.minimum(dist, off_a[:, p][:, None] + via + off_b[:, q][None, :])
    same = ea[:, None] == eb[None, :]
    direct = np.abs(xa[:, None] - xb[None, :]) * lengths[ea][:, None]
    return np.where(same, np.minimum(dist, direct), dist)


def point_node_distances(
    mesh: Mesh, element: Array, xi: Array, nodes: Sequence[int], node_dist: Array | None = None
) -> Array:
    """Along-member distances from points ``(element, xi)`` to mesh nodes, ``(n_points, n_nodes)``."""
    ends, lengths = _line_elements(mesh)
    d_nodes = node_distances(mesh) if node_dist is None else node_dist
    e = np.asarray(element, dtype=int).reshape(-1)
    x = np.asarray(xi, dtype=float).reshape(-1)
    idx = np.asarray(nodes, dtype=int).reshape(-1)
    to_start = (x * lengths[e])[:, None] + d_nodes[ends[e, 0]][:, idx]
    to_end = ((1.0 - x) * lengths[e])[:, None] + d_nodes[ends[e, 1]][:, idx]
    return np.minimum(to_start, to_end)


def natural_boundary_nodes(mesh: Mesh) -> list[int]:
    """End nodes (one element) whose rotations are all free: free and pinned ends."""
    degree = np.zeros(mesh.n_nodes, dtype=int)
    for conn in mesh.connectivity:
        for node in conn:
            degree[node] += 1
    fixed = set(mesh.fixed_dofs.tolist())
    out = []
    for node in np.flatnonzero(degree == 1):
        rotations = {DOFS_PER_NODE * int(node) + d for d in ROTATION_DOFS}
        if not rotations & fixed:
            out.append(int(node))
    return out


def matern32(distance: Array, correlation_length: float) -> Array:
    """Matern-3/2 correlation ``(1 + sqrt(3) d / l) exp(-sqrt(3) d / l)``."""
    a = np.minimum(np.sqrt(3.0) * np.asarray(distance, dtype=float) / correlation_length, _MATERN_CAP)
    return (1.0 + a) * np.exp(-a)


def prior_correlation(
    mesh: Mesh,
    prior: PriorOperator,
    correlation_length: float | None = None,
    free_ends: Sequence[int] | None = None,
    vanishing: Sequence[tuple[int, int]] = BENDING_MOMENTS,
    nugget: float = 1e-6,
) -> Array:
    """Section-strain correlation ``Gamma`` of Eq. (prior), shape ``(m, m)``.

    Parameters
    ----------
    mesh, prior:
        Inverse mesh and its dimensionless section-strain operator.
    correlation_length:
        ``ell_c`` in the length unit of the mesh; :func:`structure_length` when omitted.
    free_ends:
        Nodes at which the ``vanishing`` resultants are zero; :func:`natural_boundary_nodes` when
        omitted, ``()`` for no conditioning.
    vanishing:
        Resultant components ``(kind, component)`` that vanish at the free ends (bending moments).
    nugget:
        Nugget relative to the mean variance of the conditioned resultants.
    """
    lc = structure_length(mesh) if correlation_length is None else float(correlation_length)
    if not lc > 0.0:
        raise ValueError("correlation length must be positive")
    d_nodes = node_distances(mesh)
    dist = along_member_distances(mesh, prior.element, prior.xi, node_dist=d_nodes)
    same = (prior.kind[:, None] == prior.kind[None, :]) & (
        prior.component[:, None] == prior.component[None, :]
    )
    gamma = matern32(dist, lc) * same
    ends = natural_boundary_nodes(mesh) if free_ends is None else [int(n) for n in free_ends]
    if ends and vanishing:
        dz = point_node_distances(mesh, prior.element, prior.xi, ends, node_dist=d_nodes)
        labels = [(i, kind, comp) for i in range(len(ends)) for kind, comp in vanishing]
        cross = np.zeros((prior.n_points, len(labels)))
        zz = np.zeros((len(labels), len(labels)))
        for a, (i, kind, comp) in enumerate(labels):
            match = (prior.kind == kind) & (prior.component == comp)
            cross[:, a] = matern32(dz[:, i], lc) * match
            for b, (i2, kind2, comp2) in enumerate(labels):
                if (kind, comp) == (kind2, comp2):
                    zz[a, b] = matern32(d_nodes[ends[i], ends[i2]], lc)
        active = np.any(cross != 0.0, axis=0)
        if np.any(active):
            cross = cross[:, active]
            zz = zz[np.ix_(active, active)]
            gamma = gamma - cross @ np.linalg.solve(zz, cross.T)
    gamma = 0.5 * (gamma + gamma.T)
    gamma = gamma + nugget * float(np.mean(np.diag(gamma))) * np.eye(prior.n_points)
    inv_g = 1.0 / prior.section_factor
    return gamma * inv_g[:, None] * inv_g[None, :]


# ---- prior precision ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpatialPrior:
    """Prior of Eq. (prior) with ``K1 = R^T R`` held through its triangular factor ``R``.

    Attributes
    ----------
    operator:
        The dimensionless section-strain operator ``L``.
    correlation:
        Section-strain correlation ``Gamma``.
    factor:
        Upper-triangular ``R`` (``n x n``) from the QR factorization of ``Gamma^{-1/2} L``.
    correlation_length:
        ``ell_c`` used, or ``None`` for uncorrelated section strains.
    """

    operator: PriorOperator
    correlation: Array
    factor: Array
    correlation_length: float | None

    @property
    def n(self) -> int:
        return self.factor.shape[1]

    @property
    def precision(self) -> Array:
        """``K1 = R^T R``."""
        return self.factor.T @ self.factor

    def whiten(self, rows: Array) -> Array:
        """``R^{-T} rows^T``: functionals as vectors in ``w = R u``, where ``K1`` is the identity.

        ``K1`` spans many decades on long members (curvature integrates twice to displacement),
        while ``R`` has the square root of its condition number; products in ``w`` stay accurate.
        """
        rows = np.atleast_2d(np.asarray(rows, dtype=float))
        return solve_triangular(self.factor, rows.T, trans="T", lower=False)

    def color(self, w: Array) -> Array:
        """``R^{-1} w``: fields in ``w = R u`` mapped back to the dimensionless dofs."""
        return solve_triangular(self.factor, w, lower=False)

    def solve(self, rhs: Array) -> Array:
        """``K1^{-1} rhs`` by two triangular solves."""
        tmp = solve_triangular(self.factor, rhs, trans="T", lower=False)
        return solve_triangular(self.factor, tmp, lower=False)

    @property
    def covariance(self) -> Array:
        """``K1^{-1}``."""
        cov = self.solve(np.eye(self.n))
        return 0.5 * (cov + cov.T)


def spatial_prior(
    mesh: Mesh,
    prior: PriorOperator,
    correlated: bool = True,
    correlation_length: float | None = None,
    free_ends: Sequence[int] | None = None,
    vanishing: Sequence[tuple[int, int]] = BENDING_MOMENTS,
    nugget: float = 1e-6,
    rank_tol: float = 1e-12,
) -> SpatialPrior:
    """Form ``Gamma`` and the factor of ``K1 = L^T Gamma^{-1} L`` (Eq. prior).

    Raises ``ValueError`` when the constrained operator violates the rank condition of
    Remark rank, in which case ``K1`` is singular.
    """
    if correlated:
        lc = structure_length(mesh) if correlation_length is None else float(correlation_length)
        gamma = prior_correlation(mesh, prior, lc, free_ends, vanishing, nugget)
    else:
        lc = None
        gamma = np.diag(1.0 / prior.weights)
    try:
        chol = cholesky(gamma, lower=True)
    except np.linalg.LinAlgError as exc:
        # the Matern kernel of the graph distance is positive definite on unbranched members (the
        # distance is then one-dimensional); on branched or closed frames it can be indefinite
        raise ValueError(
            "the section-strain correlation is not positive definite on this mesh; use a shorter "
            "correlation_length or correlated_prior=False (branched or closed frames)"
        ) from exc
    whitened = solve_triangular(chol, prior.operator, lower=True)
    m, n = whitened.shape
    if m < n:
        raise ValueError(f"{m} section-strain rows cannot constrain {n} degrees of freedom (Remark rank)")
    factor = np.linalg.qr(whitened, mode="r")
    diag = np.abs(np.diag(factor))
    if diag.min() <= rank_tol * diag.max():
        raise ValueError("the constrained section-strain operator is rank deficient (Remark rank)")
    return SpatialPrior(prior, gamma, factor, lc)


def prior_solve(prior: SpatialPrior | Array, rhs: Array) -> Array:
    """``K1^{-1} rhs`` for a :class:`SpatialPrior` or a given covariance matrix ``K1^{-1}``."""
    if isinstance(prior, SpatialPrior):
        return prior.solve(rhs)
    return np.asarray(prior, dtype=float) @ rhs


# ---- static limit ------------------------------------------------------------------------------
def classical_weights(prior: PriorOperator, w_m: float, w_b: float, w_s: float) -> Array:
    """Per-row weights of the classical functional, Eq. (functional), for the rows of ``L``."""
    table = np.array([w_m, w_b, w_s], dtype=float)
    return table[prior.kind]


def static_ifem(rows: Array, weights: Array, measurements: Array) -> tuple[Array, Array]:
    """Static iFEM solution and covariance, Eq. (static): ``u = K^{-1} f``, ``P = K^{-1}``.

    Parameters
    ----------
    rows:
        Section-strain rows ``H_eps`` (quadrature weights included), shape ``(n_rows, n)``.
    weights:
        Inverse noise variances per row (the classical weights), shape ``(n_rows,)``.
    measurements:
        Measured section strains per row, shape ``(n_rows,)``.
    """
    precision = rows.T @ (weights[:, None] * rows)
    rhs = rows.T @ (weights * measurements)
    covariance = np.linalg.inv(precision)
    return covariance @ rhs, covariance
