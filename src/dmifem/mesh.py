"""Inverse mesh: nodes, elements, boundary conditions and gather matrices.

Global degrees of freedom are numbered ``6 * node + local_index``. Kinematic boundary
conditions remove degrees of freedom; the free ones form the vector ``u`` of dimension ``n``
of the manuscript. :meth:`Mesh.gather` returns the matrix ``T_e`` of Eq. (assembly), which
gathers the free degrees of freedom of an element and rotates them to the local frame.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np

from dmifem.elements.base import Array, InverseElement
from dmifem.elements.beam import InverseBeam

DOFS_PER_NODE = 6


class Mesh:
    """Inverse finite element mesh.

    Parameters
    ----------
    nodes:
        Nodal coordinates, shape ``(n_nodes, 3)``.
    connectivity:
        Node indices of each element.
    element_factory:
        Callable building an element from the coordinates of its nodes; by default an
        :class:`InverseBeam` whose local ``y`` axis follows ``reference``.
    reference:
        Reference vector of the local ``y`` axis of the default beam elements (global ``z``
        when omitted). Gauge offsets ``(y, z)`` refer to this local frame.
    """

    def __init__(
        self,
        nodes: Array,
        connectivity: Sequence[Sequence[int]],
        element_factory: Callable[[Array], InverseElement] | None = None,
        reference: Sequence[float] | None = None,
    ) -> None:
        self.nodes = np.asarray(nodes, dtype=float).reshape(-1, 3)
        self.connectivity = [tuple(int(i) for i in conn) for conn in connectivity]
        if element_factory is None:
            ref = None if reference is None else np.asarray(reference, dtype=float)

            def element_factory(coords: Array) -> InverseElement:
                return InverseBeam(coords, reference=ref)

        self.elements: list[InverseElement] = [
            element_factory(self.nodes[list(conn)]) for conn in self.connectivity
        ]
        self._fixed: set[int] = set()
        self._free_index: Array | None = None

    # ---- degrees of freedom ----------------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_total_dofs(self) -> int:
        return DOFS_PER_NODE * self.n_nodes

    def fix(self, node: int, dofs: Iterable[int] | None = None) -> None:
        """Fix degrees of freedom of a node (all six by default)."""
        local = range(DOFS_PER_NODE) if dofs is None else dofs
        for d in local:
            self._fixed.add(DOFS_PER_NODE * node + int(d))
        self._free_index = None

    @property
    def free_dofs(self) -> Array:
        """Global indices of the free degrees of freedom."""
        if self._free_index is None:
            free = [d for d in range(self.n_total_dofs) if d not in self._fixed]
            self._free_index = np.array(free, dtype=int)
        return self._free_index

    @property
    def n_free(self) -> int:
        """Dimension ``n`` of the displacement vector."""
        return self.free_dofs.size

    def _selection(self, element: int) -> Array:
        """Matrix selecting the global dofs of an element from the free-dof vector."""
        position = {dof: i for i, dof in enumerate(self.free_dofs)}
        conn = self.connectivity[element]
        sel = np.zeros((DOFS_PER_NODE * len(conn), self.n_free))
        for a, node in enumerate(conn):
            for d in range(DOFS_PER_NODE):
                g = DOFS_PER_NODE * node + d
                if g in position:
                    sel[DOFS_PER_NODE * a + d, position[g]] = 1.0
        return sel

    def gather(self, element: int) -> Array:
        """Return ``T_e``: free global dofs to local element dofs, shape ``(n_edof, n)``."""
        elem = self.elements[element]
        return elem.transformation() @ self._selection(element)

    def expand(self, u_free: Array) -> Array:
        """Embed a free-dof vector (or array of them) into the full global dof vector."""
        u_free = np.asarray(u_free, dtype=float)
        full = np.zeros(u_free.shape[:-1] + (self.n_total_dofs,))
        full[..., self.free_dofs] = u_free
        return full
