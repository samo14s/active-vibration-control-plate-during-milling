"""Reissner-Mindlin plate FEM for the thin-walled cantilever workpiece.

4-node bilinear quadrilateral, DOFs per node (w, tx, tz):
    w  : transverse deflection (out of plane, direction y)
    tx : rotation of the normal in the x-y plane   (shear:  gxy = dw/dx - tx)
    tz : rotation of the normal in the z-y plane   (shear:  gzy = dw/dz - tz)

Plate plane coordinates: x in [0, lp] (feed direction), z in [0, h]
(cantilever span; clamped at z = 0, milled along the free edge z = h).

Selective reduced integration: 2x2 Gauss for bending and mass, 1-point for
transverse shear (removes shear locking for thin plates).

Material removal is represented by rebuilding the mesh on the receded
geometry: the machined top edge moves from z = hp to z = hp - height_removed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .params import Params

NDOF = 3  # per node


@dataclass
class PlateModel:
    """Assembled FEM model on (possibly receded) geometry."""

    params: Params
    height_removed: float
    nx: int
    nz: int
    height: float            # current plate height hp - height_removed
    nodes: np.ndarray        # (n_nodes, 2) coordinates (x, z)
    elems: np.ndarray        # (n_elems, 4) connectivity, CCW
    K: sp.csr_matrix         # stiffness, free DOFs only
    M: sp.csr_matrix         # mass, free DOFs only
    free: np.ndarray         # indices of free DOFs in the full numbering
    n_free: int

    def interp_w(self, x: float, z: float) -> np.ndarray:
        """Row vector e (length n_free) with e @ q_free = w(x, z).

        Bilinear interpolation of the deflection DOFs of the containing
        element. Points on the boundary are clamped into the mesh.
        """
        ex = self.height / self.nz  # element size z
        exx = self.params.plate.lp / self.nx
        i = int(np.clip(np.floor(x / exx), 0, self.nx - 1))
        j = int(np.clip(np.floor(z / ex), 0, self.nz - 1))
        e = j * self.nx + i
        conn = self.elems[e]
        xn = self.nodes[conn]
        # local coordinates in [-1, 1]
        xi = 2.0 * (x - xn[0, 0]) / (xn[1, 0] - xn[0, 0]) - 1.0
        eta = 2.0 * (z - xn[0, 1]) / (xn[3, 1] - xn[0, 1]) - 1.0
        N = _shape(xi, eta)
        row = np.zeros(self.n_free)
        full2free = self._full2free()
        for a in range(4):
            g = full2free[NDOF * conn[a]]  # w DOF
            if g >= 0:
                row[g] = N[a]
        return row

    def _full2free(self) -> np.ndarray:
        if not hasattr(self, "_f2f"):
            f2f = -np.ones(NDOF * len(self.nodes), dtype=int)
            f2f[self.free] = np.arange(self.n_free)
            object.__setattr__(self, "_f2f", f2f)
        return self._f2f


def _shape(xi: float, eta: float) -> np.ndarray:
    return 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta),
    ])


def _shape_deriv(xi: float, eta: float) -> np.ndarray:
    """(2, 4): d N / d xi ; d N / d eta."""
    return 0.25 * np.array([
        [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
        [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)],
    ])


def element_matrices(ex: float, ez: float, E: float, nu: float, rho: float,
                     h: float, kappa: float) -> tuple[np.ndarray, np.ndarray]:
    """Ke, Me (12x12) for a rectangular ex-by-ez Mindlin element.

    DOF order: [w1 tx1 tz1  w2 tx2 tz2  w3 tx3 tz3  w4 tx4 tz4],
    nodes CCW starting lower-left.
    """
    Db = E * h ** 3 / (12.0 * (1.0 - nu ** 2)) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0],
    ])
    G = E / (2.0 * (1.0 + nu))
    Ds = kappa * G * h * np.eye(2)
    m = rho * h
    Im = rho * h ** 3 / 12.0

    J = np.array([[ex / 2.0, 0.0], [0.0, ez / 2.0]])
    detJ = ex * ez / 4.0
    Jinv = np.linalg.inv(J)

    Ke = np.zeros((12, 12))
    Me = np.zeros((12, 12))

    gp2 = [(-1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)),
           (1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)),
           (1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)),
           (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))]

    # bending + mass: full 2x2 integration
    for xi, eta in gp2:
        N = _shape(xi, eta)
        dN = Jinv @ _shape_deriv(xi, eta)   # (2,4): d/dx, d/dz
        Bb = np.zeros((3, 12))
        Nm = np.zeros((3, 12))
        for a in range(4):
            c = 3 * a
            # curvatures: kx = d tx/dx, kz = d tz/dz, kxz = d tx/dz + d tz/dx
            Bb[0, c + 1] = dN[0, a]
            Bb[1, c + 2] = dN[1, a]
            Bb[2, c + 1] = dN[1, a]
            Bb[2, c + 2] = dN[0, a]
            Nm[0, c] = N[a]
            Nm[1, c + 1] = N[a]
            Nm[2, c + 2] = N[a]
        Ke += Bb.T @ Db @ Bb * detJ
        Me += Nm.T @ np.diag([m, Im, Im]) @ Nm * detJ

    # shear: 1-point reduced integration
    xi = eta = 0.0
    N = _shape(xi, eta)
    dN = Jinv @ _shape_deriv(xi, eta)
    Bs = np.zeros((2, 12))
    for a in range(4):
        c = 3 * a
        # gx = dw/dx - tx ; gz = dw/dz - tz
        Bs[0, c] = dN[0, a]
        Bs[0, c + 1] = -N[a]
        Bs[1, c] = dN[1, a]
        Bs[1, c + 2] = -N[a]
    Ke += Bs.T @ Ds @ Bs * (detJ * 4.0)  # weight 4 for 1-point rule

    return Ke, Me


def build_plate(params: Params, height_removed: float = 0.0) -> PlateModel:
    """Assemble the cantilever plate FEM on the receded geometry."""
    pl = params.plate
    nx, nz = params.mesh.nx, params.mesh.nz
    height = pl.hp - height_removed
    if height <= 0.5 * pl.hp:
        raise ValueError("unreasonable material removal")

    xs = np.linspace(0.0, pl.lp, nx + 1)
    zs = np.linspace(0.0, height, nz + 1)
    nodes = np.array([[x, z] for z in zs for x in xs])

    def nid(i, j):
        return j * (nx + 1) + i

    elems = np.array([
        [nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
        for j in range(nz) for i in range(nx)
    ])

    ex = pl.lp / nx
    ez = height / nz
    Ke, Me = element_matrices(ex, ez, pl.E, pl.nu, pl.rho, pl.bp, pl.kappa)

    ndof = NDOF * len(nodes)
    rows, cols, kv, mv = [], [], [], []
    for conn in elems:
        dofs = np.concatenate([[NDOF * n, NDOF * n + 1, NDOF * n + 2]
                               for n in conn])
        ii, jj = np.meshgrid(dofs, dofs, indexing="ij")
        rows.append(ii.ravel())
        cols.append(jj.ravel())
        kv.append(Ke.ravel())
        mv.append(Me.ravel())
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    K = sp.coo_matrix((np.concatenate(kv), (rows, cols)),
                      shape=(ndof, ndof)).tocsr()
    M = sp.coo_matrix((np.concatenate(mv), (rows, cols)),
                      shape=(ndof, ndof)).tocsr()

    # clamp bottom edge z = 0: nodes 0..nx
    fixed = np.concatenate([[NDOF * n, NDOF * n + 1, NDOF * n + 2]
                            for n in range(nx + 1)])
    free = np.setdiff1d(np.arange(ndof), fixed)

    model = PlateModel(
        params=params, height_removed=height_removed, nx=nx, nz=nz,
        height=height, nodes=nodes, elems=elems,
        K=K[np.ix_(free, free)].tocsr(), M=M[np.ix_(free, free)].tocsr(),
        free=free, n_free=len(free),
    )
    return model
