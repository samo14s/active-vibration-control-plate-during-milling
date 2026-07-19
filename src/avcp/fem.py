"""Kirchhoff plate FEM with the 12-DOF ACM rectangular element.

Element DOFs per node: (w, w_x, w_y).  The nonconforming ACM element uses
the 12-term polynomial basis

    p(x, y) = [1, x, y, x^2, xy, y^2, x^3, x^2 y, x y^2, y^3, x^3 y, x y^3]

with shape functions N(x, y) = p(x, y)^T C^{-1}, where C collects
(p, p_x, p_y) evaluated at the four corners.  This is the same element
family as the reference MATLAB implementation shipped with the project
(Plate-FEM, Kirchhoff_plate/) and converges to Kirchhoff-Love theory.

Curvature convention: kappa = [w_xx, w_yy, 2 w_xy],
D_b = D [[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]].
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .params import PlateParams

# ---------------------------------------------------------------- basis

_EXPONENTS = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2),
              (3, 0), (2, 1), (1, 2), (0, 3), (3, 1), (1, 3)]


def _poly_row(x: float, y: float, dx: int = 0, dy: int = 0) -> np.ndarray:
    """Row of the (derivative of the) polynomial basis at (x, y)."""
    row = np.zeros(12)
    for k, (px, py) in enumerate(_EXPONENTS):
        if px < dx or py < dy:
            continue
        cx = 1.0
        for i in range(dx):
            cx *= (px - i)
        cy = 1.0
        for i in range(dy):
            cy *= (py - i)
        row[k] = cx * cy * x ** (px - dx) * y ** (py - dy)
    return row


def _gauss_1d(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(n)


class AcmElement:
    """One rectangular ACM element of size a x b (local origin at corner 0)."""

    def __init__(self, a: float, b: float, ngauss: int = 4):
        self.a, self.b = a, b
        corners = [(0.0, 0.0), (a, 0.0), (a, b), (0.0, b)]
        C = np.zeros((12, 12))
        for k, (xk, yk) in enumerate(corners):
            C[3 * k + 0] = _poly_row(xk, yk, 0, 0)
            C[3 * k + 1] = _poly_row(xk, yk, 1, 0)
            C[3 * k + 2] = _poly_row(xk, yk, 0, 1)
        self.Cinv = np.linalg.inv(C)

        gx, wx = _gauss_1d(ngauss)
        gy, wy = _gauss_1d(ngauss)
        # map to [0,a] x [0,b]
        self.xg = 0.5 * a * (gx + 1.0)
        self.yg = 0.5 * b * (gy + 1.0)
        self.wxg = 0.5 * a * wx
        self.wyg = 0.5 * b * wy

    def matrices(self, plate: PlateParams) -> tuple[np.ndarray, np.ndarray]:
        """Element stiffness and consistent mass (12 x 12)."""
        D = plate.D
        Db = D * np.array([[1.0, plate.nu, 0.0],
                           [plate.nu, 1.0, 0.0],
                           [0.0, 0.0, (1.0 - plate.nu) / 2.0]])
        Ka = np.zeros((12, 12))
        Ma = np.zeros((12, 12))
        for xi, wxi in zip(self.xg, self.wxg):
            for yj, wyj in zip(self.yg, self.wyg):
                Q = np.vstack([_poly_row(xi, yj, 2, 0),
                               _poly_row(xi, yj, 0, 2),
                               2.0 * _poly_row(xi, yj, 1, 1)])
                p = _poly_row(xi, yj)
                w = wxi * wyj
                Ka += w * Q.T @ Db @ Q
                Ma += w * np.outer(p, p)
        Ke = self.Cinv.T @ Ka @ self.Cinv
        Me = plate.rho * plate.h * self.Cinv.T @ Ma @ self.Cinv
        return Ke, Me

    def laplacian_integral(self) -> np.ndarray:
        """b_e = integral over the element of (N_xx + N_yy) dA, shape (12,).

        Used for the piezo patch coupling vector (uniform biaxial moment).
        """
        acc = np.zeros(12)
        for xi, wxi in zip(self.xg, self.wxg):
            for yj, wyj in zip(self.yg, self.wyg):
                acc += wxi * wyj * (_poly_row(xi, yj, 2, 0)
                                    + _poly_row(xi, yj, 0, 2))
        return acc @ self.Cinv

    def shape_at(self, xl: float, yl: float, dx: int = 0, dy: int = 0) -> np.ndarray:
        """Shape-function row (or derivative) at local point (xl, yl)."""
        return _poly_row(xl, yl, dx, dy) @ self.Cinv


class PlateFem:
    """Assembled cantilever plate (clamped along y = 0)."""

    def __init__(self, plate: PlateParams, nx: int, ny: int,
                 thickness_map: np.ndarray | None = None):
        """thickness_map: optional (nx, ny) per-element thickness scaling
        (1.0 = nominal) used to represent in-process material removal."""
        self.plate = plate
        self.nx, self.ny = nx, ny
        self.a = plate.Lx / nx
        self.b = plate.Ly / ny
        self.n_nodes = (nx + 1) * (ny + 1)
        self.ndof = 3 * self.n_nodes
        self.element = AcmElement(self.a, self.b)
        self._assemble(thickness_map)

    # node/DOF numbering: node (i, j) -> j * (nx + 1) + i, row-major in y
    def node_id(self, i: int, j: int) -> int:
        return j * (self.nx + 1) + i

    def element_dofs(self, i: int, j: int) -> np.ndarray:
        n = [self.node_id(i, j), self.node_id(i + 1, j),
             self.node_id(i + 1, j + 1), self.node_id(i, j + 1)]
        return np.concatenate([[3 * k, 3 * k + 1, 3 * k + 2] for k in n])

    def _assemble(self, thickness_map: np.ndarray | None):
        Ke0, Me0 = self.element.matrices(self.plate)
        rows, cols, kv, mv = [], [], [], []
        for j in range(self.ny):
            for i in range(self.nx):
                if thickness_map is None or thickness_map[i, j] == 1.0:
                    Ke, Me = Ke0, Me0
                else:
                    s = float(thickness_map[i, j])
                    # K ~ h^3, M ~ h
                    Ke, Me = Ke0 * s ** 3, Me0 * s
                dofs = self.element_dofs(i, j)
                grid = np.meshgrid(dofs, dofs, indexing="ij")
                rows.append(grid[0].ravel())
                cols.append(grid[1].ravel())
                kv.append(Ke.ravel())
                mv.append(Me.ravel())
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        self.K = sp.csr_array((np.concatenate(kv), (rows, cols)),
                              shape=(self.ndof, self.ndof))
        self.M = sp.csr_array((np.concatenate(mv), (rows, cols)),
                              shape=(self.ndof, self.ndof))
        # clamped edge y = 0: all DOFs of nodes j = 0
        clamped_nodes = [self.node_id(i, 0) for i in range(self.nx + 1)]
        fixed = np.concatenate([[3 * n, 3 * n + 1, 3 * n + 2]
                                for n in clamped_nodes])
        self.free = np.setdiff1d(np.arange(self.ndof), fixed)
        self.Kff = self.K[np.ix_(self.free, self.free)].tocsc()
        self.Mff = self.M[np.ix_(self.free, self.free)].tocsc()

    # ---------------------------------------------------------- point ops

    def locate(self, x: float, y: float) -> tuple[int, int, float, float]:
        """Element (i, j) containing (x, y) and local coordinates."""
        i = min(int(x / self.a), self.nx - 1)
        j = min(int(y / self.b), self.ny - 1)
        return i, j, x - i * self.a, y - j * self.b

    def point_row(self, x: float, y: float, dx: int = 0, dy: int = 0) -> np.ndarray:
        """Sparse row r such that r @ u_full = derivative of w at (x, y)."""
        i, j, xl, yl = self.locate(x, y)
        row = np.zeros(self.ndof)
        row[self.element_dofs(i, j)] = self.element.shape_at(xl, yl, dx, dy)
        return row

    def point_row_free(self, x: float, y: float, dx: int = 0, dy: int = 0) -> np.ndarray:
        return self.point_row(x, y, dx, dy)[self.free]

    def point_load(self, x: float, y: float, f: float = 1.0) -> np.ndarray:
        """Consistent nodal force vector (free DOFs) for a point load."""
        return self.point_row_free(x, y) * f

    # ------------------------------------------------------- piezo vector

    def piezo_vector(self, u1: float, u2: float, v1: float, v2: float,
                     q: float) -> np.ndarray:
        """Actuation vector Ba (free DOFs):  M u'' + K u = ... + Ba V(t).

        Ba_i = -q * integral over the patch of (N_i,xx + N_i,yy) dA,
        the weak form of the uniform biaxial moment q V along the patch
        boundary (paper Eqs. 15-16, integrated by parts twice).
        Patch edges must coincide with mesh lines.
        """
        be = self.element.laplacian_integral()
        Ba = np.zeros(self.ndof)
        eps = 1e-9
        for j in range(self.ny):
            yc = (j + 0.5) * self.b
            if not (v1 - eps < yc < v2 + eps):
                continue
            for i in range(self.nx):
                xc = (i + 0.5) * self.a
                if not (u1 - eps < xc < u2 + eps):
                    continue
                Ba[self.element_dofs(i, j)] += -q * be
        return Ba[self.free]

    # ---------------------------------------------------------- analysis

    def modes(self, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
        """First n_modes natural frequencies [Hz] and mass-normalized
        mode-shape matrix Phi (free DOFs x n_modes)."""
        vals, vecs = spla.eigsh(self.Kff, k=n_modes, M=self.Mff, sigma=0.0,
                                which="LM")
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
        freqs = np.sqrt(np.maximum(vals, 0.0)) / (2.0 * np.pi)
        # mass-normalize
        for k in range(vecs.shape[1]):
            mk = float(vecs[:, k] @ (self.Mff @ vecs[:, k]))
            vecs[:, k] /= np.sqrt(mk)
        return freqs, vecs

    def static_solve(self, F: np.ndarray) -> np.ndarray:
        return spla.spsolve(self.Kff.tocsc(), F)
