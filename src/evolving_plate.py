"""
evolving_plate.py
=================
Cantilever Reissner-Mindlin plate whose THICKNESS FIELD evolves as material is
removed along the tool path.

WHY THIS MODULE EXISTS
----------------------
In thin-walled milling the workpiece is not a fixed structure. A wall is
machined down from a thicker blank in successive radial layers and axial
steps; between the first and the last pass its natural frequencies can move
by a factor of two or more. A controller designed on the nominal (blank)
model is therefore mistuned over most of the operation, and a stability lobe
diagram computed once for the nominal workpiece describes none of it.

`PlateModel` assembles one element stiffness/mass pair for a single uniform
thickness and reuses it for every element. This module carries a PER-ELEMENT
thickness field instead, and exposes the same public interface as
`PlateModel` (Mp, Kp, Cp, omega_n, freq_n, V, precompute_Dp, set_observation,
add_piezo_patch, get_Dp_at) so every downstream module -- Newmark solver,
controllers, stability analysis, figures -- runs unchanged on it.

EXACT THICKNESS SCALING
-----------------------
For the Q8 element of `mindlin_q8.py` the thickness enters analytically:

    K_e(h) = h*(Kf + Ks),   Hf ∝ E h^2/(12(1-nu^2)),   Hs ∝ E kappa/(2(1+nu))
           = h^3 * KB1  +  h * KS1
    M_e(h) = rho*h * ∫ N^T diag(1, h^2/12, h^2/12) N dS
           = h * MT1  +  h^3 * MR1

so the four "unit" matrices KB1, KS1, MT1, MR1 are computed ONCE and each
element's matrices follow by scalar scaling. Re-assembling the whole plate
after a material-removal step therefore costs a sparse rebuild, not 720
element integrations.

The scaling is exact, not an approximation: it is the same integrand with h
factored out, valid because the mesh is uniform and rectangular so every
element shares the same Jacobian.

MATERIAL REMOVAL MODEL
----------------------
`remove_material(x0, x1, z0, z1, delta)` subtracts `delta` from the thickness
of every element whose centroid lies in the window, flooring at `h_min`. An
element is the finest resolution at which removal is represented, so the
mesh must be fine enough that one axial step spans at least one element row;
`check_axial_resolution` warns when it does not.

This is a structural-modification model: it captures the stiffness and
inertia change of the remaining material. It does NOT model residual-stress
redistribution or machining-induced deformation, which would require an
elasto-plastic analysis; that limitation is stated rather than hidden.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from mindlin_q8 import (
    matrix_der_M, shape_function_M, jacobian_M, _node_coor, _G2,
    build_node_map, elem_dofs_M, clamped_edge_dofs,
    shape_at_point_M, piezo_moment_load_M,
)


# ---------------------------------------------------------------------
# Unit-thickness element matrices
# ---------------------------------------------------------------------
def unit_element_matrices(E, nu, kappa, rho, lex, ley):
    """
    Return (KB1, KS1, MT1, MR1) such that, for thickness h,

        K_e(h) = h**3 * KB1 + h * KS1
        M_e(h) = h * MT1 + h**3 * MR1

    Verified against mindlin_q8.stiffness_matrix_M / mass_matrix_M in
    tests/verify_evolving.py.
    """
    nc = _node_coor(lex, ley)
    D0 = np.array([[1.0, nu, 0.0],
                   [nu, 1.0, 0.0],
                   [0.0, 0.0, (1 - nu) / 2]])
    Hf1 = E / (12 * (1 - nu ** 2)) * D0          # Hf = h^2 * Hf1
    Hs1 = E * kappa / (2 * (1 + nu)) * np.eye(2)  # Hs = Hs1

    KB1 = np.zeros((24, 24))
    KS1 = np.zeros((24, 24))
    MT1 = np.zeros((24, 24))
    MR1 = np.zeros((24, 24))

    for I in range(2):
        for J in range(2):
            Bf, Bs = matrix_der_M(nc, _G2[I], _G2[J])
            N, _, _ = shape_function_M(_G2[I], _G2[J])
            Jac, _ = jacobian_M(nc, _G2[I], _G2[J])
            detJ = np.linalg.det(Jac)
            KB1 += Bf.T @ Hf1 @ Bf * detJ
            KS1 += Bs.T @ Hs1 @ Bs * detJ
            MT1 += rho * (N.T @ np.diag([1.0, 0.0, 0.0]) @ N) * detJ
            MR1 += rho / 12.0 * (N.T @ np.diag([0.0, 1.0, 1.0]) @ N) * detJ

    return KB1, KS1, MT1, MR1


# ---------------------------------------------------------------------
class EvolvingPlateModel:
    """
    Cantilever Mindlin plate with a per-element thickness field.

    Public interface matches `PlateModel`, plus:
        thickness      : (N2, N1) current per-element thickness [m]
        remove_material(x0, x1, z0, z1, delta)
        solve_modes()  : re-assemble and re-run the modal analysis
        removed_volume : cumulative removed volume [m^3]
    """

    def __init__(self, lp, hp, h0, rho, E, nu,
                 N1=30, N2=24, n_modes=3, zeta_modes=None,
                 kappa=5.0 / 6.0, h_min=0.2e-3, verbose=True):
        self.lp, self.hp = lp, hp
        self.rho, self.E, self.nu = rho, E, nu
        self.kappa = kappa
        self.N1, self.N2 = N1, N2
        self.lex, self.ley = lp / N1, hp / N2
        self.n_modes = n_modes
        self.h_min = h_min
        self.verbose = verbose

        self.node_id, self.ntot = build_node_map(N1, N2)
        self.ndof = 3 * self.ntot

        if zeta_modes is None:
            zeta_modes = [0.0031, 0.0017, 0.0027]
        self.zeta_modes = np.asarray(zeta_modes[:n_modes], float)

        # thickness field, indexed [J, I] with I along lp, J along hp
        h0 = np.asarray(h0, float)
        self.thickness = (np.full((N2, N1), float(h0)) if h0.ndim == 0
                          else h0.copy())
        self.h_nominal = float(np.max(self.thickness))
        self._h_initial = self.thickness.copy()

        self.KB1, self.KS1, self.MT1, self.MR1 = unit_element_matrices(
            E, nu, kappa, rho, self.lex, self.ley)

        # element -> global DOF map, built once
        self._dof_map = np.empty((N2, N1, 24), dtype=np.int64)
        for J in range(1, N2 + 1):
            for I in range(1, N1 + 1):
                self._dof_map[J - 1, I - 1] = elem_dofs_M(I, J, self.node_id)

        # sparse assembly index pattern, built once
        nelem = N1 * N2
        self._rows = np.repeat(self._dof_map.reshape(nelem, 24), 24, axis=1).ravel()
        self._cols = np.tile(self._dof_map.reshape(nelem, 24), (1, 24)).ravel()

        DOFb = clamped_edge_dofs(self.node_id, edge="bottom")
        self.DOFb = DOFb
        self.DOFf = np.setdiff1d(np.arange(self.ndof), DOFb)

        self._piezo_args = None
        self._obs_args = None
        self._dp_args = None
        self.removed_volume = 0.0

        self.solve_modes()

    # -----------------------------------------------------------------
    def check_axial_resolution(self, ap):
        """Warn if one axial step is finer than one element row."""
        if ap < self.ley:
            print(f"[EvolvingPlate] WARNING: axial step a_p = {ap*1e3:.3f} mm is "
                  f"smaller than one element row ({self.ley*1e3:.3f} mm). "
                  f"Material removal cannot be resolved; increase N2 to at "
                  f"least {int(np.ceil(self.hp/ap))}.")
            return False
        return True

    # -----------------------------------------------------------------
    def remove_material(self, x0, x1, z0, z1, delta):
        """
        Subtract `delta` [m] of thickness over the window [x0,x1] x [z0,z1].

        Elements are selected by centroid. Thickness is floored at h_min.
        Returns the volume actually removed [m^3].
        """
        xc = (np.arange(self.N1) + 0.5) * self.lex
        zc = (np.arange(self.N2) + 0.5) * self.ley
        mi = (xc >= x0) & (xc <= x1)
        mj = (zc >= z0) & (zc <= z1)
        if not mi.any() or not mj.any():
            return 0.0

        sel = np.ix_(mj, mi)
        before = self.thickness[sel].copy()
        after = np.maximum(before - delta, self.h_min)
        self.thickness[sel] = after
        vol = float(np.sum(before - after) * self.lex * self.ley)
        self.removed_volume += vol
        return vol

    def reset_thickness(self):
        self.thickness = self._h_initial.copy()
        self.removed_volume = 0.0

    # -----------------------------------------------------------------
    def _assemble(self):
        h = self.thickness.reshape(-1)                      # (nelem,)
        h3 = h ** 3
        Ke = h3[:, None, None] * self.KB1 + h[:, None, None] * self.KS1
        Me = h[:, None, None] * self.MT1 + h3[:, None, None] * self.MR1
        Kg = csr_matrix((Ke.ravel(), (self._rows, self._cols)),
                        shape=(self.ndof, self.ndof))
        Mg = csr_matrix((Me.ravel(), (self._rows, self._cols)),
                        shape=(self.ndof, self.ndof))
        f = self.DOFf
        self.K_free = Kg[f, :][:, f]
        self.M_free = Mg[f, :][:, f]

    # -----------------------------------------------------------------
    def solve_modes(self):
        """Re-assemble and re-run the modal analysis on the current thickness."""
        self._assemble()
        w2, V = eigsh(self.K_free.tocsc(), k=self.n_modes,
                      M=self.M_free.tocsc(), sigma=0, which='LM')
        order = np.argsort(np.real(w2))
        w2 = np.real(w2[order])
        V = np.real(V[:, order])

        for k in range(self.n_modes):
            V[:, k] /= np.sqrt(V[:, k] @ (self.M_free @ V[:, k]))

        # Sign convention: fix the sign of each mode shape by its value at the
        # free top-right corner. eigsh returns eigenvectors up to a sign, and an
        # unfixed sign silently flips D_obs and H_Pe_modal between two models
        # built from the same data -- which inverts the feedback sign of a
        # controller designed on one and run on the other.
        N_ref = shape_at_point_M(self.lp, self.hp, self.N1, self.N2,
                                 self.lex, self.ley, self.node_id, self.ndof)
        ref = N_ref[self.DOFf] @ V
        for k in range(self.n_modes):
            if ref[k] < 0:
                V[:, k] *= -1.0

        self.omega_n = np.sqrt(np.abs(w2))
        self.freq_n = self.omega_n / (2 * np.pi)
        self.V = V
        self.Mp = np.eye(self.n_modes)
        self.Kp = np.diag(self.omega_n ** 2)
        self.Cp = np.diag(2 * self.zeta_modes * self.omega_n)

        # refresh anything that depends on the mode shapes
        if self._dp_args is not None:
            self.precompute_Dp(*self._dp_args, _refresh=True)
        if self._obs_args is not None:
            self.set_observation(*self._obs_args, _refresh=True)
        if self._piezo_args is not None:
            self.add_piezo_patch(*self._piezo_args, _refresh=True)

        if self.verbose:
            print(f"[EvolvingPlate] h in [{self.thickness.min()*1e3:.3f}, "
                  f"{self.thickness.max()*1e3:.3f}] mm   "
                  f"f = {np.round(self.freq_n, 1)} Hz")

    # -----------------------------------------------------------------
    # PlateModel-compatible interface
    # -----------------------------------------------------------------
    def precompute_Dp(self, zp_pos, n_pos=2001, _refresh=False):
        if not _refresh:
            self._dp_args = (zp_pos, n_pos)
        xp = np.linspace(0, self.lp, n_pos)
        Dp = np.zeros((self.n_modes, n_pos))
        DpT = np.zeros((self.n_modes, self.n_modes, n_pos))
        for k in range(n_pos):
            Nv = shape_at_point_M(xp[k], zp_pos, self.N1, self.N2,
                                  self.lex, self.ley, self.node_id, self.ndof)
            d = Nv[self.DOFf] @ self.V
            Dp[:, k] = d
            DpT[:, :, k] = np.outer(d, d)
        self.xp_array, self.Dp_array, self.DpT_Dp_array = xp, Dp, DpT
        self.zp_pos = zp_pos

    def set_observation(self, x_obs, z_obs, _refresh=False):
        if not _refresh:
            self._obs_args = (x_obs, z_obs)
        Nv = shape_at_point_M(x_obs, z_obs, self.N1, self.N2,
                              self.lex, self.ley, self.node_id, self.ndof)
        self.D_obs = Nv[self.DOFf] @ self.V
        self.x_obs, self.z_obs = x_obs, z_obs

    def add_piezo_patch(self, xP1, xP2, zP1, zP2,
                        d31, h_Pa, E_Pe, nu_Pe, _refresh=False):
        if not _refresh:
            self._piezo_args = (xP1, xP2, zP1, zP2, d31, h_Pa, E_Pe, nu_Pe)
        # moment arm uses the LOCAL plate thickness under the patch, which
        # changes if the patch region is ever machined
        xc = (np.arange(self.N1) + 0.5) * self.lex
        zc = (np.arange(self.N2) + 0.5) * self.ley
        mi = (xc >= xP1) & (xc <= xP2)
        mj = (zc >= zP1) & (zc <= zP2)
        h_loc = (float(np.mean(self.thickness[np.ix_(mj, mi)]))
                 if mi.any() and mj.any() else self.h_nominal)
        m_piezo = -E_Pe * d31 * (h_loc + h_Pa) / (2 * (1 - nu_Pe))
        g = piezo_moment_load_M(xP1, xP2, zP1, zP2, self.N1, self.N2,
                                self.lex, self.ley, self.node_id, self.ndof)
        self.H_Pe_modal = self.V.T @ (m_piezo * g)[self.DOFf]
        self.patch = dict(xP1=xP1, xP2=xP2, zP1=zP1, zP2=zP2,
                          m_piezo=m_piezo, h_local=h_loc)

    def get_Dp_at(self, kp):
        return self.Dp_array[:, kp], self.DpT_Dp_array[:, :, kp]
