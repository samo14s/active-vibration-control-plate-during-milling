"""
material_removal.py
===================
Material-removal-aware, time-varying FEM of the cantilever plate for a MULTI-PASS
finishing sequence — the physically accurate replacement for the phenomenological
uniform frequency-scale drift (`PlateModel.perturbed_copy` / the solver's
`omega_scale_t`).

WHY THIS EXISTS (the honest physics — verified in `phys_check`, see docs/):
  * A single pass at a_p = 0.3 mm removes a 0.1x0.3x100 mm strip = 0.0094 % of the
    volume -> frequency drift ~0.000 %/mode, MAC = 1.0. "Along one tool path" the
    plant is essentially CONSTANT; the dominant within-pass variation is the
    spatial mode-shape sampling Dp(x), which the base model already captures.
  * Uniform full-face thinning gives omega ∝ h EXACTLY (a plate bending result:
    D ∝ h^3, mass ∝ h, so omega ∝ sqrt(h^3/h) = h), so the old uniform-scale
    heuristic is the correct BETWEEN-pass model — the article's measured 9-17 %
    drift corresponds to 4-7 full 0.1 mm finishing layers (wall 4 -> 3.3-3.6 mm).
  * The genuinely new physics is NON-UNIFORM removal: when only part of the height
    is thinned, the per-mode drift acquires OPPOSITE signs across modes
    (+0.87/+0.15/-0.57 % for a top-band cut) and the mode shapes reshape
    (MAC < 1) — a single frequency scale cannot represent this, which is exactly
    what makes real-time identification necessary.

EFFICIENCY. The Kirchhoff element matrices scale as K_e = h^3 * K_unit and
M_e = h * M_unit (uniform mesh), so ONE unit element matrix is precomputed and
each snapshot only rescales per element — reassembly is cheap, no re-integration.

MODE TRACKING. As the thickness field becomes non-uniform the eigenvectors genuinely
change, and a fresh eigensolve returns modes in arbitrary order and sign. Each
snapshot is MAC-matched and sign-aligned against a reference eigenbasis so that
mode k stays mode k with a consistent sign — this preserves feedback-sign
consistency for a controller designed on an earlier snapshot (the property that
`perturbed_copy` got for free by sharing eigenvectors, now earned honestly).
"""
import copy as _copy
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from kirchhoff_q4 import (stiffness_matrix_K, mass_matrix_K,
                          shape_at_point, laplace_n_patch)


class MillingWorkpiece:
    """Cantilever plate whose per-element thickness field decreases as material is
    removed along a finishing sequence; recomputes the modal model on demand."""

    def __init__(self, lp, hp, bp, rho, E, nu, N1=30, N2=24,
                 n_modes=5, zeta_modes=None, verbose=True):
        self.lp, self.hp, self.bp = lp, hp, bp
        self.rho, self.E, self.nu = rho, E, nu
        self.N1, self.N2 = N1, N2
        self.n1, self.n2 = N1 + 1, N2 + 1
        self.lex, self.ley = lp / N1, hp / N2
        self.ndof = 3 * self.n1 * self.n2
        self.n_modes = n_modes
        self.verbose = verbose
        if zeta_modes is None:
            zeta_modes = [0.0031, 0.0017, 0.0027, 0.0056, 0.0035]
        self.zeta_modes = np.array(zeta_modes[:n_modes], float)

        # unit-thickness element matrices (h factored out): K_e = h^3 K1, M_e = h M1
        self.K1 = stiffness_matrix_K(E, nu, self.lex, self.ley, 1.0)   # actually h^3
        self.M1 = mass_matrix_K(rho, self.lex, self.ley, 1.0)          # actually h^1
        # NOTE: stiffness_matrix_K(...,h) = h^3 * stiffness_matrix_K(...,1); confirm
        # numerically once (guards against a future change to the element routine).
        Kh = stiffness_matrix_K(E, nu, self.lex, self.ley, 2.0)
        assert np.allclose(Kh, 8.0 * self.K1, rtol=1e-9), "K not ∝ h^3"
        Mh = mass_matrix_K(rho, self.lex, self.ley, 2.0)
        assert np.allclose(Mh, 2.0 * self.M1, rtol=1e-9), "M not ∝ h^1"

        # element DOF maps (global, pre-BC)
        self.edofs = np.zeros((N2 * N1, 12), dtype=np.int64)
        e = 0
        for J in range(1, N2 + 1):
            for I in range(1, N1 + 1):
                self.edofs[e] = np.r_[
                    3*(J-1)*self.n1 + 3*(I-1) + np.arange(6),
                    3*J*self.n1 + 3*(I-1) + np.arange(3, 6),
                    3*J*self.n1 + 3*(I-1) + np.arange(3)]
                e += 1
        # constant COO index pattern (same for K and M)
        aa = np.repeat(np.arange(12), 12)
        bb = np.tile(np.arange(12), 12)
        self._rows = self.edofs[:, aa].ravel()
        self._cols = self.edofs[:, bb].ravel()
        self._K1flat = self.K1.ravel()
        self._M1flat = self.M1.ravel()

        # BC: clamp bottom edge (J=1 nodes)
        DOFb = np.arange(3 * self.n1)
        self.DOFf = np.setdiff1d(np.arange(self.ndof), DOFb)

        # element centroid heights (for pass geometry) and column index
        self.ez = (np.repeat(np.arange(N2), N1) + 0.5) * self.ley   # (nelem,)
        self.ex = (np.tile(np.arange(N1), N2) + 0.5) * self.lex     # (nelem,) col x
        self.h_field = np.full(N2 * N1, bp, float)                  # thickness/elem

        self._patch = None
        self._x_obs = None
        self._z_obs = None
        self._ref_V = None          # reference eigenbasis for MAC tracking

    # ------------------------------------------------------------------
    def _assemble_free(self):
        h = np.clip(self.h_field, 1e-6, None)
        Kdat = (h**3)[:, None] * self._K1flat[None, :]
        Mdat = h[:, None] * self._M1flat[None, :]
        Kg = csr_matrix((Kdat.ravel(), (self._rows, self._cols)),
                        shape=(self.ndof, self.ndof))
        Mg = csr_matrix((Mdat.ravel(), (self._rows, self._cols)),
                        shape=(self.ndof, self.ndof))
        return (Kg[self.DOFf, :][:, self.DOFf],
                Mg[self.DOFf, :][:, self.DOFf])

    # ------------------------------------------------------------------
    def remove_layer_band(self, z_lo, z_hi, a_e):
        """Remove a radial bite a_e from the wall thickness over the height band
        [z_lo, z_hi] (elements whose centroid falls in the band), clamped >= 0.
        Models one finishing pass engaging that band (end-of-pass, all-x snapshot)."""
        sel = (self.ez >= z_lo) & (self.ez < z_hi)
        self.h_field[sel] = np.maximum(self.h_field[sel] - a_e, 1e-4)

    def begin_pass(self):
        """Snapshot the current thickness field as the PRE-PASS baseline for the
        x-resolved moving front (so repeated remove_moving_front calls during a pass
        are idempotent and each column is thinned by exactly a_e once per pass)."""
        self._front_baseline = self.h_field.copy()

    def remove_moving_front(self, x_tool, a_p, a_e):
        """Physically accurate X-RESOLVED moving thinning front (P7): the tool,
        currently at feed position x_tool, has thinned by the radial bite a_e the
        material it has already passed (columns with centroid x < x_tool) in the
        engaged top band [HP - a_p, HP]. Material ahead of the tool is still virgin
        (full thickness), the channel behind is thinned by exactly a_e. This replaces
        the end-of-pass all-x `remove_layer_band` with the correct within-pass
        geometry.

        IDEMPOTENT within a pass: the thickness of a behind-tool column is set to
        (pre-pass baseline - a_e), NOT decremented, so calling this repeatedly as
        x_tool advances (the intended event-driven usage) thins each column exactly
        once. Call begin_pass() at the start of each pass (auto-snapshotted on first
        use if omitted). Across passes, begin_pass() re-baselines, so a_e accumulates
        pass-to-pass correctly.

        NOTE (one-face removal): peripheral milling removes material from ONE face,
        offsetting the neutral surface by ~a_e/2 and inducing membrane-bending
        coupling that symmetric thickness-scaling (h^3 stiffness, h mass) ignores.
        This is ~2.5% eccentricity at a_e = 0.1 mm (negligible) but ~12.5% for a
        rough a_e = 0.5 mm — a documented modelling caveat, not implemented here.
        """
        if not hasattr(self, '_front_baseline'):
            self._front_baseline = self.h_field.copy()
        band = self.ez >= (self.hp - a_p)
        behind = band & (self.ex < x_tool)
        self.h_field[behind] = np.maximum(self._front_baseline[behind] - a_e, 1e-4)

    def thickness_stats(self):
        return dict(min=float(self.h_field.min()), max=float(self.h_field.max()),
                    mean=float(self.h_field.mean()),
                    removed_frac=float(1 - self.h_field.mean() / self.bp))

    # ------------------------------------------------------------------
    def solve_modes(self, track=True):
        """Eigensolve at the current thickness field; MAC-match + sign-align to the
        reference basis (first call sets the reference). Populates omega_n, freq_n,
        V (mass-normalised, free DOFs), Mp/Kp/Cp."""
        Kf, Mf = self._assemble_free()
        w2, V = eigsh(Kf.tocsc(), k=self.n_modes, M=Mf.tocsc(),
                      sigma=0, which='LM')
        order = np.argsort(w2.real)
        w2 = w2.real[order]
        V = V.real[:, order]
        for k in range(self.n_modes):
            V[:, k] /= np.sqrt(V[:, k] @ (Mf @ V[:, k]))

        if track and self._ref_V is not None:
            V, w2 = self._mac_align(V, w2, Mf)
        if self._ref_V is None:
            self._ref_V = V.copy()
            self._ref_Mf = Mf

        self.omega_n = np.sqrt(np.clip(w2, 0, None))
        self.freq_n = self.omega_n / (2 * np.pi)
        self.V = V
        self.Mp = np.eye(self.n_modes)
        self.Kp = np.diag(self.omega_n**2)
        self.Cp = np.diag(2 * self.zeta_modes * self.omega_n)
        # refresh observation / actuation / tool projections on the new shapes
        if self._x_obs is not None:
            self._set_observation()
        if self._patch is not None:
            self._recompute_piezo()
        return self.freq_n.copy()

    def _mac_align(self, V, w2, Mf):
        """Reorder new modes to best-match the reference basis (MAC via M inner
        product on the common free-DOF space) and flip signs. Robust to the
        occasional eigsolver mode swap under strong reshaping."""
        Vr = self._ref_V
        # MAC[i,j] = (Vr_i' M V_j)^2 / ((Vr_i'M Vr_i)(V_j'M V_j)); mass-normalised
        # so denominators are 1.
        G = Vr.T @ (Mf @ V)              # (n x n)
        MAC = G**2
        used = set()
        perm = np.zeros(self.n_modes, dtype=int)
        for i in range(self.n_modes):     # greedy assignment, strongest first
            j = int(np.argmax([MAC[i, jj] if jj not in used else -1
                               for jj in range(self.n_modes)]))
            perm[i] = j
            used.add(j)
        Vp = V[:, perm]
        w2p = w2[perm]
        Gp = np.diag(Vr.T @ (Mf @ Vp))
        signs = np.sign(Gp)
        signs[signs == 0] = 1.0
        Vp = Vp * signs[None, :]
        return Vp, w2p

    # ------------------------------------------------------------------
    def set_observation(self, x_obs, z_obs):
        self._x_obs, self._z_obs = x_obs, z_obs
        if hasattr(self, 'V'):
            self._set_observation()

    def _set_observation(self):
        N = shape_at_point(self._x_obs, self._z_obs, self.N1, self.N2,
                           self.lex, self.ley, self.n1)
        self.D_obs = N[self.DOFf] @ self.V

    def add_piezo_patch(self, xP1, xP2, zP1, zP2, d31, h_Pa, E_Pe, nu_Pe):
        self._patch = dict(xP1=xP1, xP2=xP2, zP1=zP1, zP2=zP2,
                           d31=d31, h_Pa=h_Pa, E_Pe=E_Pe, nu_Pe=nu_Pe)
        self._g_lap = laplace_n_patch(xP1, xP2, zP1, zP2, self.N1, self.N2,
                                      self.lex, self.ley, self.n1, self.ndof)
        if hasattr(self, 'V'):
            self._recompute_piezo()

    def _recompute_piezo(self):
        """Eq. (15) coupling with the CURRENT mean plate thickness under the patch
        (as the wall thins the coupling coefficient drifts too — modelled honestly)."""
        p = self._patch
        # mean thickness of elements whose centroid lies under the patch footprint
        ex = (np.tile(np.arange(self.N1), self.N2) + 0.5) * self.lex
        under = ((ex >= p['xP1']) & (ex < p['xP2'])
                 & (self.ez >= p['zP1']) & (self.ez < p['zP2']))
        bp_loc = float(self.h_field[under].mean()) if under.any() else self.bp
        E_P, nu_P = self.E, self.nu
        E_Pe, nu_Pe, d31, h_Pa = p['E_Pe'], p['nu_Pe'], p['d31'], p['h_Pa']
        P_M = (-(E_Pe / E_P) * (1 - nu_P**2) / (1 - nu_Pe**2)
               * 3 * h_Pa * bp_loc * (bp_loc + h_Pa)
               / (0.5 * bp_loc**3 + 4 * h_Pa**3 + 3 * bp_loc * h_Pa**2))
        C_P0 = (-(1.0 / 6.0) * (1 + nu_Pe) / (1 - nu_P) * E_P * bp_loc**2 * P_M
                / (1 + nu_P - (1 + nu_Pe) * P_M))
        m_piezo = -C_P0 * d31 / h_Pa
        H_phys = m_piezo * self._g_lap
        self.H_Pe_modal = self.V.T @ H_phys[self.DOFf]
        self.bp_under_patch = bp_loc

    # ------------------------------------------------------------------
    def precompute_Dp(self, zp_pos, n_pos=2001):
        """Tool-contact mode-shape lookup Dp(x) at height zp_pos on the CURRENT
        modal basis."""
        xp = np.linspace(0, self.lp, n_pos)
        Dp = np.zeros((self.n_modes, n_pos))
        DpDp = np.zeros((self.n_modes, self.n_modes, n_pos))
        for kp in range(n_pos):
            Nv = shape_at_point(xp[kp], zp_pos, self.N1, self.N2,
                                self.lex, self.ley, self.n1)
            d = Nv[self.DOFf] @ self.V
            Dp[:, kp] = d
            DpDp[:, :, kp] = np.outer(d, d)
        self.xp_array = xp
        self.Dp_array = Dp
        self.DpT_Dp_array = DpDp
        self.zp_pos = zp_pos

    def get_Dp_at(self, kp):
        return self.Dp_array[:, kp], self.DpT_Dp_array[:, :, kp]

    # ------------------------------------------------------------------
    def truncated_view(self, n_keep):
        """Controller design view of the first n_keep modes at the current snapshot
        (same interface as PlateModel.truncated_view)."""
        n = int(min(n_keep, self.n_modes))
        return SimpleNamespace(
            n_modes=n, Mp=np.eye(n),
            Kp=np.diag(self.omega_n[:n]**2),
            Cp=np.diag(2.0 * self.zeta_modes[:n] * self.omega_n[:n]),
            omega_n=self.omega_n[:n].copy(),
            freq_n=self.freq_n[:n].copy(),
            D_obs=self.D_obs[:n].copy(),
            H_Pe_modal=self.H_Pe_modal[:n].copy())

    def snapshot_plant(self):
        """A lightweight, solver-compatible plant object frozen at the current
        snapshot (carries Dp_array, D_obs, H_Pe_modal, Mp/Kp/Cp, omega_n)."""
        return SimpleNamespace(
            n_modes=self.n_modes, Mp=self.Mp.copy(), Kp=self.Kp.copy(),
            Cp=self.Cp.copy(), omega_n=self.omega_n.copy(),
            freq_n=self.freq_n.copy(), zeta_modes=self.zeta_modes.copy(),
            D_obs=self.D_obs.copy(), H_Pe_modal=self.H_Pe_modal.copy(),
            Dp_array=self.Dp_array, DpT_Dp_array=self.DpT_Dp_array,
            xp_array=self.xp_array, zp_pos=self.zp_pos,
            get_Dp_at=self.get_Dp_at)


# =====================================================================
#  Finishing-sequence generator
# =====================================================================
def finishing_sequence(wp, n_layers, a_e, a_p, zp_for_pass=None, verbose=True):
    """
    Simulate a finishing sequence: `n_layers` full layers, each descending the
    height in bands of engaged height a_p, removing a radial bite a_e per pass.
    Returns a list of per-pass snapshots (dicts) with the modal model AFTER that
    pass's material removal, plus the tool height used for the pass.

    This produces a genuinely NON-UNIFORM thickness field mid-layer (top thinned,
    bottom not), so the per-mode frequency drift is non-uniform and the mode shapes
    reshape — the regime that makes online identification necessary.
    """
    HP = wp.hp
    n_bands = int(np.ceil(HP / a_p))
    band_edges = np.linspace(HP, 0.0, n_bands + 1)     # top-down
    # establish the reference basis on the pristine plate
    wp.solve_modes(track=True)
    snaps = []
    pass_id = 0
    f0 = wp.freq_n.copy()
    for layer in range(n_layers):
        for b in range(n_bands):
            z_hi, z_lo = band_edges[b], band_edges[b + 1]
            wp.remove_layer_band(z_lo, z_hi, a_e)
            zc = 0.5 * (z_lo + z_hi)
            zp = zp_for_pass(zc) if zp_for_pass else min(zc + a_p / 2, HP - 1e-4)
            wp.solve_modes(track=True)
            wp.precompute_Dp(zp_pos=min(zp, HP - 1e-4), n_pos=2001)
            st = wp.thickness_stats()
            snaps.append(dict(
                pass_id=pass_id, layer=layer, band=b,
                z_lo=z_lo, z_hi=z_hi, zp=zp,
                freq=wp.freq_n.copy(), omega=wp.omega_n.copy(),
                removed_frac=st['removed_frac'], h_min=st['min'],
                plant=wp.snapshot_plant()))
            if verbose:
                drift = (wp.freq_n / f0 - 1) * 100
                print(f"  pass {pass_id:3d} L{layer} b{b:2d} "
                      f"z[{z_lo*1e3:5.1f},{z_hi*1e3:5.1f}]mm  "
                      f"h_min={st['min']*1e3:.3f}mm removed={st['removed_frac']*100:5.2f}%  "
                      f"f=[{wp.freq_n[0]:.0f},{wp.freq_n[1]:.0f},{wp.freq_n[2]:.0f}]Hz "
                      f"drift=[{drift[0]:+.1f},{drift[1]:+.1f},{drift[2]:+.1f}]%")
            pass_id += 1
    return snaps
