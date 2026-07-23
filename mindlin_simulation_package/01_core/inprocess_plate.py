"""
inprocess_plate.py
==================
In-process (time-varying) workpiece dynamics for the Mindlin milling model.

As the cutter feeds along its path it removes material, so the plate's local
mass and stiffness — and therefore its natural frequencies, mode shapes and the
modal projections seen by the force / sensor / piezo — evolve **during** the
cut. This module adds that fidelity on top of the (uniform-thickness) Mindlin
`PlateModel`:

  * per-element thickness `h_elem[I, J]` (instead of a single `bp`);
  * a material-removal law `remove_swept_material(x_tool, ...)` that thins the
    elements the cutter has already passed (volume-consistent);
  * `rebuild()` re-assembles the physical FEM matrices with the current
    thickness field;
  * `instantaneous_modes()` gives the current natural frequencies / shapes
    (the "in-process" dynamics as a function of tool position);
  * `reduced_on_basis(V0)` projects the current physical matrices onto a FIXED
    modal basis `V0` — a reduced-order model whose coordinates keep their
    meaning while the structure changes, ready for the Newmark solver.

`InProcessPlate` subclasses `PlateModel`, so with no material removed it is
byte-for-byte the same model (same 519 Hz cantilever, same interface).
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from plate_model import PlateModel
from mindlin_q8 import (
    stiffness_matrix_M, mass_matrix_M, elem_dofs_M, clamped_edge_dofs,
)


class InProcessPlate(PlateModel):
    """Mindlin cantilever plate with per-element thickness and material removal."""

    # ------------------------------------------------------------------
    def _assemble(self):
        """Assemble Kg, Mg from the per-element thickness field `h_elem`."""
        if not hasattr(self, "h_elem"):
            # default: uniform thickness -> identical to PlateModel
            self.h_elem = np.full((self.N1, self.N2), self.bp)

        # element matrices depend only on the thickness value -> cache them
        self._KM_cache = {}

        def get_KM(h):
            key = round(float(h), 12)
            km = self._KM_cache.get(key)
            if km is None:
                Ke = stiffness_matrix_M(self.E, self.nu, self.kappa,
                                        self.lex, self.ley, h)
                Me = mass_matrix_M(self.rho, self.lex, self.ley, h)
                km = (Ke, Me)
                self._KM_cache[key] = km
            return km

        nelem = self.N1 * self.N2
        rows = np.zeros(nelem * 576, dtype=np.int64)
        cols = np.zeros_like(rows)
        Kdat = np.zeros(nelem * 576)
        Mdat = np.zeros(nelem * 576)
        idx = 0
        for J in range(1, self.N2 + 1):
            for I in range(1, self.N1 + 1):
                Ke, Me = get_KM(self.h_elem[I - 1, J - 1])
                DofE = elem_dofs_M(I, J, self.node_id)
                for a in range(24):
                    da = DofE[a]
                    for b in range(24):
                        rows[idx] = da
                        cols[idx] = DofE[b]
                        Kdat[idx] = Ke[a, b]
                        Mdat[idx] = Me[a, b]
                        idx += 1
        self.Kg = csr_matrix((Kdat, (rows, cols)), shape=(self.ndof, self.ndof))
        self.Mg = csr_matrix((Mdat, (rows, cols)), shape=(self.ndof, self.ndof))
        if self.verbose:
            rem = 100 * (1 - self.h_elem.mean() / self.bp)
            print(f"[InProcessPlate] assembled, mean thickness "
                  f"{self.h_elem.mean()*1e3:.3f} mm ({rem:+.2f}% removed)")

    # ------------------------------------------------------------------
    #  Element-centre coordinates (X_P, Z_P) helper
    # ------------------------------------------------------------------
    def _elem_center(self, I, J):
        return ((I - 0.5) * self.lex, (J - 0.5) * self.ley)

    # ------------------------------------------------------------------
    def new_pass(self):
        """Start a new cutter pass (resets the per-pass machined mask so each
        element is thinned at most once as the tool sweeps by)."""
        self._machined = np.zeros((self.N1, self.N2), dtype=bool)

    def machine_to(self, x_tool,
                   a_p=0.3e-3, a_e=0.1e-3,
                   z_top=None, h_min_frac=0.05):
        """
        Remove the material the cutter has swept up to feed position `x_tool`
        during the CURRENT pass (call `new_pass()` first).

        The cutter engages an axial band of height `a_p` at the top of the wall
        (`z in [z_top - a_p, z_top]`, default `z_top = H_P`) and a radial depth
        `a_e` through the thickness. Each element the tool has just passed
        (`x_center <= x_tool`, not yet machined this pass) that overlaps the
        band is thinned, volume-consistently, by

            dh = a_e * (overlap_height / ley)

        (removed cross-section `a_e * overlap_height` smeared over the element
        height `ley`). Thickness is floored at `h_min_frac * bp`. Each element
        is thinned at most once per pass. Call `rebuild()` afterwards to update
        the FEM matrices. Returns the number of elements machined this call.
        """
        if not hasattr(self, "_machined"):
            self.new_pass()
        if z_top is None:
            z_top = self.hp
        z_lo_band, z_hi_band = z_top - a_p, z_top
        h_floor = h_min_frac * self.bp
        n_hit = 0
        for J in range(1, self.N2 + 1):
            ze_lo, ze_hi = (J - 1) * self.ley, J * self.ley
            overlap = max(0.0, min(ze_hi, z_hi_band) - max(ze_lo, z_lo_band))
            if overlap <= 0:
                continue
            dh = a_e * (overlap / self.ley)
            for I in range(1, self.N1 + 1):
                if self._machined[I - 1, J - 1]:
                    continue
                xc, _ = self._elem_center(I, J)
                if xc <= x_tool:
                    self.h_elem[I - 1, J - 1] = max(
                        h_floor, self.h_elem[I - 1, J - 1] - dh)
                    self._machined[I - 1, J - 1] = True
                    n_hit += 1
        return n_hit

    def set_thickness_field(self, h_elem):
        """Set the full per-element thickness field (N1 x N2) and mark dirty."""
        self.h_elem = np.array(h_elem, dtype=float).reshape(self.N1, self.N2)

    # ------------------------------------------------------------------
    def rebuild(self):
        """Re-assemble physical matrices and re-apply the clamped BC."""
        self._assemble()
        self._apply_bc()

    # ------------------------------------------------------------------
    def instantaneous_modes(self, n=None):
        """
        Natural frequencies / mass-normalised shapes of the CURRENT structure
        (call rebuild() first if the thickness field changed). Returns
        (freq_hz (n,), V (ndof_free x n)).
        """
        n = n or self.n_modes
        w2, V = eigsh(self.K_free.tocsc(), k=n, M=self.M_free.tocsc(),
                      sigma=0, which='LM')
        order = np.argsort(np.real(w2))
        w2 = np.real(w2[order]); V = np.real(V[:, order])
        for k in range(n):
            V[:, k] /= np.sqrt(V[:, k] @ (self.M_free @ V[:, k]))
        return np.sqrt(np.abs(w2)) / (2 * np.pi), V

    # ------------------------------------------------------------------
    def reduced_on_basis(self, V0):
        """
        Reduced (modal) matrices of the CURRENT structure on a FIXED basis V0
        (ndof_free x m). Returns (M_r, K_r), each (m x m). Because the basis is
        fixed, the reduced coordinates keep their physical meaning as the
        structure is machined, so D_obs / H_Pe / Dp (computed once from V0)
        stay valid and no state re-projection is needed.
        """
        M_r = V0.T @ (self.M_free @ V0)
        K_r = V0.T @ (self.K_free @ V0)
        return M_r, K_r
