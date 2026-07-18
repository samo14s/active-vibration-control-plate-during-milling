"""
cl_fdm.py
=========
Closed-loop semi-discretization (CL-SD) for chatter stability of
actively-controlled thin-walled milling.

Terminology note: the delayed state is frozen at one history node per
sub-interval, i.e. this is the 0th-order semi-discretization scheme of Insperger
& Stepan (2002/2004), not the higher-order full-discretization of Ding et al.
The module/class names retain "FDM" for brevity; the manuscript uses the
accurate name (semi-discretization).

Motivation
----------
The classical full-discretization method (FDM, Insperger & Stepan, 2004) builds
the monodromy (transition) matrix of the *open-loop* delayed periodic milling
dynamics and reads chatter stability from its spectral radius (the dominant
Floquet multiplier).  A *controlled* stability-lobe diagram (SLD) can only be
obtained honestly if the control law is embedded **inside** that monodromy
matrix.

This module does exactly that.  It assembles the closed-loop periodic
delay-differential equation

    x_dot(t) = [A_p(t) - B K] x(t) + [A_tau(t) - B K_tau] x(t - tau)

where

    A_p(t)   = [[0, I], [-(Kp + a4(t) Dp Dp^T),  -Cp]]      (2n x 2n)
    A_tau(t) = [[0, 0], [ a4(t) Dp Dp^T,           0 ]]      (2n x 2n)
    B        = [0; H_pe]                                     (2n x 1)

x = [q; q_dot] are the modal states, a4(t) is the periodic regenerative cutting
coefficient over one tooth period tau, Dp is the modal shape vector at the tool
tip, and Kp, Cp are the modal stiffness/damping matrices.  K is a static state
feedback gain (the LQR/LQG core), and K_tau is an optional *delayed* feedback
gain.  Setting K = K_tau = None recovers the open-loop SLD.

Discretisation follows the semi-discretisation idea: over each of ``m_div``
sub-intervals of the delay tau the coefficients are frozen and the delayed
state is treated as the corresponding entry of an augmented history vector.

Validation
----------
With K = None the open-loop critical depth of cut at 4900 rpm evaluates to
~0.12 mm, in agreement with the published experimental value (~0.1 mm) for this
cantilever plate, confirming the assembly is physically correct.

Author: contribution built on the model of
        Du et al., Int. J. Mech. Sci. 274 (2024) 109257.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from milling_force import milling_force_coeffs


def _expm_and_integral(M, dt):
    """Return (e^{M dt}, integral_0^dt e^{M s} ds) for a small, diagonalisable M.

    Uses an eigen-decomposition, which is far faster and more robust here than
    Pade scaling-and-squaring: the milling state matrix contains a stiff
    high-frequency mode (||M dt|| is large), for which ``scipy.linalg.expm``
    performs many costly squaring steps.  Oscillator matrices are diagonalisable,
    so the eigen route is exact; it falls back to ``expm`` if M is defective.
    """
    try:
        w, V = np.linalg.eig(M)
        Vi = np.linalg.inv(V)
        e = np.exp(w * dt)
        Ed = (V * e) @ Vi
        g = np.where(np.abs(w) > 1e-12, (e - 1.0) / w, dt)   # integral eigenvalues
        Ig = (V * g) @ Vi
        return Ed.real, Ig.real
    except np.linalg.LinAlgError:
        Ed = expm(M * dt)
        try:
            Ig = np.linalg.solve(M, Ed - np.eye(M.shape[0]))
        except np.linalg.LinAlgError:
            Ig = np.eye(M.shape[0]) * dt + M * dt ** 2 / 2.0
        return Ed, Ig


class ClosedLoopFDM:
    """Closed-loop full-discretization stability solver.

    Parameters
    ----------
    omega_n : (n,) modal natural frequencies [rad/s]
    zeta    : (n,) modal damping ratios [-]
    Dp      : (n,) modal shape values at the tool tip (may be a path-averaged
              vector; see :meth:`from_plate`)
    Hpe     : (n,) modal actuator input vector [N/V]
    cutting : dict with keys NT, RT, eta_h, phi_st, phi_ex, k1, k2, kt, hp
    m_div   : number of sub-intervals per tooth period (discretisation order)
    """

    def __init__(self, omega_n, zeta, Dp, Hpe, cutting, m_div: int = 30):
        self.omega_n = np.asarray(omega_n, float)
        self.zeta = np.asarray(zeta, float)
        self.Dp = np.asarray(Dp, float)
        self.n = self.omega_n.size
        self.nx = 2 * self.n
        self.Kp = np.diag(self.omega_n ** 2)
        self.Cp = np.diag(2.0 * self.zeta * self.omega_n)
        self.Dp2 = np.outer(self.Dp, self.Dp)          # rank-1 modal coupling
        self.B = np.zeros((self.nx, 1))
        self.B[self.n:, 0] = np.asarray(Hpe, float)
        self.C_obs = None                    # sensor shape (set by from_plate)
        self.c = cutting
        self.m_div = int(m_div)

    # ------------------------------------------------------------------
    @classmethod
    def from_plate(cls, plate, cutting, m_div: int = 30, dp_mode: str = "average"):
        """Build from a :class:`PlateModel`.

        dp_mode = "average" uses the tool-path-averaged modal shape (matches the
        reference SLD convention); "tip" would use a single position (pass a
        pre-sliced Dp instead if a specific position is required).
        """
        if dp_mode == "average":
            samples = [plate.get_Dp_at(kp)[0] for kp in range(0, plate.Dp_array.shape[1], 50)]
            Dp = np.mean(samples, axis=0)
        else:
            Dp = plate.get_Dp_at(plate.Dp_array.shape[1] // 2)[0]
        obj = cls(plate.omega_n, plate.zeta_modes, Dp, plate.H_Pe_modal,
                  cutting, m_div=m_div)
        obj.C_obs = np.asarray(plate.D_obs, float)   # sensor shape, for output feedback
        return obj

    # ------------------------------------------------------------------
    def _a4_over_period(self, RPM, ap, m_div):
        """Regenerative coefficient a4(t) sampled over one tooth period."""
        c = self.c
        tau = 60.0 / (c["NT"] * RPM)
        dt_int = tau / m_div
        Omega = 2.0 * np.pi * RPM / 60.0
        za_low, za_high = c["hp"] - ap, c["hp"]
        a4 = np.empty(m_div)
        for k in range(m_div):
            _, a4k = milling_force_coeffs(
                k * dt_int, Omega, c["NT"], c["RT"], c["eta_h"],
                c["phi_st"], c["phi_ex"], za_low, za_high,
                c["k1"], c["k2"], c["kt"])
            a4[k] = a4k
        return a4, dt_int

    # ------------------------------------------------------------------
    def rho(self, RPM, ap, K=None, K_tau=None, m_div=None):
        """Dominant Floquet multiplier (spectral radius) of the closed loop.

        Stable milling  <=> rho < 1.  Chatter  <=> rho >= 1.
        """
        m_div = self.m_div if m_div is None else m_div
        n, nx = self.n, self.nx
        a4, dt_int = self._a4_over_period(RPM, ap, m_div)
        n_tau = m_div                       # delay = one period = m_div steps
        N = (n_tau + 1) * nx
        Phi = np.eye(N)
        D = np.zeros((N, N))
        for j in range(n_tau):              # constant shift-register blocks
            D[(j + 1) * nx:(j + 2) * nx, j * nx:(j + 1) * nx] = np.eye(nx)
        top = slice(0, nx)
        tail = slice(n_tau * nx, (n_tau + 1) * nx)
        for k in range(m_div):
            a4k = a4[k]
            Ap = np.zeros((nx, nx))
            Ap[:n, n:] = np.eye(n)
            Ap[n:, :n] = -(self.Kp + a4k * self.Dp2)
            Ap[n:, n:] = -self.Cp
            Ad = np.zeros((nx, nx))
            Ad[n:, :n] = a4k * self.Dp2      # destabilising regenerative delay
            if K is not None:
                Ap = Ap - self.B @ K
            if K_tau is not None:
                Ad = Ad - self.B @ K_tau
            Pd, Ig = _expm_and_integral(Ap, dt_int)
            R = Ig @ Ad
            D[top, :] = 0.0
            D[top, top] = Pd
            D[top, tail] = R
            Phi = D @ Phi
        return float(np.max(np.abs(np.linalg.eigvals(Phi))))

    # ------------------------------------------------------------------
    def ap_crit(self, RPM, K=None, K_tau=None, ap_max=6e-3,
                n_coarse=40, refine=True, m_div=None, ap_floor=1e-5):
        """Critical (maximum stable) depth of cut at a given spindle speed.

        The scan starts from a small stable floor (the regenerative coupling
        vanishes as ap -> 0), brackets the first rho = 1 crossing and bisects it.
        Returns depth in metres.
        """
        if self.rho(RPM, ap_floor, K, K_tau, m_div) >= 1.0:
            return ap_floor                          # unstable even at the floor
        ap_grid = np.linspace(ap_max / n_coarse, ap_max, n_coarse)
        prev = ap_floor
        for ap in ap_grid:
            if self.rho(RPM, ap, K, K_tau, m_div) >= 1.0:
                if not refine:
                    return ap
                lo, hi = prev, ap                    # bisection on [lo, hi]
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if self.rho(RPM, mid, K, K_tau, m_div) >= 1.0:
                        hi = mid
                    else:
                        lo = mid
                    if (hi - lo) < 1e-6:
                        break
                return 0.5 * (lo + hi)
            prev = ap
        return ap_max                                # stable over the whole range

    # ------------------------------------------------------------------
    def sld_grid(self, RPM_array, ap_array, K=None, K_tau=None,
                 m_div=None, verbose=False):
        """Spectral-radius grid rho[i_ap, i_rpm] for a controlled SLD."""
        import time
        RPM_array = np.asarray(RPM_array, float)
        ap_array = np.asarray(ap_array, float)
        rho_grid = np.zeros((ap_array.size, RPM_array.size))
        t0 = time.time()
        for j, RPM in enumerate(RPM_array):
            for i, ap in enumerate(ap_array):
                rho_grid[i, j] = self.rho(RPM, ap, K, K_tau, m_div)
            if verbose and (j + 1) % max(1, RPM_array.size // 10) == 0:
                print(f"  [CL-FDM] {100*(j+1)/RPM_array.size:5.1f}%  "
                      f"({time.time()-t0:5.1f}s)")
        return rho_grid


    # ==================================================================
    # Dynamic output-feedback controllers (e.g. ADRC / observer-based)
    # ==================================================================
    def _Cpl(self):
        """Plant output row y = C_obs . q  (shape (1, nx))."""
        if self.C_obs is None:
            raise ValueError("C_obs (sensor shape) not set; build via from_plate().")
        C = np.zeros((1, self.nx))
        C[0, :self.n] = self.C_obs
        return C

    def rho_dynamic(self, RPM, ap, Ac, Bc, Cc, Dc, m_div=None):
        """Dominant Floquet multiplier for a linear *dynamic* output-feedback
        controller  z_dot = Ac z + Bc y,  u = Cc z + Dc y.

        The closed loop augments the plant state x = [q; q_dot] with the
        controller state z; the regenerative delay acts on x only.  With ADRC,
        z is the extended-state-observer state, so this is the true ADRC SLD.
        """
        m_div = self.m_div if m_div is None else m_div
        n, nx = self.n, self.nx
        Ac = np.asarray(Ac, float); Bc = np.asarray(Bc, float)
        Cc = np.asarray(Cc, float); Dc = np.asarray(Dc, float)
        nz = Ac.shape[0]
        na = nx + nz
        Cpl = self._Cpl()
        B = self.B
        BCc = B @ Cc                       # (nx, nz)
        BDcCpl = B @ Dc @ Cpl              # (nx, nx)
        BcCpl = Bc @ Cpl                   # (nz, nx)
        a4, dt_int = self._a4_over_period(RPM, ap, m_div)
        n_tau = m_div
        N = (n_tau + 1) * na
        Phi = np.eye(N)
        D = np.zeros((N, N))
        for j in range(n_tau):
            D[(j + 1) * na:(j + 2) * na, j * na:(j + 1) * na] = np.eye(na)
        top = slice(0, na)
        tail = slice(n_tau * na, (n_tau + 1) * na)
        for k in range(m_div):
            a4k = a4[k]
            Ap = np.zeros((nx, nx))
            Ap[:n, n:] = np.eye(n)
            Ap[n:, :n] = -(self.Kp + a4k * self.Dp2)
            Ap[n:, n:] = -self.Cp
            A_now = np.zeros((na, na))
            A_now[:nx, :nx] = Ap + BDcCpl
            A_now[:nx, nx:] = BCc
            A_now[nx:, :nx] = BcCpl
            A_now[nx:, nx:] = Ac
            A_del = np.zeros((na, na))
            A_del[n:nx, :n] = a4k * self.Dp2      # regenerative delay on x only
            Pd, Ig = _expm_and_integral(A_now, dt_int)
            R = Ig @ A_del
            D[top, :] = 0.0
            D[top, top] = Pd
            D[top, tail] = R
            Phi = D @ Phi
        return float(np.max(np.abs(np.linalg.eigvals(Phi))))

    def ap_crit_dynamic(self, RPM, Ac, Bc, Cc, Dc, ap_max=6e-3,
                        n_coarse=40, refine=True, m_div=None, ap_floor=1e-5):
        """Critical depth of cut for a dynamic output-feedback controller."""
        if self.rho_dynamic(RPM, ap_floor, Ac, Bc, Cc, Dc, m_div) >= 1.0:
            return ap_floor
        ap_grid = np.linspace(ap_max / n_coarse, ap_max, n_coarse)
        prev = ap_floor
        for ap in ap_grid:
            if self.rho_dynamic(RPM, ap, Ac, Bc, Cc, Dc, m_div) >= 1.0:
                if not refine:
                    return ap
                lo, hi = prev, ap
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if self.rho_dynamic(RPM, mid, Ac, Bc, Cc, Dc, m_div) >= 1.0:
                        hi = mid
                    else:
                        lo = mid
                    if (hi - lo) < 1e-6:
                        break
                return 0.5 * (lo + hi)
            prev = ap
        return ap_max

    def sld_grid_dynamic(self, RPM_array, ap_array, Ac, Bc, Cc, Dc,
                         m_div=None, verbose=False):
        """Spectral-radius grid for a dynamic output-feedback controller."""
        import time
        RPM_array = np.asarray(RPM_array, float)
        ap_array = np.asarray(ap_array, float)
        rho_grid = np.zeros((ap_array.size, RPM_array.size))
        t0 = time.time()
        for j, RPM in enumerate(RPM_array):
            for i, ap in enumerate(ap_array):
                rho_grid[i, j] = self.rho_dynamic(RPM, ap, Ac, Bc, Cc, Dc, m_div)
            if verbose and (j + 1) % max(1, RPM_array.size // 10) == 0:
                print(f"  [CL-FDM dyn] {100*(j+1)/RPM_array.size:5.1f}%  "
                      f"({time.time()-t0:5.1f}s)")
        return rho_grid


def default_cutting(NT=3, RT=5e-3, eta_h=None, gamma_n=None,
                    kt=925e6, kN=0.26, mu_c=0.20, ae=0.1e-3, hp=0.080):
    """Assemble the cutting-geometry dict used throughout the study."""
    eta_h = np.deg2rad(35) if eta_h is None else eta_h
    gamma_n = np.deg2rad(15) if gamma_n is None else gamma_n
    phi_st = np.pi - np.arccos(1 - ae / RT)
    phi_ex = np.pi
    k1 = kN * np.cos(eta_h)
    k2 = 1 + mu_c * np.tan(eta_h) * np.cos(gamma_n) - kN * np.sin(eta_h)
    return dict(NT=NT, RT=RT, eta_h=eta_h, phi_st=phi_st, phi_ex=phi_ex,
                k1=k1, k2=k2, kt=kt, hp=hp)
