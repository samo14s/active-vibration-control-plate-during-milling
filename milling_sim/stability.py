"""Regenerative chatter stability analysis (paper Eqs. (8)-(13)).

Implements the zero-order analytical (ZOA) stability solution of
Altintas & Budak, which the paper adopts:

* Eq. (8):  det(I + G(i w_c) H(i w_c)) = 0     characteristic equation
* Eq. (9):  oriented transfer function with directional factors a_ij
* Eq. (10): receptance H(i w) as a modal sum
* Eq. (11): a_p,lim = -2 pi Lambda_R (1 + kappa^2) / (N Kt)
* Eq. (12): n = 60 w_c / (N (2 k pi + pi - 2 psi)),  psi = arctan(kappa)
* Eq. (13): process damping Cp = Kp V0 / V (adds speed-dependent damping)

The directional factors are the classical closed-form integrals of the
linearised cutting-force model Ft = Kt h ap, Fr = Kr Ft over the engaged
arc [phi_st, phi_ex].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .parameters import EngagementParams, Mode, ToolParams


def directional_matrix(Kr: float, phi_st: float, phi_ex: float) -> np.ndarray:
    """Average directional coefficient matrix [[axx, axy], [ayx, ayy]].

    Closed-form integration of the rotating force projections over the
    engagement window (the integrals behind Eq. (9)).
    """

    def _at(phi: float) -> np.ndarray:
        c2, s2 = math.cos(2.0 * phi), math.sin(2.0 * phi)
        axx = 0.5 * (c2 - 2.0 * Kr * phi + Kr * s2)
        axy = 0.5 * (-s2 - 2.0 * phi + Kr * c2)
        ayx = 0.5 * (-s2 + 2.0 * phi + Kr * c2)
        ayy = 0.5 * (-c2 - 2.0 * Kr * phi - Kr * s2)
        return np.array([[axx, axy], [ayx, ayy]])

    return _at(phi_ex) - _at(phi_st)


def receptance_matrix(w: float, modes: list[Mode], Vr: float) -> np.ndarray:
    """H(i w): 2x2 dynamic compliance at the cutting point, Eq. (10).

    Tool x-modes populate H[0,0]; tool y-modes and wall y-modes both
    contribute to H[1,1] (the relative tool-workpiece receptance in the
    wall-normal direction).  Cross terms are neglected (symmetric tool,
    wall compliant only along its normal).
    """
    H = np.zeros((2, 2), dtype=complex)
    for m in modes:
        wn = m.wn(Vr)
        denom = (m.mass_kg * (wn * wn - w * w)
                 + 1j * (2.0 * m.zeta(Vr) * m.mass_kg * wn * w))
        g = 1.0 / denom
        if m.direction == "x":
            H[0, 0] += g
        else:
            H[1, 1] += g
    return H


@dataclass
class LobePoint:
    spindle_rpm: float
    ap_lim_m: float
    chatter_freq_hz: float
    lobe_k: int


def stability_lobes(tool: ToolParams,
                    engagement: EngagementParams,
                    modes: list[Mode],
                    Vr: float,
                    f_lo: float = 400.0,
                    f_hi: float = 3600.0,
                    n_freq: int = 900,
                    k_max: int = 12,
                    process_damping: bool = False,
                    ) -> list[LobePoint]:
    """Sweep candidate chatter frequencies and solve Eqs. (11)-(12).

    Returns the raw cloud of (n, ap_lim) lobe points; use
    :func:`critical_depth_at_speed` for the envelope at one speed.
    """
    phi_st, phi_ex = engagement.entry_exit_angles(tool)
    A = directional_matrix(tool.Kr, phi_st, phi_ex)
    N = tool.n_teeth
    points: list[LobePoint] = []

    for f in np.linspace(f_lo, f_hi, n_freq):
        w = 2.0 * math.pi * f
        H = receptance_matrix(w, modes, Vr)
        if process_damping:
            H = _apply_process_damping(H, w, tool, modes, Vr)
        G0 = A @ H                      # oriented receptance matrix, Eq. (9)
        eigs = np.linalg.eigvals(G0)
        for nu in eigs:
            if abs(nu) < 1e-18:
                continue
            Lam = -1.0 / nu             # eigenvalue of the char. eq., Eq. (8)
            LamR, LamI = Lam.real, Lam.imag
            if LamR >= 0.0:
                continue                # only negative real parts give ap > 0
            kappa = LamI / LamR
            ap = (-2.0 * math.pi * LamR / (N * tool.Kt)) * (1.0 + kappa * kappa)
            if ap <= 0.0 or ap > 0.05:
                continue
            psi = math.atan(kappa)
            for k in range(k_max):
                eps = 2.0 * k * math.pi + math.pi - 2.0 * psi   # Eq. (12)
                if eps <= 0.0:
                    continue
                n_rpm = 60.0 * w / (N * eps)
                if 2000.0 <= n_rpm <= 30000.0:
                    points.append(LobePoint(n_rpm, ap, f, k))
    return points


def _apply_process_damping(H: np.ndarray, w: float, tool: ToolParams,
                           modes: list[Mode], Vr: float) -> np.ndarray:
    """Fold the speed-dependent process damping of Eq. (13) into H[1,1].

    Modelled as extra modal damping on the dominant wall mode proportional
    to Kp V0 / V, evaluated at a representative cutting speed.
    """
    # Representative surface speed at 12 000 rpm for the damping magnitude.
    V = math.pi * tool.diameter_m * 12000.0 / 60.0
    Cp = tool.Kp * tool.V0 / max(V, 0.5)
    H = H.copy()
    g = H[1, 1]
    if abs(g) > 0.0:
        # Add Cp as parallel viscous damping: 1/g' = 1/g + i w Cp
        H[1, 1] = 1.0 / (1.0 / g + 1j * w * Cp)
    return H


def lobe_envelope(points: list[LobePoint],
                  n_grid_rpm: np.ndarray) -> np.ndarray:
    """Lower envelope ap_lim(n) [m] over a spindle-speed grid."""
    ap_env = np.full_like(n_grid_rpm, np.inf, dtype=float)
    if not points:
        return ap_env
    n_arr = np.array([p.spindle_rpm for p in points])
    ap_arr = np.array([p.ap_lim_m for p in points])
    # bin points to the grid and keep the minimum per bin, then close gaps
    idx = np.clip(np.searchsorted(n_grid_rpm, n_arr), 0, len(n_grid_rpm) - 1)
    for i, a in zip(idx, ap_arr):
        if a < ap_env[i]:
            ap_env[i] = a
    # fill empty bins by nearest-neighbour interpolation of finite values
    finite = np.isfinite(ap_env)
    if finite.any() and not finite.all():
        ap_env[~finite] = np.interp(n_grid_rpm[~finite],
                                    n_grid_rpm[finite], ap_env[finite])
    return ap_env


def fast_lobe_envelope(tool: ToolParams,
                       engagement: EngagementParams,
                       modes_fz: list[tuple[float, float, float]],
                       n_grid_rpm: np.ndarray,
                       directions: list[str] | None = None,
                       f_lo: float = 400.0,
                       f_hi: float = 3600.0,
                       n_freq: int = 2400,
                       k_max: int = 14) -> np.ndarray:
    """Vectorised ap_lim(n) envelope [m] for the controller's 100 Hz loop.

    ``modes_fz`` is a list of (fn_hz, zeta, mass_kg) tuples;  ``directions``
    gives 'x'/'y' per mode (defaults to the standard set order).  Uses the
    closed-form eigenvalues of the 2x2 oriented matrix, fully vectorised
    over the chatter-frequency grid, then scatters Eq. (12) speeds onto
    ``n_grid_rpm`` keeping the per-bin minimum of Eq. (11).
    """
    phi_st, phi_ex = engagement.entry_exit_angles(tool)
    A = directional_matrix(tool.Kr, phi_st, phi_ex)
    N = tool.n_teeth
    if directions is None:
        directions = ["y"] * len(modes_fz)

    w = 2.0 * math.pi * np.linspace(f_lo, f_hi, n_freq)
    Hxx = np.zeros(n_freq, dtype=complex)
    Hyy = np.zeros(n_freq, dtype=complex)
    for (fn, ze, ma), d in zip(modes_fz, directions):
        wn = 2.0 * math.pi * fn
        g = 1.0 / (ma * (wn * wn - w * w) + 1j * (2.0 * ze * ma * wn * w))
        if d == "x":
            Hxx += g
        else:
            Hyy += g

    # G0 = A @ diag(Hxx, Hyy):  [[a00*Hxx, a01*Hyy], [a10*Hxx, a11*Hyy]]
    a00 = A[0, 0] * Hxx
    a01 = A[0, 1] * Hyy
    a10 = A[1, 0] * Hxx
    a11 = A[1, 1] * Hyy
    tr = a00 + a11
    det = a00 * a11 - a01 * a10
    disc = np.sqrt(tr * tr - 4.0 * det)
    nus = np.stack([(tr + disc) * 0.5, (tr - disc) * 0.5])   # (2, n_freq)

    ap_env = np.full(n_grid_rpm.shape, np.inf)
    n_lo, n_hi = n_grid_rpm[0], n_grid_rpm[-1]
    bin_w = (n_hi - n_lo) / max(len(n_grid_rpm) - 1, 1)
    for nu in nus:
        ok = np.abs(nu) > 1e-18
        Lam = np.where(ok, -1.0 / np.where(ok, nu, 1.0), 0.0)
        LamR, LamI = Lam.real, Lam.imag
        valid = ok & (LamR < 0.0)
        kappa = np.where(valid, LamI / np.where(valid, LamR, 1.0), 0.0)
        ap = (-2.0 * math.pi * LamR / (N * tool.Kt)) * (1.0 + kappa ** 2)
        valid = valid & (ap <= 0.05)              # physical cap as loop version
        psi = np.arctan(kappa)
        for k in range(k_max):
            eps = 2.0 * k * math.pi + math.pi - 2.0 * psi
            n_rpm = 60.0 * w / (N * eps)
            sel = valid & (ap > 0.0) & (n_rpm >= n_lo) & (n_rpm <= n_hi)
            if not sel.any():
                continue
            idx = np.clip(((n_rpm[sel] - n_lo) / bin_w).round().astype(int),
                          0, len(n_grid_rpm) - 1)
            np.minimum.at(ap_env, idx, ap[sel])

    finite = np.isfinite(ap_env)
    if finite.any() and not finite.all():
        ap_env[~finite] = np.interp(n_grid_rpm[~finite],
                                    n_grid_rpm[finite], ap_env[finite])
    return ap_env


def critical_depth_at_speed(tool: ToolParams,
                            engagement: EngagementParams,
                            modes: list[Mode],
                            Vr: float,
                            n_rpm: float,
                            f_lo: float = 400.0,
                            f_hi: float = 3600.0,
                            n_freq: int = 700,
                            k_max: int = 14) -> float:
    """ap_lim [m] at one spindle speed: minimise ap over chatter frequencies
    whose Eq. (12) speed matches ``n_rpm``.

    Direct evaluation (no gridding): for each candidate frequency and lobe
    number the exact speed is computed; the point contributes when it lies
    within half a grid-step of the requested speed.
    """
    phi_st, phi_ex = engagement.entry_exit_angles(tool)
    A = directional_matrix(tool.Kr, phi_st, phi_ex)
    N = tool.n_teeth
    best = math.inf
    tol = 0.02 * n_rpm + 30.0

    for f in np.linspace(f_lo, f_hi, n_freq):
        w = 2.0 * math.pi * f
        H = receptance_matrix(w, modes, Vr)
        G0 = A @ H
        eigs = np.linalg.eigvals(G0)
        for nu in eigs:
            if abs(nu) < 1e-18:
                continue
            Lam = -1.0 / nu
            if Lam.real >= 0.0:
                continue
            kappa = Lam.imag / Lam.real
            ap = (-2.0 * math.pi * Lam.real / (N * tool.Kt)) * (1.0 + kappa ** 2)
            if ap <= 0.0:
                continue
            psi = math.atan(kappa)
            for k in range(k_max):
                eps = 2.0 * k * math.pi + math.pi - 2.0 * psi
                if eps <= 0.0:
                    continue
                n_cand = 60.0 * w / (N * eps)
                if abs(n_cand - n_rpm) < tol and ap < best:
                    best = ap
    return best
