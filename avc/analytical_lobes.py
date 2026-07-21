"""Classical Altintas-Budak analytical stability lobes (zero-order / ZOA).

The single-frequency (zero-order) analytical solution of the regenerative
milling stability problem (Altintas & Budak, CIRP Annals 44 (1995) 357).
For the single flexible normal direction of the thin-walled plate the
oriented transfer function reduces to the scalar tool-point receptance

    Phi(i w) = sum_r  bf_r^2 / (w_r^2 - w^2 + 2 i zeta_r w_r w)   [m/N]

and the dynamic-milling characteristic equation of the linearized loop
w = Phi (i w_c) * a_bar * (e^{-i w_c tau} - 1) * w  is

    1 + a_p * g0 * Phi(i w_c) * (1 - e^{-i w_c tau}) = 0,

with g0 = mean directional cutting coefficient per unit axial depth
(avc.milling.mean_alpha4 at a_p = 1 m; g0 < 0 for this down-milling
engagement) and tau = 60/(N rpm) the tooth-passing / regenerative delay.
Requiring a_p real gives, at each chatter frequency w_c, the classical
closed form: the phase epsilon = w_c tau from Im[.] = 0 and the critical
depth from Re[.]. Scanning w_c near the resonances and sweeping the lobe
number k traces the whole diagram. No time-domain discretization is used.

The module consumes any object exposing the ModalModel interface
(``omega``, ``zeta``, ``bf(x_T)``), so it runs on the classical Warburton
model of avc.classical_plate or on the FEM model of avc.modal.
"""
from __future__ import annotations

import numpy as np

from .milling import mean_alpha4


def oriented_frf(mm, x_T: float, omega: np.ndarray) -> np.ndarray:
    """Tool-point direct receptance Phi(i omega) [m/N] at position x_T."""
    bf = mm.bf(x_T)
    w = np.asarray(omega, float)
    H = np.zeros(w.shape, dtype=complex)
    for r in range(mm.n_modes):
        H += bf[r] ** 2 / (mm.omega[r] ** 2 - w ** 2
                           + 2j * mm.zeta[r] * mm.omega[r] * w)
    return H


def _branches(mm, x_T: float, n_freq: int = 6000, kmax: int = 60,
              ap_hi: float = 5e-3):
    """Continuous lobe branches of the zero-order solution.

    Each retained resonance contributes a chatter region (Re[Phi] > 0);
    within it every lobe number k yields a branch, a rpm-sorted pair of
    arrays (rpm, a_lim) that can be interpolated to any spindle speed. The
    lower envelope over all branches is the classical stability lobe
    diagram. Returns a list of (rpm_sorted, alim_sorted) arrays."""
    p = mm.params
    N = p.tool.n_teeth
    g0 = mean_alpha4(p, 1.0)               # per-unit-depth coefficient
    per = max(400, n_freq // mm.n_modes)
    out = []
    for wr in mm.omega[: mm.n_modes]:
        fc = wr / (2.0 * np.pi)
        wscan = 2.0 * np.pi * np.linspace(fc * 0.5, fc * 1.5, per)
        Phi = oriented_frf(mm, x_T, wscan)
        pr, pi = Phi.real, Phi.imag
        good = (pr != 0.0) & (pi != 0.0)
        a_lim = np.where(good, -1.0 / (2.0 * g0 * pr), -1.0)
        sel = good & (a_lim > 0.0) & (a_lim <= ap_hi)
        if sel.sum() < 2:
            continue
        wsel, asel, prs, pis = wscan[sel], a_lim[sel], pr[sel], pi[sel]
        eps0 = (2.0 * np.arctan2(-prs, pis)) % (2.0 * np.pi)
        for k in range(kmax + 1):
            rpm = 60.0 * wsel / (N * (eps0 + 2.0 * np.pi * k))
            o = np.argsort(rpm)
            out.append((rpm[o], asel[o]))
    return out


def _envelope_at(branches, rpms, ap_hi):
    """Lower envelope a_lim [m] on the rpm grid via per-branch interp."""
    rpms = np.atleast_1d(np.asarray(rpms, float))
    out = np.full(rpms.shape, ap_hi)
    for r_b, a_b in branches:
        if len(r_b) < 2:
            continue
        inside = (rpms >= r_b[0]) & (rpms <= r_b[-1])
        if not inside.any():
            continue
        vals = np.interp(rpms[inside], r_b, a_b)
        out[inside] = np.minimum(out[inside], vals)
    return out


def lobes(mm, x_T: float, rpm_range=(2000.0, 10000.0),
          n_freq: int = 6000, kmax: int = 60, ap_hi: float = 5e-3):
    """Raw analytical stability-boundary point cloud (rpm, a_lim), sorted
    by rpm and filtered to rpm_range (for plotting the scallops)."""
    r_all, a_all = [], []
    for r_b, a_b in _branches(mm, x_T, n_freq, kmax, ap_hi):
        m = (r_b >= rpm_range[0]) & (r_b <= rpm_range[1])
        r_all.append(r_b[m])
        a_all.append(a_b[m])
    if not r_all:
        return np.array([]), np.array([])
    r = np.concatenate(r_all)
    a = np.concatenate(a_all)
    o = np.argsort(r)
    return r[o], a[o]


def critical_depth(mm, x_T: float, rpm: float, ap_hi: float = 5e-3,
                   **kw) -> float:
    """Classical (zero-order) critical depth [m] at one spindle speed:
    lower envelope of the analytical lobes interpolated at that speed."""
    br = _branches(mm, x_T, ap_hi=ap_hi, **kw)
    return float(_envelope_at(br, [rpm], ap_hi)[0])


def lobe_envelope(mm, x_T: float, rpms: np.ndarray, ap_hi: float = 5e-3,
                  **kw) -> np.ndarray:
    """Lower-envelope a_lim [m] on a spindle-speed grid (classical SLD)."""
    br = _branches(mm, x_T, ap_hi=ap_hi, **kw)
    return _envelope_at(br, rpms, ap_hi)
