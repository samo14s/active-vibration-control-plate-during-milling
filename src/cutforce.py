"""
Paper-exact non-smooth milling-force coefficients alpha3(t), alpha4(t).

Implements the semi-analytical per-tooth milling-force model of the reference
paper (Du et al., IJMS 274 (2024) 109257, Eqs. (2)-(5)):

    [Fx; Fy] = [[a1, a3], [a2, a4]] [xr; yr]

with the time-varying, NON-SMOOTH coefficients of Eq. (3)

    alpha3(t) = sum_j kt ( k2*ss(t,j) - k1*sc(t,j) )
    alpha4(t) = sum_j kt ( k2*sc(t,j) - k1*cc(t,j) )

    k1 = kn / cos(eta) ,   k2 = 1 + mu_c*tan(eta)*(cos(gamma_n) - kn*sin(gamma_n))

and the helix-integrated engagement integrals of Eq. (4) over the engaged flute
segment [z1, z2] of tooth j (theta = tooth rotation angle at height z):

    ss = (z2-z1)/2 + R/(2 tan eta) * sin[th2-th1] * cos[th2+th1]   ( = int sin^2 )
    sc =            R/(2 tan eta) * sin[th1-th2] * sin[th1+th2]    ( = int sin cos )
    cc = (z2-z1)/2 - R/(2 tan eta) * sin[th2-th1] * cos[th2+th1]   ( = int cos^2 )

The tooth angle lags with height because of the helix:
    theta(t, j, z) = Omega t - j*(2 pi / NT) - (tan eta / R) * z
and a tooth cuts only while theta is inside the down-milling engagement window
[pi - phi_eng, pi] with phi_eng = acos(1 - ae/R).

The waveforms are periodic with the tooth-passing period tau; they are sampled
once on a fine phase grid at plant construction and interpolated during
simulation, so the true simulated plant uses the paper's actual non-smooth
milling force instead of an idealised engagement window.  In Eq. (12) they act as

    ... + alpha4(t) * Phi_m Phi_m^T [q(t) - q(t-tau)] = f_tau * alpha3(t) * Phi_m + ...

Scale note: alpha4(t) here is in physical N/m (kt * engaged length).  The overall
loop gain of the mass-normalised modal model still carries the single calibrated
multiplier CUT_GAIN (see config.py) applied to the NORMALISED waveform
w(t)/|mean(w)|, so the calibration philosophy (match the paper's open-loop
chatter at condition S) is unchanged -- only the waveform SHAPE becomes exact.
The sign convention of the printed Eq. (3) makes alpha4 negative in down-milling;
the destabilising regenerative sense (which the paper's stability lobes and the
Fig. 14(a) chatter fix) is selected by the calibration in config/plant.
"""

from __future__ import annotations
import numpy as np

from . import config as C


def _engaged_z(theta0, kappa, ap, th_st, th_ex):
    """Engaged flute segment [z1, z2] of one tooth.

    theta(z) = theta0 - kappa*z decreases with height z in [0, ap]; the tooth
    cuts while theta(z) is inside [th_st, th_ex].  Returns (z1, z2) with
    z1 < z2, or None when the tooth is out of cut.
    """
    # unwrap theta0 into the revolution containing the window
    th0 = th_st + np.mod(theta0 - th_st, 2.0 * np.pi)
    # theta(z) in [th_st, th_ex]  ->  z in [(th0-th_ex)/kappa, (th0-th_st)/kappa]
    z_lo = (th0 - th_ex) / kappa
    z_hi = (th0 - th_st) / kappa
    z1 = max(0.0, z_lo)
    z2 = min(ap, z_hi)
    if z2 <= z1:
        return None
    return z1, z2


_WAVEFORM_CACHE = {}


def paper_alpha_waveforms(n_grid=4096):
    """Sample alpha3(t), alpha4(t) (Eqs. 3-4) over one tooth-passing period.

    Returns dict with the phase grid in [0,1), the two waveforms [N/m], their
    period means, and the engaged duty fraction.  Cached per grid size (the
    waveform depends only on the fixed tool/condition parameters).
    """
    if n_grid in _WAVEFORM_CACHE:
        return _WAVEFORM_CACHE[n_grid]
    T = C.TOOL
    S = C.CONDITION_S
    NT = T["n_teeth"]
    R = T["radius"]
    ap = S["a_axial"]
    ae = S["a_radial"]
    eta = np.deg2rad(T["helix_deg"])
    gam = np.deg2rad(T["rake_deg"])
    kt = T["kt"]
    kn = T["kn"]
    mu_c = T["mu_c"]

    k1 = kn / np.cos(eta)
    k2 = 1.0 + mu_c * np.tan(eta) * (np.cos(gam) - kn * np.sin(gam))
    kappa = np.tan(eta) / R                     # helix lag rate d(theta)/dz

    # down-milling engagement window (radial immersion ae on a tool of radius R)
    phi_eng = np.arccos(np.clip(1.0 - ae / R, -1.0, 1.0))
    th_ex = np.pi
    th_st = np.pi - phi_eng

    Omega = 2.0 * np.pi * S["spindle_rpm"] / 60.0        # spindle speed [rad/s]
    tau = C.TAU_S                                        # tooth-passing period
    Reta = R / (2.0 * np.tan(eta))

    phase = np.arange(n_grid) / n_grid                   # tooth-period phase
    a3 = np.zeros(n_grid)
    a4 = np.zeros(n_grid)
    engaged = np.zeros(n_grid, bool)
    for i, ph in enumerate(phase):
        t = ph * tau
        for j in range(NT):
            th0 = Omega * t - j * (2.0 * np.pi / NT)     # angle at z = 0
            seg = _engaged_z(th0, kappa, ap, th_st, th_ex)
            if seg is None:
                continue
            z1, z2 = seg
            th1 = th0 - kappa * z1
            th2 = th0 - kappa * z2
            dz = z2 - z1
            ss = dz / 2.0 + Reta * np.sin(th2 - th1) * np.cos(th2 + th1)
            sc = Reta * np.sin(th1 - th2) * np.sin(th1 + th2)
            cc = dz / 2.0 - Reta * np.sin(th2 - th1) * np.cos(th2 + th1)
            a3[i] += kt * (k2 * ss - k1 * sc)
            a4[i] += kt * (k2 * sc - k1 * cc)
            engaged[i] = True

    out = dict(phase=phase, a3=a3, a4=a4,
               a3_mean=float(np.mean(a3)), a4_mean=float(np.mean(a4)),
               duty=float(np.mean(engaged)))
    _WAVEFORM_CACHE[n_grid] = out
    return out
