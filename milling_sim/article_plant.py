"""Bridge between the article simulation package (this repository's own
study: LQG vs DARC-MPC active vibration control of a cantilever AL6061
plate during peripheral milling, modelling per Nasiri & Moradi, MSSP 224
(2025) 112198) and the `milling_sim` adaptive-machining framework.

Two things live here:

1. ``load_article_plate()`` — constructs the package's analytical
   Von-Karman/Galerkin plate exactly as the package's own mains do and
   returns the SISO regenerative plant data at the tool point:
   modal frequencies/damping, modal masses, mode-shape gains Dp, and the
   linear regenerative coefficient slope c_reg = <alpha4>/ap.

2. ``analytic_ap_limit()`` — frequency-domain (averaged-coefficient,
   ZOA-style) stability boundary for the package's scalar NDDE

        M eta'' + C eta' + K eta = -a4 * Dp * (w(t) - w(t - tau)),
        w = Dp^T eta,   a4 ~= c_reg * ap

   whose characteristic equation is  1 + a4 (1 - e^{-i w tau}) G(i w) = 0
   with G(i w) = sum_i Dp_i^2 / (m_i (w_i^2 - w^2) + 2 i zeta_i m_i w_i w).
   This is an INDEPENDENT check of the package's time-domain Floquet
   (full-discretisation) SLD: two methods, one plant.

3. ``article_setup()`` — maps the plate into a `PaperSetup` for the
   adaptive-machining framework (modes as y-direction workpiece modes with
   effective masses m_i / Dp_i^2; force model mapped via
   Kt_eff = k2*KT, Kr_eff = k1/k2, which makes my mechanistic y-force
   Kt h (sin phi - Kr cos phi) coincide with the package's normal-force
   integrand kt (k2 sin theta - k1 cos theta) h).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import numpy as np

_PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "article_package", "workspace")


def _import_package():
    if _PKG not in sys.path:
        sys.path.insert(0, _PKG)
    import plate_model, milling_force, fdm_stability  # noqa: F401
    return plate_model, milling_force, fdm_stability


# Package physical constants (article_package/05_main/main_simulation.py)
LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU_AL = 2830.0, 69e9, 0.33
NT, DT_TOOL = 3, 0.010
ETA_H = math.radians(35.0)
GAMMA_N = math.radians(15.0)
KT_NOMINAL, KN, MU_C = 925e6, 0.26, 0.20
RPM_NOM = 4900.0
FT_NOM = 0.02e-3
AE = 0.1e-3
AP_NOM = 0.3e-3
N1, N2, N_MODES = 30, 24, 3
ZETA = [0.0031, 0.0017, 0.0027]

K1 = KN * math.cos(ETA_H)
K2 = 1.0 + MU_C * math.tan(ETA_H) * math.cos(GAMMA_N) - KN * math.sin(ETA_H)
RT = DT_TOOL / 2.0
PHI_ST = math.pi - math.acos(1.0 - AE / RT)
PHI_EX = math.pi


@dataclass
class ArticlePlant:
    freq_hz: np.ndarray          # modal frequencies [Hz]
    zeta: np.ndarray             # modal damping ratios
    mass: np.ndarray             # modal masses (Galerkin convention)
    Dp: np.ndarray               # mode-shape gain at the tool point
    c_reg: float                 # <alpha4> per metre of ap  [N/m per m]
    plate: object                # the live PlateModel instance


def load_article_plate(zp_pos: float | None = None) -> ArticlePlant:
    plate_model, milling_force, _ = _import_package()
    plate = plate_model.PlateModel(LP, HP, BP, RHO, E_AL, NU_AL,
                                   N1=N1, N2=N2, n_modes=N_MODES,
                                   zeta_modes=ZETA, verbose=False)
    plate.precompute_Dp(zp_pos=(HP - AP_NOM / 2 if zp_pos is None else zp_pos),
                        n_pos=2001)
    plate.set_observation(x_obs=LP, z_obs=HP)

    # same tool-point averaging as the package's own SLD (fig07)
    Dp_sample = [plate.get_Dp_at(kp)[0] for kp in range(0, 2001, 50)]
    Dp_avg = np.mean(Dp_sample, axis=0)

    # linear regenerative coefficient slope: <alpha4(t)> over one
    # tooth-passing period, per metre of axial engagement
    ap_ref = AP_NOM
    dt = 5e-5
    tau = 60.0 / (NT * RPM_NOM)
    n_per = max(8, int(round(tau / dt)))
    _, a4_t = milling_force.precompute_alpha_periodic(
        dt, n_per, n_per, 2 * math.pi * RPM_NOM / 60.0, NT, RT, ETA_H,
        PHI_ST, PHI_EX, HP - ap_ref, HP, K1, K2, KT_NOMINAL)
    c_reg = float(np.mean(a4_t)) / ap_ref

    return ArticlePlant(freq_hz=np.asarray(plate.freq_n, float),
                        zeta=np.asarray(ZETA, float),
                        mass=np.asarray(np.diag(plate.Mp), float),
                        Dp=np.asarray(Dp_avg, float),
                        c_reg=c_reg, plate=plate)


def tool_point_receptance(plant: ArticlePlant, w: np.ndarray,
                          freq_hz: np.ndarray | None = None,
                          zeta: np.ndarray | None = None) -> np.ndarray:
    """G(i w) at the tool point [m/N]."""
    f = plant.freq_hz if freq_hz is None else np.asarray(freq_hz, float)
    z = plant.zeta if zeta is None else np.asarray(zeta, float)
    G = np.zeros_like(w, dtype=complex)
    for fi, zi, mi, di in zip(f, z, plant.mass, plant.Dp):
        wn = 2.0 * math.pi * fi
        G += di * di / (mi * (wn * wn - w * w)
                        + 2j * zi * mi * wn * w)
    return G


def analytic_ap_limit(plant: ArticlePlant, rpm: np.ndarray,
                      freq_hz: np.ndarray | None = None,
                      zeta: np.ndarray | None = None,
                      f_lo: float = 150.0, f_hi: float = 3600.0,
                      n_freq: int = 6000) -> np.ndarray:
    """ZOA-style critical axial depth ap_lim(rpm) [m] for the scalar NDDE.

    For each spindle speed, ap_c(w) = -1 / (c_reg (1 - e^{-i w tau}) G(i w))
    is real-valued exactly on the chatter-frequency locus; the boundary is
    the minimum positive real solution.  Zeros of Im(ap_c) are located by
    sign changes on a dense frequency grid and refined by interpolation.
    """
    w = 2.0 * math.pi * np.linspace(f_lo, f_hi, n_freq)
    G = tool_point_receptance(plant, w, freq_hz, zeta)
    out = np.full(len(rpm), np.inf)
    for i, n in enumerate(np.asarray(rpm, float)):
        tau = 60.0 / (NT * n)
        D = 1.0 - np.exp(-1j * w * tau)
        denom = plant.c_reg * D * G
        ok = np.abs(denom) > 1e-16
        apc = np.where(ok, -1.0 / np.where(ok, denom, 1.0), np.nan)
        im, re = apc.imag, apc.real
        s = im[:-1] * im[1:]
        best = np.inf
        for k in np.nonzero((s < 0.0) & np.isfinite(s))[0]:
            t = im[k] / (im[k] - im[k + 1])
            ap_root = re[k] * (1 - t) + re[k + 1] * t
            if 0.0 < ap_root < best:
                best = ap_root
        out[i] = best
    return out


def article_setup():
    """`PaperSetup` view of the article plant for the adaptive framework."""
    from .parameters import PaperSetup, Mode

    plant = load_article_plate()
    setup = PaperSetup()
    setup.tool.n_teeth = NT
    setup.tool.diameter_m = DT_TOOL
    setup.tool.helix_deg = math.degrees(ETA_H)
    setup.tool.Kt = K2 * KT_NOMINAL          # mapped tangential coefficient
    setup.tool.Kr = K1 / K2                  # mapped radial ratio
    setup.engagement.ae_m = AE
    setup.engagement.down_milling = True
    setup.modes = []
    for fi, zi, mi, di in zip(plant.freq_hz, plant.zeta, plant.mass,
                              plant.Dp):
        # unit-shape convention: effective mass m / Dp^2
        m_eff = float(mi / max(di * di, 1e-12))
        setup.modes.append(Mode(f0_hz=float(fi), zeta0=float(zi),
                                mass_kg=m_eff, alpha=0.0, beta=0.0,
                                direction="y", structure="workpiece"))
    # thin finishing cut: the removed volume is a negligible fraction of
    # the plate, so the Eq. (6)-(7) evolution is dormant (alpha=beta=0)
    setup.baseline.n_rpm = RPM_NOM
    setup.baseline.vf_mm_min = FT_NOM * 1e3 * NT * RPM_NOM   # 294 mm/min
    setup.baseline.ap_mm = AP_NOM * 1e3
    return setup, plant
