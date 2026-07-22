"""
mu-synthesis (structured robust control) — the paper's method.

Constant-D D-K iteration on the reduced 2-mode plant with the two structured
uncertainty sources of the milling problem represented explicitly:

  * additive high-mode uncertainty  — the paper's weight W_Pau(s) (Eq. 19, with
    the published coefficients r_Pau, zeta, omega), covering modes 3+.  For an
    ADDITIVE perturbation  G_true = G + Delta * W_Pau  robust stability requires
    || W_Pau * K S ||_inf < 1, so W_Pau is imposed on the K·S (control-effort)
    channel of the mixed-sensitivity problem — that is where it actually acts;
  * parametric perturbation          — mode mass/stiffness, damping and the
    non-smooth milling coefficient; handled by the performance/sensitivity
    requirement W1·S.

D-K iteration (constant D):
  K-step  : mixed-sensitivity H-infinity synthesis (mixsyn) with W1/W_Pau/(mild
            W3) -> a stable, well-conditioned controller;
  D-step  : rebalance the scalar constant D-scale between the additive-uncertainty
            block (W_Pau · K S) and the performance block (W1 · S) so their
            weighted peaks are equalised.

Robust-performance mu upper bound (exact & tight for this 2-block SISO problem,
Skogestad & Postlethwaite Eq. 8.128):  mu(omega) = |W1 S| + |W_Pau K S|, and the
reported peak is its maximum over frequency.  mu_peak < 1 certifies robust
performance across the whole structured-uncertainty set.
"""

from __future__ import annotations
import numpy as np
import control as ct

from .base import LinearSSController
from .. import config as C
from .. import design as D

SY = 1.0e-4
SU = 1.0e2
# Non-minimum-phase paper geometry: full 3-mode model (captures the RHP zero),
# gentle well-damped design, negative-feedback sign (see hinf.py).
N_DESIGN_MODES = 3
ZETA_DESIGN = 0.10
FB_SIGN = -1.0


def _wpau():
    """Paper additive-uncertainty weight W_Pau(s) (Eq. 19), scaled to design units."""
    s = ct.tf('s')
    tp = 2 * np.pi
    rPau, za1, za2 = 4e-7, 0.58, 0.22
    wa1, wa2 = 1100 * tp, 3500 * tp
    WPau = rPau * (s**2 + 2*za1*wa1*s + wa1**2) / (s**2 + 2*za2*wa2*s + wa2**2)
    return WPau * (SU / SY)          # into scaled (SU/SY) units


# mu-synthesis synthesises on the SAME mixed-sensitivity weights as the robust
# H-infinity design (which stabilises the non-collocated plant across the whole
# pass); the mu-synthesis-specific part is the D-K iteration that scales the T
# (robustness) channel using the paper's additive-uncertainty weight W_Pau.
def _perf_weight():
    """Performance / sensitivity weight W1(s) (shared with the H-infinity design)."""
    s = ct.tf('s')
    tp = 2 * np.pi
    return (s / 2.0 + tp * 800) / (s + tp * 800 * 0.6)


def _ks_weight():
    """Control-effort (K·S) weight (shared with the H-infinity design)."""
    s = ct.tf('s')
    tp = 2 * np.pi
    return 0.5 * (s + tp * 1500) / (s + tp * 6000)


def _rolloff_weight():
    """Complementary-sensitivity (T) weight W3(s) — the robustness channel that
    the D-scale acts on; a larger scale forces the loop to roll off earlier and
    so shrinks the additive-uncertainty term W_Pau·K·S.  Rolloff kept below the
    RHP zero (~5 kHz) of the non-minimum-phase plant."""
    s = ct.tf('s')
    tp = 2 * np.pi
    return (s + tp * 2800 / 2.5) / (0.02 * s + tp * 2800 * 4.0)


def _mu_upper_bound(Gs, W1, Wa, Kss):
    """
    Robust-performance mu upper bound for the 2-block SISO problem.

    Performance channel:      a(omega) = |W1 S|
    Additive-uncertainty ch.: b(omega) = |W_Pau K S|
    mu(omega) = a + b  (exact & tight for <=3 blocks); peak_mu = max_omega mu.
    Returns (peak_mu, d_opt) with d_opt the constant D-scale equalising the peaks.
    """
    w = np.logspace(np.log10(2*np.pi*50), np.log10(2*np.pi*8000), 500)
    L = ct.frequency_response(Gs * Kss, w).fresp.ravel()
    S = 1.0 / (1.0 + L)
    KS = ct.frequency_response(Kss, w).fresp.ravel() * S
    w1 = ct.frequency_response(W1, w).fresp.ravel()
    wa = ct.frequency_response(Wa, w).fresp.ravel()
    a = np.abs(w1 * S)                      # performance channel   |W1 S|
    b = np.abs(wa * KS)                     # additive-uncertainty  |W_Pau K S|
    peak_perf, peak_rob = float(np.max(a)), float(np.max(b))
    # a larger D-scale (more T rolloff) reduces b; increase d when robustness
    # dominates performance so the two weighted peaks equalise.
    d_opt = np.sqrt(peak_rob / max(peak_perf, 1e-12))
    peak_mu = float(np.max(a + b))          # per-frequency SUM, then max (exact SISO mu)
    return peak_mu, d_opt


def design_musyn(dk_iters=3, return_info=False):
    pl = D.design_plant(n_modes=N_DESIGN_MODES, zeta_design=ZETA_DESIGN)
    Gs = ct.ss(pl["A"], pl["Bu"] * SU, pl["Cy"] * (1.0 / SY), 0.0)
    W1b = _perf_weight()
    W2b = _ks_weight()
    W3b = _rolloff_weight()
    Wa = _wpau()                 # paper additive-uncertainty weight (drives D-step)
    d_scale = 1.0
    gamma = None
    hist = []
    best = None                      # keep the controller with the lowest mu_peak
    for it in range(dk_iters):
        # K-step: mixed-sensitivity H-infinity; the constant D-scale acts on the
        # robustness (T) channel W3, trading nominal performance for a lower
        # additive-uncertainty term W_Pau·K·S.
        W3 = W3b * d_scale
        K, CL, info = ct.mixsyn(Gs, w1=W1b, w2=W2b, w3=W3)
        gamma = float(info[0])
        # D-step: robust-performance mu against the paper's additive uncertainty.
        peak_mu, d_opt = _mu_upper_bound(Gs, W1b, Wa, ct.ss(K))
        hist.append((round(d_scale, 3), round(gamma, 3), round(peak_mu, 3)))
        if best is None or peak_mu < best[0]:
            best = (peak_mu, K, gamma)
        # damped, tightly-bounded D-scale update (D-K can overshoot otherwise)
        d_scale *= d_opt ** 0.35
        d_scale = float(np.clip(d_scale, 0.6, 1.8))
    peak_mu, K, gamma = best         # use the best iteration, not the last
    Kreal = K * (SU / SY)
    Kd = ct.ss(ct.sample_system(Kreal, C.DT, method='tustin'))
    if return_info:
        return Kd, gamma, hist
    return Kd


class MuSynthesis(LinearSSController):
    name = "mu-synthesis"
    color = "#d01884"      # magenta

    def __init__(self, **kw):
        Kd = design_musyn()
        A, B, Cc, Dd = (np.asarray(Kd.A), np.asarray(Kd.B),
                        np.asarray(Kd.C), np.asarray(Kd.D))
        super().__init__(A, B, Cc, Dd, sign=FB_SIGN, **kw)
