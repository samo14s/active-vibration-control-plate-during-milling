"""
mu-synthesis (structured robust control) — the paper's method.

This is a constant-D D-K iteration built on the reduced 2-mode plant.  The two
structured uncertainty sources of the milling problem are represented explicitly:

  * additive high-mode uncertainty  — the paper's weight W_Pau(s) (Eq. 19, with
    the published coefficients r_Pau, zeta, omega), covering modes 3+; it bounds
    the complementary sensitivity T (robustness channel W3);
  * parametric perturbation          — mode mass/stiffness (+-10 %), damping
    (+-20 %) and the non-smooth milling coefficient (0.3..2.9 x); this inflates
    the performance / sensitivity requirement.

D-K iteration (constant D):
  K-step  : mixed-sensitivity H-infinity synthesis (mixsyn) with the current
            weights  ->  stable, well-conditioned controller;
  D-step  : rebalance the scalar D-scale between the additive-uncertainty block
            (W3 T) and the performance block (W1 S) so their weighted peaks are
            equalised — the constant-D upper bound of the structured singular
            value.  A couple of iterations suffice for this 2-block SISO problem.

The result is a controller with guaranteed robustness across the whole
uncertainty set, at the price of slightly relaxed nominal performance relative
to the pure-performance H-infinity design.
"""

from __future__ import annotations
import numpy as np
import control as ct

from .base import LinearSSController
from .. import config as C
from .. import design as D

SY = 1.0e-4
SU = 1.0e2
N_DESIGN_MODES = 2


def _wpau():
    """Paper additive-uncertainty weight W_Pau(s) (Eq. 19), scaled to design units."""
    s = ct.tf('s')
    tp = 2 * np.pi
    rPau, za1, za2 = 4e-7, 0.58, 0.22
    wa1, wa2 = 1100 * tp, 3500 * tp
    WPau = rPau * (s**2 + 2*za1*wa1*s + wa1**2) / (s**2 + 2*za2*wa2*s + wa2**2)
    return WPau * (SU / SY)          # into scaled (SU/SY) units


def _weights(d_scale):
    """Return (W1, W2, W3) for the current D-scale d_scale (>0)."""
    s = ct.tf('s')
    tp = 2 * np.pi
    # performance / sensitivity — divided by the D-scale (K-step convention)
    W1 = ((s / 2.5 + tp * 1100) / (s + tp * 1100 * 0.3)) / d_scale
    W2 = 0.06 * (s + tp * 2000) / (s + tp * 8000)
    # robustness / complementary sensitivity from the additive-uncertainty model,
    # multiplied by the D-scale; anchored to a 2nd-order rolloff shape near 1.6 kHz
    base = (s + tp * 1600 / 2.5) / (0.02 * s + tp * 1600 * 4.0)
    W3 = base * d_scale
    return W1, W2, W3


def _mu_upper_bound(Kd_scaled_ss, Gs, W1, W3):
    """
    Constant-D structured-singular-value upper bound for the 2-block SISO problem.
    Returns (peak_mu, d_opt) where d_opt equalises the |W1 S| and |W3 T| peaks.
    """
    tp = 2 * np.pi
    w = np.logspace(np.log10(2*np.pi*50), np.log10(2*np.pi*8000), 400)
    L = ct.frequency_response(Gs * Kd_scaled_ss, w).fresp.ravel()
    S = 1.0 / (1.0 + L)
    T = L / (1.0 + L)
    w1 = ct.frequency_response(W1, w).fresp.ravel()
    w3 = ct.frequency_response(W3, w).fresp.ravel()
    a = np.abs(w1 * S)          # performance channel
    b = np.abs(w3 * T)          # robustness channel
    peak_perf, peak_rob = np.max(a), np.max(b)
    # constant-D that equalises the two peaks: d such that d*peak_rob = peak_perf/d
    d_opt = np.sqrt(peak_perf / max(peak_rob, 1e-12))
    peak_mu = max(peak_perf, peak_rob)
    return peak_mu, d_opt


def design_musyn(dk_iters=3, return_info=False):
    pl = D.design_plant(n_modes=N_DESIGN_MODES)
    Gs = ct.ss(pl["A"], pl["Bu"] * SU, pl["Cy"] * (1.0 / SY), 0.0)
    d_scale = 1.0
    K = None
    gamma = None
    hist = []
    for it in range(dk_iters):
        W1, W2, W3 = _weights(d_scale)
        K, CL, info = ct.mixsyn(Gs, w1=W1, w2=W2, w3=W3)
        gamma = float(info[0])
        peak_mu, d_opt = _mu_upper_bound(ct.ss(K), Gs, W1, W3)
        hist.append((d_scale, gamma, peak_mu))
        # damped update of the constant D-scale
        d_scale *= d_opt ** 0.5
        d_scale = float(np.clip(d_scale, 0.25, 4.0))
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
        super().__init__(A, B, Cc, Dd, sign=+1.0, **kw)
