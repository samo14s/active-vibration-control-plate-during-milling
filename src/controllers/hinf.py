"""
H-infinity controller via mixed-sensitivity loop shaping (S/KS/T).

Following the paper's philosophy, the controller is synthesised on a REDUCED
2-mode design plant (modes 1 & 2, which carry the milling response); the higher
modes are left as un-modelled dynamics that the robustness weight W3 must roll
the loop off before.  We minimise the H-infinity norm of the stacked cost

        || [ W1 S ; W2 KS ; W3 T ] ||_inf

  * W1 (performance / sensitivity) — high gain from DC through the two
    resonances (540 & 1068 Hz) to force strong disturbance rejection there;
  * W2 (control effort / KS)       — sets actuator usage;
  * W3 (robustness / T)            — high-pass forcing loop rolloff around 2 kHz,
    below mode 3 (2787 Hz), so the un-modelled high modes are not excited.

Numerics
--------
The plant maps volts (O(1e2)) to metres (O(1e-5)); that span wrecks the Riccati
conditioning, so the plant is scaled (output SY, input SU) before synthesis and
the controller scaled back by the scalar SU/SY.  Designing on the reduced model
keeps the controller low order and cleanly discretisable at the control rate.
"""

from __future__ import annotations
import numpy as np
import control as ct

from .base import LinearSSController
from .. import config as C
from .. import design as D

SY = 1.0e-4      # output scale (100 um)
SU = 1.0e2       # input  scale (100 V)
N_DESIGN_MODES = 2
# Design damping (robustification): design against a well-damped model so the
# controller is a broadband damper, not a sharp inverter of the lightly damped
# resonances -> it stays stable when the true modal frequencies drift.
ZETA_DESIGN = 0.15


def _default_weights():
    s = ct.tf('s')
    tp = 2 * np.pi
    # W1: performance/sensitivity (no pure integrator; A=0.3)
    Ms, wb, A = 2.0, tp * 1200.0, 0.3
    W1 = (s / Ms + wb) / (s + wb * A)
    # W2: control effort
    ku, w2z, w2p = 0.05, tp * 2000.0, tp * 8000.0
    W2 = ku * (s + w2z) / (s + w2p)
    # W3: complementary-sensitivity rolloff ~2 kHz (below mode 3 at 2787 Hz)
    Mt, wt = 2.5, tp * 2000.0
    W3 = (s + wt / Mt) / (0.02 * s + wt * 4.0)
    return W1, W2, W3


def design_hinf(weights=None, return_info=False):
    pl = D.design_plant(n_modes=N_DESIGN_MODES, zeta_design=ZETA_DESIGN)
    Gs = ct.ss(pl["A"], pl["Bu"] * SU, pl["Cy"] * (1.0 / SY), 0.0)
    W1, W2, W3 = _default_weights() if weights is None else weights
    K, CL, info = ct.mixsyn(Gs, w1=W1, w2=W2, w3=W3)
    gamma = float(info[0])
    Kreal = K * (SU / SY)                       # undo scaling
    Kd = ct.ss(ct.sample_system(Kreal, C.DT, method='tustin'))
    if return_info:
        return Kd, Kreal, gamma
    return Kd


class Hinf(LinearSSController):
    name = "H-infinity"
    color = "#1a73e8"      # blue

    def __init__(self, **kw):
        Kd = design_hinf()
        A, B, Cc, Dd = (np.asarray(Kd.A), np.asarray(Kd.B),
                        np.asarray(Kd.C), np.asarray(Kd.D))
        super().__init__(A, B, Cc, Dd, sign=+1.0, **kw)
