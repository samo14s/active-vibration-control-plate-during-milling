"""
Sliding-Mode Control (SMC) — state-feedback (regular-form) design.

Why not an output sliding surface here
--------------------------------------
On the paper's exact geometry the piezo actuator (left-lower) and the
displacement sensor (right-upper) sit on opposite sides of the plate, so the
non-collocated transfer function u -> y is NON-MINIMUM-PHASE (a right-half-plane
zero at ~5 kHz).  A high-gain *output* sliding surface  s = y_dot + lam y  is then
destabilised by the RHP zero exactly as high-gain output feedback would be, so the
output-SMC of the collocated case simply diverges.  State feedback, however, can
still stabilise a non-minimum-phase plant, so we design a *state* sliding mode:

  1. a steady-state Kalman observer reconstructs the full modal state x = [q; q']
     from the noisy displacement y (observability is unaffected by the RHP zero);
  2. the sliding surface  s = S x  is designed in REGULAR FORM: the plant is
     rotated so the (scalar) piezo input appears in the last coordinate, and the
     remaining (2n-1) "sliding" coordinates are given damped closed-loop poles by
     pole placement (design damping `zeta_s`).  On s = 0 the state therefore slides
     along a well-damped manifold — the damping the lightly-damped plate lacks;
  3. the control is equivalent control + boundary-layer reaching law

        u = -(S B)^{-1}( S A x_hat  +  ks·s  +  eta·sat(s/phi) ).

     The S A x_hat term keeps the state on the surface (equivalent control); the
     ks·s and eta·sat(s/phi) terms are the reaching law — ks injects extra
     damping inside the boundary layer and eta rejects *matched* uncertainty.

Gains tuned by PSO (particle-swarm optimisation)
------------------------------------------------
The default gains were found by PSO (src/tuning_pso.py) minimising a ROBUST cost —
the settled vibration summed over a fine sweep of the milling-force coefficient
(alpha4 0.3..2.9x) at the paper's reduced damping, with heavy penalties for
divergence, saturation and any still-growing (slowly diverging) response, and with
the gains rounded to 4 significant figures so only IMPLEMENTABLE tunings are
credited.  PSO found that a heavily-damped sliding surface (zeta_s ~ 0.43) with a
strong switching gain (eta ~ 350 V, versus the ~60 V a hand grid settled on) gives
the reaching law enough authority to reject the regenerative force across the WHOLE
alpha4 range: the tuned SMC holds ~5.6 um flat over alpha4 in [0.3, 2.9] at only
~20 V, and survives the full 20.4 s moving pass (~6 um) — the best of the six
controllers here.  The boundary layer phi replaces sign() to avoid chattering the
piezo stack could not follow.

(This is why manual tuning matters: the SAME state-feedback SMC structure looked
non-robust with hand-picked gains but is best-in-class once PSO-tuned.  The
output-feedback PID and the plant-inverting ADRC, by contrast, could NOT be made
robust by PSO on this non-minimum-phase plant — a structural, not tuning, limit.)
"""

from __future__ import annotations
import numpy as np
import scipy.linalg as sla
from scipy.signal import place_poles
from .base import Controller
from .. import config as C
from .. import design as D


def design_sliding_surface(A, Bu, zeta_s):
    """Regular-form sliding surface S (1 x 2n) placing the (2n-1) sliding poles at
    damped versions of the modal frequencies (design damping zeta_s)."""
    nx = A.shape[0]
    n = nx // 2
    # regular form: orthonormal T whose LAST column is Bu/|Bu| (QR completion), so
    # T^T Bu = [0, ..., 0, b]^T and the input appears only in the last coordinate.
    b = Bu.ravel()
    bn = b / np.linalg.norm(b)
    M = np.eye(nx)
    M[:, -1] = bn
    Q, _ = np.linalg.qr(M)
    if np.dot(Q[:, -1], bn) < 0:
        Q[:, -1] *= -1
    T = Q
    Ar = T.T @ A @ T
    A11 = Ar[:nx - 1, :nx - 1]
    A12 = Ar[:nx - 1, nx - 1:nx]
    # desired sliding poles (2n-1): damped complex pairs for modes 1..n-1, one real
    # pole for the highest mode (keeps every complex pole with its conjugate).
    om = C.OMEGA_N[:n]
    poles = []
    for w in om[:n - 1]:
        wd = w * np.sqrt(max(1.0 - zeta_s ** 2, 0.0))
        poles += [(-zeta_s * w) + 1j * wd, (-zeta_s * w) - 1j * wd]
    poles.append(-zeta_s * om[-1])
    poles = np.array(poles[:nx - 1])
    F = place_poles(A11, A12, poles).gain_matrix.ravel()      # z2 = -F z1
    Sr = np.hstack([F, 1.0])                                   # surface in regular coords
    S = Sr @ T.T                                              # back to physical coords
    return S.reshape(1, -1)


class SMC(Controller):
    name = "SMC"
    color = "#188038"      # green

    def __init__(self, zeta_s=0.427, ks=1.276e4, eta=347.3, phi=0.0699,
                 kf_q=5e-2, kf_r=None, n_modes=3, **kw):
        self.zeta_s = zeta_s      # sliding-manifold design damping
        self.ks = ks              # linear reaching gain (damping inside boundary layer)
        self.eta = eta            # switching (matched-uncertainty) gain [V]
        self.phi = phi            # boundary-layer half-width (on the scaled surface)
        self.kf_q = kf_q
        self.kf_r = (C.MEAS_NOISE_STD ** 2) if kf_r is None else kf_r
        self.n_modes = n_modes
        super().__init__(**kw)
        self._design()

    def _design(self):
        pl = D.design_plant(n_modes=self.n_modes)
        A, Bu, Cy = pl["A"], pl["Bu"], pl["Cy"]
        self.A, self.Bu, self.Cy = A, Bu, Cy
        self.n = pl["n"]
        self.nx = 2 * pl["n"]
        S = design_sliding_surface(A, Bu, self.zeta_s).ravel()
        SB = float(S @ Bu.ravel())
        # normalise by S B so the equivalent control is  u_eq = -(S A / S B) x  and
        # the reaching gains ks, eta act directly in volts.
        self.Sn = S / SB
        self.SAn = (S @ A).ravel() / SB
        self.Ad, self.Bd = D.c2d(A, Bu, self.dt)
        self.L = D.kalman_gain(self.Ad, Cy, self.kf_q, self.kf_r)

    def reset(self):
        super().reset()
        if hasattr(self, "nx"):
            self.xhat = np.zeros(self.nx)

    @staticmethod
    def _sat(x):
        return np.clip(x, -1.0, 1.0)

    def _update(self, t, y):
        xh = self.xhat
        s = float(self.Sn @ xh)
        u_eq = -float(self.SAn @ xh)
        u_reach = -(self.ks * s + self.eta * self._sat(s / self.phi))
        u = float(np.clip(u_eq + u_reach, -self.u_max, self.u_max))
        # observer update with the applied (saturated) input
        yhat = float((self.Cy @ xh).item())
        self.xhat = self.Ad @ xh + self.Bd.flatten() * u + self.L.flatten() * (y - yhat)
        return u
