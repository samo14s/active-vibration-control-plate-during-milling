"""
AST-SMC — Adaptive Super-Twisting Sliding-Mode Control with an amplitude-
scheduled model-based regenerative feedforward (the MOVING-ENVELOPE controller).

The second developed controller, complementary to ATD-SMC (atdsmc.py): the two
form an honest PARETO PAIR — neither dominates the other, and each closes gaps
the other cannot.  AST-SMC is the one that closes the study's #1 open gap, the
FULL-MOVING-PASS force ceiling.

Structure (on the regular-form state sliding surface + Kalman observer of smc.py):

    u = u_eq + u_ff + u_reach
    u_reach = -k1*sqrt(|s|)*sign(s) - kl*s - eta*sat(s/phi_bl) + w
    w      <- w - k2*sat(s/phi_bl)*dt              (super-twisting integral state,
                                                    anti-windup clamped)
    u_ff   = -g * a4unit * proj * (Sn_force . phi_m)     (regen feedforward)
    proj   = phi_m . (q_hat - q_hat(t-tau))              (delay-buffer reconstruction
                                                    -> correct phase BY CONSTRUCTION)
    g      : amplitude-scheduled cutting-factor estimate —
             sbar = LPF(|s|);  g += dt*(gamma*max(0, sbar - s_dead) - sigma*(g - g0)),
             projected to [g_min, g_max]  (reversible, leaks back when quiescent)

What it uniquely achieves (all measured, scripts in the repo / scratch logs):
  * FULL 20.4 s moving-pass force ceiling alpha4 = 2.0 (6.4 um / 39 V) where
    ATD-SMC diverges already at 1.8 — a +25% lift of the study's hardest envelope.
    ABLATION-HONEST attribution: the credit is the SUPER-TWISTING INTEGRAL term
    (continuous disturbance rejection through the varying-coupling traverse) —
    the feedforward stayed at g=0 on the surviving passes, and a trajectory-aware
    milling direction gave IDENTICAL results to the static direction (both tested).
  * The combined extreme (alpha4=4.0, -40% damping) CLEANLY: 20.8 um / 76 V,
    where ATD-SMC is bounded but rail-saturated (45 um at 150 V) and SMC/TD-SMC
    diverge.  Here the g-schedule DOES engage (g -> 8) and carries the case.

What it honestly does NOT do (ATD-SMC remains the static/economy champion):
  * NEGATIVE frequency detuning blind spot: at -4% shift (-20% damping) the
    amplitude schedule misreads model mismatch as cutting force, winds g to its
    ceiling and rails the actuator (109-181 um at 150 V for -4..-8%), where
    ATD-SMC holds 6.6 um.  Material removal shifts frequencies UPWARD, so the
    physical drift direction is safe — but the blind spot is real and documented.
  * The tight 20 V actuator limit (needs ~29 V at alpha4=2.9 -> saturates), and
    nominal economy (~19-22 V vs ~10 V for ATD-SMC).

Lineage note: the structure is the design-panel 'adv_ST' variant (amplitude-
scheduled super-twisting) promoted verbatim after it won the moving-pass tests;
a soft-start ramp was tried and REJECTED (it fixes nothing here and breaks the
+17% frequency case by letting vibration build during the ramp).  Neither
super-twisting, amplitude-scheduled adaptation, nor delayed-force feedforward is
novel — the contribution is the combination and the measured envelope extension.
"""

from __future__ import annotations
import numpy as np

from .smc import SMC
from .. import config as C


class ASTSMC(SMC):
    name = "AST-SMC (developed)"
    color = "#d81b60"      # crimson

    def __init__(self, zeta_s=0.3882, k1=500.0, k2=1.0e4, kl=1125.0, phi_bl=0.24,
                 eta=254.0, gamma=1.5e6, g0=0.0, g_min=0.0, g_max=8.0, sigma=30.0,
                 s_dead=5.0e-4, tau_f=2.0e-3, moving=False, **kw):
        # super-twisting reaching gains
        self.k1 = k1          # sqrt(|s|) gain
        self.k2 = k2          # integral (w) gain
        self.kl = kl          # linear -kl*s damping
        self.phi_bl = phi_bl  # boundary-layer half-width (smooths sign -> sat)
        self._eta_r = eta     # bounded switching robustifier [V]
        # amplitude-scheduled regenerative feedforward
        self.gamma = gamma    # g growth rate per unit sliding-amplitude excess
        self.g0 = g0          # rest value of g
        self.g_min = g_min
        self.g_max = g_max
        self.sigma = sigma    # leakage rate back toward g0
        self.s_dead = s_dead  # amplitude dead-band below which g does not grow
        self.tau_f = tau_f    # |s| low-pass time constant
        self.moving = moving  # True: recompute phi_m from the known tool trajectory
        super().__init__(zeta_s=zeta_s, **kw)

    def _design(self):
        super()._design()
        self.phi_mill = C.phi_mill(C.MILL_POS_STATIC).ravel()[:self.n]
        self.a4unit = C.CUT_GAIN * C.ALPHA4_BASE * (C.DUTY / 2.0)   # factor = 1
        self.Sn_force = self.Sn.ravel()[self.n:]                    # force -> s row
        self.c_sf = float(self.Sn_force @ self.phi_mill)
        self.ndel = C.TAU_S / self.dt
        self.nbuf = int(np.ceil(self.ndel)) + 2

    def reset(self):
        super().reset()
        if hasattr(self, "nbuf"):
            self.w = 0.0
            self.g = self.g0
            self.sbar = 0.0
            self.qbuf = np.zeros((self.nbuf, self.n))
            self.kstep = 0

    def _delayed_q(self):
        idx = self.kstep - self.ndel
        if idx < 0:
            return np.zeros(self.n)
        i0 = int(np.floor(idx))
        fr = idx - i0
        return (1.0 - fr) * self.qbuf[i0 % self.nbuf] + fr * self.qbuf[(i0 + 1) % self.nbuf]

    def _update(self, t, y):
        xh = self.xhat
        s = float(self.Sn @ xh)
        u_eq = -float(self.SAn @ xh)

        # trajectory-aware milling direction (moving pass; the CNC knows x_T(t))
        if self.moving:
            prog = min(t / C.T_PASS, 1.0)
            self.phi_mill = C.phi_mill(prog).ravel()[:self.n]
            self.c_sf = float(self.Sn_force @ self.phi_mill)

        # amplitude-scheduled regenerative feedforward (correct phase via buffer)
        q = xh[:self.n]
        self.qbuf[self.kstep % self.nbuf] = q
        proj = float(self.phi_mill @ (q - self._delayed_q()))
        u_ff = -self.g * self.a4unit * proj * self.c_sf

        # super-twisting reaching law
        sg = float(np.clip(s / self.phi_bl, -1.0, 1.0))
        u_reach = -self.k1 * np.sqrt(abs(s)) * np.sign(s) - self.kl * s \
                  - self._eta_r * sg + self.w

        u = float(np.clip(u_eq + u_ff + u_reach, -self.u_max, self.u_max))

        # observer update with the applied (saturated) input
        yhat = float((self.Cy @ xh).item())
        self.xhat = self.Ad @ xh + self.Bd.flatten() * u + self.L.flatten() * (y - yhat)

        # super-twisting integral state (anti-windup clamped)
        self.w -= self.k2 * sg * self.dt
        if self.w > self.u_max:
            self.w = self.u_max
        elif self.w < -self.u_max:
            self.w = -self.u_max

        # amplitude schedule for the feedforward factor (reversible)
        self.sbar += (abs(s) - self.sbar) * (self.dt / self.tau_f)
        self.g += self.dt * (self.gamma * max(0.0, self.sbar - self.s_dead)
                             - self.sigma * (self.g - self.g0))
        if self.g < self.g_min:
            self.g = self.g_min
        elif self.g > self.g_max:
            self.g = self.g_max

        self.kstep += 1
        return u
