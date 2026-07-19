"""Milling force model with regenerative coupling to the flexible plate.

Convention (thin-wall side milling, workpiece-flexible):

* The plate deflection w at the cutting point is positive TOWARD the tool
  center (out of the material): if the wall bulges toward the tool, the
  tooth immersion — hence the chip thickness — increases.
* Tooth angle phi_j is measured from the surface-normal axis; the tooth is
  cutting when phi_st <= phi_j (mod 2 pi) <= phi_ex.
* Chip thickness  h = fz sin(phi) + [w(t) - w(t - tau)] cos(phi)
  (surface-normal motion projected on the tooth radial direction).
* Elemental forces dFt = Kt ap h, dFr = Kr dFt.  With the tool-center
  frame derivation (tooth at P = R(sin phi, -cos phi) from the center,
  cutting velocity along (cos phi, sin phi)): the tangential force drags
  the workpiece material along the cutting velocity (+dFt sin(phi) on
  the wall normal), while the radial ploughing pushes the wall outward
  from the tool center (-dFr cos(phi)).  The resultant on the WORKPIECE
  along the surface normal (positive toward the tool) is
      dFn = dFt sin(phi) - dFr cos(phi) = Kt ap h (sin(phi) - Kr cos(phi)),
  whose engagement-window mean is negative for the low-immersion cuts
  considered (the wall is pushed away), and whose orientation-averaged
  regenerative gain alpha is negative (force opposes penetration).
  Consistency of the sign structure with linear chatter theory is
  verified against the direct Nyquist criterion in tests/test_milling.py.

Only up-milling is supported: under the single-normal-direction
workpiece-flexible convention the down-milling window would need a
re-derivation of the projection signs.
"""
from __future__ import annotations

import numpy as np

from .params import MillingParams


class MillingForce:
    """Instantaneous plate-normal milling force with regeneration."""

    def __init__(self, mil: MillingParams):
        if not mil.up_milling:
            raise NotImplementedError(
                "down-milling is not supported by the single-normal "
                "workpiece-flexible force convention")
        self.m = mil
        self.phi_st, self.phi_ex = mil.engagement_angles()
        # per-tooth radial runout: tooth radii rho_j; the chip-thickness
        # offset is the radius difference to the tooth that generated the
        # reference surface (m passes earlier)
        j = np.arange(mil.n_teeth)
        self.rho_j = mil.runout * np.cos(2.0 * np.pi * j / mil.n_teeth)
        self.dr = self.rho_j - np.roll(self.rho_j, 1)

    def tooth_angles(self, t: float) -> np.ndarray:
        m = self.m
        base = m.omega_spindle * t
        return base + np.arange(m.n_teeth) * m.phi_pitch

    def normal_force(self, t: float, w_now: float = 0.0,
                     w_tau: float = 0.0, ap: float | None = None) -> float:
        """Force on the workpiece along +w (toward the tool) [N]."""
        m = self.m
        ap = m.ap if ap is None else ap
        phis = np.mod(self.tooth_angles(t), 2.0 * np.pi)
        F = 0.0
        for j, phi in enumerate(phis):
            if not (self.phi_st <= phi <= self.phi_ex):
                continue
            h = m.fz * np.sin(phi) + (w_now - w_tau) * np.cos(phi) \
                + self.dr[j]
            if h <= 0.0:
                continue  # tooth jumped out of cut
            dFt = m.Kt * ap * h
            F += dFt * (np.sin(phi) - m.Kr * np.cos(phi))
        return F

    def normal_force_series(self, t: np.ndarray) -> np.ndarray:
        """Rigid-workpiece force history (no regeneration)."""
        return np.array([self.normal_force(tk) for tk in t])

    # ----------------------------- multi-delay surface-memory regeneration

    def reset_regen(self, max_miss: int = 8):
        """Reset the surface-memory state for a new simulation.

        When the vibration amplitude approaches the feed per tooth, teeth
        jump out of the cut and the machined surface is left by an OLDER
        pass: the regenerative delay becomes m tau and the accumulated
        feed m fz, with m the number of consecutive missed passes.  This
        surface memory is the physical mechanism that bounds
        large-amplitude (saturated) chatter; a single-delay model without
        it diverges when the cut is strongly supercritical."""
        self._m = 1
        self._cut_in_window = False
        self._in_window_prev = False
        self._max_miss = max_miss

    def normal_force_regen(self, t: float, w_now: float, w_lookup,
                           ap: float | None = None) -> float:
        """Force on the workpiece with surface-memory regeneration.

        w_lookup(m) must return the plate deflection at the cutting point
        at time t - m*tau.  Call reset_regen() before a simulation."""
        m = self.m
        ap = m.ap if ap is None else ap
        phis = np.mod(self.tooth_angles(t), 2.0 * np.pi)
        F = 0.0
        in_window = False
        for j, phi in enumerate(phis):
            if not (self.phi_st <= phi <= self.phi_ex):
                continue
            in_window = True
            mm = self._m
            # runout offset vs the tooth that left the reference surface
            dr = self.rho_j[j] - self.rho_j[(j - mm) % m.n_teeth]
            h = mm * m.fz * np.sin(phi) \
                + (w_now - w_lookup(mm)) * np.cos(phi) + dr
            if h <= 0.0:
                continue
            self._cut_in_window = True
            # radially available material bounds the chip thickness
            h_cap = m.ae + mm * m.fz
            dFt = m.Kt * ap * min(h, h_cap)
            F += dFt * (np.sin(phi) - m.Kr * np.cos(phi))
        if self._in_window_prev and not in_window:
            # engagement window just ended: update the surface memory.
            # Approximation (documented): a single miss counter per window;
            # any positive-h event in the window renews the surface.
            self._m = 1 if self._cut_in_window \
                else min(self._m + 1, self._max_miss)
            self._cut_in_window = False
        self._in_window_prev = in_window
        return F

    # ------------------------------------------------- ZOA stability lobes

    def zoa_alpha_nn(self) -> float:
        """Orientation-averaged regenerative directional factor.

        Linearizing the force law about the static chip thickness and
        averaging over one rotation with N teeth gives
            F_reg = (ap Kt alpha / 2 pi) [w(t) - w(t - tau)],
            alpha = N * [G(phi_ex) - G(phi_st)],
            G(phi) = sin(phi)^2 / 2 - Kr (phi / 2 + sin(2 phi) / 4),
        which is negative for low-immersion engagements (the radial
        ploughing term dominates: force opposes penetration).
        """
        Kr = self.m.Kr

        def G(phi: float) -> float:
            return (0.5 * np.sin(phi) ** 2
                    - Kr * (phi / 2.0 + np.sin(2.0 * phi) / 4.0))

        return self.m.n_teeth * (G(self.phi_ex) - G(self.phi_st))

    def a_crit_nyquist(self, frf_nn, rpm: float,
                       f_lo: float = 50.0, f_hi: float = 3000.0,
                       nf: int = 60000) -> float:
        """Critical axial depth at a given spindle speed by the direct
        Nyquist criterion on  L(w) = ap C (1 - e^{-j w tau}) H(jw),
        C = Kt alpha / 2 pi:  the loop is on the stability boundary when
        the locus crosses +1, i.e.  a_crit = 1 / max{Re L1 : Im L1 = 0},
        with L1 the locus per unit depth.  Robust for multi-mode FRFs and
        closed-loop FRFs alike (unlike the lobe-scatter mapping, which is
        fragile near stability pockets); validated against the linear
        time-domain growth threshold within ~6 % (tests/test_milling.py).
        """
        m = self.m
        tau = 60.0 / (rpm * m.n_teeth)
        alpha = self.zoa_alpha_nn()
        w = 2.0 * np.pi * np.linspace(f_lo, f_hi, nf)
        L1 = (m.Kt * alpha / (2.0 * np.pi)) * (1.0 - np.exp(-1j * w * tau)) \
            * frf_nn(w)
        im, re = np.imag(L1), np.real(L1)
        best = 0.0
        crossings = np.where(np.sign(im[:-1]) * np.sign(im[1:]) < 0)[0]
        for i in crossings:
            t = im[i] / (im[i] - im[i + 1])
            r = re[i] + t * (re[i + 1] - re[i])
            best = max(best, r)
        return 1.0 / best if best > 0.0 else np.inf

    def a_crit_sweep(self, frf_nn, rpms: np.ndarray, **kw) -> np.ndarray:
        return np.array([self.a_crit_nyquist(frf_nn, r, **kw) for r in rpms])
