"""Milling force model with regenerative coupling to the flexible plate.

Convention (thin-wall side milling, workpiece-flexible):

* The plate deflection w at the cutting point is positive TOWARD the tool
  center (out of the material): if the wall bulges toward the tool, the
  tooth immersion — hence the chip thickness — increases.
* Tooth angle phi_j is measured from the surface-normal axis; the tooth is
  cutting when phi_st <= phi_j (mod 2 pi) <= phi_ex.
* Chip thickness  h = fz sin(phi) + [w(t) - w(t - tau)] cos(phi)
  (surface-normal motion projected on the tooth radial direction).
* Elemental forces dFt = Kt ap h, dFr = Kr dFt.  Their resultant on the
  WORKPIECE along the surface normal (positive toward the tool) is
      dFn = -(dFt sin(phi) + dFr cos(phi)),
  i.e. the mean cutting force pushes the wall away from the tool, and the
  regenerative gain dFn/dw is negative (force opposes penetration).  This
  reproduces classical zero-order-approximation (ZOA) stability lobes,
  verified in tests/test_milling.py.
"""
from __future__ import annotations

import numpy as np

from .params import MillingParams


class MillingForce:
    """Instantaneous plate-normal milling force with regeneration."""

    def __init__(self, mil: MillingParams):
        self.m = mil
        self.phi_st, self.phi_ex = mil.engagement_angles()
        # per-tooth radial runout: chip-thickness offset of tooth j is the
        # radius difference to the preceding tooth
        j = np.arange(mil.n_teeth)
        rho_j = mil.runout * np.cos(2.0 * np.pi * j / mil.n_teeth)
        self.dr = rho_j - np.roll(rho_j, 1)

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
            F += -(dFt * np.sin(phi) + m.Kr * dFt * np.cos(phi))
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
        h_cap = m.ae + self._max_miss * m.fz   # radially available material
        for j, phi in enumerate(phis):
            if not (self.phi_st <= phi <= self.phi_ex):
                continue
            in_window = True
            mm = self._m
            h = mm * m.fz * np.sin(phi) \
                + (w_now - w_lookup(mm)) * np.cos(phi) + self.dr[j]
            if h <= 0.0:
                continue
            self._cut_in_window = True
            dFt = m.Kt * ap * min(h, h_cap)
            F += -(dFt * np.sin(phi) + m.Kr * dFt * np.cos(phi))
        if self._in_window_prev and not in_window:
            # engagement window just ended: update the surface memory
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
            G(phi) = -sin(phi)^2 / 2 - Kr (phi / 2 + sin(2 phi) / 4),
        which is negative for realistic engagements (force opposes
        penetration).
        """
        Kr = self.m.Kr

        def G(phi: float) -> float:
            return (-0.5 * np.sin(phi) ** 2
                    - Kr * (phi / 2.0 + np.sin(2.0 * phi) / 4.0))

        return self.m.n_teeth * (G(self.phi_ex) - G(self.phi_st))

    def zoa_lobes(self, frf_nn, f_grid: np.ndarray,
                  k_lobes: int = 40) -> tuple[np.ndarray, np.ndarray]:
        """ZOA stability lobes for the single (workpiece-normal) direction.

        Characteristic equation  1 = ap C (1 - e^{-j w tau}) H(jw),
        C = Kt alpha / (2 pi) (< 0), H = frf_nn.  For each candidate
        chatter frequency, the depth limit is  ap = pi / (Kt alpha Re[H]),
        positive where Re[H] < 0, and the phase condition
        tan(theta/2) = Re[H] / Im[H], tau = (theta + 2 k pi) / w  maps each
        lobe k to a spindle speed.

        Parameters
        ----------
        frf_nn : callable omega[rad/s] -> complex FRF, point force at the
            cutting point to deflection at the cutting point (closed- or
            open-loop).
        f_grid : candidate chatter frequencies [Hz].

        Returns (rpm, ap_lim) scatter points (all lobes, unenveloped).
        """
        m = self.m
        alpha = self.zoa_alpha_nn()
        w = 2.0 * np.pi * f_grid
        H = frf_nn(w)
        G, I = np.real(H), np.imag(H)
        mask = G < 0.0
        w, G, I = w[mask], G[mask], I[mask]
        ap_lim = np.pi / (m.Kt * alpha * G)          # > 0 since alpha, G < 0
        half = np.arctan2(G, I)                      # theta/2 branch
        theta = np.mod(2.0 * half, 2.0 * np.pi)
        rpms, aps = [], []
        for k in range(k_lobes):
            tau = (theta + 2.0 * np.pi * k) / w
            rpm = 60.0 / (m.n_teeth * tau)
            rpms.append(rpm)
            aps.append(ap_lim)
        return np.concatenate(rpms), np.concatenate(aps)

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

    @staticmethod
    def lobes_envelope(rpm: np.ndarray, ap: np.ndarray,
                       rpm_range: tuple[float, float] = (5000.0, 12000.0),
                       n_bins: int = 300) -> tuple[np.ndarray, np.ndarray]:
        """Lower envelope of the lobe scatter on an rpm grid."""
        grid = np.linspace(rpm_range[0], rpm_range[1], n_bins)
        env = np.full(n_bins, np.inf)
        inside = (rpm >= rpm_range[0]) & (rpm <= rpm_range[1]) & np.isfinite(ap)
        idx = np.clip(np.searchsorted(grid, rpm[inside]) - 1, 0, n_bins - 1)
        for i, a in zip(idx, ap[inside]):
            env[i] = min(env[i], a)
        return grid, env
