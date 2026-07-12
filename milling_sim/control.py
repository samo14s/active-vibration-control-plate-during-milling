"""Machining strategies: adaptive controller (Sec. 3) and the two
baselines of Sec. 4.1 (conventional fixed-parameter, SSV).

Adaptive controller structure (hierarchical coordination of Eq. (22)):
  1. spindle speed  — fast loop, gradient ascent on the stability margin
                      (Eqs. (16), (18)-(19)) with gain scheduling (Eq. (25))
                      and slew limit (Eq. (26)); emergency lobe-jump when
                      the chatter indicator fires;
  2. feed rate      — receding-horizon MPC toward the reference MRR
                      (Eq. (20));
  3. depth of cut   — slow vibration-regulating loop (Eq. (21)) capped by
                      the stability constraint ap <= eta * ap_lim (Eq. (17)).

The controller never reads the simulator's ground truth: it works from
the RLS-identified modal parameters (Sec. 3.1) and the measured vibration
and force signals, exactly as the physical system would.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .engine import IntervalMeasurement
from .identification import RLSModalIdentifier
from .parameters import PaperSetup
from .stability import fast_lobe_envelope


class BaseController:
    name = "base"

    def command(self, t: float) -> tuple[float, float, float]:
        raise NotImplementedError

    def observe(self, meas: IntervalMeasurement) -> None:
        pass


# ---------------------------------------------------------------------------
# Conventional baseline: offline-optimised constants + operator emulation.
# ---------------------------------------------------------------------------
class FixedParameterController(BaseController):
    """Fixed parameters; "conservative parameter reductions were manually
    implemented when visible chatter occurred" (Sec. 4.1)."""

    name = "conventional"

    def __init__(self, setup: PaperSetup):
        b = setup.baseline
        self.b = b
        self.n = b.n_rpm
        self.vf = b.vf_mm_min
        self.ap = b.ap_mm
        self._over_since: float | None = None
        self._cooldown_until = 0.0

    def command(self, t: float) -> tuple[float, float, float]:
        return self.n, self.vf, self.ap

    def observe(self, meas: IntervalMeasurement) -> None:
        b = self.b
        if meas.rms_disp_um > b.operator_rms_threshold_um:
            if self._over_since is None:
                self._over_since = meas.t
            visible_for = meas.t - self._over_since
            if (visible_for > b.operator_reaction_s
                    and meas.t > self._cooldown_until):
                self.vf = max(b.operator_vf_floor_mm_min,
                              self.vf * b.operator_feed_factor)
                self.ap = max(b.operator_ap_floor_mm,
                              self.ap - b.operator_ap_step_mm)
                self._cooldown_until = meas.t + 2.0
        else:
            self._over_since = None


# ---------------------------------------------------------------------------
# SSV baseline: +/-10 % sinusoidal spindle-speed modulation at 5 Hz.
# ---------------------------------------------------------------------------
class SSVController(FixedParameterController):
    """+/-10 % sinusoidal SSV command at 5 Hz; the realised speed is slew-
    limited by the spindle acceleration capability (Eq. (26)), as on the
    physical machine."""

    name = "ssv"

    def __init__(self, setup: PaperSetup):
        super().__init__(setup)
        self._n_act = self.n
        self._t_prev = 0.0
        self._n_dot_max = setup.limits.n_dot_max_rpm_s

    def command(self, t: float) -> tuple[float, float, float]:
        b = self.b
        n_cmd = self.n * (1.0 + b.ssv_amplitude
                          * math.sin(2.0 * math.pi * b.ssv_freq_hz * t))
        dt = max(t - self._t_prev, 1e-6)
        self._t_prev = t
        dn = np.clip(n_cmd - self._n_act,
                     -self._n_dot_max * dt, self._n_dot_max * dt)
        self._n_act = float(self._n_act + dn)
        return self._n_act, self.vf, self.ap


# ---------------------------------------------------------------------------
# Adaptive controller (Sec. 3).
# ---------------------------------------------------------------------------
@dataclass
class AdaptiveLog:
    t: list = None
    fn_id: list = None
    fn_true: list = None
    ap_lim: list = None
    margin: list = None
    p_inst: list = None

    def __post_init__(self):
        for f in ("t", "fn_id", "fn_true", "ap_lim", "margin", "p_inst"):
            setattr(self, f, [])


class AdaptiveController(BaseController):
    name = "adaptive"

    def __init__(self, setup: PaperSetup,
                 boundary_learner=None):
        self.setup = setup
        self.cp = setup.controller
        self.lim = setup.limits
        b = setup.baseline
        # start from the same offline-optimised point as the baseline
        self.n = b.n_rpm
        self.vf = b.vf_mm_min
        self.ap = b.ap_mm

        self.ident = RLSModalIdentifier(fs=setup.sim.accel_sample_hz)
        self.learner = boundary_learner

        # dead-reckoned removed-volume fraction: the NC program knows the
        # stock volume, and the controller integrates its own commanded
        # MRR - this drives the Eq. (6)-(7) model evolution directly
        self._Vr_cmd = 0.0
        self._stock_cm3 = setup.sim.removal_volume_cm3
        self._steps_since_id = 10 ** 9

        # calibrated initial modal model (from the "three to five
        # representative cutting passes for controller initialization")
        self.model_modes = [(m.f0_hz, m.zeta0, m.mass_kg) for m in setup.modes]
        self.model_dirs = [m.direction for m in setup.modes]
        self._alpha1 = setup.modes[0].alpha       # Eq. (6) sensitivity
        self._f0_wall = setup.modes[0].f0_hz

        self.n_grid = np.linspace(self.lim.n_min_rpm, self.lim.n_max_rpm, 141)
        self.ap_env_mm = None
        self._lobe_refresh = 0
        self._force_harmonics = self._tooth_force_harmonics()

        self.p_inst = 0.0
        self._rms_filt = 0.0
        self._slope_um_per_mm = 15.0              # dVrms/dap prior, Eq. (21)
        self._prev_ap = self.ap
        self._prev_rms = 0.0
        self._ap_timer = 0.0
        # strategic relocation state (slow supervisory loop, Sec. 3.1)
        self._n_target: float | None = None
        self._strat_timer = 0.0
        self._strat_holdoff = 0.0
        # feedback-driven trust relaxation of the model constraint:
        # sustained low vibration lets ap exceed the (imperfect) model
        # boundary; any instability indication snaps it back (the in-pass
        # counterpart of the paper's boundary-learning, Sec. 3.3)
        self._relax = 0.0
        self._chatter_streak = 0
        self.log = AdaptiveLog()

    # -- controller-side model update -------------------------------------
    def _update_model(self) -> None:
        """Refresh the modal model: Eq. (6)-(7) evolution driven by the
        dead-reckoned removed-volume fraction, refined by the RLS estimate
        when it is fresh and consistent.

        The RLS frequency can go stale when a spindle harmonic sits on the
        mode (the notch and the mode collide); the Vr-driven prediction
        carries the model through those blind intervals.  The RLS damping
        estimate reflects the CLOSED-LOOP poles (structure + regeneration),
        which tend to zero near the stability boundary, so the structural
        damping always comes from the calibrated Eq. (7) law.
        """
        Vr = self._Vr_cmd
        fn_pred = self._f0_wall * math.sqrt(
            max(0.05, 1.0 - self._alpha1 * Vr))          # Eq. (6)
        f1 = fn_pred
        if (self.ident.fn_hz is not None
                and self._steps_since_id < 150
                and abs(self.ident.fn_hz - fn_pred) < 0.08 * fn_pred):
            f1 = 0.7 * self.ident.fn_hz + 0.3 * fn_pred  # sensor fusion
        mm = []
        for i, m in enumerate(self.setup.modes):
            f0, z0, ma = m.f0_hz, m.zeta0, m.mass_kg
            if m.structure == "workpiece":
                z_hat = z0 * (1.0 + m.beta * Vr)         # Eq. (7)
                if i == 0:
                    mm.append((f1, z_hat, ma))
                else:
                    s = max(0.05, 1.0 - m.alpha * Vr)
                    mm.append((f0 * math.sqrt(s), z_hat, ma))
            else:
                mm.append((f0, z0, ma))
        self.model_modes = mm

    def _refresh_lobes(self) -> None:
        env_m = fast_lobe_envelope(self.setup.tool, self.setup.engagement,
                                   self.model_modes, self.n_grid,
                                   directions=self.model_dirs)
        env_mm = env_m * 1000.0
        if self.learner is not None:
            env_mm = env_mm * self.learner.correction(self.n_grid,
                                                      Vr=self._Vr_cmd)
        # smoothing across the grid stabilises the gradient, Eq. (19);
        # edge-padded so the boundary is not biased down at the grid ends
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        kernel /= kernel.sum()
        padded = np.pad(env_mm, (2, 2), mode="edge")
        self.ap_env_mm = np.convolve(padded, kernel, mode="same")[2:-2]

    def _ap_lim_at(self, n_rpm: float) -> float:
        return float(np.interp(n_rpm, self.n_grid, self.ap_env_mm))

    def _ap_cap_mm(self) -> float:
        """Depth of cut beyond which extra stability limit is not usable
        (machine envelope + no productivity gain past the MRR ceiling)."""
        return self.lim.ap_max_mm

    def _vf_max_at(self, n_rpm: float | np.ndarray):
        """Feed ceiling: drive limit and the D_surf feed-per-tooth cap."""
        ft_lim = (self.setup.tool.n_teeth * n_rpm * self.cp.ft_max_mm)
        return np.minimum(self.lim.vf_max_mm_min, ft_lim)

    def _tooth_force_harmonics(self, n_harm: int = 6) -> np.ndarray:
        """|W_j| Fourier magnitudes of the periodic y cutting force per unit
        (Kt ap ft): the excitation spectrum at tooth-passing harmonics used
        for the forced-response prediction."""
        tool = self.setup.tool
        phi_st, phi_ex = self.setup.engagement.entry_exit_angles(tool)
        N = tool.n_teeth
        m = 512
        phi = np.linspace(0.0, 2.0 * math.pi / N, m, endpoint=False)
        w = np.zeros(m)
        for j in range(N):
            pj = (phi + 2.0 * math.pi * j / N) % (2.0 * math.pi)
            g = (pj >= phi_st) & (pj <= phi_ex)
            s, c = np.sin(pj), np.cos(pj)
            w += g * s * (s - tool.Kr * c)
        W = np.fft.rfft(w) / m
        mags = 2.0 * np.abs(W[1: n_harm + 1])
        return mags

    def _ap_forced_mm(self, n_rpm: np.ndarray,
                      vf_mm_min: float | None = None) -> np.ndarray:
        """Depth of cut that keeps the PREDICTED forced vibration at the
        wall sensor within the Eq. (21) target, per candidate speed.

        Forced amplitude at harmonic j:  |W_j| Kt ap ft |Hw(i 2 pi j f_t)|;
        the model receptance uses the workpiece modes only (wall sensor).
        Robustness: the worst case over a +/-3 % natural-frequency band
        (the identification accuracy) is used, per the min-max formulation
        of Eq. (24).
        """
        tool = self.setup.tool
        N = tool.n_teeth
        n_rpm = np.asarray(n_rpm, float)
        f_tooth = n_rpm / 60.0 * N
        if vf_mm_min is None:
            ft_m = (self._vf_max_at(n_rpm) / 1000.0 / 60.0
                    / (N * n_rpm / 60.0))
        else:
            ft_m = np.full_like(f_tooth,
                                vf_mm_min / 1000.0 / 60.0) / (N * n_rpm / 60.0)
        rms2_per_ap2 = np.zeros_like(f_tooth)
        for j, Wj in enumerate(self._force_harmonics, start=1):
            wjw = 2.0 * math.pi * j * f_tooth
            Habs = np.zeros_like(f_tooth)
            for shift in (0.97, 1.0, 1.03):       # Eq. (24) worst case
                H = np.zeros_like(f_tooth, dtype=complex)
                for (fn, ze, ma), m0, d in zip(self.model_modes,
                                               self.setup.modes,
                                               self.model_dirs):
                    if d != "y" or m0.structure != "workpiece":
                        continue
                    wn = 2.0 * math.pi * fn * shift
                    H += 1.0 / (ma * (wn * wn - wjw * wjw)
                                + 1j * (2.0 * ze * ma * wn * wjw))
                Habs = np.maximum(Habs, np.abs(H))
            amp_per_ap = Wj * tool.Kt * ft_m * Habs
            rms2_per_ap2 += 0.5 * amp_per_ap ** 2
        rms_per_ap = np.sqrt(rms2_per_ap2)          # [m RMS per m of ap]
        target_m = self.cp.V_target_um * 1e-6
        with np.errstate(divide="ignore"):
            ap_m = np.where(rms_per_ap > 1e-12, target_m / rms_per_ap, 1.0)
        return np.clip(ap_m * 1000.0, 0.2, self.lim.ap_max_mm)

    def _achievable_q(self) -> np.ndarray:
        """Achievable-MRR score [cm^3/min] over the speed grid: the
        relocation objective (productivity term of Eq. (16)) under BOTH
        stability (Eq. (17)) and predicted forced-vibration constraints,
        plus the surface/feed cap.  A lobe peak is also a forced-resonance
        speed (tooth harmonic on the mode), so the usable depth is the
        minimum of the two boundaries - this is the multiobjective
        coordination of Eq. (16) in practice.  Deliberately NOT capped at
        the productivity ceiling so deeper pockets keep a higher score."""
        ae_mm = self.setup.engagement.ae_m * 1000.0
        ap_use = np.minimum(
            np.minimum(self.cp.eta * self.ap_env_mm, self._ap_cap_mm()),
            self._ap_forced_mm(self.n_grid))
        return ap_use * ae_mm * self._vf_max_at(self.n_grid) / 1000.0

    # -- main update --------------------------------------------------------
    def observe(self, meas: IntervalMeasurement) -> None:
        cp, lim = self.cp, self.lim
        dt_c = 1.0 / cp.update_rate_hz

        # dead-reckoned stock removal from the COMMANDED parameters
        ae_mm = self.setup.engagement.ae_m * 1000.0
        q_cmd = self.vf * self.ap * ae_mm / 1000.0       # cm^3/min
        self._Vr_cmd = min(1.0, self._Vr_cmd
                           + q_cmd * dt_c / 60.0 / self._stock_cm3)

        f_rev = meas.n_rpm / 60.0
        acc0 = self.ident.n_accepts
        self.ident.update(meas.id_samples, f_rev,
                          protect_hz=self.model_modes[0][0])
        self._steps_since_id = (0 if self.ident.n_accepts > acc0
                                else self._steps_since_id + 1)
        self._update_model()

        # low-pass the vibration measurement (0.1 s window equivalent)
        self._rms_filt = 0.9 * self._rms_filt + 0.1 * meas.rms_disp_um

        # instability penalty P_inst of Eq. (16): driven by the
        # NONSYNCHRONOUS residual (chatter energy), which reacts long
        # before the total RMS - the paper's variance-analysis detector
        chatter_um = self.ident.resid_rms * 1e6
        warmed_up = meas.t > 0.5                 # entry transients settle
        self.p_inst = (max(0.0, (chatter_um - 2.5) / 5.0)
                       if warmed_up else 0.0)
        # temporary retraction (Sec. 3.1 tool-path modifier) on a SUSTAINED
        # strong onset - a single hot window is not chatter
        self._chatter_streak = (self._chatter_streak + 1
                                if chatter_um > 6.0 else 0)
        if warmed_up and self._chatter_streak >= 3:
            self.ap = max(lim.ap_min_mm, self.ap * 0.85)

        if self.ap_env_mm is None or self._lobe_refresh <= 0:
            self._refresh_lobes()
            self._lobe_refresh = 10               # 10 Hz lobe refresh
        self._lobe_refresh -= 1

        ap_lim = self._ap_lim_at(self.n)
        margin = cp.eta * ap_lim - self.ap        # S_margin, Eqs. (16)-(17)

        # -- gain scheduling, Eq. (25) ------------------------------------
        sigma2 = min(self.ident.uncertainty, 0.05)
        alpha_n = (cp.alpha_n0 * math.exp(-cp.beta1 * min(self.p_inst, 2.0))
                   * math.exp(-cp.beta2 * sigma2))

        # -- 1. spindle speed --------------------------------------------
        # Two time scales (Sec. 3.1): a fast gradient/emergency loop
        # (Eqs. (18)-(19), (25)) and a slow supervisory relocation toward
        # the globally best lobe when the local one degrades.  The ramp
        # policy is gentler than the machine's Eq. (26) capability.
        dn_max = min(cp.n_slew_rpm_s, lim.n_dot_max_rpm_s) * dt_c
        self._strat_timer += dt_c
        q_ach = self._achievable_q()
        if self._strat_timer >= 1.0 and self.p_inst < 1.0:
            self._strat_timer = 0.0
            i_best = int(np.argmax(q_ach))
            n_best = float(self.n_grid[i_best])
            ae_mm = self.setup.engagement.ae_m * 1000.0
            q_now = self.vf * self.ap * ae_mm / 1000.0   # realised MRR
            q_best = min(float(q_ach[i_best]), self.cp.Q_cap_cm3_min)
            if (meas.t >= self._strat_holdoff
                    and q_best > min(1.2 * q_now + 1.0, q_now + 4.0)
                    and abs(n_best - self.n) > 400.0):
                self._n_target = n_best
                self._strat_holdoff = meas.t + 4.0
                self._relax = 0.0                 # fresh pocket: trust model
        if self.p_inst > 0.8:
            # fast correction: slew toward the best achievable-MRR lobe
            i_best = int(np.argmax(q_ach))
            self._n_target = float(self.n_grid[i_best])
        if self._n_target is not None:
            step = np.clip(self._n_target - self.n, -dn_max, dn_max)
            if abs(self._n_target - self.n) < 1.5 * dn_max:
                self._n_target = None
        else:
            # local gradient of the full achievable-MRR score (stability
            # AND forced-response terms), the realised form of Eq. (19)
            i0 = np.searchsorted(self.n_grid, self.n)
            i0 = np.clip(i0, 2, len(self.n_grid) - 3)
            dn_grid = self.n_grid[1] - self.n_grid[0]
            grad = ((q_ach[i0 + 2] - q_ach[i0 - 2]) / (4.0 * dn_grid))
            step = np.clip(alpha_n * cp.w4 * grad * 4.0e2, -dn_max, dn_max)
        self.n = float(np.clip(self.n + step, lim.n_min_rpm, lim.n_max_rpm))

        # -- 2. feed rate MPC, Eq. (20) -------------------------------------
        # Context-weighted reference (Eq. (16): weights shift with margin):
        # push above the nominal reference where the stability margin is
        # generous, back off as the margin or the chatter indicator tightens.
        margin_now = cp.eta * self._ap_lim_at(self.n) - self.ap
        q_ref = cp.Q_ref_cm3_min * float(np.clip(
            1.0 + 0.06 * (margin_now - 1.0), 0.85, 1.20))
        if self.p_inst > 0.4:
            q_ref *= max(0.4, 1.0 - 0.35 * min(self.p_inst, 2.0))
        ae_mm = self.setup.engagement.ae_m * 1000.0
        vf_star = q_ref * 1000.0 / (self.ap * ae_mm)     # mm/min
        vf_seq = self._solve_feed_mpc(vf_star)
        dv_max = lim.vf_dot_max_mm_min_s * dt_c
        vf_floor = max(lim.vf_min_mm_min,
                       self.setup.tool.n_teeth * self.n * cp.ft_min_mm)
        vf_hi = min(self.vf + dv_max, float(self._vf_max_at(self.n)))
        self.vf = float(np.clip(vf_seq,
                                max(self.vf - dv_max, vf_floor),
                                max(vf_hi, vf_floor)))

        # -- 3. depth of cut, Eq. (21) + Eq. (17), every 0.5 s ---------------
        self._ap_timer += dt_c
        if self._ap_timer >= 0.5:
            self._ap_timer = 0.0
            d_ap = self.ap - self._prev_ap
            d_rms = self._rms_filt - self._prev_rms
            if abs(d_ap) > 1e-3:
                slope = d_rms / d_ap
                if np.isfinite(slope):
                    self._slope_um_per_mm = float(np.clip(
                        0.8 * self._slope_um_per_mm + 0.2 * slope, 3.0, 120.0))
            self._prev_ap, self._prev_rms = self.ap, self._rms_filt
            err = self._rms_filt - cp.V_target_um
            gain = cp.gamma_a * (2.2 if err > 0.0 else 1.0)
            d_ap_cmd = -gain * err / self._slope_um_per_mm
            self.ap = self.ap + float(np.clip(d_ap_cmd, -0.35, 0.25))
        # feedback-driven trust in the model boundary: sustained quiet
        # cutting relaxes the cap; any chatter indication resets it
        # In-pass trust relaxation.  When a cross-part boundary learner is
        # attached it becomes the sole learning channel (clean attribution
        # for the Eq. (23) study) and the in-pass relaxation is disabled.
        if self.p_inst > 0.3 or self.learner is not None:
            self._relax = 0.0
        elif (chatter_um < 2.0 and self._rms_filt < 1.1 * cp.V_target_um
              and self._n_target is None):
            self._relax = min(0.35, self._relax + 0.0007)
        # stability constraint, Eq. (17), enforced every period, together
        # with the forced-response cap at the current speed and feed.
        # The trust relaxation may soften the safety factor eta but must
        # never push the commanded depth past the predicted boundary
        # itself (eta_eff < 1 always).
        eta_eff = min(cp.eta * (1.0 + self._relax), 0.95)
        ap_cap = eta_eff * self._ap_lim_at(self.n)
        ap_forced_now = float(self._ap_forced_mm(
            np.array([self.n]), vf_mm_min=self.vf)[0])
        ap_cap = min(ap_cap, 1.15 * ap_forced_now)
        if self._n_target is not None:
            # crossing lobes toward a new pocket: respect the worst
            # boundary along the remaining path, with a safety factor
            sel = ((self.n_grid >= min(self.n, self._n_target))
                   & (self.n_grid <= max(self.n, self._n_target)))
            if sel.any():
                ap_cap = min(ap_cap,
                             0.9 * cp.eta * float(self.ap_env_mm[sel].min()))
        ap_cap = max(ap_cap, lim.ap_min_mm)      # keep clip bounds ordered
        self.ap = float(np.clip(self.ap, lim.ap_min_mm,
                                min(lim.ap_max_mm, ap_cap)))

        if self.learner is not None:
            self.learner.observe(self.n, meas.Vr, self.ap, ap_lim,
                                 self._rms_filt)

        lg = self.log
        lg.t.append(meas.t)
        lg.fn_id.append(self.model_modes[0][0])   # fused Eq. (6) + RLS
        lg.fn_true.append(np.nan)                 # filled by the runner
        lg.ap_lim.append(ap_lim)
        lg.margin.append(margin)
        lg.p_inst.append(self.p_inst)

    def _solve_feed_mpc(self, vf_star: float) -> float:
        """Unconstrained receding-horizon solution of Eq. (20).

        With QMRR linear in vf and ap held over the horizon, the KKT system
        is tridiagonal in the moves; only the first move is applied.
        The tracking error is expressed in feed units, so lambda_f here
        equals the paper's lambda_f divided by (ap*ae)^2 - a fixed smoothing
        ratio rather than Eq. (20)'s literal MRR-unit weighting.
        """
        cp = self.cp
        Np = cp.Np
        # minimise sum (vf_i - vf*)^2 + lambda_f sum (dvf_i)^2,  vf_0 given
        # tridiagonal system in vf_1..vf_Np
        a = np.full(Np, 1.0 + 2.0 * cp.lambda_f)
        a[-1] = 1.0 + cp.lambda_f
        b = np.full(Np - 1, -cp.lambda_f)
        M = (np.diag(a) + np.diag(b, 1) + np.diag(b, -1))
        rhs = np.full(Np, vf_star)
        rhs[0] += cp.lambda_f * self.vf
        sol = np.linalg.solve(M, rhs)
        return float(sol[0])

    def command(self, t: float) -> tuple[float, float, float]:
        # small speed dither keeps the identification excited (Sec. 3.1)
        cp = self.cp
        n = self.n * (1.0 + cp.dither_frac
                      * math.sin(2.0 * math.pi * cp.dither_hz * t))
        return n, self.vf, self.ap
