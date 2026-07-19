"""Controllers: constant PD (CPD), time-space varying PD (VPD, the
reference paper's method re-derived by its published tuning procedure),
and the proposed position-scheduled harmonic LQG (PSH-LQG).

All controllers run at the hardware rate fs = 10 kHz, receive the filtered
accelerometer signal y [m/s^2] and the (known) cutter position, and return
the high-voltage command in volts; the simulator applies saturation and a
one-sample computation delay.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.linalg as la
import scipy.signal as sig

from .discrete import CompositeCt, CompositeDt
from .modal import ModalModel
from .params import SystemParams


# ====================================================================== PD

class PDController:
    """u_k = -(kp y_k + kd (y_k - y_{k-1})), gains possibly position-varying."""

    def __init__(self, kp, kd, x_ref: np.ndarray | None = None):
        """kp, kd: scalars, or arrays sampled at positions x_ref."""
        self.kp, self.kd, self.x_ref = kp, kd, x_ref
        self.y_prev = 0.0

    def gains_at(self, xc: float) -> tuple[float, float]:
        if self.x_ref is None:
            return float(self.kp), float(self.kd)
        return (float(np.interp(xc, self.x_ref, self.kp)),
                float(np.interp(xc, self.x_ref, self.kd)))

    def update(self, k: int, t: float, y: float, xc: float) -> float:
        kp, kd = self.gains_at(xc)
        u = -(kp * y + kd * (y - self.y_prev))
        self.y_prev = y
        return u


def quintic_fit(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Quintic polynomial fit of gains vs position (paper Eq. 26)."""
    return np.polyfit(x, g, 5)


# =========================================================== loop analysis

def pd_loop_frf(sys: SystemParams, model: ModalModel, kp: float, kd: float,
                x_force: float, w: np.ndarray,
                x_out: float | None = None) -> np.ndarray:
    """Closed-loop FRF (force at x_force -> deflection at x_out) with a
    discrete PD on the filtered acceleration, incl. ZOH-equivalent delay.
    """
    ctl, sen = sys.control, sys.sensor
    x_out = x_force if x_out is None else x_out
    z = np.exp(1j * w * ctl.Ts)
    ba_f, aa_f = sig.butter(sen.lp_order, 2.0 * np.pi * sen.lp_cutoff,
                            btype="low", analog=True)
    Hf = sig.freqs(ba_f, aa_f, worN=w)[1]
    Hzoh = np.where(w > 0, (1.0 - 1.0 / z) / (1j * w * ctl.Ts), 1.0)
    # PD + analog filter + ZOH reconstruction + computation delay
    C = (kp + kd * (1.0 - 1.0 / z)) * Hf * Hzoh / z
    G_fa = model.frf_force_to_acc(x_force, w)
    G_va = model.frf_voltage_to_acc(w)
    G_fw = model.frf_force_to_disp(x_force, x_out, w)
    G_vw = model.frf_voltage_to_disp(x_out, w)
    return G_fw - G_vw * C * G_fa / (1.0 + G_va * C)


def force_harmonics(sys: SystemParams, n_harm: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Complex Fourier coefficients of the rigid-workpiece milling force
    at multiples of the tooth-passing frequency (h = 0 .. n_harm)."""
    from .milling import MillingForce
    mil = sys.milling
    mf = MillingForce(mil)
    n = 2048
    t = np.arange(n) / n * mil.tau
    F = mf.normal_force_series(t)
    spec = np.fft.rfft(F) / n
    amps = np.concatenate([[spec[0]], 2.0 * spec[1:n_harm + 1]])
    w_h = 2.0 * np.pi * mil.f_tpf * np.arange(0, n_harm + 1)
    return w_h, amps


def _pd_stable(sys, model, kp, kd, x_positions, margin=0.997) -> bool:
    from .stability import closed_loop_matrix
    ctl = PDController(kp, kd)
    for x in x_positions:
        Acl = closed_loop_matrix(sys, model, ctl, x)
        if np.max(np.abs(np.linalg.eigvals(Acl))) > margin:
            return False
    return True


def _pd_harmonic_cost(sys, model, kp, kd, x_pos, w_h, F_h,
                      v_lim: float = 150.0, peak_guard: float = 1.2):
    """Predicted AC deflection RMS at the cutting point under the actual
    milling-force spectrum, with a soft actuator-voltage budget and a
    broadband waterbed guard."""
    ctl, sen = sys.control, sys.sensor
    w = w_h[1:]
    z = np.exp(1j * w * ctl.Ts)
    ba_f, aa_f = sig.butter(sen.lp_order, 2.0 * np.pi * sen.lp_cutoff,
                            btype="low", analog=True)
    Hf = sig.freqs(ba_f, aa_f, worN=w)[1]
    Hzoh = (1.0 - 1.0 / z) / (1j * w * ctl.Ts)
    C = (kp + kd * (1.0 - 1.0 / z)) * Hf * Hzoh / z
    G_fa = model.frf_force_to_acc(x_pos, w)
    G_va = model.frf_voltage_to_acc(w)
    G_fw = model.frf_force_to_disp(x_pos, x_pos, w)
    G_vw = model.frf_voltage_to_disp(x_pos, w)
    H_cl = G_fw - G_vw * C * G_fa / (1.0 + G_va * C)
    V_of_F = -C * G_fa / (1.0 + G_va * C)
    rms2 = 0.5 * np.sum(np.abs(H_cl * F_h[1:]) ** 2)
    v_rms = np.sqrt(0.5 * np.sum(np.abs(V_of_F * F_h[1:]) ** 2))
    cost = np.sqrt(rms2)
    if v_rms > v_lim:
        cost *= 1.0 + 10.0 * (v_rms / v_lim - 1.0)
    # waterbed guard on the broadband peak
    wb = 2.0 * np.pi * np.geomspace(50.0, 2600.0, 160)
    zb = np.exp(1j * wb * ctl.Ts)
    Hfb = sig.freqs(ba_f, aa_f, worN=wb)[1]
    Hzb = (1.0 - 1.0 / zb) / (1j * wb * ctl.Ts)
    Cb = (kp + kd * (1.0 - 1.0 / zb)) * Hfb * Hzb / zb
    H_ol = model.frf_force_to_disp(x_pos, x_pos, wb)
    H_cb = H_ol - model.frf_voltage_to_disp(x_pos, wb) * Cb \
        * model.frf_force_to_acc(x_pos, wb) \
        / (1.0 + model.frf_voltage_to_acc(wb) * Cb)
    ratio = np.max(np.abs(H_cb)) / np.max(np.abs(H_ol))
    if ratio > peak_guard:
        cost *= 1.0 + 10.0 * (ratio - peak_guard)
    return cost


def _pd_regen_ok(sys, model, kp, kd, x_positions, a_crit_open,
                 degrade: float = 0.9, target: float = 1.3) -> bool:
    """The closed loop must not degrade the regenerative (chatter) margin:
    at each position, a_crit(closed) >= min(degrade * a_crit(open),
    target * ap).  Where the open loop is already beyond its linear limit
    (amplitude-bounded chatter regime), the controller may not worsen it;
    where the open loop has margin, the margin must be preserved."""
    from .milling import MillingForce
    mf = MillingForce(sys.milling)
    for x, a_ol in zip(x_positions, a_crit_open):
        frf = lambda w: pd_loop_frf(sys, model, kp, kd, x, w)
        a_cl = mf.a_crit_nyquist(frf, sys.milling.rpm, nf=12000)
        if a_cl < min(degrade * a_ol, target * sys.milling.ap):
            return False
    return True


def _pd_grid_search(sys, model, x_cost_positions, x_stab_positions,
                    w_h, F_h, seed_cands=()) -> tuple[float, float, float]:
    """Best (kp, kd) over a signed log grid, minimizing the worst
    harmonic-excitation cost subject to frozen stability and to the
    closed-loop regenerative stability margin."""
    from .milling import MillingForce
    mf = MillingForce(sys.milling)
    a_crit_open = [mf.a_crit_nyquist(
        lambda w: model.frf_force_to_disp(x, x, w), sys.milling.rpm,
        nf=12000) for x in x_cost_positions]

    def cost(kp, kd):
        if (kp, kd) != (0.0, 0.0) and \
                not _pd_stable(sys, model, kp, kd, x_stab_positions):
            return np.inf
        c = max(_pd_harmonic_cost(sys, model, kp, kd, x, w_h, F_h)
                for x in x_cost_positions)
        if np.isfinite(c) and (kp, kd) != (0.0, 0.0) and \
                not _pd_regen_ok(sys, model, kp, kd, x_cost_positions,
                                 a_crit_open):
            return np.inf
        return c

    mags = np.concatenate([[0.0], np.logspace(-3.0, 0.0, 7)])
    cands = [(sp * mp, sd * md)
             for sp in (1.0, -1.0) for mp in mags
             for sd in (1.0, -1.0) for md in mags
             if not (mp == 0.0 and sp < 0) and not (md == 0.0 and sd < 0)]
    cands = list(seed_cands) + cands
    best = (0.0, 0.0)
    best_c = cost(0.0, 0.0)
    for kp, kd in cands:
        c = cost(kp, kd)
        if c < best_c:
            best_c, best = c, (kp, kd)
    for _ in range(2):
        kp0, kd0 = best
        dp = abs(kp0) if kp0 else 0.005
        dd = abs(kd0) if kd0 else 0.005
        for fp in np.linspace(-0.6, 0.6, 5):
            for fd in np.linspace(-0.6, 0.6, 5):
                c = cost(kp0 + fp * dp, kd0 + fd * dd)
                if c < best_c:
                    best_c, best = c, (kp0 + fp * dp, kd0 + fd * dd)
    return best[0], best[1], best_c


def _pd_sim_score(sys, model, kp, kd,
                  probes=(0.012, 0.06, 0.108), T: float = 0.9) -> float:
    """Nonlinear-simulation cost of a PD pair: geometric-mean RMS ratio
    (controlled / open) over probe positions, with saturation penalized.
    The frequency-domain surrogate costs are unreliable in the
    finite-amplitude regenerative regime, so candidate SELECTION (not
    just verification) is simulation-based."""
    import copy as _copy
    from .simulate import simulate as _sim
    from .metrics import summarize as _summ
    logsum = 0.0
    for x in probes:
        sp = _copy.deepcopy(sys)
        sp.milling.x_start, sp.milling.x_end = x, x + 1e-6
        sp.milling.pass_time = T
        r_ol = _sim(sp, model, controller=None, T=T, seed=1)
        r_cl = _sim(sp, model, controller=PDController(kp, kd), T=T, seed=1)
        s_ol = _summ(r_ol, T, settle=0.3)
        s_cl = _summ(r_cl, T, settle=0.3)
        ratio = s_cl["rms_ac_wc"] / max(s_ol["rms_ac_wc"], 1e-12)
        if s_cl["sat_frac"] > 0.01:
            ratio *= 10.0
        logsum += np.log(max(ratio, 1e-6))
    return float(np.exp(logsum / len(probes)))


def tune_cpd(sys: SystemParams, model: ModalModel,
             n_check: int = 5, n_sim: int = 12) -> tuple[float, float]:
    """Best CONSTANT PD: candidates on a signed log grid are pre-filtered
    by frozen stability and regenerative-margin non-degradation, ranked
    by the harmonic-excitation surrogate, and the top `n_sim` are scored
    by nonlinear time-domain simulation; the simulation winner is
    verified by _pd_sim_ok."""
    from .milling import MillingForce
    mil = sys.milling
    xs = list(np.linspace(mil.x_start, mil.x_end, n_check))
    w_h, F_h = force_harmonics(sys)
    mf = MillingForce(sys.milling)
    x3 = [mil.x_start + 0.1 * (mil.x_end - mil.x_start),
          0.5 * (mil.x_start + mil.x_end),
          mil.x_end - 0.1 * (mil.x_end - mil.x_start)]
    a_ol = [mf.a_crit_nyquist(
        lambda w: model.frf_force_to_disp(x, x, w), mil.rpm, nf=12000)
        for x in x3]

    mags_d = np.logspace(-2.3, -0.8, 6)
    mags_p = [0.0, 0.01, -0.01, 0.03, -0.03]
    cands = [(kp, sd * md) for kp in mags_p
             for sd in (1.0, -1.0) for md in mags_d]
    scored = []
    for kp, kd in cands:
        if not _pd_stable(sys, model, kp, kd, xs):
            continue
        if not _pd_regen_ok(sys, model, kp, kd, x3, a_ol):
            continue
        c = max(_pd_harmonic_cost(sys, model, kp, kd, x, w_h, F_h)
                for x in x3)
        scored.append((c, kp, kd))
    scored.sort()
    best, best_cost = (0.0, 0.0), 1.0
    for _, kp, kd in scored[:n_sim]:
        c = _pd_sim_score(sys, model, kp, kd)
        if c < best_cost:
            best_cost, best = c, (kp, kd)
    kp, kd = best
    for _ in range(10):
        if _pd_sim_ok(sys, model, PDController(kp, kd)):
            return kp, kd
        kp *= 0.85
        kd *= 0.85
    return 0.0, 0.0


def tune_vpd(sys: SystemParams, model: ModalModel, n_points: int = 31,
             cpd_seed: tuple[float, float] | None = None
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Position-varying PD, the reference paper's VPD strategy given its
    best shot: at each path position the (kp, kd) pair minimizes the local
    response to the actual milling-force harmonics subject to frozen
    stability (at that position and the path extremes); the gain profiles
    are then smoothed by the paper's quintic-polynomial parameterization
    (their Eq. 26).

    Returns (x_grid, kp, kd) with the quintic-smoothed profiles.
    """
    import copy as _copy
    from .simulate import simulate as _sim
    from .metrics import summarize as _summ
    mil = sys.milling
    xg = np.linspace(mil.x_start, mil.x_end, n_points)
    if cpd_seed is None:
        cpd_seed = tune_cpd(sys, model)
    kp_a, kd_a = cpd_seed
    if kp_a == 0.0 and kd_a == 0.0:
        return xg, np.zeros(n_points), np.zeros(n_points)

    # position-dependent intensity of the anchor gains, scored by short
    # nonlinear simulations at each grid point (the paper's per-position
    # tuning idea, made simulation-truthful)
    scales = np.array([0.5, 0.75, 1.0, 1.5, 2.2])
    x_ends = [mil.x_start, 0.5 * (mil.x_start + mil.x_end), mil.x_end]
    s_best = np.ones(n_points)
    for i, x in enumerate(xg):
        sp = _copy.deepcopy(sys)
        sp.milling.x_start, sp.milling.x_end = x, x + 1e-6
        sp.milling.pass_time = 0.8
        r_ol = _sim(sp, model, controller=None, T=0.8, seed=1)
        ref = _summ(r_ol, 0.8, settle=0.3)["rms_ac_wc"]
        best_c = np.inf
        for s in scales:
            if not _pd_stable(sys, model, s * kp_a, s * kd_a, [x] + x_ends):
                continue
            r_cl = _sim(sp, model,
                        controller=PDController(s * kp_a, s * kd_a),
                        T=0.8, seed=1)
            sc = _summ(r_cl, 0.8, settle=0.3)
            c = sc["rms_ac_wc"] / max(ref, 1e-12)
            if sc["sat_frac"] > 0.01:
                c *= 10.0
            if c < best_c:
                best_c, s_best[i] = c, s
    kp_arr = s_best * kp_a
    kd_arr = s_best * kd_a
    kp_fit = np.polyval(quintic_fit(xg, kp_arr), xg)
    kd_fit = np.polyval(quintic_fit(xg, kd_arr), xg)

    # The quintic smoothing can bridge over local dips of the pointwise
    # stability-constrained gains and yield a locally destabilizing
    # profile (a genuine hazard of the reference paper's Eq. 26
    # parameterization).  Verify the fitted profile along a dense grid and
    # shrink it uniformly until the frozen loop and the regenerative
    # margin verify everywhere.
    from .milling import MillingForce
    from .stability import closed_loop_matrix
    mf = MillingForce(sys.milling)
    x_dense = np.linspace(mil.x_start, mil.x_end, 25)
    a_ol = [mf.a_crit_nyquist(lambda w: model.frf_force_to_disp(x, x, w),
                              sys.milling.rpm, nf=12000) for x in x_dense]
    def profile_ok(kp_p, kd_p) -> bool:
        ctl = PDController(kp_p, kd_p, x_ref=xg)
        for x, a in zip(x_dense, a_ol):
            Acl = closed_loop_matrix(sys, model, ctl, x)
            if np.max(np.abs(np.linalg.eigvals(Acl))) > 0.997:
                return False
            kp_x, kd_x = ctl.gains_at(x)
            frf = lambda w: pd_loop_frf(sys, model, kp_x, kd_x, x, w)
            a_cl = mf.a_crit_nyquist(frf, sys.milling.rpm, nf=12000)
            if a_cl < min(0.9 * a, 1.3 * sys.milling.ap):
                return False
        # frequency-domain checks are fragile near stability pockets of
        # the interrupted-cutting regenerative loop; the final arbiter is
        # a short nonlinear time-domain simulation at probe positions
        return _pd_sim_ok(sys, model, PDController(kp_p, kd_p, x_ref=xg))

    for _ in range(20):
        if profile_ok(kp_fit, kd_fit):
            return xg, kp_fit, kd_fit
        kp_fit *= 0.85
        kd_fit *= 0.85
    raise RuntimeError("VPD profile failed post-fit verification")


def _pd_sim_ok(sys, model, controller, probes=(0.012, 0.06, 0.108),
               T: float = 1.2) -> bool:
    """Short closed-loop simulations at probe positions: the controller
    must not amplify the open-loop response nor saturate persistently."""
    import copy as _copy
    from .modal import build_modal_model
    from .simulate import simulate as _sim
    from .metrics import summarize as _summ
    for x in probes:
        sp = _copy.deepcopy(sys)
        sp.milling.x_start, sp.milling.x_end = x, x + 1e-6
        sp.milling.pass_time = T
        m2 = model  # mode shapes at fixed x via the original path grid
        r_ol = _sim(sp, m2, controller=None, T=T, seed=1)
        if hasattr(controller, "reset"):
            controller.reset()
        if hasattr(controller, "y_prev"):
            controller.y_prev = 0.0
        r_cl = _sim(sp, m2, controller=controller, T=T, seed=1)
        s_ol, s_cl = _summ(r_ol, T, settle=0.4), _summ(r_cl, T, settle=0.4)
        # tolerance: 20 % relative or 2 um absolute (bounded-chatter limit
        # cycles shift slightly under any control input, and micron-level
        # positions carry no meaningful signal)
        limit = max(1.2 * s_ol["rms_ac_wc"], s_ol["rms_ac_wc"] + 2e-6)
        if s_cl["rms_ac_wc"] > limit or s_cl["sat_frac"] > 0.01:
            return False
    return True


# ============================================================== PSH-LQG

@dataclass
class PshlqgConfig:
    n_modes: int = 6            # design-model order
    n_harmonics: int = 4        # TPF harmonics in the internal model
    include_dc: bool = True
    q_perf: float = 1.0         # weight on cutting-point deflection^2
    r_u: float = 1e-14          # weight on voltage^2 (sets authority usage)
    q_modal: float = 1e-8       # process noise, modal accelerations
    q_dist: float = 1e-3        # process noise, harmonic force states [N^2/step]
    q_dc: float = 1.0           # broadband (random-walk) force state: large
                                # enough to track the sub-harmonic content of
                                # amplitude-bounded chatter in the
                                # quasi-static band below mode 1, where the
                                # static inversion phase is reliable
    r_meas: float = 2.5e-3      # measurement noise variance [(m/s^2)^2]
    reg_ff: float = 0.05        # Tikhonov regularization of harmonic inversion
    rho_dist: float = 0.99995   # internal-model leakage: trades an O(0.01)
                                # steady cancellation bias for a guaranteed
                                # contraction budget against model mismatch
                                # (standard practice in repetitive/adaptive
                                # feedforward control)
    n_sched: int = 31           # scheduling grid points along the path


def tune_r_u(sys: SystemParams, model: ModalModel,
             cfg: "PshlqgConfig | None" = None,
             n_check: int = 7, margin_struct: float = 0.997,
             safety: float = 3.0) -> float:
    """Auto-tune the LQR voltage weight: find the smallest r_u (most
    aggressive control) whose frozen closed loop against the FULL
    evaluation model stays stable along the whole path (this bounds
    spillover), then back off by `safety`.

    Two-tier criterion: eigenvalues inside a +/-30 Hz band around the
    TPF harmonics (the internal-model pole cluster plus any plate pole
    that happens to fall there, e.g. mode 4 near the 4th harmonic) must
    stay within half the leakage budget of the unit circle; all
    eigenvalues outside the bands must meet the stricter `margin_struct`.
    A structural pole destabilized INSIDE a band cannot hide: the
    spillover mask can only stabilize poles that respond to backing off
    the harmonic gains, so a K-induced in-band instability fails the mask
    and forces this tuner to a weaker r_u (attribution by damping)."""
    from .stability import closed_loop_matrix
    cfg = cfg or PshlqgConfig()
    mil = sys.milling
    xs = np.linspace(mil.x_start, mil.x_end, n_check)
    Ts = sys.control.Ts
    w_h = 2.0 * np.pi * sys.milling.f_tpf * np.arange(0, cfg.n_harmonics + 1)
    margin_im = 1.0 - 0.5 * (1.0 - cfg.rho_dist)

    def feasible(log_r: float) -> bool:
        c = PshlqgConfig(**{**cfg.__dict__, "r_u": 10.0 ** log_r})
        ctl = PshlqgController(sys, model, c)
        if not ctl.spillover_mask(model):
            return False
        for xc in xs:
            ev = np.linalg.eigvals(closed_loop_matrix(sys, model, ctl, xc))
            f_ev = np.abs(np.angle(ev)) / Ts
            near_im = np.min(np.abs(f_ev[:, None] - w_h[None, :]),
                             axis=1) < 2.0 * np.pi * 30.0
            if np.any(np.abs(ev[near_im]) > margin_im):
                return False
            if np.any(np.abs(ev[~near_im]) > margin_struct):
                return False
        return True

    lo, hi = -16.0, -8.0
    if not feasible(hi):
        raise RuntimeError("PSH-LQG infeasible even at the weakest gain")
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    return float(10.0 ** hi * safety)


class DesignPlant:
    """Discrete design model at a frozen cutter position.

    State: [eta (nm), etad (nm), x_f (nf), z (ndc + 2H), u_prev];
    the modal + analog-filter block is discretized jointly (ZOH of the
    composite continuous system), and the harmonic phasor states couple
    into it through EXACT sinusoid discretization (see discrete.py).
    Input: commanded (saturated) voltage; output: sampled filtered
    accelerometer signal (no direct feedthrough).
    """

    def __init__(self, sys: SystemParams, model: ModalModel,
                 cfg: "PshlqgConfig", f_dist: float):
        self.sys, self.cfg = sys, cfg
        self.dm = model.truncate(cfg.n_modes)
        self.nm = cfg.n_modes
        self.H = cfg.n_harmonics
        self.ndc = 1 if cfg.include_dc else 0
        self.nz = self.ndc + 2 * self.H
        self.Ts = sys.control.Ts
        self.w_h = 2.0 * np.pi * f_dist * np.arange(1, self.H + 1)

        self.ct = CompositeCt(self.dm, sys)
        self.dt = CompositeDt(self.ct, self.Ts, self.w_h)
        self.nmf = self.ct.nx                     # modal + filter states
        self.zoff = self.nmf
        self.i_u = self.nmf + self.nz
        self.nx = self.nmf + self.nz + 1
        # LQR acts on the controllable subsystem [x_mf, u_prev]
        self.idx_lqr = np.r_[np.arange(self.nmf), self.i_u]

        # constant blocks
        self._A0 = np.zeros((self.nx, self.nx))
        self._A0[:self.nmf, :self.nmf] = self.dt.Ad
        self._A0[:self.nmf, self.i_u] = self.dt.Bd_V
        j = self.zoff
        rho = cfg.rho_dist
        if self.ndc:
            self._A0[j, j] = rho
            j += 1
        for h in range(self.H):
            th = self.w_h[h] * self.Ts
            self._A0[j:j + 2, j:j + 2] = rho * np.array(
                [[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
            j += 2
        self._B = np.zeros(self.nx)
        self._B[self.i_u] = 1.0
        self._C = np.zeros(self.nx)
        self._C[:self.nmf] = self.ct.C_y

    def matrices(self, xc: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A, B (command input), C (sampled output) at cutter position xc."""
        A = self._A0.copy()
        phi_c = self.dm.phi_at(xc)
        j = self.zoff
        if self.ndc:
            A[:self.nmf, j] = self.dt.Pd @ phi_c
            j += 1
        for h in range(self.H):
            c1, c2 = self.dt.force_cols(phi_c, h)
            A[:self.nmf, j] = c1
            A[:self.nmf, j + 1] = c2
            j += 2
        return A, self._B, self._C

    def perf_row(self, xc: float) -> np.ndarray:
        """y_p = deflection at the cutting point (design model)."""
        row = np.zeros(self.nx)
        row[:self.nm] = self.dm.phi_at(xc)
        return row


def _dare_gain_lqr(A, B, Q, R):
    P = la.solve_discrete_are(A, B.reshape(-1, 1), Q, np.atleast_2d(R))
    BPB = float(B @ P @ B) + float(R)
    K = (B @ P @ A) / BPB
    return K


def _dare_gain_kf(A, C, Qn, Rn):
    """Steady-state Kalman predictor gain.  With an exact internal model
    (unit-circle disturbance eigenvalues) scipy's DARE solver fails, so we
    fall back to iterating the Riccati difference equation, which converges
    geometrically as long as (A, C) is detectable and the marginally
    stable states carry process noise."""
    try:
        P = la.solve_discrete_are(A.T, C.reshape(-1, 1), Qn, np.atleast_2d(Rn))
    except Exception:
        P = Qn.copy()
        L_prev = None
        for it in range(200000):
            S = float(C @ P @ C) + float(Rn)
            L = (A @ P @ C) / S
            P = A @ P @ A.T - np.outer(L, C @ P @ A.T) + Qn
            P = 0.5 * (P + P.T)
            if it % 200 == 0:
                if L_prev is not None and np.max(np.abs(L - L_prev)) \
                        < 1e-10 * max(1.0, np.max(np.abs(L))):
                    break
                L_prev = L.copy()
    S = float(C @ P @ C) + float(Rn)
    return (A @ P @ C) / S


class PshlqgController:
    """Position-scheduled harmonic LQG (the proposed strategy).

    Structure: a Kalman predictor on the augmented design model (composite
    modal + analog-filter states, harmonic/DC milling-force phasors at the
    tooth-passing frequency, computation delay), a position-scheduled LQR
    state feedback weighting the deflection at the *moving cutting point*,
    and a position-scheduled regularized-inverse feedforward that cancels
    the estimated periodic milling force at the performance point
    (internal-model / disturbance-accommodation action).
    """

    def __init__(self, sys: SystemParams, model: ModalModel,
                 cfg: PshlqgConfig | None = None, f_dist: float | None = None):
        self.sys = sys
        self.cfg = cfg or PshlqgConfig()
        f_dist = f_dist or sys.milling.f_tpf
        self.plant = DesignPlant(sys, model, self.cfg, f_dist)
        self._design_gains()
        self.x_hat = np.zeros(self.plant.nx)
        self.sat_lo, self.sat_hi = sys.piezo.V_min, sys.piezo.V_max

    # ------------------------------------------------------------ design

    def _design_gains(self):
        p, cfg, sys = self.plant, self.cfg, self.sys
        mil = sys.milling
        self.x_sched = np.linspace(mil.x_start, mil.x_end, cfg.n_sched)
        nm, nmf = p.nm, p.nmf

        Qn = np.zeros((p.nx, p.nx))
        Qn[nm:2 * nm, nm:2 * nm] = cfg.q_modal * np.eye(nm)
        j = p.zoff
        if p.ndc:
            Qn[j, j] = cfg.q_dc
            j += 1
        for h in range(p.H):
            Qn[j, j] = Qn[j + 1, j + 1] = cfg.q_dist / (h + 1)
            j += 2
        Qn += 1e-12 * np.eye(p.nx)
        Rn = cfg.r_meas

        nK = nmf + 1
        self.K_grid = np.zeros((cfg.n_sched, nK))
        self.L_grid = np.zeros((cfg.n_sched, p.nx))
        self.Lam_grid = np.zeros((cfg.n_sched, p.H), dtype=complex)
        self.Lam0_grid = np.zeros(cfg.n_sched)

        dm = p.dm
        Gu_all = np.zeros((cfg.n_sched, p.H + 1), dtype=complex)
        Gd_all = np.zeros((cfg.n_sched, p.H + 1), dtype=complex)
        for g, xg in enumerate(self.x_sched):
            A, B, C = p.matrices(xg)
            self.L_grid[g] = _dare_gain_kf(A, C, Qn, Rn)

            phi_c = dm.phi_at(xg)
            Ac = np.zeros((nK, nK))
            Ac[:nmf, :nmf] = p.dt.Ad
            Ac[:nmf, -1] = p.dt.Bd_V
            Bc = np.zeros(nK)
            Bc[-1] = 1.0
            cp = np.zeros(nK)
            cp[:nm] = phi_c
            Qc = cfg.q_perf * np.outer(cp, cp) + 1e-14 * np.eye(nK)
            K = _dare_gain_lqr(Ac, Bc, Qc, cfg.r_u)
            self.K_grid[g] = K

            # harmonic-inversion transfers evaluated on the LQR-closed
            # loop (the state feedback also responds to the disturbance-
            # driven states, so the open-loop ratio Gd/Gu would leave a
            # systematic residual)
            Acl = Ac - np.outer(Bc, K)
            for h in range(p.H):
                zh = np.exp(1j * p.w_h[h] * p.Ts)
                R = np.linalg.solve(zh * np.eye(nK) - Acl,
                                    np.column_stack([Bc, np.zeros(nK)]))
                Gu_all[g, h] = cp @ R[:, 0]
                Bd = np.zeros(nK, dtype=complex)
                Bd[:nmf] = p.dt.W[h] @ phi_c
                Gd_all[g, h] = cp @ np.linalg.solve(
                    zh * np.eye(nK) - Acl, Bd)
            Bd0 = np.zeros(nK)
            Bd0[:nmf] = p.dt.Pd @ phi_c
            Gu_all[g, p.H] = cp @ np.linalg.solve(np.eye(nK) - Acl, Bc)
            Gd_all[g, p.H] = cp @ np.linalg.solve(np.eye(nK) - Acl, Bd0)

        # Tikhonov regularization with an ABSOLUTE floor per harmonic:
        # eps_h is a fraction of the path-maximum authority, so the
        # inversion gain is genuinely bounded where the local authority
        # collapses (weak-authority positions between modes)
        for h in range(p.H + 1):
            eps = cfg.reg_ff * np.max(np.abs(Gu_all[:, h])) + 1e-30
            lam = (Gd_all[:, h] * np.conj(Gu_all[:, h])
                   / (np.abs(Gu_all[:, h]) ** 2 + eps ** 2))
            if h < p.H:
                self.Lam_grid[:, h] = lam
            else:
                self.Lam0_grid[:] = np.real(lam)

    # ------------------------------------------------ spillover masking

    def spillover_mask(self, eval_model: ModalModel, n_check: int = 13,
                       max_iter: int = 15, backoff: float = 0.6) -> bool:
        """Verification-driven per-harmonic gain masking.

        The internal-model inversion is computed on the reduced design
        model; where residual-mode truncation flips its phase (weak-
        authority positions between modes), the corresponding closed-loop
        internal-model pole can drift outside the unit circle.  This
        standard spillover check closes the design loop against the
        full-order FEM model: any internal-model pole found outside its
        leakage budget at a frozen position triggers a local backoff of
        that harmonic's feedforward gain until the whole path verifies.

        Returns True when the frozen loop verifies along the path.
        """
        from .stability import closed_loop_matrix
        mil = self.sys.milling
        xs = np.linspace(mil.x_start, mil.x_end, n_check)
        Ts = self.sys.control.Ts
        w_h = np.concatenate([[0.0], self.plant.w_h])
        margin_im = 1.0 - 0.5 * (1.0 - self.cfg.rho_dist)
        band = 2.0 * np.pi * 30.0
        for _ in range(max_iter):
            viol = []
            for xc in xs:
                ev = np.linalg.eigvals(
                    closed_loop_matrix(self.sys, eval_model, self, xc))
                bad = np.abs(ev) > margin_im
                for e in ev[bad]:
                    f_e = abs(np.angle(e)) / Ts
                    h = int(np.argmin(np.abs(w_h - f_e)))
                    if abs(w_h[h] - f_e) < band:
                        viol.append((xc, h))
            if not viol:
                return True
            for xc, h in viol:
                sel = np.abs(self.x_sched - xc) <= 1.5 * (
                    self.x_sched[1] - self.x_sched[0])
                if h == 0:
                    self.Lam0_grid[sel] *= backoff
                else:
                    self.Lam_grid[sel, h - 1] *= backoff
        return False

    # ------------------------------------------------------------ runtime

    def _interp(self, arr: np.ndarray, xc: float) -> np.ndarray:
        x = self.x_sched
        i = int(np.clip(np.searchsorted(x, xc) - 1, 0, len(x) - 2))
        s = float(np.clip((xc - x[i]) / (x[i + 1] - x[i]), 0.0, 1.0))
        return (1.0 - s) * arr[i] + s * arr[i + 1]

    def reset(self):
        self.x_hat = np.zeros(self.plant.nx)

    def update(self, k: int, t: float, y: float, xc: float) -> float:
        p = self.plant
        A, B, C = p.matrices(xc)
        L = self._interp(self.L_grid, xc)
        K = self._interp(self.K_grid, xc)
        Lam = self._interp(self.Lam_grid, xc)
        Lam0 = float(self._interp(self.Lam0_grid, xc))

        xh = self.x_hat
        u_fb = -float(K @ xh[p.idx_lqr])
        # cancelling phasor: d_h(k) = Re{conj(c_h(k))}, c_h = z1 + j z2;
        # b = -Lam_h conj(c_h)  =>  u_h = -(Re(Lam) z1 + Im(Lam) z2)
        j = p.zoff
        u_ff = 0.0
        if p.ndc:
            u_ff -= Lam0 * xh[j]
            j += 1
        for h in range(p.H):
            u_ff -= (np.real(Lam[h]) * xh[j] + np.imag(Lam[h]) * xh[j + 1])
            j += 2
        u = u_fb + u_ff
        u_sat = min(max(u, self.sat_lo), self.sat_hi)

        # Kalman predictor driven by the *saturated* input (anti-windup)
        yh = float(C @ xh)
        self.x_hat = A @ xh + B * u_sat + L * (y - yh)
        return u
