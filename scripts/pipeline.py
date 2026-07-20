"""Full computational campaign for the manuscript.

Builds every controller, runs every certificate, and produces the cached
artifacts consumed by the fig_*.py scripts:

  A. models + weights + scheduled/frozen/DPD/DR controllers + certificates
  B. closed-loop stability lobes (SLD) per strategy and position
  C. time-domain passes at the reference condition
  D. robustness Monte-Carlo
  E. material-removal (multi-pass) scheduling study

Run:  python3 scripts/pipeline.py [stage ...]   (default: all stages)
"""
from __future__ import annotations

import dataclasses
import sys
import time

import numpy as np

from common import RESDIR, cached  # noqa: F401  (path setup side effect)

from avc.params import DEFAULT
from avc.modal import reduce_model
from avc.synthesis import (additive_uncertainty_weight, best_frozen_design,
                           delayed_pd_controller, deploy, gain_schedule,
                           rs_margin, sampled_rho)
from avc.delayed_feedback import tune_kr
from avc.simulate import simulate
from avc import sld

P = DEFAULT
N_D = P.design.n_modes_design
N_F = P.design.n_modes_full
RPM_REF = P.process.rpm_ref
X_EVAL = (0.005, 0.025, 0.05, 0.075, 0.095)          # evaluation positions
X_GRID = tuple(np.linspace(0.005, 0.095, P.design.n_pos_grid))
REMOVALS = (0.0, 0.5e-3, 1.0e-3)
RPMS = np.linspace(2000.0, 10000.0, 33)
AP_DESIGN = 0.3e-3
AP_HI = 5.0e-3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- stage A
def models():
    def build():
        out = {}
        for r in REMOVALS:
            out[r] = {"design": reduce_model(P, N_D, height_removed=r),
                      "full": reduce_model(P, N_F, height_removed=r)}
        return out
    return cached("pipe_models", build)


def stage_a():
    mdl = models()
    theta_all = [(x, r) for x in X_GRID for r in REMOVALS]

    def build():
        log("A: Wa fit")
        Wa = additive_uncertainty_weight(P, N_D, N_F, theta_all)
        log("A: gain schedule (9 x 3 grid)")
        S = gain_schedule(P, X_GRID, REMOVALS, AP_DESIGN, n_modes=N_D)
        midpt = S.validate_midpoints()
        log(f"A: schedule gamma={S.gammas[0, 0]:.2f}, midpoints {midpt:.1f}")
        log("A: best frozen design")
        Kf_raw = best_frozen_design(P, X_GRID, REMOVALS, AP_DESIGN,
                                    n_modes=N_D)
        log(f"A: frozen at {Kf_raw.meta['design_point']}, "
            f"wc_gamma={Kf_raw.meta['worst_case_gamma']:.1f}")

        log("A: DPD tuning (coarse grid, worst-position criterion)")
        mm12_0 = mdl[0.0]["full"]
        best_dpd, best_score = None, -1.0
        for kp in (-3e6, -1e6, -3e5, 0.0, 3e5, 1e6, 3e6):
            for kd in (0.0, 20.0, 60.0):
                if kp == 0.0 and kd == 0.0:
                    continue
                Kd = delayed_pd_controller(kp, kd)
                score = min(sld.critical_depth(mm12_0, x, RPM_REF,
                                               controller=Kd, ap_hi=AP_HI)
                            for x in (0.005, 0.05, 0.095))
                if score > best_score:
                    best_dpd, best_score = (kp, kd), score
        log(f"A: DPD kp={best_dpd[0]:.3g} kd={best_dpd[1]:.3g} "
            f"min a_lim={best_score * 1e3:.3f} mm")

        log("A: DR gain tuning k_r(theta) on closed-loop lobes")
        kr_map = {}
        for x in X_GRID:
            base = S.at_theta(float(x), 0.0, latency=True)
            kr, alim = tune_kr(mdl[0.0]["design"], float(x), RPM_REF, base,
                               k_max=3e6, n_coarse=9, n_refine=4,
                               ap_hi=AP_HI)
            kr_map[float(x)] = kr
            log(f"A: k_r({x * 1e3:.0f} mm) = {kr:.3g} V/m "
                f"(a_lim {alim * 1e3:.2f} mm)")

        log("A: certificates")
        certs = {}
        for x in X_EVAL:
            Kd = S.at_theta(x, 0.0)
            certs[("sched", x)] = {
                "rs": rs_margin(P, Kd, Wa, theta_all),
                "s_rho": sampled_rho(P, Kd, mm12_0)}
        Kf_dep = deploy(Kf_raw, P)
        certs[("frozen", None)] = {"rs": rs_margin(P, Kf_dep, Wa, theta_all),
                                   "s_rho": sampled_rho(P, Kf_dep, mm12_0)}
        return {"Wa": Wa, "sched": S, "frozen_raw": Kf_raw,
                "dpd": best_dpd, "dpd_alim": best_score,
                "kr_map": kr_map, "certs": certs,
                "midpoint_abscissa": midpt}
    return cached("pipe_stageA", build)


def controller_for(strategy, art, x, removal=0.0, latency=True):
    """Deployed controller (analysis flavour) for a strategy at theta."""
    if strategy == "OL":
        return None
    if strategy == "DPD":
        kp, kd = art["dpd"]
        return delayed_pd_controller(kp, kd)
    if strategy == "FROZEN":
        return deploy(art["frozen_raw"], P, latency=latency)
    if strategy == "PS-LPV":
        return art["sched"].at_theta(float(x), float(removal),
                                     latency=latency)
    if strategy == "PS-LPV-DR":
        K = art["sched"].at_theta(float(x), float(removal), latency=latency)
        xs = np.array(sorted(art["kr_map"]))
        kr = float(np.interp(float(x), xs,
                             [art["kr_map"][v] for v in xs]))
        return K.with_delayed_feedback(kr)
    raise ValueError(strategy)


STRATEGIES = ("OL", "DPD", "FROZEN", "PS-LPV", "PS-LPV-DR")


# ---------------------------------------------------------------- stage B
def stage_b():
    art = stage_a()
    mm12 = models()[0.0]["full"]

    def build():
        out = {}
        for strat in STRATEGIES:
            for x in X_EVAL:
                K = controller_for(strat, art, x)
                curve = sld.sld_curve(mm12, float(x), RPMS, controller=K,
                                      ap_hi=AP_HI)
                out[(strat, x)] = np.asarray(curve)
                log(f"B: SLD {strat} @x={x * 1e3:.0f} mm "
                    f"min={np.min(curve) * 1e3:.3f} mm")
        return {"rpms": RPMS, "curves": out}
    return cached("pipe_stageB", build)


# ---------------------------------------------------------------- stage C
def stage_c():
    art = stage_a()
    mm12 = models()[0.0]["full"]

    def build():
        out = {}
        # (i) fixed-position runs at the reference condition (comparable
        # to ref. Fig. 14): x = 5 mm (worst scheduled position)
        for strat in STRATEGIES:
            K = controller_for(strat, art, 0.005, latency=False)
            r = simulate(mm12, RPM_REF, P.process.ap_ref, controller=K,
                         T=0.5, x0=0.005, moving=False, fs=250e3, seed=3)
            out[("fixed", strat)] = {
                "t": r.t[:: 5], "w": r.w_mill[:: 5], "u": r.u[:: 5],
                "rms_w": r.rms_w(0.1), "peak_w": r.peak_w(0.1),
                "rms_u": r.rms_u(0.1), "peak_u": r.peak_u(0.1),
                "psd": r.psd_w(), "extras": r.extras}
            log(f"C: fixed {strat}: rms={r.rms_w(0.1) * 1e6:.2f} um "
                f"u={r.rms_u(0.1):.1f} V")
        # (ii) full moving pass, aggressive depth ap = 1.0 mm (far above
        # the open-loop limit): scheduled controller updating along the
        # path vs frozen
        for strat in ("FROZEN", "PS-LPV", "PS-LPV-DR"):
            sched = art["sched"] if strat.startswith("PS-LPV") else None
            K = controller_for(strat, art, 0.005, latency=False)
            r = simulate(mm12, RPM_REF, 1.0e-3, controller=K,
                         x0=0.005, x1=0.095, moving=True, fs=250e3,
                         seed=3, scheduled=sched)
            out[("pass", strat)] = {
                "t": r.t[:: 25], "w": r.w_mill[:: 25], "u": r.u[:: 25],
                "x": r.x_tool[:: 25],
                "rms_w": r.rms_w(0.5), "peak_w": r.peak_w(0.5),
                "rms_u": r.rms_u(0.5), "peak_u": r.peak_u(0.5),
                "extras": r.extras}
            log(f"C: pass {strat}: rms={r.rms_w(0.5) * 1e6:.2f} um "
                f"u={r.rms_u(0.5):.1f} V "
                f"div={r.extras['diverged_at']}")
        return out
    return cached("pipe_stageC", build)


# ---------------------------------------------------------------- stage D
def stage_d(n_mc=40):
    art = stage_a()
    mm12 = models()[0.0]["full"]

    def build():
        rng = np.random.default_rng(7)
        rows = {s: [] for s in ("FROZEN", "PS-LPV", "PS-LPV-DR")}
        x_worst = 0.095
        for i in range(n_mc):
            a4_scale = float(np.exp(rng.uniform(np.log(0.3), np.log(2.9))))
            dm = 1.0 + rng.uniform(-0.10, 0.10)      # mass-like freq shift
            dk = 1.0 + rng.uniform(-0.10, 0.10)      # stiffness-like
            dz = rng.uniform(0.8, 1.0)               # damping reduction
            # mass perturbation also rescales the mass-normalized mode
            # shapes (Phi ~ m^-1/2), hence bu, cs, and bf(x) coherently
            sm = 1.0 / np.sqrt(dm)
            mm_p = dataclasses.replace(
                mm12, omega=mm12.omega * np.sqrt(dk / dm),
                zeta=mm12.zeta * dz, Phi=mm12.Phi * sm,
                bu=mm12.bu * sm, cs=mm12.cs * sm)
            for strat in rows:
                K = controller_for(strat, art, x_worst)
                al = sld.critical_depth(mm_p, x_worst, RPM_REF,
                                        controller=K, ap_hi=AP_HI,
                                        a4_scale=a4_scale)
                rows[strat].append(al)
            if (i + 1) % 10 == 0:
                log(f"D: {i + 1}/{n_mc}")
        return {k: np.array(v) for k, v in rows.items()}
    return cached("pipe_stageD", build)


# ---------------------------------------------------------------- stage E
def stage_e():
    art = stage_a()
    mdl = models()

    def build():
        out = {}
        for r in REMOVALS:
            mmf = mdl[r]["full"]
            for x in X_EVAL:
                right = art["sched"].at_theta(float(x), float(r),
                                              latency=True)
                wrong = art["sched"].at_theta(float(x), 0.0, latency=True)
                froz = deploy(art["frozen_raw"], P, latency=True)
                out[("sched", x, r)] = sld.critical_depth(
                    mmf, float(x), RPM_REF, controller=right, ap_hi=AP_HI)
                out[("sched-fresh", x, r)] = sld.critical_depth(
                    mmf, float(x), RPM_REF, controller=wrong, ap_hi=AP_HI)
                out[("frozen", x, r)] = sld.critical_depth(
                    mmf, float(x), RPM_REF, controller=froz, ap_hi=AP_HI)
            log(f"E: removal {r * 1e3:.1f} mm done")
        return out
    return cached("pipe_stageE", build)


# ---------------------------------------------------------------- stage C2
def stage_c2():
    """Differentiating fixed-position runs at the frozen design's weakest
    position (end of the pass), above its local stability limit."""
    art = stage_a()
    mm12 = models()[0.0]["full"]

    def build():
        out = {}
        x_hard = 0.095
        for ap in (1.0e-3, 2.0e-3):
            for strat in ("OL", "DPD", "FROZEN", "PS-LPV", "PS-LPV-DR"):
                K = controller_for(strat, art, x_hard, latency=False)
                r = simulate(mm12, RPM_REF, ap, controller=K, T=0.5,
                             x0=x_hard, moving=False, fs=250e3, seed=3)
                out[(ap, strat)] = {
                    "t": r.t[:: 5], "w": r.w_mill[:: 5], "u": r.u[:: 5],
                    "rms_w": r.rms_w(0.1), "peak_w": r.peak_w(0.1),
                    "rms_u": r.rms_u(0.1), "peak_u": r.peak_u(0.1),
                    "psd": r.psd_w(), "extras": r.extras}
                log(f"C2: x=95mm ap={ap * 1e3:.1f} {strat}: "
                    f"rms={r.rms_w(0.1) * 1e6:.2f} um "
                    f"u={r.rms_u(0.1):.1f} V div={r.extras['diverged_at']}")
        return out
    return cached("pipe_stageC2", build)


STAGES = {"a": stage_a, "b": stage_b, "c": stage_c, "c2": stage_c2,
          "d": stage_d, "e": stage_e}

if __name__ == "__main__":
    which = [s.lower() for s in sys.argv[1:]] or list(STAGES)
    t0 = time.time()
    for s in which:
        log(f"=== stage {s.upper()} ===")
        STAGES[s]()
    log(f"pipeline done in {(time.time() - t0) / 60:.1f} min")
