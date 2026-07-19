"""
fopid_study.py
==============
Fractional-order PID (FOPID / PI^lambda D^mu) controller: honest design and
head-to-head comparison with the open loop, LQG and ADRC on the SAME plant,
sensors, saturation and metrics used throughout the study.

Why a fixed-structure fractional controller is worth testing here
----------------------------------------------------------------
LQG needs the full state model (and pays ~21 % margin for its observer); ADRC
needs only b0 but carries a third-order ESO.  A FOPID needs NEITHER a model nor
an observer -- it is a two-parameter-richer classical PID acting directly on the
measured signal.  If a well-tuned FOPID could match the observer-based designs it
would be the cheapest option to deploy.  This study asks, honestly, how close it
gets, and what the fractional orders buy.

Design (chatter-margin optimal, and the damping contrast)
---------------------------------------------------------
The FOPID is designed against the true chatter metric: for each loop sign the
gains and orders are chosen by differential evolution to minimize the dominant
Floquet multiplier of the CL-SD at a reference depth, then selected by the
voltage-feasible metric via a gain-scale line search (the analogue of ADRC's
bandwidth grid / LQG's weight search).  As a computed control (not an
assertion) we ALSO design a damping-optimal FOPID (maximizing the minimum
closed-loop modal damping, no cutting) and report its chatter boundary: it is
far below the chatter-optimal design, showing that modal damping is the wrong
objective for regenerative chatter (Sec. 4.11, out["damping_vs_chatter"]).

Realization / metric consistency
--------------------------------
The fractional operators use the Oustaloup approximation over [5, 8000] Hz
(order N = 3).  The upper corner is kept well below the Nyquist frequency
(1/2dt = 20 kHz at dt = 25 us; wh*dt/pi = 0.4) so that the ZOH-discretized
controller used in the time-domain feasibility metric and the continuous
realization embedded in the CL-SD monodromy agree closely -- the two metrics
describe the same controller.  design == deployment (N, band, orders, gains all
fixed across CL-SD and time domain).

Fair-comparison rules (identical to the rest of the study)
----------------------------------------------------------
* Plant: refined 5-mode Kirchhoff model, dt = 25 us; +/-150 V saturation on ALL.
* Sensor: collocated piezo-corner (minimum phase) for the output-feedback
  controllers FOPID and ADRC; tip for LQG.  To separate the model-free effect
  from the sensor effect, an LQG-on-collocated feasible number is also reported.
* Linear a_p,crit is given on BOTH the nominal 3-mode design model (spillover-
  free) and the refined 5-mode deployment plant, for every deployed design, so
  the "linear boundary is deceptive" point is applied symmetrically.
* Robustness uses the voltage-feasible depth under +/-20 % frequency drift.

Outputs: results/fopid.json (+ results/fopid_traces.npz)
Run:     python experiments/fopid_study.py [--quick]
"""
import os, sys, json, time, argparse
import numpy as np
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

from milling_force import precompute_alpha_periodic
from lqg_controller import LQGController
from adrc_control import ADRCController
from fopid_control import FOPIDController
from newmark_solver import NewmarkSimulator
from cl_fdm import ClosedLoopFDM, default_cutting
from model_refinement import build, _SelfStateWrapper, ZETA5

LP, HP = 0.100, 0.080
NT = 3; RPM = 4900; FT = 0.02e-3
DT = 2.5e-5
U_MAX = 150.0
CUT = default_cutting(hp=HP)
OMEGA = 2 * np.pi * RPM / 60
NX5 = 2 * 5

# Oustaloup realization used for EVERY FOPID here (design == deployment).
# wh kept below Nyquist (1/2dt = 20 kHz): wh*dt/pi = 0.4.
FOPID_N = 3
FOPID_WB = 2 * np.pi * 5.0
FOPID_WH = 2 * np.pi * 8000.0
FOPID_KW = dict(N=FOPID_N, wb=FOPID_WB, wh=FOPID_WH)
GAIN_BOUNDS = [(2.0, 10.5), (2.0, 10.5), (2.0, 10.5)]   # log10 Kp, Ki, Kd


def _fopid(plate, P):
    return FOPIDController(plate, dt=DT, Kp=P["Kp"], Ki=P["Ki"], Kd=P["Kd"],
                          lam=P["lam"], mu=P["mu"], sign=P["sign"],
                          u_max=U_MAX, **FOPID_KW)


def _rho(clf, plate, P, ap):
    try:
        r = clf.rho_dynamic(RPM, ap, *_fopid(plate, P).export_lti())
        return r if np.isfinite(r) else 9.0
    except Exception:
        return 9.0


# ---- closed-loop (no cutting) modal damping surrogate, for the contrast ----
def _plant_ss(plate):
    n = plate.n_modes
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -np.diag(plate.omega_n ** 2)
    A[n:, n:] = -np.diag(2 * np.array(ZETA5[:n]) * plate.omega_n)
    B = np.zeros((2 * n, 1)); B[n:, 0] = plate.H_Pe_modal
    Cpl = np.zeros((1, 2 * n)); Cpl[0, :n] = plate.D_obs
    return A, B, Cpl


def _min_damping(plate, Ap, B, Cpl, P):
    try:
        Ac, Bc, Cc, Dc = _fopid(plate, P).export_lti()
    except Exception:
        return -1.0
    nx, nz = Ap.shape[0], Ac.shape[0]
    Acl = np.zeros((nx + nz, nx + nz))
    Acl[:nx, :nx] = Ap + B @ Dc @ Cpl
    Acl[:nx, nx:] = B @ Cc
    Acl[nx:, :nx] = Bc @ Cpl
    Acl[nx:, nx:] = Ac
    ev = np.linalg.eigvals(Acl)
    if not np.all(np.isfinite(ev)) or np.max(ev.real) >= 0:
        return -1.0
    osc = ev[np.abs(ev.imag) > 1.0]
    if osc.size == 0:
        return -1.0
    return float(np.min(-osc.real / np.abs(osc)))


# ---- FOPID design: minimize the CL-SD spectral radius (chatter margin) ----
def design_fopid(design_plate, quick=False):
    clf = ClosedLoopFDM.from_plate(design_plate, CUT, m_div=20)
    ap_ref = 1.0e-3
    maxiter = 12 if quick else 16
    popsize = 8 if quick else 10
    bounds = GAIN_BOUNDS + [(0.2, 0.95), (0.2, 0.95)]
    winners = []
    t0 = time.time()
    for sign in (+1.0, -1.0):
        def obj(x):
            P = dict(sign=sign, lam=x[3], mu=x[4],
                     Kp=10.0 ** x[0], Ki=10.0 ** x[1], Kd=10.0 ** x[2])
            r0 = _rho(clf, design_plate, P, 0.02e-3)
            if r0 >= 1.0:
                return 5.0 + min(r0, 9.0)
            return _rho(clf, design_plate, P, ap_ref)
        res = differential_evolution(obj, bounds, seed=0, maxiter=maxiter,
                                     popsize=popsize, tol=1e-2, init='sobol',
                                     polish=True)
        P = dict(sign=sign, lam=float(res.x[3]), mu=float(res.x[4]),
                 Kp=10.0 ** res.x[0], Ki=10.0 ** res.x[1], Kd=10.0 ** res.x[2],
                 log_gains=[float(res.x[0]), float(res.x[1]), float(res.x[2])])
        ap = clf.ap_crit_dynamic(RPM, *_fopid(design_plate, P).export_lti(),
                                 ap_max=6e-3, n_coarse=25) * 1e3
        P["ap_lin_design_mm"] = ap
        winners.append(P)
        print(f"  [design] sign={sign:+.0f}: obj={res.fun:.3f} "
              f"lam={P['lam']:.2f} mu={P['mu']:.2f} ap_lin(design)={ap:.3f}mm "
              f"logKi={res.x[1]:.2f} logKd={res.x[2]:.2f} logKp={res.x[0]:.2f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    winners.sort(key=lambda P: -P["ap_lin_design_mm"])
    return winners


def design_damping_optimal(design_plate, sign, quick=False):
    """Damping-optimal FOPID (max min closed-loop modal damping, no cutting)."""
    Ap, B, Cpl = _plant_ss(design_plate)
    maxiter = 15 if quick else 25
    bounds = GAIN_BOUNDS + [(0.2, 0.95), (0.2, 0.95)]

    def neg(x):
        P = dict(sign=sign, lam=x[3], mu=x[4],
                 Kp=10.0 ** x[0], Ki=10.0 ** x[1], Kd=10.0 ** x[2])
        return -_min_damping(design_plate, Ap, B, Cpl, P)
    res = differential_evolution(neg, bounds, seed=0, maxiter=maxiter,
                                 popsize=10, tol=1e-3, init='sobol', polish=True)
    if -res.fun <= 0:
        return None
    return dict(sign=sign, lam=float(res.x[3]), mu=float(res.x[4]),
                Kp=10.0 ** res.x[0], Ki=10.0 ** res.x[1], Kd=10.0 ** res.x[2],
                damping=-res.fun)


# ---- metrics (identical routines to the rest of the study) ----
def _stable_at(make_ctrl, plant, ap, T=0.3, dt=DT, ret=False):
    sim = NewmarkSimulator(plant, dt=dt, T_end=T, ft=FT, tau=60 / (NT * RPM),
                           verbose=False)
    n_per = int(round((60 / (NT * RPM)) / dt))
    a3, a4 = precompute_alpha_periodic(dt, n_per, sim.nstep, OMEGA, NT, CUT['RT'],
                                       CUT['eta_h'], CUT['phi_st'], CUT['phi_ex'],
                                       HP - ap, HP, CUT['k1'], CUT['k2'], 925e6)
    vf = FT * NT * RPM / 60
    xp = np.minimum(vf * sim.t_vec, LP)
    kp = np.clip(np.round(xp / LP * 2000).astype(int), 0, 2000)
    r = sim.simulate(a3, a4, kp, controller=make_ctrl(), progress=False)
    ie = r['stop_idx']
    if ie < sim.nstep - 1:
        return (False, None) if ret else False
    ytip = plant.D_tip @ r['qm'][:, :ie + 1]
    yr = float(np.sqrt(np.mean(ytip[len(ytip) // 2:] ** 2)) * 1e6)
    ok = yr < 50.0
    return (ok, r) if ret else ok


def _feasible(make_ctrl, plant_fn, lo=0.2e-3, hi=6e-3, tol=0.1e-3, dt=DT):
    if not _stable_at(make_ctrl, plant_fn(), lo, dt=dt):
        return 0.0
    if _stable_at(make_ctrl, plant_fn(), hi, dt=dt):
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _stable_at(make_ctrl, plant_fn(), mid, dt=dt):
            lo = mid
        else:
            hi = mid
    return lo


def drift_plate(plate, scale):
    """Scale modal natural frequencies by `scale` (stiffness/temperature drift);
    mode shapes, actuator and sensor rows held fixed (as in the CL-SD drift)."""
    n = plate.n_modes
    plate.omega_n = plate.omega_n * scale
    plate.Kp = np.diag(plate.omega_n ** 2)
    plate.Cp = np.diag(2.0 * np.array(ZETA5[:n]) * plate.omega_n)
    if hasattr(plate, "freq_n"):
        plate.freq_n = plate.freq_n * scale
    return plate


def make_fopid_wrapped(plate_fn, P):
    c = _fopid(plate_fn(), P)
    return _SelfStateWrapper(c, NX5, c.nz)


def mk_lqg(weights, sensor="tip"):
    d3 = build(3, include_dynamics=False, sensor=sensor)
    l = LQGController(d3, dt=DT, verbose=False, u_max=U_MAX)
    if weights is None:
        l.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                           w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
    else:
        l.optimize_weights(w_q_list=[weights[0]], w_qd_list=[weights[1]],
                           w_r=1.0, gain_norm_max=1e12)
    l.discretize_observer()
    return l


def _lqg_lti(l):
    return (l.A - l.L_kal @ l.C - l.B @ l.K_lqr, l.L_kal, -l.K_lqr, np.zeros((1, 1)))


# ----------------------------------------------------------------------
def run(quick=False):
    T0 = time.time()
    out = dict(dt_us=DT * 1e6, u_max=U_MAX, rpm=RPM, sensor_fopid="collocated",
               nyquist_hz=0.5 / DT,
               oustaloup=dict(N=FOPID_N, wb_hz=FOPID_WB / 2 / np.pi,
                              wh_hz=FOPID_WH / 2 / np.pi,
                              wh_over_nyquist=FOPID_WH / 2 / np.pi / (0.5 / DT)))

    # ---- plants + CL-SD solvers ----
    design_pz3 = build(3, include_dynamics=False, sensor="piezo")
    design_tip3 = build(3, include_dynamics=False, sensor="tip")
    eval_pz = lambda: build(5, include_dynamics=True, sensor="piezo")
    eval_tip = lambda: build(5, include_dynamics=True, sensor="tip")
    clf3_pz = ClosedLoopFDM.from_plate(design_pz3, CUT, m_div=30)
    clf3_tip = ClosedLoopFDM.from_plate(design_tip3, CUT, m_div=30)
    clf5_pz = ClosedLoopFDM.from_plate(eval_pz(), CUT, m_div=30)
    clf5_tip = ClosedLoopFDM.from_plate(eval_tip(), CUT, m_div=30)

    out["OL"] = dict(ap_crit_nominal_mm=clf3_tip.ap_crit(RPM, None, ap_max=6e-3) * 1e3,
                     ap_crit_refined_mm=clf5_tip.ap_crit(RPM, None, ap_max=6e-3) * 1e3)
    print(f"OL a_p,crit: nominal={out['OL']['ap_crit_nominal_mm']:.3f} "
          f"refined={out['OL']['ap_crit_refined_mm']:.3f} mm")

    # ================= FOPID design + feasible selection =================
    print("\n=== FOPID design (CL-SD spectral-radius optimal) ===")
    winners = design_fopid(design_pz3, quick=quick)
    top = winners[:1 if quick else 2]

    print("  [design] feasible-metric selection (gain scale) on refined plant:")
    scales = [0.35, 0.5, 0.75, 1.0, 1.5] if quick else \
             [0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]
    best = dict(feasible=0.0, params=None, base=None)
    for P0 in top:
        for g in scales:
            Pg = dict(P0); Pg.update(Kp=P0["Kp"] * g, Ki=P0["Ki"] * g,
                                     Kd=P0["Kd"] * g, scale=g)
            f = _feasible(lambda Pg=Pg: make_fopid_wrapped(eval_pz, Pg), eval_pz)
            print(f"     sign={P0['sign']:+.0f} lam={P0['lam']:.2f} mu={P0['mu']:.2f}"
                  f" g={g:.2f}: feasible={f*1e3:.2f} mm [{time.time()-T0:.0f}s]",
                  flush=True)
            if f > best["feasible"]:
                best = dict(feasible=f, params=Pg, base=P0)
    P = best["params"]
    edge_note = ("scale at grid edge" if P["scale"] in (scales[0], scales[-1])
                 else "interior")
    print(f"  [design] SELECTED FOPID: sign={P['sign']:+.0f} lam={P['lam']:.2f} "
          f"mu={P['mu']:.2f} Kp={P['Kp']:.3g} Ki={P['Ki']:.3g} Kd={P['Kd']:.3g} "
          f"(scale {P['scale']}, {edge_note}) feasible={best['feasible']*1e3:.2f} mm")

    ap_lin_fop_nom = clf3_pz.ap_crit_dynamic(RPM, *_fopid(design_pz3, P).export_lti(),
                                             ap_max=6e-3) * 1e3
    ap_lin_fop_ref = clf5_pz.ap_crit_dynamic(RPM, *_fopid(design_pz3, P).export_lti(),
                                             ap_max=6e-3) * 1e3
    out["FOPID"] = dict(design={k: P[k] for k in
                               ("sign", "lam", "mu", "Kp", "Ki", "Kd", "scale")},
                        nz=_fopid(design_pz3, P).nz,
                        scale_grid=scales, scale_selected_edge=(edge_note != "interior"),
                        ap_crit_linear_nominal_mm=ap_lin_fop_nom,
                        ap_crit_linear_refined_mm=ap_lin_fop_ref,
                        ap_feasible_mm=best["feasible"] * 1e3)

    # ---- damping-optimal contrast (computed, not asserted) ----
    print("\n=== damping-optimal FOPID (contrast) ===")
    Pd = design_damping_optimal(design_pz3, P["sign"], quick=quick)
    dvc = dict(chatter_optimal=dict(
        lam=P["lam"], mu=P["mu"], ap_lin_nominal_mm=winners[0]["ap_lin_design_mm"]))
    if Pd is not None:
        ap_lin_damp = clf3_pz.ap_crit_dynamic(RPM, *_fopid(design_pz3, Pd).export_lti(),
                                              ap_max=6e-3) * 1e3
        f_damp = _feasible(lambda: make_fopid_wrapped(eval_pz, Pd), eval_pz)
        dvc["damping_optimal"] = dict(
            lam=Pd["lam"], mu=Pd["mu"], min_damping_pct=Pd["damping"] * 100,
            ap_lin_nominal_mm=ap_lin_damp, ap_feasible_mm=f_damp * 1e3)
        print(f"  damping-opt: min-damp={Pd['damping']*100:.2f}%  "
              f"ap_lin(nominal)={ap_lin_damp:.3f}mm  feasible={f_damp*1e3:.2f}mm  "
              f"vs chatter-opt ap_lin={winners[0]['ap_lin_design_mm']:.3f}mm "
              f"feasible={best['feasible']*1e3:.2f}mm")
    out["damping_vs_chatter"] = dvc

    # ================= ADRC (feasible-selected bandwidth) =================
    print("\n=== ADRC / LQG under identical conditions ===")
    adrc_rows = []
    for wc, wo in ([(1200, 14400), (1800, 21600)] if quick
                   else [(1200, 14400), (1800, 21600), (2400, 28800)]):
        f = _feasible(lambda wc=wc, wo=wo: _SelfStateWrapper(
            ADRCController(eval_pz(), dt=DT, wc=wc, wo=wo, u_max=U_MAX), NX5, NX5),
            eval_pz)
        adrc_rows.append(dict(wc=wc, wo=wo, feasible_mm=f * 1e3))
        print(f"  ADRC wc={wc}: feasible={f*1e3:.2f} mm [{time.time()-T0:.0f}s]",
              flush=True)
    best_adrc = max(adrc_rows, key=lambda r: r["feasible_mm"])
    a_dep = ADRCController(eval_pz(), dt=DT, wc=best_adrc["wc"], wo=best_adrc["wo"],
                          u_max=U_MAX)
    Aa, Ba, Ca, Da = a_dep.export_lti()
    out["ADRC"] = dict(tune=best_adrc, rows=adrc_rows,
                       ap_crit_linear_nominal_mm=clf3_pz.ap_crit_dynamic(
                           RPM, *ADRCController(design_pz3, dt=DT, wc=best_adrc["wc"],
                                                wo=best_adrc["wo"], u_max=U_MAX
                                                ).export_lti(), ap_max=6e-3) * 1e3,
                       ap_crit_linear_refined_mm=clf5_pz.ap_crit_dynamic(
                           RPM, Aa, Ba, Ca, Da, ap_max=6e-3) * 1e3,
                       ap_feasible_mm=best_adrc["feasible_mm"])

    # ================= LQG (feasible-selected weights), tip + collocated =====
    lqg_rows = []
    for label, w in ([("grid-default", None), ("w_q=1e16,w_qd=1e8", (1e16, 1e8))]
                     if quick else
                     [("grid-default", None), ("w_q=1e16,w_qd=1e8", (1e16, 1e8)),
                      ("w_q=1e12,w_qd=1e8", (1e12, 1e8))]):
        f = _feasible(lambda w=w: _SelfStateWrapper(mk_lqg(w), NX5, 2 * 3), eval_tip)
        lqg_rows.append(dict(label=label, weights=w, feasible_mm=f * 1e3))
        print(f"  LQG tip ({label}): feasible={f*1e3:.2f} mm [{time.time()-T0:.0f}s]",
              flush=True)
    best_lqg = max(lqg_rows, key=lambda r: r["feasible_mm"])
    l_dep = mk_lqg(best_lqg["weights"], sensor="tip")
    Acl, Bcl, Ccl, Dcl = _lqg_lti(l_dep)

    # LQG on the collocated sensor (isolate the sensor from the model-free effect)
    f_lqg_col = _feasible(lambda: _SelfStateWrapper(
        mk_lqg(best_lqg["weights"], sensor="piezo"), NX5, 2 * 3), eval_pz)
    print(f"  LQG collocated (same weights): feasible={f_lqg_col*1e3:.2f} mm")

    out["LQG"] = dict(
        tune_feasible=best_lqg["label"], rows=lqg_rows,
        ap_crit_linear_nominal_mm=clf3_tip.ap_crit_dynamic(
            RPM, Acl, Bcl, Ccl, Dcl, ap_max=6e-3) * 1e3,
        ap_crit_linear_refined_mm=clf5_tip.ap_crit_dynamic(
            RPM, Acl, Bcl, Ccl, Dcl, ap_max=6e-3) * 1e3,
        ap_feasible_mm=best_lqg["feasible_mm"],
        ap_feasible_collocated_mm=f_lqg_col * 1e3)

    # convenience summaries (ALL deployed designs, both linear plants)
    out["linear_nominal_mm"] = dict(
        OL=out["OL"]["ap_crit_nominal_mm"], LQG=out["LQG"]["ap_crit_linear_nominal_mm"],
        ADRC=out["ADRC"]["ap_crit_linear_nominal_mm"], FOPID=ap_lin_fop_nom)
    out["linear_refined_mm"] = dict(
        OL=out["OL"]["ap_crit_refined_mm"], LQG=out["LQG"]["ap_crit_linear_refined_mm"],
        ADRC=out["ADRC"]["ap_crit_linear_refined_mm"], FOPID=ap_lin_fop_ref)
    print(f"  linear (nominal 3-mode): {out['linear_nominal_mm']}")
    print(f"  linear (refined 5-mode): {out['linear_refined_mm']}")

    # ================= robustness: feasible depth under +/-20% drift =========
    print("\n=== robustness: voltage-feasible depth under +/-20% drift ===")
    rob = {}
    for scale, tag in [(0.8, "-20%"), (1.0, "nominal"), (1.2, "+20%")]:
        dpz = lambda s=scale: drift_plate(build(5, include_dynamics=True,
                                                sensor="piezo"), s)
        dtip = lambda s=scale: drift_plate(build(5, include_dynamics=True,
                                                 sensor="tip"), s)
        f_fop = _feasible(lambda: make_fopid_wrapped(dpz, P), dpz)
        f_adr = _feasible(lambda: _SelfStateWrapper(
            ADRCController(eval_pz(), dt=DT, wc=best_adrc["wc"], wo=best_adrc["wo"],
                          u_max=U_MAX), NX5, NX5), dpz)
        f_lqg = _feasible(lambda: _SelfStateWrapper(
            mk_lqg(best_lqg["weights"]), NX5, 6), dtip)
        rob[tag] = dict(FOPID=f_fop * 1e3, ADRC=f_adr * 1e3, LQG=f_lqg * 1e3)
        print(f"  {tag}: FOPID={f_fop*1e3:.2f} ADRC={f_adr*1e3:.2f} "
              f"LQG={f_lqg*1e3:.2f} mm [{time.time()-T0:.0f}s]", flush=True)
    out["robustness_feasible_mm"] = rob

    # ================= time traces at a common depth =========================
    print("\n=== time traces at common depth ===")
    ap_common = 1.2e-3
    traces = {}
    ctrls = {"FOPID": lambda: make_fopid_wrapped(eval_pz, P),
             "ADRC": lambda: _SelfStateWrapper(ADRCController(
                 eval_pz(), dt=DT, wc=best_adrc["wc"], wo=best_adrc["wo"],
                 u_max=U_MAX), NX5, NX5),
             "LQG": lambda: _SelfStateWrapper(mk_lqg(best_lqg["weights"]), NX5, 6)}
    plants = {"FOPID": eval_pz, "ADRC": eval_pz, "LQG": eval_tip}
    for name, mk in ctrls.items():
        pl = plants[name]()
        ok, r = _stable_at(mk, pl, ap_common, T=0.4, ret=True)
        ie = r['stop_idx']
        ytip = (pl.D_tip @ r['qm'][:, :ie + 1]) * 1e6
        u = r['u'][:ie + 1]; dec = 8
        traces[name] = dict(t=r['t'][:ie + 1][::dec].tolist(),
                            y_um=ytip[::dec].tolist(), u_V=u[::dec].tolist(),
                            yrms_um=float(np.sqrt(np.mean(ytip[len(ytip)//2:]**2))),
                            umax_V=float(np.max(np.abs(u))), stable=bool(ok))
        print(f"  {name} @ {ap_common*1e3:.1f}mm: yrms={traces[name]['yrms_um']:.3f}um "
              f"umax={traces[name]['umax_V']:.1f}V stable={ok}", flush=True)
    out["traces_common_depth_mm"] = ap_common * 1e3
    out["traces_summary"] = {k: {f: traces[k][f] for f in ("yrms_um", "umax_V", "stable")}
                             for k in traces}
    np.savez(os.path.join(RESULTS, "fopid_traces.npz"),
             **{f"{k}_{f}": np.array(traces[k][f])
                for k in traces for f in ("t", "y_um", "u_V")})

    # ---- FOPID controller Bode (fractional character) ----
    wgrid = np.logspace(np.log10(2 * np.pi * 50), np.log10(2 * np.pi * 8000), 200)
    Gc = _fopid(design_pz3, P).freqresp(wgrid)
    out["fopid_bode"] = dict(w=wgrid.tolist(),
                             mag_db=(20 * np.log10(np.abs(Gc))).tolist(),
                             phase_deg=(np.angle(Gc) * 180 / np.pi).tolist())

    out["summary"] = dict(
        ap_feasible_mm=dict(LQG=out["LQG"]["ap_feasible_mm"],
                            LQG_collocated=out["LQG"]["ap_feasible_collocated_mm"],
                            ADRC=out["ADRC"]["ap_feasible_mm"],
                            FOPID=out["FOPID"]["ap_feasible_mm"]),
        ap_crit_linear_nominal_mm=out["linear_nominal_mm"],
        ap_crit_linear_refined_mm=out["linear_refined_mm"])
    out["elapsed_s"] = time.time() - T0
    out["note"] = ("All controllers saturated at +/-150 V; FOPID and ADRC use the "
                   "collocated (minimum-phase) sensor, LQG the tip (an LQG-on-"
                   "collocated feasible is also reported to isolate the sensor). "
                   "Linear a_p,crit is given on the nominal 3-mode model (spillover-"
                   "free) AND the refined 5-mode plant for every deployed design. "
                   "Feasible a_p is saturated nonlinear time domain. FOPID orders+"
                   "gains minimize the CL-SD spectral radius, then a gain-scale line "
                   "search selects by the feasible metric; only the scalar scale is "
                   "feasible-selected, so the FOPID feasible number is, if anything, "
                   "conservative. Oustaloup wh kept below Nyquist (wh*dt/pi=0.4) so "
                   "the CL-SD (continuous) and time-domain (ZOH) controllers agree.")
    json.dump(out, open(os.path.join(RESULTS, "fopid.json"), "w"), indent=2)
    print(f"\nTOTAL {out['elapsed_s']:.0f}s -> results/fopid.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run(quick=args.quick)
