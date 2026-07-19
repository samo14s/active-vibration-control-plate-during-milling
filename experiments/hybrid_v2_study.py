"""
hybrid_v2_study.py
==================
Improvements to the HYBRID ADRC-FOPID tip-sensor controller (Sec. 4.11), each
targeting one of its three COMPUTED weaknesses:

  W1  -20 % frequency drift collapses the feasible depth to zero
      -> ROBUST MIN-MAX CO-DESIGN: the DE objective becomes the worst dominant
         Floquet multiplier over the drifted plants {-20 %, nominal, +20 %}
         (falls back to +/-10 % if the +/-20 % set is infeasible).
  W2  saturated ~37 um limit cycle at low depth (finish far worse than LQG)
      -> ESO LEAKAGE eps_leak (z3' += -eps z3): bounds the disturbance
         estimate so the mode-1 inversion cannot ram the actuator; and the
         margin-vs-finish trade is reported as a PARETO front over the design
         family instead of a single point.
  W3  linear boundary on the refined 5-mode plant collapses to the floor
      (spillover through the 3.4-4.1 kHz modes)
      -> LOW-PASS ROLL-OFF wf on the ESO injection: the mode-1 inversion is
         rolled off before the spillover modes.

Both new elements are LTI (leak: A_o[2,2] = -eps; filter: one extra state), so
the v2 hybrid embeds in the CL-SD monodromy exactly as before (module
validation: defaults reproduce v1 bit-exactly; the v2 export equals the
analytic (LP*Gy + F)/(1 - LP*Gu) closed form to 5e-15; closed-loop step vs
export agree to 0.03 %).

Also closes a disclosed v1 cap: the search runs BOTH ESO gain signs; the FOPID
loop sign stays sigma = +1 (its winning sign standalone on the tip; disclosed).
A bonus stage tests the regeneration-aware delayed channel
u_tau = g [y(t-tau) - y(t)] on the selected v2 via the CL-SD D_tau interface
(negative or positive, it is reported as computed).

Outputs: results/hybrid_v2.json (+ results/hybrid_v2_traces.npz)
Run:     python experiments/hybrid_v2_study.py [--quick]
"""
import os, sys, json, time, argparse
import numpy as np
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
RESULTS = os.path.join(ROOT, "results")

from hybrid_adrc_fopid import HybridADRCFOPID
from cl_fdm import ClosedLoopFDM
from model_refinement import build, _SelfStateWrapper
from fopid_study import (DT, U_MAX, RPM, CUT, NX5, FOPID_KW,
                         _feasible, _stable_at, drift_plate, mk_lqg)

AP_REF = 0.8e-3
M_DIV_DESIGN = 16
BOUNDS_V2 = ([(2.0, 10.5)] * 3            # log10 Kp, Ki, Kd
             + [(0.2, 0.95), (0.2, 0.95)]  # lam, mu
             + [(1.7, 3.3), (3.0, 12.0), (-1.5, 1.0)]   # log wc, wo/wc, log|b0|
             + [(-0.5, 3.5), (3.3, 4.6)])  # log10 eps_leak, log10 wf [rad/s]


def H_from_x(x, sign_b, sigma=+1.0):
    wc = 10.0 ** x[5]; wo = x[6] * wc
    return dict(sign=sigma, lam=float(x[3]), mu=float(x[4]),
                Kp=10.0 ** x[0], Ki=10.0 ** x[1], Kd=10.0 ** x[2],
                wc=wc, wo=wo, b0_eff=sign_b * 10.0 ** x[7],
                eps_leak=10.0 ** x[8], wf=10.0 ** x[9])


def make_v2(plate, H):
    return HybridADRCFOPID(plate, DT, wc=H["wc"], wo=H["wo"], b0_eff=H["b0_eff"],
                           Kp=H["Kp"], Ki=H["Ki"], Kd=H["Kd"],
                           lam=H["lam"], mu=H["mu"], sign=H["sign"],
                           eps_leak=H.get("eps_leak", 0.0), wf=H.get("wf"),
                           u_max=U_MAX, **FOPID_KW)


def _rho(clf, plate, H, ap):
    try:
        r = clf.rho_dynamic(RPM, ap, *make_v2(plate, H).export_lti())
        return r if np.isfinite(r) else 9.0
    except Exception:
        return 9.0


def drift_clf(plate, scale, m_div):
    clf = ClosedLoopFDM.from_plate(plate, CUT, m_div=m_div)
    Hpe = clf.B[clf.n:, 0].copy()
    d = ClosedLoopFDM(clf.omega_n * scale, clf.zeta, clf.Dp, Hpe, CUT, m_div=m_div)
    d.C_obs = clf.C_obs
    return d


def run(quick=False):
    T0 = time.time()
    v1 = json.load(open(os.path.join(RESULTS, "hybrid_tip.json")))
    H1 = dict(v1["HYBRID"]["design"]); H1.setdefault("eps_leak", 0.0)
    out = dict(dt_us=DT * 1e6, u_max=U_MAX, rpm=RPM, sensor="tip only",
               v1_design=v1["HYBRID"]["design"],
               v1_feasible_mm=v1["HYBRID"]["ap_feasible_mm"])

    tip3 = build(3, include_dynamics=False, sensor="tip")
    eval_tip = lambda: build(5, include_dynamics=True, sensor="tip")
    clf3f = ClosedLoopFDM.from_plate(tip3, CUT, m_div=30)
    clf5 = ClosedLoopFDM.from_plate(eval_tip(), CUT, m_div=30)
    solvers20 = {s: drift_clf(tip3, s, M_DIV_DESIGN) for s in (0.8, 1.0, 1.2)}
    solvers10 = {s: drift_clf(tip3, s, M_DIV_DESIGN) for s in (0.9, 1.0, 1.1)}

    maxiter = 6 if quick else 10
    popsize = 6 if quick else 8

    def de_search(objective, tag):
        best = None
        for sign_b in (+1.0, -1.0):
            def obj(x):
                return objective(H_from_x(x, sign_b))
            res = differential_evolution(obj, BOUNDS_V2, seed=0, maxiter=maxiter,
                                         popsize=popsize, tol=1e-2, init='sobol',
                                         polish=True)
            H = H_from_x(res.x, sign_b)
            print(f"  [{tag}] sign_b={sign_b:+.0f}: obj={res.fun:.3f} "
                  f"lam={H['lam']:.2f} mu={H['mu']:.2f} wc={H['wc']:.0f} "
                  f"wo={H['wo']:.0f} b0={H['b0_eff']:+.3g} eps={H['eps_leak']:.3g} "
                  f"wf={H['wf']/2/np.pi:.0f}Hz [{time.time()-T0:.0f}s]", flush=True)
            if best is None or res.fun < best[0]:
                best = (float(res.fun), H)
        return best

    # ---------- nominal-optimal v2 ----------
    print("=== v2-N: nominal-optimal co-design (leak + roll-off in the search) ===")
    def obj_nom(H):
        r0 = _rho(solvers20[1.0], tip3, H, 0.02e-3)
        if r0 >= 1.0:
            return 5.0 + min(r0, 9.0)
        return _rho(solvers20[1.0], tip3, H, AP_REF)
    objN, HN = de_search(obj_nom, "v2-N")

    # ---------- robust min-max v2 ----------
    print("=== v2-R: robust min-max co-design over drifted plants ===")
    def make_obj_robust(solvers):
        def obj_rob(H):
            # floors on ALL drifted plants first; penalty uses the WORST floor
            # rho (not the first failure) so the DE landscape is continuous and
            # iteration-order-independent.  Floor penalties sit >= 6.0.
            worst_floor = 0.0
            for s, clf in solvers.items():
                worst_floor = max(worst_floor, _rho(clf, tip3, H, 0.02e-3))
            if worst_floor >= 1.0:
                return 5.0 + min(worst_floor, 9.0)
            worst = 0.0
            for s, clf in solvers.items():
                worst = max(worst, _rho(clf, tip3, H, AP_REF))
            return worst
        return obj_rob
    objR, HR = de_search(make_obj_robust(solvers20), "v2-R(20%)")
    robust_band = 20
    if objR >= 6.0:     # floor-infeasible over +/-20 % (>= 6.0 by construction)
        print("  [v2-R] +/-20% floor-infeasible in budget -> retrying +/-10%")
        objR, HR = de_search(make_obj_robust(solvers10), "v2-R(10%)")
        robust_band = 10
    out["robust_band_pct"] = robust_band
    out["obj_v2N"], out["obj_v2R"] = float(objN), float(objR)

    # ---------- Stage C: feasible + Pareto (margin vs finish) ----------
    print("=== feasible selection + Pareto (margin vs low-depth finish) ===")
    scales = [0.5, 1.0, 1.5] if quick else [0.2, 0.25, 0.35, 0.5, 0.75, 1.0,
                                            1.5, 2.0]

    def evaluate_family(H0, tag):
        rows, best = [], dict(feasible=0.0, H=None)
        for g in scales:
            Hg = dict(H0); Hg.update(Kp=H0["Kp"] * g, Ki=H0["Ki"] * g,
                                     Kd=H0["Kd"] * g, scale=g)
            f = _feasible(lambda: _SelfStateWrapper(
                make_v2(eval_tip(), Hg), NX5, NX5), eval_tip)
            pl = eval_tip()
            ok, r = _stable_at(lambda: _SelfStateWrapper(
                make_v2(pl, Hg), NX5, NX5), pl, 0.15e-3, T=0.3, ret=True)
            rms15 = umax15 = None
            if ok and r is not None:
                ie = r['stop_idx']
                y = (pl.D_tip @ r['qm'][:, :ie + 1]) * 1e6
                rms15 = float(np.sqrt(np.mean(y[len(y) // 2:] ** 2)))
                umax15 = float(np.max(np.abs(r['u'][:ie + 1])))
            rows.append(dict(scale=g, feasible_mm=f * 1e3, rms15_um=rms15,
                             umax15_V=umax15))
            print(f"  [{tag}] g={g:.2f}: feasible={f*1e3:.2f} mm  "
                  f"rms@0.15={rms15 if rms15 is None else round(rms15,2)} um "
                  f"[{time.time()-T0:.0f}s]", flush=True)
            if f > best["feasible"]:
                best = dict(feasible=f, H=Hg)
        edge = (best["H"] is not None
                and best["H"]["scale"] in (scales[0], scales[-1]))
        return rows, best, edge

    rowsN, bestN, edgeN = evaluate_family(HN, "v2-N")
    rowsR, bestR, edgeR = evaluate_family(HR, "v2-R")
    out["pareto_v2N"] = rowsN
    out["pareto_v2R"] = rowsR
    out["scale_selected_edge"] = dict(v2N=bool(edgeN), v2R=bool(edgeR))

    # ---------- boundaries + drift for the selected designs ----------
    print("=== boundaries + drift robustness ===")
    res_designs = {}
    for tag, best, H0 in (("v2N", bestN, HN), ("v2R", bestR, HR)):
        H = best["H"] if best["H"] is not None else H0
        hyb = make_v2(tip3, H)
        ap_nom = clf3f.ap_crit_dynamic(RPM, *hyb.export_lti(), ap_max=6e-3) * 1e3
        ap_ref5 = clf5.ap_crit_dynamic(RPM, *hyb.export_lti(), ap_max=6e-3) * 1e3
        drift = {}
        for scale, dtag in [(0.8, "-20%"), (1.0, "nominal"), (1.2, "+20%")]:
            dtip = lambda s=scale: drift_plate(build(5, include_dynamics=True,
                                                     sensor="tip"), s)
            f = _feasible(lambda: _SelfStateWrapper(
                make_v2(eval_tip(), H), NX5, NX5), dtip)
            drift[dtag] = f * 1e3
        res_designs[tag] = dict(
            design={k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in H.items()},
            ap_feasible_mm=best["feasible"] * 1e3,
            ap_crit_linear_nominal_mm=ap_nom,
            ap_crit_linear_refined_mm=ap_ref5,
            drift_feasible_mm=drift)
        print(f"  [{tag}] feasible={best['feasible']*1e3:.2f}  lin_nom={ap_nom:.2f}  "
              f"lin_refined={ap_ref5:.3f}  drift={ {k: round(v,2) for k,v in drift.items()} }",
              flush=True)
    out["designs"] = res_designs

    # ---------- in-run v1 baseline + attribution ablation ----------
    print("=== v1 baseline in-pipeline + stripped-elements ablation ===")
    H1 = dict(v1["HYBRID"]["design"]); H1.setdefault("eps_leak", 0.0)
    H1.setdefault("wf", None)
    f1 = _feasible(lambda: _SelfStateWrapper(make_v2(eval_tip(), H1), NX5, NX5),
                   eval_tip)
    out["v1_reevaluated_feasible_mm"] = f1 * 1e3
    print(f"  v1 in-pipeline feasible = {f1*1e3:.2f} mm "
          f"(stored: {v1['HYBRID']['ap_feasible_mm']:.2f})", flush=True)
    for tag in ("v2N", "v2R"):
        Hs = dict(res_designs[tag]["design"]); Hs.update(eps_leak=0.0, wf=None)
        fs = _feasible(lambda: _SelfStateWrapper(make_v2(eval_tip(), Hs),
                                                 NX5, NX5), eval_tip)
        res_designs[tag]["ablation_stripped_feasible_mm"] = fs * 1e3
        print(f"  [{tag}] elements stripped (eps=0, wf=None, gains kept): "
              f"feasible={fs*1e3:.2f} mm "
              f"(with: {res_designs[tag]['ap_feasible_mm']:.2f})", flush=True)

    # ---------- bonus: regeneration-aware delayed channel on the winner ----
    # ESO-AWARE realization: u_tau = g [y(t-tau) - y(t)] enters the total
    # voltage, so the exported ESO must see it too:
    #   Bc'  = Bc - g B_o (ESO rows),  Bc_tau = +g B_o,  Dc' = Dc - g,
    #   D_tau = g.  (The ESO-blind plant-input-only variant is also reported.)
    print("=== bonus: delayed channel u_tau = g[y(t-tau) - y(t)] (CL-SD) ===")
    pick = "v2R" if (res_designs["v2R"]["drift_feasible_mm"]["-20%"] > 0.05
                     ) else "v2N"
    Hbest = res_designs[pick]["design"]
    hyb = make_v2(tip3, Hbest)
    Ac, Bc, Cc, Dc = hyb.export_lti()
    B_o = hyb.B_o
    base = res_designs[pick]["ap_crit_linear_nominal_mm"]
    rows_tau, rows_tau_blind = [], []
    for g in ([-2.0, -0.5, 0.5, 2.0] if quick
              else [-5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0]):
        Bc_mod = Bc.copy(); Bc_mod[:3, 0] -= g * B_o
        Bc_tau = np.zeros_like(Bc); Bc_tau[:3, 0] = g * B_o
        ap = clf3f.ap_crit_dynamic(RPM, Ac, Bc_mod, Cc, Dc - g, ap_max=6e-3,
                                   Bc_tau=Bc_tau, D_tau=np.array([[g]])) * 1e3
        ap_b = clf3f.ap_crit_dynamic(RPM, Ac, Bc, Cc, Dc - g, ap_max=6e-3,
                                     D_tau=np.array([[g]])) * 1e3
        rows_tau.append(dict(g=g, ap_lin_nominal_mm=ap))
        rows_tau_blind.append(dict(g=g, ap_lin_nominal_mm=ap_b))
        print(f"  g={g:+.1f}: ap_lin={ap:.3f} (eso-aware) / {ap_b:.3f} "
              f"(eso-blind) mm  (base {base:.3f})", flush=True)
    best_tau = max(rows_tau, key=lambda r: r["ap_lin_nominal_mm"])
    out["delayed_channel"] = dict(applied_to=pick, base_ap_lin_mm=base,
                                  rows_eso_aware=rows_tau,
                                  rows_eso_blind=rows_tau_blind,
                                  best_gain_mm=best_tau["ap_lin_nominal_mm"],
                                  improvement_pct=(best_tau["ap_lin_nominal_mm"]
                                                   / base - 1) * 100)

    # ---------- traces for the selected v2 designs ----------
    print("=== traces ===")
    traces = {}
    for tag in ("v2N", "v2R"):
        H = res_designs[tag]["design"]
        for ap_c in (0.15e-3, 0.5e-3):
            pl = eval_tip()
            ok, r = _stable_at(lambda: _SelfStateWrapper(
                make_v2(pl, H), NX5, NX5), pl, ap_c, T=0.3, ret=True)
            if r is None:
                continue
            ie = r['stop_idx']
            y = (pl.D_tip @ r['qm'][:, :ie + 1]) * 1e6
            key = f"{tag}_{ap_c*1e3:.2f}"
            traces[key] = dict(t=r['t'][:ie + 1][::8], y_um=y[::8],
                               u_V=r['u'][:ie + 1][::8], stable=bool(ok),
                               yrms_um=(float(np.sqrt(np.mean(y[len(y)//2:]**2)))
                                        if ok else None),
                               umax_V=float(np.max(np.abs(r['u'][:ie + 1]))))
            print(f"  {key}: stable={ok} yrms={traces[key]['yrms_um']} "
                  f"umax={traces[key]['umax_V']:.1f}", flush=True)
    out["trace_summary"] = {k: dict(stable=v["stable"], yrms_um=v["yrms_um"],
                                    umax_V=v["umax_V"]) for k, v in traces.items()}
    np.savez(os.path.join(RESULTS, "hybrid_v2_traces.npz"),
             **{f"{k}_{f}": np.asarray(traces[k][f])
                for k in traces for f in ("t", "y_um", "u_V")})

    out["elapsed_s"] = time.time() - T0
    out["disclosures"] = dict(
        de_budget=dict(maxiter=maxiter, popsize=popsize, tol=1e-2, init="sobol",
                       seed=0, b0_signs=[+1, -1], sigma_fixed=+1, quick=quick),
        design_stage_vs_v1=dict(
            ap_ref_mm=[AP_REF * 1e3, 1.0], m_div_design=[M_DIV_DESIGN, 20],
            note=("v2 design stage used ap_ref=0.8mm / m_div=16 (speed) vs "
                  "v1's 1.0mm / 20; all REPORTED boundaries and feasible "
                  "numbers use the identical evaluation machinery (m_div=30, "
                  "same bisection), and the stripped-elements ablation "
                  "addresses attribution of the improvement.")),
        step_vs_export=("The deployed step() carries the ESO's one-sample "
                        "u-channel delay (ADRCController convention); the CL-SD "
                        "export folds u instantaneously. All feasibility "
                        "numbers use step() itself (conservative)."),
        validation="experiments/test_hybrid_validation.py")
    out["note"] = ("v2 hybrid = v1 + ESO leakage eps_leak + first-order roll-off "
                   "wf on the ESO injection, both searched by DE alongside the 8 "
                   "v1 parameters. v2-N optimizes the nominal CL-SD margin; v2-R "
                   "optimizes the WORST margin over drifted plants (min-max, "
                   "worst-floor penalty). sigma fixed +1 (winning standalone tip "
                   "sign; disclosed). Pareto rows expose the margin-vs-finish "
                   "trade instead of a single point. Delayed-channel bonus uses "
                   "the CL-SD Bc_tau/D_tau interface, ESO-aware realization "
                   "(ESO-blind variant reported for comparison).")
    json.dump(out, open(os.path.join(RESULTS, "hybrid_v2.json"), "w"), indent=2)
    print(f"\nTOTAL {out['elapsed_s']:.0f}s -> results/hybrid_v2.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run(quick=args.quick)
