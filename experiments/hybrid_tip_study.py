"""
hybrid_tip_study.py
===================
HYBRID (band-limited ADRC + FOPID) controller on the article's original tip
sensor, and the final tip-only comparison:

    open loop | LQG | plain ADRC | plain FOPID | HYBRID   -- all on y_tip.

The tip measurement is the sensor of Du et al. (2024); this study keeps that
system unchanged (no collocated sensor anywhere).  Component baselines (LQG,
plain ADRC, plain FOPID on the tip) come from fopid_tip_study.py; this script
designs the hybrid and evaluates everything a controller must survive:

  A. frozen-FOPID stage: keep the best tip FOPID and add ONLY the band-limited
     ESO (3 free parameters: wc, wo/wc, signed b0_eff) -- the direct answer to
     "does merging ADRC into FOPID help on the tip?"
  B. joint stage: differential evolution over all 8 hybrid parameters
     (FOPID gains+orders and the ESO triple), spectral-radius objective.
  C. feasible-metric selection (gain-scale line search, as for every other
     controller), linear boundaries on the nominal AND refined plants,
     voltage-feasible depth, +/-20% drift robustness, and rejection traces.

Design metric: dominant Floquet multiplier of the closed-loop
semi-discretization at a reference depth (chatter margin), controller states in
the monodromy -- identical to the FOPID/ADRC/LQG treatment.  Saturation
+/-150 V everywhere; dt = 25 us; refined 5-mode plant for evaluation.

Outputs: results/hybrid_tip.json (+ results/hybrid_tip_traces.npz)
Run:     python experiments/hybrid_tip_study.py [--quick]
"""
import os, sys, json, time, argparse
import numpy as np
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
RESULTS = os.path.join(ROOT, "results")

from adrc_control import ADRCController
from hybrid_adrc_fopid import HybridADRCFOPID
from cl_fdm import ClosedLoopFDM
from model_refinement import build, _SelfStateWrapper
from fopid_study import (DT, U_MAX, RPM, CUT, NX5, FOPID_KW,
                         _fopid, _feasible, _stable_at, drift_plate,
                         make_fopid_wrapped, mk_lqg, _lqg_lti)

AP_REF = 1.0e-3          # design reference depth for the rho objective
ESO_BOUNDS = [(1.7, 3.3),        # log10 wc      (50 .. 2000 rad/s)
              (3.0, 12.0),       # wo / wc
              (-1.5, 1.0)]       # log10 |b0_eff|
FOPID_BOUNDS = [(2.0, 10.5)] * 3 + [(0.2, 0.95), (0.2, 0.95)]


def make_hybrid(plate, H):
    return HybridADRCFOPID(plate, DT, wc=H["wc"], wo=H["wo"], b0_eff=H["b0_eff"],
                           Kp=H["Kp"], Ki=H["Ki"], Kd=H["Kd"],
                           lam=H["lam"], mu=H["mu"], sign=H["sign"],
                           u_max=U_MAX, **FOPID_KW)


def _rho_h(clf, plate, H, ap):
    try:
        r = clf.rho_dynamic(RPM, ap, *make_hybrid(plate, H).export_lti())
        return r if np.isfinite(r) else 9.0
    except Exception:
        return 9.0


def run(quick=False):
    T0 = time.time()
    tipjson = json.load(open(os.path.join(RESULTS, "fopid_tip.json")))
    P_tip = tipjson["FOPID"]["design"]          # best plain tip FOPID (sign=+1)
    out = dict(dt_us=DT * 1e6, u_max=U_MAX, rpm=RPM,
               sensor="tip (article's original measurement; only sensor used)",
               baselines_from="fopid_tip.json",
               fopid_tip_design=P_tip)

    tip3 = build(3, include_dynamics=False, sensor="tip")
    eval_tip = lambda: build(5, include_dynamics=True, sensor="tip")
    clf3 = ClosedLoopFDM.from_plate(tip3, CUT, m_div=20)
    clf3f = ClosedLoopFDM.from_plate(tip3, CUT, m_div=30)     # fine, for ap_crit
    clf5 = ClosedLoopFDM.from_plate(eval_tip(), CUT, m_div=30)

    # ---------- Stage A: frozen FOPID + ESO search ----------
    print("=== Stage A: frozen tip-FOPID + band-limited ESO (3-param DE) ===")
    stageA = []
    maxiterA = 10 if quick else 15
    for sign_b in (+1.0, -1.0):
        def objA(x):
            wc = 10.0 ** x[0]; wo = x[1] * wc; b0 = sign_b * 10.0 ** x[2]
            H = dict(P_tip); H.update(wc=wc, wo=wo, b0_eff=b0)
            r0 = _rho_h(clf3, tip3, H, 0.02e-3)
            if r0 >= 1.0:
                return 5.0 + min(r0, 9.0)
            return _rho_h(clf3, tip3, H, AP_REF)
        res = differential_evolution(objA, ESO_BOUNDS, seed=0, maxiter=maxiterA,
                                     popsize=10, tol=1e-2, init='sobol', polish=True)
        wc = 10.0 ** res.x[0]; wo = res.x[1] * wc; b0 = sign_b * 10.0 ** res.x[2]
        H = dict(P_tip); H.update(wc=wc, wo=wo, b0_eff=b0)
        ap = clf3f.ap_crit_dynamic(RPM, *make_hybrid(tip3, H).export_lti(),
                                   ap_max=6e-3, n_coarse=25) * 1e3
        stageA.append(dict(H=H, ap_lin_nominal_mm=ap, obj=float(res.fun)))
        print(f"  [A] sign_b={sign_b:+.0f}: obj={res.fun:.3f} wc={wc:.0f} "
              f"wo={wo:.0f} b0={b0:+.3g} ap_lin={ap:.3f}mm [{time.time()-T0:.0f}s]",
              flush=True)

    # ---------- Stage B: joint 8-parameter search ----------
    print("=== Stage B: joint FOPID+ESO DE (8 params; sigma=+1, both b0 signs) ===")
    stageB = []
    maxiterB = 8 if quick else 10
    for sign_b in (+1.0, -1.0):
        def objB(x):
            wc = 10.0 ** x[5]; wo = x[6] * wc; b0 = sign_b * 10.0 ** x[7]
            H = dict(sign=+1.0, lam=x[3], mu=x[4],
                     Kp=10.0 ** x[0], Ki=10.0 ** x[1], Kd=10.0 ** x[2],
                     wc=wc, wo=wo, b0_eff=b0)
            r0 = _rho_h(clf3, tip3, H, 0.02e-3)
            if r0 >= 1.0:
                return 5.0 + min(r0, 9.0)
            return _rho_h(clf3, tip3, H, AP_REF)
        res = differential_evolution(objB, FOPID_BOUNDS + ESO_BOUNDS, seed=0,
                                     maxiter=maxiterB, popsize=8, tol=1e-2,
                                     init='sobol', polish=True)
        x = res.x
        wc = 10.0 ** x[5]; wo = x[6] * wc; b0 = sign_b * 10.0 ** x[7]
        H = dict(sign=+1.0, lam=float(x[3]), mu=float(x[4]),
                 Kp=10.0 ** x[0], Ki=10.0 ** x[1], Kd=10.0 ** x[2],
                 wc=wc, wo=wo, b0_eff=b0)
        ap = clf3f.ap_crit_dynamic(RPM, *make_hybrid(tip3, H).export_lti(),
                                   ap_max=6e-3, n_coarse=25) * 1e3
        stageB.append(dict(H=H, ap_lin_nominal_mm=ap, obj=float(res.fun)))
        print(f"  [B] sign_b={sign_b:+.0f}: obj={res.fun:.3f} lam={H['lam']:.2f} "
              f"mu={H['mu']:.2f} wc={wc:.0f} wo={wo:.0f} b0={b0:+.3g} "
              f"ap_lin={ap:.3f}mm [{time.time()-T0:.0f}s]", flush=True)

    cands = sorted(stageA + stageB, key=lambda c: -c["ap_lin_nominal_mm"])
    out["stageA"] = [dict(ap_lin_nominal_mm=c["ap_lin_nominal_mm"], obj=c["obj"],
                          **{k: c["H"][k] for k in
                             ("wc", "wo", "b0_eff", "lam", "mu")}) for c in stageA]
    out["stageB"] = [dict(ap_lin_nominal_mm=c["ap_lin_nominal_mm"], obj=c["obj"],
                          **{k: c["H"][k] for k in
                             ("wc", "wo", "b0_eff", "lam", "mu")}) for c in stageB]

    # ---------- Stage C: feasible-metric selection ----------
    print("=== Stage C: feasible selection (gain scale on the FOPID branch) ===")
    scales = [0.5, 1.0, 1.5] if quick else [0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]
    best = dict(feasible=0.0, H=None)
    for c in cands[:2 if quick else 3]:
        for g in scales:
            Hg = dict(c["H"]); Hg.update(Kp=c["H"]["Kp"] * g, Ki=c["H"]["Ki"] * g,
                                         Kd=c["H"]["Kd"] * g, scale=g)
            f = _feasible(lambda Hg=Hg: _SelfStateWrapper(
                make_hybrid(eval_tip(), Hg), NX5, NX5), eval_tip)
            print(f"  [C] ap_lin={c['ap_lin_nominal_mm']:.2f} g={g:.2f}: "
                  f"feasible={f*1e3:.2f} mm [{time.time()-T0:.0f}s]", flush=True)
            if f > best["feasible"]:
                best = dict(feasible=f, H=Hg)
    H = best["H"]
    print(f"  SELECTED HYBRID: wc={H['wc']:.0f} wo={H['wo']:.0f} "
          f"b0={H['b0_eff']:+.3g} lam={H['lam']:.2f} mu={H['mu']:.2f} "
          f"scale={H['scale']} feasible={best['feasible']*1e3:.2f} mm")

    hyb = make_hybrid(tip3, H)
    ap_nom = clf3f.ap_crit_dynamic(RPM, *hyb.export_lti(), ap_max=6e-3) * 1e3
    ap_ref = clf5.ap_crit_dynamic(RPM, *hyb.export_lti(), ap_max=6e-3) * 1e3
    out["HYBRID"] = dict(design={k: H[k] for k in
                                ("sign", "lam", "mu", "Kp", "Ki", "Kd",
                                 "wc", "wo", "b0_eff", "scale")},
                         nz=hyb.nz,
                         ap_crit_linear_nominal_mm=ap_nom,
                         ap_crit_linear_refined_mm=ap_ref,
                         ap_feasible_mm=best["feasible"] * 1e3)
    print(f"  HYBRID linear: nominal={ap_nom:.3f} refined={ap_ref:.3f} mm")

    # ---------- drift robustness (feasible metric), all tip controllers ------
    print("=== drift robustness (voltage-feasible, +/-20%) ===")
    lqg_w = {"grid-default": None, "w_q=1e16,w_qd=1e8": (1e16, 1e8),
             "w_q=1e12,w_qd=1e8": (1e12, 1e8)}[tipjson["LQG"]["tune"]]
    P_fop = tipjson["FOPID"]["design"]
    rob = {}
    for scale, tag in [(0.8, "-20%"), (1.0, "nominal"), (1.2, "+20%")]:
        dtip = lambda s=scale: drift_plate(build(5, include_dynamics=True,
                                                 sensor="tip"), s)
        f_hyb = _feasible(lambda: _SelfStateWrapper(
            make_hybrid(eval_tip(), H), NX5, NX5), dtip)
        f_lqg = _feasible(lambda: _SelfStateWrapper(mk_lqg(lqg_w, "tip"), NX5, 6), dtip)
        f_fop = _feasible(lambda: make_fopid_wrapped(eval_tip, P_fop), dtip)
        rob[tag] = dict(HYBRID=f_hyb * 1e3, LQG=f_lqg * 1e3, FOPID=f_fop * 1e3)
        print(f"  {tag}: HYBRID={f_hyb*1e3:.2f} LQG={f_lqg*1e3:.2f} "
              f"FOPID={f_fop*1e3:.2f} mm [{time.time()-T0:.0f}s]", flush=True)
    out["robustness_feasible_mm"] = rob

    # ---------- traces: rejection at 0.15 mm and stability demo at 0.5 mm ----
    print("=== traces ===")
    traces = {}
    mk = {"LQG": lambda: _SelfStateWrapper(mk_lqg(lqg_w, "tip"), NX5, 6),
          "FOPID": lambda: make_fopid_wrapped(eval_tip, P_fop),
          "HYBRID": lambda: _SelfStateWrapper(make_hybrid(eval_tip(), H), NX5, NX5),
          "ADRC": lambda: _SelfStateWrapper(ADRCController(
              eval_tip(), dt=DT, wc=800, wo=9600, u_max=U_MAX), NX5, NX5)}
    for ap_c, taglist in [(0.15e-3, ["LQG", "FOPID", "HYBRID"]),
                          (0.5e-3, ["LQG", "FOPID", "HYBRID", "ADRC"])]:
        for name in taglist:
            pl = eval_tip()
            ok, r = _stable_at(mk[name], pl, ap_c, T=0.3, dt=DT, ret=True)
            if r is None:
                continue
            ie = r['stop_idx']
            y = (pl.D_tip @ r['qm'][:, :ie + 1]) * 1e6
            u = r['u'][:ie + 1]
            key = f"{name}_{ap_c*1e3:.2f}"
            traces[key] = dict(t=r['t'][:ie + 1][::8], y_um=y[::8], u_V=u[::8],
                               stable=bool(ok),
                               yrms_um=(float(np.sqrt(np.mean(y[len(y)//2:]**2)))
                                        if ok else None),
                               umax_V=float(np.max(np.abs(u))))
            print(f"  {name} @ {ap_c*1e3:.2f}mm: stable={ok} "
                  f"yrms={traces[key]['yrms_um']} umax={traces[key]['umax_V']:.1f}",
                  flush=True)
    out["trace_summary"] = {k: dict(stable=v["stable"], yrms_um=v["yrms_um"],
                                    umax_V=v["umax_V"]) for k, v in traces.items()}
    np.savez(os.path.join(RESULTS, "hybrid_tip_traces.npz"),
             **{f"{k}_{f}": np.asarray(traces[k][f])
                for k in traces for f in ("t", "y_um", "u_V")})

    # ---------- hybrid Bode + final tip-only summary ----------
    wgrid = np.logspace(np.log10(2 * np.pi * 30), np.log10(2 * np.pi * 8000), 200)
    Gc = hyb.freqresp(wgrid)
    out["hybrid_bode"] = dict(w=wgrid.tolist(),
                              mag_db=(20 * np.log10(np.abs(Gc))).tolist(),
                              phase_deg=(np.angle(Gc) * 180 / np.pi).tolist())

    out["summary_tip_feasible_mm"] = dict(
        OL_linear=tipjson["OL"]["ap_crit_refined_mm"],
        LQG=tipjson["LQG"]["ap_feasible_mm"],
        ADRC=tipjson["ADRC"]["best_feasible_mm"],
        FOPID=tipjson["FOPID"]["ap_feasible_mm"],
        HYBRID=out["HYBRID"]["ap_feasible_mm"])
    out["elapsed_s"] = time.time() - T0
    out["note"] = ("Tip sensor only (the article's original measurement). Hybrid = "
                   "band-limited ESO (wc, wo, signed b0_eff free) + FOPID, one "
                   "voltage output; both branches in the CL-SD monodromy. Design: "
                   "stage-A frozen-FOPID + ESO, stage-B joint DE (sigma=+1 only, "
                   "disclosed budget cap), feasible-metric gain-scale selection. "
                   "Baselines (LQG/ADRC/FOPID on tip) from fopid_tip.json. All "
                   "controllers saturated at +/-150 V.")
    json.dump(out, open(os.path.join(RESULTS, "hybrid_tip.json"), "w"), indent=2)
    print(f"\nTOTAL {out['elapsed_s']:.0f}s -> results/hybrid_tip.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run(quick=args.quick)
