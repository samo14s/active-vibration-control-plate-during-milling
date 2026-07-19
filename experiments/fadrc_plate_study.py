"""
fadrc_plate_study.py
====================
The ADRC-FOPID strategy of thesis Chapter 2 (ESO + fractional PID on the
estimates, metaheuristic tuning), DEVELOPED for the thin flexible plate and
evaluated on the article's original tip sensor by the full honest pipeline.

Developments relative to the chapter-2 beam version (all LTI, all searched):
  * signed free effective gain b0_eff  (NMP survival: lock onto mode 1)
  * free observer bandwidth wo         (band-limit the inversion)
  * disturbance-state leakage eps      (bound the estimate under saturation)
  * injection roll-off wf              (protect the 3.4-4.1 kHz modes)
The fractional orders lambda, mu and the gains Kp, Ki, Kd are searched jointly
with these, by differential evolution on the closed-loop semi-discretization
spectral radius (the same family of population metaheuristics as the PSO of
chapter 2, applied to the true chatter metric), then the deployed gain scale is
selected by the voltage-feasible metric.

Baselines read from the committed studies: LQG / plain ADRC / plain FOPID on
the tip (fopid_tip.json), parallel hybrid v1 (hybrid_tip.json) and v2
(hybrid_v2.json).

Outputs: results/fadrc_tip.json (+ results/fadrc_tip_traces.npz)
Run:     python experiments/fadrc_plate_study.py [--quick]
"""
import os, sys, json, time, argparse
import numpy as np
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
RESULTS = os.path.join(ROOT, "results")

from fadrc_control import FADRCController
from cl_fdm import ClosedLoopFDM
from model_refinement import build, _SelfStateWrapper
from fopid_study import (DT, U_MAX, RPM, CUT, NX5, FOPID_KW,
                         _feasible, _stable_at, drift_plate)

AP_REF = 1.0e-3
BOUNDS = ([(2.0, 10.5)] * 3                    # log10 Kp, Ki, Kd
          + [(0.2, 0.95), (0.2, 0.95)]         # lam, mu
          + [(2.0, 4.3), (-1.5, 1.0)]          # log10 wo, log10 |b0|
          + [(-0.5, 3.5), (3.3, 4.6)])         # log10 eps, log10 wf


def H_from_x(x, sign_b):
    return dict(Kp=10.0 ** x[0], Ki=10.0 ** x[1], Kd=10.0 ** x[2],
                lam=float(x[3]), mu=float(x[4]),
                wo=10.0 ** x[5], b0_eff=sign_b * 10.0 ** x[6],
                eps_leak=10.0 ** x[7], wf=10.0 ** x[8])


def make_fadrc(plate, H):
    return FADRCController(plate, DT, Kp=H["Kp"], Ki=H["Ki"], Kd=H["Kd"],
                           lam=H["lam"], mu=H["mu"], wo=H["wo"],
                           b0_eff=H["b0_eff"], eps_leak=H.get("eps_leak", 0.0),
                           wf=H.get("wf"), u_max=U_MAX, **FOPID_KW)


def _rho(clf, plate, H, ap):
    try:
        r = clf.rho_dynamic(RPM, ap, *make_fadrc(plate, H).export_lti())
        return r if np.isfinite(r) else 9.0
    except Exception:
        return 9.0


def run(quick=False):
    T0 = time.time()
    out = dict(dt_us=DT * 1e6, u_max=U_MAX, rpm=RPM,
               sensor="tip only (article's measurement)",
               strategy="chapter-2 ADRC-FOPID developed for the plate")

    tip3 = build(3, include_dynamics=False, sensor="tip")
    eval_tip = lambda: build(5, include_dynamics=True, sensor="tip")
    clf3 = ClosedLoopFDM.from_plate(tip3, CUT, m_div=20)
    clf3f = ClosedLoopFDM.from_plate(tip3, CUT, m_div=30)
    clf5 = ClosedLoopFDM.from_plate(eval_tip(), CUT, m_div=30)

    # ---------- DE co-design on the CL-SD spectral radius ----------
    print("=== FADRC design (chapter-2 strategy + plate developments) ===")
    maxiter = 6 if quick else 10
    popsize = 6 if quick else 8
    cands = []
    for sign_b in (+1.0, -1.0):
        def obj(x):
            H = H_from_x(x, sign_b)
            r0 = _rho(clf3, tip3, H, 0.02e-3)
            if r0 >= 1.0:
                return 5.0 + min(r0, 9.0)
            return _rho(clf3, tip3, H, AP_REF)
        res = differential_evolution(obj, BOUNDS, seed=0, maxiter=maxiter,
                                     popsize=popsize, tol=1e-2, init='sobol',
                                     polish=True)
        H = H_from_x(res.x, sign_b)
        ap = clf3f.ap_crit_dynamic(RPM, *make_fadrc(tip3, H).export_lti(),
                                   ap_max=6e-3, n_coarse=25) * 1e3
        cands.append(dict(H=H, obj=float(res.fun), ap_lin_nominal_mm=ap))
        print(f"  [design] sign_b={sign_b:+.0f}: obj={res.fun:.3f} "
              f"lam={H['lam']:.2f} mu={H['mu']:.2f} wo={H['wo']:.0f} "
              f"b0={H['b0_eff']:+.3g} eps={H['eps_leak']:.3g} "
              f"wf={H['wf']/2/np.pi:.0f}Hz ap_lin={ap:.3f}mm "
              f"[{time.time()-T0:.0f}s]", flush=True)
    cands.sort(key=lambda c: -c["ap_lin_nominal_mm"])
    out["design_candidates"] = [dict(obj=c["obj"],
                                     ap_lin_nominal_mm=c["ap_lin_nominal_mm"],
                                     **{k: c["H"][k] for k in
                                        ("lam", "mu", "wo", "b0_eff",
                                         "eps_leak", "wf")}) for c in cands]

    # ---------- feasible-metric gain-scale selection ----------
    print("=== feasible selection (gain scale) ===")
    scales = [0.5, 1.0, 1.5] if quick else [0.2, 0.25, 0.35, 0.5, 0.75,
                                            1.0, 1.5, 2.0]
    best = dict(feasible=0.0, H=None)
    rows = []
    for c in cands[:1 if quick else 2]:
        for g in scales:
            Hg = dict(c["H"]); Hg.update(Kp=c["H"]["Kp"] * g,
                                         Ki=c["H"]["Ki"] * g,
                                         Kd=c["H"]["Kd"] * g, scale=g)
            f = _feasible(lambda Hg=Hg: _SelfStateWrapper(
                make_fadrc(eval_tip(), Hg), NX5, NX5), eval_tip)
            pl = eval_tip()
            ok, r = _stable_at(lambda: _SelfStateWrapper(
                make_fadrc(pl, Hg), NX5, NX5), pl, 0.15e-3, T=0.3, ret=True)
            rms15 = umax15 = None
            if ok and r is not None:
                ie = r['stop_idx']
                y = (pl.D_tip @ r['qm'][:, :ie + 1]) * 1e6
                rms15 = float(np.sqrt(np.mean(y[len(y) // 2:] ** 2)))
                umax15 = float(np.max(np.abs(r['u'][:ie + 1])))
            rows.append(dict(ap_lin_base=c["ap_lin_nominal_mm"], scale=g,
                             feasible_mm=f * 1e3, rms15_um=rms15,
                             umax15_V=umax15))
            print(f"  g={g:.2f}: feasible={f*1e3:.2f} mm rms15={rms15} "
                  f"[{time.time()-T0:.0f}s]", flush=True)
            if f > best["feasible"]:
                best = dict(feasible=f, H=Hg)
    out["gain_scale_rows"] = rows
    H = best["H"]
    out["scale_selected_edge"] = bool(H and H["scale"] in (scales[0], scales[-1]))

    # ---------- boundaries, drift, traces for the selected design ----------
    fad = make_fadrc(tip3, H)
    ap_nom = clf3f.ap_crit_dynamic(RPM, *fad.export_lti(), ap_max=6e-3) * 1e3
    ap_ref5 = clf5.ap_crit_dynamic(RPM, *fad.export_lti(), ap_max=6e-3) * 1e3
    drift = {}
    for scale, tag in [(0.8, "-20%"), (1.0, "nominal"), (1.2, "+20%")]:
        dtip = lambda s=scale: drift_plate(build(5, include_dynamics=True,
                                                 sensor="tip"), s)
        drift[tag] = _feasible(lambda: _SelfStateWrapper(
            make_fadrc(eval_tip(), H), NX5, NX5), dtip) * 1e3
    out["FADRC"] = dict(
        design={k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in H.items()},
        nz=fad.nz, ap_feasible_mm=best["feasible"] * 1e3,
        ap_crit_linear_nominal_mm=ap_nom,
        ap_crit_linear_refined_mm=ap_ref5,
        drift_feasible_mm=drift)
    print(f"  SELECTED FADRC: feasible={best['feasible']*1e3:.2f} "
          f"lin_nom={ap_nom:.2f} lin_ref={ap_ref5:.3f} "
          f"drift={ {k: round(v,2) for k,v in drift.items()} }", flush=True)

    traces = {}
    for ap_c in (0.15e-3, 0.5e-3):
        pl = eval_tip()
        ok, r = _stable_at(lambda: _SelfStateWrapper(
            make_fadrc(pl, H), NX5, NX5), pl, ap_c, T=0.3, ret=True)
        if r is None:
            continue
        ie = r['stop_idx']
        y = (pl.D_tip @ r['qm'][:, :ie + 1]) * 1e6
        key = f"FADRC_{ap_c*1e3:.2f}"
        traces[key] = dict(t=r['t'][:ie + 1][::8], y_um=y[::8],
                           u_V=r['u'][:ie + 1][::8], stable=bool(ok),
                           yrms_um=(float(np.sqrt(np.mean(y[len(y)//2:]**2)))
                                    if ok else None),
                           umax_V=float(np.max(np.abs(r['u'][:ie + 1]))))
        print(f"  {key}: stable={ok} yrms={traces[key]['yrms_um']} "
              f"umax={traces[key]['umax_V']:.1f}", flush=True)
    out["trace_summary"] = {k: dict(stable=v["stable"], yrms_um=v["yrms_um"],
                                    umax_V=v["umax_V"]) for k, v in traces.items()}
    np.savez(os.path.join(RESULTS, "fadrc_tip_traces.npz"),
             **{f"{k}_{f}": np.asarray(traces[k][f])
                for k in traces for f in ("t", "y_um", "u_V")})

    # ---------- baselines summary (from committed studies) ----------
    tipj = json.load(open(os.path.join(RESULTS, "fopid_tip.json")))
    hybj = json.load(open(os.path.join(RESULTS, "hybrid_tip.json")))
    v2j = json.load(open(os.path.join(RESULTS, "hybrid_v2.json")))
    out["baselines_feasible_mm"] = dict(
        OL_linear=tipj["OL"]["ap_crit_refined_mm"],
        LQG=tipj["LQG"]["ap_feasible_mm"],
        ADRC_plain=tipj["ADRC"]["best_feasible_mm"],
        FOPID_plain=tipj["FOPID"]["ap_feasible_mm"],
        HYBRID_v1=hybj["HYBRID"]["ap_feasible_mm"],
        HYBRID_v2R=v2j["designs"]["v2R"]["ap_feasible_mm"])
    out["de_budget"] = dict(maxiter=maxiter, popsize=popsize, tol=1e-2,
                            init="sobol", seed=0, b0_signs=[+1, -1], quick=quick)
    out["elapsed_s"] = time.time() - T0
    out["note"] = ("Chapter-2 ADRC-FOPID (fractional law ON THE ESO ESTIMATES) "
                   "developed for the plate: signed free b0_eff, free wo, "
                   "leakage, injection roll-off, all co-searched by DE on the "
                   "CL-SD spectral radius; gain scale selected by the "
                   "voltage-feasible metric. Deployed step() carries the "
                   "one-sample ESO u-channel convention (feasibility numbers "
                   "use step(), conservative); validation in the module "
                   "docstring and test session.")
    json.dump(out, open(os.path.join(RESULTS, "fadrc_tip.json"), "w"), indent=2)
    print(f"\nTOTAL {out['elapsed_s']:.0f}s -> results/fadrc_tip.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run(quick=args.quick)
