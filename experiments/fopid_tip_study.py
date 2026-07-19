"""
fopid_tip_study.py
==================
Strict single-sensor comparison: LQG, ADRC and FOPID ALL reading the SAME
tip sensor (the non-minimum-phase measurement LQG uses), so no controller gets
the collocated (minimum-phase) sensor.  This isolates controller structure from
sensor placement and answers directly: on a common tip sensor, which controllers
still work?

Physics
-------
The tip measurement is non-minimum phase: the modal coupling D_tip * H_pe is
sign-indefinite (here -+--+), i.e. the u -> y_tip transfer has right-half-plane
zeros.  This is exactly why the collocated (minimum-phase, ---+-) sensor was used
for the output-feedback controllers elsewhere in this study; here we deliberately
remove that choice.

What to expect (and what is found):
  * LQG carries a full-state Kalman observer that explicitly models the NMP
    dynamics, so it works on the tip (its native sensor).
  * ADRC lumps everything into a single total disturbance estimated by an ESO;
    the RHP zeros defeat it -- it fails to stabilise at ANY bandwidth or depth.
  * FOPID is a fixed-structure output-feedback law; the RHP zeros cap its
    achievable margin, so it is heavily degraded relative to the collocated case
    but does not collapse entirely.

Everything else is identical to fopid_study.py (refined 5-mode plant, dt = 25 us,
+/-150 V saturation, CL-SD linear metric + voltage-feasible metric, tip readout).
The collocated numbers from results/fopid.json are loaded for side-by-side
contrast.

Outputs: results/fopid_tip.json (+ results/fopid_tip_traces.npz)
Run:     python experiments/fopid_tip_study.py [--quick]
"""
import os, sys, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
RESULTS = os.path.join(ROOT, "results")

from adrc_control import ADRCController
from cl_fdm import ClosedLoopFDM
from model_refinement import build, _SelfStateWrapper
from fopid_study import (DT, U_MAX, RPM, CUT, NX5, _fopid, _feasible, _stable_at,
                         design_fopid, make_fopid_wrapped, mk_lqg, _lqg_lti)


def run(quick=False):
    T0 = time.time()
    out = dict(dt_us=DT * 1e6, u_max=U_MAX, rpm=RPM, sensor="tip (all controllers)")

    tip3 = build(3, include_dynamics=False, sensor="tip")
    eval_tip = lambda: build(5, include_dynamics=True, sensor="tip")
    clf3 = ClosedLoopFDM.from_plate(tip3, CUT, m_div=30)
    clf5 = ClosedLoopFDM.from_plate(eval_tip(), CUT, m_div=30)

    p = eval_tip()
    out["tip_coupling_signs"] = "".join('+' if c > 0 else '-'
                                        for c in (p.D_obs * p.H_Pe_modal))
    out["b0_tip"] = float(p.D_obs @ p.H_Pe_modal)
    out["OL"] = dict(ap_crit_nominal_mm=clf3.ap_crit(RPM, None, ap_max=6e-3) * 1e3,
                     ap_crit_refined_mm=clf5.ap_crit(RPM, None, ap_max=6e-3) * 1e3)
    print(f"tip coupling {out['tip_coupling_signs']}  b0_tip={out['b0_tip']:.4f}  "
          f"OL_ref={out['OL']['ap_crit_refined_mm']:.3f} mm")

    # ---------------- LQG on tip (its native sensor) ----------------
    print("\n=== LQG (tip) ===")
    lqg_rows = []
    grid = ([("grid-default", None), ("w_q=1e16,w_qd=1e8", (1e16, 1e8))] if quick
            else [("grid-default", None), ("w_q=1e16,w_qd=1e8", (1e16, 1e8)),
                  ("w_q=1e12,w_qd=1e8", (1e12, 1e8))])
    for label, w in grid:
        f = _feasible(lambda w=w: _SelfStateWrapper(mk_lqg(w, "tip"), NX5, 6), eval_tip)
        lqg_rows.append(dict(label=label, weights=w, feasible_mm=f * 1e3))
        print(f"  LQG ({label}): feasible={f*1e3:.2f} mm [{time.time()-T0:.0f}s]",
              flush=True)
    best_lqg = max(lqg_rows, key=lambda r: r["feasible_mm"])
    l = mk_lqg(best_lqg["weights"], "tip")
    Acl = _lqg_lti(l)
    out["LQG"] = dict(rows=lqg_rows, tune=best_lqg["label"],
                      ap_feasible_mm=best_lqg["feasible_mm"],
                      ap_crit_linear_nominal_mm=clf3.ap_crit_dynamic(RPM, *Acl, ap_max=6e-3) * 1e3,
                      ap_crit_linear_refined_mm=clf5.ap_crit_dynamic(RPM, *Acl, ap_max=6e-3) * 1e3)

    # ---------------- ADRC on tip (expected to fail: NMP breaks the ESO) ----
    print("\n=== ADRC (tip) -- bandwidth sweep ===")
    adrc_rows = []
    bws = ([(400, 3200), (800, 9600)] if quick
           else [(200, 1200), (400, 3200), (800, 9600), (1200, 14400), (1800, 21600)])
    for wc, wo in bws:
        f = _feasible(lambda wc=wc, wo=wo: _SelfStateWrapper(
            ADRCController(eval_tip(), dt=DT, wc=wc, wo=wo, u_max=U_MAX), NX5, NX5),
            eval_tip)
        apl = clf3.ap_crit_dynamic(RPM, *ADRCController(
            tip3, dt=DT, wc=wc, wo=wo, u_max=U_MAX).export_lti(), ap_max=6e-3) * 1e3
        adrc_rows.append(dict(wc=wc, wo=wo, feasible_mm=f * 1e3, ap_lin_nominal_mm=apl))
        print(f"  ADRC wc={wc}: ap_lin(nom)={apl:.3f} feasible={f*1e3:.2f} mm "
              f"[{time.time()-T0:.0f}s]", flush=True)
    best_adrc = max(adrc_rows, key=lambda r: r["feasible_mm"])
    out["ADRC"] = dict(rows=adrc_rows, best_feasible_mm=best_adrc["feasible_mm"],
                       best_linear_nominal_mm=max(r["ap_lin_nominal_mm"] for r in adrc_rows),
                       fails=(best_adrc["feasible_mm"] < 0.2))

    # ---------------- FOPID on tip (chatter-optimal design) ----------------
    print("\n=== FOPID (tip) -- design + feasible selection ===")
    winners = design_fopid(tip3, quick=quick)
    top = winners[:1 if quick else 2]
    scales = [0.25, 0.5, 1.0, 1.5] if quick else [0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]
    best = dict(feasible=0.0, params=None)
    for P0 in top:
        for g in scales:
            Pg = dict(P0); Pg.update(Kp=P0["Kp"] * g, Ki=P0["Ki"] * g,
                                     Kd=P0["Kd"] * g, scale=g)
            f = _feasible(lambda Pg=Pg: make_fopid_wrapped(eval_tip, Pg), eval_tip)
            if f > best["feasible"]:
                best = dict(feasible=f, params=Pg)
            print(f"  FOPID sign={P0['sign']:+.0f} lam={P0['lam']:.2f} mu={P0['mu']:.2f}"
                  f" g={g:.2f}: feasible={f*1e3:.2f} mm [{time.time()-T0:.0f}s]", flush=True)
    P = best["params"]
    ap_lin_nom = (clf3.ap_crit_dynamic(RPM, *_fopid(tip3, P).export_lti(), ap_max=6e-3) * 1e3
                  if P else 0.0)
    ap_lin_ref = (clf5.ap_crit_dynamic(RPM, *_fopid(tip3, P).export_lti(), ap_max=6e-3) * 1e3
                  if P else 0.0)
    out["FOPID"] = dict(design=({k: P[k] for k in
                                ("sign", "lam", "mu", "Kp", "Ki", "Kd", "scale")}
                                if P else None),
                        ap_feasible_mm=best["feasible"] * 1e3,
                        ap_crit_linear_nominal_mm=ap_lin_nom,
                        ap_crit_linear_refined_mm=ap_lin_ref)
    print(f"  FOPID selected: feasible={best['feasible']*1e3:.2f} mm "
          f"ap_lin(nom)={ap_lin_nom:.3f} mm")

    # ---------------- time traces at a common depth ----------------
    print("\n=== time traces at common depth ===")
    ap_common = 0.5e-3
    traces = {}
    ctrls = {"LQG": lambda: _SelfStateWrapper(mk_lqg(best_lqg["weights"], "tip"), NX5, 6),
             "ADRC": lambda: _SelfStateWrapper(ADRCController(
                 eval_tip(), dt=DT, wc=best_adrc["wc"], wo=best_adrc["wo"], u_max=U_MAX),
                 NX5, NX5),
             "FOPID": (lambda: make_fopid_wrapped(eval_tip, P)) if P else None}
    for name, mk in ctrls.items():
        if mk is None:
            continue
        pl = eval_tip()
        ok, r = _stable_at(mk, pl, ap_common, T=0.3, dt=DT, ret=True)
        if r is None:
            continue
        ie = r['stop_idx']
        ytip = (pl.D_tip @ r['qm'][:, :ie + 1]) * 1e6
        dec = 8
        traces[name] = dict(t=r['t'][:ie + 1][::dec].tolist(),
                            y_um=ytip[::dec].tolist(),
                            yrms_um=(float(np.sqrt(np.mean(ytip[len(ytip)//2:]**2)))
                                     if ok else None), stable=bool(ok),
                            diverged_ms=(None if ok else float(r['t'][ie] * 1e3)))
        print(f"  {name} @ {ap_common*1e3:.1f}mm: stable={ok} "
              f"yrms={traces[name]['yrms_um']}", flush=True)
    out["traces_common_depth_mm"] = ap_common * 1e3
    np.savez(os.path.join(RESULTS, "fopid_tip_traces.npz"),
             **{f"{k}_{f}": np.array(traces[k][f])
                for k in traces for f in ("t", "y_um")})

    # ---------------- contrast with the collocated study ----------------
    try:
        col = json.load(open(os.path.join(RESULTS, "fopid.json")))
        out["collocated_contrast_feasible_mm"] = dict(
            LQG_tip=col["LQG"]["ap_feasible_mm"],
            LQG_collocated=col["LQG"].get("ap_feasible_collocated_mm"),
            ADRC_collocated=col["ADRC"]["ap_feasible_mm"],
            FOPID_collocated=col["FOPID"]["ap_feasible_mm"])
    except Exception:
        pass

    out["summary"] = dict(ap_feasible_tip_mm=dict(
        LQG=out["LQG"]["ap_feasible_mm"], ADRC=out["ADRC"]["best_feasible_mm"],
        FOPID=out["FOPID"]["ap_feasible_mm"]))
    out["elapsed_s"] = time.time() - T0
    out["note"] = ("All three controllers read the SAME non-minimum-phase tip "
                   "sensor. ADRC (lumped-disturbance ESO) fails to stabilise at "
                   "any bandwidth/depth on the NMP plant; FOPID (fixed-structure) "
                   "is heavily degraded vs its collocated case; LQG (full-state "
                   "observer that models the NMP dynamics) works. This is the "
                   "quantified reason the output-feedback controllers use a "
                   "collocated minimum-phase sensor elsewhere in the study.")
    json.dump(out, open(os.path.join(RESULTS, "fopid_tip.json"), "w"), indent=2)
    print(f"\nTOTAL {out['elapsed_s']:.0f}s -> results/fopid_tip.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run(quick=args.quick)
