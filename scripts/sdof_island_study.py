"""SDOF (Ozsoy-class) saturation-island study under the
matched-authority protocol.

Parameter provenance: see avc/sdof.py header — this runs on the
clearly-labeled class-representative stand-in unless OZSOY_PUBLISHED
is filled from the OA PDF (network policy blocks it in this session);
all results are labeled accordingly in the JSON.

Protocol:
  1. Linear boundaries: open-loop (g_v = 0) and DVF closed-loop
     critical depths over the spindle-speed grid (Floquet radius of
     the sampled lift).
  2. Census at each speed: zone string over ap (certified at h_req /
     linear-only / forced-saturated / unstable) + forced-orbit
     UTILIZATION U = max|F*| / F_max — the matched-authority metric
     shared with the plate campaign (where U = max|v*| / V_max).
  3. Island hunt at linear-only points: signed surface-step ladder,
     causal control with the force bound lifted — identical protocol
     to scripts/satcert_islands.py.
  4. Matched-authority comparison: island occurrence vs U, side by
     side with the plate campaign's island points (U ~ 0.91 and
     ~0.999 at PS-LPV x=50 mm).
  5. Saturated-regime margins (circle criterion) at the island points.

Output: results/sdof_island_study.json
"""
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from avc.satcert import certify_sampled, saturated_margins  # noqa: E402
from avc.sdof import (OZSOY_PUBLISHED, REPRESENTATIVE, SDOFConfig,  # noqa: E402
                      lift_sdof, simulate_sdof, tooth_period)
from dataclasses import replace  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "sdof_island_study.json"
H_REQ = 10e-6
RPMS = (2000.0, 2500.0, 3000.0, 3500.0, 4000.0)
SETTLE, OBSERVE = 120, 150


def log(msg):
    print(f"[sdof {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def crit_depth(cfg, rpm, ap_hi, tol_rel=5e-3):
    lo, hi = 0.0, ap_hi
    if lift_sdof(cfg, hi, rpm).rho() < 1.0:
        return hi
    while hi - lo > tol_rel * ap_hi:
        mid = 0.5 * (lo + hi)
        if lift_sdof(cfg, mid, rpm).rho() < 1.0:
            lo = mid
        else:
            hi = mid
    return lo


def utilization(lift, bound):
    return float(np.max(np.abs(lift.forced_voltages()))) / bound


def zone(cfg, ap, rpm):
    lift = lift_sdof(cfg, ap, rpm)
    c = certify_sampled(lift, cfg.f_max)
    u = utilization(lift, cfg.f_max)
    if c.status == "lin-unstable":
        return "U", u, c
    if c.status == "forced-saturated":
        return "S", u, c
    return ("C" if c.h_max >= H_REQ else "L"), u, c


def run(cfg, ap, rpm, h, f_max):
    tau = tooth_period(cfg, rpm)
    r = simulate_sdof(cfg, ap, rpm, T=(SETTLE + OBSERVE) * tau, fs=50e3,
                      f_max=f_max, loss_of_contact=True,
                      surf_step=(SETTLE * tau, h))
    t_inj = SETTLE * tau
    pre = (r["t"] >= t_inj - 20 * tau) & (r["t"] < t_inj)
    post = r["t"] >= r["t"][-1] - 20 * tau
    rp = float(np.sqrt(np.mean(r["x"][pre] ** 2))) if pre.any() else 0.0
    rq = float(np.sqrt(np.mean(r["x"][post] ** 2))) if post.any() else 0.0
    return {"ratio": rq / max(rp, 1e-15), "clip_frac": r["clip_frac"],
            "diverged": r["diverged"]}


def classify(res):
    if res["diverged"] is not None:
        return "diverged"
    return "chatter" if res["ratio"] > 3.0 else "decays"


def main():
    cfg = OZSOY_PUBLISHED if OZSOY_PUBLISHED is not None else REPRESENTATIVE
    out = {"config": cfg.label, "h_req_um": H_REQ * 1e6,
           "provenance": ("published parameter set" if OZSOY_PUBLISHED
                          else "class-representative stand-in — NOT the "
                          "published values (OA sources blocked by the "
                          "session network policy; see avc/sdof.py)")}

    # 1. linear boundaries
    cfg_ol = replace(cfg, g_v=0.0)
    ap_hi0 = 5e-3
    bounds = {}
    for rpm in RPMS:
        a_ol = crit_depth(cfg_ol, rpm, ap_hi0)
        a_cl = crit_depth(cfg, rpm, ap_hi0)
        bounds[f"{rpm:g}"] = {"ol_mm": a_ol * 1e3, "dvf_mm": a_cl * 1e3,
                              "dvf_capped": a_cl >= ap_hi0 - 1e-9}
        log(f"rpm={rpm:g}: OL crit {a_ol*1e3:.3f} mm, DVF crit "
            f"{a_cl*1e3:.3f} mm{' (cap)' if a_cl >= ap_hi0-1e-9 else ''}")
    out["linear_boundaries"] = bounds

    # 2. census + utilization: the EXPOSURE BAND between the open-loop
    # boundary (below which even a dead actuator is stable) and the DVF
    # boundary — for the DVF class the forced utilization stays low, so
    # islands are TRANSIENT-driven and live at large perturbations even
    # inside cells certified at the (small) h_req
    out["census"] = {}
    island_candidates = []
    for rpm in RPMS:
        a_ol = bounds[f"{rpm:g}"]["ol_mm"] * 1e-3
        a_cl = bounds[f"{rpm:g}"]["dvf_mm"] * 1e-3
        aps = np.linspace(0.5 * a_ol, min(a_cl, 12 * a_ol), 13)
        zs, us, hmaxes = [], [], []
        for ap in aps:
            z, u, c = zone(cfg, float(ap), rpm)
            zs.append(z)
            us.append(round(u, 3))
            hmaxes.append(c.h_max * 1e6 if c.feasible else None)
        out["census"][f"{rpm:g}"] = {
            "ap_mm": (aps * 1e3).round(4).tolist(),
            "zones": "".join(zs), "utilization": us,
            "cert_h_um": hmaxes}
        log(f"rpm={rpm:g}: {''.join(zs)}  h_um={[None if h is None else round(h,1) for h in hmaxes]}")
        for ap, z, u, hm in zip(aps, zs, us, hmaxes):
            if z in "CL" and ap > a_ol:
                island_candidates.append((rpm, float(ap), u,
                                          hm if hm else 1e9))

    # 3. island hunt: linearly-stable cells ABOVE the open-loop
    # boundary, closest to the clip onset first; absolute step ladder
    island_candidates.sort(key=lambda c: c[3])
    island_candidates = [(r, a, u) for r, a, u, _ in island_candidates]
    out["islands"] = {}
    for rpm, ap, u in island_candidates[:4]:
        key = f"rpm={rpm:g}|ap={ap*1e3:.3f}mm|U={u:.3f}"
        lift = lift_sdof(cfg, ap, rpm)
        cert = certify_sampled(lift, cfg.f_max)
        rec = {"cert_h_um": cert.h_max * 1e6, "rho": cert.lin_rho,
               "u_bar_N": cert.u_bar, "utilization": u, "runs": {}}
        found = None
        done = False
        for habs in (25e-6, 50e-6, 100e-6, 200e-6, 350e-6, 500e-6):
            for sign in (-1.0, +1.0):
                h = sign * habs
                r_on = run(cfg, ap, rpm, h, cfg.f_max)
                v = classify(r_on)
                rec["runs"][f"{sign*habs*1e6:+g}um"] = {**r_on,
                                                       "verdict": v}
                log(f"  {key} h={h*1e6:+8.2f}um: {v} "
                    f"(clip {r_on['clip_frac']:.3f})")
                if v != "decays":
                    r_off = run(cfg, ap, rpm, h, 1e12)
                    v_off = classify(r_off)
                    rec["runs"]["control_nosat"] = {**r_off,
                                                    "verdict": v_off}
                    rec["island_h_um"] = h * 1e6
                    rec["verdict"] = (
                        "ISLAND CONFIRMED (saturation-induced)"
                        if v_off == "decays" else
                        f"not saturation-specific ({v_off} unbounded)")
                    log(f"  {key}: bound lifted -> {v_off}")
                    found = h
                    done = True
                    break
            if done or abs(h) >= 500e-6:
                break
        if found is None:
            rec["verdict"] = "no island found in tested range"
        mar = saturated_margins(lift, n_grid=512)
        rec["circle_max"] = mar["circle_max"]
        rec["gamma_2"] = mar["gamma_2"]
        out["islands"][key] = rec
        log(f"  {key}: {rec['verdict']} (circle {mar['circle_max']:.2f})")

    # 4. matched-authority protocol vs the plate campaign: two
    # rig-agnostic metrics. U (forced utilization) drives islands on
    # forced-dominated rigs (plate H-inf: U ~ 0.91/0.999 at its two
    # islands); D (unclipped transient demand at the island step,
    # divided by the bound) is the like-for-like clipping depth for
    # transient-dominated rigs (SDOF/DVF). D_plate at its islands
    # (computed from the plate lifts, same formula as dd below):
    # (v_pk_per_m * |h| + forced peak) / V_max = 2.80 and 4.14.
    dd = {}
    for key, rec in out["islands"].items():
        if rec.get("island_h_um"):
            lift = None
            for (rpm, ap, u) in island_candidates[:4]:
                if f"rpm={rpm:g}|ap={ap*1e3:.3f}mm" in key:
                    lift = lift_sdof(cfg, ap, rpm)
                    break
            if lift is not None:
                c = certify_sampled(lift, cfg.f_max)
                vpk = c.meta.get("v_peak_per_m", 0.0)
                fpk = c.meta.get("forced_peak", 0.0)
                dd[key] = (vpk * abs(rec["island_h_um"]) * 1e-6 + fpk) \
                    / cfg.f_max
    out["matched_authority"] = {
        "U_metric": "max|forced actuation| / bound",
        "D_metric": "unclipped transient demand at island step / bound",
        "plate_islands": {"PS-LPV x=50mm ap=1.5mm": {"U": 0.906,
                                                     "D": 2.80},
                          "PS-LPV x=50mm ap=2.0mm": {"U": 0.999,
                                                     "D": 4.14}},
        "sdof_islands_D": dd}

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    log(f"written {OUT}")


if __name__ == "__main__":
    main()
