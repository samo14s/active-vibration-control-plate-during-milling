#!/usr/bin/env python3
"""Finish the campaign fast: stages 6 (Monte-Carlo) and 7 (rpm sweep)
parallelized across worker processes; then numbers + figures."""
from __future__ import annotations

import copy
import json
import sys as _sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"


def _mc_case(i: int) -> dict:
    from avcp import SystemParams, build_modal_model, simulate
    from avcp.metrics import summarize
    from run_campaign import make_controllers  # noqa: E402

    _sys.path.insert(0, str(ROOT / "scripts"))
    rng = np.random.default_rng(7)
    perts = []
    for _ in range(20):
        perts.append({
            "df": float(rng.uniform(-0.03, 0.03)),
            "dz": float(rng.uniform(-0.3, 0.3)),
            "dF": float(rng.uniform(-0.2, 0.2)),
            "runout": float(rng.uniform(0.0, 5e-6)),
            "delay2": bool(rng.random() < 0.3),
        })
    pert = perts[i]
    sys_p = SystemParams()
    model, _ = build_modal_model(sys_p)
    tuning = json.loads((RES / "tuning.json").read_text())
    T = sys_p.milling.pass_time + 1.0
    sp = copy.deepcopy(sys_p)
    sp.milling.runout = pert["runout"]
    if pert["delay2"]:
        sp.control.delay_steps = 2
    pm = copy.deepcopy(model)
    pm.freqs = model.freqs * (1.0 + pert["df"])
    pm.zeta = model.zeta * (1.0 + pert["dz"])
    row = {"pert": pert}
    r = simulate(sp, pm, controller=None, T=T, seed=100 + i,
                 force_scale=1.0 + pert["dF"])
    row["open"] = summarize(r, sp.milling.pass_time)["rms_ac_wc"]
    for name in ("cpd", "vpd", "pshlqg"):
        ctl = make_controllers(sys_p, model, tuning)[name]
        rc = simulate(sp, pm, controller=ctl, T=T, seed=100 + i,
                      force_scale=1.0 + pert["dF"])
        s = summarize(rc, sp.milling.pass_time)
        row[name] = s["rms_ac_wc"]
        row[name + "_sat"] = s["sat_frac"]
    return row


def _build_psh_for_speed(sp, model, r_u0, f_dist):
    """Per-speed design with a regenerative-margin gate: the closed loop
    must not degrade the chatter limit at that speed's regenerative
    delay; if it does, feedback authority is reduced and the design
    re-verified."""
    from avcp import PshlqgController, PshlqgConfig, MillingForce
    from avcp.stability import closed_loop_frf_nn
    mf = MillingForce(sp.milling)
    r_u = r_u0
    for _ in range(4):
        ctl = PshlqgController(sp, model, PshlqgConfig(r_u=r_u),
                               f_dist=f_dist)
        if not ctl.spillover_mask(model):
            r_u *= 10.0
            continue
        ok = True
        for xp in (0.012, 0.06, 0.108):
            a_ol = mf.a_crit_nyquist(
                lambda w: closed_loop_frf_nn(sp, model, None, xp, w),
                sp.milling.rpm, nf=12000)
            a_cl = mf.a_crit_nyquist(
                lambda w: closed_loop_frf_nn(sp, model, ctl, xp, w),
                sp.milling.rpm, nf=12000)
            if a_cl < 0.9 * a_ol:
                ok = False
                break
        if ok:
            return ctl
        r_u *= 10.0
    return ctl


def _rpm_case(args) -> tuple[str, dict]:
    rpm, = args
    from avcp import SystemParams, build_modal_model, simulate
    from avcp.metrics import summarize

    sys_p = SystemParams()
    model, _ = build_modal_model(sys_p)
    tuning = json.loads((RES / "tuning.json").read_text())
    sp = copy.deepcopy(sys_p)
    sp.milling.rpm = rpm
    T = sp.milling.pass_time + 1.0
    r_open = simulate(sp, model, controller=None, T=T, seed=42)
    ctl_a = _build_psh_for_speed(sp, model, tuning["r_u"],
                                 f_dist=sp.milling.f_tpf)
    r_a = simulate(sp, model, controller=ctl_a, T=T, seed=42)
    ctl_d = _build_psh_for_speed(sp, model, tuning["r_u"],
                                 f_dist=466.6667)
    r_d = simulate(sp, model, controller=ctl_d, T=T, seed=42)
    out = {
        "open": summarize(r_open, sp.milling.pass_time)["rms_ac_wc"],
        "adaptive": summarize(r_a, sp.milling.pass_time)["rms_ac_wc"],
        "detuned": summarize(r_d, sp.milling.pass_time)["rms_ac_wc"],
    }
    return str(int(rpm)), out


def main():
    _sys.path.insert(0, str(ROOT / "scripts"))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        mc_rows = list(ex.map(_mc_case, range(20)))
    (RES / "robust.json").write_text(json.dumps(mc_rows, indent=1))
    for i, r in enumerate(mc_rows):
        print(f"mc{i}: open={r['open']:.2e} cpd={r['cpd']:.2e} "
              f"vpd={r['vpd']:.2e} pshlqg={r['pshlqg']:.2e}", flush=True)
    print(f"MC done in {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=3) as ex:
        rpm_out = dict(ex.map(_rpm_case, [(6000.0,), (7000.0,), (8000.0,)]))
    (RES / "rpm_sweep.json").write_text(json.dumps(rpm_out, indent=1))
    print(json.dumps(rpm_out, indent=1), flush=True)
    print(f"rpm sweep done in {time.time()-t0:.0f}s", flush=True)
    print("campaign complete", flush=True)


if __name__ == "__main__":
    main()
