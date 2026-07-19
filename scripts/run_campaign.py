#!/usr/bin/env python3
"""Full simulation campaign for the manuscript.

Stages (each saves its products under results/ as .npz/.json so the run
can resume after interruption and figures can be rebuilt at will):

  1. model      — FEM validation data: frequencies, mode shapes, static
                  piezo deflection, milling-force spectrum, material-
                  removal frequency drift
  2. tuning     — CPD, VPD (31 points, CPD-seeded), PSH-LQG (auto r_u)
  3. passes     — full 11-s milling passes: open / CPD / VPD / PSH-LQG,
                  plus PSH-LQG with the material-removal frequency drift
  4. frozen     — frozen closed-loop spectral radius along the path
  5. sld        — stability lobes (Nyquist a_crit sweep) at points 1/5/9
  6. robust     — Monte-Carlo plant perturbations + runout + delay
  7. rpm        — spindle-speed sweep with internal-model adaptation
"""
from __future__ import annotations

import copy
import json
import pickle
import sys as _sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(ROOT / "src"))

from avcp import (SystemParams, PlateFem, build_modal_model,
                  removal_thickness_map, simulate, MillingForce,
                  PDController, PshlqgController, PshlqgConfig,
                  tune_cpd, tune_vpd, tune_r_u)
from avcp.controllers import force_harmonics
from avcp.metrics import summarize, along_path_rms, uniformity
from avcp.stability import frozen_stability_map, closed_loop_frf_nn

RES = ROOT / "results"
RES.mkdir(exist_ok=True)

MONITOR = np.linspace(0.0, 0.12, 11)
POINTS_159 = (0.012, 0.060, 0.108)          # paper's points 1, 5, 9


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_result(res, name):
    np.savez_compressed(RES / f"pass_{name}.npz", t=res.t, x_cut=res.x_cut,
                        w_cut=res.w_cut, w_points=res.w_points, acc=res.acc,
                        V=res.V, F=res.F, surf_t=res.surf_t,
                        surf_x=res.surf_x, surf_w=res.surf_w)


# ----------------------------------------------------------------- stage 1

def stage_model(sys_p, model, fem):
    log("stage 1: model validation data")
    p = sys_p.plate
    lam2 = (2 * np.pi * model.freqs * p.Lx ** 2
            * np.sqrt(p.rho * p.h / p.D))
    # static deflection vs voltage along the top edge
    q = sys_p.piezo.moment_constant(p.h)
    Ba = fem.piezo_vector(sys_p.piezo.u1, sys_p.piezo.u2,
                          sys_p.piezo.v1, sys_p.piezo.v2, q)
    xs = np.linspace(0.0, 0.12, 61)
    stat = {}
    for V in (500.0, 900.0, 1200.0):
        u = fem.static_solve(Ba * V)
        stat[str(int(V))] = [float(fem.point_row_free(x, 0.1199) @ u)
                             for x in xs]
    # mode shapes on a coarse grid for contour plots
    gx = np.linspace(0.0, 0.12, 31)
    gy = np.linspace(0.0, 0.12, 31)
    freqs, Phi = fem.modes(6)
    shapes = np.zeros((6, len(gy), len(gx)))
    for k in range(6):
        for iy, y in enumerate(gy):
            if y == 0.0:
                continue
            for ix, x in enumerate(gx):
                shapes[k, iy, ix] = float(
                    fem.point_row_free(min(x, 0.11999), min(y, 0.11999))
                    @ Phi[:, k])
    # force spectrum
    mil = sys_p.milling
    mf = MillingForce(mil)
    tt = np.linspace(0.0, 4 * mil.tau, 4096, endpoint=False)
    Fser = mf.normal_force_series(tt)
    w_h, F_h = force_harmonics(sys_p, n_harm=8)
    # material removal frequency drift
    tm = removal_thickness_map(sys_p, x_cut=0.12)
    fem_end = PlateFem(sys_p.plate, sys_p.config.nx, sys_p.config.ny, tm)
    freqs_end, _ = fem_end.modes(sys_p.config.n_modes_eval)
    np.savez_compressed(
        RES / "model.npz", freqs=model.freqs, lam2=lam2, ba=model.ba,
        phi_acc=model.phi_acc, path_x=model.path_x, phi_path=model.phi_path,
        stat_x=xs, stat_500=stat["500"], stat_900=stat["900"],
        stat_1200=stat["1200"], shapes=shapes, shapes_x=gx, shapes_y=gy,
        force_t=tt, force_F=Fser, harm_w=w_h, harm_F=np.abs(F_h),
        freqs_end=freqs_end)
    log(f"  freqs = {np.round(model.freqs[:8], 1)}")
    log(f"  removal drift = {np.round((freqs_end / model.freqs - 1) * 100, 3)} %")
    return freqs_end


# ----------------------------------------------------------------- stage 2

def stage_tuning(sys_p, model):
    log("stage 2: controller tuning")
    t0 = time.time()
    kp_c, kd_c = tune_cpd(sys_p, model)
    log(f"  CPD kp={kp_c:.4f} kd={kd_c:.4f} ({time.time()-t0:.0f}s)")
    t0 = time.time()
    xg, kp_v, kd_v = tune_vpd(sys_p, model, n_points=31,
                              cpd_seed=(kp_c, kd_c))
    log(f"  VPD tuned ({time.time()-t0:.0f}s); kp range "
        f"[{kp_v.min():.3f}, {kp_v.max():.3f}]")
    t0 = time.time()
    r_u = tune_r_u(sys_p, model)
    log(f"  PSH-LQG r_u={r_u:.3e} ({time.time()-t0:.0f}s)")
    tuning = {"kp_c": float(kp_c), "kd_c": float(kd_c),
              "xg": xg.tolist(), "kp_v": kp_v.tolist(),
              "kd_v": kd_v.tolist(), "r_u": float(r_u)}
    (RES / "tuning.json").write_text(json.dumps(tuning, indent=1))
    return tuning


def make_controllers(sys_p, model, tuning, f_dist=None):
    cpd = PDController(tuning["kp_c"], tuning["kd_c"])
    vpd = PDController(np.array(tuning["kp_v"]), np.array(tuning["kd_v"]),
                       x_ref=np.array(tuning["xg"]))
    cfg = PshlqgConfig(r_u=tuning["r_u"])
    psh = PshlqgController(sys_p, model, cfg, f_dist=f_dist)
    psh.spillover_mask(model)
    return {"cpd": cpd, "vpd": vpd, "pshlqg": psh}


# ----------------------------------------------------------------- stage 3

def stage_passes(sys_p, model, tuning, freqs_end):
    log("stage 3: full milling passes")
    T = sys_p.milling.pass_time + 1.0
    metrics = {}
    ctls = make_controllers(sys_p, model, tuning)
    for name, ctl in [("open", None)] + list(ctls.items()):
        t0 = time.time()
        res = simulate(sys_p, model, controller=ctl, T=T,
                       monitor_points=MONITOR, seed=42)
        save_result(res, name)
        s = summarize(res, sys_p.milling.pass_time)
        cx, rms = along_path_rms(res, sys_p.milling.pass_time)
        s["uniformity"] = uniformity(cx, rms)
        metrics[name] = s
        log(f"  {name:7s}: rms_ac={s['rms_ac_wc']:.3e} max={s['max_wc']:.3e} "
            f"unif={s['uniformity']:.2f} Vrms={s['v_rms']:.0f} "
            f"sat={s['sat_frac']*100:.1f}% ({time.time()-t0:.0f}s)")
    # PSH-LQG against the material-removal drifting plant
    scale_end = freqs_end[:model.n] / model.freqs
    ctl = make_controllers(sys_p, model, tuning)["pshlqg"]
    res = simulate(sys_p, model, controller=ctl, T=T,
                   monitor_points=MONITOR, seed=42,
                   freq_scale_end=scale_end)
    save_result(res, "pshlqg_removal")
    s = summarize(res, sys_p.milling.pass_time)
    metrics["pshlqg_removal"] = s
    log(f"  pshlqg+removal: rms_ac={s['rms_ac_wc']:.3e}")
    (RES / "metrics_passes.json").write_text(json.dumps(metrics, indent=1))
    return metrics


# ----------------------------------------------------------------- stage 4

def stage_frozen(sys_p, model, tuning):
    log("stage 4: frozen stability maps")
    xs = np.linspace(0.0, 0.12, 41)
    out = {"x": xs.tolist()}
    ctls = make_controllers(sys_p, model, tuning)
    for name, ctl in ctls.items():
        out[name] = frozen_stability_map(sys_p, model, ctl, xs).tolist()
        log(f"  {name}: max = {max(out[name]):.5f}")
    (RES / "frozen.json").write_text(json.dumps(out, indent=1))


# ----------------------------------------------------------------- stage 5

def stage_sld(sys_p, model, tuning):
    log("stage 5: stability lobes (closed-loop Nyquist a_crit sweep)")
    mf = MillingForce(sys_p.milling)
    rpms = np.linspace(5000.0, 12000.0, 141)
    f_dense = np.linspace(30.0, 3000.0, 30000)
    w_dense = 2 * np.pi * f_dense
    ctls = make_controllers(sys_p, model, tuning)
    out = {"rpm": rpms.tolist()}
    sens = {"f": f_dense}
    for name, ctl in [("open", None)] + list(ctls.items()):
        out[name] = {}
        for xp in POINTS_159:
            H = closed_loop_frf_nn(sys_p, model, ctl, xp, w_dense)
            sens[f"{name}_{xp*1e3:.0f}"] = np.abs(H)
            frf = lambda w: np.interp(w, w_dense, H.real) \
                + 1j * np.interp(w, w_dense, H.imag)
            a = mf.a_crit_sweep(frf, rpms, nf=30000)
            out[name][f"{xp*1e3:.0f}"] = np.where(
                np.isfinite(a), a, 1.0).tolist()
        log(f"  {name}: a_crit@7000rpm x=60mm = "
            f"{out[name]['60'][np.argmin(np.abs(rpms-7000))]*1e3:.2f} mm")
    (RES / "sld.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(RES / "frf_sens.npz", **sens)


# ----------------------------------------------------------------- stage 6

def stage_robust(sys_p, model, tuning):
    log("stage 6: Monte-Carlo robustness")
    rng = np.random.default_rng(7)
    T = sys_p.milling.pass_time + 1.0
    rows = []
    ctl_specs = ["cpd", "pshlqg"]
    for i in range(12):
        pert = {
            "df": float(rng.uniform(-0.03, 0.03)),
            "dz": float(rng.uniform(-0.3, 0.3)),
            "dF": float(rng.uniform(-0.2, 0.2)),
            "runout": float(rng.uniform(0.0, 5e-6)),
            "delay2": bool(rng.random() < 0.3),
        }
        sp = copy.deepcopy(sys_p)
        sp.milling.runout = pert["runout"]
        if pert["delay2"]:
            sp.control.delay_steps = 2
        # perturbed evaluation plant
        pm = copy.deepcopy(model)
        pm.freqs = model.freqs * (1.0 + pert["df"])
        pm.zeta = model.zeta * (1.0 + pert["dz"])
        row = {"pert": pert}
        r_open = simulate(sp, pm, controller=None, T=T, seed=100 + i,
                          force_scale=1.0 + pert["dF"])
        row["open"] = summarize(r_open, sp.milling.pass_time)["rms_ac_wc"]
        for name in ctl_specs:
            ctl = make_controllers(sys_p, model, tuning)[name]
            r = simulate(sp, pm, controller=ctl, T=T, seed=100 + i,
                         force_scale=1.0 + pert["dF"])
            s = summarize(r, sp.milling.pass_time)
            row[name] = s["rms_ac_wc"]
            row[name + "_sat"] = s["sat_frac"]
        rows.append(row)
        log(f"  mc{i}: open={row['open']:.2e} cpd={row['cpd']:.2e} "
            f"pshlqg={row['pshlqg']:.2e} (df={pert['df']:+.3f} "
            f"dz={pert['dz']:+.2f} dF={pert['dF']:+.2f} "
            f"ro={pert['runout']*1e6:.1f}um d2={pert['delay2']})")
    (RES / "robust.json").write_text(json.dumps(rows, indent=1))


# ----------------------------------------------------------------- stage 7

def stage_rpm(sys_p, model, tuning):
    log("stage 7: spindle-speed sweep (internal-model adaptation)")
    out = {}
    for rpm in (6000.0, 7000.0, 8000.0):
        sp = copy.deepcopy(sys_p)
        sp.milling.rpm = rpm
        T = sp.milling.pass_time + 1.0
        r_open = simulate(sp, model, controller=None, T=T, seed=42)
        # adaptive: internal model at the true TPF
        ctl_a = make_controllers(sp, model, tuning,
                                 f_dist=sp.milling.f_tpf)["pshlqg"]
        r_a = simulate(sp, model, controller=ctl_a, T=T, seed=42)
        # detuned: internal model left at 7000 rpm's TPF
        ctl_d = make_controllers(sp, model, tuning,
                                 f_dist=466.6667)["pshlqg"]
        r_d = simulate(sp, model, controller=ctl_d, T=T, seed=42)
        out[str(int(rpm))] = {
            "open": summarize(r_open, sp.milling.pass_time)["rms_ac_wc"],
            "adaptive": summarize(r_a, sp.milling.pass_time)["rms_ac_wc"],
            "detuned": summarize(r_d, sp.milling.pass_time)["rms_ac_wc"],
        }
        log(f"  {int(rpm)} rpm: open={out[str(int(rpm))]['open']:.2e} "
            f"adaptive={out[str(int(rpm))]['adaptive']:.2e} "
            f"detuned={out[str(int(rpm))]['detuned']:.2e}")
    (RES / "rpm_sweep.json").write_text(json.dumps(out, indent=1))


def main():
    stages = _sys.argv[1:] or ["model", "tuning", "passes", "frozen",
                               "sld", "robust", "rpm"]
    sys_p = SystemParams()
    log("building models...")
    model, fem = build_modal_model(sys_p)
    freqs_end = None
    if "model" in stages:
        freqs_end = stage_model(sys_p, model, fem)
    else:
        freqs_end = np.load(RES / "model.npz")["freqs_end"]
    if "tuning" in stages:
        tuning = stage_tuning(sys_p, model)
    else:
        tuning = json.loads((RES / "tuning.json").read_text())
    if "passes" in stages:
        stage_passes(sys_p, model, tuning, freqs_end)
    if "frozen" in stages:
        stage_frozen(sys_p, model, tuning)
    if "sld" in stages:
        stage_sld(sys_p, model, tuning)
    if "robust" in stages:
        stage_robust(sys_p, model, tuning)
    if "rpm" in stages:
        stage_rpm(sys_p, model, tuning)
    log("campaign complete")


if __name__ == "__main__":
    main()
