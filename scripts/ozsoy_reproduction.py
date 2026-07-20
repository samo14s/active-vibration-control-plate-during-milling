"""Published-parameter reproduction of the Ozsoy/Sims/Ozturk (MSSP
224:111942, 2025) DVF saturation islands, on OZSOY_PUBLISHED
(avc/sdof.py; Tables 1-2 + Eq. (2), y-projection of Eq. (4) with the
regenerative sign calibrated against the published uncontrolled
boundary — declared).

Deliverables (results/ozsoy_reproduction.json +
docs/figures/ozsoy_reproduction.*):
 1. OL / DVF linear boundaries over 1700-2250 rpm (shape vs Fig. 12).
 2. Forced actuator-force map F*(rpm, doc) and cell-by-cell validation
    against the measured amplitudes printed in Fig. 10.
 3. Zone census (certified / linear-only / forced-SATURATED /
    unstable): the S-zones are the predicted islands (their
    frequency-domain model predicts exactly this object, without
    regeneration; ours adds it).
 4. Causal probes: sims from rest at island cells with the 27 N bound
    active vs lifted.
Declared limitations: single mode (their rig also carries 23/47 Hz
robot modes and a 129.3 Hz preloaded mode), no edge coefficients
(untabulated), no runout, 10 kHz DVF rate assumption.
"""
import json
import pathlib
import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from avc.satcert import certify_sampled  # noqa: E402
from avc.sdof import (OZSOY_PUBLISHED, lift_sdof, simulate_sdof,  # noqa: E402
                      tooth_period)
from dataclasses import replace  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ozsoy_reproduction.json"
FIG = ROOT / "docs" / "figures"
H_REQ = 10e-6
RPMS = np.arange(1700.0, 2251.0, 50.0)
DOCS = np.arange(2e-3, 26.1e-3, 2e-3)

# (rpm, doc_mm, measured actuator force N) read from Fig. 10
CELLS = [(1700, 20, 13), (1700, 10, 10), (1700, 5, 12), (1700, 4, 9),
         (1750, 20, 16), (1750, 5, 14), (1800, 10, 15), (1800, 4, 13),
         (1850, 5, 19), (1900, 5, 20), (2100, 5, 16), (2150, 5, 17),
         (2200, 4, 7)]


def log(m):
    print(f"[ozsoy {time.strftime('%H:%M:%S')}] {m}", flush=True)


def crit(cfg, rpm, hi=30e-3):
    lo = 0.0
    if lift_sdof(cfg, hi, rpm).rho() < 1.0:
        return hi
    while hi - lo > 0.1e-3:
        mid = 0.5 * (lo + hi)
        if lift_sdof(cfg, mid, rpm).rho() < 1.0:
            lo = mid
        else:
            hi = mid
    return lo


def probe(cfg, rpm, doc, f_max, h=0.0, settle=120, observe=150):
    tau = tooth_period(cfg, rpm)
    r = simulate_sdof(cfg, doc, rpm, T=(settle + observe) * tau, fs=50e3,
                      f_max=f_max, surf_step=(settle * tau, h))
    post = r["t"] >= r["t"][-1] - 20 * tau
    rms = float(np.sqrt(np.mean(r["x"][post] ** 2))) if post.any() else 0.0
    return {"rms_um": rms * 1e6, "clip_frac": r["clip_frac"],
            "diverged": r["diverged"]}


def main():
    cfg = OZSOY_PUBLISHED
    out = {"config": cfg.label, "h_req_um": H_REQ * 1e6}

    cfg_ol = replace(cfg, g_v=0.0)
    out["boundaries"] = {}
    for rpm in RPMS:
        out["boundaries"][f"{rpm:g}"] = {
            "ol_mm": crit(cfg_ol, rpm) * 1e3,
            "dvf_mm": crit(cfg, rpm) * 1e3}
    log("boundaries done: " + ", ".join(
        f"{r}:{v['ol_mm']:.1f}/{v['dvf_mm']:.1f}"
        for r, v in out["boundaries"].items()))

    # forced-force map + zones
    Fmap = np.full((len(DOCS), len(RPMS)), np.nan)
    zones = np.full((len(DOCS), len(RPMS)), "U", dtype="U1")
    for j, rpm in enumerate(RPMS):
        for i, doc in enumerate(DOCS):
            lift = lift_sdof(cfg, float(doc), float(rpm))
            c = certify_sampled(lift, cfg.f_max)
            if c.status == "lin-unstable":
                zones[i, j] = "U"
                continue
            Fmap[i, j] = float(np.max(np.abs(lift.forced_voltages())))
            if c.status == "forced-saturated":
                zones[i, j] = "S"
            elif c.h_max >= H_REQ:
                zones[i, j] = "C"
            else:
                zones[i, j] = "L"
        log(f"rpm={rpm:g}: " + "".join(zones[:, j]))
    out["zones"] = {"doc_mm": (DOCS * 1e3).round(1).tolist(),
                    "rpm": RPMS.tolist(),
                    "rows_doc_x_rpm": ["".join(zones[i, :])
                                       for i in range(len(DOCS))]}

    # cell validation
    val = []
    for rpm, doc, meas in CELLS:
        lift = lift_sdof(cfg, doc * 1e-3, float(rpm))
        if lift.rho() >= 1.0:
            val.append({"rpm": rpm, "doc_mm": doc, "meas_N": meas,
                        "model_N": None, "note": "lin-unstable (model)"})
            continue
        val.append({"rpm": rpm, "doc_mm": doc, "meas_N": meas,
                    "model_N": round(float(np.max(np.abs(
                        lift.forced_voltages()))), 1)})
    ratios = [v["model_N"] / v["meas_N"] for v in val if v["model_N"]]
    out["cell_validation"] = {"cells": val,
                              "median_model_over_meas":
                              float(np.median(ratios))}
    log(f"cell validation: median model/meas = {np.median(ratios):.2f}")

    # causal probes: island cells (S) and a saturation-free cell
    probes = {}
    for rpm, doc, tag in ((1900.0, 5e-3, "central island (S)"),
                          (1850.0, 6e-3, "central island (S)"),
                          (2100.0, 5e-3, "post-lobe band"),
                          (1750.0, 10e-3, "saturation-free cell")):
        on = probe(cfg, rpm, doc, cfg.f_max)
        off = probe(cfg, rpm, doc, 1e12)
        probes[f"rpm={rpm:g}|doc={doc*1e3:g}mm"] = {
            "tag": tag, "bound_on": on, "bound_off": off,
            "saturation_induced_degradation":
                bool(on["rms_um"] > 3.0 * max(off["rms_um"], 1e-9)
                     or (on["diverged"] and not off["diverged"]))}
        log(f"probe {rpm:g}/{doc*1e3:g}mm [{tag}]: on rms "
            f"{on['rms_um']:.1f}um clip {on['clip_frac']:.3f} "
            f"div={on['diverged']} | off rms {off['rms_um']:.1f}um "
            f"div={off['diverged']}")
    out["probes"] = probes

    OUT.write_text(json.dumps(out, indent=1))
    log(f"written {OUT}")

    # ---------------- figure -----------------------------------------
    plt.rcParams.update({"font.size": 7.5, "font.family": "serif"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.9))
    R, D = np.meshgrid(RPMS, DOCS * 1e3)
    pc = ax1.contourf(R, D, np.clip(Fmap, 0, 27), levels=12,
                      cmap="turbo")
    ax1.contour(R, D, Fmap, levels=[27.0], colors="r", linewidths=1.6)
    bs = [out["boundaries"][f"{r:g}"] for r in RPMS]
    ax1.plot(RPMS, [b["dvf_mm"] for b in bs], "b-", lw=1.4,
             label="DVF boundary (model)")
    ax1.plot(RPMS, [b["ol_mm"] for b in bs], "b--", lw=1.2,
             label="uncontrolled boundary (model)")
    for rpm, doc, meas in CELLS:
        ax1.plot(rpm, doc, "wo", ms=9, mec="k")
        ax1.text(rpm, doc, f"{meas}", ha="center", va="center",
                 fontsize=5.4)
    ax1.set_ylim(0, 27)
    ax1.set_xlabel("spindle speed [rpm]")
    ax1.set_ylabel("depth of cut [mm]")
    ax1.set_title("forced actuator force [N]; red contour = 27 N "
                  "(island boundary);\ncircles: measured amplitudes "
                  "(Ozsoy Fig. 10)", fontsize=8)
    ax1.legend(fontsize=6, loc="upper right")
    fig.colorbar(pc, ax=ax1, shrink=0.9)

    zmap = np.zeros_like(Fmap)
    code = {"C": 0, "L": 1, "S": 2, "U": 3}
    for i in range(len(DOCS)):
        for j in range(len(RPMS)):
            zmap[i, j] = code[zones[i, j]]
    from matplotlib.colors import ListedColormap
    ax2.pcolormesh(R, D, zmap, cmap=ListedColormap(
        ["#3f8f4f", "#f2c14e", "#6b1f2a", "#d1948f"]), vmin=-0.5,
        vmax=3.5, shading="nearest")
    ax2.set_xlabel("spindle speed [rpm]")
    ax2.set_title("SatCERT zones: green certified / yellow linear-only"
                  " /\ndark-red forced-SATURATED (= predicted island) "
                  "/ pink unstable", fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"ozsoy_reproduction.{ext}", dpi=250,
                    bbox_inches="tight")
    log("figure saved")


if __name__ == "__main__":
    main()
