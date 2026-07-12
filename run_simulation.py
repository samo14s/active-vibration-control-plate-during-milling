#!/usr/bin/env python3
"""Re-run of the simulation study behind:

    Wang, Sun & Sun, "Adaptive Control Strategy for High-Speed Milling of
    Thin-Walled Components", ASME Open Journal of Engineering, Vol. 5,
    051006 (2026).  DOI 10.1115/1.4070743

Three strategies are compared on a time-compressed wall-thinning pass of
an Al 7075-T6 thin-walled workpiece (n = 5 trials each, matching the
paper's replication count):

  * conventional  - offline-optimised fixed parameters + operator-style
                    manual reductions on visible chatter (Sec. 4.1)
  * ssv           - +/-10 % sinusoidal spindle-speed variation at 5 Hz
  * adaptive      - the full Sec. 3 architecture: RLS identification,
                    online stability lobes, multiparameter coordination

Outputs (figures, results.json, REPORT.md) are written to results/.

Usage:
    python3 run_simulation.py            # full campaign (5 trials)
    python3 run_simulation.py --quick    # 2 trials, faster smoke run
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from milling_sim.metrics import summarize
from milling_sim.parameters import PaperSetup
from milling_sim.plots import (fig3_performance, fig4_quality,
                               fig_identification, fig_lobes, fig_spectrum)
from milling_sim.runner import aggregate, run_campaign

STRATEGIES = ["conventional", "ssv", "adaptive"]

# Paper reference values used in the comparison table of the report.
PAPER = {
    "avg_mrr": {"conventional": 17.2, "adaptive": 24.8},
    "peak_mrr": {"conventional": 20.0, "adaptive": 32.5},
    "mrr_improvement_pct": 44.2,
    "ssv_mrr_improvement_pct": 18.0,
    "time_reduction_pct": -29.6,
    "energy_reduction_pct": -25.8,
    "ra_band_adaptive": (0.9, 1.3),
    "ra_band_conventional": (0.8, 4.5),
    "ra_sigma": {"conventional": 1.24, "adaptive": 0.15},
    "thickness_mm": {"conventional": 0.08, "adaptive": 0.02},
    "rms_limit_um": 25.0,
    "id_accuracy_pct": 3.2,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="2 trials per strategy instead of 5")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    setup = PaperSetup()
    n_trials = 2 if args.quick else 5

    print(f"Running campaign: {STRATEGIES} x {n_trials} trials "
          f"(dt = {setup.sim.dt * 1e6:.0f} us, "
          f"stock = {setup.sim.removal_volume_cm3:.0f} cm3)")
    campaign = run_campaign(setup, STRATEGIES, n_trials=n_trials,
                            base_seed=117)

    # ---- figures ---------------------------------------------------------
    fig3_performance(campaign, os.path.join(args.out, "fig3_performance.png"))
    fig4_quality(campaign, setup, os.path.join(args.out, "fig4_quality.png"))
    fig_lobes(campaign, setup, os.path.join(args.out, "fig_stability_lobes.png"))
    fig_identification(campaign,
                       os.path.join(args.out, "fig_identification.png"))
    fig_spectrum(campaign, setup, os.path.join(args.out, "fig_spectrum.png"))

    # ---- aggregate numbers ------------------------------------------------
    agg = {name: aggregate(campaign[name]["summaries"])
           for name in STRATEGIES}
    durations = {name: [h.t[-1] for h in campaign[name]["histories"]]
                 for name in STRATEGIES}

    def m(name, key):
        return agg[name][key]["mean"]

    mrr_gain = 100.0 * (m("adaptive", "avg_mrr_cm3_min")
                        / m("conventional", "avg_mrr_cm3_min") - 1.0)
    ssv_gain = 100.0 * (m("ssv", "avg_mrr_cm3_min")
                        / m("conventional", "avg_mrr_cm3_min") - 1.0)
    t_gain = 100.0 * (np.mean(durations["adaptive"])
                      / np.mean(durations["conventional"]) - 1.0)
    e_gain = 100.0 * (m("adaptive", "specific_energy_W_min_cm3")
                      / m("conventional", "specific_energy_W_min_cm3") - 1.0)

    # identification accuracy across adaptive trials
    id_errs = []
    for ctl in campaign["adaptive"]["controllers"]:
        f_id = np.asarray(ctl.log.fn_id, float)
        f_tr = np.asarray(ctl.log.fn_true, float)
        ok = np.isfinite(f_id) & np.isfinite(f_tr)
        if ok.any():
            id_errs.append(np.median(
                np.abs(f_id[ok] - f_tr[ok]) / f_tr[ok] * 100.0))
    id_err = float(np.mean(id_errs)) if id_errs else float("nan")

    results = {
        "config": {
            "n_trials": n_trials,
            "dt_s": setup.sim.dt,
            "stock_cm3": setup.sim.removal_volume_cm3,
            "baseline": {"n_rpm": setup.baseline.n_rpm,
                         "vf_mm_min": setup.baseline.vf_mm_min,
                         "ap_mm": setup.baseline.ap_mm},
        },
        "aggregate": agg,
        "durations_s": {k: list(map(float, v)) for k, v in durations.items()},
        "headline": {
            "mrr_improvement_pct": mrr_gain,
            "ssv_mrr_improvement_pct": ssv_gain,
            "machining_time_change_pct": t_gain,
            "specific_energy_change_pct": e_gain,
            "identification_median_error_pct": id_err,
        },
        "paper_reference": PAPER,
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    _write_report(args.out, setup, agg, durations, results)

    print("\n==== headline comparison (simulation vs paper) ====")
    print(f"MRR improvement:      {mrr_gain:+6.1f}%   (paper +44.2%)")
    print(f"SSV MRR improvement:  {ssv_gain:+6.1f}%   (paper +18%)")
    print(f"Machining time:       {t_gain:+6.1f}%   (paper -29.6%)")
    print(f"Specific energy:      {e_gain:+6.1f}%   (paper -25.8%)")
    print(f"ID median error:      {id_err:6.2f}%   (paper 3.2%)")
    print(f"\nOutputs written to {args.out}/")


def _write_report(out: str, setup: PaperSetup, agg: dict,
                  durations: dict, results: dict) -> None:
    h = results["headline"]

    def row(label, key, paper_conv, paper_adap, fmt="{:.2f}", scale=1.0):
        c = agg["conventional"][key]["mean"] * scale
        a = agg["adaptive"][key]["mean"] * scale
        return (f"| {label} | {fmt.format(c)} | {fmt.format(a)} | "
                f"{paper_conv} | {paper_adap} |")

    lines = [
        "# Simulation re-run report",
        "",
        "Re-implementation and re-run of the study in Wang, Sun & Sun,",
        '"Adaptive Control Strategy for High-Speed Milling of Thin-Walled',
        'Components", ASME Open J. Eng. 5, 051006 (2026), '
        "DOI 10.1115/1.4070743.",
        "",
        f"Campaign: 3 strategies x {results['config']['n_trials']} trials, "
        f"time-compressed wall-thinning pass "
        f"({results['config']['stock_cm3']:.0f} cm3 stock, "
        f"100 kHz time-domain integration, 100 Hz control).",
        "",
        "## Headline comparison",
        "",
        "| Quantity | This simulation | Paper |",
        "|---|---|---|",
        f"| MRR improvement (adaptive vs conventional) | "
        f"{h['mrr_improvement_pct']:+.1f}% | +44.2% |",
        f"| MRR improvement (SSV vs conventional) | "
        f"{h['ssv_mrr_improvement_pct']:+.1f}% | +18% |",
        f"| Total machining time | {h['machining_time_change_pct']:+.1f}% | "
        f"-29.6% |",
        f"| Specific cutting energy | "
        f"{h['specific_energy_change_pct']:+.1f}% | -25.8% |",
        f"| Modal identification median error | "
        f"{h['identification_median_error_pct']:.1f}% | 3.2% |",
        "",
        "## Per-strategy detail (means over trials; paper values in the "
        "last two columns)",
        "",
        "| Metric | Conventional (sim) | Adaptive (sim) | Conventional "
        "(paper) | Adaptive (paper) |",
        "|---|---|---|---|---|",
        row("Average MRR (cm3/min)", "avg_mrr_cm3_min", "17.2", "24.8",
            "{:.1f}"),
        row("Peak MRR (cm3/min)", "peak_mrr_cm3_min", "20.0", "32.5",
            "{:.1f}"),
        row("Max RMS vibration (um)", "rms_max_um", "peaks > 150*", "< 25",
            "{:.0f}"),
        row("Time above 25 um (%)", "pct_time_above_25um", "-", "~0",
            "{:.1f}"),
        row("Ra min (um)", "ra_min_um", "0.8", "0.9"),
        row("Ra max (um)", "ra_max_um", "4.5", "1.3"),
        row("Ra sigma (um)", "ra_std_um", "1.24", "0.15"),
        row("Thickness deviation (mm)", "thick_dev_max_mm", "±0.08",
            "±0.02", "{:.3f}"),
        row("Specific energy (W·min/cm3)", "specific_energy_W_min_cm3",
            "3.1*", "2.3*", "{:.1f}"),
        "",
        "\\* Notes on scale: the paper's Fig. 3(a) reports RMS peaks "
        "exceeding 150 um for conventional cutting; our conventional runs "
        "saturate near 115 um RMS (~165 um peak amplitude), set by the "
        "tooth jump-out limit cycle.  The absolute specific energy in the "
        "mechanistic force model (~13 W·min/cm3, i.e. ~0.8 J/mm3) matches "
        "aluminium machining physics; the paper's absolute values (3.1 -> "
        "2.3) are on a different accounting basis, so the *relative* "
        "reduction is the comparable quantity.",
        "",
        "## Simulated machining times (s, compressed pass)",
        "",
        "| Strategy | mean | trials |",
        "|---|---|---|",
    ]
    for name in ("conventional", "ssv", "adaptive"):
        ds = durations[name]
        lines.append(f"| {name} | {np.mean(ds):.1f} | "
                     f"{', '.join(f'{d:.1f}' for d in ds)} |")

    lines += [
        "",
        "## Figures",
        "",
        "* `fig3_performance.png` - vibration / MRR / spindle speed vs "
        "time (paper Fig. 3)",
        "* `fig4_quality.png` - roughness and wall-thickness comparison "
        "(paper Fig. 4)",
        "* `fig_stability_lobes.png` - boundary evolution with material "
        "removal (Eqs. (6)-(12)) and the adaptive trajectory",
        "* `fig_identification.png` - real-time wall-mode tracking",
        "* `fig_spectrum.png` - harmonic vs nonharmonic (chatter) force "
        "content",
        "",
        "## Known deviations from the paper",
        "",
        "1. **SSV baseline** improves MRR by more than the paper's +18% "
        "here (no operator reductions are triggered because its chatter "
        "bursts stay short), while its surface/thickness quality is worse "
        "than the paper reports.  SSV effectiveness is highly "
        "plant-specific; our calibrated wall dynamics sit where 5 Hz "
        "modulation only fragments, rather than suppresses, chatter.",
        "2. **Chatter frequency band**: our calibration places the wall "
        "mode at 1580 Hz falling to ~1050 Hz across the pass, so chatter "
        "appears at 1100-1500 Hz vs the paper's 800-1200 Hz; the paper's "
        "physical walls are larger.  The stability mechanics (lobe "
        "migration through the fixed operating point) are identical.",
        "3. The economics rows of the paper's Table 2 (cost/part, tooling "
        "cost, ROI) and yield/tool-life statistics require shop-floor "
        "data and are outside the physics simulation.",
        "",
        "## Equation-to-code map",
        "",
        "| Paper | Implementation |",
        "|---|---|",
        "| Eqs. (1)-(4) cutting forces | `milling_sim/engine.py` "
        "(`_integrate_interval`) |",
        "| Eq. (5) modal dynamics | `milling_sim/engine.py` |",
        "| Eqs. (6)-(7) parameter evolution | `milling_sim/parameters.py` "
        "(`Mode.wn`, `Mode.zeta`) |",
        "| Eqs. (8)-(12) stability lobes | `milling_sim/stability.py` |",
        "| Eq. (13) process damping | `engine.py` + `stability.py` |",
        "| Eq. (15) stiffness update concept | modal update per control "
        "period (`MillingSimulator._update_modal_arrays`) |",
        "| Eq. (16) objective | `milling_sim/control.py` "
        "(`_achievable_q`, weights in `ControllerParams`) |",
        "| Eq. (17) stability constraint | `AdaptiveController.observe` |",
        "| Eqs. (18)-(19) speed adaptation | gradient + relocation loop |",
        "| Eq. (20) feed MPC | `AdaptiveController._solve_feed_mpc` |",
        "| Eq. (21) depth adaptation | slow loop in `observe` |",
        "| Eq. (22) coordination | hierarchical structure of `observe` |",
        "| Eq. (23) GP learning | `milling_sim/learning.py` |",
        "| Eq. (24) min-max robustness | worst-case fn band in "
        "`_ap_forced_mm` |",
        "| Eq. (25) gain scheduling | `alpha_n` in `observe` |",
        "| Eq. (26) actuator limits | slew clamps in `observe` |",
        "| RLS identification (Sec. 3.1) | `milling_sim/identification.py` |",
    ]
    with open(os.path.join(out, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
