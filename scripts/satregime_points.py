"""Saturated-regime record: circle-criterion / small-gain margins at
the reference points of the island campaign, the low-depth structural
sweep, and the anti-windup line search. Writes
results/satregime_points.json (the machine record behind §6 of
docs/paper2_foundational_text_final.md)."""
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pipeline import RPM_REF, controller_for, models, stage_a  # noqa: E402

from avc import sld  # noqa: E402
from avc.satcert import (aw_min_gamma, certify_saturated,  # noqa: E402
                         lift_sampled)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "satregime_points.json"

POINTS = (("PS-LPV", 0.05, 1.5e-3, "island -37.6um"),
          ("PS-LPV", 0.05, 2.0e-3, "island -50um"),
          ("FROZEN", 0.095, 0.9e-3, "island -42um"),
          ("FROZEN", 0.05, 2.0e-3, "no island to +/-500um"),
          ("PS-LPV", 0.095, 2.0e-3, "hard condition"))


def log(m):
    print(f"[satregime {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    art = stage_a()
    mm = models()[0.0]["full"]
    out = {"open_loop_crit_x50_mm":
           sld.critical_depth(mm, 0.05, RPM_REF, controller=None,
                              ap_hi=5e-3) * 1e3,
           "points": {}, "low_depth": {}, "aw": {}}

    for strat, x, ap, note in POINTS:
        K = controller_for(strat, art, x, latency=False)
        r = certify_saturated(lift_sampled(mm, x, ap, RPM_REF, K), 150.0)
        out["points"][f"{strat}|x={x*1e3:g}mm|ap={ap*1e3:g}mm"] = {
            "circle_max": r["circle_max"], "gamma_2": r["gamma_2"],
            "global_recovery": r["global_recovery"],
            "rho": r["linear_cert"].lin_rho, "note": note}
        log(f"{strat} x={x*1e3:g} ap={ap*1e3:g}: circle "
            f"{r['circle_max']:.3f} gamma2 {r['gamma_2']:.3f}")

    for strat in ("FROZEN", "PS-LPV"):
        K = controller_for(strat, art, 0.05, latency=False)
        row = {}
        for ap in (0.05e-3, 0.15e-3, 0.25e-3, 0.4e-3):
            r = certify_saturated(lift_sampled(mm, 0.05, ap, RPM_REF, K),
                                  150.0, n_grid=512)
            row[f"{ap*1e3:g}mm"] = {"circle_max": r["circle_max"],
                                    "gamma_2": r["gamma_2"]}
            log(f"low-depth {strat} ap={ap*1e3:g}: circle "
                f"{r['circle_max']:.3f}")
        out["low_depth"][strat] = row

    for strat in ("FROZEN", "PS-LPV"):
        K = controller_for(strat, art, 0.05, latency=False)
        a, cb, c0, _ = aw_min_gamma(mm, 0.05, 0.1e-3, RPM_REF, K,
                                    alphas=(-1.0, -0.5, -0.2, -0.1,
                                            -0.05, 0.0, 0.05, 0.1,
                                            0.2, 0.5, 1.0))
        out["aw"][strat] = {"alpha_best": a, "circle_best": cb,
                           "circle_no_aw": c0}
        log(f"AW {strat}: {c0:.3f} -> {cb:.3f} at alpha={a}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    log(f"written {OUT}")


if __name__ == "__main__":
    main()
