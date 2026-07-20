"""Print every number the manuscript needs, in manuscript order."""
import sys

import numpy as np

from common import cached  # noqa: F401
from pipeline import (P, REMOVALS, RPM_REF, STRATEGIES, X_EVAL, models,
                      stage_a, stage_b, stage_c, stage_c2, stage_d, stage_e)


def fm(v, n=3):
    return f"{v * 1e3:.{n}f}"


def main():
    A = stage_a()
    B = stage_b()
    C = stage_c()
    C2 = stage_c2()
    D = stage_d()
    E = stage_e()

    print("=" * 70)
    print("SYNTHESIS (Sec. 3 / 5.2)")
    S = A["sched"]
    print(f"gamma_common = {S.gammas[0, 0]:.2f}")
    print(f"midpoint abscissa = {A['midpoint_abscissa']:.2f} 1/s")
    print(f"frozen design point = {A['frozen_raw'].meta['design_point']}")
    print(f"frozen worst-case gamma = "
          f"{A['frozen_raw'].meta['worst_case_gamma']:.1f}")
    print(f"DPD gains kp={A['dpd'][0]:.3g} kd={A['dpd'][1]:.3g}, "
          f"worst-pos a_lim {fm(A['dpd_alim'])} mm")
    print(f"kr_map = { {round(k * 1e3): round(v, 1) for k, v in A['kr_map'].items()} }")
    print("certificates:")
    for k, v in A["certs"].items():
        print(f"  {k}: rs={v['rs']:.3f}  s_rho={v['s_rho']:.5f}")

    print("=" * 70)
    print(f"SLD TABLE at {RPM_REF:.0f} rpm (stage E removal-0 rows + stage B)")
    rpms = B["rpms"]
    i_ref = int(np.argmin(np.abs(rpms - RPM_REF)))
    print(f"(stage B grid speed nearest ref: {rpms[i_ref]:.0f} rpm)")
    for s in STRATEGIES:
        row_b = [fm(B["curves"][(s, x)][i_ref]) for x in X_EVAL]
        row_min = [fm(min(B["curves"][(s, x)])) for x in X_EVAL]
        print(f"{s:10s} @{rpms[i_ref]:.0f}: {row_b}   band-min: {row_min}")
    for key, lbl in (("sched", "PS-LPV"), ("frozen", "FROZEN")):
        row = [fm(E[(key, x, 0.0)]) for x in X_EVAL]
        print(f"{lbl:10s} @{RPM_REF:.0f} (exact, stage E): {row}")

    print("=" * 70)
    print("TIME DOMAIN (Sec. 5.3)")
    for k, d in C.items():
        if k[0] == "fixed":
            print(f"fixed  {k[1]:10s} rms_w={d['rms_w'] * 1e6:8.2f} um "
                  f"peak_w={d['peak_w'] * 1e6:8.2f} um "
                  f"rms_u={d['rms_u']:6.2f} V peak_u={d['peak_u']:6.1f} V")
    for k, d in C.items():
        if k[0] == "pass":
            print(f"pass   {k[1]:10s} rms_w={d['rms_w'] * 1e6:8.2f} um "
                  f"rms_u={d['rms_u']:6.2f} V div={d['extras']['diverged_at']}")
    for (ap, s), d in C2.items():
        print(f"hard ap={ap * 1e3:.1f} {s:10s} rms_w={d['rms_w'] * 1e6:12.2f} um "
              f"rms_u={d['rms_u']:6.1f} V peak_u={d['peak_u']:6.1f} V")

    print("=" * 70)
    print("MONTE-CARLO (Sec. 5.5), n =", len(next(iter(D.values()))))
    for s, v in D.items():
        v = np.asarray(v) * 1e3
        print(f"{s:10s} med={np.median(v):.2f} IQR=[{np.percentile(v, 25):.2f},"
              f"{np.percentile(v, 75):.2f}] "
              f"P(>=0.3mm)={np.mean(v >= 0.3) * 100:.0f}% "
              f"P(>=1mm)={np.mean(v >= 1.0) * 100:.0f}%")

    print("=" * 70)
    print("REMOVAL STUDY (Sec. 5.6)")
    for r in REMOVALS:
        for key in ("sched", "sched-fresh", "frozen"):
            row = [fm(E[(key, x, r)]) for x in X_EVAL]
            print(f"removal {r * 1e3:.1f}mm {key:12s}: {row}")
    mdl = models()
    f0 = mdl[0.0]["design"].freqs_hz
    f1 = mdl[max(REMOVALS)]["design"].freqs_hz
    print("freq drift over full recession [%]:",
          np.round((f1 / f0 - 1) * 100, 2))


if __name__ == "__main__":
    main()
