"""SatCERT certified campaign (WP3-4 numbers; fills the X/Y/Z
placeholders of docs/paper2_foundational_text_final.md).

All certificates are the implementation-exact sampled-data
region-of-linearity certificates of avc.satcert (Gilbert-Tan maximal
saturation-free admissible set, phase-resolved headroom, both step
signs), evaluated on the 12-mode plant at the production speed.

Stages (results/satcert_campaign.json):
  table    : per position x certified depth at h_req (both strategies)
             + linear (Floquet, sampled loop) boundary  -> Y
  validate : certificate vs 100 kHz nonlinear-simulator clip onset
             (both step signs) at reference points
  hard     : companion-protocol hard-condition sims (x=95 mm)  -> X, Z
  census   : (x, ap) grid zone classification {certified | linear-only
             | forced-saturated | unstable} + island counts
  nsub     : a4-averaging convergence record
"""
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pipeline import RPM_REF, X_EVAL, controller_for, models, stage_a  # noqa: E402

from avc.satcert import certified_depth, certify_sampled, lift_sampled  # noqa: E402
from avc.simulate import simulate  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "satcert_campaign.json"

V_MAX = 150.0
H_REQ = 20e-6                  # declared required tolerance [m]
AP_HI = 5e-3
STRATS = ("FROZEN", "PS-LPV")


def log(msg):
    print(f"[satcert {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def lin_depth(mm, x, K, ap_hi=AP_HI, tol=2e-5):
    """Linear (Floquet) boundary of the sampled-data loop."""
    def rho(ap):
        return lift_sampled(mm, x, ap, RPM_REF, K).rho()
    lo, hi = 0.0, ap_hi
    if rho(hi) < 1.0:
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if rho(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return lo


def stage_table(mm, art):
    out = {}
    for strat in STRATS:
        for x in X_EVAL:
            K = controller_for(strat, art, x, latency=False)
            t0 = time.time()
            ap_c, cert = certified_depth(mm, float(x), RPM_REF, K, V_MAX,
                                         H_REQ, ap_hi=AP_HI)
            ap_l = lin_depth(mm, float(x), K)
            out[f"{strat}|{x}"] = {
                "ap_cert_mm": ap_c * 1e3, "ap_lin_mm": ap_l * 1e3,
                "h_max_um": (cert.h_max * 1e6 if cert else 0.0),
                "u_bar_V": (cert.u_bar if cert else 0.0),
                "rho": (cert.lin_rho if cert else None)}
            log(f"table {strat} x={x*1e3:.0f}mm: cert {ap_c*1e3:.3f} mm, "
                f"lin {ap_l*1e3:.3f} mm  [{time.time()-t0:.0f}s]")
    return out


def stage_validate(mm, art):
    """Certificate vs nonlinear-simulator clip onset, both step signs."""
    x = 0.05
    Ks = controller_for("PS-LPV", art, x, latency=False)
    tau = mm.params.tooth_passing_delay(RPM_REF)
    INJ, TOT = 120, 160

    def clips(ap, h):
        r = simulate(mm, RPM_REF, ap, controller=Ks, T=TOT * tau, x0=x,
                     moving=False, fs=100e3, noise_rms=0.0,
                     loss_of_contact=False, surf_step=(INJ * tau, h))
        return int(np.sum(np.abs(r.u[r.t >= INJ * tau]) >= V_MAX - 1e-9))

    def onset(ap, sign, hi0=400e-6):
        lo, hi = 0.0, hi0
        if clips(ap, sign * hi) == 0:
            return None
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            if clips(ap, sign * mid) > 0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    out = {}
    for ap in (0.3e-3, 1.0e-3):
        cert = certify_sampled(lift_sampled(mm, x, ap, RPM_REF, Ks), V_MAX)
        sp, sm = onset(ap, +1), onset(ap, -1)
        out[f"{ap*1e3:.1f}mm"] = {
            "cert_h_um": cert.h_max * 1e6,
            "cert_h_plus_um": cert.meta["h_plus"] * 1e6,
            "cert_h_minus_um": cert.meta["h_minus"] * 1e6,
            "sim_onset_plus_um": (sp * 1e6 if sp else None),
            "sim_onset_minus_um": (sm * 1e6 if sm else None)}
        log(f"validate ap={ap*1e3}mm: cert ±{cert.h_max*1e6:.2f} "
            f"(+{cert.meta['h_plus']*1e6:.2f}/-{cert.meta['h_minus']*1e6:.2f})"
            f" vs sim +{sp*1e6 if sp else -1:.2f}/-{sm*1e6 if sm else -1:.2f} um")
    return out


def stage_hard(mm, art):
    """Companion-protocol hard condition (x=95 mm), saturation active."""
    x_hard = 0.095
    out = {}
    for ap in (1.0e-3, 2.0e-3):
        for strat in STRATS:
            K = controller_for(strat, art, x_hard, latency=False)
            r = simulate(mm, RPM_REF, ap, controller=K, T=0.5, x0=x_hard,
                         moving=False, fs=250e3, seed=3, v_max=V_MAX)
            out[f"{ap*1e3:.0f}mm|{strat}"] = {
                "rms_w_um": r.rms_w(0.1) * 1e6, "peak_w_um": r.peak_w(0.1) * 1e6,
                "rms_u_V": r.rms_u(0.1), "peak_u_V": r.peak_u(0.1),
                "sat_frac": r.extras["sat_frac"],
                "diverged": r.extras["diverged_at"]}
            log(f"hard ap={ap*1e3:.0f}mm {strat}: "
                f"rms {r.rms_w(0.1)*1e6:.2f} um, peak u {r.peak_u(0.1):.1f} V, "
                f"sat {r.extras['sat_frac']:.4f}")
    return out


def stage_census(mm, art):
    aps = np.arange(0.1e-3, 3.55e-3, 0.1e-3)
    out = {"ap_grid_mm": (aps * 1e3).round(2).tolist(), "zones": {}}
    for strat in STRATS:
        for x in X_EVAL:
            K = controller_for(strat, art, x, latency=False)
            zones = []
            for ap in aps:
                c = certify_sampled(lift_sampled(mm, float(x), float(ap),
                                                 RPM_REF, K), V_MAX)
                if c.status == "lin-unstable":
                    zones.append("U")           # linearly unstable
                elif c.status == "forced-saturated":
                    zones.append("S")           # forced orbit clips
                elif c.h_max >= H_REQ:
                    zones.append("C")           # certified at h_req
                else:
                    zones.append("L")           # linear-only (island zone)
            out["zones"][f"{strat}|{x}"] = "".join(zones)
            log(f"census {strat} x={x*1e3:.0f}mm: {''.join(zones)}")
    return out


def stage_nsub(mm, art):
    K = controller_for("PS-LPV", art, 0.05, latency=False)
    out = {}
    for n_sub in (4, 8, 16):
        c = certify_sampled(lift_sampled(mm, 0.05, 1.0e-3, RPM_REF, K,
                                         n_sub=n_sub), V_MAX)
        out[str(n_sub)] = {"h_max_um": c.h_max * 1e6, "u_bar_V": c.u_bar}
        log(f"nsub={n_sub}: h={c.h_max*1e6:.3f} um")
    return out


def main():
    art = stage_a()
    mm = models()[0.0]["full"]
    res = {"rpm": RPM_REF, "v_max": V_MAX, "h_req_um": H_REQ * 1e6,
           "certificate": "O-infinity-sampled (region of linear behavior, "
                          "phase-resolved headroom, both step signs)"}
    res["table"] = stage_table(mm, art)
    res["validate"] = stage_validate(mm, art)
    res["hard"] = stage_hard(mm, art)
    res["census"] = stage_census(mm, art)
    res["nsub"] = stage_nsub(mm, art)

    worst = {s: min(res["table"][f"{s}|{x}"]["ap_cert_mm"] for x in X_EVAL)
             for s in STRATS}
    res["worst_position_cert_mm"] = worst
    res["Y_ratio"] = worst["PS-LPV"] / worst["FROZEN"] if worst["FROZEN"] > 0 \
        else None
    h2 = res["hard"]["2mm|FROZEN"], res["hard"]["2mm|PS-LPV"]
    res["X_rms_reduction_pct"] = (1.0 - h2[1]["rms_w_um"] / h2[0]["rms_w_um"]) \
        * 100.0
    res["Z_voltage_fraction_pct"] = h2[1]["peak_u_V"] / h2[0]["peak_u_V"] * 100.0
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    log(f"Y = {res['Y_ratio']:.2f}x   X = {res['X_rms_reduction_pct']:.1f}%   "
        f"Z = {res['Z_voltage_fraction_pct']:.1f}%")
    log(f"written {OUT}")


if __name__ == "__main__":
    main()
