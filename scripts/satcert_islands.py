"""SatCERT WP4: causal demonstration of saturation islands.

An operating point exhibits a saturation island when it is LINEARLY
stable in closed loop (sampled Floquet radius < 1) yet a finite,
physically meaningful perturbation — a surface step h left by the
previous pass — triggers sustained chatter, and the SAME perturbation
decays once the amplifier bound is removed. The second leg is the
causal control: it proves the instability is saturation-induced, not a
property of the linear loop or of the perturbation size per se.

Protocol per candidate point (x_T, ap, strategy), all deterministic
(noise off) with the unilateral loss-of-contact nonlinearity ON (it
bounds the large-amplitude regime, as in the physical process):
  1. certificate: h_max (both signs) from certify_sampled;
  2. settle 150 tooth periods, inject the step, observe 150 periods;
  3. escalate h over a ladder (multiples of the clip-onset scale) with
     the ±150 V bound active until the response fails to return to the
     pre-injection orbit (post/pre RMS ratio > 3, measured over the
     last 20 periods) or diverges;
  4. causal control: repeat the smallest chattering h with the bound
     lifted (v_max = 1e9 V) — expect decay;
  5. record the island interval [h_cert, h_chatter) exposure.

Output: results/satcert_islands.json + per-run traces (downsampled)
for the figure script.
"""
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pipeline import RPM_REF, controller_for, models, stage_a  # noqa: E402

from avc.satcert import certify_sampled, lift_sampled  # noqa: E402
from avc.simulate import simulate  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "satcert_islands.json"
V_MAX = 150.0
SETTLE, OBSERVE = 150, 150

# candidates: linear-only ("L") census cells of both strategies
POINTS = (
    ("PS-LPV", 0.05, 1.5e-3),
    ("PS-LPV", 0.05, 2.0e-3),
    ("FROZEN", 0.095, 0.9e-3),
    ("FROZEN", 0.05, 2.0e-3),
)
LADDER = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)     # x clip-onset scale


def log(msg):
    print(f"[islands {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_once(mm, K, ap, x, h, v_max, tau):
    T = (SETTLE + OBSERVE) * tau
    r = simulate(mm, RPM_REF, ap, controller=K, T=T, x0=x, moving=False,
                 fs=100e3, noise_rms=0.0, loss_of_contact=True,
                 v_max=v_max, surf_step=(SETTLE * tau, h))
    t_inj = SETTLE * tau
    pre = (r.t >= t_inj - 20 * tau) & (r.t < t_inj)
    post = r.t >= r.t[-1] - 20 * tau
    rms_pre = float(np.sqrt(np.mean(r.w_mill[pre] ** 2)))
    rms_post = float(np.sqrt(np.mean(r.w_mill[post] ** 2)))
    clip = float(np.mean(np.abs(r.u[r.t >= t_inj]) >= v_max - 1e-9))
    return {"rms_pre_um": rms_pre * 1e6, "rms_post_um": rms_post * 1e6,
            "ratio": rms_post / max(rms_pre, 1e-15),
            "clip_frac_post": clip,
            "diverged": r.extras["diverged_at"],
            "peak_w_post_um": float(np.max(np.abs(r.w_mill[post]))) * 1e6,
            "trace": {"t": r.t[::25].tolist(),
                      "w_um": (r.w_mill[::25] * 1e6).round(4).tolist(),
                      "u": r.u[::25].round(3).tolist()}}


def classify(res):
    if res["diverged"] is not None:
        return "diverged"
    return "chatter" if res["ratio"] > 3.0 else "decays"


def main():
    art = stage_a()
    mm = models()[0.0]["full"]
    tau = mm.params.tooth_passing_delay(RPM_REF)
    out = {"rpm": RPM_REF, "v_max": V_MAX, "ladder": LADDER,
           "settle_periods": SETTLE, "observe_periods": OBSERVE,
           "points": {}}

    for strat, x, ap in POINTS:
        key = f"{strat}|x={x*1e3:g}mm|ap={ap*1e3:g}mm"
        K = controller_for(strat, art, x, latency=False)
        cert = certify_sampled(lift_sampled(mm, x, ap, RPM_REF, K), V_MAX)
        rec = {"cert_h_um": cert.h_max * 1e6, "cert_status": cert.status,
               "rho": cert.lin_rho, "u_bar_V": cert.u_bar, "runs": {}}
        log(f"{key}: rho={cert.lin_rho:.4f} status={cert.status} "
            f"h_cert={cert.h_max*1e6:.2f}um u_bar={cert.u_bar:.1f}V")
        if cert.lin_rho >= 1.0:
            rec["verdict"] = "not-linear-stable (skip)"
            out["points"][key] = rec
            continue
        h0 = cert.h_max if cert.feasible and cert.h_max > 0 else 5e-6
        island_h = None
        for mult in LADDER:
            h = min(h0 * mult, 500e-6)
            r_on = run_once(mm, K, ap, x, h, V_MAX, tau)
            verdict = classify(r_on)
            rec["runs"][f"{mult:g}x_sat"] = {k: v for k, v in r_on.items()
                                             if k != "trace"}
            log(f"  h={h*1e6:7.2f}um sat-on : {verdict} "
                f"(ratio {r_on['ratio']:.2f}, clip {r_on['clip_frac_post']:.3f})")
            if verdict != "decays":
                island_h = h
                rec["island_trace_on"] = r_on["trace"]
                # causal control: same h, bound lifted
                r_off = run_once(mm, K, ap, x, h, 1e9, tau)
                v_off = classify(r_off)
                rec["runs"][f"{mult:g}x_nosat"] = {
                    k: v for k, v in r_off.items() if k != "trace"}
                rec["island_trace_off"] = r_off["trace"]
                log(f"  h={h*1e6:7.2f}um sat-OFF: {v_off} "
                    f"(ratio {r_off['ratio']:.2f})")
                rec["verdict"] = ("ISLAND CONFIRMED (saturation-induced)"
                                  if v_off == "decays" else
                                  f"instability not saturation-specific "
                                  f"({v_off} without bound)")
                break
            if h >= 500e-6:
                break
        if island_h is None:
            rec["verdict"] = "no island found up to 500 um"
        rec["island_h_um"] = island_h * 1e6 if island_h else None
        out["points"][key] = rec

    OUT.write_text(json.dumps(out, indent=1))
    log(f"written {OUT}")
    for k, v in out["points"].items():
        log(f"  {k}: {v['verdict']}")


if __name__ == "__main__":
    main()
