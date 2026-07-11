"""
main_gap_spindle_sync.py
=========================
RESEARCH-GAP EXPERIMENT : spindle-speed uncertainty vs. learned feedforward
---------------------------------------------------------------------------

The DARC-MPC v3 feedforward is indexed by an OPEN-LOOP clock
    φ_clock = 2π (k mod n_per) / n_per,
i.e. it assumes the tooth-passing period is exactly known and constant.
This experiment quantifies what happens when that assumption is violated
(spindle-speed offset or fluctuation), and shows that the v4 PLAD
controller (sensorless phase-locked feedforward + confidence gating)
closes the gap.

Scenario matrix
---------------
  A. Constant spindle-speed offset (tiled periodic coefficients):
       ap = 0.3 mm : δ_eff = 0 %, +1.23 %, +2.50 %, −1.20 %
       ap = 0.6 mm : δ_eff = 0 %, +2.50 %
     δ_eff = n_per_nom/n_per_act − 1 is the EFFECTIVE frequency offset
     realised by the discrete simulation (the disturbance period is
     exactly n_per_act·dt).
  B. Sinusoidal spindle-speed fluctuation ±1 % @ 2 Hz (untiled,
     phase-continuous coefficients), ap = 0.3 mm, T = 1 s.
  C. Long pass (T = 4 s) with +2.5 % offset, ap = 0.3 mm — sustained
     lock while the tool travels along the plate (position-scheduled
     phase reference).

Controllers
-----------
  LQG      : baseline feedback (grid-optimised weights, as in article)
  DARC v3  : LQG base + NN feedforward on the open-loop clock
  DARC v4  : SAME trained NN weights as v3, synchronised by the
             sensorless spindle-phase observer (PLL) with
             confidence-gated feedforward (PLAD)

All controllers are designed/trained at NOMINAL speed only; the actual
speed of each scenario is unknown to them.

Outputs: results_gap_sync/  (figures + metrics.json + summary.md)
"""
import os
import sys
import json
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _d in ("01_core", "02_controllers", "03_analysis", "05_main"):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plate_model import PlateModel
from milling_force import milling_force_coeffs, precompute_alpha_periodic
from lqg_controller import LQGController
from darc_mpc_v3_controller import DARC_MPC_v3_Controller
from darc_mpc_v4_plad_controller import DARC_MPC_v4_PLAD_Controller
from newmark_solver import NewmarkSimulator

# ============================================================
# Physical parameters (identical to main_simulation.py)
# ============================================================
LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU_AL = 2830, 69e9, 0.33
NT = 3; DT_TOOL = 0.010
ETA_H = np.deg2rad(35); GAMMA_N = np.deg2rad(15)
KT_NOMINAL = 925e6; KN = 0.26; MU_C = 0.20
RPM_NOM = 4900
FT = 0.02e-3; AE = 0.1e-3
D31 = 175e-12; H_PA = 0.7e-3
E_PE = 63e9; NU_PE = 0.35
N1, N2 = 30, 24
N_MODES = 3
ZETA = [0.0031, 0.0017, 0.0027]

DT = 5e-5
T_STEADY = 0.15          # metrics window start (transient + PLL lock-in)

PHI_ST = np.pi - np.arccos(1 - AE/(DT_TOOL/2))
PHI_EX = np.pi
RT = DT_TOOL/2
K1 = KN*np.cos(ETA_H)
K2 = 1 + MU_C*np.tan(ETA_H)*np.cos(GAMMA_N) - KN*np.sin(ETA_H)

OUT_DIR = os.path.join(_ROOT, "results_gap_sync")
os.makedirs(OUT_DIR, exist_ok=True)

COL = {"LQG": "#2E8B57", "DARC v3": "#DC143C", "DARC v4": "#1E5AA8"}


# ============================================================
# Helpers
# ============================================================
def build_plate(ap):
    plate = PlateModel(LP, HP, BP, RHO, E_AL, NU_AL,
                       N1=N1, N2=N2, n_modes=N_MODES,
                       zeta_modes=ZETA, verbose=False)
    plate.precompute_Dp(zp_pos=HP - ap/2, n_pos=2001)
    plate.set_observation(x_obs=LP, z_obs=HP)
    plate.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    return plate


def setup_const(plate, ap, n_per_act, T_end):
    """Tiled periodic coefficients with an exact period of n_per_act steps."""
    rpm_eff = 60.0/(NT*n_per_act*DT)
    Omega = 2*np.pi*rpm_eff/60
    tau = 60/(NT*rpm_eff)
    sim = NewmarkSimulator(plate, dt=DT, T_end=T_end, ft=FT, tau=tau,
                           verbose=False)
    v_feed = FT*NT*rpm_eff/60
    kp_idx = np.clip(np.round(np.minimum(v_feed*sim.t_vec, LP)/LP*2000
                              ).astype(int), 0, 2000)
    a3, a4 = precompute_alpha_periodic(DT, n_per_act, sim.nstep, Omega,
                                       NT, RT, ETA_H, PHI_ST, PHI_EX,
                                       HP-ap, HP, K1, K2, KT_NOMINAL)
    # true clock phase of the disturbance (for diagnostics)
    k_arr = np.arange(sim.nstep)
    phase_true = 2*np.pi*(k_arr % n_per_act)/n_per_act
    return sim, kp_idx, a3, a4, phase_true


def setup_ssv(plate, ap, n_per_nom, A_ssv, f_ssv, T_end):
    """
    Phase-continuous coefficients for a sinusoidal spindle-speed
    fluctuation Ω(t) = Ω_car·(1 + A_ssv·sin(2π f_ssv t)), where Ω_car is
    the carrier realised by n_per_nom steps (so A_ssv = 0 reproduces the
    nominal tiled case exactly).  The regenerative delay is kept at its
    carrier value (delay modulation is second order for A_ssv ≈ 1 %).
    """
    rpm_car = 60.0/(NT*n_per_nom*DT)
    Omega_car = 2*np.pi*rpm_car/60
    tau = 60/(NT*rpm_car)
    sim = NewmarkSimulator(plate, dt=DT, T_end=T_end, ft=FT, tau=tau,
                           verbose=False)
    v_feed = FT*NT*rpm_car/60
    kp_idx = np.clip(np.round(np.minimum(v_feed*sim.t_vec, LP)/LP*2000
                              ).astype(int), 0, 2000)

    # fine profile of (a3, a4) over one tooth period vs clock phase
    n_sub = 16*n_per_nom
    prof3 = np.zeros(n_sub); prof4 = np.zeros(n_sub)
    for k in range(n_sub):
        t_eq = k*(n_per_nom*DT)/n_sub
        prof3[k], prof4[k] = milling_force_coeffs(
            t_eq, Omega_car, NT, RT, ETA_H,
            PHI_ST, PHI_EX, HP-ap, HP, K1, K2, KT_NOMINAL)

    # accumulated spindle angle -> tooth-clock phase per step
    t = sim.t_vec
    t_eq = t + A_ssv*(1 - np.cos(2*np.pi*f_ssv*t))/(2*np.pi*f_ssv)
    phase_true = (Omega_car*NT*t_eq) % (2*np.pi)
    idx = phase_true/(2*np.pi)*n_sub
    a3 = np.interp(idx, np.arange(n_sub+1),
                   np.append(prof3, prof3[0]))
    a4 = np.interp(idx, np.arange(n_sub+1),
                   np.append(prof4, prof4[0]))
    return sim, kp_idx, a3, a4, phase_true


def rolling_rms(y, win_s=0.02):
    n = max(1, int(win_s/DT))
    y2 = np.asarray(y)**2
    c = np.cumsum(np.insert(y2, 0, 0.0))
    out = np.full(len(y2), np.nan)
    out[n-1:] = np.sqrt((c[n:] - c[:-n])/n)
    return out


def reset_v3(v3):
    v3.history_u_lqg = []; v3.history_u_ff = []; v3.history_u_total = []
    v3.history_phase = []; v3.history_safety = []


def run_all(sim, kp_idx, a3, a4, lqg, v3, v4):
    res = {}
    res["LQG"] = sim.simulate(a3, a4, kp_idx, controller=lqg, progress=False)
    reset_v3(v3)
    res["DARC v3"] = sim.simulate(a3, a4, kp_idx, controller=v3, progress=False)
    res["DARC v3"]["u_ff"] = np.array(v3.history_u_ff)
    res["DARC v3"]["phase_used"] = np.array(v3.history_phase)
    v4.reset_runtime()
    res["DARC v4"] = sim.simulate(a3, a4, kp_idx, controller=v4, progress=False)
    res["DARC v4"]["u_ff"] = np.array(v4.history_u_ff)
    res["DARC v4"]["phase_used"] = np.array(v4.history_phase_est)
    res["DARC v4"]["conf"] = np.array(v4.history_conf)
    res["DARC v4"]["omega_hat"] = np.array(v4.history_omega_hat)
    res["DARC v4"]["alpha_eff"] = np.array(v4.history_alpha_eff)
    res["DARC v4"]["n_safety"] = int(np.sum(v4.history_safety))
    return res


def metrics_of(res, T_end):
    i0 = int(T_STEADY/DT)
    out = {}
    for name, r in res.items():
        y = r["y"]; u = r["u"]
        m = {
            "y_rms_steady_um": float(np.sqrt(np.mean(y[i0:]**2))*1e6),
            "y_rms_full_um":   float(np.sqrt(np.mean(y**2))*1e6),
            "y_max_um":        float(np.max(np.abs(y[i0:]))*1e6),
            "u_rms_V":         float(np.sqrt(np.mean(u[i0:]**2))),
            "u_max_V":         float(np.max(np.abs(u))),
        }
        if "conf" in r:
            m["conf_mean"] = float(r["conf"][i0:].mean())
        out[name] = m
    for name in ("DARC v3", "DARC v4"):
        out[name]["gain_vs_LQG_pct"] = float(
            (1 - out[name]["y_rms_steady_um"]/out["LQG"]["y_rms_steady_um"])*100)
    return out


# ============================================================
# Phase 0 : design & training at NOMINAL conditions
# ============================================================
print("="*72)
print(" RESEARCH-GAP EXPERIMENT : spindle-speed uncertainty vs learned FF")
print("="*72)
t_glob = time.time()

N_PER_NOM = int(np.round(60/(NT*RPM_NOM)/DT))          # 82
controllers = {}                                        # per ap

for ap in (0.3e-3, 0.6e-3):
    print(f"\n--- design/training @ nominal, ap = {ap*1e3:.1f} mm ---")
    plate = build_plate(ap)
    sim_n, kp_n, a3_n, a4_n, _ = setup_const(plate, ap, N_PER_NOM, 0.5)

    lqg = LQGController(plate, dt=DT, verbose=False)
    lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                         w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
    lqg.discretize_observer()

    v3 = DARC_MPC_v3_Controller(plate, dt=DT,
                                ff_lr=0.005, ff_max=10.0, ff_alpha=1.0,
                                alpha4_periodic=a4_n[:N_PER_NOM],
                                n_per=N_PER_NOM,
                                safety_alpha=5.0, enable_adaptation=True,
                                u_max=150.0, verbose=False)
    t0 = time.time()
    v3.pretrain_iterative_simulation(sim_n, a3_n, a4_n, kp_n,
                                     n_iterations=30, n_epochs_per_iter=15,
                                     verbose=False)
    print(f"  v3 trained (30 iterations) in {time.time()-t0:.1f} s")

    v4 = DARC_MPC_v4_PLAD_Controller(plate, dt=DT,
                                     alpha3_periodic=a3_n[:N_PER_NOM],
                                     ff_lr=0.005, ff_max=10.0, ff_alpha=1.0,
                                     alpha4_periodic=a4_n[:N_PER_NOM],
                                     n_per=N_PER_NOM,
                                     safety_alpha=5.0, u_max=150.0,
                                     verbose=False)
    v4.copy_feedforward_from(v3)
    v4.calibrate_phase_reference(sim_n, a3_n, a4_n, kp_n,
                                 T_cal=0.35, t_lock=0.15, verbose=True)
    controllers[ap] = dict(plate=plate, lqg=lqg, v3=v3, v4=v4)


# ============================================================
# Phase A : constant speed offsets
# ============================================================
print("\n" + "="*72)
print(" A. Constant spindle-speed offset (controllers believe nominal)")
print("="*72)

SCEN_A = [
    (0.3e-3, 82, "A1  ap=0.3  δ=0%"),
    (0.3e-3, 81, "A2  ap=0.3  δ=+1.23%"),
    (0.3e-3, 80, "A3  ap=0.3  δ=+2.50%"),
    (0.3e-3, 83, "A4  ap=0.3  δ=-1.20%"),
    (0.6e-3, 82, "A5  ap=0.6  δ=0%"),
    (0.6e-3, 80, "A6  ap=0.6  δ=+2.50%"),
]

all_metrics = {}
saved = {}          # keep selected raw results for figures

for ap, n_act, label in SCEN_A:
    c = controllers[ap]
    sim, kp_idx, a3, a4, ph_true = setup_const(c["plate"], ap, n_act, 0.5)
    res = run_all(sim, kp_idx, a3, a4, c["lqg"], c["v3"], c["v4"])
    m = metrics_of(res, 0.5)
    delta = (N_PER_NOM/n_act - 1)*100
    all_metrics[label] = dict(ap_mm=ap*1e3, delta_pct=round(delta, 2), **m)
    print(f"  {label}: LQG {m['LQG']['y_rms_steady_um']:.4f} µm | "
          f"v3 {m['DARC v3']['y_rms_steady_um']:.4f} "
          f"({m['DARC v3']['gain_vs_LQG_pct']:+.1f}%) | "
          f"v4 {m['DARC v4']['y_rms_steady_um']:.4f} "
          f"({m['DARC v4']['gain_vs_LQG_pct']:+.1f}%)  "
          f"conf={m['DARC v4'].get('conf_mean', 0):.2f}")
    if label.startswith(("A3", "A1")):
        saved[label] = (sim, res, ph_true, n_act)


# ============================================================
# Phase B : sinusoidal spindle-speed fluctuation
# ============================================================
print("\n" + "="*72)
print(" B. Sinusoidal spindle-speed fluctuation ±1 % @ 2 Hz (ap = 0.3 mm)")
print("="*72)

ap = 0.3e-3
c = controllers[ap]
sim_b, kp_b, a3_b, a4_b, ph_true_b = setup_ssv(c["plate"], ap, N_PER_NOM,
                                               A_ssv=0.01, f_ssv=2.0,
                                               T_end=1.0)
res_b = run_all(sim_b, kp_b, a3_b, a4_b, c["lqg"], c["v3"], c["v4"])
m_b = metrics_of(res_b, 1.0)
all_metrics["B   ap=0.3  SSV ±1% @2Hz"] = dict(ap_mm=0.3, delta_pct="±1 sin",
                                               **m_b)
print(f"  B : LQG {m_b['LQG']['y_rms_steady_um']:.4f} µm | "
      f"v3 {m_b['DARC v3']['y_rms_steady_um']:.4f} "
      f"({m_b['DARC v3']['gain_vs_LQG_pct']:+.1f}%) | "
      f"v4 {m_b['DARC v4']['y_rms_steady_um']:.4f} "
      f"({m_b['DARC v4']['gain_vs_LQG_pct']:+.1f}%)  "
      f"conf={m_b['DARC v4'].get('conf_mean', 0):.2f}")


# ============================================================
# Phase C : long pass with offset (position-scheduled reference)
# ============================================================
print("\n" + "="*72)
print(" C. Long pass T = 4 s, δ = +2.5 % (ap = 0.3 mm)")
print("="*72)

sim_c, kp_c, a3_c, a4_c, ph_true_c = setup_const(c["plate"], ap, 80, 4.0)
res_c = run_all(sim_c, kp_c, a3_c, a4_c, c["lqg"], c["v3"], c["v4"])
m_c = metrics_of(res_c, 4.0)
all_metrics["C   ap=0.3  δ=+2.5%  T=4s"] = dict(ap_mm=0.3, delta_pct=2.5,
                                                **m_c)
print(f"  C : LQG {m_c['LQG']['y_rms_steady_um']:.4f} µm | "
      f"v3 {m_c['DARC v3']['y_rms_steady_um']:.4f} "
      f"({m_c['DARC v3']['gain_vs_LQG_pct']:+.1f}%) | "
      f"v4 {m_c['DARC v4']['y_rms_steady_um']:.4f} "
      f"({m_c['DARC v4']['gain_vs_LQG_pct']:+.1f}%)  "
      f"conf={m_c['DARC v4'].get('conf_mean', 0):.2f}")


# ============================================================
# FIGURE 1 : steady-state RMS + gain over the scenario matrix
# ============================================================
labels = list(all_metrics.keys())
short = [l.split()[0] for l in labels]
x = np.arange(len(labels))
w = 0.26

fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
ax = axes[0]
for i, name in enumerate(("LQG", "DARC v3", "DARC v4")):
    vals = [all_metrics[l][name]["y_rms_steady_um"] for l in labels]
    ax.bar(x + (i-1)*w, vals, w, color=COL[name], alpha=0.88,
           edgecolor="k", linewidth=1.2, label=name)
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=10)
ax.set_ylabel(r"$y_{RMS}$ steady (µm)", fontsize=12)
ax.set_title("Steady-state vibration (t > %.2f s)" % T_STEADY,
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10); ax.grid(True, axis="y", alpha=0.4)

ax = axes[1]
for i, name in enumerate(("DARC v3", "DARC v4")):
    vals = [all_metrics[l][name]["gain_vs_LQG_pct"] for l in labels]
    b = ax.bar(x + (i-0.5)*w, vals, w, color=COL[name], alpha=0.88,
               edgecolor="k", linewidth=1.2, label=name)
    for bar, v in zip(b, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                v + (0.15 if v >= 0 else -0.45), f"{v:+.1f}",
                ha="center", fontsize=8.5, fontweight="bold")
ax.axhline(0, color="k", linewidth=1)
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=10)
ax.set_ylabel(r"RMS reduction vs LQG (%)", fontsize=12)
ax.set_title("Feedforward benefit vs LQG baseline",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10); ax.grid(True, axis="y", alpha=0.4)

plt.suptitle("Spindle-speed uncertainty: open-loop clock FF (v3) vs "
             "phase-locked FF (v4 PLAD)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig01_gap_summary.png", dpi=170, bbox_inches="tight")
plt.close()
print(f"\n  ✓ fig01_gap_summary.png")


# ============================================================
# FIGURE 2 : rolling RMS, scenarios A3 (δ=+2.5%) and B (SSV)
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(13, 7.6), sharex=False)

sim_a3, res_a3, _, _ = saved["A3  ap=0.3  δ=+2.50%"]
for ax, (ttl, sim_i, res_i) in zip(
        axes,
        [("A3 : constant offset δ = +2.5 %", sim_a3, res_a3),
         ("B : sinusoidal fluctuation ±1 % @ 2 Hz", sim_b, res_b)]):
    for name in ("LQG", "DARC v3", "DARC v4"):
        rr = rolling_rms(res_i[name]["y"])*1e6
        ax.plot(sim_i.t_vec, rr, color=COL[name], linewidth=1.3, label=name)
    ax.axvspan(0, T_STEADY, color="gray", alpha=0.12)
    ax.text(T_STEADY/2, ax.get_ylim()[0], " transient\n excluded",
            fontsize=8, va="bottom", ha="center", color="gray")
    ax.set_ylabel("rolling RMS $y$ (µm, 20 ms)", fontsize=11)
    ax.set_title(ttl, fontsize=11, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right"); ax.grid(True, alpha=0.4)
axes[-1].set_xlabel("time (s)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig02_rolling_rms.png", dpi=170, bbox_inches="tight")
plt.close()
print(f"  ✓ fig02_rolling_rms.png")


# ============================================================
# FIGURE 3 : v4 phase-observer diagnostics (scenario B)
# ============================================================
r4 = res_b["DARC v4"]
n_b = len(r4["conf"])
t_b = sim_b.t_vec[1:n_b+1]

fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

ax = axes[0]
f_true_b = np.gradient(np.unwrap(ph_true_b), DT)/(2*np.pi)
# smooth the numerically-differentiated true frequency for display
kern = np.ones(400)/400
f_true_s = np.convolve(f_true_b, kern, mode="same")
ax.plot(sim_b.t_vec, f_true_s, "k--", linewidth=1.4,
        label="true tooth-passing freq")
ax.plot(t_b, r4["omega_hat"]/(2*np.pi), color=COL["DARC v4"],
        linewidth=1.4, label=r"PLL estimate $\hat f$")
ax.set_ylabel("frequency (Hz)", fontsize=11)
ax.set_title("B : sensorless tracking of the tooth-passing frequency",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10); ax.grid(True, alpha=0.4)

ax = axes[1]
err4 = np.angle(np.exp(1j*(r4["phase_used"] - ph_true_b[1:n_b+1])))
ph_v3 = res_b["DARC v3"]["phase_used"]
err3 = np.angle(np.exp(1j*(ph_v3 - ph_true_b[1:len(ph_v3)+1])))
ax.plot(sim_b.t_vec[1:len(ph_v3)+1], np.degrees(err3), color=COL["DARC v3"],
        linewidth=1.0, label="v3 open-loop clock error")
ax.plot(t_b, np.degrees(err4), color=COL["DARC v4"], linewidth=1.2,
        label="v4 PLL phase error")
ax.axhline(0, color="k", linewidth=0.8)
ax.set_ylabel("clock phase error (deg)", fontsize=11)
ax.legend(fontsize=10); ax.grid(True, alpha=0.4)

ax = axes[2]
ax.plot(t_b, r4["conf"], color="#7A28A8", linewidth=1.4,
        label="lock confidence c(t)")
ax.plot(t_b, r4["alpha_eff"], color="#E8A000", linewidth=1.2,
        linestyle="--", label=r"FF gain $\alpha_{eff}(t)$")
ax.set_ylim([-0.05, 1.1])
ax.set_xlabel("time (s)", fontsize=11)
ax.set_ylabel("gating (–)", fontsize=11)
ax.legend(fontsize=10); ax.grid(True, alpha=0.4)

plt.suptitle("DARC v4 PLAD internal diagnostics (scenario B)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig03_v4_diagnostics.png", dpi=170,
            bbox_inches="tight")
plt.close()
print(f"  ✓ fig03_v4_diagnostics.png")


# ============================================================
# FIGURE 4 : feedforward voltage phase drift (scenario A3)
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(13, 6.8), sharex=True)
r3a = res_a3["DARC v3"]; r4a = res_a3["DARC v4"]
n_show0, n_show1 = int(0.30/DT), int(0.36/DT)
sl = slice(n_show0, n_show1)
t_sl = sim_a3.t_vec[1:len(r3a["u_ff"])+1][sl]

ax = axes[0]
ax.plot(t_sl, r3a["u_ff"][sl], color=COL["DARC v3"], linewidth=1.1,
        label="v3 $u_{FF}$ (desynchronised)")
ax.plot(t_sl, r4a["u_ff"][sl], color=COL["DARC v4"], linewidth=1.1,
        label="v4 $u_{FF}$ (phase-locked)")
ax.set_ylabel("$u_{FF}$ (V)", fontsize=11)
ax.set_title("A3 (δ = +2.5 %) : feedforward voltage, 60 ms window",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10); ax.grid(True, alpha=0.4)

ax = axes[1]
ax.plot(t_sl, res_a3["LQG"]["y"][sl]*1e6, color=COL["LQG"], linewidth=0.9,
        label="LQG")
ax.plot(t_sl, res_a3["DARC v3"]["y"][sl]*1e6, color=COL["DARC v3"],
        linewidth=0.9, label="DARC v3")
ax.plot(t_sl, res_a3["DARC v4"]["y"][sl]*1e6, color=COL["DARC v4"],
        linewidth=0.9, label="DARC v4")
ax.set_xlabel("time (s)", fontsize=11)
ax.set_ylabel("$y_p$ (µm)", fontsize=11)
ax.legend(fontsize=10); ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig04_ff_drift.png", dpi=170, bbox_inches="tight")
plt.close()
print(f"  ✓ fig04_ff_drift.png")


# ============================================================
# FIGURE 5 : long pass (scenario C)
# ============================================================
fig, ax = plt.subplots(figsize=(13, 4.6))
for name in ("LQG", "DARC v3", "DARC v4"):
    rr = rolling_rms(res_c[name]["y"], win_s=0.05)*1e6
    ax.plot(sim_c.t_vec, rr, color=COL[name], linewidth=1.2, label=name)
ax2 = ax.twinx()
ax2.plot(sim_c.t_vec[1:len(res_c['DARC v4']['conf'])+1],
         res_c["DARC v4"]["conf"], color="#7A28A8", linewidth=1.0,
         alpha=0.6, label="v4 lock confidence")
ax2.set_ylabel("v4 lock confidence (–)", fontsize=10, color="#7A28A8")
ax2.set_ylim([0, 1.15])
ax.set_xlabel("time (s)", fontsize=11)
ax.set_ylabel("rolling RMS $y$ (µm, 50 ms)", fontsize=11)
ax.set_title("C : long pass (T = 4 s, δ = +2.5 %) — sustained lock while "
             "the tool advances", fontsize=11, fontweight="bold")
ax.legend(fontsize=10, loc="upper right"); ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig05_long_pass.png", dpi=170, bbox_inches="tight")
plt.close()
print(f"  ✓ fig05_long_pass.png")


# ============================================================
# Save metrics + summary
# ============================================================
with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

lines = ["# Spindle-speed uncertainty experiment — summary", "",
         "Steady-state window: t > %.2f s.  Gains are RMS reduction vs "
         "the LQG baseline of the SAME scenario." % T_STEADY, "",
         "| Scenario | ap (mm) | δ speed | LQG (µm) | v3 (µm) | v3 gain | "
         "v4 (µm) | v4 gain | v4 conf |",
         "|---|---|---|---|---|---|---|---|---|"]
for l, m in all_metrics.items():
    lines.append(
        f"| {l} | {m['ap_mm']:.1f} | {m['delta_pct']} | "
        f"{m['LQG']['y_rms_steady_um']:.4f} | "
        f"{m['DARC v3']['y_rms_steady_um']:.4f} | "
        f"{m['DARC v3']['gain_vs_LQG_pct']:+.1f}% | "
        f"{m['DARC v4']['y_rms_steady_um']:.4f} | "
        f"{m['DARC v4']['gain_vs_LQG_pct']:+.1f}% | "
        f"{m['DARC v4'].get('conf_mean', float('nan')):.2f} |")
with open(f"{OUT_DIR}/summary.md", "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\n  ✓ metrics.json, summary.md")
print(f"\nTotal wall time : {time.time()-t_glob:.1f} s")
