"""
main_adaptive_removal.py
========================
MATERIAL-REMOVAL drift benchmark: fixed controllers vs A-PALF-LQG (online
parameter updating at every step).

The plant's modal frequencies DRIFT DURING THE PASS (omega_scale_t schedule in the
Newmark solver — stiffness Kp*s(t)^2, damping Cp*s(t)), emulating progressive
material removal. Du et al. (2024) measured +9 %/+17 % frequency increase over a
machining series; this package's fixed-controller margin is ~-9 %. Scenarios:

  R1  ramp 1.00 -> 1.10  (+10 % — the article's direction and magnitude)
  R2  ramp 1.00 -> 0.90  (-10 % — crosses the fixed-design stability margin)
  R3  no drift            (sanity: adaptation must not hurt on the nominal plant)

Controllers (all designed/trained ONCE on the nominal model, then frozen):
  LQG        fixed feedback
  PALF-LQG   fixed feedback + phase-locked learned feedforward
  A-PALF-LQG same + per-step probe-FRF parameter tracking (sliding DFT at
             non-harmonic probe lines, model-grid matching) and dwell-limited
             gain/observer rescheduling

Run: python main_adaptive_removal.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from plate_model import PlateModel
from milling_force import precompute_alpha_periodic, cutting_constants
from lqg_controller import LQGController
from palf_lqg_controller import PALF_LQG_Controller
from adaptive_palf_lqg_controller import AdaptivePALF_LQG_Controller
from newmark_solver import NewmarkSimulator

# --- Parameters (identical to main_simulation) ---
LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU_AL = 2830, 69e9, 0.33
NT = 3;  DT_TOOL = 0.010
ETA_H = np.deg2rad(35);  GAMMA_N = np.deg2rad(15)
KT_NOMINAL = 925e6;  KN = 0.26;  MU_C = 0.20
RPM = 4900;  FT = 0.02e-3;  AE = 0.1e-3
D31 = 175e-12;  H_PA = 0.7e-3;  E_PE = 63e9;  NU_PE = 0.35
N1, N2 = 30, 24
N_MODES_PLANT, N_MODES_CTRL = 5, 3
ZETA_PLANT = [0.0031, 0.0017, 0.0027, 0.0056, 0.0035]
MEAS_NOISE_STD, KALMAN_V = 1e-8, 1e-12
DT, T_END = 5e-5, 0.5
AP = 0.3e-3
OUT_DIR = "figs_lqg_vs_palf"
os.makedirs(OUT_DIR, exist_ok=True)

RT = DT_TOOL / 2
PHI_ST = np.pi - np.arccos(1 - AE / RT);  PHI_EX = np.pi
OMEGA = 2 * np.pi * RPM / 60;  TAU = 60 / (NT * RPM)
K1, K2 = cutting_constants(KN, MU_C, ETA_H, GAMMA_N)
N_PER = int(np.round(TAU / DT))

# --- Nominal plant + one-time controller design/training -------------------
print("[design] nominal plant, shared LQG, PALF ILC, A-PALF bank...")
plant = PlateModel(LP, HP, BP, RHO, E_AL, NU_AL, N1=N1, N2=N2,
                   n_modes=N_MODES_PLANT, zeta_modes=ZETA_PLANT, verbose=False)
plant.precompute_Dp(zp_pos=HP - AP / 2, n_pos=2001)
plant.set_observation(x_obs=LP, z_obs=HP)
plant.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
design = plant.truncated_view(N_MODES_CTRL)

sim = NewmarkSimulator(plant, dt=DT, T_end=T_END, ft=FT, tau=TAU, verbose=False)
v_feed = FT * NT * RPM / 60
kp_idx = np.clip(np.round(np.minimum(v_feed * sim.t_vec, LP) / LP * 2000).astype(int),
                 0, 2000)
a3, a4 = precompute_alpha_periodic(DT, N_PER, sim.nstep, OMEGA, NT, RT, ETA_H,
                                   PHI_ST, PHI_EX, HP - AP, HP, K1, K2, KT_NOMINAL)

LQG = LQGController(design, dt=DT, verbose=False, kalman_V=KALMAN_V, u_max=150.0)
LQG.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                     w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
LQG.discretize_observer()

PALF = PALF_LQG_Controller(design, dt=DT, base_w_q=LQG.w_q, base_w_qd=LQG.w_qd,
                           base_w_r=1.0, ff_lr=0.005, ff_max=10.0, ff_alpha=1.0,
                           n_per=N_PER, safety_alpha=5.0, kalman_V=KALMAN_V,
                           u_max=150.0, verbose=False)
PALF.pretrain_iterative_simulation(sim, a3, a4, kp_idx,
                                   n_iterations=10, n_epochs_per_iter=30,
                                   verbose=False)

APALF = AdaptivePALF_LQG_Controller(design, dt=DT,
                                    base_w_q=LQG.w_q, base_w_qd=LQG.w_qd,
                                    base_w_r=1.0, ff_lr=0.005, ff_max=10.0,
                                    ff_alpha=1.0, n_per=N_PER, safety_alpha=5.0,
                                    kalman_V=KALMAN_V, u_max=150.0,
                                    theta_grid=(0.86, 0.90, 0.94, 0.98, 1.00,
                                                1.04, 1.08, 1.12, 1.16),
                                    verbose=True)
# Identical frozen feedforward for PALF and A-PALF (copy trained weights)
for w in ('W1', 'b1', 'W2', 'b2'):
    setattr(APALF.ff_nn, w, getattr(PALF.ff_nn, w).copy())

# --- Drift scenarios: ramp over the first 60 % of the pass, then HOLD -------
# (a sustained level is the realistic end-state of a machining series; the article
#  measured +9 %/+17 %; the fixed-design stability margin here is ~-9 %)
t = sim.t_vec


def ramp_hold(s_end, frac=0.6):
    n_ramp = int(frac * sim.nstep)
    return np.concatenate([np.linspace(1.0, s_end, n_ramp),
                           np.full(sim.nstep - n_ramp, s_end)])


scenarios = [
    ("R1  removal +15% (hold)", ramp_hold(1.15)),
    ("R2  removal -12% (hold)", ramp_hold(0.88)),
    ("R3  no drift (sanity)",   None),
]
controllers = [("LQG", LQG), ("PALF-LQG", PALF), ("A-PALF-LQG", APALF)]

results = {}
print("\n" + "=" * 74)
print(f"  {'Scenario':<26}{'LQG':>14}{'PALF-LQG':>14}{'A-PALF-LQG':>14}")
print("-" * 74)
for sname, sched in scenarios:
    row = {}
    for cname, ctrl in controllers:
        if hasattr(ctrl, 'reset_adaptation'):
            ctrl.reset_adaptation()
        elif hasattr(ctrl, 'history_u_lqg'):
            for h in ('history_u_lqg', 'history_u_ff', 'history_u_total',
                      'history_phase', 'history_safety'):
                setattr(ctrl, h, [])
        r = sim.simulate(a3, a4, kp_idx, controller=ctrl,
                         meas_noise_std=MEAS_NOISE_STD,
                         rng=np.random.default_rng(1234),
                         omega_scale_t=sched, progress=False)
        i = r['stop_idx']
        rms = np.sqrt(np.mean(r['y'][:i + 1]**2)) * 1e6
        div = i < sim.nstep - 2
        row[cname] = dict(rms=rms, div=div, res=r,
                          theta=(list(APALF.history_theta)
                                 if cname == "A-PALF-LQG" else None))
    results[sname] = row
    fmt = lambda d: f"{d['rms']:9.3f}um" + ("*" if d['div'] else " ")
    print(f"  {sname:<26}{fmt(row['LQG']):>14}{fmt(row['PALF-LQG']):>14}"
          f"{fmt(row['A-PALF-LQG']):>14}")
print("-" * 74)
print("  (* = diverged; RMS up to the stop index)")
for sname, _ in scenarios[:2]:
    rL = results[sname]['PALF-LQG'];  rA = results[sname]['A-PALF-LQG']
    if not rA['div'] and not rL['div']:
        print(f"  {sname}: A-PALF vs fixed PALF gain = "
              f"{(1 - rA['rms'] / rL['rms']) * 100:+.2f}%")

# --- Figure: responses + tracked theta --------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(15, 11),
                         gridspec_kw={'width_ratios': [2.2, 1.0]})
colors = {"LQG": '#2E8B57', "PALF-LQG": '#DC143C', "A-PALF-LQG": '#1E63C8'}
for r_i, (sname, sched) in enumerate(scenarios):
    ax = axes[r_i, 0]
    for cname, _ in controllers:
        d = results[sname][cname]
        i = d['res']['stop_idx']
        ax.plot(t[:i + 1] * 1e3, d['res']['y'][:i + 1] * 1e6,
                color=colors[cname], lw=0.5, alpha=0.85,
                label=f"{cname} (RMS {d['rms']:.2f}µm{'*' if d['div'] else ''})")
    ax.set_title(sname, fontweight='bold')
    ax.set_ylabel("$y_p$ (µm)");  ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.4)
    if r_i == 2:
        ax.set_xlabel("time (ms)")

    ax = axes[r_i, 1]
    if sched is not None:
        ax.plot(t * 1e3, sched, 'k--', lw=1.5, label="true scale $s(t)$")
    else:
        ax.axhline(1.0, color='k', ls='--', lw=1.5, label="true scale $s(t)$")
    th = results[sname]['A-PALF-LQG']['theta']
    if th:
        ax.plot(np.arange(len(th)) * DT * 1e3, th, color=colors['A-PALF-LQG'],
                lw=1.5, label=r"tracked $\hat{\theta}(t)$")
    ax.set_ylim(0.85, 1.15);  ax.set_ylabel(r"$\omega/\omega_{nom}$")
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.4)
    if r_i == 2:
        ax.set_xlabel("time (ms)")
plt.suptitle("Material-removal drift: fixed LQG / PALF vs A-PALF-LQG "
             "(per-step probe-FRF parameter update)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_adaptive_removal.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig_adaptive_removal.pdf", bbox_inches='tight')
plt.close()
print(f"\n  >>> fig_adaptive_removal saved in {OUT_DIR}/")
