"""
main_robustness_mc.py
=====================
Monte-Carlo robustness: LQG vs ESO-ADRC (certified fixed design) vs A-ESO-ADRC
over parametric uncertainty (±15 % cutting constants, ±3 % modal frequencies,
±20 % damping). All controllers designed ONCE on the nominal model and frozen
(A-ESO-ADRC's supervisor runs, but its adaptation state is reset before every
sample). Divergence is reported explicitly — no survivorship bias.

Output: figs_lqg_vs_adrc/fig_robustness_mc.png|pdf + console stats.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from plate_model import PlateModel
from milling_force import cutting_constants
from lqg_controller import LQGController
from adrc_controller import ESO_ADRC_Controller, AdaptiveESO_ADRC_Controller
from newmark_solver import NewmarkSimulator
from uncertainty_analysis import run_mc_controllers

# ---------------- physical parameters (article) ----------------
LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU_AL = 2830, 69e9, 0.33
NT = 3;  DT_TOOL = 0.010
ETA_H = np.deg2rad(35);  GAMMA_N = np.deg2rad(15)
KT_NOMINAL = 925e6;  KN = 0.26;  MU_C = 0.20
RPM = 4900
FT = 0.02e-3;  AE = 0.1e-3
D31 = 175e-12;  H_PA = 0.7e-3;  E_PE = 63e9;  NU_PE = 0.35
N1, N2 = 30, 24
N_MODES_PLANT, N_MODES_CTRL = 5, 3
ZETA_PLANT = [0.0031, 0.0017, 0.0027, 0.0056, 0.0035]
MEAS_NOISE_STD = 1e-8
KALMAN_V = 1e-12
DT = 5e-5
T_END = 0.5
AP = 0.3e-3
N_SAMPLES = 50

OUT_DIR = "figs_lqg_vs_adrc"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- nominal plant + frozen designs ----------------
print("[MC] building nominal plant and frozen controllers...")
plant = PlateModel(LP, HP, BP, RHO, E_AL, NU_AL, N1=N1, N2=N2,
                   n_modes=N_MODES_PLANT, zeta_modes=ZETA_PLANT, verbose=False)
plant.precompute_Dp(zp_pos=HP - AP/2, n_pos=2001)
plant.set_observation(x_obs=LP, z_obs=HP)
plant.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
design = plant.truncated_view(N_MODES_CTRL)

lqg = LQGController(design, dt=DT, verbose=False, kalman_V=KALMAN_V, u_max=150.0)
lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                     w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
lqg.discretize_observer()

# Design tuples from the main_simulation grid (performance / certified; see
# main_simulation.py — regenerate there if the model changes).
PERF_DESIGN = (1e16, 1e8, 3e3)
CERT_DESIGN = (1e14, 1e8, 1e4)
adrc = ESO_ADRC_Controller(design, DT, *CERT_DESIGN, kalman_V=KALMAN_V,
                           verbose=False)
aadrc = AdaptiveESO_ADRC_Controller(design, DT, rungs=(PERF_DESIGN, CERT_DESIGN),
                                    perf_rung=0, robust_rung=1,
                                    kalman_V=KALMAN_V, verbose=False)

# ---------------- MC setup ----------------
PHI_ST = np.pi - np.arccos(1 - AE/(DT_TOOL/2))
PHI_EX = np.pi
OMEGA = 2*np.pi*RPM/60
RT = DT_TOOL/2
TAU = 60/(NT*RPM)
N_PER = int(np.round(TAU/DT))

sim_t = NewmarkSimulator(plant, dt=DT, T_end=T_END, ft=FT, tau=TAU, verbose=False)
v_feed = FT * NT * RPM / 60
xp_path = np.minimum(v_feed * sim_t.t_vec, LP)
kp_idx = np.clip(np.round(xp_path/LP * 2000).astype(int), 0, 2000)

nominal_params = dict(
    kt=KT_NOMINAL, kn=KN, mu_c=MU_C,
    omega_n=plant.omega_n.copy(), zeta=np.array(ZETA_PLANT),
    eta_h=ETA_H, gamma_n=GAMMA_N, tau=TAU, Omega=OMEGA, NT=NT, RT=RT,
    phi_st=PHI_ST, phi_ex=PHI_EX, za_low=HP-AP, za_high=HP,
)

UNC = dict(kt_pct=0.15, kn_pct=0.15, mu_c_pct=0.15,
           omega_pct=0.03, zeta_pct=0.20, E_pct=0.0)

stats = run_mc_controllers(
    plant, {"LQG": lqg, "ESO-ADRC": adrc, "A-ESO-ADRC": aadrc},
    nominal_params, kp_idx, DT, T_END, FT, TAU, N_PER,
    unc=UNC, n_samples=N_SAMPLES, meas_noise_std=MEAS_NOISE_STD,
    seed=42, verbose=True)

# ---------------- report ----------------
print("\n=== Monte-Carlo summary ===")
for nm in stats['names']:
    r = stats['rms'][nm][stats['conv'][nm]]
    print(f"  {nm:<12} converged {stats['n_conv'][nm]}/{stats['n_samples']}, "
          f"RMS median {np.median(r):.4f} µm "
          f"[p05 {np.percentile(r, 5):.4f}, p95 {np.percentile(r, 95):.4f}]")
for nm, gs in stats['gain_stats'].items():
    print(f"  {nm} vs {stats['baseline']}: median {gs['median']:+.2f} % "
          f"[p05 {gs['p05']:+.2f}, p95 {gs['p95']:+.2f}], "
          f"better in {gs['pct_better']:.0f} % of samples "
          f"(n_both={gs['n_both']})")

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
COL = {"LQG": '#2E8B57', "ESO-ADRC": '#1E5AA8', "A-ESO-ADRC": '#DC143C'}

ax = axes[0]
data = [stats['rms'][nm][stats['conv'][nm]] for nm in stats['names']]
try:
    bp = ax.boxplot(data, tick_labels=stats['names'], patch_artist=True,
                    showmeans=True)
except TypeError:                                   # matplotlib < 3.9
    bp = ax.boxplot(data, labels=stats['names'], patch_artist=True,
                    showmeans=True)
for patch, nm in zip(bp['boxes'], stats['names']):
    patch.set_facecolor(COL[nm]); patch.set_alpha(0.6)
for i, nm in enumerate(stats['names']):
    n_div = stats['n_samples'] - stats['n_conv'][nm]
    ax.text(i+1, ax.get_ylim()[1]*0.95,
            f"{n_div} div." if n_div else "0 div.",
            ha='center', fontsize=10,
            color=('red' if n_div else 'green'), fontweight='bold')
ax.set_ylabel("y_RMS (µm) — converged samples", fontsize=12)
ax.set_title(f"Monte-Carlo ({stats['n_samples']} samples)\n"
             "±15% cutting, ±3% freq, ±20% damping",
             fontsize=12, fontweight='bold')
ax.grid(True, axis='y', alpha=0.5)

ax = axes[1]
for nm in stats['names'][1:]:
    g = stats['gains'][nm]
    g = g[~np.isnan(g)]
    ax.hist(g, bins=15, alpha=0.6, color=COL[nm],
            label=f"{nm} vs LQG (median {np.median(g):+.1f} %)",
            edgecolor='k')
ax.axvline(0, color='k', linewidth=1.5)
ax.set_xlabel("RMS gain vs LQG (%) — positive = better than LQG", fontsize=12)
ax.set_ylabel("count", fontsize=12)
ax.set_title("Held-out gain distribution over the uncertainty ball",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.5)

plt.suptitle("Robustness Monte-Carlo — frozen designs, divergence reported",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_robustness_mc.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig_robustness_mc.pdf", bbox_inches='tight')
plt.close()
print(f"\n  ✓ {OUT_DIR}/fig_robustness_mc.png|pdf")
