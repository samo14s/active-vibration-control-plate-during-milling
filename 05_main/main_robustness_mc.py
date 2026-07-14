"""
main_robustness_mc.py
=====================
Monte-Carlo robustness comparison of the FROZEN LQG and PALF-LQG controllers over
parametric uncertainty (per-mode omega/zeta + cutting kt/kn/mu_c), on the 5-mode
plant with spillover and measurement noise — the P2 "wire the Monte Carlo module in"
deliverable. Both controllers are designed once on the nominal model (held-out) and
evaluated unchanged on every random sample. Divergence is reported explicitly (no
survivorship bias).

Run: python main_robustness_mc.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from plate_model import PlateModel
from milling_force import precompute_alpha_periodic, cutting_constants
from lqg_controller import LQGController
from palf_lqg_controller import PALF_LQG_Controller
from newmark_solver import NewmarkSimulator
from uncertainty_analysis import run_mc_lqg_vs_palf

# --- Physical / cutting parameters (same as main_simulation) ---
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
T_END_MC = 0.25          # shorter horizon per MC sample (enough for the RMS ratio)
N_SAMPLES = 50
AP = 0.3e-3
OUT_DIR = "figs_lqg_vs_palf"
os.makedirs(OUT_DIR, exist_ok=True)

RT = DT_TOOL / 2
PHI_ST = np.pi - np.arccos(1 - AE / RT);  PHI_EX = np.pi
OMEGA = 2 * np.pi * RPM / 60;  TAU = 60 / (NT * RPM)
K1, K2 = cutting_constants(KN, MU_C, ETA_H, GAMMA_N)
N_PER = int(np.round(TAU / DT))

# --- Nominal plant (5 modes) + reduced controller design (3 modes) ---
print("[setup] Building nominal plant and training controllers (held-out)...")
plant = PlateModel(LP, HP, BP, RHO, E_AL, NU_AL, N1=N1, N2=N2,
                   n_modes=N_MODES_PLANT, zeta_modes=ZETA_PLANT, verbose=False)
plant.precompute_Dp(zp_pos=HP - AP / 2, n_pos=2001)
plant.set_observation(x_obs=LP, z_obs=HP)
plant.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
design = plant.truncated_view(N_MODES_CTRL)

sim = NewmarkSimulator(plant, dt=DT, T_end=T_END, ft=FT, tau=TAU, verbose=False)
v_feed = FT * NT * RPM / 60
kp_idx = np.clip(np.round(np.minimum(v_feed * sim.t_vec, LP) / LP * 2000).astype(int), 0, 2000)
a3_nom, a4_nom = precompute_alpha_periodic(DT, N_PER, sim.nstep, OMEGA, NT, RT, ETA_H,
                                           PHI_ST, PHI_EX, HP - AP, HP, K1, K2, KT_NOMINAL)

lqg = LQGController(design, dt=DT, verbose=False, kalman_V=KALMAN_V, u_max=150.0)
lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16], w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
lqg.discretize_observer()
palf = PALF_LQG_Controller(design, dt=DT, base_w_q=lqg.w_q, base_w_qd=lqg.w_qd,
                           base_w_r=1.0, ff_lr=0.005, ff_max=10.0, ff_alpha=1.0,
                           n_per=N_PER, safety_alpha=5.0, kalman_V=KALMAN_V,
                           u_max=150.0, verbose=False)
palf.pretrain_iterative_simulation(sim, a3_nom, a4_nom, kp_idx,
                                   n_iterations=30, n_epochs_per_iter=15, verbose=False)

nominal_params = dict(
    omega_n=plant.omega_n.copy(), zeta=np.array(ZETA_PLANT),
    kt=KT_NOMINAL, kn=KN, mu_c=MU_C, eta_h=ETA_H, gamma_n=GAMMA_N,
    Omega=OMEGA, NT=NT, RT=RT, phi_st=PHI_ST, phi_ex=PHI_EX,
    za_low=HP - AP, za_high=HP, tau=TAU,
)
# Uncertainty: +/-15% cutting force, +/-3% modal frequency, +/-20% damping
UNC = dict(kt_pct=0.15, kn_pct=0.15, mu_c_pct=0.15,
           omega_pct=0.03, zeta_pct=0.20, E_pct=0.0)

st = run_mc_lqg_vs_palf(plant, lqg, palf, nominal_params, kp_idx,
                        DT, T_END_MC, FT, TAU, N_PER, unc=UNC, n_samples=N_SAMPLES,
                        meas_noise_std=MEAS_NOISE_STD, seed=42, verbose=True)

print("\n" + "=" * 60)
print(" Monte-Carlo robustness (held-out; divergence reported)")
print("=" * 60)
print(f"  Converged: LQG {st['n_conv_lqg']}/{st['n_samples']}, "
      f"PALF {st['n_conv_palf']}/{st['n_samples']}, both {st['n_both']}")
print(f"  RMS gain over both-converged samples:")
print(f"     median {st['gain_median']:+.2f}%   mean {st['gain_mean']:+.2f}%   "
      f"[p05 {st['gain_p05']:+.2f}%, p95 {st['gain_p95']:+.2f}%]")
print(f"  PALF better than LQG in {st['pct_palf_better']:.0f}% of both-converged samples")

# Figure: gain distribution + RMS scatter
both = st['both_converged']
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].hist(st['gains'][both], bins=15, color='#DC143C', alpha=0.8, edgecolor='k')
ax[0].axvline(0, color='k', lw=1)
ax[0].axvline(st['gain_median'], color='blue', lw=2, ls='--',
              label=f"median {st['gain_median']:+.1f}%")
ax[0].set_xlabel("RMS gain PALF vs LQG (%)"); ax[0].set_ylabel("count")
ax[0].set_title(f"Gain distribution ({st['n_both']} both-converged samples)")
ax[0].legend(); ax[0].grid(True, alpha=0.4)

ax[1].plot([0, 3], [0, 3], 'k:', alpha=0.5)
ax[1].scatter(st['rms_lqg'][both], st['rms_palf'][both], c='#DC143C', alpha=0.7,
              edgecolors='k', s=40)
ax[1].set_xlabel("LQG y_RMS (µm)"); ax[1].set_ylabel("PALF y_RMS (µm)")
ax[1].set_title("Below the diagonal = PALF better")
ax[1].grid(True, alpha=0.4); ax[1].set_aspect('equal', 'box')

plt.suptitle("Monte-Carlo robustness: LQG vs PALF-LQG", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_robustness_mc.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"\n  >>> fig_robustness_mc.png saved in {OUT_DIR}/")
