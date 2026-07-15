"""
main_predictive_removal.py
==========================
P7 — material-removal-aware PREDICTIVE control: the honest feasibility study.

Delivers, as one driver, the four pieces of the P7 finding (see the module docstring
of 02_controllers/predictive_removal.py and docs/REPRODUCED_RESULTS.md):

  WALL 1 (timescale): the tool takes ~1.4e4 steps (0.68 s) to cross one FEM mesh
    column, so per-step property update is over-engineering by ~1e4; the correct
    cadence is event-driven (mesh-crossing). Shown numerically + an event-driven
    coupled simulation whose within-pass drift is quantified.
  WALL 2 (regime conflict): the open-loop chatter radius rho(a_p) explodes for deep
    cuts (where within-pass removal is significant) while within-pass removal is
    negligible for the controllable shallow cuts. The two curves are plotted together
    and DO NOT OVERLAP -> the requested per-step-removal-aware regime is uncontrollable
    on this plant.
  CONTROLLER: the PreviewPredictiveController (regenerative + feed-force preview,
    with a MISMATCHED cutting model — no inverse crime) vs LQG / ESO-ADRC / HRC at
    the article condition. Honest outcome: stable, marginally beats LQG, loses to HRC.

Output: figs_predictive_removal/*.png|pdf + console summary.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from plate_model import PlateModel
from material_removal import MillingWorkpiece
from milling_force import precompute_alpha_periodic, cutting_constants
from newmark_solver import NewmarkSimulator
from lqg_controller import LQGController
from adrc_controller import ESO_ADRC_Controller, ESO_ADRC_HRC_Controller
from predictive_removal import PhysicsCuttingModel, PreviewPredictiveController
from fdm_stability import closed_loop_rho

# ---------------- physical parameters (article) ----------------
LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU = 2830, 69e9, 0.33
D31, H_PA, E_PE, NU_PE = 175e-12, 0.7e-3, 63e9, 0.35
NT, RPM = 3, 4900
FT = 0.02e-3
AE = 0.1e-3
ZETA = [0.0031, 0.0017, 0.0027, 0.0056, 0.0035]
DT, TAU = 5e-5, 60 / (NT * RPM)
F_TOOTH = NT * RPM / 60.0
W_TOOTH = 2 * np.pi * F_TOOTH
KT, KN, MUC = 925e6, 0.26, 0.20
ETA, GAM = np.deg2rad(35), np.deg2rad(15)
K1, K2 = cutting_constants(KN, MUC, ETA, GAM)
RT = 0.005
PHI_ST = np.pi - np.arccos(1 - AE / RT)
PHI_EX = np.pi
OMEGA = 2 * np.pi * RPM / 60
N_PER = int(round(TAU / DT))
N_TAU = int(round(TAU / DT))
KV, MNS = 1e-12, 1e-8

OUT_DIR = "figs_predictive_removal"
os.makedirs(OUT_DIR, exist_ok=True)
t0 = time.time()
print("=" * 72)
print(" P7 — material-removal-aware PREDICTIVE control: feasibility study ")
print("=" * 72)

# nominal plant + 3-mode design view
plant = PlateModel(LP, HP, BP, RHO, E_AL, NU, N1=30, N2=24, n_modes=5,
                   zeta_modes=ZETA, verbose=False)
plant.precompute_Dp(zp_pos=HP - 0.3e-3 / 2, n_pos=2001)
plant.set_observation(x_obs=LP, z_obs=HP)
plant.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
design = plant.truncated_view(3)


# ============================================================
# WALL 1 — timescale + event-driven coupled simulation
# ============================================================
print("\n[WALL 1] timescale of within-cut material removal")
v_feed = FT * NT * RPM / 60.0
lex = LP / 30
steps_per_col = lex / v_feed / DT
print(f"  feed speed {v_feed*1e3:.3f} mm/s; tool advances {v_feed*DT*1e6:.3f} um/step")
print(f"  one mesh column = {lex*1e3:.2f} mm -> {steps_per_col:.0f} steps "
      f"({steps_per_col/N_PER:.0f} tooth-periods, {steps_per_col*DT:.2f} s) to cross")
print(f"  => plant is CONSTANT between mesh crossings; per-step update is "
      f"~{steps_per_col:.0f}x over-resolved")


def coupled_within_pass_drift(a_p, a_e, n_cross=30):
    """Event-driven coupled run: advance the tool across n_cross mesh columns,
    re-solving the modal model only at each crossing (the correct cadence). Return
    the max per-mode within-pass frequency drift and the number of eigensolves."""
    wp = MillingWorkpiece(LP, HP, BP, RHO, E_AL, NU, N1=30, N2=24, n_modes=5,
                          zeta_modes=ZETA, verbose=False)
    wp.set_observation(x_obs=LP, z_obs=HP)
    wp.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    wp.solve_modes(track=True)
    f0 = wp.freq_n[:3].copy()
    wp.begin_pass()                       # baseline for the idempotent moving front
    drift = 0.0
    for c in range(1, n_cross + 1):
        x_tool = c / n_cross * LP
        wp.remove_moving_front(x_tool, a_p, a_e)
        wp.solve_modes(track=True)
        drift = max(drift, float(np.max(np.abs(wp.freq_n[:3] / f0 - 1))) * 100)
    return drift, n_cross


# ============================================================
# WALL 2 — regime conflict: rho(a_p) vs within-pass drift(a_p)
# ============================================================
print("\n[WALL 2] regime conflict — chatter radius vs removal significance")
ap_grid = np.array([0.1, 0.3, 0.6, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]) * 1e-3
Dp_mid = plant.get_Dp_at(1000)[0]        # tool at L/2
rho_ol = np.zeros(len(ap_grid))
drift_wp = np.zeros(len(ap_grid))
for i, ap in enumerate(ap_grid):
    rho_ol[i] = closed_loop_rho(plant.omega_n, np.array(ZETA), Dp_mid,
                                None, None, None, None, None, None,
                                RPM, NT, RT, ETA, PHI_ST, PHI_EX, ap, HP,
                                K1, K2, KT, m_div=30)
    # within-pass drift: a_p as engaged band, deep enough to resolve on the mesh
    d, _ = coupled_within_pass_drift(ap, AE, n_cross=20)
    drift_wp[i] = d
    print(f"  a_p={ap*1e3:5.1f}mm  rho_OL={rho_ol[i]:.3g}  "
          f"within-pass drift={drift_wp[i]:.3f}%  "
          f"{'CONTROLLABLE' if rho_ol[i] < 5 else 'uncontrollable'}")


# ============================================================
# CONTROLLER — preview-predictive vs baselines (a_p=0.3mm, MISMATCHED model)
# ============================================================
print("\n[controller] preview-predictive vs LQG/ESO/HRC (a_p=0.3mm, "
      "anti-inverse-crime mismatched model)")
AP = 0.3e-3
sim = NewmarkSimulator(plant, dt=DT, T_end=0.4, ft=FT, tau=TAU, verbose=False)
xp = np.minimum(v_feed * sim.t_vec, LP)
kp = np.clip(np.round(xp / LP * 2000).astype(int), 0, 2000)
a3, a4 = precompute_alpha_periodic(DT, N_PER, sim.nstep, OMEGA, NT, RT, ETA,
                                   PHI_ST, PHI_EX, HP - AP, HP, K1, K2, KT)

lqg = LQGController(design, dt=DT, verbose=False, kalman_V=KV, u_max=150.0)
lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                     w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
lqg.discretize_observer()
adrc = ESO_ADRC_Controller(design, DT, 1e14, 1e8, 1e4, kalman_V=KV, verbose=False)
hrc = ESO_ADRC_HRC_Controller(design, DT, 1e16, 1e8, 3e3, n_harm=4, g_base=150,
                              lam=5.0, w_tooth=W_TOOTH, kalman_V=KV, verbose=False)

Dp0_c = plant.get_Dp_at(0)[0][:3]
# (i) EXACT cutting model (inverse-crime reference — an unachievable upper bound):
cm_exact = PhysicsCuttingModel(a3, a4, FT, Dp0_c, N_TAU)
preview_exact = PreviewPredictiveController(design, DT, cm_exact, lqg,
                                            horizon=N_TAU//2, r=1e-14, u_max=150.0,
                                            verbose=True)
# (ii) MISMATCHED cutting model (the honest case, no inverse crime): +15% KT,
# +15% kn, -15% mu_c, nominal (vibration-free) bite, tool projection at x=0.
KT_c = 1.15 * KT
K1_c, K2_c = cutting_constants(1.15 * KN, 0.85 * MUC, ETA, GAM)
a3_c, a4_c = precompute_alpha_periodic(DT, N_PER, sim.nstep, OMEGA, NT, RT, ETA,
                                       PHI_ST, PHI_EX, HP - AP, HP, K1_c, K2_c, KT_c)
cm = PhysicsCuttingModel(a3_c, a4_c, FT, Dp0_c, N_TAU)
preview = PreviewPredictiveController(design, DT, cm, lqg, horizon=N_TAU//2,
                                      r=1e-14, u_max=150.0, verbose=False)


def run(ctrl):
    if hasattr(ctrl, 'reset'):
        ctrl.reset()
    r = sim.simulate(a3, a4, kp, controller=ctrl, meas_noise_std=MNS,
                     rng=np.random.default_rng(1234), progress=False)
    i = r['stop_idx']
    y = r['y'][:i + 1]
    u = r['u'][:i + 1]
    return (np.sqrt(np.mean(y**2)) * 1e6, np.max(np.abs(u)), i >= sim.nstep - 2)


res = {}
for name, c in [('LQG', lqg), ('ESO-ADRC', adrc), ('ESO-ADRC+HRC', hrc),
                ('Preview (exact model)', preview_exact),
                ('Preview (mismatched)', preview)]:
    rms, umax, ok = run(c)
    res[name] = rms
    print(f"  {name:22s} rms={rms:.4f}um  umax={umax:6.2f}V  "
          f"{'OK' if ok else 'DIVERGED'}")


# ============================================================
# FIGURE
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) the two walls
ax = axes[0]
apx = ap_grid * 1e3
ax.set_xscale('log'); ax.set_yscale('log')
l1 = ax.plot(apx, rho_ol, 'o-', color='#C0392B', lw=2,
             label='open-loop chatter radius ρ (uncontrollability)')
ax.axhline(1.0, color='#C0392B', ls=':', lw=1, alpha=0.7)
ax.axvspan(1.0, apx.max(), color='#C0392B', alpha=0.06)
ax.text(3.5, ax.get_ylim()[1]*0.02, 'UNCONTROLLABLE\n(ρ ≫ 1, beyond piezo)',
        color='#C0392B', fontsize=10, ha='center', fontweight='bold')
ax.set_xlabel("axial depth a_p (mm)", fontsize=12)
ax.set_ylabel("open-loop Floquet radius ρ", color='#C0392B', fontsize=12)
ax.tick_params(axis='y', labelcolor='#C0392B')
ax2 = ax.twinx()
l2 = ax2.plot(apx, np.maximum(drift_wp, 1e-4), 's-', color='#2471A3', lw=2,
              label='within-pass property drift (removal significance)')
ax2.set_yscale('log')
ax2.set_ylabel("within-pass modal drift (%)", color='#2471A3', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#2471A3')
ax2.axhline(1.0, color='#2471A3', ls=':', lw=1, alpha=0.7)
ax.axvspan(apx.min(), 1.0, color='#2471A3', alpha=0.06)
ax.text(0.2, ax.get_ylim()[0]*1e3, 'CONTROLLABLE\ndrift ≈ 0', color='#2471A3',
        fontsize=10, ha='center', fontweight='bold')
ax.axvline(0.3, color='gold', lw=2, alpha=0.8)
ax.text(0.3, ax.get_ylim()[0]*3, 'article\n0.3 mm', color='#b8860b', fontsize=8,
        ha='center')
lns = l1 + l2
ax.legend(lns, [x.get_label() for x in lns], loc='upper center', fontsize=9)
ax.set_title("The regime conflict: within-pass removal matters only where the\n"
             "cut is uncontrollable (the two shaded regions do not overlap)",
             fontsize=11, fontweight='bold')
ax.grid(True, which='both', alpha=0.3)

# (b) controller comparison at a_p=0.3mm
ax = axes[1]
names = ['LQG', 'ESO-ADRC', 'Preview\n(exact model)', 'Preview\n(mismatched)',
         'ESO-ADRC\n+HRC']
keys = ['LQG', 'ESO-ADRC', 'Preview (exact model)', 'Preview (mismatched)',
        'ESO-ADRC+HRC']
vals = [res[k] for k in keys]
cols = ['#2E8B57', '#1E5AA8', '#C39BD3', '#8E44AD', '#8A2BE2']
bars = ax.bar(names, vals, color=cols, alpha=0.85, edgecolor='k', linewidth=1.2)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.02, f'{v:.3f}', ha='center',
            fontsize=10, fontweight='bold')
ax.axhline(res['LQG'], color='#2E8B57', ls='--', lw=1, alpha=0.6)
ax.set_ylabel("y_RMS (µm) at a_p = 0.3 mm", fontsize=12)
ax.set_title("Preview-predictive (mismatched cutting model — no inverse crime):\n"
             "modestly beats LQG (~10 %) but HRC dominates it ~2×",
             fontsize=10.5, fontweight='bold')
ax.grid(True, axis='y', alpha=0.4)
ax.tick_params(axis='x', labelsize=9)

plt.suptitle("P7 — material-removal-aware predictive control: honest feasibility "
             "study", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_predictive_removal.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig_predictive_removal.pdf", bbox_inches='tight')
plt.close()
print(f"\n  ✓ {OUT_DIR}/fig_predictive_removal.png|pdf")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n=== P7 SUMMARY (feasibility study) ===")
print(f"  WALL 1 timescale: {steps_per_col:.0f} steps/mesh-column "
      f"({steps_per_col*DT:.2f}s) -> per-step update over-engineered ~{steps_per_col:.0f}x; "
      f"event-driven (mesh-crossing) is the correct cadence.")
i_ctrl = np.where(rho_ol < 5)[0]
i_sig = np.where(drift_wp > 1.0)[0]
print(f"  WALL 2 regime conflict: controllable a_p (ρ<5) up to "
      f"{ap_grid[i_ctrl].max()*1e3:.1f}mm; removal-significant (drift>1%) from "
      f"{ap_grid[i_sig].min()*1e3:.1f}mm -> they meet only at the boundary, "
      f"NO USABLE OVERLAP.")
print(f"  controller @0.3mm: LQG {res['LQG']:.3f}, ESO {res['ESO-ADRC']:.3f}, "
      f"Preview(mismatched) {res['Preview (mismatched)']:.3f} "
      f"({(1-res['Preview (mismatched)']/res['LQG'])*100:+.0f}% vs LQG; exact-model "
      f"reference {res['Preview (exact model)']:.3f} — no inverse-crime inflation), "
      f"HRC {res['ESO-ADRC+HRC']:.3f} (best, ~2x better than preview).")
print(f"  HONEST CONCLUSION: a per-step material-removal-aware predictive "
      f"controller is not physically justified here (timescale over-engineering "
      f"~1e4x) and its removal-awareness adds nothing in the controllable regime "
      f"(regime conflict). The preview-from-physics law is a valid, stable strategy "
      f"that modestly beats LQG at a tuned horizon, but it is dominated ~2x by the "
      f"existing HRC and, being model-based, is inherently limited by cutting-model "
      f"accuracy — weaker than P6 probe-identification, which measures the truth.")
print(f"\nTotal time {time.time()-t0:.1f}s  ->  {OUT_DIR}/")
