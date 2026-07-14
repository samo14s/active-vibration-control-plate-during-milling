"""
main_delayed_pd_baseline.py
===========================
Four-way benchmark: Open-Loop / Delayed PD (article Eq. 30) / LQG / PALF-LQG.

The delayed PD gains (K_Pp, K_Pd) are grid-tuned on the NOMINAL scenario only (design-
time tuning, same standard as the LQG weight grid search); all controllers are then
FROZEN and evaluated on the four scenarios (S2/S3/S4 held-out). Same plant realism as
main_simulation.py: 5-mode plant / 3-mode designs (spillover), 10 nm measurement
noise, corrected Eq. (3) forces, Eq. (15) piezo coupling, identical +/-150 V clipping.

Run: python main_delayed_pd_baseline.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from plate_model import PlateModel
from milling_force import precompute_alpha_periodic, cutting_constants
from lqg_controller import LQGController
from palf_lqg_controller import PALF_LQG_Controller
from delayed_pd_controller import DelayedPDController
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
T_TUNE = 0.2                      # shorter horizon for the PD gain grid search
OUT_DIR = "figs_lqg_vs_palf"
os.makedirs(OUT_DIR, exist_ok=True)

RT = DT_TOOL / 2
PHI_ST = np.pi - np.arccos(1 - AE / RT);  PHI_EX = np.pi
OMEGA = 2 * np.pi * RPM / 60;  TAU = 60 / (NT * RPM)
K1, K2 = cutting_constants(KN, MU_C, ETA_H, GAMMA_N)
N_PER = int(np.round(TAU / DT))


def build_plant(zp_pos, freq_perturb=0.0):
    p = PlateModel(LP, HP, BP, RHO, E_AL, NU_AL, N1=N1, N2=N2,
                   n_modes=N_MODES_PLANT, zeta_modes=ZETA_PLANT, verbose=False)
    p.precompute_Dp(zp_pos=zp_pos, n_pos=2001)
    p.set_observation(x_obs=LP, z_obs=HP)
    p.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    if abs(freq_perturb) > 1e-6:
        p = p.perturbed_copy(freq_perturb)
    return p


def forces_for(sim, ap, kt):
    v_feed = FT * NT * RPM / 60
    kp = np.clip(np.round(np.minimum(v_feed * sim.t_vec, LP) / LP * 2000).astype(int),
                 0, 2000)
    a3, a4 = precompute_alpha_periodic(DT, N_PER, sim.nstep, OMEGA, NT, RT, ETA_H,
                                       PHI_ST, PHI_EX, HP - ap, HP, K1, K2, kt)
    return a3, a4, kp


def run(sim, a3, a4, kp, ctrl):
    if hasattr(ctrl, 'reset'):
        ctrl.reset()
    if hasattr(ctrl, 'history_u_lqg'):
        for h in ('history_u_lqg', 'history_u_ff', 'history_u_total',
                  'history_phase', 'history_safety'):
            setattr(ctrl, h, [])
    r = sim.simulate(a3, a4, kp, controller=ctrl,
                     meas_noise_std=MEAS_NOISE_STD,
                     rng=np.random.default_rng(1234), progress=False)
    i = r['stop_idx']
    rms = np.sqrt(np.mean(r['y'][:i + 1]**2)) * 1e6
    diverged = i < sim.nstep - 2
    return rms, diverged, i * sim.dt


# ------------------------------------------------------------------
# Design phase (nominal): LQG grid, PALF ILC (frozen), PD gain grid
# ------------------------------------------------------------------
print("[design] nominal plant + controllers...")
PLANT = build_plant(zp_pos=HP - 0.3e-3 / 2)
design = PLANT.truncated_view(N_MODES_CTRL)
sim_nom = NewmarkSimulator(PLANT, dt=DT, T_end=T_END, ft=FT, tau=TAU, verbose=False)
a3n, a4n, kpn = forces_for(sim_nom, 0.3e-3, KT_NOMINAL)

LQG = LQGController(design, dt=DT, verbose=False, kalman_V=KALMAN_V, u_max=150.0)
LQG.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                     w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
LQG.discretize_observer()

PALF = PALF_LQG_Controller(design, dt=DT, base_w_q=LQG.w_q, base_w_qd=LQG.w_qd,
                           base_w_r=1.0, ff_lr=0.005, ff_max=10.0, ff_alpha=1.0,
                           n_per=N_PER, safety_alpha=5.0, kalman_V=KALMAN_V,
                           u_max=150.0, verbose=False)
PALF.pretrain_iterative_simulation(sim_nom, a3n, a4n, kpn,
                                   n_iterations=10, n_epochs_per_iter=30,
                                   verbose=False)

print("[design] grid-tuning delayed PD (Eq. 30) on the nominal scenario...")
sim_tune = NewmarkSimulator(PLANT, dt=DT, T_end=T_TUNE, ft=FT, tau=TAU, verbose=False)
a3t, a4t, kpt = forces_for(sim_tune, 0.3e-3, KT_NOMINAL)
# Rank by: (1) survived the horizon, (2) longest survival time, (3) lowest RMS.
# NOTE: the article itself reports that delayed PD ALONE is still unstable at the
# start/end positions of this very cutting condition (Du et al. 2024, Fig. 14) — so
# a fully stabilizing gain pair may simply not exist; we then keep the least-bad one.
best = dict(score=(-1, -np.inf, np.inf), kpp=0.0, kpd=0.0, rms=np.inf, div=True)
for kpp in [-1e8, -3e7, -1e7, -3e6, -1e6, 0.0, 1e6, 3e6, 1e7, 3e7, 1e8]:
    for kpd in [-3e4, -1e4, -3e3, -1e3, -3e2, 0.0, 3e2, 1e3, 3e3, 1e4, 3e4]:
        if kpp == 0.0 and kpd == 0.0:
            continue
        pd = DelayedPDController(DT, TAU, kpp, kpd, u_max=150.0)
        rms, div, t_surv = run(sim_tune, a3t, a4t, kpt, pd)
        score = (0 if div else 1, t_surv, -rms)
        if score > best['score']:
            best = dict(score=score, kpp=kpp, kpd=kpd, rms=rms, div=div)
print(f"[design] PD gains: K_Pp = {best['kpp']:.1e} V/m, "
      f"K_Pd = {best['kpd']:.1e} V.s/m  "
      f"(tuning rms {best['rms']:.3f} um{', DIVERGED' if best['div'] else ''})")
if best['div']:
    print("[design] NOTE: no grid gain pair stabilizes S1 — consistent with the")
    print("         article's own finding (Fig. 14: delayed PD alone is unstable at")
    print("         the start/end positions of this cutting condition).")
PD = DelayedPDController(DT, TAU, best['kpp'], best['kpd'], u_max=150.0)

# ------------------------------------------------------------------
# Held-out evaluation: 4 scenarios x {OL, PD, LQG, PALF}
# ------------------------------------------------------------------
scenarios = [
    ("S1 - Nominal",             0.3e-3, KT_NOMINAL, 0.0),
    ("S2 - Aggressive ap=0.6mm", 0.6e-3, KT_NOMINAL, 0.0),
    ("S3 - Uncertainty w-8%",    0.3e-3, KT_NOMINAL, -0.08),
    ("S4 - High K_T +30%",       0.3e-3, 1.3 * KT_NOMINAL, 0.0),
]
names, table = [], {}
print("\n" + "=" * 76)
print(f"  {'Scenario':<26}{'Open-Loop':>12}{'DelayedPD':>12}{'LQG':>10}{'PALF':>10}")
print("-" * 76)
for name, ap, kt, fp in scenarios:
    plant_s = build_plant(zp_pos=HP - ap / 2, freq_perturb=fp) \
        if (fp != 0.0 or ap != 0.3e-3) else PLANT
    sim = NewmarkSimulator(plant_s, dt=DT, T_end=T_END, ft=FT, tau=TAU, verbose=False)
    a3, a4, kp = forces_for(sim, ap, kt)
    row = {}
    # Open loop (no controller)
    r = sim.simulate(a3, a4, kp, controller=None, progress=False)
    i = r['stop_idx']
    ol_div = i < sim.nstep - 2
    row['OL'] = (np.sqrt(np.mean(r['y'][:i + 1]**2)) * 1e6, ol_div)
    for tag, ctrl in [('PD', PD), ('LQG', LQG), ('PALF', PALF)]:
        rms, div, _ = run(sim, a3, a4, kp, ctrl)
        row[tag] = (rms, div)
    names.append(name)
    table[name] = row
    fmt = lambda v: (f"{v[0]:9.3f}um" + ("*" if v[1] else " "))
    print(f"  {name:<26}{fmt(row['OL']):>12}{fmt(row['PD']):>12}"
          f"{fmt(row['LQG']):>10}{fmt(row['PALF']):>10}")
print("-" * 76)
print("  (* = diverged / stopped early; RMS computed up to the stop index)")

# ------------------------------------------------------------------
# Figure: grouped bars (log scale, divergence hatched)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5.5))
tags = ['OL', 'PD', 'LQG', 'PALF']
colors = {'OL': '#888888', 'PD': '#6A5ACD', 'LQG': '#2E8B57', 'PALF': '#DC143C'}
x = np.arange(len(names));  w = 0.2
for j, tag in enumerate(tags):
    vals = [table[n][tag][0] for n in names]
    divs = [table[n][tag][1] for n in names]
    bars = ax.bar(x + (j - 1.5) * w, vals, w, color=colors[tag], alpha=0.85,
                  edgecolor='k', label={'OL': 'Open-Loop', 'PD': 'Delayed PD (Eq. 30)',
                                        'LQG': 'LQG', 'PALF': 'PALF-LQG'}[tag])
    for b, d in zip(bars, divs):
        if d:
            b.set_hatch('//');  b.set_alpha(0.45)
ax.set_yscale('log')
ax.set_xticks(x); ax.set_xticklabels([n.split(' - ')[1] for n in names], fontsize=10)
ax.set_ylabel(r"$y_{RMS}$ (µm, log)")
ax.set_title("Four-way benchmark (hatched = diverged): OL / Delayed PD / LQG / PALF-LQG")
ax.legend(fontsize=10); ax.grid(True, axis='y', which='both', alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_delayed_pd_benchmark.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig_delayed_pd_benchmark.pdf", bbox_inches='tight')
plt.close()
print(f"\n  >>> fig_delayed_pd_benchmark saved in {OUT_DIR}/")
