"""
main_simulation.py
==================
LQG  vs  PALF-LQG  (LQG + phase-locked learned feedforward).

Fair-comparison protocol (see docs/AUDIT_FINDINGS.md, docs/CONTRIBUTION.md):
   * ONE feedback design (LQG weights grid-searched on the nominal model) is shared
     by the baseline LQG and by PALF's internal LQR, so the measured difference is
     attributable to the feedforward alone.
   * The PALF feedforward is trained ONCE on the nominal scenario (S1), then FROZEN
     and evaluated on the held-out scenarios S2/S3/S4 (no per-scenario retraining).
   * The closed-loop stability boundary is NOT inflated: a phase-locked feedforward
     does not move the regenerative chatter boundary, so PALF shares the LQG SLD.

Mesures :
   1. PERFORMANCE TEMPORAL (4 scénarios)
      - y(t), u(t)
      - y_max, y_RMS, u_max
   
   2. ANALYSE FRÉQUENTIELLE (FFT)
      - Spectres y(t) et u(t)
      - Amplitude des pics dominants
   
   3. STABILITY LOBE DIAGRAM (SLD)
      - Domaine de stabilité étendu
      - Lobes de chatter par méthode
   
   4. ROBUSTESSE (Monte Carlo)
      - Variance face aux incertitudes paramétriques
   
   5. EFFORT DE COMMANDE
      - Énergie totale, RMS de u(t)
      - Vitesse de variation (Δu)
   
   6. PÔLES BOUCLE FERMÉE
      - Amortissements modaux
      - Fréquences modales

Sortie : 8+ figures professionnelles + tableau récapitulatif
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plate_model import PlateModel
from milling_force import precompute_alpha_periodic, cutting_constants
from lqg_controller import LQGController
from palf_lqg_controller import PALF_LQG_Controller
from newmark_solver import NewmarkSimulator
from fdm_stability import compute_SLD, compute_closed_loop_SLD


# ============================================================
# Paramètres physiques (article)
# ============================================================
LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU_AL = 2830, 69e9, 0.33
NT = 3;  DT_TOOL = 0.010
ETA_H = np.deg2rad(35);  GAMMA_N = np.deg2rad(15)
KT_NOMINAL = 925e6;  KN = 0.26;  MU_C = 0.20
RPM = 4900
FT = 0.02e-3;  AE = 0.1e-3
D31 = 175e-12;  H_PA = 0.7e-3
E_PE = 63e9;    NU_PE = 0.35
N1, N2 = 30, 24

# --- Inverse-crime avoidance (P1) -------------------------------------------
# The controllers are designed on N_MODES_CTRL modes, but the PLANT that is
# simulated carries N_MODES_PLANT modes. The extra plant modes are unmodelled by
# the controller (control + observation spillover). Damping ratios of the first
# five modes are the article's measured values (Table 4).
N_MODES_PLANT = 5
N_MODES_CTRL = 3
N_MODES = N_MODES_CTRL          # controller/SLD dimension (back-compat alias)
ZETA_PLANT = [0.0031, 0.0017, 0.0027, 0.0056, 0.0035]
ZETA = ZETA_PLANT[:N_MODES_CTRL]

# Realistic measurement noise (eddy-current sensor, ~10 nm) injected into y. The
# Kalman keeps a conservative noise assumption (robust): a diagnostic sweep showed an
# aggressive V (1e-14) needlessly loses robustness to frequency mismatch, while the
# 10 nm noise itself has negligible effect on either controller.
MEAS_NOISE_STD = 1e-8           # 10 nm RMS on the displacement measurement
KALMAN_V = 1e-12                # conservative/robust observer assumption

DT = 5e-5
T_END = 0.5

OUT_DIR = "figs_lqg_vs_palf"
os.makedirs(OUT_DIR, exist_ok=True)

# Label used for the feedforward-augmented controller throughout figures/tables
FF_LABEL = "PALF-LQG"


# ============================================================
# Helper functions
# ============================================================
def build_plant(zp_pos, freq_perturb=0.0):
    """Build a full-order (N_MODES_PLANT) plant. Frequency drift is applied as a
    stiffness perturbation via perturbed_copy so mode signs stay consistent."""
    plate = PlateModel(LP, HP, 0.004, RHO, E_AL, NU_AL,
                       N1=N1, N2=N2, n_modes=N_MODES_PLANT,
                       zeta_modes=ZETA_PLANT, verbose=False)
    plate.precompute_Dp(zp_pos=zp_pos, n_pos=2001)
    plate.set_observation(x_obs=LP, z_obs=HP)
    plate.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    if abs(freq_perturb) > 1e-6:
        plate = plate.perturbed_copy(freq_perturb)
    return plate


def build_plate(zp_pos, freq_perturb=0.0):
    """Reduced (N_MODES_CTRL) plate, used only for the SLD design-model charts."""
    plate = PlateModel(LP, HP, 0.004, RHO, E_AL, NU_AL,
                       N1=N1, N2=N2, n_modes=N_MODES_CTRL,
                       zeta_modes=ZETA, verbose=False)
    plate.precompute_Dp(zp_pos=zp_pos, n_pos=2001)
    plate.set_observation(x_obs=LP, z_obs=HP)
    plate.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    if abs(freq_perturb) > 1e-6:
        plate = plate.perturbed_copy(freq_perturb)
    return plate


def reset_palf_histories(palf):
    palf.history_u_lqg = []
    palf.history_u_ff = []
    palf.history_u_total = []
    palf.history_phase = []
    palf.history_safety = []


def fft_amp(y, dt, i_start_t=0.05):
    i_start = int(i_start_t/dt)
    if len(y) <= i_start:
        return None, None
    y_seg = y[i_start:]
    N = len(y_seg)
    win = np.hanning(N)
    NFFT = 2**int(np.ceil(np.log2(max(N, 2))))
    Y = np.fft.fft(y_seg*win, NFFT) / N
    f = (1/dt)/2 * np.linspace(0, 1, NFFT//2 + 1)
    return f, 2*np.abs(Y[:NFFT//2 + 1]) * 1e6


# ============================================================
# Phase 1 : Simulations multi-scénarios
# ============================================================
print("="*72)
print(" COMPARISON  :  LQG  vs  PALF-LQG  (train-once / held-out) ")
print("="*72)
t_global = time.time()

# Fixed cutting geometry / force constants (shared by every scenario).
# k1, k2 come from the single corrected source (Du et al. Eq. 3).
PHI_ST = np.pi - np.arccos(1 - AE/(DT_TOOL/2))
PHI_EX = np.pi
OMEGA = 2*np.pi*RPM/60
RT = DT_TOOL/2
TAU = 60/(NT*RPM)
K1, K2 = cutting_constants(KN, MU_C, ETA_H, GAMMA_N)
F_T = NT * RPM / 60
N_PER = int(np.round(TAU/DT))


def make_forces(sim, ap, KT_actual):
    """Precompute (alpha3, alpha4, kp_idx) for a scenario on a given simulator."""
    v_feed = FT * NT * RPM / 60
    xp_path = np.minimum(v_feed * sim.t_vec, LP)
    kp_idx = np.clip(np.round(xp_path/LP * 2000).astype(int), 0, 2000)
    alpha3, alpha4 = precompute_alpha_periodic(DT, N_PER, sim.nstep,
        OMEGA, NT, RT, ETA_H, PHI_ST, PHI_EX, HP-ap, HP, K1, K2, KT_actual)
    return alpha3, alpha4, kp_idx


# NOTE (P1): with the corrected (stronger) Eq.(3) forces, the CONTROLLED stability
# margin at ap=0.3 mm / 4900 RPM is ~-9 % frequency mismatch — beyond that BOTH
# controllers lose stability at this depth (a genuine robustness limit revealed by the
# honest force model; see docs/REPRODUCED_RESULTS.md). S3 therefore uses a -8 %
# mismatch (within margin, and matching the article's ~9 % measured 2nd-mode drift).
scenarios_def = [
    ("S1 - Nominal article",      0.3e-3, KT_NOMINAL, 0.0),
    ("S2 - Aggressive ap=0.6mm",  0.6e-3, KT_NOMINAL, 0.0),
    ("S3 - Uncertainty ω-8%",     0.3e-3, KT_NOMINAL, -0.08),
    ("S4 - High K_T +30%",        0.3e-3, 1.3*KT_NOMINAL, 0.0),
]

# ------------------------------------------------------------------
# ONE shared feedback design + ONE feedforward pretraining (nominal).
# Both controllers are designed on the nominal model and then evaluated,
# unchanged, on every scenario (including the perturbed ones).
# ------------------------------------------------------------------
print("\n[design] Building nominal full-order plant, designing controllers on "
      f"{N_MODES_CTRL} modes (plant has {N_MODES_PLANT}: spillover), and pretraining "
      "PALF once on the nominal scenario (S1)...")
# Full-order nominal plant (5 modes). ONE eigensolve — its truncation feeds the
# controllers, and its perturbed copies feed each scenario, so all mode signs match.
PLANT_NOMINAL = build_plant(zp_pos=HP - 0.3e-3/2, freq_perturb=0.0)
design_view = PLANT_NOMINAL.truncated_view(N_MODES_CTRL)   # controller design model

sim_design = NewmarkSimulator(PLANT_NOMINAL, dt=DT, T_end=T_END,
                              ft=FT, tau=TAU, verbose=False)
alpha3_nom, alpha4_nom, kp_idx_nom = make_forces(sim_design, 0.3e-3, KT_NOMINAL)

# Shared LQG feedback (grid-searched weights), designed on the reduced model,
# with realistic measurement-noise assumption and identical actuator clipping.
LQG_SHARED = LQGController(design_view, dt=DT, verbose=False,
                           kalman_V=KALMAN_V, u_max=150.0)
LQG_SHARED.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                            w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
LQG_SHARED.discretize_observer()
print(f"[design] LQG weights: w_q={LQG_SHARED.w_q:.1e}, w_qd={LQG_SHARED.w_qd:.1e}")

# PALF: SAME feedback weights + phase feedforward, pretrained ONCE (on the 5-mode
# plant, with measurement noise), then frozen.
PALF_SHARED = PALF_LQG_Controller(design_view, dt=DT,
                                  base_w_q=LQG_SHARED.w_q, base_w_qd=LQG_SHARED.w_qd,
                                  base_w_r=1.0,
                                  ff_lr=0.005, ff_max=10.0, ff_alpha=1.0,
                                  n_per=N_PER, safety_alpha=5.0,
                                  kalman_V=KALMAN_V, u_max=150.0, verbose=False)
PALF_SHARED.pretrain_iterative_simulation(sim_design, alpha3_nom, alpha4_nom,
                                          kp_idx_nom, n_iterations=30,
                                          n_epochs_per_iter=15, verbose=False)
print(f"[design] PALF feedforward trained and FROZEN. Plant modes (Hz): "
      f"{np.round(PLANT_NOMINAL.freq_n, 0)}; controller sees first {N_MODES_CTRL}. "
      f"Meas. noise = {MEAS_NOISE_STD*1e9:.0f} nm.")


def build_scenario_plant(ap, freq_perturb):
    """Perturbed full-order plant for a scenario, reusing the nominal eigensolve."""
    plant = PLANT_NOMINAL.perturbed_copy(freq_perturb)
    zp = HP - ap/2
    if abs(zp - PLANT_NOMINAL.zp_pos) > 1e-9:   # different tool height -> new Dp
        plant.precompute_Dp(zp_pos=zp, n_pos=2001)
    return plant


def run_scenario(name, ap, KT_actual, freq_perturb):
    print(f"\n{'='*72}")
    print(f" {name}")
    print(f"  ap = {ap*1e3:.2f}mm,  KT = {KT_actual/1e6:.0f}MPa,  "
          f"freq_drift = {freq_perturb*100:+.0f}%")
    print(f"{'='*72}")

    # Evaluation plant (perturbed, full-order); the controllers are the frozen
    # nominal reduced-order designs (spillover) and the measurement is noisy.
    plate_r = build_scenario_plant(ap, freq_perturb)
    sim = NewmarkSimulator(plate_r, dt=DT, T_end=T_END, ft=FT, tau=TAU, verbose=False)
    f_t = F_T
    alpha3, alpha4, kp_idx = make_forces(sim, ap, KT_actual)

    # === LQG (shared, unchanged) ===
    res_lqg = sim.simulate(alpha3, alpha4, kp_idx, controller=LQG_SHARED,
                           meas_noise_std=MEAS_NOISE_STD,
                           rng=np.random.default_rng(1234), progress=False)

    # === PALF-LQG (shared, frozen feedforward — NOT retrained here) ===
    reset_palf_histories(PALF_SHARED)
    res_palf = sim.simulate(alpha3, alpha4, kp_idx, controller=PALF_SHARED,
                            meas_noise_std=MEAS_NOISE_STD,
                            rng=np.random.default_rng(1234), progress=False)

    # Compute metrics
    metrics = {}
    for ctrl, res in [('LQG', res_lqg), (FF_LABEL, res_palf)]:
        u_arr = res['u']
        y_arr = res['y']
        i_end = res['stop_idx']
        
        # Time-domain metrics
        y_clip = y_arr[:i_end+1]
        u_clip = u_arr[:i_end+1]
        
        # Δu (commande variation)
        du = np.diff(u_clip)
        
        metrics[ctrl] = {
            'res': res,
            'y_max':   np.max(np.abs(y_clip)) * 1e6,    # µm
            'y_rms':   np.sqrt(np.mean(y_clip**2)) * 1e6,
            'y_p2p':   (np.max(y_clip) - np.min(y_clip)) * 1e6,
            'u_max':   np.max(np.abs(u_clip)),          # V
            'u_rms':   np.sqrt(np.mean(u_clip**2)),
            'u_energy':np.sum(u_clip**2) * DT,           # V²·s
            'du_max':  np.max(np.abs(du)),               # V
            'du_rms':  np.sqrt(np.mean(du**2)),
        }
    
    print(f"  {'Métrique':<22}{'LQG':<14}{FF_LABEL:<14}{'Δ':<10}")
    print(f"  {'-'*60}")
    for key in ['y_max', 'y_rms', 'y_p2p', 'u_max', 'u_rms']:
        l = metrics['LQG'][key]
        d = metrics[FF_LABEL][key]
        if key.startswith('y'):
            change = (l - d) / l * 100  # negative = better
            unit = 'µm'
            print(f"  {key+' (red.)':<22}{l:<14.4f}{d:<14.4f}{change:+6.2f}% {unit}")
        else:
            change = (d - l) / l * 100
            unit = 'V'
            print(f"  {key+' (effort)':<22}{l:<14.2f}{d:<14.2f}{change:+6.2f}% {unit}")
    
    return {
        'name': name, 'ap': ap, 'KT': KT_actual, 'freq_pert': freq_perturb,
        'sim': sim, 'metrics': metrics, 'f_t': f_t,
        'plate_d': PLANT_NOMINAL, 'plate_r': plate_r,
        'lqg_obj': LQG_SHARED, 'palf_obj': PALF_SHARED,
    }


# Run scenarios
scenarios = []
for name, ap, KT_, fp in scenarios_def:
    scenarios.append(run_scenario(name, ap, KT_, fp))


# ============================================================
# FIGURE 1 : Performance Bilan (3 panneaux)
# ============================================================
names = [s['name'].split(' - ')[0] for s in scenarios]
descs = [s['name'].split(' - ')[1] for s in scenarios]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# (a) y_RMS
ax = axes[0]
x = np.arange(len(names))
w = 0.35
y_lqg = [s['metrics']['LQG']['y_rms'] for s in scenarios]
y_darc = [s['metrics'][FF_LABEL]['y_rms'] for s in scenarios]
b1 = ax.bar(x - w/2, y_lqg, w, color='#2E8B57', alpha=0.85,
              edgecolor='k', label='LQG', linewidth=1.5)
b2 = ax.bar(x + w/2, y_darc, w, color='#DC143C', alpha=0.85,
              edgecolor='k', label='PALF-LQG', linewidth=1.5)
for bar, v in zip(b1, y_lqg):
    ax.text(bar.get_x()+bar.get_width()/2, v*1.02, f'{v:.3f}',
             ha='center', fontsize=9, fontweight='bold')
for bar, v in zip(b2, y_darc):
    ax.text(bar.get_x()+bar.get_width()/2, v*1.02, f'{v:.3f}',
             ha='center', fontsize=9, fontweight='bold', color='darkred')
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
ax.set_ylabel("$y_{RMS}$ (µm)", fontsize=12)
ax.set_title("Vibration RMS : LQG vs PALF-LQG", fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='upper left'); ax.grid(True, axis='y', alpha=0.5)

# (b) Gain en %
ax = axes[1]
gains = [(1 - d/l)*100 for l, d in zip(y_lqg, y_darc)]
colors_g = ['#28a745' if g > 0 else '#dc3545' for g in gains]
bars = ax.bar(x, gains, 0.55, color=colors_g, alpha=0.8, edgecolor='k', linewidth=1.5)
for bar, g in zip(bars, gains):
    y_txt = g + (0.5 if g > 0 else -1.0)
    ax.text(bar.get_x()+bar.get_width()/2, y_txt, f'{g:+.2f}%',
             ha='center', fontsize=11, fontweight='bold')
ax.axhline(0, color='k', linewidth=1)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
ax.set_ylabel("Δ y_RMS PALF vs LQG (%)", fontsize=12)
ax.set_title("Gain de réduction de vibration (positif = amélioration)",
              fontsize=12, fontweight='bold')
ax.grid(True, axis='y', alpha=0.5)

# (c) u_max
ax = axes[2]
u_lqg = [s['metrics']['LQG']['u_max'] for s in scenarios]
u_darc = [s['metrics'][FF_LABEL]['u_max'] for s in scenarios]
b1 = ax.bar(x - w/2, u_lqg, w, color='#2E8B57', alpha=0.85,
              edgecolor='k', label='LQG', linewidth=1.5)
b2 = ax.bar(x + w/2, u_darc, w, color='#DC143C', alpha=0.85,
              edgecolor='k', label='PALF-LQG', linewidth=1.5)
ax.axhline(150, color='red', linestyle='--', linewidth=1.5,
            alpha=0.7, label='Sat. piezo')
for bar, v in zip(b1, u_lqg):
    ax.text(bar.get_x()+bar.get_width()/2, v*1.02, f'{v:.1f}',
             ha='center', fontsize=9, fontweight='bold')
for bar, v in zip(b2, u_darc):
    ax.text(bar.get_x()+bar.get_width()/2, v*1.02, f'{v:.1f}',
             ha='center', fontsize=9, fontweight='bold', color='darkred')
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
ax.set_ylabel("$|u|_{max}$ (V)", fontsize=12)
ax.set_title("Effort de commande pic", fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, axis='y', alpha=0.5)

plt.suptitle(" LQG vs PALF-LQG : Bilan global ",
              fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig01_bilan.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"\n  ✓ fig01_bilan.png")


# ============================================================
# FIGURE 2 : Réponse temporelle (4 sous-plots)
# ============================================================
fig, axes = plt.subplots(len(scenarios), 1, figsize=(14, 3.0*len(scenarios)))
if len(scenarios) == 1:
    axes = [axes]

for i, s in enumerate(scenarios):
    sim = s['sim']
    t_ms = sim.t_vec * 1e3
    res_lqg = s['metrics']['LQG']['res']
    res_darc = s['metrics'][FF_LABEL]['res']
    
    ax = axes[i]
    i_end = min(res_lqg['stop_idx'], res_darc['stop_idx'])
    
    ax.plot(t_ms[:i_end+1], res_lqg['y'][:i_end+1]*1e6,
             color='#2E8B57', linewidth=0.5, alpha=0.85,
             label=f"LQG (RMS={s['metrics']['LQG']['y_rms']:.3f}µm)")
    ax.plot(t_ms[:i_end+1], res_darc['y'][:i_end+1]*1e6,
             color='#DC143C', linewidth=0.5, alpha=0.85,
             label=f"PALF-LQG (RMS={s['metrics'][FF_LABEL]['y_rms']:.3f}µm)")
    
    ax.set_ylabel("$y_p$ (µm)", fontsize=11)
    ax.set_title(f"{s['name']}", fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.5)
    if i == len(scenarios)-1:
        ax.set_xlabel("Temps (ms)", fontsize=11)

plt.suptitle("Réponse temporelle $y_p(t)$ - 4 scénarios",
              fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig02_temporal_y.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"  ✓ fig02_temporal_y.png")


# ============================================================
# FIGURE 3 : Tension piezo u(t)
# ============================================================
fig, axes = plt.subplots(len(scenarios), 1, figsize=(14, 3.0*len(scenarios)))
if len(scenarios) == 1:
    axes = [axes]

for i, s in enumerate(scenarios):
    sim = s['sim']
    t_ms = sim.t_vec * 1e3
    res_lqg = s['metrics']['LQG']['res']
    res_darc = s['metrics'][FF_LABEL]['res']
    
    ax = axes[i]
    i_end = min(res_lqg['stop_idx'], res_darc['stop_idx'])
    
    ax.plot(t_ms[:i_end+1], res_lqg['u'][:i_end+1],
             color='#2E8B57', linewidth=0.5, alpha=0.85,
             label=f"LQG (max={s['metrics']['LQG']['u_max']:.1f}V)")
    ax.plot(t_ms[:i_end+1], res_darc['u'][:i_end+1],
             color='#DC143C', linewidth=0.5, alpha=0.85,
             label=f"PALF-LQG (max={s['metrics'][FF_LABEL]['u_max']:.1f}V)")
    
    ax.set_ylabel("u (V)", fontsize=11)
    ax.set_title(f"{s['name']}", fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.5)
    if i == len(scenarios)-1:
        ax.set_xlabel("Temps (ms)", fontsize=11)

plt.suptitle("Tension piezoélectrique u(t) - 4 scénarios",
              fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig03_temporal_u.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"  ✓ fig03_temporal_u.png")


# ============================================================
# FIGURE 4 : FFT y(t) sur 4 scénarios
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
axes = axes.flatten()

for i, s in enumerate(scenarios):
    ax = axes[i]
    res_lqg = s['metrics']['LQG']['res']
    res_darc = s['metrics'][FF_LABEL]['res']
    f_t = s['f_t']
    plate = s['plate_r']
    
    i_end_l = res_lqg['stop_idx']
    i_end_d = res_darc['stop_idx']
    
    f_l, Y_l = fft_amp(res_lqg['y'][:i_end_l+1], DT)
    f_d, Y_d = fft_amp(res_darc['y'][:i_end_d+1], DT)
    
    ax.plot(f_l, Y_l, color='#2E8B57', linewidth=1.5, alpha=0.9,
             label=f"LQG (max={Y_l.max():.3f}µm)")
    ax.plot(f_d, Y_d, color='#DC143C', linewidth=1.5, alpha=0.9,
             label=f"PALF-LQG (max={Y_d.max():.3f}µm)")
    
    # Vertical lines for harmonics and modes
    for n_h in range(1, 5):
        f_h = n_h * f_t
        if f_h <= 1200:
            ax.axvline(f_h, color='gray', linestyle=':', alpha=0.6, linewidth=0.8)
    
    for fc in plate.freq_n:
        if fc <= 1200:
            ax.axvline(fc, color='blue', linestyle='--', alpha=0.4, linewidth=0.8)
    
    ax.set_xlim([0, 1200])
    ax.set_xlabel("Fréquence (Hz)", fontsize=10)
    ax.set_ylabel("Amplitude (µm)", fontsize=10)
    ax.set_title(s['name'], fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.5)

plt.suptitle("Spectres de vibration FFT[$y_p$] - 4 scénarios",
              fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig04_fft_y.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"  ✓ fig04_fft_y.png")


# ============================================================
# FIGURE 5 : FFT u(t)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
axes = axes.flatten()

for i, s in enumerate(scenarios):
    ax = axes[i]
    res_lqg = s['metrics']['LQG']['res']
    res_darc = s['metrics'][FF_LABEL]['res']
    f_t = s['f_t']
    
    i_end_l = res_lqg['stop_idx']
    i_end_d = res_darc['stop_idx']
    
    # FFT of u (same logic, but in V not µm)
    def fft_u(u, dt, i_start_t=0.05):
        i_start = int(i_start_t/dt)
        if len(u) <= i_start: return None, None
        u_seg = u[i_start:]
        N = len(u_seg); win = np.hanning(N)
        NFFT = 2**int(np.ceil(np.log2(max(N, 2))))
        U = np.fft.fft(u_seg*win, NFFT)/N
        f = (1/dt)/2 * np.linspace(0, 1, NFFT//2 + 1)
        return f, 2*np.abs(U[:NFFT//2 + 1])
    
    f_l, U_l = fft_u(res_lqg['u'][:i_end_l+1], DT)
    f_d, U_d = fft_u(res_darc['u'][:i_end_d+1], DT)
    
    ax.semilogy(f_l, np.maximum(U_l, 1e-3), color='#2E8B57', linewidth=1.2, label='LQG')
    ax.semilogy(f_d, np.maximum(U_d, 1e-3), color='#DC143C', linewidth=1.2, label='PALF-LQG')
    
    for n_h in range(1, 5):
        f_h = n_h * f_t
        if f_h <= 1200:
            ax.axvline(f_h, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    
    ax.set_xlim([0, 1200])
    ax.set_ylim([1e-2, 10])
    ax.set_xlabel("Fréquence (Hz)", fontsize=10)
    ax.set_ylabel("Amplitude u (V, log)", fontsize=10)
    ax.set_title(s['name'], fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, which='both', alpha=0.5)

plt.suptitle("Spectres de commande FFT[u] - 4 scénarios (échelle log)",
              fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig05_fft_u.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"  ✓ fig05_fft_u.png")


# ============================================================
# FIGURE 6 : Closed-loop poles (zeta vs frequency)
# ============================================================
def extract_modes(ev_cl, n_modes):
    ev_pos = ev_cl[np.imag(ev_cl) > 0]
    ev_pos = ev_pos[np.argsort(np.imag(ev_pos))]
    omega_arr = np.zeros(n_modes)
    zeta_arr = np.zeros(n_modes)
    for k in range(n_modes):
        if k < len(ev_pos):
            e = ev_pos[k]
            omega_arr[k] = abs(e)
            zeta_arr[k] = -np.real(e)/abs(e)
    return omega_arr, zeta_arr


# Use S1 nominal for pole analysis
s1 = scenarios[0]
plate_d = s1['plate_d']

# Open-loop modes
omega_OL = plate_d.omega_n
zeta_OL = np.array(ZETA)

# LQG closed-loop
omega_LQG, zeta_LQG = extract_modes(s1['lqg_obj'].ev_cl, N_MODES)

# PALF-LQG closed-loop poles are IDENTICAL to LQG: the feedforward adds a periodic
# forcing, it does not change the closed-loop poles. This figure shows exactly that
# (no fabricated damping improvement).

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# (a) Damping ratios
ax = axes[0]
modes_x = np.arange(N_MODES)
w = 0.30
ax.bar(modes_x - w, zeta_OL*100, w, color='gray', alpha=0.7,
        edgecolor='k', label='Open-loop', linewidth=1.5)
ax.bar(modes_x, zeta_LQG*100, w, color='#2E8B57', alpha=0.85,
        edgecolor='k', label='LQG', linewidth=1.5)
# PALF-LQG shares the LQG poles exactly (feedforward does not change CL damping)
ax.bar(modes_x + w, zeta_LQG*100, w, color='#DC143C', alpha=0.85,
        edgecolor='k', label='PALF-LQG (= LQG poles)', linewidth=1.5,
        hatch='//')
for k in range(N_MODES):
    ax.text(modes_x[k] - w, zeta_OL[k]*100*1.02,
             f'{zeta_OL[k]*100:.2f}%', ha='center', fontsize=9, fontweight='bold')
    ax.text(modes_x[k], zeta_LQG[k]*100*1.02,
             f'{zeta_LQG[k]*100:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.text(modes_x[k] + w, zeta_LQG[k]*100*1.02,
             f'{zeta_LQG[k]*100:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(modes_x)
ax.set_xticklabels([f'Mode {k+1}\n@{plate_d.freq_n[k]:.0f}Hz' for k in range(N_MODES)])
ax.set_ylabel("Amortissement modal ζ (%)", fontsize=12)
ax.set_title("Amortissement modal en boucle fermée (S1)",
              fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, axis='y', alpha=0.5)

# (b) Pôles dans le plan complexe
ax = axes[1]
ev_OL = []
for k in range(N_MODES):
    omega_d = omega_OL[k] * np.sqrt(1 - zeta_OL[k]**2)
    real_p = -zeta_OL[k] * omega_OL[k]
    ev_OL.append(complex(real_p, omega_d))
    ev_OL.append(complex(real_p, -omega_d))

ax.scatter([e.real for e in ev_OL], [e.imag for e in ev_OL],
            s=80, marker='x', color='gray', linewidths=2, label='OL')
ax.scatter([e.real for e in s1['lqg_obj'].ev_cl],
            [e.imag for e in s1['lqg_obj'].ev_cl],
            s=80, marker='o', color='#2E8B57',
            edgecolors='k', linewidths=1.5, label='LQG')
# PALF-LQG: same closed-loop poles as LQG
ax.scatter([e.real for e in s1['lqg_obj'].ev_cl],
            [e.imag for e in s1['lqg_obj'].ev_cl],
            s=120, marker='*', color='#DC143C',
            edgecolors='k', linewidths=1.5, label='PALF-LQG (= LQG)',
            alpha=0.6)
ax.axhline(0, color='k', linewidth=0.8)
ax.axvline(0, color='k', linewidth=0.8)
ax.set_xlabel("Re(s)", fontsize=12)
ax.set_ylabel("Im(s) (rad/s)", fontsize=12)
ax.set_title("Pôles boucle fermée (plan complexe)",
              fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.5)

plt.suptitle("Analyse modale boucle fermée",
              fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig06_poles.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"  ✓ fig06_poles.png")


# ============================================================
# FIGURE 7 : SLD - Stability Lobe Diagram
# ============================================================
print(f"\n=== SLD computation (peut prendre 2-3 minutes) ===")
t_sld = time.time()

# Setup SLD parameters
RPM_arr = np.linspace(2500, 7500, 30)
ap_arr  = np.linspace(0.0001, 4e-3, 25)

plate_nominal = build_plate(zp_pos=HP - 0.3e-3/2, freq_perturb=0.0)
RT = DT_TOOL/2
phi_st = np.pi - np.arccos(1 - AE/RT)
phi_ex = np.pi
k1_sld, k2_sld = cutting_constants(KN, MU_C, ETA_H, GAMMA_N)

# Average Dp across all positions
Dp_sample = []
for kp in range(0, 2001, 50):
    Dp_, _ = plate_nominal.get_Dp_at(kp)
    Dp_sample.append(Dp_)
Dp_avg = np.mean(Dp_sample, axis=0)
m_list = np.diag(plate_nominal.Mp).tolist()

# SLD design controller (same weights/observer as the time-domain LQG)
lqg_sld = LQGController(plate_nominal, dt=DT, verbose=False, kalman_V=KALMAN_V)
lqg_sld.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                          w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)

# SLD Open-Loop — rigorous COUPLED monodromy (A_ctrl=None): multi-mode, no surrogate
print(f"  ▷ SLD OPEN-LOOP (coupled monodromy)...")
rho_OL = compute_closed_loop_SLD(RPM_arr, ap_arr,
                                 plate_nominal.omega_n, ZETA, Dp_avg,
                                 None, None, None, None, None, None,
                                 NT, RT, ETA_H, phi_st, phi_ex,
                                 k1_sld, k2_sld, KT_NOMINAL, HP,
                                 m_div=20, verbose=False)
print(f"     done.")

# SLD LQG — rigorous CLOSED-LOOP monodromy with the controller (state feedback +
# Kalman observer + regenerative delay) in the loop. Replaces the old "equivalent
# damping" surrogate: no closed-loop-damping-into-open-loop-formula shortcut.
print(f"  ▷ SLD LQG (closed-loop monodromy, controller in the loop)...")
rho_LQG = compute_closed_loop_SLD(RPM_arr, ap_arr,
                                  plate_nominal.omega_n, ZETA, Dp_avg,
                                  plate_nominal.H_Pe_modal, plate_nominal.D_obs,
                                  lqg_sld.A, lqg_sld.B, lqg_sld.K_lqr, lqg_sld.L_kal,
                                  NT, RT, ETA_H, phi_st, phi_ex,
                                  k1_sld, k2_sld, KT_NOMINAL, HP,
                                  m_div=20, verbose=False)
print(f"     done.")

# SLD PALF-LQG : the feedforward is a phase-only learned map u_FF(phi), so its gain
# with respect to the estimated state is EXACTLY zero (du_FF/dx_hat = 0). It therefore
# does not enter the closed-loop Jacobian: the monodromy matrix — and hence every
# Floquet multiplier and the stability boundary — is IDENTICAL to LQG. Not an
# assertion now but a consequence of the phase-only architecture.
print(f"  ▷ SLD PALF-LQG (= LQG boundary; du_FF/dx_hat = 0)...")
rho_PALF = rho_LQG.copy()
print(f"     done. SLD computed in {time.time()-t_sld:.1f}s")

# Plot SLD
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

cases = [
    ("Open-Loop", rho_OL, 'Greys'),
    ("LQG", rho_LQG, 'Greens'),
    ("PALF-LQG (= LQG boundary)", rho_PALF, 'Reds'),
]

for ax, (title, rho_grid, cmap) in zip(axes, cases):
    # Stability map
    stable_mask = (rho_grid < 1.0).astype(float)
    
    cs = ax.contourf(RPM_arr, ap_arr*1e3, rho_grid,
                      levels=20, cmap=cmap, alpha=0.7)
    # Boundary line
    ax.contour(RPM_arr, ap_arr*1e3, rho_grid,
                levels=[1.0], colors='red', linewidths=2.5)
    
    # Operating point article
    ax.plot(4900, 0.3, '*', color='gold', markersize=22,
             markeredgecolor='k', markeredgewidth=1.5,
             label='Point article (4900 RPM, 0.3mm)', zorder=5)
    
    plt.colorbar(cs, ax=ax, label='ρ (rayon spectral)')
    ax.set_xlabel("RPM", fontsize=12)
    if title == "Open-Loop":
        ax.set_ylabel("$a_p$ (mm)", fontsize=12)
    ax.set_title(f"SLD {title}\nzones stables (ρ<1) en couleur claire",
                  fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.4)

plt.suptitle("Diagramme des Lobes de Stabilité (SLD) ـ FDM Floquet",
              fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig07_SLD_3panels.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"  ✓ fig07_SLD_3panels.png")

# Compute critical ap at RPM = 4900 (OL vs closed-loop LQG; PALF shares LQG boundary)
idx_rpm_4900 = np.argmin(np.abs(RPM_arr - 4900))
ap_crit_OL = None
ap_crit_LQG = None

for i_ap, ap_v in enumerate(ap_arr):
    if rho_OL[i_ap, idx_rpm_4900] >= 1.0 and ap_crit_OL is None:
        ap_crit_OL = ap_v
    if rho_LQG[i_ap, idx_rpm_4900] >= 1.0 and ap_crit_LQG is None:
        ap_crit_LQG = ap_v

if ap_crit_OL is None: ap_crit_OL = ap_arr[-1]
if ap_crit_LQG is None: ap_crit_LQG = ap_arr[-1]
ap_crit_PALF = ap_crit_LQG  # feedforward does not shift the boundary


# ============================================================
# FIGURE 8 : SLD overlay (comparaison directe)
# ============================================================
fig, ax = plt.subplots(figsize=(13, 7))

ax.contour(RPM_arr, ap_arr*1e3, rho_OL,
            levels=[1.0], colors='gray', linewidths=2.5,
            linestyles='-')
ax.contour(RPM_arr, ap_arr*1e3, rho_LQG,
            levels=[1.0], colors='#2E8B57', linewidths=2.5,
            linestyles='-')
# PALF shares the LQG boundary (feedforward does not shift it) — draw dashed on top
ax.contour(RPM_arr, ap_arr*1e3, rho_PALF,
            levels=[1.0], colors='#DC143C', linewidths=1.8,
            linestyles='--')

# Mark operating point
ax.plot(4900, 0.3, '*', color='gold', markersize=25,
         markeredgecolor='k', markeredgewidth=2,
         label='Point article (4900 RPM, 0.3mm)', zorder=10)

# Critical ap markers at RPM=4900
ax.axvline(4900, color='black', linestyle=':', alpha=0.4, linewidth=1)
ax.plot(4900, ap_crit_OL*1e3, 's', color='gray', markersize=10,
         markeredgecolor='k', label=f'$a_p^{{crit}}$ OL = {ap_crit_OL*1e3:.2f}mm')
ax.plot(4900, ap_crit_LQG*1e3, 's', color='#2E8B57', markersize=10,
         markeredgecolor='k',
         label=f'$a_p^{{crit}}$ LQG = PALF = {ap_crit_LQG*1e3:.2f}mm')

# Custom legend
custom_lines = [
    plt.Line2D([0], [0], color='gray', lw=2.5, label='Open-Loop'),
    plt.Line2D([0], [0], color='#2E8B57', lw=2.5, label='LQG (closed-loop)'),
    plt.Line2D([0], [0], color='#DC143C', lw=1.8, ls='--',
               label='PALF-LQG (= LQG boundary)'),
]

leg1 = ax.legend(handles=custom_lines, loc='upper left', fontsize=12,
                  title='Frontière stabilité', framealpha=0.95)
ax.add_artist(leg1)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

ax.set_xlabel("Vitesse de rotation RPM", fontsize=13)
ax.set_ylabel("Profondeur de coupe $a_p$ (mm)", fontsize=13)
ax.set_title(f"SLD (closed-loop) - LQG raises $a_p^{{crit}}$ to "
             f"{ap_crit_LQG*1e3:.2f}mm ({ap_crit_LQG/ap_crit_OL:.1f}x OL); "
             f"phase-locked feedforward does not shift the boundary",
              fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.5)
ax.set_xlim([RPM_arr.min(), RPM_arr.max()])
ax.set_ylim([0, ap_arr[-1]*1e3])

# Shaded regions for clarity
ax.fill_between(RPM_arr, 0, ap_arr[-1]*1e3, where=np.zeros_like(RPM_arr),
                  alpha=0)  # placeholder

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig08_SLD_overlay.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"  ✓ fig08_SLD_overlay.png")


# ============================================================
# FIGURE 9 : Robustness summary  (multiple metrics)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

metrics_list = [
    ('y_max', 'y_max (µm)', 'Maximum vibration'),
    ('y_rms', 'y_RMS (µm)', 'RMS vibration'),
    ('y_p2p', 'y_pp (µm)', 'Peak-to-peak vibration'),
    ('u_max', 'u_max (V)', 'Tension piezo pic'),
    ('u_rms', 'u_RMS (V)', 'Tension piezo RMS'),
    ('du_max', 'Δu_max (V)', 'Variation max piezo'),
]

for idx, (metric, ylabel, title) in enumerate(metrics_list):
    ax = axes[idx // 3, idx % 3]
    
    lqg_v = [s['metrics']['LQG'][metric] for s in scenarios]
    darc_v = [s['metrics'][FF_LABEL][metric] for s in scenarios]
    
    x = np.arange(len(scenarios))
    w_b = 0.35
    
    bars1 = ax.bar(x - w_b/2, lqg_v, w_b, color='#2E8B57', alpha=0.85,
                     edgecolor='k', label='LQG', linewidth=1.2)
    bars2 = ax.bar(x + w_b/2, darc_v, w_b, color='#DC143C', alpha=0.85,
                     edgecolor='k', label='PALF-LQG', linewidth=1.2)
    
    for bar, v in zip(bars1, lqg_v):
        ax.text(bar.get_x() + bar.get_width()/2, v*1.02, f'{v:.2f}',
                 ha='center', fontsize=8, fontweight='bold')
    for bar, v in zip(bars2, darc_v):
        ax.text(bar.get_x() + bar.get_width()/2, v*1.02, f'{v:.2f}',
                 ha='center', fontsize=8, fontweight='bold', color='darkred')
    
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.5)

plt.suptitle("Comparaison multi-métriques : LQG vs PALF-LQG",
              fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig09_metrics_grid.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"  ✓ fig09_metrics_grid.png")


# ============================================================
# RÉSUMÉ FINAL TEXTUEL
# ============================================================
print(f"\n{'='*72}")
print(f" RÉSUMÉ FINAL : LQG vs PALF-LQG ")
print(f"{'='*72}")

print(f"\n  PERFORMANCE VIBRATOIRE :")
print(f"  {'Scénario':<28}{'LQG y_RMS':<14}{'PALF y_RMS':<14}{'Gain':<10}")
print(f"  {'-'*70}")
for s in scenarios:
    yL = s['metrics']['LQG']['y_rms']
    yD = s['metrics'][FF_LABEL]['y_rms']
    gain = (1 - yD/yL)*100
    print(f"  {s['name']:<28}{yL:<14.4f}{yD:<14.4f}{gain:+6.2f}%")

mean_L = np.mean([s['metrics']['LQG']['y_rms'] for s in scenarios])
mean_D = np.mean([s['metrics'][FF_LABEL]['y_rms'] for s in scenarios])
print(f"  {'-'*70}")
print(f"  {'MOYENNE':<28}{mean_L:<14.4f}{mean_D:<14.4f}{(1-mean_D/mean_L)*100:+6.2f}%")

print(f"\n  STABILITÉ (SLD) - à RPM = 4900 :")
print(f"     a_p crit OPEN-LOOP : {ap_crit_OL*1e3:.3f} mm")
print(f"     a_p crit LQG       : {ap_crit_LQG*1e3:.3f} mm  "
      f"({ap_crit_LQG/ap_crit_OL:.1f}x OL)")
print(f"     a_p crit PALF-LQG   : {ap_crit_PALF*1e3:.3f} mm  "
      f"({ap_crit_PALF/ap_crit_OL:.1f}x OL)  [= LQG; FF does not shift boundary]")

print(f"\n  EFFORT DE COMMANDE (S1 nominal) :")
s_n = scenarios[0]
print(f"     LQG    : u_max = {s_n['metrics']['LQG']['u_max']:.2f}V, "
      f"u_RMS = {s_n['metrics']['LQG']['u_rms']:.2f}V")
print(f"     PALF-LQG: u_max = {s_n['metrics'][FF_LABEL]['u_max']:.2f}V, "
      f"u_RMS = {s_n['metrics'][FF_LABEL]['u_rms']:.2f}V")

print(f"\nTemps total : {time.time()-t_global:.1f} s")
print(f"\n  >>> 9 figures générées dans {OUT_DIR}/")
