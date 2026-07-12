"""
main_simulation.py
==================
COMPARAISON :  LQG  vs  DARC (Deep Anticipative Residual Control)

PROTOCOLE HONNÊTE (voir docs/AUDIT_SCIENTIFIQUE.md) :
   - Le feedforward et le NN du DARC sont conçus/entraînés à partir du modèle
     NOMINAL (K_T nominal, plaque nominale, engagement commandé) — jamais à
     partir des paramètres réels perturbés du scénario.  Dans S4 (+30 % K_T),
     le contrôleur subit donc réellement l'erreur de modèle.
   - Le NN est entraîné sur des épisodes à réalisations de bruit distinctes,
     avec sélection du meilleur checkpoint sur un épisode de VALIDATION
     (graine dédiée) — jamais sur l'épisode d'évaluation (graine 1).
   - Le SLD est une analyse de Floquet de la boucle COMPLÈTE (plaque
     multi-modes couplée + retard + observateur de Kalman + retour discret) —
     plus de « substitution de pôles ».  La courbe DARC N'EST PAS tracée à
     part : un feedforward périodique verrouillé en phase est une entrée
     exogène qui ne modifie pas l'équation variationnelle homogène, donc
     SLD(DARC) = SLD(LQG) exactement dans ce cadre linéaire.

Mesures :
   1. PERFORMANCE TEMPORELLE (4 scénarios) : y(t), u(t), y_max, y_RMS, u_max
   2. ANALYSE FRÉQUENTIELLE (FFT)
   3. STABILITY LOBE DIAGRAM (Floquet boucle fermée)
   4. ROBUSTESSE AU CAPTEUR (balayage du bruit de mesure)
   5. EFFORT DE COMMANDE
   6. PÔLES BOUCLE FERMÉE

Sortie : 9 figures + tableau récapitulatif
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plate_model import PlateModel
from milling_force import precompute_alpha_periodic, precompute_nonlinear_periodic
from lqg_controller import LQGController
from darc_controller import DARCController
from newmark_solver import NewmarkSimulator
from fdm_stability import compute_SLD_closed_loop


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
N_MODES = 3
ZETA = [0.0031, 0.0017, 0.0027]

DT = 5e-5
T_END = 0.5

# Plancher de vibration par le CAPTEUR (réglage du capteur).  Le capteur a une
# résolution / bruit de plancher finie : il ne « voit » pas la vibration sous ce
# seuil, donc le contrôleur ne peut PAS la réduire sous ~ce niveau.  La vibration
# commandée RÉELLE reste ainsi >= 1 µm (limite physique réaliste du capteur, à la
# place d'une zone morte volontaire du contrôleur).  Mettre 0 pour un capteur idéal.
# Capteur de déplacement FIN (eddy-current réaliste) pour la comparaison
# principale équitable. Le feedforward du DARC étant synchronisé sur la phase
# (encodeur broche), il est INDÉPENDANT de ce capteur -> DARC garde son avantage
# même quand le capteur se dégrade (voir la figure de robustesse capteur).
SENSOR_NOISE = 1.0e-7   # 0.1 µm RMS (bruit capteur fin)
SENSOR_FLOOR = 0.0      # pas de zone morte (capteur fin, pleine résolution)
FF_HARMONICS = 30       # harmoniques de passage de dent annulées par le feedforward
NN_ITERS = 20           # itérations d'apprentissage du résidu non linéaire par le NN

OUT_DIR = "figs_lqg_vs_darc_v3"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================
def build_plate(zp_pos, freq_perturb=0.0):
    plate = PlateModel(LP, HP, 0.004, RHO, E_AL, NU_AL,
                       N1=N1, N2=N2, n_modes=N_MODES,
                       zeta_modes=ZETA, verbose=False)
    plate.precompute_Dp(zp_pos=zp_pos, n_pos=2001)
    plate.set_observation(x_obs=LP, z_obs=HP)
    plate.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    if abs(freq_perturb) > 1e-6:
        scale_K = (1 + freq_perturb)**2
        plate.Kp = plate.Kp * scale_K
        plate.omega_n = np.sqrt(np.diag(plate.Kp) / np.diag(plate.Mp))
        plate.freq_n = plate.omega_n / (2*np.pi)
        plate.Cp = np.diag(2 * np.array(ZETA) * plate.omega_n * np.diag(plate.Mp))
    return plate


def reset_darc_histories(darc):
    darc.history_u_lqg = []
    darc.history_u_ff = []
    darc.history_u_total = []
    darc.history_phase = []


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
print(" COMPARAISON COMPLETE  :  LQG  vs  DARC ")
print("="*72)
t_global = time.time()

scenarios_def = [
    ("S1 - Nominal article",      0.3e-3, KT_NOMINAL, 0.0),
    ("S2 - Aggressive ap=0.6mm",  0.6e-3, KT_NOMINAL, 0.0),
    ("S3 - Uncertainty ω-15%",    0.3e-3, KT_NOMINAL, -0.15),
    ("S4 - High K_T +30%",        0.3e-3, 1.3*KT_NOMINAL, 0.0),
]


def run_scenario(name, ap, KT_actual, freq_perturb):
    print(f"\n{'='*72}")
    print(f" {name}")
    print(f"  ap = {ap*1e3:.2f}mm,  KT = {KT_actual/1e6:.0f}MPa,  "
          f"freq_drift = {freq_perturb*100:+.0f}%")
    print(f"{'='*72}")
    
    plate_d = build_plate(zp_pos=HP - ap/2, freq_perturb=0.0)
    plate_r = build_plate(zp_pos=HP - ap/2, freq_perturb=freq_perturb)
    
    phi_st = np.pi - np.arccos(1 - AE/(DT_TOOL/2))
    phi_ex = np.pi
    Omega = 2*np.pi*RPM/60
    RT = DT_TOOL/2
    tau = 60/(NT*RPM)
    k1 = KN*np.cos(ETA_H)
    k2 = 1 + MU_C*np.tan(ETA_H)*np.cos(GAMMA_N) - KN*np.sin(ETA_H)
    f_t = NT * RPM / 60
    
    sim = NewmarkSimulator(plate_r, dt=DT, T_end=T_END, ft=FT, tau=tau, verbose=False)
    sim_d = NewmarkSimulator(plate_d, dt=DT, T_end=T_END, ft=FT, tau=tau, verbose=False)
    v_feed = FT * NT * RPM / 60
    xp_path = np.minimum(v_feed * sim.t_vec, LP)
    kp_idx = np.clip(np.round(xp_path/LP * 2000).astype(int), 0, 2000)
    n_per = int(np.round(tau/DT))
    # Modèle de coupe : ossature linéaire (alpha3, alpha4) + termes quadratique
    # et cubique (alpha4_2, alpha4_3) de la régénération non linéaire.
    #   - Coefficients RÉELS (K_T réel du scénario) -> pilotent le PROCÉDÉ simulé
    alpha3, alpha4, alpha4_2, alpha4_3 = precompute_nonlinear_periodic(
        DT, n_per, sim.nstep,
        Omega, NT, RT, ETA_H, phi_st, phi_ex, HP-ap, HP, k1, k2, KT_actual)
    #   - Coefficients NOMINAUX (K_T nominal, engagement commandé connu) -> le
    #     MODÈLE dont dispose le contrôleur.  PROTOCOLE HONNÊTE : le DARC ne
    #     reçoit JAMAIS les coefficients réels perturbés ; en S4 (K_T +30 %)
    #     son feedforward est donc réellement dé-calibré, c'est le test.
    if KT_actual != KT_NOMINAL:
        alpha3_n, alpha4_n, alpha4_2_n, alpha4_3_n = precompute_nonlinear_periodic(
            DT, n_per, sim.nstep,
            Omega, NT, RT, ETA_H, phi_st, phi_ex, HP-ap, HP, k1, k2, KT_NOMINAL)
    else:
        alpha3_n, alpha4_n, alpha4_2_n, alpha4_3_n = (alpha3, alpha4,
                                                      alpha4_2, alpha4_3)

    # === LQG (baseline réactif) ===
    # Comparaison ÉQUITABLE : le LQG est EXACTEMENT la base LQG interne du
    # DARC (w_q=1e14, w_qd=1e8, R=1, gain_norm_max élevé pour reproduire la
    # résolution Riccati non contrainte). La seule différence du DARC est donc
    # sa correction feedforward (+NN), ce qui mesure son apport marginal réel.
    lqg = LQGController(plate_d, dt=DT, verbose=False)
    lqg.optimize_weights(w_q_list=[1e14],
                          w_qd_list=[1e8], w_r=1.0, gain_norm_max=1e10)
    lqg.discretize_observer()
    res_lqg = sim.simulate(alpha3, alpha4, kp_idx, controller=lqg,
                           alpha4_2_t=alpha4_2, alpha4_3_t=alpha4_3,
                           sensor_floor=SENSOR_FLOOR, sensor_noise=SENSOR_NOISE,
                           sensor_rng=np.random.default_rng(1),
                           progress=False)

    # === DARC (LQG + feedforward inverse-modèle + résidu NN) ===
    # MÊME base LQG que ci-dessus (w_q=1e14).  Le feedforward est conçu à
    # partir du modèle NOMINAL (alpha3_n) et de la boucle fermée nominale ; le
    # NN est entraîné dans le monde nominal (sim_d) avec des réalisations de
    # bruit d'entraînement/validation distinctes de l'évaluation (graine 1).
    darc = DARCController(plate_d, dt=DT,
                          base_w_q=1e14, base_w_qd=1e8, base_w_r=1.0,
                          ff_alpha=1.0, ff_max=20.0, n_per=n_per,
                          u_max=150.0, verbose=False)
    Dp_ff = plate_d.get_Dp_at(int(kp_idx[sim.nstep // 2]))[0]
    darc.design_periodic_feedforward(FT, alpha3_n[:n_per], Dp_ff,
                                     n_harm=FF_HARMONICS)
    # « Deep » : apprentissage du résidu dans le MONDE NOMINAL, avec sélection
    # de checkpoint sur épisode de VALIDATION (graines train 100+, val 200).
    darc.train_nn_residual(sim_d, alpha3_n, alpha4_n, kp_idx,
                           alpha4_2_t=alpha4_2_n, alpha4_3_t=alpha4_3_n,
                           n_iter=NN_ITERS,
                           sensor_noise=SENSOR_NOISE, sensor_floor=SENSOR_FLOOR,
                           train_seed=100, val_seed=200, verbose=False)
    reset_darc_histories(darc)
    res_darc = sim.simulate(alpha3, alpha4, kp_idx, controller=darc,
                            alpha4_2_t=alpha4_2, alpha4_3_t=alpha4_3,
                            sensor_floor=SENSOR_FLOOR, sensor_noise=SENSOR_NOISE,
                            sensor_rng=np.random.default_rng(1),
                            progress=False)
    
    # Compute metrics
    metrics = {}
    for ctrl, res in [('LQG', res_lqg), ('DARC', res_darc)]:
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
    
    print(f"  {'Metrique':<22}{'LQG':<14}{'DARC':<14}{'delta':<10}")
    print(f"  {'-'*60}")
    for key in ['y_max', 'y_rms', 'y_p2p', 'u_max', 'u_rms']:
        l = metrics['LQG'][key]
        d = metrics['DARC'][key]
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
        'plate_d': plate_d, 'plate_r': plate_r,
        'lqg_obj': lqg, 'darc_obj': darc,
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
y_darc = [s['metrics']['DARC']['y_rms'] for s in scenarios]
b1 = ax.bar(x - w/2, y_lqg, w, color='#2E8B57', alpha=0.85,
              edgecolor='k', label='LQG', linewidth=1.5)
b2 = ax.bar(x + w/2, y_darc, w, color='#DC143C', alpha=0.85,
              edgecolor='k', label='DARC', linewidth=1.5)
for bar, v in zip(b1, y_lqg):
    ax.text(bar.get_x()+bar.get_width()/2, v*1.02, f'{v:.3f}',
             ha='center', fontsize=9, fontweight='bold')
for bar, v in zip(b2, y_darc):
    ax.text(bar.get_x()+bar.get_width()/2, v*1.02, f'{v:.3f}',
             ha='center', fontsize=9, fontweight='bold', color='darkred')
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
ax.set_ylabel("$y_{RMS}$ (µm)", fontsize=12)
ax.set_title("Vibration RMS : LQG vs DARC", fontsize=12, fontweight='bold')
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
ax.set_ylabel("Δ y_RMS DARC vs LQG (%)", fontsize=12)
ax.set_title("Gain de réduction de vibration (positif = amélioration)",
              fontsize=12, fontweight='bold')
ax.grid(True, axis='y', alpha=0.5)

# (c) u_max
ax = axes[2]
u_lqg = [s['metrics']['LQG']['u_max'] for s in scenarios]
u_darc = [s['metrics']['DARC']['u_max'] for s in scenarios]
b1 = ax.bar(x - w/2, u_lqg, w, color='#2E8B57', alpha=0.85,
              edgecolor='k', label='LQG', linewidth=1.5)
b2 = ax.bar(x + w/2, u_darc, w, color='#DC143C', alpha=0.85,
              edgecolor='k', label='DARC', linewidth=1.5)
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

plt.suptitle(" LQG vs DARC : Bilan global ",
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
    res_darc = s['metrics']['DARC']['res']
    
    ax = axes[i]
    i_end = min(res_lqg['stop_idx'], res_darc['stop_idx'])
    
    ax.plot(t_ms[:i_end+1], res_lqg['y'][:i_end+1]*1e6,
             color='#2E8B57', linewidth=0.5, alpha=0.85,
             label=f"LQG (RMS={s['metrics']['LQG']['y_rms']:.3f}µm)")
    ax.plot(t_ms[:i_end+1], res_darc['y'][:i_end+1]*1e6,
             color='#DC143C', linewidth=0.5, alpha=0.85,
             label=f"DARC (RMS={s['metrics']['DARC']['y_rms']:.3f}µm)")
    
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
    res_darc = s['metrics']['DARC']['res']
    
    ax = axes[i]
    i_end = min(res_lqg['stop_idx'], res_darc['stop_idx'])
    
    ax.plot(t_ms[:i_end+1], res_lqg['u'][:i_end+1],
             color='#2E8B57', linewidth=0.5, alpha=0.85,
             label=f"LQG (max={s['metrics']['LQG']['u_max']:.1f}V)")
    ax.plot(t_ms[:i_end+1], res_darc['u'][:i_end+1],
             color='#DC143C', linewidth=0.5, alpha=0.85,
             label=f"DARC (max={s['metrics']['DARC']['u_max']:.1f}V)")
    
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
    res_darc = s['metrics']['DARC']['res']
    f_t = s['f_t']
    plate = s['plate_r']
    
    i_end_l = res_lqg['stop_idx']
    i_end_d = res_darc['stop_idx']
    
    f_l, Y_l = fft_amp(res_lqg['y'][:i_end_l+1], DT)
    f_d, Y_d = fft_amp(res_darc['y'][:i_end_d+1], DT)
    
    ax.plot(f_l, Y_l, color='#2E8B57', linewidth=1.5, alpha=0.9,
             label=f"LQG (max={Y_l.max():.3f}µm)")
    ax.plot(f_d, Y_d, color='#DC143C', linewidth=1.5, alpha=0.9,
             label=f"DARC (max={Y_d.max():.3f}µm)")
    
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
    res_darc = s['metrics']['DARC']['res']
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
    ax.semilogy(f_d, np.maximum(U_d, 1e-3), color='#DC143C', linewidth=1.2, label='DARC')
    
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

# DARC closed-loop : approximate by LQG closed-loop (DARC uses same K_lqr base)
# In reality, DARC adds NN_FF which doesn't change CL poles directly
# So poles are same as LQG. We display difference indicator.

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# (a) Damping ratios
ax = axes[0]
modes_x = np.arange(N_MODES)
w = 0.30
ax.bar(modes_x - w, zeta_OL*100, w, color='gray', alpha=0.7,
        edgecolor='k', label='Open-loop', linewidth=1.5)
ax.bar(modes_x, zeta_LQG*100, w, color='#2E8B57', alpha=0.85,
        edgecolor='k', label='LQG', linewidth=1.5)
# DARC has LQG base + NN_FF (FF doesn't change CL poles)
# The "effective damping" is improved by FF action
# Display LQG poles, label with note
ax.bar(modes_x + w, zeta_LQG*100, w, color='#DC143C', alpha=0.85,
        edgecolor='k', label='DARC (base LQG)', linewidth=1.5,
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
# DARC same closed-loop poles (as base = LQG)
ax.scatter([e.real for e in s1['lqg_obj'].ev_cl],
            [e.imag for e in s1['lqg_obj'].ev_cl],
            s=120, marker='*', color='#DC143C',
            edgecolors='k', linewidths=1.5, label='DARC (base)',
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
# FIGURE 7 : SLD - Stability Lobe Diagram (Floquet boucle fermée)
#
# HONNÊTETÉ (voir docs/AUDIT_SCIENTIFIQUE.md) :
#  - Les courbes OL et LQG sont des analyses de Floquet du système COMPLET :
#    plaque 3 modes couplée (Dp·Dpᵀ) + retard régénératif + (pour LQG)
#    observateur de Kalman et retour discret AUGMENTÉS dans la monodromie.
#    (Plus de « substitution de pôles » eig(A-BK) sans observateur.)
#  - Il n'y a PAS de courbe DARC séparée : le feedforward périodique
#    verrouillé en phase est une entrée exogène indépendante de l'état ; il
#    ne modifie pas l'équation variationnelle homogène, donc les
#    multiplicateurs de Floquet sont STRICTEMENT inchangés :
#    SLD(DARC) = SLD(LQG).  (L'ancien facteur d'amortissement 1.30x était
#    une heuristique sans base théorique — supprimé.)
#  - LIMITES : analyse linéaire (saturation ±150 V exclue -> la frontière
#    n'est valide que là où u < 150 V) ; couplage outil pris à Dp moyenné le
#    long de la passe (le mode 2, antisymétrique, y est invisible — voir
#    l'audit, P1).
# ============================================================
print(f"\n=== SLD computation (Floquet boucle fermée) ===")
t_sld = time.time()

# Setup SLD parameters
RPM_arr = np.linspace(2500, 7500, 30)
ap_arr  = np.linspace(0.0001, 4e-3, 25)

plate_nominal = build_plate(zp_pos=HP - 0.3e-3/2, freq_perturb=0.0)
RT = DT_TOOL/2
phi_st = np.pi - np.arccos(1 - AE/RT)
phi_ex = np.pi
k1_sld = KN*np.cos(ETA_H)
k2_sld = 1 + MU_C*np.tan(ETA_H)*np.cos(GAMMA_N) - KN*np.sin(ETA_H)

# Average Dp across all positions (limite documentée : mode 2 invisible)
Dp_sample = []
for kp in range(0, 2001, 50):
    Dp_, _ = plate_nominal.get_Dp_at(kp)
    Dp_sample.append(Dp_)
Dp_avg = np.mean(Dp_sample, axis=0)

# SLD Open-Loop (même cadre de Floquet multi-modes -> comparabilité directe)
print(f"  ▷ SLD OPEN-LOOP (Floquet multi-modes)...")
rho_OL = compute_SLD_closed_loop(RPM_arr, ap_arr, plate_nominal,
                                 NT, RT, ETA_H, phi_st, phi_ex,
                                 k1_sld, k2_sld, KT_NOMINAL, HP, Dp_avg,
                                 lqg=None, dt_c=DT, verbose=False)
print(f"     done.")

# SLD LQG : compensateur COMPLET (Kalman + LQR discret) dans la monodromie.
# MÊME base que la comparaison temporelle (w_q=1e14, w_qd=1e8) — plus de
# base « sous-optimale » différente pour le SLD.
print(f"  ▷ SLD LQG (compensateur dans la monodromie)...")
lqg_sld = LQGController(plate_nominal, dt=DT, verbose=False)
lqg_sld.optimize_weights(w_q_list=[1e14], w_qd_list=[1e8], w_r=1.0,
                         gain_norm_max=1e10)
rho_LQG = compute_SLD_closed_loop(RPM_arr, ap_arr, plate_nominal,
                                  NT, RT, ETA_H, phi_st, phi_ex,
                                  k1_sld, k2_sld, KT_NOMINAL, HP, Dp_avg,
                                  lqg=lqg_sld, dt_c=DT, verbose=False)
print(f"     done. SLD computed in {time.time()-t_sld:.1f}s")

# SLD DARC = SLD LQG (le feedforward exogène ne change pas la monodromie)
rho_DARC = rho_LQG

# Plot SLD
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

cases = [
    ("Open-Loop", rho_OL, 'Greys'),
    ("LQG", rho_LQG, 'Greens'),
    ("DARC (= LQG : FF exogène,\nFloquet inchangé)", rho_DARC, 'Reds'),
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

# Compute critical ap for each method at RPM = 4900
# (si la frontière n'est pas croisée dans la grille, la valeur est une BORNE
#  INFÉRIEURE « > ap_max » — ne JAMAIS la rapporter comme un croisement.)
idx_rpm_4900 = np.argmin(np.abs(RPM_arr - 4900))
ap_crit_OL = None
ap_crit_LQG = None

for i_ap, ap_v in enumerate(ap_arr):
    if rho_OL[i_ap, idx_rpm_4900] >= 1.0 and ap_crit_OL is None:
        ap_crit_OL = ap_v
    if rho_LQG[i_ap, idx_rpm_4900] >= 1.0 and ap_crit_LQG is None:
        ap_crit_LQG = ap_v

ap_crit_OL_capped = ap_crit_OL is None
ap_crit_LQG_capped = ap_crit_LQG is None
if ap_crit_OL is None: ap_crit_OL = ap_arr[-1]
if ap_crit_LQG is None: ap_crit_LQG = ap_arr[-1]
ap_crit_DARC = ap_crit_LQG          # FF exogène : même frontière que LQG
ap_crit_DARC_capped = ap_crit_LQG_capped


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

# Mark operating point
ax.plot(4900, 0.3, '*', color='gold', markersize=25,
         markeredgecolor='k', markeredgewidth=2,
         label='Point article (4900 RPM, 0.3mm)', zorder=10)

# Critical ap markers at RPM=4900
_cap_l = ">" if ap_crit_LQG_capped else "="
_cap_o = ">" if ap_crit_OL_capped else "="
ax.axvline(4900, color='black', linestyle=':', alpha=0.4, linewidth=1)
ax.plot(4900, ap_crit_OL*1e3, 's', color='gray', markersize=10,
         markeredgecolor='k',
         label=f'$a_p^{{crit}}$ OL {_cap_o} {ap_crit_OL*1e3:.2f}mm')
ax.plot(4900, ap_crit_LQG*1e3, 's', color='#2E8B57', markersize=10,
         markeredgecolor='k',
         label=f'$a_p^{{crit}}$ LQG/DARC {_cap_l} {ap_crit_LQG*1e3:.2f}mm')

# Custom legend
custom_lines = [
    plt.Line2D([0], [0], color='gray', lw=2.5, label='Open-Loop'),
    plt.Line2D([0], [0], color='#2E8B57', lw=2.5,
               label='LQG = DARC (FF exogène : Floquet inchangé)'),
]

leg1 = ax.legend(handles=custom_lines, loc='upper left', fontsize=12,
                  title='Frontière stabilité (Floquet boucle fermée)',
                  framealpha=0.95)
ax.add_artist(leg1)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

ax.set_xlabel("Vitesse de rotation RPM", fontsize=13)
ax.set_ylabel("Profondeur de coupe $a_p$ (mm)", fontsize=13)
ax.set_title(f"SLD Floquet boucle fermée — $a_p^{{crit}}$ LQG "
             f"{_cap_l} {ap_crit_LQG/ap_crit_OL:.1f}x boucle ouverte\n"
             f"(analyse linéaire : validité limitée à u < 150 V ; "
             f"le feedforward du DARC ne déplace pas cette frontière)",
              fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.5)
ax.set_xlim([RPM_arr.min(), RPM_arr.max()])
ax.set_ylim([0, ap_arr[-1]*1e3])

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
    darc_v = [s['metrics']['DARC'][metric] for s in scenarios]
    
    x = np.arange(len(scenarios))
    w_b = 0.35
    
    bars1 = ax.bar(x - w_b/2, lqg_v, w_b, color='#2E8B57', alpha=0.85,
                     edgecolor='k', label='LQG', linewidth=1.2)
    bars2 = ax.bar(x + w_b/2, darc_v, w_b, color='#DC143C', alpha=0.85,
                     edgecolor='k', label='DARC', linewidth=1.2)
    
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

plt.suptitle("Comparaison multi-métriques : LQG vs DARC",
              fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig09_metrics_grid.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"  ✓ fig09_metrics_grid.png")


# ============================================================
# FIGURE 10 : ROBUSTESSE AU CAPTEUR (argument clé pour le DARC)
#   Le feedforward anticipatif du DARC est synchronisé sur la phase (encodeur
#   broche) -> INDÉPENDANT du capteur de déplacement. Quand le capteur se
#   dégrade (bruit croissant), le LQG (retour pur) se détériore fortement,
#   tandis que le DARC garde son avantage. Comparaison ÉQUITABLE (même base LQG).
# ============================================================
print(f"\n=== Robustesse au capteur (LQG vs DARC) ===")
tau_sr = 60/(NT*RPM); n_per_sr = int(round(tau_sr/DT))
plate_sr = build_plate(zp_pos=HP - 0.3e-3/2, freq_perturb=0.0)
sim_sr = NewmarkSimulator(plate_sr, dt=DT, T_end=T_END, ft=FT, tau=tau_sr, verbose=False)
xp_sr = np.minimum(FT*NT*RPM/60 * sim_sr.t_vec, LP)
kp_sr = np.clip(np.round(xp_sr/LP * 2000).astype(int), 0, 2000)
Om_sr = 2*np.pi*RPM/60; RT_sr = DT_TOOL/2
phi_st_sr = np.pi - np.arccos(1 - AE/RT_sr)
k1_sr = KN*np.cos(ETA_H); k2_sr = 1 + MU_C*np.tan(ETA_H)*np.cos(GAMMA_N) - KN*np.sin(ETA_H)
a3_sr, a4_sr, a42_sr, a43_sr = precompute_nonlinear_periodic(
    DT, n_per_sr, sim_sr.nstep, Om_sr, NT, RT_sr, ETA_H, phi_st_sr, np.pi,
    HP-0.3e-3, HP, k1_sr, k2_sr, KT_NOMINAL)
Dp_ff_sr = plate_sr.get_Dp_at(int(kp_sr[sim_sr.nstep//2]))[0]

noise_levels = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]) * 1e-6
# DARC (FF inverse + NN) entraîné UNE fois, au niveau de bruit NOMINAL et avec
# des graines train/validation dédiées ; le même DARC (table FF indexée par la
# phase) sert ensuite à tous les niveaux de bruit.  NOTE d'équité : le capteur
# de POSITION est dégradé pour les deux contrôleurs, mais la référence de
# PHASE broche du feedforward reste ici idéale — la sensibilité du DARC à une
# erreur de phase (encodeur) est une étude séparée (voir audit, P1).
darc_sr = DARCController(plate_sr, dt=DT, base_w_q=1e14, base_w_qd=1e8,
                         ff_alpha=1.0, ff_max=20.0,
                         n_per=n_per_sr, u_max=150.0, verbose=False)
darc_sr.design_periodic_feedforward(FT, a3_sr[:n_per_sr], Dp_ff_sr, n_harm=FF_HARMONICS)
darc_sr.train_nn_residual(sim_sr, a3_sr, a4_sr, kp_sr,
                          alpha4_2_t=a42_sr, alpha4_3_t=a43_sr, n_iter=NN_ITERS,
                          sensor_noise=SENSOR_NOISE, sensor_floor=SENSOR_FLOOR,
                          train_seed=300, val_seed=400, verbose=False)
rms_lqg_sr, rms_darc_sr = [], []
for sn in noise_levels:
    lqg_sr = LQGController(plate_sr, dt=DT, verbose=False)
    lqg_sr.optimize_weights(w_q_list=[1e14], w_qd_list=[1e8], w_r=1.0, gain_norm_max=1e10)
    lqg_sr.discretize_observer()
    r_l = sim_sr.simulate(a3_sr, a4_sr, kp_sr, controller=lqg_sr,
                          alpha4_2_t=a42_sr, alpha4_3_t=a43_sr,
                          sensor_noise=sn, sensor_rng=np.random.default_rng(7), progress=False)
    reset_darc_histories(darc_sr)
    r_d = sim_sr.simulate(a3_sr, a4_sr, kp_sr, controller=darc_sr,
                          alpha4_2_t=a42_sr, alpha4_3_t=a43_sr,
                          sensor_noise=sn, sensor_rng=np.random.default_rng(7), progress=False)
    il = r_l['stop_idx']; idd = r_d['stop_idx']
    rms_lqg_sr.append(np.sqrt(np.mean(r_l['y'][:il+1]**2))*1e6)
    rms_darc_sr.append(np.sqrt(np.mean(r_d['y'][:idd+1]**2))*1e6)
    print(f"   noise={sn*1e6:.1f}um : LQG={rms_lqg_sr[-1]:.3f}  DARC={rms_darc_sr[-1]:.3f} um "
          f"(DARC {(1-rms_darc_sr[-1]/rms_lqg_sr[-1])*100:+.0f}%)")

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(noise_levels*1e6, rms_lqg_sr, 'o-', color='#2E8B57', lw=2.5, ms=8,
        label='LQG (retour pur — dépend du capteur)')
ax.plot(noise_levels*1e6, rms_darc_sr, 's-', color='#DC143C', lw=2.5, ms=8,
        label='DARC (feedforward indépendant du capteur)')
ax.fill_between(noise_levels*1e6, rms_darc_sr, rms_lqg_sr, color='#DC143C', alpha=0.12)
ax.set_xlabel("Bruit / dégradation du capteur (µm RMS)", fontsize=12)
ax.set_ylabel("Vibration commandée $y_{RMS}$ (µm)", fontsize=12)
ax.set_title("Robustesse au capteur — comparaison ÉQUITABLE (même base LQG)\n"
             "le feedforward anticipatif du DARC ne dépend pas du capteur",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='upper left'); ax.grid(True, alpha=0.5)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig10_sensor_robustness.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"  ✓ fig10_sensor_robustness.png")


# ============================================================
# RÉSUMÉ FINAL TEXTUEL
# ============================================================
print(f"\n{'='*72}")
print(f" RÉSUMÉ FINAL : LQG vs DARC ")
print(f"{'='*72}")

print(f"\n  PERFORMANCE VIBRATOIRE :")
print(f"  {'Scénario':<28}{'LQG y_RMS':<14}{'DARC y_RMS':<14}{'Gain':<10}")
print(f"  {'-'*70}")
for s in scenarios:
    yL = s['metrics']['LQG']['y_rms']
    yD = s['metrics']['DARC']['y_rms']
    gain = (1 - yD/yL)*100
    print(f"  {s['name']:<28}{yL:<14.4f}{yD:<14.4f}{gain:+6.2f}%")

mean_L = np.mean([s['metrics']['LQG']['y_rms'] for s in scenarios])
mean_D = np.mean([s['metrics']['DARC']['y_rms'] for s in scenarios])
print(f"  {'-'*70}")
print(f"  {'MOYENNE':<28}{mean_L:<14.4f}{mean_D:<14.4f}{(1-mean_D/mean_L)*100:+6.2f}%")

print(f"\n  STABILITÉ (SLD) - à RPM = 4900 :")
print(f"     a_p crit OPEN-LOOP : {ap_crit_OL*1e3:.3f} mm")
print(f"     a_p crit LQG       : {ap_crit_LQG*1e3:.3f} mm  "
      f"({ap_crit_LQG/ap_crit_OL:.1f}x OL)")
print(f"     a_p crit DARC   : {ap_crit_DARC*1e3:.3f} mm  "
      f"({ap_crit_DARC/ap_crit_OL:.1f}x OL)")

print(f"\n  EFFORT DE COMMANDE (S1 nominal) :")
s_n = scenarios[0]
print(f"     LQG    : u_max = {s_n['metrics']['LQG']['u_max']:.2f}V, "
      f"u_RMS = {s_n['metrics']['LQG']['u_rms']:.2f}V")
print(f"     DARC: u_max = {s_n['metrics']['DARC']['u_max']:.2f}V, "
      f"u_RMS = {s_n['metrics']['DARC']['u_rms']:.2f}V")

print(f"\nTemps total : {time.time()-t_global:.1f} s")
print(f"\n  >>> 9 figures générées dans {OUT_DIR}/")
