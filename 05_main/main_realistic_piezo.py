"""
main_realistic_piezo.py
========================
Simulation avec modèle PIEZO RÉALISTE :
   - Saturation de tension (±150 V)
   - Slew-rate limité (300 V/ms)
   - Dynamique d'amplificateur (BW 5 kHz)
   - Hystérésis matérielle (10%)
   - Bruit de mesure capteur (0.1 µm RMS)
   - Retard capteur (50 µs)

Ce script génère :
   - Comparaison piezo idéal vs piezo réaliste
   - Tension de commande vs tension réelle
   - Saturations détectées
   - FFT (frequency vs amplitude µm) avec/sans piezo réaliste
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from plate_model import PlateModel
from milling_force import precompute_alpha_periodic
from lqg_controller import LQGController
from newmark_solver import NewmarkSimulator
from piezo_actuator import PiezoActuator


# =====================================================================
#  PARAMETRES
# =====================================================================
LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU_AL = 2830, 69e9, 0.33

NT = 3;  DT_TOOL = 0.010
ETA_H = np.deg2rad(35);  GAMMA_N = np.deg2rad(15)
KT = 925e6;  KN = 0.26;  MU_C = 0.20

RPM = 4900
FT  = 0.02e-3;  AE = 0.1e-3;  AP = 0.3e-3

D31  = 175e-12;  H_PA = 0.7e-3
E_PE = 63e9;     NU_PE = 0.35
PATCH_X = (0.0, 0.020);  PATCH_Z = (0.0, 0.060)

N1, N2 = 30, 24
N_MODES = 3
ZETA = [0.0031, 0.0017, 0.0027]

DT = 5e-5
T_STOP_NO_CTRL = 0.20
T_END_DEMO = 0.5      # 0.5 s = 2.45 mm parcourus

# Spécifications PIEZO RÉALISTES (datasheet QDA60-200.7)
PIEZO_SPECS = dict(
    V_max            = 150.0,        # Tension maximale (bipolaire)
    V_min            = -150.0,
    slew_rate        = 300e3,        # 300 V/ms = 0.3 V/µs
    omega_amp        = 2*np.pi*5e3,  # BW amplificateur 5 kHz
    hysteresis_pct   = 0.10,         # 10% hystérésis
    sensor_noise_rms = 1e-7,         # 0.1 µm RMS
    sensor_delay     = 50e-6,        # 50 µs retard
    dt               = DT,
)

OUT_DIR = "figs_realistic"
os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================================
#  1) Plaque + LQG
# =====================================================================
print("="*70)
print(" SIMULATION AVEC MODELE PIEZO REALISTE")
print("="*70)

t_start = time.time()
plate = PlateModel(lp=LP, hp=HP, bp=BP,
                   rho=RHO, E=E_AL, nu=NU_AL,
                   N1=N1, N2=N2,
                   n_modes=N_MODES, zeta_modes=ZETA)

v_feed = FT * NT * RPM / 60
zp_pos = HP - AP/2

plate.precompute_Dp(zp_pos=zp_pos, n_pos=2001)
plate.set_observation(x_obs=LP, z_obs=HP)
plate.add_piezo_patch(xP1=PATCH_X[0], xP2=PATCH_X[1],
                      zP1=PATCH_Z[0], zP2=PATCH_Z[1],
                      d31=D31, h_Pa=H_PA, E_Pe=E_PE, nu_Pe=NU_PE)

print(f"\n=== Conception LQG ===")
ctrl = LQGController(plate, dt=DT)
ctrl.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                      w_qd_list=[1e4, 1e6, 1e8],
                      w_r=1.0)
ctrl.discretize_observer()


# =====================================================================
#  2) Simulation pour 0.5s
# =====================================================================
phi_st = np.pi - np.arccos(1 - AE/(DT_TOOL/2))
phi_ex = np.pi
za_low  = HP - AP
za_high = HP
Omega = 2*np.pi*RPM/60
RT = DT_TOOL/2
tau = 60/(NT*RPM)
k1 = KN*np.cos(ETA_H)
k2 = 1 + MU_C*np.tan(ETA_H)*np.cos(GAMMA_N) - KN*np.sin(ETA_H)

sim = NewmarkSimulator(plate, dt=DT, T_end=T_END_DEMO, ft=FT, tau=tau)

xp_path = np.minimum(v_feed * sim.t_vec, LP)
kp_idx  = np.clip(np.round(xp_path/LP * (len(plate.xp_array) - 1)).astype(int),
                  0, len(plate.xp_array) - 1)

print(f"\n=== Pré-calcul alpha3, alpha4 ===")
n_per = int(np.round(tau/DT))
alpha3_t, alpha4_t = precompute_alpha_periodic(
    DT, n_per, sim.nstep,
    Omega, NT, RT, ETA_H, phi_st, phi_ex, za_low, za_high, k1, k2, KT
)


# =====================================================================
#  3) Trois simulations comparatives
# =====================================================================
rng = np.random.default_rng(42)

# Sim 1 : Sans contrôle
print(f"\n--- Sim 1 : Sans contrôle (arret a {T_STOP_NO_CTRL}s) ---")
res_open = sim.simulate(alpha3_t, alpha4_t, kp_idx,
                         controller=None,
                         stop_at_time=T_STOP_NO_CTRL,
                         stop_threshold=5e-3,
                         progress=False)
print(f"   Max |y_p| = {np.max(np.abs(res_open['y']))*1e6:.1f} µm")

# Sim 2 : LQG idéal (piezo parfait)
print(f"\n--- Sim 2 : LQG avec piezo IDEAL (référence) ---")
res_ideal = sim.simulate(alpha3_t, alpha4_t, kp_idx,
                          controller=ctrl,
                          piezo=None,
                          stop_threshold=5e-3,
                          progress=False)
print(f"   Max |y_p|     = {np.max(np.abs(res_ideal['y']))*1e6:.2f} µm")
print(f"   Max |u_cmd|   = {np.max(np.abs(res_ideal['u'])):.1f} V")

# Sim 3 : LQG avec piezo réaliste
print(f"\n--- Sim 3 : LQG avec piezo REALISTE ---")
piezo_real = PiezoActuator(**PIEZO_SPECS)
rng2 = np.random.default_rng(42)
res_real = sim.simulate(alpha3_t, alpha4_t, kp_idx,
                         controller=ctrl,
                         piezo=piezo_real,
                         rng=rng2,
                         stop_threshold=5e-3,
                         progress=False)
print(f"   Max |y_p|     = {np.max(np.abs(res_real['y']))*1e6:.2f} µm")
print(f"   Max |u_cmd|   = {np.max(np.abs(res_real['u_cmd'])):.1f} V (commande LQR)")
print(f"   Max |u_real|  = {np.max(np.abs(res_real['u'])):.1f} V (tension réelle au piezo)")
ps = res_real['piezo_stats']
print(f"   Saturations   = {ps['n_saturations']} / {sim.nstep}")
print(f"   Slew limites  = {ps['n_slew_limits']} / {sim.nstep}")

reduction_ideal = (1 - np.max(np.abs(res_ideal['y']))/np.max(np.abs(res_open['y']))) * 100
reduction_real  = (1 - np.max(np.abs(res_real['y'])) /np.max(np.abs(res_open['y']))) * 100
print(f"\n=== COMPARAISON ===")
print(f"   Réduction piezo IDEAL    : {reduction_ideal:.2f} %")
print(f"   Réduction piezo REALISTE : {reduction_real:.2f} %")
print(f"   Dégradation due au piezo : {reduction_ideal-reduction_real:.2f} %")
print(f"   Temps total : {time.time()-t_start:.1f} s")


# =====================================================================
#  4) TRACES
# =====================================================================
i1 = res_open['stop_idx']
i2 = res_ideal['stop_idx']
i3 = res_real['stop_idx']
t_ms = sim.t_vec * 1e3

# --- Fig 1 : Comparaison temporelle des 3 cas ---
fig, axes = plt.subplots(3, 1, figsize=(13, 9))

ax = axes[0]
ax.plot(t_ms[:i1+1], res_open['y'][:i1+1]*1e6, 'r', linewidth=1.0)
ax.set_ylabel("$y_p$ (µm)")
ax.set_title(f"Sans contrôle  (max = {np.max(np.abs(res_open['y']))*1e6:.0f} µm)")
ax.grid(True, alpha=0.5)

ax = axes[1]
ax.plot(t_ms[:i2+1], res_ideal['y'][:i2+1]*1e6, 'g', linewidth=1.0,
         label='LQG (piezo idéal)')
ax.set_ylabel("$y_p$ (µm)")
ax.set_title(f"LQG avec piezo IDEAL  (max = {np.max(np.abs(res_ideal['y']))*1e6:.2f} µm, "
             f"réduction {reduction_ideal:.1f}%)")
ax.grid(True, alpha=0.5);  ax.legend()

ax = axes[2]
ax.plot(t_ms[:i3+1], res_real['y'][:i3+1]*1e6, 'b', linewidth=1.0,
         label='LQG (piezo réaliste)')
ax.set_xlabel("Temps (ms)");  ax.set_ylabel("$y_p$ (µm)")
ax.set_title(f"LQG avec piezo REALISTE  (max = {np.max(np.abs(res_real['y']))*1e6:.2f} µm, "
             f"réduction {reduction_real:.1f}%)")
ax.grid(True, alpha=0.5);  ax.legend()

plt.suptitle("Effet du modèle piezo réaliste sur la performance LQG",
              fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_realistic_temps.png", dpi=120)
plt.close()


# --- Fig 2 : Tension commande vs tension réelle ---
fig, axes = plt.subplots(3, 1, figsize=(13, 9))

ax = axes[0]
ax.plot(t_ms[:i3+1], res_real['u_cmd'][:i3+1], 'g', linewidth=0.7,
         alpha=0.8, label='u commande LQR')
ax.plot(t_ms[:i3+1], res_real['u'][:i3+1], 'b', linewidth=0.7,
         alpha=0.8, label='u réel (après piezo)')
ax.axhline(PIEZO_SPECS['V_max'],  color='r', linestyle='--', alpha=0.5,
            label=f"V_max = ±{PIEZO_SPECS['V_max']:.0f}V")
ax.axhline(-PIEZO_SPECS['V_max'], color='r', linestyle='--', alpha=0.5)
ax.set_ylabel("Tension (V)")
ax.set_title("Tension de commande LQR vs tension réelle après piezo")
ax.legend(loc='upper right');  ax.grid(True, alpha=0.5)

ax = axes[1]
diff = res_real['u_cmd'][:i3+1] - res_real['u'][:i3+1]
ax.plot(t_ms[:i3+1], diff, 'm', linewidth=0.6)
ax.set_ylabel("Erreur (V)")
ax.set_title(f"Erreur commande - réel  (RMS = {np.sqrt(np.mean(diff**2)):.2f} V)")
ax.grid(True, alpha=0.5)

ax = axes[2]
ax.plot(t_ms[:i3+1], np.abs(res_real['u'][:i3+1]), 'b', linewidth=0.6,
         label='|u_real|')
ax.fill_between(t_ms[:i3+1], 0, np.abs(res_real['u'][:i3+1]),
                 color='b', alpha=0.2)
ax.axhline(PIEZO_SPECS['V_max'], color='r', linestyle='--', alpha=0.5)
ax.set_xlabel("Temps (ms)"); ax.set_ylabel("|u_real| (V)")
ax.set_title(f"Magnitude tension réelle  (saturations : {ps['n_saturations']}, "
             f"slew limits : {ps['n_slew_limits']})")
ax.legend();  ax.grid(True, alpha=0.5)

plt.suptitle("Comportement du piezo réaliste",
              fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_realistic_voltage.png", dpi=120)
plt.close()


# --- Fig 3 : FFT comparaison (Frequency vs Amplitude µm) ---
def fft_amp(y, dt, i_start_t=0.05):
    i_start = int(i_start_t/dt)
    if len(y) <= i_start:
        return None, None
    y_seg = y[i_start:]
    N = len(y_seg);  win = np.hanning(N)
    NFFT = 2**int(np.ceil(np.log2(max(N, 2))))
    Y = np.fft.fft(y_seg*win, NFFT)/N
    f = (1/dt)/2 * np.linspace(0, 1, NFFT//2 + 1)
    return f, 2*np.abs(Y[:NFFT//2 + 1]) * 1e6

f1, Y1 = fft_amp(res_open['y'][:i1+1], DT, 0.05)
f2, Y2 = fft_amp(res_ideal['y'][:i2+1], DT, 0.05)
f3, Y3 = fft_amp(res_real['y'][:i3+1], DT, 0.05)

f_t = NT * RPM / 60

fig, axes = plt.subplots(2, 1, figsize=(13, 8))

# Subplot supérieur : linéaire
ax = axes[0]
ax.plot(f1, Y1, color=(0.85, 0.55, 0.25), linewidth=1.4,
         label=f"Sans contrôle (max={Y1.max():.1f} µm)")
ax.fill_between(f1, 0, Y1, color=(0.85, 0.55, 0.25), alpha=0.25)
ax.plot(f2, Y2, 'g', linewidth=1.2,
         label=f"LQG piezo idéal (max={Y2.max():.3f} µm)")
ax.plot(f3, Y3, 'b', linewidth=1.2,
         label=f"LQG piezo réaliste (max={Y3.max():.3f} µm)")
ax.set_ylabel("Amplitude (µm)")
ax.set_title("FFT - Frequence (Hz) vs Amplitude (µm)  -  Comparaison piezo idéal/réaliste")
ax.set_xlim([0, 1200])
ax.grid(True, alpha=0.5)
ax.legend(loc='upper right')

yl = ax.get_ylim()
for n_h, lab in [(1, '$f_t$'), (2, '$2f_t$'), (3, '$3f_t$')]:
    f_h = n_h * f_t
    if f_h > 1200: break
    ax.axvline(f_h, color='gray', linestyle=':', alpha=0.7)
    ax.text(f_h+5, yl[1]*0.95, f"{lab}={f_h:.0f}Hz", fontsize=9)
for k in range(N_MODES):
    if plate.freq_n[k] > 1200: continue
    ax.axvline(plate.freq_n[k], color='red', linestyle='--', alpha=0.5)
    ax.text(plate.freq_n[k]+5, yl[1]*(0.85-0.07*k),
             f"$f_{{c{k+1}}}$={plate.freq_n[k]:.0f}Hz",
             color='red', fontsize=9)

# Subplot inférieur : log
ax = axes[1]
ax.semilogy(f1, np.maximum(Y1, 1e-4), color=(0.85, 0.55, 0.25), linewidth=1.4,
             label=f"Sans contrôle")
ax.semilogy(f2, np.maximum(Y2, 1e-4), 'g', linewidth=1.2,
             label=f"LQG piezo idéal")
ax.semilogy(f3, np.maximum(Y3, 1e-4), 'b', linewidth=1.2,
             label=f"LQG piezo réaliste")
ax.set_xlabel("Fréquence (Hz)");  ax.set_ylabel("Amplitude (µm, log)")
ax.set_title(f"FFT échelle log  (réduction au pic dominant : "
             f"idéal {Y1.max()/max(Y2.max(),1e-6):.0f}× , "
             f"réaliste {Y1.max()/max(Y3.max(),1e-6):.0f}×)")
ax.set_xlim([0, 1200])
ax.set_ylim([1e-3, max(Y1.max(), Y2.max(), Y3.max())*2])
ax.grid(True, which='both', alpha=0.5)
ax.legend(loc='upper right')
yl = ax.get_ylim()
for n_h, lab in [(1, '$f_t$'), (2, '$2f_t$'), (3, '$3f_t$')]:
    f_h = n_h * f_t
    if f_h > 1200: break
    ax.axvline(f_h, color='gray', linestyle=':', alpha=0.7)
for k in range(N_MODES):
    if plate.freq_n[k] > 1200: continue
    ax.axvline(plate.freq_n[k], color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_realistic_fft.png", dpi=120)
plt.close()


# --- Fig 4 : Mesure capteur (bruitée + retardée) ---
fig, axes = plt.subplots(2, 1, figsize=(13, 6))

# Vue zoom sur 50 ms pour voir le bruit
n_zoom = min(int(0.05/DT), i3)
ax = axes[0]
ax.plot(t_ms[:n_zoom], res_real['y'][:n_zoom]*1e6, 'b', linewidth=1.5,
         label='$y$ vraie déflection', alpha=0.9)
ax.plot(t_ms[:n_zoom], res_real['y_meas'][:n_zoom]*1e6, 'r',
         linewidth=0.6, alpha=0.6, label='$y_{meas}$ mesurée (bruit + retard)')
ax.set_ylabel("Déflection (µm)")
ax.set_title(f"Mesure capteur réaliste : bruit {PIEZO_SPECS['sensor_noise_rms']*1e6:.2f} µm RMS, "
             f"retard {PIEZO_SPECS['sensor_delay']*1e6:.0f} µs")
ax.legend(loc='upper right');  ax.grid(True, alpha=0.5)
ax.set_xlim([0, 50])

# Erreur de mesure
ax = axes[1]
err = (res_real['y_meas'][:i3+1] - res_real['y'][:i3+1]) * 1e6
ax.plot(t_ms[:i3+1], err, 'm', linewidth=0.5)
ax.set_xlabel("Temps (ms)");  ax.set_ylabel("Erreur (µm)")
ax.set_title(f"Erreur de mesure  (RMS = {np.sqrt(np.mean(err**2)):.3f} µm)")
ax.grid(True, alpha=0.5)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_realistic_sensor.png", dpi=120)
plt.close()


# --- Fig 5 : Bilan résumé ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# (a) Max |y| par cas
ax = axes[0]
labels = ['Sans\ncontrôle', 'LQG\npiezo idéal', 'LQG\npiezo réaliste']
values = [np.max(np.abs(res_open['y']))*1e6,
          np.max(np.abs(res_ideal['y']))*1e6,
          np.max(np.abs(res_real['y']))*1e6]
colors = ['red', 'green', 'blue']
bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='k')
ax.set_yscale('log')
ax.set_ylabel("Max |y_p| (µm)")
ax.set_title("Performance vibratoire")
ax.grid(True, axis='y', alpha=0.5)
for bar, v in zip(bars, values):
    ax.text(bar.get_x()+bar.get_width()/2, v*1.1, f'{v:.1f} µm',
             ha='center', fontweight='bold')

# (b) Tensions
ax = axes[1]
labels = ['u_cmd LQR\n(idéal)', 'u_cmd\n(réel)', 'u_actuel\n(piezo)']
values = [np.max(np.abs(res_ideal['u'])),
          np.max(np.abs(res_real['u_cmd'])),
          np.max(np.abs(res_real['u']))]
ax.bar(labels, values, color=['green', 'orange', 'blue'], alpha=0.7, edgecolor='k')
ax.axhline(PIEZO_SPECS['V_max'], color='red', linestyle='--',
            label=f"V_max ±{PIEZO_SPECS['V_max']:.0f}V")
ax.set_ylabel("Tension max (V)")
ax.set_title("Tensions piézoélectriques")
ax.legend(loc='upper left')
ax.grid(True, axis='y', alpha=0.5)
for i, v in enumerate(values):
    ax.text(i, v + 5, f'{v:.1f} V', ha='center', fontweight='bold')

# (c) Réduction
ax = axes[2]
labels = ['Idéal', 'Réaliste']
values = [reduction_ideal, reduction_real]
ax.bar(labels, values, color=['green', 'blue'], alpha=0.7, edgecolor='k')
ax.set_ylabel("Réduction d'amplitude (%)")
ax.set_title("Efficacité du contrôle")
ax.set_ylim([0, 105])
ax.grid(True, axis='y', alpha=0.5)
for i, v in enumerate(values):
    ax.text(i, v - 5, f'{v:.2f}%', ha='center', fontweight='bold', color='white')

plt.suptitle("Bilan : impact du modèle piezo réaliste sur le LQG",
              fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_realistic_summary.png", dpi=120)
plt.close()


print(f"\n=== Figures générées dans {OUT_DIR}/ ===")
print(f"   fig_realistic_temps.png    : 3 simulations (sans/idéal/réaliste)")
print(f"   fig_realistic_voltage.png  : tension commande vs réelle + erreur")
print(f"   fig_realistic_fft.png      : FFT (Hz vs µm) comparaison")
print(f"   fig_realistic_sensor.png   : mesure bruitée vs vraie")
print(f"   fig_realistic_summary.png  : bilan barres")
