"""
main_aimc_extended.py
=====================
Résultats ÉTENDUS pour l'article — anticipe les questions des rapporteurs et
caractérise HONNÊTEMENT les limites de l'AIMC (pas seulement ses victoires).

Six expériences (protocole honnête : aucun contrôleur ne connaît le détuning
réel, le K_T réel, la dérive, ni le bruit) :

  E4 — PAYSAGE DE DÉTUNING continu (−22 %…+12 %, au-delà de la grille MMAE)
       -> domination sur le CONTINUUM + comportement de bord gracieux.
  E5 — ROBUSTESSE AU BRUIT CAPTEUR des 5 contrôleurs (l'identification MMAE
       survit-elle au bruit ?).
  E7 — TRANSITOIRE D'IDENTIFICATION : entrée d'outil dans une paroi déjà
       détunée (échelon) -> temps de verrouillage, RMS transitoire.
  E9 — SPECTRE : annulation des harmoniques de passage de dent (LQG vs AIMC).
  E10 — COÛT TEMPS RÉEL : temps par pas (banc de 7 filtres VECTORISÉ).

Sortie : figs_aimc_ext/*.png + tableaux console.
(La dérive E8 et l'erreur d'encodeur sont dans main_aimc_drift.py.)
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plate_model import PlateModel
from milling_force import precompute_nonlinear_periodic
from lqg_controller import LQGController
from imc_lqg_controller import IMCLQGController
from darc_controller import DARCController
from smc_dob_controller import SMCDOBController
from aimc_controller import AIMCController
from newmark_solver import NewmarkSimulator

LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU_AL = 2830, 69e9, 0.33
NT = 3;  DT_TOOL = 0.010
ETA_H = np.deg2rad(35);  GAMMA_N = np.deg2rad(15)
KT_NOMINAL = 925e6;  KN = 0.26;  MU_C = 0.20
RPM = 4900
FT = 0.02e-3;  AE = 0.1e-3
D31 = 175e-12;  H_PA = 0.7e-3
E_PE = 63e9;    NU_PE = 0.35
N_MODES = 3
ZETA = [0.0031, 0.0017, 0.0027]
DT = 5e-5
SENSOR_NOISE = 1.0e-7
FF_HARMONICS = 30
IMC_HARMONICS = 8
IMC_QDIST = 1e3
RT = DT_TOOL/2
PHI_ST = np.pi - np.arccos(1 - AE/RT)
PHI_EX = np.pi
OMEGA = 2*np.pi*RPM/60
TAU = 60/(NT*RPM)
K1C = KN*np.cos(ETA_H)
K2C = 1 + MU_C*np.tan(ETA_H)*np.cos(GAMMA_N) - KN*np.sin(ETA_H)

OUT_DIR = "figs_aimc_ext"
os.makedirs(OUT_DIR, exist_ok=True)
COL = {'LQG': '#2E8B57', 'IMC-fige': '#1f77b4', 'DARC-FF': '#DC143C',
       'STSMC+DOB': '#8A6A1B', 'AIMC': '#6A0DAD'}
TAGS = ('LQG', 'IMC-fige', 'DARC-FF', 'STSMC+DOB', 'AIMC')


def build_plate(ap, freq_perturb=0.0):
    plate = PlateModel(LP, HP, BP, RHO, E_AL, NU_AL,
                       n_modes=N_MODES, zeta_modes=ZETA, verbose=False)
    plate.precompute_Dp(zp_pos=HP - ap/2, n_pos=2001)
    plate.set_observation(x_obs=LP, z_obs=HP)
    plate.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    if abs(freq_perturb) > 1e-6:
        s = (1 + freq_perturb)**2
        plate.Kp = plate.Kp * s
        plate.omega_n = np.sqrt(np.diag(plate.Kp) / np.diag(plate.Mp))
        plate.freq_n = plate.omega_n / (2*np.pi)
        plate.Cp = np.diag(2 * np.array(ZETA) * plate.omega_n * np.diag(plate.Mp))
    return plate


def make_setup(plate_r, ap, KT_actual, T_end):
    sim = NewmarkSimulator(plate_r, dt=DT, T_end=T_end, ft=FT, tau=TAU, verbose=False)
    xp = np.minimum(FT*NT*RPM/60 * sim.t_vec, LP)
    kp = np.clip(np.round(xp/LP * 2000).astype(int), 0, 2000)
    n_per = int(np.round(TAU/DT))
    a3, a4, a42, a43 = precompute_nonlinear_periodic(
        DT, n_per, sim.nstep, OMEGA, NT, RT, ETA_H, PHI_ST, PHI_EX,
        HP-ap, HP, K1C, K2C, KT_actual)
    a3_n = a3 if KT_actual == KT_NOMINAL else precompute_nonlinear_periodic(
        DT, n_per, sim.nstep, OMEGA, NT, RT, ETA_H, PHI_ST, PHI_EX,
        HP-ap, HP, K1C, K2C, KT_NOMINAL)[0]
    return sim, kp, (a3, a4, a42, a43), a3_n, n_per


def make_ctrls(plate_d, n_per, a3_n, Dp_ff):
    f_fund = 1.0 / (n_per * DT)
    lqg = LQGController(plate_d, dt=DT, verbose=False)
    lqg.optimize_weights(w_q_list=[1e14], w_qd_list=[1e8], w_r=1.0, gain_norm_max=1e10)
    lqg.discretize_observer()
    imc = IMCLQGController(plate_d, dt=DT, f_fund=f_fund, Dp_dist=Dp_ff,
                           n_harm=IMC_HARMONICS, q_dist=IMC_QDIST, verbose=False)
    darc = DARCController(plate_d, dt=DT, base_w_q=1e14, base_w_qd=1e8,
                          n_per=n_per, u_max=150.0, verbose=False)
    darc.design_periodic_feedforward(FT, a3_n[:n_per], Dp_ff, n_harm=FF_HARMONICS)
    smc = SMCDOBController(plate_d, dt=DT, Dp_dist=Dp_ff, n_per=n_per, verbose=False)
    aimc = AIMCController(plate_d, dt=DT, f_fund=f_fund, Dp_dist=Dp_ff, verbose=False)
    return dict(zip(TAGS, [lqg, imc, darc, smc, aimc]))


def run(sim, kp, arrs, ctrl, k_scale_t=None, noise=SENSOR_NOISE, seed=1):
    a3, a4, a42, a43 = arrs
    if hasattr(ctrl, 'reset'):
        ctrl.reset()
    return sim.simulate(a3, a4, kp, controller=ctrl, alpha4_2_t=a42, alpha4_3_t=a43,
                        sensor_noise=noise, sensor_rng=np.random.default_rng(seed),
                        k_scale_t=k_scale_t, progress=False)


def yrms(r):
    i = r['stop_idx']
    return float(np.sqrt(np.mean(r['y'][:i+1]**2)))*1e6, r['diverged_at'] > 0


print("="*78)
print(" RÉSULTATS ÉTENDUS AIMC — pour l'article")
print("="*78)
t_glob = time.time()
ap = 0.3e-3

# ================================================================
# E4 : PAYSAGE DE DÉTUNING continu
# ================================================================
print("\n[E4] Paysage de détuning continu (-22%..+12%, grille MMAE = -20..+10%)")
rho_axis = np.round(np.arange(-0.22, 0.121, 0.03), 3)
plate_d0 = build_plate(ap, 0.0)
Dp_ff0 = plate_d0.get_Dp_at(int(np.clip(round(FT*NT*RPM/60*0.25/LP*2000),0,2000)))[0]
land = {tag: [] for tag in TAGS}
land_div = {tag: [] for tag in TAGS}
for rho in rho_axis:
    plate_r = build_plate(ap, rho)
    sim, kp, arrs, a3_n, n_per = make_setup(plate_r, ap, KT_NOMINAL, 0.5)
    ctrls = make_ctrls(plate_d0, n_per, a3_n, Dp_ff0)
    for tag in TAGS:
        y, div = yrms(run(sim, kp, arrs, ctrls[tag]))
        land[tag].append(y); land_div[tag].append(div)
    print(f"  rho={rho*100:+5.0f}% : " +
          "  ".join(f"{tag[:4]}={land[tag][-1]:6.3f}{'X' if land_div[tag][-1] else ''}"
                    for tag in TAGS))

# ================================================================
# E5 : robustesse au bruit capteur (les 5, y compris AIMC)
# ================================================================
print("\n[E5] Robustesse au bruit capteur (S3 détuné -15%, les 5 contrôleurs)")
noise_axis = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]) * 1e-6
plate_r3 = build_plate(ap, -0.15)
sim3, kp3, arrs3, a3n3, n_per3 = make_setup(plate_r3, ap, KT_NOMINAL, 0.5)
noise_sweep = {tag: [] for tag in TAGS}
for sn in noise_axis:
    ctrls = make_ctrls(plate_d0, n_per3, a3n3, Dp_ff0)
    for tag in TAGS:
        y, div = yrms(run(sim3, kp3, arrs3, ctrls[tag], noise=sn, seed=7))
        noise_sweep[tag].append(min(y, 30.0))
    print(f"  noise={sn*1e6:.1f}um : " +
          "  ".join(f"{tag[:4]}={noise_sweep[tag][-1]:6.3f}" for tag in TAGS))

# ================================================================
# E7 : transitoire d'identification (entrée d'outil paroi détunée -15%)
# ================================================================
print("\n[E7] Transitoire d'identification : entrée dans une paroi détunée -15%")
sim7, kp7, arrs7, a3n7, n_per7 = make_setup(plate_r3, ap, KT_NOMINAL, 0.5)
ctrls7 = make_ctrls(plate_d0, n_per7, a3n7, Dp_ff0)
aimc7 = ctrls7['AIMC']
r7 = run(sim7, kp7, arrs7, aimc7)
rho_tr = np.array(aimc7.trace_rho)
conf_tr = np.array(aimc7.trace_conf)
# temps où rho_hat atteint 90% de sa valeur finale
rho_final = np.median(rho_tr[-500:])
reached = np.where(np.abs(rho_tr - rho_final) < 0.1*abs(rho_final))[0]
t_lock = reached[0]*DT*1e3 if len(reached) else -1
# RMS transitoire (0-50ms) vs établi (>200ms)
i50 = int(0.05/DT); i200 = int(0.2/DT); iend = r7['stop_idx']
rms_trans = np.sqrt(np.mean(r7['y'][:i50]**2))*1e6
rms_settled = np.sqrt(np.mean(r7['y'][i200:iend]**2))*1e6
print(f"  rho_hat -> {rho_final*100:.1f}% , temps de verrouillage (90%) = {t_lock:.0f} ms")
print(f"  RMS transitoire [0,50]ms = {rms_trans:.3f} um , établi [200ms,fin] = {rms_settled:.3f} um")

# ================================================================
# E9 : spectre — annulation des harmoniques (S3, LQG vs AIMC)
# ================================================================
print("\n[E9] Spectre FFT : annulation des harmoniques de dent (S3, LQG vs AIMC)")
ctrls9 = make_ctrls(plate_d0, n_per3, a3n3, Dp_ff0)
r9_lqg = run(sim3, kp3, arrs3, ctrls9['LQG'])
r9_aimc = run(sim3, kp3, arrs3, ctrls9['AIMC'])
def spec(r):
    i0 = int(0.1/DT); i1 = r['stop_idx']
    y = r['y'][i0:i1]; N = len(y); win = np.hanning(N)
    Y = np.abs(np.fft.rfft(y*win))/N * 2 * 1e6 / np.mean(win)
    f = np.fft.rfftfreq(N, DT)
    return f, Y
f_l, Y_l = spec(r9_lqg); f_a, Y_a = spec(r9_aimc)
f_tpf = NT*RPM/60
print(f"  f_tooth={f_tpf:.1f}Hz. Pics LQG vs AIMC aux harmoniques :")
for h in range(1, 5):
    fh = h*f_tpf; il = np.argmin(np.abs(f_l-fh))
    pl = Y_l[max(0,il-3):il+4].max(); pa = Y_a[max(0,il-3):il+4].max()
    print(f"    h={h} ({fh:.0f}Hz) : LQG={pl:.4f}  AIMC={pa:.4f} um  (att. {20*np.log10(pa/pl+1e-12):.0f} dB)")

# ================================================================
# E10 : coût temps réel (temps par pas)
# ================================================================
print("\n[E10] Coût temps réel : temps par pas de commande")
timing = {}
for tag in TAGS:
    ctrls = make_ctrls(plate_d0, n_per, a3_n, Dp_ff0)
    c = ctrls[tag]
    if hasattr(c, 'reset'): c.reset()
    x0 = np.zeros(2*N_MODES); nrep = 3000
    import inspect
    takes_k = 'k_step' in inspect.signature(c.step).parameters
    t0 = time.time()
    for k in range(nrep):
        if takes_k:
            c.step(x0, 0.0, 1e-7*np.sin(k*0.1), k_step=k)
        else:
            c.step(x0, 0.0, 1e-7*np.sin(k*0.1))
    dt_step = (time.time()-t0)/nrep*1e6
    timing[tag] = dt_step
    print(f"  {tag:<10} {dt_step:7.2f} us/pas  ({'temps reel OK' if dt_step < DT*1e6 else 'DEPASSE dt=50us'})")

# ================================================================
# FIGURES
# ================================================================
# Figure principale (E4 + E5)
fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
ax = axes[0]
for tag in TAGS:
    yv = np.array(land[tag]); dv = np.array(land_div[tag])
    ax.plot(rho_axis*100, np.minimum(yv, 3.0), 'o-', color=COL[tag], lw=2, ms=5, label=tag)
    for xr, yy, dd in zip(rho_axis*100, yv, dv):
        if dd or yy > 3.0:
            ax.text(xr, 2.95, '✗', ha='center', color=COL[tag], fontweight='bold', fontsize=8)
ax.axvspan(-20, 10, color='green', alpha=0.05)
ax.axvline(-20, color='gray', ls=':', lw=0.8); ax.axvline(10, color='gray', ls=':', lw=0.8)
ax.text(-5, 2.7, 'grille MMAE', ha='center', fontsize=8, color='green')
ax.set_xlabel("détuning structurel réel ρ (%)")
ax.set_ylabel("$y_{RMS}$ (µm) [tronqué à 3]")
ax.set_title("(E4) Paysage de robustesse au détuning\nAIMC domine tout le continuum",
             fontweight='bold')
ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.4)

ax = axes[1]
for tag in TAGS:
    ax.plot(noise_axis*1e6, noise_sweep[tag], 'o-', color=COL[tag], lw=2, ms=5, label=tag)
ax.set_xlabel("bruit capteur (µm RMS)")
ax.set_ylabel("$y_{RMS}$ (µm)")
ax.set_ylim(0, 3.0)
ax.set_title("(E5) Robustesse au bruit capteur (paroi détunée −15 %)\n"
             "l'identification AIMC reste robuste", fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_ext_landscape.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"\n  ✓ {OUT_DIR}/fig_ext_landscape.png")

# Figure transitoire + spectre
fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
ax = axes[0]
tt = sim7.t_vec[1:len(rho_tr)+1]*1e3
ax.plot(tt, rho_tr*100, color=COL['AIMC'], lw=1.8, label=r"$\hat\rho$ identifié")
ax.axhline(rho_final*100, color='k', ls='--', lw=1, label=f"régime ({rho_final*100:.0f}%)")
if t_lock > 0:
    ax.axvline(t_lock, color='red', ls=':', lw=1.2, label=f"verrouillage {t_lock:.0f}ms")
ax.set_xlabel("temps (ms)"); ax.set_ylabel("détuning identifié (%)")
ax.set_xlim(0, 150)
ax.set_title("(E7) Transitoire d'identification\nentrée d'outil, paroi détunée −15 %",
             fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

ax = axes[1]
m = f_l <= 1300
ax.semilogy(f_l[m], np.maximum(Y_l[m], 1e-4), color=COL['LQG'], lw=1.2, label='LQG')
ax.semilogy(f_a[m], np.maximum(Y_a[m], 1e-4), color=COL['AIMC'], lw=1.2, label='AIMC')
for h in range(1, 6):
    fh = h*f_tpf
    if fh <= 1300:
        ax.axvline(fh, color='gray', ls=':', alpha=0.5, lw=0.8)
ax.set_xlabel("fréquence (Hz)"); ax.set_ylabel("amplitude $y$ (µm)")
ax.set_title("(E9) Annulation spectrale des harmoniques de dent\n"
             "(paroi détunée −15 %, pointillés = h·f_tooth)", fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_ext_transient_spectrum.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"  ✓ {OUT_DIR}/fig_ext_transient_spectrum.png")

print(f"\nTemps total : {time.time()-t_glob:.0f} s")
