"""
main_aimc_study.py
==================
AIMC — annulation harmonique ADAPTATIVE PAR IDENTIFICATION : l'étude qui
comble la lacune « l'identification en cours d'usinage n'alimente jamais le
chemin d'annulation active ».

Trois expériences (protocole honnête : aucun contrôleur ne connaît le
détuning réel, le K_T réel, ni la dérive simulée) :

  E1 — CARTE STATIQUE (5 contrôleurs x 5 scénarios, T = 0.5 s)
       LQG / IMC-LQG figé / DARC-FF / STSMC+DOB / AIMC.
       Attendu : AIMC repare S3 (ou IMC fige diverge) en restant pres du
       meilleur partout.
  E2 — DÉTUNING HORS GRILLE (-12 %, entre deux candidats MMAE)
       Vérifie que le mélange bayésien interpole entre les modèles.
  E3 — DÉRIVE EN COURS D'USINAGE (rampe 0 -> -15 % pendant la coupe,
       proxy de l'enlèvement de matière, T = 2 s, k_scale_t du solveur)
       L'expérience-clé de la lacune : l'IMC figé se dégrade à mesure que la
       paroi s'écarte du modèle nominal ; l'AIMC ré-identifie et re-synthétise
       son chemin d'annulation EN LIGNE.

Sortie : tableaux console + figs_aimc/fig_aimc_study.png
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
from material_removal import thinning_schedule

# ============================================================
# Paramètres physiques (identiques à main_simulation.py)
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

OUT_DIR = "figs_aimc"
os.makedirs(OUT_DIR, exist_ok=True)


def build_plate(ap, freq_perturb=0.0):
    plate = PlateModel(LP, HP, BP, RHO, E_AL, NU_AL,
                       n_modes=N_MODES, zeta_modes=ZETA, verbose=False)
    plate.precompute_Dp(zp_pos=HP - ap/2, n_pos=2001)
    plate.set_observation(x_obs=LP, z_obs=HP)
    plate.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    if abs(freq_perturb) > 1e-6:
        scale_K = (1 + freq_perturb)**2
        plate.Kp = plate.Kp * scale_K
        plate.omega_n = np.sqrt(np.diag(plate.Kp) / np.diag(plate.Mp))
        plate.freq_n = plate.omega_n / (2*np.pi)
        plate.Cp = np.diag(2 * np.array(ZETA) * plate.omega_n * np.diag(plate.Mp))
    return plate


def make_setup(plate_r, ap, KT_actual, T_end):
    sim = NewmarkSimulator(plate_r, dt=DT, T_end=T_end, ft=FT, tau=TAU,
                           verbose=False)
    xp = np.minimum(FT*NT*RPM/60 * sim.t_vec, LP)
    kp = np.clip(np.round(xp/LP * 2000).astype(int), 0, 2000)
    n_per = int(np.round(TAU/DT))
    a3, a4, a42, a43 = precompute_nonlinear_periodic(
        DT, n_per, sim.nstep, OMEGA, NT, RT, ETA_H, PHI_ST, PHI_EX,
        HP-ap, HP, K1C, K2C, KT_actual)
    if KT_actual != KT_NOMINAL:
        a3_n, _, _, _ = precompute_nonlinear_periodic(
            DT, n_per, sim.nstep, OMEGA, NT, RT, ETA_H, PHI_ST, PHI_EX,
            HP-ap, HP, K1C, K2C, KT_NOMINAL)
    else:
        a3_n = a3
    return sim, kp, (a3, a4, a42, a43), a3_n, n_per


def controllers_for(plate_d, n_per, a3_n, Dp_ff):
    """Les 5 contrôleurs, tous conçus sur le modèle NOMINAL uniquement."""
    f_fund = 1.0 / (n_per * DT)
    lqg = LQGController(plate_d, dt=DT, verbose=False)
    lqg.optimize_weights(w_q_list=[1e14], w_qd_list=[1e8], w_r=1.0,
                         gain_norm_max=1e10)
    lqg.discretize_observer()
    imc = IMCLQGController(plate_d, dt=DT, f_fund=f_fund, Dp_dist=Dp_ff,
                           n_harm=IMC_HARMONICS, q_dist=IMC_QDIST,
                           verbose=False)
    darc = DARCController(plate_d, dt=DT, base_w_q=1e14, base_w_qd=1e8,
                          n_per=n_per, u_max=150.0, verbose=False)
    darc.design_periodic_feedforward(FT, a3_n[:n_per], Dp_ff,
                                     n_harm=FF_HARMONICS)
    smc = SMCDOBController(plate_d, dt=DT, Dp_dist=Dp_ff, n_per=n_per,
                           verbose=False)
    aimc = AIMCController(plate_d, dt=DT, f_fund=f_fund, Dp_dist=Dp_ff,
                          verbose=False)
    return [('LQG', lqg), ('IMC-fige', imc), ('DARC-FF', darc),
            ('STSMC+DOB', smc), ('AIMC', aimc)]


def run_one(sim, kp, arrs, ctrl, k_scale_t=None):
    a3, a4, a42, a43 = arrs
    if hasattr(ctrl, 'reset'):
        ctrl.reset()
    return sim.simulate(a3, a4, kp, controller=ctrl,
                        alpha4_2_t=a42, alpha4_3_t=a43,
                        sensor_noise=SENSOR_NOISE,
                        sensor_rng=np.random.default_rng(1),
                        k_scale_t=k_scale_t, progress=False)


def stats(r):
    i = r['stop_idx']
    return (float(np.sqrt(np.mean(r['y'][:i+1]**2)))*1e6,
            float(np.max(np.abs(r['u'][:i+1]))),
            r['diverged_at'] > 0)


print("="*78)
print(" AIMC : annulation adaptative par IDENTIFICATION — étude complète")
print("="*78)
t_glob = time.time()

# ================================================================
# E1 + E2 : carte statique (T = 0.5 s), incl. détuning hors grille
# ================================================================
scenarios = [
    ("S1  nominal",          0.3e-3, KT_NOMINAL,     0.00),
    ("S2  ap=0.6mm",         0.6e-3, KT_NOMINAL,     0.00),
    ("S3  omega-15%",        0.3e-3, KT_NOMINAL,    -0.15),
    ("S3b omega-12% (hors grille)", 0.3e-3, KT_NOMINAL, -0.12),
    ("S4  K_T+30% inconnu",  0.3e-3, 1.3*KT_NOMINAL, 0.00),
]
tags = ('LQG', 'IMC-fige', 'DARC-FF', 'STSMC+DOB', 'AIMC')
E1 = {}
print("\n[E1/E2] Carte statique 5 contrôleurs x 5 scénarios (T = 0.5 s)")
for name, ap, KT, fp in scenarios:
    plate_d = build_plate(ap, 0.0)
    plate_r = build_plate(ap, fp)
    sim, kp, arrs, a3_n, n_per = make_setup(plate_r, ap, KT, 0.5)
    Dp_ff = plate_d.get_Dp_at(int(kp[sim.nstep // 2]))[0]
    row = {}
    for tag, ctrl in controllers_for(plate_d, n_per, a3_n, Dp_ff):
        y, u, div = stats(run_one(sim, kp, arrs, ctrl))
        row[tag] = (y, u, div)
    E1[name] = row
    print(f"  {name:<30} " + "  ".join(
        f"{tag}={row[tag][0]:.3f}{'X' if row[tag][2] else ''}" for tag in tags))

# ================================================================
# E3 : dérive en cours d'usinage (rampe 0 -> -15 %, T = 2 s)
# ================================================================
print("\n[E3] Dérive PHYSIQUE (amincissement paroi 4->3.4mm, loi ω∝B) sur t=[0.4,1.6]s (T=2s)")
ap = 0.3e-3
plate_d = build_plate(ap, 0.0)
sim, kp, arrs, a3_n, n_per = make_setup(plate_d, ap, KT_NOMINAL, 2.0)
Dp_ff = plate_d.get_Dp_at(int(kp[sim.nstep // 2]))[0]
t = sim.t_vec
# dérive de fréquence PHYSIQUE : amincissement de la paroi (loi plaque ω ∝ B).
# 0.6 mm retirés d'une paroi de 4 mm = 15 % -> détuning -15 % (cf. material_removal).
k_scale, B_t, rho_true = thinning_schedule(
    t, B_nominal=BP, removed_total=0.15*BP, t_start=0.4, t_end=1.6)

segs = [(0.0, 0.4), (0.4, 1.0), (1.0, 1.6), (1.6, 2.0)]
def rms_seg(r, t0, t1):
    i0, i1 = int(t0/DT), min(int(t1/DT), r['stop_idx'])
    return float(np.sqrt(np.mean(r['y'][i0:i1+1]**2)))*1e6 if i1 > i0 else float('nan')

E3 = {}
aimc_obj = None
for tag, ctrl in controllers_for(plate_d, n_per, a3_n, Dp_ff):
    r = run_one(sim, kp, arrs, ctrl, k_scale_t=k_scale)
    E3[tag] = dict(res=r, segs=[rms_seg(r, a, b) for a, b in segs])
    if tag == 'AIMC':
        aimc_obj = ctrl
    print(f"  {tag:<10} segments RMS: " +
          "  ".join(f"{v:.3f}" for v in E3[tag]['segs']) +
          f"   u_max={np.max(np.abs(r['u'][:r['stop_idx']+1])):.1f}V" +
          ("  DIVERGE" if r['diverged_at'] > 0 else ""))

# ================================================================
# Figure (3 panneaux)
# ================================================================
colors = {'LQG': '#2E8B57', 'IMC-fige': '#1f77b4', 'DARC-FF': '#DC143C',
          'STSMC+DOB': '#8A6A1B', 'AIMC': '#6A0DAD'}
fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

# (a) carte statique
ax = axes[0]
names = [n.split()[0] for n in E1.keys()]
x = np.arange(len(names)); w = 0.16
for j, tag in enumerate(tags):
    vals = [min(E1[n][tag][0], 2.0) for n in E1]
    raw = [E1[n][tag][0] for n in E1]
    bars = ax.bar(x + (j-2)*w, vals, w, color=colors[tag], alpha=0.88,
                  edgecolor='k', linewidth=0.6, label=tag)
    for b, v, rv in zip(bars, vals, raw):
        if rv > 2.0:
            ax.text(b.get_x()+b.get_width()/2, 2.02, f'{rv:.0f}✗',
                    ha='center', fontsize=7, fontweight='bold', rotation=90)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("$y_{RMS}$ (µm) [tronqué à 2]")
ax.set_title("(a) Carte statique — AIMC répare S3\nsans sacrifier le nominal",
             fontweight='bold', fontsize=11)
ax.legend(fontsize=8, ncol=2); ax.grid(True, axis='y', alpha=0.4)

# (b) dérive : RMS par segment
ax = axes[1]
seg_labels = [f"[{a},{b}]s" for a, b in segs]
xs = np.arange(len(segs))
for tag in tags:
    ax.plot(xs, E3[tag]['segs'], 'o-', color=colors[tag], lw=2, ms=6,
            label=tag)
ax.set_xticks(xs); ax.set_xticklabels(seg_labels, fontsize=9)
ax.set_ylabel("$y_{RMS}$ du segment (µm)")
ax.set_ylim(0, 1.0)
ax.set_title("(b) Dérive en usinage 0→−15 % :\nl'IMC figé se dégrade, l'AIMC suit",
             fontweight='bold', fontsize=11)
ax.axvspan(0.5, 2.5, color='gray', alpha=0.12)
ax.text(1.5, 0.92, "rampe de détuning", ha='center', fontsize=9, color='gray')
ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

# (c) identification en ligne
ax = axes[2]
tt = t[1:len(aimc_obj.trace_rho)+1]
ax.plot(tt, np.array(aimc_obj.trace_rho)*100, color=colors['AIMC'], lw=1.8,
        label=r"$\hat\rho$ identifié (AIMC)")
ax.plot(t, rho_true*100, 'k--', lw=1.5, label=r"$\rho$ réel (dérive simulée)")
ax.set_xlabel("temps (s)"); ax.set_ylabel("détuning (%)")
ax.set_title("(c) Identification MMAE en ligne\n(biais ≈ fréquence effective EN COUPE)",
             fontweight='bold', fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_aimc_study.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"\n  ✓ {OUT_DIR}/fig_aimc_study.png")

# ================================================================
# Résumé
# ================================================================
print(f"\n{'='*78}")
print(" RESUME — carte statique (y_RMS, um ; X = divergence)")
print(f"{'='*78}")
print(f"  {'Scenario':<30}" + "".join(f"{tag:>11}" for tag in tags))
for name in E1:
    print(f"  {name:<30}" + "".join(
        f"{E1[name][tag][0]:>10.3f}{'X' if E1[name][tag][2] else ' '}"
        for tag in tags))
print(f"\n  Derive (E3), RMS segment final [1.6,2.0]s :")
for tag in tags:
    print(f"    {tag:<10} {E3[tag]['segs'][-1]:.3f} um")
print(f"\nTemps total : {time.time()-t_glob:.0f} s")
