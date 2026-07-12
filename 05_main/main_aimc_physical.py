"""
main_aimc_physical.py
====================
DÉRIVE PHYSIQUE PAR ENLÈVEMENT DE MATIÈRE — le modèle de paroi est RÉ-DÉRIVÉ
de la géométrie réellement usinée (machined_plate.py, Rayleigh-Ritz à épaisseur
variable), au lieu du proxy scalaire k_scale.  L'usinage local décale les modes
de manière NON UNIFORME (mode dominant vers le haut OU le bas selon la zone),
ce qui est le test le plus honnête de l'AIMC.

RÉSULTAT-CLÉ (nuancé, honnête) : l'avantage de l'AIMC dépend de la GÉOMÉTRIE
d'usinage :
  - finition de BORD (petite zone près du bord libre) : mode dominant monte,
    l'IMC figé reste robuste, l'AIMC ≈ IMC (il ne nuit pas) ;
  - poche PROFONDE / usinage vers l'encastrement : le mode dominant DESCEND,
    l'IMC figé DIVERGE, l'AIMC ré-identifie et reste sous-µm (décisif) ;
  - au-delà de la grille MMAE (−20 %), l'AIMC se dégrade aussi (limite).
Le cas idéalisé « détuning uniforme −15 % » (S3) SURESTIME donc la fragilité de
l'IMC ; le modèle physique donne l'image réelle, scénario-dépendante.

Trois scénarios d'usinage (protocole honnête : aucun contrôleur n'est informé
de la dérive) sur T = 2 s, l'outil enlève la matière sur [0.4, 1.6] s :
  P1 finition de bord   : zone [top..40 % H_P], −0.4 mm  (mode 1 ↑)
  P2 poche profonde     : zone [top..10 % H_P], −0.8 mm  (mode 1 ↓ -> IMC diverge)
  P3 usinage agressif   : bande jusqu'à l'encastrement, −0.8 mm (seul AIMC tient)

Sortie : figs_aimc_phys/fig_aimc_physical.png + tableaux console.
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plate_model import PlateModel
from machined_plate import MachinedPlate
from milling_force import precompute_nonlinear_periodic
from lqg_controller import LQGController
from imc_lqg_controller import IMCLQGController
from darc_controller import DARCController
from aimc_controller import AIMCController
from newmark_solver import NewmarkSimulator
from material_removal import machining_modal_schedule

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
DT = 5e-5;  T_END = 2.0
SENSOR_NOISE = 1.0e-7
FF_HARMONICS = 30
IMC_HARMONICS = 8;  IMC_QDIST = 1e3
RT = DT_TOOL/2
PHI_ST = np.pi - np.arccos(1 - AE/RT);  PHI_EX = np.pi
OMEGA = 2*np.pi*RPM/60;  TAU = 60/(NT*RPM)
K1C = KN*np.cos(ETA_H)
K2C = 1 + MU_C*np.tan(ETA_H)*np.cos(GAMMA_N) - KN*np.sin(ETA_H)
T_MACH0, T_MACH1 = 0.4, 1.6            # fenêtre d'enlèvement de matière

OUT_DIR = "figs_aimc_phys"
os.makedirs(OUT_DIR, exist_ok=True)
COL = {'LQG': '#2E8B57', 'IMC-fige': '#1f77b4', 'DARC-FF': '#DC143C',
       'AIMC': '#6A0DAD'}
TAGS = ('LQG', 'IMC-fige', 'DARC-FF', 'AIMC')

plate = PlateModel(LP, HP, BP, RHO, E_AL, NU_AL, n_modes=N_MODES,
                   zeta_modes=ZETA, verbose=False)
ap = 0.3e-3
plate.precompute_Dp(zp_pos=HP - ap/2, n_pos=2001)
plate.set_observation(x_obs=LP, z_obs=HP)
plate.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
mplate = MachinedPlate(LP, HP, BP, RHO, E_AL, NU_AL, verbose=True)

sim = NewmarkSimulator(plate, dt=DT, T_end=T_END, ft=FT, tau=TAU, verbose=False)
xp = np.minimum(FT*NT*RPM/60 * sim.t_vec, LP)
kp = np.clip(np.round(xp/LP * 2000).astype(int), 0, 2000)
n_per = int(round(TAU/DT))
a3, a4, a42, a43 = precompute_nonlinear_periodic(
    DT, n_per, sim.nstep, OMEGA, NT, RT, ETA_H, PHI_ST, PHI_EX,
    HP-ap, HP, K1C, K2C, KT_NOMINAL)
Dp_ff = plate.get_Dp_at(int(kp[sim.nstep // 2]))[0]
f_fund = 1.0 / (n_per * DT)
t = sim.t_vec


def make(kind):
    if kind == 'LQG':
        c = LQGController(plate, dt=DT, verbose=False)
        c.optimize_weights(w_q_list=[1e14], w_qd_list=[1e8], w_r=1.0,
                           gain_norm_max=1e10);  c.discretize_observer();  return c
    if kind == 'IMC-fige':
        return IMCLQGController(plate, dt=DT, f_fund=f_fund, Dp_dist=Dp_ff,
                                n_harm=IMC_HARMONICS, q_dist=IMC_QDIST, verbose=False)
    if kind == 'DARC-FF':
        d = DARCController(plate, dt=DT, base_w_q=1e14, base_w_qd=1e8,
                           n_per=n_per, u_max=150.0, verbose=False)
        d.design_periodic_feedforward(FT, a3[:n_per], Dp_ff, n_harm=FF_HARMONICS)
        return d
    return AIMCController(plate, dt=DT, f_fund=f_fund, Dp_dist=Dp_ff, verbose=False)


def run(c, sched):
    if hasattr(c, 'reset'):
        c.reset()
    return sim.simulate(a3, a4, kp, controller=c, alpha4_2_t=a42, alpha4_3_t=a43,
                        sensor_noise=SENSOR_NOISE, sensor_rng=np.random.default_rng(1),
                        modal_schedule=sched, progress=False)


def rms_win(r, t0, t1):
    i0, i1 = int(t0/DT), min(int(t1/DT), r['stop_idx'])
    return float(np.sqrt(np.mean(r['y'][i0:i1+1]**2)))*1e6 if i1 > i0 else float('nan')


print("="*80)
print(" DÉRIVE PHYSIQUE PAR ENLÈVEMENT DE MATIÈRE (Ritz épaisseur variable)")
print("="*80)
t_glob = time.time()

scen = [
    ("P1 finition de bord (-0.4mm)",   1.0, 0.4, 0.4e-3),
    ("P2 poche profonde (-0.8mm)",     1.0, 0.1, 0.8e-3),
    ("P3 agressif vers encastr.(-0.8mm)", 0.6, 0.0, 0.8e-3),
]
results = {}
scheds = {}
for tag, zs, ze, rem in scen:
    sched = machining_modal_schedule(t, mplate, BP, rem, z_frac_start=zs,
                                     z_frac_end=ze, t_start=T_MACH0,
                                     t_end=T_MACH1, n_knots=25, verbose=False)
    scheds[tag] = sched
    drift = (sched['omega'][-1]/sched['omega'][0] - 1)*100
    row = {}
    for ct in TAGS:
        r = run(make(ct), sched)
        row[ct] = dict(before=rms_win(r, 0.0, T_MACH0),
                       after=min(rms_win(r, T_MACH1, T_END), 50.0),
                       res=r)
    results[tag] = row
    print(f"\n{tag}  | dérive modes finale = "
          f"{np.array2string(drift, precision=1, floatmode='fixed')} %")
    for ct in TAGS:
        print(f"   {ct:<9} RMS avant usinage {row[ct]['before']:.3f}  "
              f"-> après {row[ct]['after']:.3f} um")

# ================================================================
# FIGURE : (a) dérive par mode, (b) barres avant/après, (c) trace P2
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))

# (a) trajectoires de fréquence par mode (scénario décisif P2)
ax = axes[0]
sd = scheds[scen[1][0]]
tk = sd['t_knots']
om = np.array(sd['omega'])            # (n_knots, 3)
for i, cmk in enumerate(['#1f77b4', '#2E8B57', '#DC143C']):
    ax.plot(tk, om[:, i]/(2*np.pi), 'o-', color=cmk, ms=3, lw=1.6,
            label=f"mode {i+1}")
ax.axvspan(T_MACH0, T_MACH1, color='gray', alpha=0.12)
ax.set_xlabel("temps (s)"); ax.set_ylabel("fréquence propre (Hz)")
ax.set_title("(a) Dérive PAR MODE — usinage local\n(P2 : le mode 1 DESCEND, mode 3 ↓↓)",
             fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

# (b) barres RMS après usinage, 3 scénarios × 4 contrôleurs
ax = axes[1]
names = [s[0].split('(')[0].strip() for s in scen]
x = np.arange(len(scen)); w = 0.2
for j, ct in enumerate(TAGS):
    vals = [min(results[s[0]][ct]['after'], 3.0) for s in scen]
    raw = [results[s[0]][ct]['after'] for s in scen]
    bars = ax.bar(x + (j-1.5)*w, vals, w, color=COL[ct], alpha=0.88,
                  edgecolor='k', linewidth=0.5, label=ct)
    for b, v, rv in zip(bars, vals, raw):
        if rv >= 3.0:
            ax.text(b.get_x()+b.get_width()/2, 3.02,
                    '≥50✗' if rv >= 49 else f'{rv:.0f}',
                    ha='center', fontsize=7, fontweight='bold', rotation=90)
ax.set_xticks(x); ax.set_xticklabels(['P1 bord', 'P2 poche', 'P3 agressif'],
                                     fontsize=9)
ax.set_ylabel("$y_{RMS}$ après usinage (µm) [tronqué 3]")
ax.set_title("(b) Selon la géométrie d'usinage :\nAIMC décisif quand le mode "
             "dominant descend", fontweight='bold')
ax.legend(fontsize=8, ncol=2); ax.grid(True, axis='y', alpha=0.4)

# (c) trace temporelle du scénario décisif P2
ax = axes[2]
p2 = results[scen[1][0]]
ds = max(1, sim.nstep // 3000)
tt = t[::ds]
for ct in ('LQG', 'IMC-fige', 'AIMC'):
    y = p2[ct]['res']['y']
    ax.plot(tt, np.clip(y[::ds]*1e6, -6, 6), color=COL[ct], lw=0.6,
            alpha=0.9, label=ct)
ax.axvspan(T_MACH0, T_MACH1, color='gray', alpha=0.12)
ax.text((T_MACH0+T_MACH1)/2, 5.3, "enlèvement de matière", ha='center',
        fontsize=9, color='gray')
ax.set_xlabel("temps (s)"); ax.set_ylabel("vibration $y$ (µm)")
ax.set_title("(c) P2 : l'IMC figé diverge quand la paroi\ns'amincit ; "
             "l'AIMC ré-identifie et tient", fontweight='bold')
ax.legend(fontsize=9, loc='upper left'); ax.grid(True, alpha=0.4)
ax.set_ylim(-6, 6)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_aimc_physical.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"\n  ✓ {OUT_DIR}/fig_aimc_physical.png")
print(f"\nTemps total : {time.time()-t_glob:.0f} s")
