"""
main_aimc_drift.py
==================
Résultats ÉTENDUS (2) pour l'article — la BANDE PASSANTE de suivi et la
sensibilité à l'erreur d'encodeur.  Caractérisation HONNÊTE des limites.

  E8  — LIMITE DE VITESSE DE DÉRIVE : la fenêtre d'oubli de 10 ms fixe une
        bande passante de suivi.  On balaie la durée de la rampe de détuning
        (0 -> -15 %) : rampe lente -> AIMC suit ; rampe très rapide -> il
        prend du retard.  On situe le point de rupture.  (Comparé à l'IMC
        figé qui, lui, diverge quelle que soit la vitesse.)
  E12 — ERREUR D'ENCODEUR / VARIATION DE VITESSE DE BROCHE : le modèle interne
        (IMC, AIMC) suppose une fréquence de dent f_fund.  Si la vitesse réelle
        diffère (erreur d'encodeur, dérive RPM), les oscillateurs internes sont
        décalés.  On balaie l'écart de fréquence et on mesure la dégradation
        (la question « et si la vitesse de broche varie ? »).

Sortie : figs_aimc_ext/fig_ext_drift.png + tableaux console.
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
from aimc_controller import AIMCController
from newmark_solver import NewmarkSimulator
from material_removal import thinning_schedule   # dérive de fréquence PHYSIQUE

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
COL = {'LQG': '#2E8B57', 'IMC-fige': '#1f77b4', 'DARC-FF': '#DC143C', 'AIMC': '#6A0DAD'}


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


def setup(plate_r, ap, T_end):
    sim = NewmarkSimulator(plate_r, dt=DT, T_end=T_end, ft=FT, tau=TAU, verbose=False)
    xp = np.minimum(FT*NT*RPM/60 * sim.t_vec, LP)
    kp = np.clip(np.round(xp/LP * 2000).astype(int), 0, 2000)
    n_per = int(np.round(TAU/DT))
    a3, a4, a42, a43 = precompute_nonlinear_periodic(
        DT, n_per, sim.nstep, OMEGA, NT, RT, ETA_H, PHI_ST, PHI_EX,
        HP-ap, HP, K1C, K2C, KT_NOMINAL)
    return sim, kp, (a3, a4, a42, a43), n_per


def run(sim, kp, arrs, ctrl, k_scale_t=None):
    a3, a4, a42, a43 = arrs
    if hasattr(ctrl, 'reset'):
        ctrl.reset()
    return sim.simulate(a3, a4, kp, controller=ctrl, alpha4_2_t=a42, alpha4_3_t=a43,
                        sensor_noise=SENSOR_NOISE, sensor_rng=np.random.default_rng(1),
                        k_scale_t=k_scale_t, progress=False)


print("="*78)
print(" RÉSULTATS ÉTENDUS AIMC (2) : bande passante de suivi + erreur d'encodeur")
print("="*78)
t_glob = time.time()
ap = 0.3e-3
plate_d = build_plate(ap, 0.0)
Dp_ff = plate_d.get_Dp_at(int(np.clip(round(FT*NT*RPM/60*0.25/LP*2000),0,2000)))[0]
f_fund = 1.0 / (int(np.round(TAU/DT)) * DT)


def make_ctrls(n_per):
    lqg = LQGController(plate_d, dt=DT, verbose=False)
    lqg.optimize_weights(w_q_list=[1e14], w_qd_list=[1e8], w_r=1.0, gain_norm_max=1e10)
    lqg.discretize_observer()
    imc = IMCLQGController(plate_d, dt=DT, f_fund=f_fund, Dp_dist=Dp_ff,
                           n_harm=IMC_HARMONICS, q_dist=IMC_QDIST, verbose=False)
    aimc = AIMCController(plate_d, dt=DT, f_fund=f_fund, Dp_dist=Dp_ff, verbose=False)
    return lqg, imc, aimc


# ================================================================
# E8 : limite de vitesse de dérive
# ================================================================
print("\n[E8] Bande passante de suivi : amincissement PHYSIQUE de paroi "
      "(0.6 mm = 15% de 4 mm) à débits variables")
ramp_durs = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05]   # s (du plus lent au plus rapide)
REMOVED = 0.15 * BP                            # 0.6 mm retirés (15% -> -15% freq)
e8 = {'ramp': [], 'rate_mm': [], 'IMC-fige': [], 'AIMC': [], 'lag': []}
for T_ramp in ramp_durs:
    T_end = 0.2 + T_ramp + 0.3            # avant + rampe + après
    sim, kp, arrs, n_per = setup(plate_d, ap, T_end)
    t = sim.t_vec
    # dérive PHYSIQUE : ω(t) ∝ B(t), amincissement de REMOVED sur [0.2, 0.2+T_ramp]
    k_scale, B_t, rho_true = thinning_schedule(
        t, B_nominal=BP, removed_total=REMOVED, t_start=0.2, t_end=0.2+T_ramp)
    rate_mm = REMOVED*1e3 / T_ramp             # débit d'amincissement [mm/s]
    lqg, imc, aimc = make_ctrls(n_per)
    # RMS sur la fenêtre "après rampe établie" (dernier 0.2 s)
    i0 = int((T_end-0.2)/DT)
    r_imc = run(sim, kp, arrs, imc, k_scale_t=k_scale)
    r_aimc = run(sim, kp, arrs, aimc, k_scale_t=k_scale)
    def rms_tail(r):
        i1 = r['stop_idx']
        return float(np.sqrt(np.mean(r['y'][min(i0,i1):i1+1]**2)))*1e6 if i1 > i0 else float('nan')
    # retard de suivi : rho_hat vs rho_true au moment où la rampe finit
    rr = np.array(aimc.trace_rho)
    i_end_ramp = min(int((0.2+T_ramp)/DT), len(rr)-1)
    lag = (rho_true[i_end_ramp] - rr[i_end_ramp])*100   # % de retard
    e8['ramp'].append(T_ramp); e8['rate_mm'].append(rate_mm)
    e8['IMC-fige'].append(rms_tail(r_imc))
    e8['AIMC'].append(rms_tail(r_aimc)); e8['lag'].append(lag)
    print(f"  amincissement {rate_mm:5.2f} mm/s ({15.0/T_ramp:5.0f}%/s) : "
          f"IMC-fige={rms_tail(r_imc):6.3f}  AIMC={rms_tail(r_aimc):6.3f} um  "
          f"retard_ID={lag:+.1f}%")

# ================================================================
# E12 : erreur d'encodeur / variation de vitesse de broche
# ================================================================
print("\n[E12] Erreur de fréquence de dent supposée (encodeur/RPM) — S1 nominal")
# La plante est nominale ; le contrôleur suppose f_fund*(1+df).
df_axis = np.array([-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4]) / 100.0
sim1, kp1, arrs1, n_per1 = setup(plate_d, ap, 0.5)
e12 = {'df': [], 'IMC-fige': [], 'DARC-FF': [], 'AIMC': []}
for df in df_axis:
    f_err = f_fund*(1+df)
    imc = IMCLQGController(plate_d, dt=DT, f_fund=f_err, Dp_dist=Dp_ff,
                           n_harm=IMC_HARMONICS, q_dist=IMC_QDIST, verbose=False)
    aimc = AIMCController(plate_d, dt=DT, f_fund=f_err, Dp_dist=Dp_ff, verbose=False)
    darc = DARCController(plate_d, dt=DT, base_w_q=1e14, base_w_qd=1e8,
                          n_per=n_per1, u_max=150.0, verbose=False)
    darc.design_periodic_feedforward(FT, arrs1[0][:n_per1], Dp_ff, n_harm=FF_HARMONICS)
    def rr(c):
        r = run(sim1, kp1, arrs1, c); i = r['stop_idx']
        return min(float(np.sqrt(np.mean(r['y'][:i+1]**2)))*1e6, 30.0)
    e12['df'].append(df*100)
    e12['IMC-fige'].append(rr(imc)); e12['AIMC'].append(rr(aimc)); e12['DARC-FF'].append(rr(darc))
    print(f"  df={df*100:+.1f}% : IMC={e12['IMC-fige'][-1]:6.3f}  "
          f"DARC={e12['DARC-FF'][-1]:6.3f}  AIMC={e12['AIMC'][-1]:6.3f} um")

# ================================================================
# FIGURE
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
ax = axes[0]
rates = e8['rate_mm']
ax.plot(rates, e8['IMC-fige'], 'o-', color=COL['IMC-fige'], lw=2, ms=6, label='IMC-figé')
ax.plot(rates, e8['AIMC'], 's-', color=COL['AIMC'], lw=2, ms=6, label='AIMC')
ax.set_xlabel("débit d'amincissement de paroi (mm/s)")
ax.set_ylabel("$y_{RMS}$ après rampe (µm)")
ax.set_xscale('log')
ax.set_title("(E8) Suivi de la dérive PHYSIQUE (amincissement)\nAIMC suit, IMC figé diverge",
             fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.4)

ax = axes[1]
for tag in ('IMC-fige', 'DARC-FF', 'AIMC'):
    ax.plot(e12['df'], e12[tag], 'o-', color=COL[tag], lw=2, ms=6, label=tag)
ax.set_xlabel("erreur de fréquence de dent supposée (%)")
ax.set_ylabel("$y_{RMS}$ (µm)")
ax.set_title("(E12) Sensibilité à l'erreur d'encodeur / vitesse broche\n"
             "(les modèles internes exigent une bonne fréquence)", fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_ext_drift.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"\n  ✓ {OUT_DIR}/fig_ext_drift.png")
print(f"\nTemps total : {time.time()-t_glob:.0f} s")
