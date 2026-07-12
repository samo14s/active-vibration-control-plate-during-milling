"""
main_imc_baseline.py
====================
BASELINE À MODÈLE INTERNE : la question que posera tout rapporteur.

L'argument central de l'ancienne version du package (« un retour ne peut pas
rejeter une perturbation périodique ») est FAUX dès que la période de broche
est connue — et l'encodeur est déjà supposé disponible pour le feedforward du
DARC.  Ce script compare donc, à information égale :

   1. LQG          : retour pur (baseline historique)
   2. IMC-LQG      : LQG + modèle interne harmonique (Kalman augmenté,
                     principe du modèle interne / disturbance accommodation).
                     N'utilise AUCUN modèle de coupe — seulement la période
                     de dent (encodeur) et le capteur de déplacement.
   3. DARC-FF      : LQG + feedforward inverse-modèle pré-calculé sur le
                     modèle de coupe NOMINAL (sans résidu NN, pour isoler la
                     couche anticipative).  Indépendant du capteur, mais
                     dépendant du modèle de coupe et de la phase.

Scénarios identiques à main_simulation.py (S1..S4, protocole honnête : les
contrôleurs ne reçoivent JAMAIS les paramètres réels perturbés), plus un
balayage de bruit capteur qui matérialise le compromis structurel :
   - IMC-LQG : meilleur rejet, s'adapte à K_T inconnu, mais repose sur le
     CAPTEUR de déplacement ;
   - DARC-FF : insensible au capteur (table de phase), mais figé sur son
     modèle de coupe nominal.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from plate_model import PlateModel
from milling_force import precompute_nonlinear_periodic
from lqg_controller import LQGController
from imc_lqg_controller import IMCLQGController
from darc_controller import DARCController
from newmark_solver import NewmarkSimulator

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
T_END = 0.5
SENSOR_NOISE = 1.0e-7
FF_HARMONICS = 30
IMC_HARMONICS = 8       # modèle interne : 8 harmoniques (244...1951 Hz)
IMC_QDIST = 1e3         # agilité de poursuite des états de perturbation

OUT_DIR = "figs_imc_baseline"
os.makedirs(OUT_DIR, exist_ok=True)


def build_plate(zp_pos, freq_perturb=0.0):
    plate = PlateModel(LP, HP, BP, RHO, E_AL, NU_AL,
                       n_modes=N_MODES, zeta_modes=ZETA, verbose=False)
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


print("="*72)
print(" BASELINE MODELE INTERNE :  LQG  vs  IMC-LQG  vs  DARC-FF ")
print("="*72)
t0 = time.time()

scenarios_def = [
    ("S1 - Nominal article",      0.3e-3, KT_NOMINAL, 0.0),
    ("S2 - Aggressive ap=0.6mm",  0.6e-3, KT_NOMINAL, 0.0),
    ("S3 - Uncertainty w-15%",    0.3e-3, KT_NOMINAL, -0.15),
    ("S4 - High K_T +30%",        0.3e-3, 1.3*KT_NOMINAL, 0.0),
]

RT = DT_TOOL/2
phi_st = np.pi - np.arccos(1 - AE/RT)
phi_ex = np.pi
Omega = 2*np.pi*RPM/60
tau = 60/(NT*RPM)
k1 = KN*np.cos(ETA_H)
k2 = 1 + MU_C*np.tan(ETA_H)*np.cos(GAMMA_N) - KN*np.sin(ETA_H)

results = []
for name, ap, KT_actual, freq_perturb in scenarios_def:
    print(f"\n--- {name} ---")
    plate_d = build_plate(zp_pos=HP - ap/2, freq_perturb=0.0)
    plate_r = build_plate(zp_pos=HP - ap/2, freq_perturb=freq_perturb)
    sim = NewmarkSimulator(plate_r, dt=DT, T_end=T_END, ft=FT, tau=tau,
                           verbose=False)
    xp_path = np.minimum(FT*NT*RPM/60 * sim.t_vec, LP)
    kp_idx = np.clip(np.round(xp_path/LP * 2000).astype(int), 0, 2000)
    n_per = int(np.round(tau/DT))

    # Coefficients REELS (pilotent le procédé)
    a3, a4, a42, a43 = precompute_nonlinear_periodic(
        DT, n_per, sim.nstep, Omega, NT, RT, ETA_H, phi_st, phi_ex,
        HP-ap, HP, k1, k2, KT_actual)
    # Coefficients NOMINAUX (seuls connus des contrôleurs)
    if KT_actual != KT_NOMINAL:
        a3_n, _, _, _ = precompute_nonlinear_periodic(
            DT, n_per, sim.nstep, Omega, NT, RT, ETA_H, phi_st, phi_ex,
            HP-ap, HP, k1, k2, KT_NOMINAL)
    else:
        a3_n = a3

    Dp_ff = plate_d.get_Dp_at(int(kp_idx[sim.nstep // 2]))[0]
    f_fund = 1.0 / (n_per * DT)   # fondamentale simulée (info encodeur)

    def run(ctrl):
        return sim.simulate(a3, a4, kp_idx, controller=ctrl,
                            alpha4_2_t=a42, alpha4_3_t=a43,
                            sensor_noise=SENSOR_NOISE,
                            sensor_rng=np.random.default_rng(1),
                            progress=False)

    # 1. LQG
    lqg = LQGController(plate_d, dt=DT, verbose=False)
    lqg.optimize_weights(w_q_list=[1e14], w_qd_list=[1e8], w_r=1.0,
                         gain_norm_max=1e10)
    lqg.discretize_observer()
    r_lqg = run(lqg)

    # 2. IMC-LQG (modèle interne, pas de modèle de coupe)
    imc = IMCLQGController(plate_d, dt=DT, f_fund=f_fund, Dp_dist=Dp_ff,
                           n_harm=IMC_HARMONICS, q_dist=IMC_QDIST,
                           verbose=False)
    imc.reset()
    r_imc = run(imc)

    # 3. DARC-FF (feedforward inverse-modèle NOMINAL, sans NN)
    darc = DARCController(plate_d, dt=DT, base_w_q=1e14, base_w_qd=1e8,
                          n_per=n_per, u_max=150.0, verbose=False)
    darc.design_periodic_feedforward(FT, a3_n[:n_per], Dp_ff,
                                     n_harm=FF_HARMONICS)
    r_ff = run(darc)

    row = {'name': name}
    for tag, r in (('LQG', r_lqg), ('IMC-LQG', r_imc), ('DARC-FF', r_ff)):
        i = r['stop_idx']
        row[tag] = dict(
            y_rms=float(np.sqrt(np.mean(r['y'][:i+1]**2)))*1e6,
            u_max=float(np.max(np.abs(r['u'][:i+1]))))
    results.append(row)
    print(f"   LQG     : {row['LQG']['y_rms']:.4f} um "
          f"(u_max {row['LQG']['u_max']:.1f} V)")
    print(f"   IMC-LQG : {row['IMC-LQG']['y_rms']:.4f} um "
          f"(u_max {row['IMC-LQG']['u_max']:.1f} V)")
    print(f"   DARC-FF : {row['DARC-FF']['y_rms']:.4f} um "
          f"(u_max {row['DARC-FF']['u_max']:.1f} V)")

# ============================================================
# Balayage bruit capteur : le compromis structurel IMC vs FF
# ============================================================
print(f"\n--- Balayage bruit capteur (S1) ---")
ap = 0.3e-3
plate_sr = build_plate(zp_pos=HP - ap/2)
sim_sr = NewmarkSimulator(plate_sr, dt=DT, T_end=T_END, ft=FT, tau=tau,
                          verbose=False)
xp_sr = np.minimum(FT*NT*RPM/60 * sim_sr.t_vec, LP)
kp_sr = np.clip(np.round(xp_sr/LP * 2000).astype(int), 0, 2000)
n_per = int(np.round(tau/DT))
a3, a4, a42, a43 = precompute_nonlinear_periodic(
    DT, n_per, sim_sr.nstep, Omega, NT, RT, ETA_H, phi_st, phi_ex,
    HP-ap, HP, k1, k2, KT_NOMINAL)
Dp_ff = plate_sr.get_Dp_at(int(kp_sr[sim_sr.nstep // 2]))[0]
f_fund = 1.0 / (n_per * DT)

darc_sr = DARCController(plate_sr, dt=DT, base_w_q=1e14, base_w_qd=1e8,
                         n_per=n_per, u_max=150.0, verbose=False)
darc_sr.design_periodic_feedforward(FT, a3[:n_per], Dp_ff, n_harm=FF_HARMONICS)

noise_levels = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0]) * 1e-6
sweep = {'LQG': [], 'IMC-LQG': [], 'DARC-FF': []}
for sn in noise_levels:
    lqg_sr = LQGController(plate_sr, dt=DT, verbose=False)
    lqg_sr.optimize_weights(w_q_list=[1e14], w_qd_list=[1e8], w_r=1.0,
                            gain_norm_max=1e10)
    lqg_sr.discretize_observer()
    imc_sr = IMCLQGController(plate_sr, dt=DT, f_fund=f_fund, Dp_dist=Dp_ff,
                              n_harm=IMC_HARMONICS, q_dist=IMC_QDIST,
                              verbose=False)
    for tag, ctrl in (('LQG', lqg_sr), ('IMC-LQG', imc_sr),
                      ('DARC-FF', darc_sr)):
        if hasattr(ctrl, 'reset'):
            ctrl.reset()
        r = sim_sr.simulate(a3, a4, kp_sr, controller=ctrl,
                            alpha4_2_t=a42, alpha4_3_t=a43,
                            sensor_noise=sn,
                            sensor_rng=np.random.default_rng(7),
                            progress=False)
        i = r['stop_idx']
        sweep[tag].append(float(np.sqrt(np.mean(r['y'][:i+1]**2)))*1e6)
    print(f"   noise={sn*1e6:.1f}um : LQG={sweep['LQG'][-1]:.3f}  "
          f"IMC={sweep['IMC-LQG'][-1]:.3f}  FF={sweep['DARC-FF'][-1]:.3f} um")

# ============================================================
# Figure + résumé
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
names = [r['name'].split(' - ')[0] for r in results]
x = np.arange(len(names)); w = 0.26
colors = {'LQG': '#2E8B57', 'IMC-LQG': '#1f77b4', 'DARC-FF': '#DC143C'}
ax = axes[0]
for j, tag in enumerate(('LQG', 'IMC-LQG', 'DARC-FF')):
    vals = [r[tag]['y_rms'] for r in results]
    bars = ax.bar(x + (j-1)*w, vals, w, color=colors[tag], alpha=0.85,
                  edgecolor='k', label=tag)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v*1.02, f'{v:.3f}',
                ha='center', fontsize=8, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel("$y_{RMS}$ (µm)")
ax.set_title("Rejet de la perturbation périodique\n"
             "(protocole honnête : aucun contrôleur ne connaît "
             "les paramètres réels)", fontweight='bold')
ax.legend(); ax.grid(True, axis='y', alpha=0.5)

ax = axes[1]
for tag in ('LQG', 'IMC-LQG', 'DARC-FF'):
    ax.plot(noise_levels*1e6, sweep[tag], 'o-', color=colors[tag],
            lw=2.2, ms=7, label=tag)
ax.set_xlabel("Bruit capteur (µm RMS)")
ax.set_ylabel("$y_{RMS}$ (µm)")
ax.set_title("Compromis structurel :\n"
             "IMC-LQG dépend du capteur, DARC-FF de son modèle de coupe",
             fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.5); ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_imc_baseline.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"\n  ✓ {OUT_DIR}/fig_imc_baseline.png")

print(f"\n{'='*72}")
print(" RESUME : LQG vs IMC-LQG vs DARC-FF (y_RMS, um)")
print(f"{'='*72}")
print(f"  {'Scenario':<28}{'LQG':>10}{'IMC-LQG':>10}{'DARC-FF':>10}")
for r in results:
    print(f"  {r['name']:<28}{r['LQG']['y_rms']:>10.4f}"
          f"{r['IMC-LQG']['y_rms']:>10.4f}{r['DARC-FF']['y_rms']:>10.4f}")
print(f"\nTemps total : {time.time()-t0:.1f} s")
