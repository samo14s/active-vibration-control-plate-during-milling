"""
main_piezo_structure.py
=======================
P9 — a MODELING refinement of the P5 plant: include the bonded piezo patch's
STRUCTURAL mass and bending stiffness in the plate FEM (previously the patch entered
ONLY as an actuation force, and the modal analysis ran on the bare aluminium plate).

WHY. The 20x60x0.7 mm PZT patch bonded near the clamped root is not massless/stiffness-
less: as a surface layer offset from the plate mid-plane it raises the local bending
rigidity by ~58 % and the local areal mass by ~46 % (PlateModel.add_piezo_structure,
composite-laminate rigidity about the neutral axis + added mass). Near the root — where
the fundamental cantilever mode has its highest curvature — the stiffening dominates and
lifts the modes.

RESULT (vs Du et al. 2024 Table 4 measured). Including the patch structure moves the
FEM natural frequencies UP 0.4-3.8 % and IMPROVES the match to the article's measured
plate, most strikingly for mode 1 (bare -3.5 % -> instrumented +0.2 %, i.e. 541 vs 540
Hz). Mode 2 goes the other way (+0.2 % -> +2.4 %); the net RMS frequency error vs the
measured column improves 1.83 % -> 1.67 %. The A-ESO-ADRC controller, re-designed and
evaluated consistently on the (stiffer) instrumented plant, holds ~0.35 um nominally.

HONEST CAVEATS.
  * Bending-only (Kirchhoff, w-DOF) elements cannot carry the bending-membrane coupling
    the offset neutral axis introduces (the laminate B-matrix); this smears in the
    dominant stiffening + mass loading and neglects that second-order coupling — the
    usual treatment for thin surface-bonded patches.
  * The PZT areal density (rho_Pe = 7500 kg/m^3, PZT-5H) is an ASSUMPTION — the article
    reports d31/E_Pe but not density. Sensitivity is shown below (7500 vs 7800).
  * Whether the article's *measured* Table-4 frequencies were taken on the bare or the
    instrumented plate is not stated; the improved mode-1 match is consistent with (but
    not proof of) the patch having been bonded during the modal test.
This refinement is offered as an AVAILABLE higher-fidelity plant (opt-in via
add_piezo_structure); the committed P1-P8 results keep the bare-plate model so their
numbers stay comparable. Run:  python main_piezo_structure.py  ->  figs_piezo_structure/
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plate_model import PlateModel
from newmark_solver import NewmarkSimulator
from lqg_controller import LQGController
from milling_force import cutting_constants, precompute_alpha_periodic
from adrc_controller import AdaptiveESO_ADRC_Controller

LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU_AL = 2830, 69e9, 0.33
NT = 3; DT_TOOL = 0.010
ETA_H = np.deg2rad(35); GAMMA_N = np.deg2rad(15)
KT_NOMINAL = 925e6; KN = 0.26; MU_C = 0.20
RPM = 4900; FT = 0.02e-3; AE = 0.1e-3
D31 = 175e-12; H_PA = 0.7e-3; E_PE = 63e9; NU_PE = 0.35
RHO_PE = 7500.0
N1, N2 = 30, 24
N_MODES_PLANT = 5; N_MODES_CTRL = 3
ZETA_PLANT = [0.0031, 0.0017, 0.0027, 0.0056, 0.0035]
MEAS_NOISE_STD = 1e-8; KALMAN_V = 1e-12
DT = 5e-5; T_END = 0.5
PHI_ST = np.pi - np.arccos(1 - AE/(DT_TOOL/2)); PHI_EX = np.pi
OMEGA = 2*np.pi*RPM/60; RT = DT_TOOL/2; TAU = 60/(NT*RPM)
K1, K2 = cutting_constants(KN, MU_C, ETA_H, GAMMA_N)
N_PER = int(np.round(TAU/DT)); W_TOOTH = 2*np.pi*NT*RPM/60.0
PERF = dict(w_q=1e16, w_qd=1e8, sd=3e3); CERT = dict(w_q=1e14, w_qd=1e8, sd=1e4)
# Du et al. (2024) Table 4
ART_THEORY = [537, 1101, 2805, 3423, 4254]
ART_MEAS = [540, 1068, 2787, 3351, 4122]
OUT_DIR = "figs_piezo_structure"
os.makedirs(OUT_DIR, exist_ok=True)


def build(instrumented, rho_pe=RHO_PE):
    p = PlateModel(LP, HP, BP, RHO, E_AL, NU_AL, N1=N1, N2=N2, n_modes=N_MODES_PLANT,
                   zeta_modes=ZETA_PLANT, verbose=False)
    if instrumented:
        p.add_piezo_structure(0, 0.020, 0, 0.060, E_PE, NU_PE, H_PA, rho_pe, verbose=False)
    p.precompute_Dp(zp_pos=HP-0.3e-3/2, n_pos=2001)
    p.set_observation(x_obs=LP, z_obs=HP)
    p.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    return p


def make_forces(sim, ap, KT):
    v = FT*NT*RPM/60
    kp = np.clip(np.round(np.minimum(v*sim.t_vec, LP)/LP*2000).astype(int), 0, 2000)
    a3, a4 = precompute_alpha_periodic(DT, N_PER, sim.nstep, OMEGA, NT, RT, ETA_H,
                                       PHI_ST, PHI_EX, HP-ap, HP, K1, K2, KT)
    return a3, a4, kp


def aeso(dv):
    R = [dict(kind='hrc', w_q=PERF['w_q'], w_qd=PERF['w_qd'], sigma_d=PERF['sd'],
              n_harm=4, g_base=150.0, lam=5.0),
         dict(kind='eso', w_q=PERF['w_q'], w_qd=PERF['w_qd'], sigma_d=PERF['sd']),
         dict(kind='eso', w_q=CERT['w_q'], w_qd=CERT['w_qd'], sigma_d=1e3),
         dict(kind='eso', w_q=CERT['w_q'], w_qd=CERT['w_qd'], sigma_d=CERT['sd'])]
    return AdaptiveESO_ADRC_Controller(dv, DT, rungs=R, w_tooth=W_TOOTH,
                                       kalman_V=KALMAN_V, verbose=False)


def nominal_rms(plant):
    dv = plant.truncated_view(N_MODES_CTRL)
    c = aeso(dv)
    sim = NewmarkSimulator(plant, dt=DT, T_end=T_END, ft=FT, tau=TAU, verbose=False)
    a3, a4, kp = make_forces(sim, 0.3e-3, KT_NOMINAL)
    c.reset_adaptation()
    r = sim.simulate(a3, a4, kp, controller=c, meas_noise_std=MEAS_NOISE_STD,
                     rng=np.random.default_rng(1234), progress=False)
    i = r['stop_idx']
    return np.inf if i < sim.nstep-2 else np.sqrt(np.mean(r['y'][:i+1]**2))*1e6


print("="*72)
print(" P9 — piezo-patch STRUCTURE in the plate FEM (modeling refinement) ")
print("="*72)
t0 = time.time()
bare = build(False)
inst = build(True)
ps = inst.piezo_structure
print(f"\npatch structure: {ps['n_patch_elem']} elements, bending x{ps['k_scale']:.3f}, "
      f"mass x{ps['m_scale']:.3f}, neutral-axis shift {ps['z_na']*1e3:.3f} mm")

print("\n--- natural frequencies: bare vs instrumented vs article (Table 4) ---")
print(f"{'mode':>4} {'bare(Hz)':>9} {'instr(Hz)':>10} {'shift':>7} {'art.meas':>9} "
      f"{'bare err':>9} {'instr err':>10}")
eb, ei = [], []
for k in range(N_MODES_PLANT):
    fb, fi = bare.freq_n[k], inst.freq_n[k]
    db, di = 100*(fb/ART_MEAS[k]-1), 100*(fi/ART_MEAS[k]-1)
    eb.append(db); ei.append(di)
    print(f"{k+1:>4} {fb:>9.1f} {fi:>10.1f} {100*(fi/fb-1):>+6.1f}% {ART_MEAS[k]:>9} "
          f"{db:>+8.1f}% {di:>+9.1f}%")
rms_b = np.sqrt(np.mean(np.array(eb)**2)); rms_i = np.sqrt(np.mean(np.array(ei)**2))
print(f"\nRMS frequency error vs measured:  bare {rms_b:.2f}%  ->  instrumented {rms_i:.2f}%")

# density sensitivity
inst78 = build(True, rho_pe=7800.0)
print(f"\nrho_Pe sensitivity: mode-1 = {inst.freq_n[0]:.1f} Hz (7500) vs "
      f"{inst78.freq_n[0]:.1f} Hz (7800)  [measured 540]")

print("\n--- A-ESO-ADRC nominal (designed + evaluated consistently per plant) ---")
r_bare = nominal_rms(bare); r_inst = nominal_rms(inst)
print(f"  bare plant:         y_RMS = {r_bare:.4f} um")
print(f"  instrumented plant: y_RMS = {r_inst:.4f} um")

# ---------------- figure ----------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))
x = np.arange(N_MODES_PLANT); w = 0.27
axA.bar(x-w, [bare.freq_n[k] for k in range(5)], w, label='bare plate', color='#8A8A8A')
axA.bar(x, [inst.freq_n[k] for k in range(5)], w, label='+ piezo structure', color='#DC143C')
axA.plot(x, ART_MEAS, 'k_', ms=18, mew=2, label='article measured')
axA.set_xticks(x); axA.set_xticklabels([f"m{k+1}" for k in range(5)])
axA.set_ylabel('natural frequency (Hz)'); axA.set_yscale('log')
axA.set_title('(a) modes: bare vs instrumented vs article'); axA.legend(fontsize=8)
axA.grid(alpha=0.3, axis='y')
axB.axhline(0, color='k', lw=0.8)
axB.bar(x-w/2, eb, w, label='bare', color='#8A8A8A')
axB.bar(x+w/2, ei, w, label='instrumented', color='#DC143C')
axB.set_xticks(x); axB.set_xticklabels([f"m{k+1}" for k in range(5)])
axB.set_ylabel('frequency error vs measured (%)')
axB.set_title(f'(b) article match improves: RMS {rms_b:.2f}% -> {rms_i:.2f}%\n'
              '(mode 1 -3.5% -> +0.2%)')
axB.legend(fontsize=8); axB.grid(alpha=0.3, axis='y')
plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f"{OUT_DIR}/fig_piezo_structure.{ext}", dpi=300, bbox_inches='tight')
print(f"\n  saved {OUT_DIR}/fig_piezo_structure.png|pdf")

print("\n=== P9 SUMMARY ===")
print(f"  patch structure raises modes 0.4-3.8%; mode-1 match {eb[0]:+.1f}% -> {ei[0]:+.1f}% "
      f"(541 vs 540 Hz measured); net RMS error {rms_b:.2f}% -> {rms_i:.2f}%.")
print(f"  A-ESO-ADRC nominal {r_bare:.3f} (bare) vs {r_inst:.3f} um (instrumented).")
print("  Offered as an opt-in higher-fidelity plant (add_piezo_structure); committed "
      "P1-P8 keep the bare model. Caveats: bending-membrane coupling neglected; "
      "rho_Pe assumed 7500 kg/m^3; article modal-test instrumentation unstated.")
print(f"\nTotal time {time.time()-t0:.1f}s  ->  {OUT_DIR}/")
