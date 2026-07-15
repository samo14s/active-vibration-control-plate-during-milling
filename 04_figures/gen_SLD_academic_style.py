"""
gen_SLD_academic_style.py
==========================
Academic-style (serif, clean-line) rendering of the stability lobe diagrams:
Open-Loop vs LQG vs ESO-ADRC (certified design).

Same RIGOROUS computation as the authoritative main_simulation.py — coupled
closed-loop monodromy with the controller embedded, worst of 3 tool positions
(x = 0, L/4, L/2) — only the styling differs (Schmitz / Altintas / Insperger
publication look). The historical per-mode "closed-loop equivalent damping"
surrogate of this script was removed together with the PALF-era labels.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from plate_model import PlateModel
from milling_force import cutting_constants
from lqg_controller import LQGController
from adrc_controller import ESO_ADRC_Controller
from fdm_stability import compute_closed_loop_SLD, compute_closed_loop_SLD_generic

# Style publication
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 1.4
mpl.rcParams['xtick.major.width'] = 1.4
mpl.rcParams['ytick.major.width'] = 1.4
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

COLOR_OL = '#444444'
COLOR_LQG = '#2E8B57'
COLOR_ADRC = '#1E5AA8'

# Parameters (article)
LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU_AL = 2830, 69e9, 0.33
NT = 3;  DT_TOOL = 0.010
ETA_H = np.deg2rad(35);  GAMMA_N = np.deg2rad(15)
KT_NOMINAL = 925e6;  KN = 0.26;  MU_C = 0.20
AE = 0.1e-3
D31 = 175e-12;  H_PA = 0.7e-3
E_PE = 63e9;    NU_PE = 0.35
N1, N2 = 30, 24
N_MODES = 3
ZETA = [0.0031, 0.0017, 0.0027]
DT_FAST = 5e-5
KALMAN_V = 1e-12
CERT_DESIGN = (1e14, 1e8, 1e4)      # from the main_simulation design grid

OUT_DIR = "figs_article_publication"
os.makedirs(OUT_DIR, exist_ok=True)


def build_plate(zp_pos):
    plate = PlateModel(LP, HP, BP, RHO, E_AL, NU_AL,
                       N1=N1, N2=N2, n_modes=N_MODES,
                       zeta_modes=ZETA, verbose=False)
    plate.precompute_Dp(zp_pos=zp_pos, n_pos=2001)
    plate.set_observation(x_obs=LP, z_obs=HP)
    plate.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    return plate


def save_fig(fig, name):
    fig.savefig(f"{OUT_DIR}/{name}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{OUT_DIR}/{name}.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {name}.png + .pdf")


# ============================================================
# Rigorous monodromy grids (worst of 3 tool positions)
# ============================================================
print("="*70)
print(" SLD ACADEMIC STYLE — rigorous closed-loop monodromy, worst of 3 pos.")
print("="*70)

RPM_arr = np.linspace(2500, 7500, 30)
ap_arr = np.linspace(0.0001, 4e-3, 25)

plate_n = build_plate(zp_pos=HP - 0.3e-3/2)
RT = DT_TOOL/2
phi_st = np.pi - np.arccos(1 - AE/RT)
phi_ex = np.pi
k1_sld, k2_sld = cutting_constants(KN, MU_C, ETA_H, GAMMA_N)

KP_POSITIONS = [0, 500, 1000]
Dp_positions = [plate_n.get_Dp_at(kp)[0] for kp in KP_POSITIONS]

lqg_n = LQGController(plate_n, dt=DT_FAST, verbose=False, kalman_V=KALMAN_V)
lqg_n.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                       w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
adrc_n = ESO_ADRC_Controller(plate_n, DT_FAST, *CERT_DESIGN,
                             kalman_V=KALMAN_V, verbose=False)
A_con, B_cy, K_con = adrc_n.controller_realization()

t0 = time.time()
print("\n[1/3] Open-Loop (coupled monodromy) ...")
rho_OL = None
for Dp_pos in Dp_positions:
    r = compute_closed_loop_SLD(RPM_arr, ap_arr, plate_n.omega_n, ZETA, Dp_pos,
                                None, None, None, None, None, None,
                                NT, RT, ETA_H, phi_st, phi_ex,
                                k1_sld, k2_sld, KT_NOMINAL, HP,
                                m_div=20, verbose=False)
    rho_OL = r if rho_OL is None else np.maximum(rho_OL, r)
print(f"   done ({time.time()-t0:.1f}s)")

t0 = time.time()
print("[2/3] LQG (closed-loop monodromy) ...")
rho_LQG = None
for Dp_pos in Dp_positions:
    r = compute_closed_loop_SLD(RPM_arr, ap_arr, plate_n.omega_n, ZETA, Dp_pos,
                                plate_n.H_Pe_modal, plate_n.D_obs,
                                lqg_n.A, lqg_n.B, lqg_n.K_lqr, lqg_n.L_kal,
                                NT, RT, ETA_H, phi_st, phi_ex,
                                k1_sld, k2_sld, KT_NOMINAL, HP,
                                m_div=20, verbose=False)
    rho_LQG = r if rho_LQG is None else np.maximum(rho_LQG, r)
print(f"   done ({time.time()-t0:.1f}s)")

t0 = time.time()
print("[3/3] ESO-ADRC certified design (generic closed-loop monodromy) ...")
rho_ADRC = None
for Dp_pos in Dp_positions:
    r = compute_closed_loop_SLD_generic(RPM_arr, ap_arr, plate_n.omega_n, ZETA,
                                        Dp_pos, plate_n.H_Pe_modal, plate_n.D_obs,
                                        A_con, B_cy, K_con,
                                        NT, RT, ETA_H, phi_st, phi_ex,
                                        k1_sld, k2_sld, KT_NOMINAL, HP,
                                        m_div=20, verbose=False)
    rho_ADRC = r if rho_ADRC is None else np.maximum(rho_ADRC, r)
print(f"   done ({time.time()-t0:.1f}s)")

# Critical depths at 4900 RPM
idx_4900 = np.argmin(np.abs(RPM_arr - 4900))


def ap_crit_of(rho_grid):
    for i_ap, ap_v in enumerate(ap_arr):
        if rho_grid[i_ap, idx_4900] >= 1.0:
            return ap_v
    return ap_arr[-1]


ap_crit_OL = ap_crit_of(rho_OL)
ap_crit_LQG = ap_crit_of(rho_LQG)
ap_crit_ADRC = ap_crit_of(rho_ADRC)


# ============================================================
# FIGURE A : single overlay — academic classic style
# ============================================================
print("\n[Figure A] SLD academic overlay ...")
fig, ax = plt.subplots(figsize=(13, 8))

ax.contourf(RPM_arr, ap_arr*1e3, rho_OL,
            levels=[1.0, 1e10], colors=['#F5F5F5'], alpha=0.3, zorder=0)
ax.contour(RPM_arr, ap_arr*1e3, rho_OL, levels=[1.0], colors=COLOR_OL,
           linewidths=2.8, zorder=3)
ax.contour(RPM_arr, ap_arr*1e3, rho_LQG, levels=[1.0], colors=COLOR_LQG,
           linewidths=2.8, zorder=3)
ax.contour(RPM_arr, ap_arr*1e3, rho_ADRC, levels=[1.0], colors=COLOR_ADRC,
           linewidths=2.4, linestyles='--', zorder=3)

ax.plot(4900, 0.3, marker='*', color='gold', markersize=28,
        markeredgecolor='black', markeredgewidth=2, zorder=10)
ax.axvline(4900, color='#888', linestyle=':', alpha=0.5, linewidth=1.2, zorder=2)
for apc, colr in [(ap_crit_OL, COLOR_OL), (ap_crit_LQG, COLOR_LQG),
                  (ap_crit_ADRC, COLOR_ADRC)]:
    ax.plot(4900, apc*1e3, marker='s', color=colr, markersize=12,
            markeredgecolor='black', markeredgewidth=1.5, zorder=8)

ax.annotate(f'$a_p^{{crit}} = {ap_crit_OL*1e3:.2f}$ mm  (Open-loop)',
            xy=(4900, ap_crit_OL*1e3), xytext=(5200, 0.6),
            fontsize=11, color=COLOR_OL, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLOR_OL, lw=1.3))
ax.annotate(f'$a_p^{{crit}} = {ap_crit_LQG*1e3:.2f}$ mm  (LQG)',
            xy=(4900, ap_crit_LQG*1e3), xytext=(5500, ap_crit_LQG*1e3 - 0.35),
            fontsize=11, color=COLOR_LQG, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLOR_LQG, lw=1.3))
ax.annotate(f'$a_p^{{crit}} = {ap_crit_ADRC*1e3:.2f}$ mm  (ESO-ADRC)',
            xy=(4900, ap_crit_ADRC*1e3), xytext=(5500, ap_crit_ADRC*1e3 + 0.30),
            fontsize=11, color=COLOR_ADRC, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLOR_ADRC, lw=1.3))

ax.text(2800, 3.7, 'UNSTABLE\nzone', fontsize=14, fontweight='bold',
        color='#888', alpha=0.8, ha='center', style='italic', va='top')
ax.text(2800, 0.6, 'STABLE\nzone', fontsize=14, fontweight='bold',
        color='#444', alpha=0.8, ha='center', style='italic')

custom_lines = [
    plt.Line2D([0], [0], color=COLOR_OL, lw=3, label='Open-loop'),
    plt.Line2D([0], [0], color=COLOR_LQG, lw=3, label='LQG'),
    plt.Line2D([0], [0], color=COLOR_ADRC, lw=3, ls='--',
               label='ESO-ADRC (certified rung)'),
    plt.Line2D([0], [0], marker='*', color='gold', markersize=18,
               markeredgecolor='black', linestyle='None',
               label='Operating point (4900 RPM, 0.3 mm)'),
]
ax.legend(handles=custom_lines, loc='upper left', fontsize=12,
          framealpha=0.95, edgecolor='black')
ax.set_xlabel("Spindle speed  $\\Omega$  (RPM)", fontsize=14, fontweight='bold')
ax.set_ylabel("Axial depth of cut  $a_p$  (mm)", fontsize=14, fontweight='bold')
ax.set_xlim([RPM_arr.min(), RPM_arr.max()])
ax.set_ylim([0, ap_arr[-1]*1e3])
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(which='both', width=1.4, length=6, direction='in')
ax.minorticks_on()
plt.tight_layout()
save_fig(fig, "fig09_SLD_overlay")


# ============================================================
# FIGURE B : side-by-side panels
# ============================================================
print("\n[Figure B] SLD 3 panels (academic) ...")
fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
cases = [
    ("(a) Open-loop", rho_OL, COLOR_OL, '#E8E8E8'),
    ("(b) LQG", rho_LQG, COLOR_LQG, '#E8F5E8'),
    ("(c) ESO-ADRC (certified)", rho_ADRC, COLOR_ADRC, '#E8EEF8'),
]
for ax, (title, rho_grid, line_color, fill_color) in zip(axes, cases):
    ax.contourf(RPM_arr, ap_arr*1e3, rho_grid,
                levels=[1.0, 1e10], colors=[fill_color], alpha=0.7, zorder=0)
    ax.contour(RPM_arr, ap_arr*1e3, rho_grid,
               levels=[1.0], colors=line_color, linewidths=3.0, zorder=3)
    ax.plot(4900, 0.3, marker='*', color='gold', markersize=24,
            markeredgecolor='black', markeredgewidth=1.8, zorder=10)
    ax.text(3000, 3.7, 'UNSTABLE', fontsize=12, fontweight='bold',
            color='#666', alpha=0.85, ha='center', style='italic')
    ax.text(3000, 0.4, 'STABLE', fontsize=12, fontweight='bold',
            color='#333', alpha=0.85, ha='center', style='italic')
    ax.set_xlabel("Spindle speed  $\\Omega$  (RPM)", fontsize=13,
                  fontweight='bold')
    if title.startswith("(a)"):
        ax.set_ylabel("Axial depth  $a_p$  (mm)", fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(which='both', width=1.4, length=6, direction='in')
    ax.minorticks_on()
    ax.set_xlim([RPM_arr.min(), RPM_arr.max()])
    ax.set_ylim([0, ap_arr[-1]*1e3])
plt.tight_layout()
save_fig(fig, "fig08_SLD_3panels")


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*70}")
print(f" SUMMARY (rigorous monodromy, worst of 3 tool positions)")
print(f"{'='*70}")
print(f"  Critical a_p at 4900 RPM:")
print(f"    Open-loop : {ap_crit_OL*1e3:.3f} mm")
print(f"    LQG       : {ap_crit_LQG*1e3:.3f} mm  ({ap_crit_LQG/ap_crit_OL:.1f}x OL)")
print(f"    ESO-ADRC  : {ap_crit_ADRC*1e3:.3f} mm  ({ap_crit_ADRC/ap_crit_OL:.1f}x OL)")
print(f"\n  Figures saved: fig09_SLD_overlay + fig08_SLD_3panels")
