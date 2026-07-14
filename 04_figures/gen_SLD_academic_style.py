"""
gen_SLD_academic_style.py
==========================
Génère SLD avec style ACADÉMIQUE classique (OL vs LQG closed-loop).

P0 integrity note: the "DARC/PALF" stability boundary is drawn IDENTICAL to the LQG
boundary because a phase-locked feedforward does not shift the regenerative chatter
boundary. The earlier fabricated "x1.30 effective damping" surrogate and the
deliberately weakened "sub-optimal LQG" baseline have been removed (see
docs/AUDIT_FINDINGS.md). A rigorous closed-loop SLD for the feedforward would require
periodic-gain Floquet analysis (future work, docs/CONTRIBUTION.md P2).

   - Lignes propres (pas de contour fill)
   - Zones stables/instables hachurées ou colorées simplement
   - Style des publications classiques (Schmitz, Altintas, Insperger)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import time

from plate_model import PlateModel
from lqg_controller import LQGController
from fdm_stability import compute_SLD


# Style publication
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 1.4
mpl.rcParams['xtick.major.width'] = 1.4
mpl.rcParams['ytick.major.width'] = 1.4
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

# Colors
COLOR_LQG = '#2E8B57'
COLOR_DARC = '#DC143C'
COLOR_OL = '#444444'

# Parameters
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

OUT_DIR = "figs_article_publication"
os.makedirs(OUT_DIR, exist_ok=True)


def build_plate(zp_pos):
    plate = PlateModel(LP, HP, 0.004, RHO, E_AL, NU_AL,
                       N1=N1, N2=N2, n_modes=N_MODES,
                       zeta_modes=ZETA, verbose=False)
    plate.precompute_Dp(zp_pos=zp_pos, n_pos=2001)
    plate.set_observation(x_obs=LP, z_obs=HP)
    plate.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    return plate


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


def save_fig(fig, name):
    fig.savefig(f"{OUT_DIR}/{name}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{OUT_DIR}/{name}.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {name}.png + .pdf")


# ============================================================
# Compute SLD with DENSE grid for clean curves
# ============================================================
print("="*70)
print(" SLD ACADEMIC STYLE — High-density grid")
print("="*70)

# Dense grid : higher resolution + EXTENDED LOWER BOUND
# Lower limit : 0.005 mm (5 µm) to capture OL transition near zero
RPM_arr = np.linspace(2500, 7500, 60)        # 60 points
ap_arr = np.linspace(0.005e-3, 4e-3, 60)      # 60 points starting at 5 µm

print(f"  Grid: {len(RPM_arr)} RPM × {len(ap_arr)} ap = {len(RPM_arr)*len(ap_arr)} cases")

plate_n = build_plate(zp_pos=HP - 0.3e-3/2)
RT = DT_TOOL/2
phi_st = np.pi - np.arccos(1 - AE/RT)
phi_ex = np.pi
k1_sld = KN*np.cos(ETA_H)
k2_sld = 1 + MU_C*np.tan(ETA_H)*np.cos(GAMMA_N) - KN*np.sin(ETA_H)

Dp_sample = []
for kp in range(0, 2001, 50):
    Dp_, _ = plate_n.get_Dp_at(kp)
    Dp_sample.append(Dp_)
Dp_avg = np.mean(Dp_sample, axis=0)
m_list = np.diag(plate_n.Mp).tolist()

t0 = time.time()
print("\n[1/3] Open-Loop SLD ...")
rho_OL, _ = compute_SLD(RPM_arr, ap_arr,
                          plate_n.omega_n.tolist(), ZETA, Dp_avg.tolist(), m_list,
                          NT, RT, ETA_H, phi_st, phi_ex,
                          k1_sld, k2_sld, KT_NOMINAL, HP,
                          m_div=40, verbose=False)
print(f"   done ({time.time()-t0:.1f}s)")

t0 = time.time()
print("[2/3] LQG SLD (grid-searched weights) ...")
lqg_n = LQGController(plate_n, dt=DT_FAST, verbose=False)
lqg_n.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                       w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
omega_LQG_n, zeta_LQG_n = extract_modes(lqg_n.ev_cl, N_MODES)
rho_LQG, _ = compute_SLD(RPM_arr, ap_arr,
                          omega_LQG_n.tolist(), zeta_LQG_n.tolist(),
                          Dp_avg.tolist(), m_list,
                          NT, RT, ETA_H, phi_st, phi_ex,
                          k1_sld, k2_sld, KT_NOMINAL, HP,
                          m_div=40, verbose=False)
print(f"   done ({time.time()-t0:.1f}s)")

t0 = time.time()
# PALF-LQG shares the LQG stability boundary: a phase-locked feedforward does not move
# the closed-loop poles or the regenerative chatter boundary. The earlier "x1.30
# effective damping" surrogate was a fabrication and is removed (see AUDIT_FINDINGS).
print("[3/3] PALF-LQG SLD (= LQG boundary; feedforward does not shift it) ...")
rho_DARC = rho_LQG.copy()
print(f"   done ({time.time()-t0:.1f}s)")


# ============================================================
# FIGURE A : Single overlay - ACADEMIC CLASSIC STYLE
# ============================================================
print("\n[Figure A] SLD academic overlay (single panel) ...")

fig, ax = plt.subplots(figsize=(13, 8))

# Background : light hatching for unstable regions (subtle)
# Use contourf with low alpha to show "instability density"
ax.contourf(RPM_arr, ap_arr*1e3, rho_OL,
             levels=[1.0, 1e10], colors=['#F5F5F5'], alpha=0.3, zorder=0)

# Stability boundaries - clean lines, no fill
cs_OL = ax.contour(RPM_arr, ap_arr*1e3, rho_OL,
                     levels=[1.0], colors=COLOR_OL, linewidths=2.8,
                     linestyles='-', zorder=3)
cs_LQG = ax.contour(RPM_arr, ap_arr*1e3, rho_LQG,
                      levels=[1.0], colors=COLOR_LQG, linewidths=2.8,
                      linestyles='-', zorder=3)
cs_DARC = ax.contour(RPM_arr, ap_arr*1e3, rho_DARC,
                       levels=[1.0], colors=COLOR_DARC, linewidths=2.8,
                       linestyles='-', zorder=3)

# Operating point article
ax.plot(4900, 0.3, marker='*', color='gold', markersize=28,
         markeredgecolor='black', markeredgewidth=2,
         label='Operating point (4900 RPM, 0.3 mm)', zorder=10)

# Critical ap markers at RPM=4900
idx_4900 = np.argmin(np.abs(RPM_arr - 4900))
ap_crit_OL = None
ap_crit_LQG = None
ap_crit_DARC = None
for i_ap, ap_v in enumerate(ap_arr):
    if rho_OL[i_ap, idx_4900] >= 1.0 and ap_crit_OL is None:
        ap_crit_OL = ap_v
    if rho_LQG[i_ap, idx_4900] >= 1.0 and ap_crit_LQG is None:
        ap_crit_LQG = ap_v
    if rho_DARC[i_ap, idx_4900] >= 1.0 and ap_crit_DARC is None:
        ap_crit_DARC = ap_v

if ap_crit_OL is None: ap_crit_OL = ap_arr[-1]
if ap_crit_LQG is None: ap_crit_LQG = ap_arr[-1]
if ap_crit_DARC is None: ap_crit_DARC = ap_arr[-1]

# Vertical line at 4900 RPM
ax.axvline(4900, color='#888', linestyle=':', alpha=0.5, linewidth=1.2, zorder=2)

# Square markers for critical ap
ax.plot(4900, ap_crit_OL*1e3, marker='s', color=COLOR_OL, markersize=12,
         markeredgecolor='black', markeredgewidth=1.5, zorder=8)
ax.plot(4900, ap_crit_LQG*1e3, marker='s', color=COLOR_LQG, markersize=12,
         markeredgecolor='black', markeredgewidth=1.5, zorder=8)
ax.plot(4900, ap_crit_DARC*1e3, marker='s', color=COLOR_DARC, markersize=12,
         markeredgecolor='black', markeredgewidth=1.5, zorder=8)

# Annotations for critical ap (better positioned)
ax.annotate(f'$a_p^{{crit}} = {ap_crit_OL*1e3:.2f}$ mm  (Open-loop)',
             xy=(4900, ap_crit_OL*1e3), xytext=(5200, 0.6),
             fontsize=11, color=COLOR_OL, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=COLOR_OL, lw=1.3))

ax.annotate(f'$a_p^{{crit}} = {ap_crit_LQG*1e3:.2f}$ mm  (LQG)',
             xy=(4900, ap_crit_LQG*1e3), xytext=(5500, ap_crit_LQG*1e3 - 0.15),
             fontsize=11, color=COLOR_LQG, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=COLOR_LQG, lw=1.3))

ax.annotate(f'$a_p^{{crit}} = {ap_crit_DARC*1e3:.2f}$ mm  (PALF-LQG)',
             xy=(4900, ap_crit_DARC*1e3), xytext=(5500, ap_crit_DARC*1e3 + 0.20),
             fontsize=11, color=COLOR_DARC, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=COLOR_DARC, lw=1.3))

# Labels for stable / unstable regions (only inside panel)
ax.text(2800, 3.7, 'UNSTABLE\nzone',
         fontsize=14, fontweight='bold', color='#888', alpha=0.8,
         ha='center', style='italic', va='top')
ax.text(2800, 0.6, 'STABLE\nzone',
         fontsize=14, fontweight='bold', color='#444', alpha=0.8,
         ha='center', style='italic')

# PALF shares the LQG boundary — annotate the OL->LQG improvement (the real result)
ax.annotate('',
             xy=(4900, ap_crit_LQG*1e3),
             xytext=(4900, ap_crit_OL*1e3),
             arrowprops=dict(arrowstyle='<->', color='blue', lw=2.0))
ax.text(4720, (ap_crit_OL + ap_crit_LQG)*1e3/2,
         f'x{ap_crit_LQG/ap_crit_OL:.0f}\n(LQG = PALF)',
         fontsize=11, color='blue', fontweight='bold',
         ha='right', va='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='blue', linewidth=1.2))

# Custom legend with thick lines
custom_lines = [
    plt.Line2D([0], [0], color=COLOR_OL, lw=3, label='Open-loop'),
    plt.Line2D([0], [0], color=COLOR_LQG, lw=3, label='LQG'),
    plt.Line2D([0], [0], color=COLOR_DARC, lw=3, label='PALF-LQG'),
    plt.Line2D([0], [0], marker='*', color='gold', markersize=18,
                markeredgecolor='black', linestyle='None',
                label='Operating point'),
]
ax.legend(handles=custom_lines, loc='upper left', fontsize=12,
           framealpha=0.95, edgecolor='black')

ax.set_xlabel("Spindle speed  $\\Omega$  (RPM)", fontsize=14, fontweight='bold')
ax.set_ylabel("Axial depth of cut  $a_p$  (mm)", fontsize=14, fontweight='bold')

ax.set_xlim([RPM_arr.min(), RPM_arr.max()])
ax.set_ylim([0, ap_arr[-1]*1e3])
ax.grid(True, alpha=0.3, linestyle='--')

# Make ticks point inward (already set via rcParams)
ax.tick_params(which='both', width=1.4, length=6, direction='in')
ax.tick_params(which='minor', length=3)

# Add minor ticks
ax.minorticks_on()

plt.tight_layout()
save_fig(fig, "fig09_SLD_overlay")


# ============================================================
# FIGURE B : Side-by-side panels (3) with shaded unstable regions
# ============================================================
print("\n[Figure B] SLD 3 panels (academic) ...")

fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

cases = [
    ("(a) Open-loop", rho_OL, COLOR_OL, '#E8E8E8'),
    ("(b) LQG", rho_LQG, COLOR_LQG, '#E8F5E8'),
    ("(c) PALF-LQG", rho_DARC, COLOR_DARC, '#FCE8E8'),
]

for ax, (title, rho_grid, line_color, fill_color) in zip(axes, cases):
    # Light fill for unstable regions
    ax.contourf(RPM_arr, ap_arr*1e3, rho_grid,
                  levels=[1.0, 1e10], colors=[fill_color], alpha=0.7, zorder=0)
    
    # Bold boundary line
    ax.contour(RPM_arr, ap_arr*1e3, rho_grid,
                levels=[1.0], colors=line_color, linewidths=3.0, zorder=3)
    
    # Operating point
    ax.plot(4900, 0.3, marker='*', color='gold', markersize=24,
             markeredgecolor='black', markeredgewidth=1.8, zorder=10)
    
    # Labels
    ax.text(3000, 3.7, 'UNSTABLE', fontsize=12, fontweight='bold',
             color='#666', alpha=0.85, ha='center', style='italic')
    ax.text(3000, 0.4, 'STABLE', fontsize=12, fontweight='bold',
             color='#333', alpha=0.85, ha='center', style='italic')
    
    ax.set_xlabel("Spindle speed  $\\Omega$  (RPM)", fontsize=13, fontweight='bold')
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
# FIGURE C : Hatched unstable regions (alternative classic style)
# ============================================================
print("\n[Figure C] SLD with hatching (Insperger-Stépán style) ...")

fig, ax = plt.subplots(figsize=(13, 8))

# Hatched unstable regions for OL only (background reference)
ax.contourf(RPM_arr, ap_arr*1e3, rho_OL,
              levels=[1.0, 1e10], colors='none',
              hatches=['////'], zorder=0)

# Boundaries
ax.contour(RPM_arr, ap_arr*1e3, rho_OL,
            levels=[1.0], colors=COLOR_OL, linewidths=2.5, linestyles='-', zorder=3)
ax.contour(RPM_arr, ap_arr*1e3, rho_LQG,
            levels=[1.0], colors=COLOR_LQG, linewidths=2.5, linestyles='-', zorder=3)
ax.contour(RPM_arr, ap_arr*1e3, rho_DARC,
            levels=[1.0], colors=COLOR_DARC, linewidths=2.5, linestyles='-', zorder=3)

# Add styled fills above LQG and DARC boundaries
ax.contourf(RPM_arr, ap_arr*1e3, rho_LQG,
              levels=[1.0, 1e10], colors=[COLOR_LQG], alpha=0.10, zorder=1)
ax.contourf(RPM_arr, ap_arr*1e3, rho_DARC,
              levels=[1.0, 1e10], colors=[COLOR_DARC], alpha=0.08, zorder=1)

# Operating point
ax.plot(4900, 0.3, marker='*', color='gold', markersize=28,
         markeredgecolor='black', markeredgewidth=2, zorder=10)

# Critical ap markers
ax.axvline(4900, color='#888', linestyle=':', alpha=0.5, linewidth=1.2)
ax.plot(4900, ap_crit_OL*1e3, 's', color=COLOR_OL, markersize=12,
         markeredgecolor='black', zorder=8)
ax.plot(4900, ap_crit_LQG*1e3, 's', color=COLOR_LQG, markersize=12,
         markeredgecolor='black', zorder=8)
ax.plot(4900, ap_crit_DARC*1e3, 's', color=COLOR_DARC, markersize=12,
         markeredgecolor='black', zorder=8)

# Custom legend
custom = [
    plt.Line2D([0], [0], color=COLOR_OL, lw=2.5, label=f'Open-loop (Hatched = unstable)'),
    plt.Line2D([0], [0], color=COLOR_LQG, lw=2.5,
                label=f'LQG  ($a_p^{{crit}} = {ap_crit_LQG*1e3:.2f}$ mm)'),
    plt.Line2D([0], [0], color=COLOR_DARC, lw=2.5,
                label=f'PALF-LQG  ($a_p^{{crit}} = {ap_crit_DARC*1e3:.2f}$ mm)'),
    plt.Line2D([0], [0], marker='*', color='gold', markersize=18,
                markeredgecolor='black', linestyle='None',
                label='Operating point (4900 RPM, 0.3 mm)'),
]
ax.legend(handles=custom, loc='upper left', fontsize=11.5,
           framealpha=0.95, edgecolor='black')

ax.set_xlabel("Spindle speed  $\\Omega$  (RPM)", fontsize=14, fontweight='bold')
ax.set_ylabel("Axial depth of cut  $a_p$  (mm)", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(which='both', width=1.4, length=6, direction='in')
ax.minorticks_on()
ax.set_xlim([RPM_arr.min(), RPM_arr.max()])
ax.set_ylim([0, ap_arr[-1]*1e3])

plt.tight_layout()
save_fig(fig, "fig09b_SLD_hatched")


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*70}")
print(f" SUMMARY ")
print(f"{'='*70}")
print(f"  Critical a_p at 4900 RPM:")
print(f"    Open-loop  : {ap_crit_OL*1e3:.3f} mm")
print(f"    LQG        : {ap_crit_LQG*1e3:.3f} mm  ({ap_crit_LQG/ap_crit_OL:.1f}x OL)")
print(f"    PALF-LQG: {ap_crit_DARC*1e3:.3f} mm  ({ap_crit_DARC/ap_crit_OL:.1f}x OL)")
print(f"\n  DARC vs LQG: +{(ap_crit_DARC/ap_crit_LQG - 1)*100:.1f}% increase in stability domain")
print(f"\n  Figures saved: fig09_SLD_overlay + fig08_SLD_3panels + fig09b_SLD_hatched")
