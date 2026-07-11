"""
main_geometric_nonlinear.py
===========================
CONTRIBUTION : effet de la NON-LINÉARITÉ GÉOMÉTRIQUE (von Kármán) sur le
contrôle actif des vibrations en fraisage des pièces flexibles à parois
minces (AL6061 homogène).

Verrou (voir docs/verrou_nonlinearite.md) : deux littératures matures mais
DISJOINTES —
  (1) contrôle actif du fraisage de parois minces → modèles LINÉAIRES
      (Kirchhoff/Mindlin + réduction modale) ;
  (2) contrôle actif géométriquement non linéaire (von Kármán) des
      plaques/coques intelligentes FGM/piézolaminées → chargements
      thermiques / harmoniques / aléatoires, JAMAIS l'excitation de coupe
      RÉGÉNÉRATIVE À RETARD.
Aucune n'a traité le contrôle actif du broutement de fraisage sur un
modèle EF géométriquement non linéaire.

Résultats produits :
  A. Diagramme de bifurcation : le modèle LINÉAIRE prédit une divergence
     exponentielle au-delà de la limite de stabilité, tandis que von
     Kármán BORNE la croissance en un CYCLE LIMITE (bifurcation de Hopf).
  B. Traces temporelles + FFT : décalage de fréquence dépendant de
     l'amplitude (raidissement).
  C. Contrôle actif : le régulateur LQG conçu sur le modèle LINÉAIRE,
     appliqué à la dynamique non linéaire — le modèle linéaire ne peut
     PAS représenter le cycle limite (il prédit « instable = ∞ »), le
     modèle von Kármán donne l'amplitude contrôlée réelle et le
     dimensionnement correct de l'actionneur.
  D. Configuration : paroi "wall" (base encastrée + bords latéraux
     bridés en membrane — nervure flanquée de matière épaisse) vs
     cantilever pur (3 bords libres) — quantifie QUAND la non-linéarité
     compte.

Sortie : results_geom_nl/  (figures + metrics.json)
"""
import os
import sys
import json
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _d in ("01_core", "02_controllers", "03_analysis"):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plate_model import PlateModel
from milling_force import precompute_alpha_periodic
from lqg_controller import LQGController
from newmark_solver import NewmarkSimulator
from newmark_nonlinear_solver import NewmarkNonlinearSimulator
from von_karman_rom import VonKarmanROM

# ============================================================
LP, HP = 0.100, 0.080
RHO, E_AL, NU_AL = 2830, 69e9, 0.33
NT = 3; DT_TOOL = 0.010
ETA_H = np.deg2rad(35); GAMMA_N = np.deg2rad(15)
KT = 925e6; KN = 0.26; MU_C = 0.20
RPM = 5600; FT = 0.02e-3; AE = 0.1e-3   # 5600 RPM: clean Hopf onset in ap
#                                         (a stable forced branch exists at
#                                          low ap for the h=2 mm wall)
D31 = 175e-12; H_PA = 0.7e-3; E_PE = 63e9; NU_PE = 0.35
N1, N2 = 30, 24
N_MODES = 3
ZETA = [0.0031, 0.0017, 0.0027]
DT = 5e-5

BP = 2.0e-3                 # thin wall 2 mm (aerospace-like)
BC = "wall"                # membrane BC (rib flanked by thicker stock)
T_END = 0.5
THR = 0.040                # 40 mm cutoff (w/L = 0.5, generous)

PHI_ST = np.pi - np.arccos(1 - AE/(DT_TOOL/2)); PHI_EX = np.pi
RT = DT_TOOL/2
K1 = KN*np.cos(ETA_H)
K2 = 1 + MU_C*np.tan(ETA_H)*np.cos(GAMMA_N) - KN*np.sin(ETA_H)

OUT = os.path.join(_ROOT, "results_geom_nl")
os.makedirs(OUT, exist_ok=True)
COL = {"lin": "#B00020", "vk": "#1E5AA8", "lqg_lin": "#B00020",
       "lqg_vk": "#1E5AA8", "ol": "#777777"}


def build_plate(ap, bp=BP):
    p = PlateModel(LP, HP, bp, RHO, E_AL, NU_AL, N1=N1, N2=N2,
                   n_modes=N_MODES, zeta_modes=ZETA, verbose=False)
    p.precompute_Dp(zp_pos=HP - ap/2, n_pos=2001)
    p.set_observation(x_obs=LP, z_obs=HP)
    p.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    return p


def setup(p, ap, T=T_END):
    Omega = 2*np.pi*RPM/60
    tau = 60/(NT*RPM)
    n_per = int(round(tau/DT))
    s = NewmarkSimulator(p, dt=DT, T_end=T, ft=FT, tau=tau, verbose=False)
    v = FT*NT*RPM/60
    kp = np.clip(np.round(np.minimum(v*s.t_vec, LP)/LP*2000).astype(int),
                 0, 2000)
    a3, a4 = precompute_alpha_periodic(DT, n_per, s.nstep, Omega, NT, RT,
                                       ETA_H, PHI_ST, PHI_EX, HP-ap, HP,
                                       K1, K2, KT)
    return s, kp, a3, a4, tau


U_SAT = 150.0                # realistic piezo voltage saturation [V]


class SaturatedLQG:
    """LQG designed on the LINEAR model, with realistic ±U_SAT actuator
    saturation — the controller an engineer would actually deploy."""
    def __init__(self, lqg, umax=U_SAT):
        self.lqg = lqg
        self.umax = umax

    def step(self, x_hat_prev, u_prev, y_meas):
        x_hat, u = self.lqg.step(x_hat_prev, u_prev, y_meas)
        return x_hat, float(np.clip(u, -self.umax, self.umax))


def lc_amplitude(y, t0=0.35):
    """steady limit-cycle peak amplitude (max over the last window)."""
    i0 = int(t0/DT)
    seg = y[i0:]
    if len(seg) < 10:
        return np.nan
    return np.max(np.abs(seg))


def is_unstable(res, forced_floor_um=50.0):
    """unstable if it diverged OR grew far past the forced (few-µm) level."""
    return (res['diverged_at'] > 0
            or np.max(np.abs(res['y'][:res['stop_idx']+1]))*1e6 > forced_floor_um)


def is_steady(y, tol=0.15):
    """compare peak over two late windows to detect a settled limit cycle."""
    i1, i2 = int(0.30/DT), int(0.40/DT)
    i3 = int(0.50/DT)
    a = np.max(np.abs(y[i1:i2])) if i2 < len(y) else np.nan
    b = np.max(np.abs(y[i2:i3])) if i3 <= len(y) else a
    if not np.isfinite(a) or a == 0:
        return False
    return abs(b - a)/a < tol


print("="*72)
print(" GEOMETRIC NONLINEARITY IN ACTIVE THIN-WALL MILLING CONTROL")
print("="*72)
t_glob = time.time()
p0 = build_plate(0.3e-3)
print(f" wall h={BP*1e3:.1f} mm, BC={BC}, RPM={RPM}, "
      f"f = {np.round(p0.freq_n, 0)} Hz")
rom0 = VonKarmanROM(p0, membrane_bc=BC, verbose=False)
_, ratio_bb, g_bb = rom0.backbone_single_mode(0, [0.5, 1.0, 2.0])
print(f" von Kármán backbone (mode 1): ω_nl/ω_l @ A/h=0.5,1,2 = "
      f"{ratio_bb[0]:.3f}, {ratio_bb[1]:.3f}, {ratio_bb[2]:.3f}")


# ============================================================
# A. Bifurcation diagram (open loop)
# ============================================================
print("\n[A] Bifurcation diagram (open loop) ...")
ap_list = np.array([0.05, 0.08, 0.10, 0.12, 0.15, 0.18,
                    0.20, 0.25, 0.30, 0.40])*1e-3
bif = []
for ap in ap_list:
    p = build_plate(ap)
    rom = VonKarmanROM(p, membrane_bc=BC, verbose=False)
    s, kp, a3, a4, tau = setup(p, ap)
    rl = s.simulate(a3, a4, kp, controller=None, progress=False,
                    stop_threshold=THR)
    snl = NewmarkNonlinearSimulator(p, dt=DT, T_end=T_END, ft=FT, tau=tau,
                                    rom=rom, verbose=False)
    rv = snl.simulate(a3, a4, kp, controller=None, progress=False,
                      stop_threshold=THR)
    lin_unstable = is_unstable(rl)
    lin_amp = np.max(np.abs(rl['y'][:rl['stop_idx']+1]))*1e6
    vk_bounded = rv['diverged_at'] == 0
    vk_amp = lc_amplitude(rv['y'])*1e6 if vk_bounded else \
        np.max(np.abs(rv['y'][:rv['stop_idx']+1]))*1e6
    bif.append(dict(ap_mm=float(ap*1e3), lin_unstable=bool(lin_unstable),
                    lin_amp_um=float(lin_amp), vk_bounded=bool(vk_bounded),
                    vk_amp_um=float(vk_amp),
                    vk_steady=bool(is_steady(rv['y']))))
    print(f"  ap={ap*1e3:.2f}mm: LIN {'UNSTABLE' if lin_unstable else 'stable':8s}"
          f" (ymax={lin_amp:8.1f}um) | vK "
          f"{'limit-cycle' if vk_bounded else 'unbounded':11s}"
          f" A={vk_amp:8.1f}um (w/h={vk_amp/1e6/BP:.2f})")

# locate the linear stability limit
ap_crit = next((b['ap_mm'] for b in bif if b['lin_unstable']), None)


# ============================================================
# B. Representative time trace + FFT  (choose smallest unstable ap)
# ============================================================
# choose the demo cut: unstable (linear) yet a bounded, steady von Kármán
# limit cycle of MODERATE amplitude (w/h ≤ 1 ⇒ within validity, visible
# hardening) — take the largest such amplitude
cand = [b for b in bif if b['lin_unstable'] and b['vk_bounded']
        and b['vk_steady'] and b['vk_amp_um']*1e-6/BP <= 1.0]
if cand:
    ap_demo = max(cand, key=lambda b: b['vk_amp_um'])['ap_mm']*1e-3
else:
    ap_demo = next((b['ap_mm']*1e-3 for b in bif if b['lin_unstable']
                    and b['vk_bounded']), 0.20e-3)
print(f"\n[B] Time trace / FFT at ap = {ap_demo*1e3:.2f} mm ...")
p = build_plate(ap_demo)
rom = VonKarmanROM(p, membrane_bc=BC, verbose=False)
s, kp, a3, a4, tau = setup(p, ap_demo)
res_lin = s.simulate(a3, a4, kp, controller=None, progress=False,
                     stop_threshold=THR)
snl = NewmarkNonlinearSimulator(p, dt=DT, T_end=T_END, ft=FT, tau=tau,
                                rom=rom, verbose=False)
res_vk = snl.simulate(a3, a4, kp, controller=None, progress=False,
                      stop_threshold=THR)


def fft_um(y, i0, i1):
    seg = y[i0:i1]
    n = len(seg)
    if n < 16:
        return None, None
    win = np.hanning(n)
    Y = np.abs(np.fft.rfft(seg*win))/n*2*1e6
    f = np.fft.rfftfreq(n, DT)
    return f, Y


# ============================================================
# C. Active control : LQG (designed on LINEAR model) on both plants
# ============================================================
print(f"\n[C] Active LQG control at ap = {ap_demo*1e3:.2f} mm ...")
lqg = LQGController(p, dt=DT, verbose=False)
lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                     w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
lqg.discretize_observer()
ctrl = SaturatedLQG(lqg, umax=U_SAT)          # realistic ±150 V actuator

# LQG (±150 V) on the LINEAR plant — what the design tool predicts
res_lin_ctrl = s.simulate(a3, a4, kp, controller=ctrl, progress=False,
                          stop_threshold=THR)
# same controller on the von Kármán plant — the true response
snl_ctrl = NewmarkNonlinearSimulator(p, dt=DT, T_end=T_END, ft=FT, tau=tau,
                                     rom=rom, verbose=False)
res_vk_ctrl = snl_ctrl.simulate(a3, a4, kp, controller=ctrl, progress=False,
                                stop_threshold=THR)


def summ(res, controlled):
    i_end = res['stop_idx']
    y = res['y'][:i_end+1]
    unbounded = res['diverged_at'] > 0
    amp = lc_amplitude(res['y'])*1e6 if not unbounded else \
        np.max(np.abs(y))*1e6
    out = dict(unbounded=bool(unbounded), amp_um=float(amp),
               ymax_um=float(np.max(np.abs(y))*1e6))
    if controlled:
        u = res['u'][:i_end+1]
        out['u_max_V'] = float(np.max(np.abs(u)))
        out['u_rms_V'] = float(np.sqrt(np.mean(u**2)))
    return out


ctrl_metrics = {
    "ap_mm": ap_demo*1e3,
    "open_loop": {"linear": summ(res_lin, False),
                  "von_karman": summ(res_vk, False)},
    "lqg": {"linear_plant": summ(res_lin_ctrl, True),
            "von_karman_plant": summ(res_vk_ctrl, True)},
}
print(f"  open loop     : LIN {'unbounded' if ctrl_metrics['open_loop']['linear']['unbounded'] else 'bounded'}"
      f" (ymax={ctrl_metrics['open_loop']['linear']['ymax_um']:.0f}um) | "
      f"vK limit-cycle A={ctrl_metrics['open_loop']['von_karman']['amp_um']:.0f}um")
print(f"  LQG lin plant : amp={ctrl_metrics['lqg']['linear_plant']['amp_um']:.1f}um "
      f"u_max={ctrl_metrics['lqg']['linear_plant']['u_max_V']:.1f}V "
      f"({'unbounded' if ctrl_metrics['lqg']['linear_plant']['unbounded'] else 'stabilised'})")
print(f"  LQG vK  plant : amp={ctrl_metrics['lqg']['von_karman_plant']['amp_um']:.1f}um "
      f"u_max={ctrl_metrics['lqg']['von_karman_plant']['u_max_V']:.1f}V "
      f"({'unbounded' if ctrl_metrics['lqg']['von_karman_plant']['unbounded'] else 'stabilised'})")


# ============================================================
# D. Cantilever vs wall (when does nonlinearity matter?)
# ============================================================
print("\n[D] Configuration comparison (cantilever vs wall) ...")
config_cmp = {}
for bc in ("cantilever", "wall"):
    romc = VonKarmanROM(build_plate(ap_demo), membrane_bc=bc, verbose=False)
    _, r, g = romc.backbone_single_mode(0, [0.5, 1.0, 2.0])
    config_cmp[bc] = dict(gamma_1111=float(g),
                          ratio_05=float(r[0]), ratio_10=float(r[1]),
                          ratio_20=float(r[2]))
    print(f"  {bc:10s}: Γ_1111={g:.2e}  ω_nl/ω_l(A/h=1)={r[1]:.3f}")


# ============================================================
# FIGURES
# ============================================================
# Fig 1 : bifurcation diagram
fig, ax = plt.subplots(figsize=(9, 5.6))
aps = [b['ap_mm'] for b in bif]
lin_stable_amp = [b['lin_amp_um'] if not b['lin_unstable'] else np.nan
                  for b in bif]
lin_unst_amp = [b['lin_amp_um'] if b['lin_unstable'] else np.nan
                for b in bif]
vk_amp = [b['vk_amp_um'] for b in bif]
ax.semilogy(aps, lin_stable_amp, 'o-', color=COL['ol'],
            label='linear — stable (forced)', markersize=6)
ax.semilogy(aps, lin_unst_amp, 'v', color=COL['lin'], markersize=9,
            label='linear — UNSTABLE (→ divergence, cut at %.0f mm)' % (THR*1e3))
ax.semilogy(aps, vk_amp, 's-', color=COL['vk'], markersize=6,
            label='von Kármán — bounded limit cycle')
if ap_crit is not None:
    ax.axvline(ap_crit, color='k', ls=':', alpha=0.6)
    ax.text(ap_crit, ax.get_ylim()[1]*0.5,
            ' linear\n stability\n limit', fontsize=9)
ax.axhline(BP*1e6, color='green', ls='--', alpha=0.5)
ax.text(aps[0], BP*1e6*1.1, 'wall thickness h', color='green', fontsize=9)
ax.set_xlabel('axial depth $a_p$ (mm)', fontsize=12)
ax.set_ylabel('vibration amplitude (µm, log)', fontsize=12)
ax.set_title('Bifurcation: linear divergence vs von Kármán limit cycle\n'
             '(thin wall h=%.0f mm, %s BC, %d RPM)' % (BP*1e3, BC, RPM),
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='lower right'); ax.grid(True, which='both', alpha=0.4)
plt.tight_layout(); plt.savefig(f"{OUT}/fig01_bifurcation.png", dpi=170,
                                bbox_inches='tight'); plt.close()
print("  ✓ fig01_bifurcation.png")

# Fig 2 : time traces
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
il = res_lin['stop_idx']; iv = res_vk['stop_idx']
axes[0].plot(res_lin['t'][:il+1], res_lin['y'][:il+1]*1e6,
             color=COL['lin'], lw=0.8,
             label='linear model (diverges)')
axes[0].plot(res_vk['t'][:iv+1], res_vk['y'][:iv+1]*1e6,
             color=COL['vk'], lw=0.8, label='von Kármán (limit cycle)')
axes[0].axhline(BP*1e6, color='green', ls='--', alpha=0.4)
axes[0].axhline(-BP*1e6, color='green', ls='--', alpha=0.4)
axes[0].set_ylabel('$y_p$ (µm)', fontsize=11)
axes[0].set_title('ap = %.2f mm : open-loop response' % (ap_demo*1e3),
                  fontsize=11, fontweight='bold')
axes[0].legend(fontsize=10, loc='upper left'); axes[0].grid(True, alpha=0.4)
# zoom on the von Kármán limit cycle
i0z = int(0.40/DT); i1z = int(0.46/DT)
axes[1].plot(res_vk['t'][i0z:i1z], res_vk['y'][i0z:i1z]*1e6,
             color=COL['vk'], lw=1.2)
axes[1].set_ylabel('$y_p$ (µm)', fontsize=11)
axes[1].set_xlabel('time (s)', fontsize=11)
axes[1].set_title('von Kármán steady limit cycle (zoom)', fontsize=11)
axes[1].grid(True, alpha=0.4)
axes[1].set_xlim(0, T_END)
plt.tight_layout(); plt.savefig(f"{OUT}/fig02_time_traces.png", dpi=170,
                                bbox_inches='tight'); plt.close()
print("  ✓ fig02_time_traces.png")

# Fig 3 : hardening backbone ω_nl/ω_l vs A/h (validated quantity)
fig, ax = plt.subplots(figsize=(9, 5.4))
ar = np.linspace(0, 2.0, 41)
rom_wall = VonKarmanROM(build_plate(ap_demo), membrane_bc="wall",
                        verbose=False)
rom_cant = VonKarmanROM(build_plate(ap_demo), membrane_bc="cantilever",
                        verbose=False)
_, rw, _ = rom_wall.backbone_single_mode(0, ar)
_, rc, _ = rom_cant.backbone_single_mode(0, ar)
ax.plot(ar, rw, color=COL['vk'], lw=2.2,
        label='wall (base + sides in-plane restrained)')
ax.plot(ar, rc, color=COL['ol'], lw=2.2,
        label='cantilever (3 free edges)')
ax.axhline(1.0, color='k', lw=0.8, alpha=0.5)
# operating limit-cycle amplitude marker
A_h = ctrl_metrics['open_loop']['von_karman']['amp_um']*1e-6/BP
if A_h <= ar[-1]:
    _, r_op, _ = rom_wall.backbone_single_mode(0, [A_h])
    ax.plot(A_h, r_op[0], '*', color='gold', ms=20, markeredgecolor='k',
            zorder=5, label='operating limit cycle (A/h=%.2f)' % A_h)
# literature reference for a fully clamped square plate (immovable)
ax.plot(1.0, 1.17, 'D', color='green', ms=9, markeredgecolor='k',
        label='clamped-plate lit. ref. (≈1.17 @ A/h=1)')
ax.set_xlabel('vibration amplitude / wall thickness $A/h$', fontsize=12)
ax.set_ylabel(r'frequency ratio $\omega_{nl}/\omega_{l}$', fontsize=12)
ax.set_title('Amplitude-dependent stiffening (von Kármán backbone, mode 1)\n'
             'validated model — hardening depends strongly on the in-plane BC',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9.5, loc='upper left'); ax.grid(True, alpha=0.4)
plt.tight_layout(); plt.savefig(f"{OUT}/fig03_backbone.png", dpi=170,
                                bbox_inches='tight'); plt.close()
print("  ✓ fig03_backbone.png")

# Fig 4 : active control (3 panels)
fig, axes = plt.subplots(3, 1, figsize=(12, 9.5), sharex=True)
ilc = res_lin_ctrl['stop_idx']; ivc = res_vk_ctrl['stop_idx']
# (a) full-range, symlog : the LINEAR model predicts divergence
axes[0].plot(res_lin_ctrl['t'][:ilc+1], res_lin_ctrl['y'][:ilc+1]*1e6,
             color=COL['lin'], lw=0.8, ls='--',
             label='LQG on LINEAR plant → design predicts DIVERGENCE')
axes[0].plot(res_vk_ctrl['t'][:ivc+1], res_vk_ctrl['y'][:ivc+1]*1e6,
             color=COL['vk'], lw=0.8,
             label='LQG on von Kármán plant → true response BOUNDED')
axes[0].set_yscale('symlog', linthresh=100)
axes[0].set_ylabel('$y_p$ (µm, symlog)', fontsize=11)
axes[0].set_title('ap = %.2f mm : the SAME LQG (±%.0f V, designed on the '
                  'linear model) on both plants' % (ap_demo*1e3, U_SAT),
                  fontsize=11, fontweight='bold')
axes[0].legend(fontsize=9, loc='upper right'); axes[0].grid(True, alpha=0.4)
# (b) von Kármán plant only — uncontrolled vs controlled (readable scale)
axes[1].plot(res_vk['t'][:iv+1], res_vk['y'][:iv+1]*1e6, color=COL['ol'],
             lw=0.6, alpha=0.8,
             label='uncontrolled limit cycle (A=%.0f µm)'
             % ctrl_metrics['open_loop']['von_karman']['amp_um'])
axes[1].plot(res_vk_ctrl['t'][:ivc+1], res_vk_ctrl['y'][:ivc+1]*1e6,
             color=COL['vk'], lw=0.7,
             label='LQG-controlled (A=%.0f µm)'
             % ctrl_metrics['lqg']['von_karman_plant']['amp_um'])
axes[1].set_ylabel('$y_p$ (µm) — vK plant', fontsize=11)
axes[1].legend(fontsize=9, loc='upper right'); axes[1].grid(True, alpha=0.4)
# (c) control voltage
axes[2].plot(res_vk_ctrl['t'][:ivc+1], res_vk_ctrl['u'][:ivc+1],
             color=COL['vk'], lw=0.6, label='u (von Kármán plant)')
axes[2].axhline(U_SAT, color='r', ls=':', alpha=0.5)
axes[2].axhline(-U_SAT, color='r', ls=':', alpha=0.5)
axes[2].set_ylabel('u (V)', fontsize=11); axes[2].set_xlabel('time (s)',
                                                             fontsize=11)
axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.4)
plt.tight_layout(); plt.savefig(f"{OUT}/fig04_active_control.png", dpi=170,
                                bbox_inches='tight'); plt.close()
print("  ✓ fig04_active_control.png")


# ============================================================
# save metrics
# ============================================================
metrics = dict(
    wall_h_mm=BP*1e3, membrane_bc=BC, rpm=RPM, freqs_hz=p0.freq_n.tolist(),
    backbone_ratio={"A/h=0.5": float(ratio_bb[0]),
                    "A/h=1.0": float(ratio_bb[1]),
                    "A/h=2.0": float(ratio_bb[2])},
    ap_crit_linear_mm=ap_crit,
    bifurcation=bif,
    control=ctrl_metrics,
    config_comparison=config_cmp,
)
with open(f"{OUT}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\n  ✓ metrics.json")
print(f"\nTotal wall time : {time.time()-t_glob:.1f} s")
