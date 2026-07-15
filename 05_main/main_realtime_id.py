"""
main_realtime_id.py
===================
Real-time system identification along the finishing sequence — the P6 driver.

Ties together the three P6 pieces:
  1. 01_core/material_removal.MillingWorkpiece  — physically accurate, per-element
     thickness-field FEM; a finishing sequence of descending passes thins the wall
     non-uniformly (modes reshape, frequencies drift non-uniformly, reaching the
     article's measured 9-17 %).
  2. 03_analysis/realtime_id.transit_probe_identify — active piezo probe at each
     (non-cutting) pass boundary identifies the modal frequencies UNBIASED (the
     documented finding: passive innovation-based ID is biased under periodic
     cutting; persistent excitation is required).
  3. 02_controllers/adaptive_id.IDScheduledController — re-tunes the controller's
     internal model to the identified frequencies for the upcoming pass.

Headline comparison over the sequence (each pass = a cutting run on that pass's
true snapshot plant):
  * LQG (fixed, pristine)      — high-performance regulator, DIVERGES once the wall
                                 thins past its ~-9 % margin;
  * LQG (ID-scheduled)         — tracks the softening, matches the oracle;
  * LQG (oracle, true freqs)   — upper bound with perfect frequency knowledge;
  * ESO-ADRC (fixed)           — architecturally robust, survives WITHOUT ID.
Plus the ID tracking figure (identified vs true frequency) and the passive-ID bias.

Output: figs_realtime_id/*.png|pdf + console summary.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from material_removal import MillingWorkpiece, finishing_sequence
from realtime_id import transit_probe_identify, passive_frequency_estimate
from milling_force import precompute_alpha_periodic, cutting_constants
from newmark_solver import NewmarkSimulator
from lqg_controller import LQGController
from adrc_controller import ESO_ADRC_Controller
from adaptive_id import IDScheduledController, _design_view

# ---------------- physical parameters (article) ----------------
LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU = 2830, 69e9, 0.33
D31, H_PA, E_PE, NU_PE = 175e-12, 0.7e-3, 63e9, 0.35
NT, RPM, AP = 3, 4900, 0.3e-3
AE = 0.1e-3                      # radial bite per finishing layer
ZETA = [0.0031, 0.0017, 0.0027, 0.0056, 0.0035]
DT, TAU, FT = 5e-5, 60 / (NT * RPM), 0.02e-3
F_TOOTH = NT * RPM / 60.0
W_TOOTH = 2 * np.pi * F_TOOTH
KT, KN, MUC = 925e6, 0.26, 0.20
ETA, GAM = np.deg2rad(35), np.deg2rad(15)
K1, K2 = cutting_constants(KN, MUC, ETA, GAM)
RT = 0.005
PHI_ST = np.pi - np.arccos(1 - AE / RT)
PHI_EX = np.pi
OMEGA = 2 * np.pi * RPM / 60
N_PER = int(round(TAU / DT))
KV, MNS = 1e-12, 1e-8
N_LAYERS = 6
BAND_H = 0.020                  # engaged-height per pass (coarse bands: 4 per layer)
T_CUT = 0.4

OUT_DIR = "figs_realtime_id"
os.makedirs(OUT_DIR, exist_ok=True)


def run_cut(pl, ctrl, T=T_CUT):
    sim = NewmarkSimulator(pl, dt=DT, T_end=T, ft=FT, tau=TAU, verbose=False)
    v = FT * NT * RPM / 60
    xp = np.minimum(v * sim.t_vec, LP)
    kp = np.clip(np.round(xp / LP * 2000).astype(int), 0, 2000)
    a3, a4 = precompute_alpha_periodic(DT, N_PER, sim.nstep, OMEGA, NT, RT, ETA,
                                       PHI_ST, PHI_EX, HP - AP, HP, K1, K2, KT)
    r = sim.simulate(a3, a4, kp, controller=ctrl, meas_noise_std=MNS,
                     rng=np.random.default_rng(1234), progress=False)
    i = r['stop_idx']
    y = r['y'][:i + 1]
    return dict(rms=np.sqrt(np.mean(y**2)) * 1e6,
                diverged=i < sim.nstep - 2, y=r['y'], stop=i)


# ============================================================
# 1. Finishing sequence (physically accurate material removal)
# ============================================================
print("=" * 72)
print(" REAL-TIME IDENTIFICATION over a finishing sequence (P6) ")
print("=" * 72)
t0 = time.time()
wp = MillingWorkpiece(LP, HP, BP, RHO, E_AL, NU, N1=30, N2=24,
                      n_modes=5, zeta_modes=ZETA, verbose=False)
wp.set_observation(x_obs=LP, z_obs=HP)
wp.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
print(f"[seq] {N_LAYERS} finishing layers, a_e = {AE*1e3:.2f} mm/layer, "
      f"engaged band {BAND_H*1e3:.0f} mm ...")
snaps = finishing_sequence(wp, n_layers=N_LAYERS, a_e=AE, a_p=BAND_H, verbose=False)
p0 = snaps[0]['plant']
f_pristine = np.array([521.1, 1069.9, 2733.0])
print(f"[seq] {len(snaps)} passes; final drift "
      f"{(snaps[-1]['freq'][:3]/f_pristine-1)*100} % ({time.time()-t0:.1f}s)")

# fixed (pristine-designed) controllers
lqg_fixed = LQGController(_design_view(p0.freq_n[:3], ZETA[:3],
                                       p0.D_obs[:3], p0.H_Pe_modal[:3]),
                          dt=DT, verbose=False, kalman_V=KV, u_max=150.0)
lqg_fixed.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                           w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
lqg_fixed.discretize_observer()
adrc_fixed = ESO_ADRC_Controller(_design_view(p0.freq_n[:3], ZETA[:3],
                                              p0.D_obs[:3], p0.H_Pe_modal[:3]),
                                 DT, 1e14, 1e8, 1e4, kalman_V=KV, verbose=False)

# ID-scheduled LQG (re-tuned per pass)
lqg_id = IDScheduledController('lqg', DT, p0.D_obs[:3], p0.H_Pe_modal[:3],
                               ZETA[:3], p0.freq_n[:3], w_tooth=W_TOOTH,
                               kalman_V=KV)

# ============================================================
# 2. March along the sequence: probe -> retune -> cut
# ============================================================
print("\n[run] probe-identify + cut, each pass:")
rec = dict(pass_id=[], removed=[], f_true=[], f_id=[], f_passive=[],
           rms_fix=[], div_fix=[], rms_id=[], div_id=[], rms_orc=[], div_orc=[],
           rms_adrc=[], div_adrc=[])
prior = np.array([521.0, 1070.0])
for s in snaps:
    pl = s['plant']
    ftrue = s['freq'][:3]
    # --- transit probe (unbiased) ---
    fhat = transit_probe_identify(pl, DT, FT, TAU, amp=3.0, expect_freqs=prior,
                                  n_modes_id=2, seed=11)
    prior = fhat.copy()
    # --- ID-scheduled controller for this pass ---
    lqg_id.retune(fhat)
    lqg_orc = LQGController(_design_view(ftrue[:3], ZETA[:3],
                                         p0.D_obs[:3], p0.H_Pe_modal[:3]),
                            dt=DT, verbose=False, kalman_V=KV, u_max=150.0)
    lqg_orc.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                             w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
    lqg_orc.discretize_observer()
    # --- cut with each controller ---
    rf = run_cut(pl, lqg_fixed)
    ri = run_cut(pl, lqg_id)
    ro = run_cut(pl, lqg_orc)
    ra = run_cut(pl, adrc_fixed)
    rec['pass_id'].append(s['pass_id'])
    rec['removed'].append(s['removed_frac'] * 100)
    rec['f_true'].append(ftrue[:2])
    rec['f_id'].append(fhat)
    rec['rms_fix'].append(rf['rms']); rec['div_fix'].append(rf['diverged'])
    rec['rms_id'].append(ri['rms']); rec['div_id'].append(ri['diverged'])
    rec['rms_orc'].append(ro['rms']); rec['div_orc'].append(ro['diverged'])
    rec['rms_adrc'].append(ra['rms']); rec['div_adrc'].append(ra['diverged'])
    if s['pass_id'] % 3 == 0 or s['pass_id'] == len(snaps) - 1:
        fix_str = 'DIV' if rf['diverged'] else f"{rf['rms']:.3f}"
        print(f"  pass {s['pass_id']:2d} rem {s['removed_frac']*100:5.1f}%  "
              f"id f2={fhat[1]:6.1f} (true {ftrue[1]:6.1f})  "
              f"LQG-fix={fix_str:>7}  LQG-ID={ri['rms']:.3f}  ADRC={ra['rms']:.3f}")

# passive-ID bias on one mid-sequence cutting run (documented negative result)
mid = len(snaps) // 2
r_mid = run_cut(snaps[mid]['plant'], lqg_id)
f_passive = passive_frequency_estimate(r_mid['y'][:r_mid['stop'] + 1], DT,
                                       n_modes_id=2)

# ============================================================
# 3. Figures
# ============================================================
pid = np.array(rec['pass_id'])
rem = np.array(rec['removed'])
f_true = np.array(rec['f_true'])
f_id = np.array(rec['f_id'])
COL = dict(fix='#2E8B57', idc='#DC143C', orc='#E8A020', adrc='#1E5AA8')


def _plot_rms(ax, key, div, label, color, ls='-'):
    v = np.array(rec[key], float)
    d = np.array(div)
    vp = v.copy()
    vp[d] = np.nan
    ax.plot(rem, vp, ls, color=color, lw=2, marker='o', ms=4, label=label)
    if d.any():                    # mark divergences at the plot ceiling
        ax.plot(rem[d], np.full(d.sum(), ax_ceiling), 'x', color=color, ms=9,
                mew=2.5)


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) ID tracking
ax = axes[0]
ax.plot(rem, f_true[:, 0], '-', color='k', lw=2, label='true mode 1')
ax.plot(rem, f_id[:, 0], '--', color=COL['idc'], lw=1.8, marker='.',
        label='identified mode 1')
ax.plot(rem, f_true[:, 1], '-', color='#444', lw=2, label='true mode 2')
ax.plot(rem, f_id[:, 1], '--', color=COL['orc'], lw=1.8, marker='.',
        label='identified mode 2')
for h in (2, 3, 4):
    ax.axhline(h * F_TOOTH, color='gray', ls=':', alpha=0.6, lw=0.9)
    ax.text(rem[-1], h * F_TOOTH + 6, f'{h}·f_tooth', fontsize=8, color='gray',
            ha='right')
ax.plot([], [], 's', color='purple', label=f'passive-ID (biased): {f_passive[0]:.0f} Hz')
ax.set_xlabel("material removed (%)", fontsize=12)
ax.set_ylabel("modal frequency (Hz)", fontsize=12)
ax.set_title("Active-probe real-time identification tracks the drifting modes\n"
             "(passive ID would return the tooth harmonics — biased)",
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='center left')
ax.grid(True, alpha=0.4)

# (b) RMS over the sequence
ax = axes[1]
ax_ceiling = 20.0
_plot_rms(ax, 'rms_fix', rec['div_fix'], 'LQG fixed (pristine)', COL['fix'])
_plot_rms(ax, 'rms_id', rec['div_id'], 'LQG ID-scheduled', COL['idc'])
_plot_rms(ax, 'rms_orc', rec['div_orc'], 'LQG oracle (true freq)', COL['orc'], '--')
_plot_rms(ax, 'rms_adrc', rec['div_adrc'], 'ESO-ADRC fixed (robust)', COL['adrc'])
ax.set_yscale('log')
ax.set_xlabel("material removed (%)", fontsize=12)
ax.set_ylabel("y_RMS over the pass (µm, log)", fontsize=12)
ax.set_title("Fixed LQG loses control past its margin (RMS -> ~10 um); real-time\n"
             "ID keeps it on the oracle. ESO-ADRC is robust without ID.",
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.4)

plt.suptitle("Real-time identification over an accurate material-removal finishing "
             "sequence", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_realtime_id.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig_realtime_id.pdf", bbox_inches='tight')
plt.close()
print(f"\n  ✓ {OUT_DIR}/fig_realtime_id.png|pdf")

# ============================================================
# 4. Summary
# ============================================================
id_err = np.abs(f_id / f_true - 1) * 100
print(f"\n=== SUMMARY ===")
print(f"  finishing sequence: {len(snaps)} passes, final wall "
      f"{snaps[-1]['h_min']*1e3:.2f} mm, final drift "
      f"{(snaps[-1]['freq'][1]/f_pristine[1]-1)*100:+.1f}% (mode 2)")
print(f"  probe ID error over the sequence: mode1 max {id_err[:,0].max():.2f}%, "
      f"mode2 max {id_err[:,1].max():.2f}% (mean {id_err.mean():.2f}%)")
print(f"  passive ID at mid-sequence: {f_passive[0]:.0f} Hz "
      f"(true modes {f_true[mid].round(0)}; tooth harmonics "
      f"{[round(h*F_TOOTH) for h in (1,2,3,4,5)]}) — BIASED, returns a harmonic")
# "control lost" = RMS above a 1 um usability ceiling (all stay below the 5 mm
# hard stop, so this is graceful catastrophic degradation, not formal divergence)
lost = lambda k: int(np.sum(np.array(rec[k]) > 1.0))
print(f"  passes with RMS > 1 um (control lost) over {len(snaps)}: "
      f"LQG-fixed {lost('rms_fix')}, LQG-ID {lost('rms_id')}, "
      f"ESO-ADRC {lost('rms_adrc')}")
worst_fix = max(rec['rms_fix']); worst_id = max(rec['rms_id'])
print(f"  worst-pass RMS: LQG-fixed {worst_fix:.2f} um, LQG-ID {worst_id:.3f} um "
      f"(ID is {worst_fix/worst_id:.0f}x better at the worst pass)")
fin = -1
fix_fin = 'DIV' if rec['div_fix'][fin] else f"{rec['rms_fix'][fin]:.3f}um"
print(f"  final pass ({rem[fin]:.1f}% removed): "
      f"LQG-fixed {fix_fin}, LQG-ID {rec['rms_id'][fin]:.3f}um, "
      f"ESO-ADRC {rec['rms_adrc'][fin]:.3f}um")
print(f"\nTotal time {time.time()-t0:.1f}s  ->  {OUT_DIR}/")
