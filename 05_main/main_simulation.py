"""
main_simulation.py
==================
LQG  vs  ESO-ADRC  vs  A-ESO-ADRC  (modal extended-state-observer ADRC and its
adaptive, cost-supervised development — see 02_controllers/adrc_controller.py).

Fair-comparison protocol (see docs/AUDIT_FINDINGS.md, docs/CONTRIBUTION.md):
   * SYMMETRIC feedback design: both controllers use the SAME output-weighted LQR
     construction and both are tuned by grid search. The LQG grid selects its
     weights on the design model; the ESO-ADRC grid selects (w_q, w_qd, sigma_d).
   * TWO selection criteria are reported for ESO-ADRC, both computed at design time
     with NO access to the held-out scenarios:
       - PERFORMANCE design  : lowest RMS on the nominal scenario (short sim);
       - CERTIFIED design    : lowest worst-case Floquet radius (rigorous coupled
         closed-loop monodromy) over a design-time uncertainty ball
         (frequency mismatch x depth x tool position) — this one is the fixed
         "ESO-ADRC" entry in every table.
   * A-ESO-ADRC = the two designs as rungs of a supervised ladder (shared physical
     observer state; cost-driven switching with panic fallback to the certified
     rung). It is evaluated UNCHANGED on every scenario.
   * Everything runs on the 5-mode PLANT with 3-mode controllers (spillover),
     10 nm measurement noise, identical +/-150 V clipping, corrected Eq. (3)
     cutting constants, Eq. (15) piezo coupling.

Outputs: figs_lqg_vs_adrc/*.png|pdf + console summary.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from plate_model import PlateModel
from milling_force import precompute_alpha_periodic, cutting_constants
from lqg_controller import LQGController
from adrc_controller import (ESO_ADRC_Controller, AdaptiveESO_ADRC_Controller,
                             modal_matrices)
from newmark_solver import NewmarkSimulator
from fdm_stability import (compute_closed_loop_SLD, compute_closed_loop_SLD_generic,
                           closed_loop_rho, closed_loop_rho_generic)


# ============================================================
# Paramètres physiques (article)
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
N1, N2 = 30, 24

# --- Inverse-crime avoidance (P1): 5-mode plant, 3-mode controllers -----------
N_MODES_PLANT = 5
N_MODES_CTRL = 3
N_MODES = N_MODES_CTRL
ZETA_PLANT = [0.0031, 0.0017, 0.0027, 0.0056, 0.0035]
ZETA = ZETA_PLANT[:N_MODES_CTRL]

MEAS_NOISE_STD = 1e-8           # 10 nm RMS on the displacement measurement
KALMAN_V = 1e-12                # conservative/robust observer assumption

DT = 5e-5
T_END = 0.5

OUT_DIR = "figs_lqg_vs_adrc"
os.makedirs(OUT_DIR, exist_ok=True)

LBL_LQG = "LQG"
LBL_ADRC = "ESO-ADRC"
LBL_AADRC = "A-ESO-ADRC"
COL = {LBL_LQG: '#2E8B57', LBL_ADRC: '#1E5AA8', LBL_AADRC: '#DC143C'}


# ============================================================
# Helper functions
# ============================================================
def build_plant(zp_pos, freq_perturb=0.0):
    """Full-order (N_MODES_PLANT) plant; drift applied via perturbed_copy so mode
    signs stay consistent."""
    plate = PlateModel(LP, HP, BP, RHO, E_AL, NU_AL,
                       N1=N1, N2=N2, n_modes=N_MODES_PLANT,
                       zeta_modes=ZETA_PLANT, verbose=False)
    plate.precompute_Dp(zp_pos=zp_pos, n_pos=2001)
    plate.set_observation(x_obs=LP, z_obs=HP)
    plate.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    if abs(freq_perturb) > 1e-6:
        plate = plate.perturbed_copy(freq_perturb)
    return plate


def build_plate(zp_pos, freq_perturb=0.0):
    """Reduced (N_MODES_CTRL) plate — the design model, also used for the SLD."""
    plate = PlateModel(LP, HP, BP, RHO, E_AL, NU_AL,
                       N1=N1, N2=N2, n_modes=N_MODES_CTRL,
                       zeta_modes=ZETA, verbose=False)
    plate.precompute_Dp(zp_pos=zp_pos, n_pos=2001)
    plate.set_observation(x_obs=LP, z_obs=HP)
    plate.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    if abs(freq_perturb) > 1e-6:
        plate = plate.perturbed_copy(freq_perturb)
    return plate


def fft_amp(y, dt, i_start_t=0.05):
    i_start = int(i_start_t/dt)
    if len(y) <= i_start:
        return None, None
    y_seg = y[i_start:]
    N = len(y_seg)
    win = np.hanning(N)
    NFFT = 2**int(np.ceil(np.log2(max(N, 2))))
    Y = np.fft.fft(y_seg*win, NFFT) / N
    f = (1/dt)/2 * np.linspace(0, 1, NFFT//2 + 1)
    return f, 2*np.abs(Y[:NFFT//2 + 1]) * 1e6


# ============================================================
# Cutting geometry / force constants
# ============================================================
print("="*72)
print(" COMPARISON : LQG vs ESO-ADRC vs A-ESO-ADRC (train-once / held-out) ")
print("="*72)
t_global = time.time()

PHI_ST = np.pi - np.arccos(1 - AE/(DT_TOOL/2))
PHI_EX = np.pi
OMEGA = 2*np.pi*RPM/60
RT = DT_TOOL/2
TAU = 60/(NT*RPM)
K1, K2 = cutting_constants(KN, MU_C, ETA_H, GAMMA_N)
F_T = NT * RPM / 60
N_PER = int(np.round(TAU/DT))


def make_forces(sim, ap, KT_actual):
    """Precompute (alpha3, alpha4, kp_idx) for a scenario on a given simulator."""
    v_feed = FT * NT * RPM / 60
    xp_path = np.minimum(v_feed * sim.t_vec, LP)
    kp_idx = np.clip(np.round(xp_path/LP * 2000).astype(int), 0, 2000)
    alpha3, alpha4 = precompute_alpha_periodic(DT, N_PER, sim.nstep,
        OMEGA, NT, RT, ETA_H, PHI_ST, PHI_EX, HP-ap, HP, K1, K2, KT_actual)
    return alpha3, alpha4, kp_idx


# NOTE (P1): with the corrected (stronger) Eq.(3) forces, the LQG-controlled
# stability margin at ap = 0.3 mm / 4900 RPM is ~-9 % frequency mismatch. S3 uses
# -8 % (within the LQG margin, matching the article's ~9 % measured 2nd-mode drift).
# Drift BEYOND the margin (-12 %) is exercised in main_adaptive_removal.py.
scenarios_def = [
    ("S1 - Nominal article",      0.3e-3, KT_NOMINAL, 0.0),
    ("S2 - Aggressive ap=0.6mm",  0.6e-3, KT_NOMINAL, 0.0),
    ("S3 - Uncertainty ω-8%",     0.3e-3, KT_NOMINAL, -0.08),
    ("S4 - High K_T +30%",        0.3e-3, 1.3*KT_NOMINAL, 0.0),
]

# ------------------------------------------------------------------
# DESIGN (nominal model only — every scenario below is held-out)
# ------------------------------------------------------------------
print("\n[design] Building nominal full-order plant; designing controllers on "
      f"{N_MODES_CTRL} modes (plant has {N_MODES_PLANT}: spillover).")
PLANT_NOMINAL = build_plant(zp_pos=HP - 0.3e-3/2, freq_perturb=0.0)
design_view = PLANT_NOMINAL.truncated_view(N_MODES_CTRL)

sim_design = NewmarkSimulator(PLANT_NOMINAL, dt=DT, T_end=0.25,
                              ft=FT, tau=TAU, verbose=False)
alpha3_nom, alpha4_nom, kp_idx_nom = make_forces(sim_design, 0.3e-3, KT_NOMINAL)

# --- LQG baseline: grid-searched weights ---------------------------------------
LQG_SHARED = LQGController(design_view, dt=DT, verbose=False,
                           kalman_V=KALMAN_V, u_max=150.0)
LQG_SHARED.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                            w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
LQG_SHARED.discretize_observer()
print(f"[design] LQG weights: w_q={LQG_SHARED.w_q:.1e}, w_qd={LQG_SHARED.w_qd:.1e}")

# --- ESO-ADRC grid: performance + certification criteria ------------------------
print("[design] ESO-ADRC grid search (w_q x w_qd x sigma_d): nominal RMS + "
      "worst-case monodromy over the design uncertainty ball...")
GRID_WQ = [1e12, 1e14, 1e16]
GRID_WQD = [1e6, 1e8]
GRID_SD = [1e3, 3e3, 1e4]

# Design-time uncertainty ball for certification (NO held-out time simulations:
# only the design/plant MODEL enters the Floquet computation). The ball combines
# (a) the frequency-mismatch range at the nominal depth (model UNCERTAINTY) and
# (b) the doubled depth at the nominal frequency (operating ENVELOPE) — the corner
# "wrong frequency AND doubled depth" is beyond every fixed design and would
# degenerate the argmin.
CERT_MISMATCH = [-0.12, -0.08, 0.0, 0.08, 0.15]
CERT_KP = [0, 1000]                 # tool at x = 0 and x = L/2
CERT_POINTS = [(fp, 0.3e-3) for fp in CERT_MISMATCH] + [(0.0, 0.6e-3)]
CERT_PLANTS = {fp: (PLANT_NOMINAL if fp == 0.0
                    else PLANT_NOMINAL.perturbed_copy(fp))
               for fp in CERT_MISMATCH}


def nocut_stable(ctrl):
    """Linear (no cutting, no delay) closed loop vs the 5-mode plant."""
    A_con, B_con_y, K_con = ctrl.controller_realization()
    npl = N_MODES_PLANT
    m = A_con.shape[0]
    Kp = np.diag(PLANT_NOMINAL.omega_n**2)
    Cp = np.diag(2*np.array(ZETA_PLANT)*PLANT_NOMINAL.omega_n)
    A = np.zeros((2*npl+m, 2*npl+m))
    A[0:npl, npl:2*npl] = np.eye(npl)
    A[npl:2*npl, 0:npl] = -Kp
    A[npl:2*npl, npl:2*npl] = -Cp
    A[npl:2*npl, 2*npl:] = -np.outer(PLANT_NOMINAL.H_Pe_modal, K_con)
    A[2*npl:, 0:npl] = np.outer(B_con_y, PLANT_NOMINAL.D_obs)
    A[2*npl:, 2*npl:] = A_con
    return np.max(np.linalg.eigvals(A).real) < 0


def cert_worst_rho(ctrl, m_div=20):
    """Worst-case coupled closed-loop Floquet radius over the design ball."""
    A_con, B_con_y, K_con = ctrl.controller_realization()
    worst = 0.0
    for fp, ap_c in CERT_POINTS:
        pl = CERT_PLANTS[fp]
        for kp in CERT_KP:
            Dp5 = pl.get_Dp_at(kp)[0]
            r = closed_loop_rho_generic(
                pl.omega_n, np.array(ZETA_PLANT), Dp5,
                pl.H_Pe_modal, pl.D_obs, A_con, B_con_y, K_con,
                RPM, NT, RT, ETA_H, PHI_ST, PHI_EX, ap_c, HP,
                K1, K2, KT_NOMINAL, m_div=m_div)
            worst = max(worst, r)
    return worst


t_grid = time.time()
grid_rows = []
for w_q in GRID_WQ:
    for w_qd in GRID_WQD:
        for sd in GRID_SD:
            try:
                c = ESO_ADRC_Controller(design_view, DT, w_q, w_qd, sd,
                                        kalman_V=KALMAN_V, verbose=False)
            except Exception:
                continue
            if not nocut_stable(c):
                continue
            r = sim_design.simulate(alpha3_nom, alpha4_nom, kp_idx_nom,
                                    controller=c, meas_noise_std=MEAS_NOISE_STD,
                                    rng=np.random.default_rng(1234), progress=False)
            i = r['stop_idx']
            rms_nom = np.sqrt(np.mean(r['y'][:i+1]**2))*1e6
            if i < sim_design.nstep - 2:
                rms_nom = np.inf
            rho_w = cert_worst_rho(c)
            grid_rows.append(dict(w_q=w_q, w_qd=w_qd, sd=sd,
                                  rms=rms_nom, rho=rho_w))

perf_row = min(grid_rows, key=lambda d: d['rms'])
# Certified design: among designs within 5 % of the smallest worst-case radius,
# take the one with the best nominal RMS (robust argmin, grid-noise tolerant).
rho_min = min(d['rho'] for d in grid_rows)
cert_row = min((d for d in grid_rows if d['rho'] <= 1.05 * rho_min),
               key=lambda d: d['rms'])
print(f"[design] grid done in {time.time()-t_grid:.0f}s "
      f"({len(grid_rows)} feasible designs)")
print(f"[design] PERFORMANCE design: (w_q={perf_row['w_q']:.0e}, "
      f"w_qd={perf_row['w_qd']:.0e}, sigma_d={perf_row['sd']:.0e}) "
      f"rms_nom={perf_row['rms']:.4f}um, worst-rho={perf_row['rho']:.3f}")
print(f"[design] CERTIFIED design  : (w_q={cert_row['w_q']:.0e}, "
      f"w_qd={cert_row['w_qd']:.0e}, sigma_d={cert_row['sd']:.0e}) "
      f"rms_nom={cert_row['rms']:.4f}um, worst-rho={cert_row['rho']:.3f}")

# Fixed ESO-ADRC = the CERTIFIED design (robust selection, no held-out peeking).
ADRC_FIXED = ESO_ADRC_Controller(design_view, DT, cert_row['w_q'],
                                 cert_row['w_qd'], cert_row['sd'],
                                 kalman_V=KALMAN_V, verbose=False)
# A-ESO-ADRC ladder = [performance rung, certified rung].
RUNGS = ((perf_row['w_q'], perf_row['w_qd'], perf_row['sd']),
         (cert_row['w_q'], cert_row['w_qd'], cert_row['sd']))
AADRC = AdaptiveESO_ADRC_Controller(design_view, DT, rungs=RUNGS,
                                    perf_rung=0, robust_rung=1,
                                    kalman_V=KALMAN_V, verbose=False)
print(f"[design] Plant modes (Hz): {np.round(PLANT_NOMINAL.freq_n, 0)}; "
      f"controllers see first {N_MODES_CTRL}. Meas. noise = "
      f"{MEAS_NOISE_STD*1e9:.0f} nm.")


def build_scenario_plant(ap, freq_perturb):
    """Perturbed full-order plant for a scenario, reusing the nominal eigensolve."""
    plant = PLANT_NOMINAL.perturbed_copy(freq_perturb)
    zp = HP - ap/2
    if abs(zp - PLANT_NOMINAL.zp_pos) > 1e-9:   # different tool height -> new Dp
        plant.precompute_Dp(zp_pos=zp, n_pos=2001)
    return plant


CONTROLLERS = [(LBL_LQG, LQG_SHARED), (LBL_ADRC, ADRC_FIXED), (LBL_AADRC, AADRC)]


def run_scenario(name, ap, KT_actual, freq_perturb):
    print(f"\n{'='*72}")
    print(f" {name}")
    print(f"  ap = {ap*1e3:.2f}mm,  KT = {KT_actual/1e6:.0f}MPa,  "
          f"freq_drift = {freq_perturb*100:+.0f}%")
    print(f"{'='*72}")

    plate_r = build_scenario_plant(ap, freq_perturb)
    sim = NewmarkSimulator(plate_r, dt=DT, T_end=T_END, ft=FT, tau=TAU, verbose=False)
    alpha3, alpha4, kp_idx = make_forces(sim, ap, KT_actual)

    metrics = {}
    for lbl, ctrl in CONTROLLERS:
        if hasattr(ctrl, 'reset_adaptation'):
            ctrl.reset_adaptation()
        res = sim.simulate(alpha3, alpha4, kp_idx, controller=ctrl,
                           meas_noise_std=MEAS_NOISE_STD,
                           rng=np.random.default_rng(1234), progress=False)
        i_end = res['stop_idx']
        y_clip = res['y'][:i_end+1]
        u_clip = res['u'][:i_end+1]
        du = np.diff(u_clip)
        metrics[lbl] = {
            'res': res,
            'diverged': i_end < sim.nstep - 2,
            'y_max':   np.max(np.abs(y_clip)) * 1e6,
            'y_rms':   np.sqrt(np.mean(y_clip**2)) * 1e6,
            'y_p2p':   (np.max(y_clip) - np.min(y_clip)) * 1e6,
            'u_max':   np.max(np.abs(u_clip)),
            'u_rms':   np.sqrt(np.mean(u_clip**2)),
            'u_energy': np.sum(u_clip**2) * DT,
            'du_max':  np.max(np.abs(du)),
            'du_rms':  np.sqrt(np.mean(du**2)),
        }
        if lbl == LBL_AADRC and len(AADRC.history_rung):
            hr = np.array(AADRC.history_rung)
            metrics[lbl]['n_switch'] = int(np.sum(np.diff(hr) != 0))
            metrics[lbl]['frac_robust'] = float(np.mean(hr == 1))
            metrics[lbl]['rung_hist'] = hr.copy()

    hdr = "".join(f"{lbl:<14}" for lbl, _ in CONTROLLERS)
    print(f"  {'Metric':<22}{hdr}")
    print(f"  {'-'*64}")
    for key in ['y_max', 'y_rms', 'y_p2p', 'u_max', 'u_rms']:
        vals = "".join(f"{metrics[lbl][key]:<14.4f}" for lbl, _ in CONTROLLERS)
        print(f"  {key:<22}{vals}")
    for lbl, _ in CONTROLLERS:
        if metrics[lbl]['diverged']:
            print(f"  !! {lbl} DIVERGED")
    if 'n_switch' in metrics[LBL_AADRC]:
        print(f"  A-ESO-ADRC: {metrics[LBL_AADRC]['n_switch']} rung switches, "
              f"{metrics[LBL_AADRC]['frac_robust']*100:.0f}% of pass on the "
              f"certified rung")

    return {'name': name, 'ap': ap, 'KT': KT_actual, 'freq_pert': freq_perturb,
            'sim': sim, 'metrics': metrics, 'plate_r': plate_r}


scenarios = []
for name, ap, KT_, fp in scenarios_def:
    scenarios.append(run_scenario(name, ap, KT_, fp))


# ============================================================
# FIGURE 1 : summary (y_RMS + gains + u_max)
# ============================================================
names = [s['name'].split(' - ')[0] for s in scenarios]
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
x = np.arange(len(names))
w = 0.26

ax = axes[0]
for j, (lbl, _) in enumerate(CONTROLLERS):
    v = [s['metrics'][lbl]['y_rms'] for s in scenarios]
    bars = ax.bar(x + (j-1)*w, v, w, color=COL[lbl], alpha=0.85,
                  edgecolor='k', label=lbl, linewidth=1.2)
    for bar, vi in zip(bars, v):
        ax.text(bar.get_x()+bar.get_width()/2, vi*1.02, f'{vi:.2f}',
                ha='center', fontsize=8, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
ax.set_ylabel("$y_{RMS}$ (µm)", fontsize=12)
ax.set_title("Vibration RMS", fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, axis='y', alpha=0.5)

ax = axes[1]
for j, lbl in enumerate([LBL_ADRC, LBL_AADRC]):
    g = [(1 - s['metrics'][lbl]['y_rms']/s['metrics'][LBL_LQG]['y_rms'])*100
         for s in scenarios]
    bars = ax.bar(x + (j-0.5)*0.38, g, 0.38, color=COL[lbl], alpha=0.85,
                  edgecolor='k', label=f"{lbl} vs LQG", linewidth=1.2)
    for bar, gi in zip(bars, g):
        ax.text(bar.get_x()+bar.get_width()/2, gi + (1 if gi > 0 else -3),
                f'{gi:+.1f}%', ha='center', fontsize=9, fontweight='bold')
ax.axhline(0, color='k', linewidth=1)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
ax.set_ylabel("Δ y_RMS vs LQG (%)", fontsize=12)
ax.set_title("RMS gain vs LQG (positive = better than LQG)",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, axis='y', alpha=0.5)

ax = axes[2]
for j, (lbl, _) in enumerate(CONTROLLERS):
    v = [s['metrics'][lbl]['u_max'] for s in scenarios]
    ax.bar(x + (j-1)*w, v, w, color=COL[lbl], alpha=0.85,
           edgecolor='k', label=lbl, linewidth=1.2)
ax.axhline(150, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
           label='Sat. piezo')
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
ax.set_ylabel("$|u|_{max}$ (V)", fontsize=12)
ax.set_title("Peak control effort", fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.5)

plt.suptitle("LQG vs ESO-ADRC vs A-ESO-ADRC — held-out scenarios",
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig01_bilan.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig01_bilan.pdf", bbox_inches='tight')
plt.close()
print(f"\n  ✓ fig01_bilan")


# ============================================================
# FIGURE 2 : temporal responses
# ============================================================
fig, axes = plt.subplots(len(scenarios), 1, figsize=(14, 3.0*len(scenarios)))
if len(scenarios) == 1:
    axes = [axes]
for i, s in enumerate(scenarios):
    ax = axes[i]
    t_ms = s['sim'].t_vec * 1e3
    i_end = min(s['metrics'][lbl]['res']['stop_idx'] for lbl, _ in CONTROLLERS)
    for lbl, _ in CONTROLLERS:
        res = s['metrics'][lbl]['res']
        ax.plot(t_ms[:i_end+1], res['y'][:i_end+1]*1e6, color=COL[lbl],
                linewidth=0.5, alpha=0.85,
                label=f"{lbl} (RMS={s['metrics'][lbl]['y_rms']:.3f}µm)")
    ax.set_ylabel("$y_p$ (µm)", fontsize=11)
    ax.set_title(s['name'], fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.5)
    if i == len(scenarios)-1:
        ax.set_xlabel("Temps (ms)", fontsize=11)
plt.suptitle("Réponse temporelle $y_p(t)$ — 4 scénarios held-out",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig02_temporal_y.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig02_temporal_y.pdf", bbox_inches='tight')
plt.close()
print(f"  ✓ fig02_temporal_y")


# ============================================================
# FIGURE 3 : control voltage u(t)
# ============================================================
fig, axes = plt.subplots(len(scenarios), 1, figsize=(14, 3.0*len(scenarios)))
if len(scenarios) == 1:
    axes = [axes]
for i, s in enumerate(scenarios):
    ax = axes[i]
    t_ms = s['sim'].t_vec * 1e3
    i_end = min(s['metrics'][lbl]['res']['stop_idx'] for lbl, _ in CONTROLLERS)
    for lbl, _ in CONTROLLERS:
        res = s['metrics'][lbl]['res']
        ax.plot(t_ms[:i_end+1], res['u'][:i_end+1], color=COL[lbl],
                linewidth=0.5, alpha=0.85,
                label=f"{lbl} (max={s['metrics'][lbl]['u_max']:.1f}V)")
    ax.set_ylabel("u (V)", fontsize=11)
    ax.set_title(s['name'], fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.5)
    if i == len(scenarios)-1:
        ax.set_xlabel("Temps (ms)", fontsize=11)
plt.suptitle("Tension piézoélectrique u(t) — 4 scénarios",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig03_temporal_u.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig03_temporal_u.pdf", bbox_inches='tight')
plt.close()
print(f"  ✓ fig03_temporal_u")


# ============================================================
# FIGURE 4 : FFT of y(t)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
axes = axes.flatten()
for i, s in enumerate(scenarios):
    ax = axes[i]
    for lbl, _ in CONTROLLERS:
        res = s['metrics'][lbl]['res']
        i_end = res['stop_idx']
        f_a, Y_a = fft_amp(res['y'][:i_end+1], DT)
        ax.plot(f_a, Y_a, color=COL[lbl], linewidth=1.3, alpha=0.9,
                label=f"{lbl} (max={Y_a.max():.3f}µm)")
    for n_h in range(1, 5):
        if n_h * F_T <= 1200:
            ax.axvline(n_h * F_T, color='gray', linestyle=':', alpha=0.6,
                       linewidth=0.8)
    for fc in s['plate_r'].freq_n:
        if fc <= 1200:
            ax.axvline(fc, color='blue', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.set_xlim([0, 1200])
    ax.set_xlabel("Fréquence (Hz)", fontsize=10)
    ax.set_ylabel("Amplitude (µm)", fontsize=10)
    ax.set_title(s['name'], fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.5)
plt.suptitle("Spectres de vibration FFT[$y_p$] — 4 scénarios",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig04_fft_y.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig04_fft_y.pdf", bbox_inches='tight')
plt.close()
print(f"  ✓ fig04_fft_y")


# ============================================================
# FIGURE 5 : rung supervision trace (A-ESO-ADRC, all scenarios)
# ============================================================
fig, axes = plt.subplots(len(scenarios), 1, figsize=(14, 2.2*len(scenarios)),
                         sharex=True)
if len(scenarios) == 1:
    axes = [axes]
for i, s in enumerate(scenarios):
    ax = axes[i]
    m = s['metrics'][LBL_AADRC]
    if 'rung_hist' in m:
        t_ms = s['sim'].t_vec[1:len(m['rung_hist'])+1] * 1e3
        ax.step(t_ms, m['rung_hist'], where='post', color=COL[LBL_AADRC],
                linewidth=1.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['perf rung', 'certified rung'])
    ax.set_ylim([-0.2, 1.2])
    ax.set_title(f"{s['name']} — {m.get('n_switch', 0)} switches",
                 fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.5)
axes[-1].set_xlabel("Temps (ms)", fontsize=11)
plt.suptitle("A-ESO-ADRC : supervision de rung (coût mesuré + panic)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig05_rung_supervision.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig05_rung_supervision.pdf", bbox_inches='tight')
plt.close()
print(f"  ✓ fig05_rung_supervision")


# ============================================================
# FIGURE 6 : design-time certification map (rho vs mismatch)
# ============================================================
print(f"\n=== Certification map (Floquet radius vs frequency mismatch) ===")
mismatch_grid = [-0.12, -0.10, -0.08, -0.06, -0.04, 0.0, 0.04, 0.08, 0.12, 0.15]
CERT_PLANTS_FINE = {fp: (PLANT_NOMINAL if fp == 0.0
                         else PLANT_NOMINAL.perturbed_copy(fp))
                    for fp in mismatch_grid}
ADRC_PERF_CTRL = ESO_ADRC_Controller(design_view, DT, perf_row['w_q'],
                                     perf_row['w_qd'], perf_row['sd'],
                                     kalman_V=KALMAN_V, verbose=False)


def rho_curve_generic(ctrl):
    A_con, B_con_y, K_con = ctrl.controller_realization()
    out = []
    for fp in mismatch_grid:
        pl = CERT_PLANTS_FINE[fp]
        r = max(closed_loop_rho_generic(
                    pl.omega_n, np.array(ZETA_PLANT), pl.get_Dp_at(kp)[0],
                    pl.H_Pe_modal, pl.D_obs, A_con, B_con_y, K_con,
                    RPM, NT, RT, ETA_H, PHI_ST, PHI_EX, 0.3e-3, HP,
                    K1, K2, KT_NOMINAL, m_div=25) for kp in CERT_KP)
        out.append(r)
    return np.array(out)


def rho_curve_lqg():
    n3 = N_MODES_CTRL
    D3 = design_view.D_obs
    A_con = LQG_SHARED.A \
        - np.outer(np.asarray(LQG_SHARED.B).reshape(2*n3), LQG_SHARED.K_lqr) \
        - np.outer(np.asarray(LQG_SHARED.L_kal).reshape(2*n3),
                   np.concatenate([D3, np.zeros(n3)]))
    L_v = np.asarray(LQG_SHARED.L_kal).reshape(2*n3)
    K_v = np.asarray(LQG_SHARED.K_lqr).reshape(2*n3)
    out = []
    for fp in mismatch_grid:
        pl = CERT_PLANTS_FINE[fp]
        r = max(closed_loop_rho_generic(
                    pl.omega_n, np.array(ZETA_PLANT), pl.get_Dp_at(kp)[0],
                    pl.H_Pe_modal, pl.D_obs, A_con, L_v, K_v,
                    RPM, NT, RT, ETA_H, PHI_ST, PHI_EX, 0.3e-3, HP,
                    K1, K2, KT_NOMINAL, m_div=25) for kp in CERT_KP)
        out.append(r)
    return np.array(out)


rho_lqg_curve = rho_curve_lqg()
rho_perf_curve = rho_curve_generic(ADRC_PERF_CTRL)
rho_cert_curve = rho_curve_generic(ADRC_FIXED)

fig, ax = plt.subplots(figsize=(11, 6))
mm = np.array(mismatch_grid)*100
ax.plot(mm, rho_lqg_curve, 'o-', color=COL[LBL_LQG], linewidth=2,
        label='LQG')
ax.plot(mm, rho_perf_curve, 's-', color='#E8A020', linewidth=2,
        label=f"ESO-ADRC performance rung "
              f"({perf_row['w_q']:.0e},{perf_row['w_qd']:.0e},{perf_row['sd']:.0e})")
ax.plot(mm, rho_cert_curve, 'd-', color=COL[LBL_ADRC], linewidth=2,
        label=f"ESO-ADRC certified rung "
              f"({cert_row['w_q']:.0e},{cert_row['w_qd']:.0e},{cert_row['sd']:.0e})")
ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5,
           label='Floquet limit ρ = 1')
ax.set_xlabel("Frequency mismatch (%)", fontsize=12)
ax.set_ylabel("max Floquet radius ρ (worst of x = 0, L/2)", fontsize=12)
ax.set_title("Design-time certification: closed-loop monodromy over the mismatch "
             "ball\n(ap = 0.3 mm, 4900 RPM — complementary robustness of the two "
             "rungs motivates the supervised ladder)", fontsize=11,
             fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig06_certification.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig06_certification.pdf", bbox_inches='tight')
plt.close()
print(f"  ✓ fig06_certification")


# ============================================================
# FIGURE 7-8 : SLD — rigorous closed-loop monodromy, worst of 3 tool positions
# ============================================================
print(f"\n=== SLD computation (rigorous monodromy, 3 positions — a few minutes) ===")
t_sld = time.time()

RPM_arr = np.linspace(2500, 7500, 30)
ap_arr = np.linspace(0.0001, 4e-3, 25)

plate_sld = build_plate(zp_pos=HP - 0.3e-3/2, freq_perturb=0.0)
KP_POSITIONS = [0, 500, 1000]          # x = 0, L/4, L/2 (article Fig. 6 treatment)
Dp_positions = [plate_sld.get_Dp_at(kp)[0] for kp in KP_POSITIONS]

# SLD design controllers (same construction as the time-domain ones, on the
# 3-mode SLD plate)
lqg_sld = LQGController(plate_sld, dt=DT, verbose=False, kalman_V=KALMAN_V)
lqg_sld.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                         w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
adrc_sld = ESO_ADRC_Controller(plate_sld, DT, cert_row['w_q'], cert_row['w_qd'],
                               cert_row['sd'], kalman_V=KALMAN_V, verbose=False)
A_con_sld, B_cy_sld, K_con_sld = adrc_sld.controller_realization()

print("  ▷ SLD OPEN-LOOP (coupled monodromy, worst of 3 tool positions)...")
rho_OL = None
for Dp_pos in Dp_positions:
    rho_p = compute_closed_loop_SLD(RPM_arr, ap_arr,
                                    plate_sld.omega_n, ZETA, Dp_pos,
                                    None, None, None, None, None, None,
                                    NT, RT, ETA_H, PHI_ST, PHI_EX,
                                    K1, K2, KT_NOMINAL, HP,
                                    m_div=20, verbose=False)
    rho_OL = rho_p if rho_OL is None else np.maximum(rho_OL, rho_p)
print("     done.")

print("  ▷ SLD LQG (closed-loop monodromy, worst of 3 tool positions)...")
rho_LQG = None
for Dp_pos in Dp_positions:
    rho_p = compute_closed_loop_SLD(RPM_arr, ap_arr,
                                    plate_sld.omega_n, ZETA, Dp_pos,
                                    plate_sld.H_Pe_modal, plate_sld.D_obs,
                                    lqg_sld.A, lqg_sld.B, lqg_sld.K_lqr,
                                    lqg_sld.L_kal,
                                    NT, RT, ETA_H, PHI_ST, PHI_EX,
                                    K1, K2, KT_NOMINAL, HP,
                                    m_div=20, verbose=False)
    rho_LQG = rho_p if rho_LQG is None else np.maximum(rho_LQG, rho_p)
print("     done.")

print("  ▷ SLD ESO-ADRC certified design (generic closed-loop monodromy)...")
rho_ADRC = None
for Dp_pos in Dp_positions:
    rho_p = compute_closed_loop_SLD_generic(RPM_arr, ap_arr,
                                            plate_sld.omega_n, ZETA, Dp_pos,
                                            plate_sld.H_Pe_modal, plate_sld.D_obs,
                                            A_con_sld, B_cy_sld, K_con_sld,
                                            NT, RT, ETA_H, PHI_ST, PHI_EX,
                                            K1, K2, KT_NOMINAL, HP,
                                            m_div=20, verbose=False)
    rho_ADRC = rho_p if rho_ADRC is None else np.maximum(rho_ADRC, rho_p)
print(f"     done. SLD computed in {time.time()-t_sld:.1f}s")
# Note: the beta_d leakage pole of the ESO contributes a constant Floquet
# multiplier exp(-beta_d*tau) ~ 0.96 — irrelevant to the chatter boundary.

fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
cases = [("Open-Loop", rho_OL, 'Greys'),
         ("LQG", rho_LQG, 'Greens'),
         ("ESO-ADRC (certified)", rho_ADRC, 'Blues')]
for ax, (title, rho_grid, cmap) in zip(axes, cases):
    cs = ax.contourf(RPM_arr, ap_arr*1e3, rho_grid, levels=20, cmap=cmap, alpha=0.7)
    ax.contour(RPM_arr, ap_arr*1e3, rho_grid, levels=[1.0], colors='red',
               linewidths=2.5)
    ax.plot(4900, 0.3, '*', color='gold', markersize=22, markeredgecolor='k',
            markeredgewidth=1.5, label='Point article (4900 RPM, 0.3mm)', zorder=5)
    plt.colorbar(cs, ax=ax, label='ρ (rayon spectral)')
    ax.set_xlabel("RPM", fontsize=12)
    if title == "Open-Loop":
        ax.set_ylabel("$a_p$ (mm)", fontsize=12)
    ax.set_title(f"SLD {title}\nzones stables (ρ<1) en couleur claire",
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.4)
plt.suptitle("Diagramme des Lobes de Stabilité — monodromie couplée en boucle "
             "fermée, pire des 3 positions", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig07_SLD_3panels.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig07_SLD_3panels.pdf", bbox_inches='tight')
plt.close()
print(f"  ✓ fig07_SLD_3panels")

# critical depths at 4900 RPM
idx_rpm = np.argmin(np.abs(RPM_arr - 4900))


def ap_crit(rho_grid):
    for i_ap, ap_v in enumerate(ap_arr):
        if rho_grid[i_ap, idx_rpm] >= 1.0:
            return ap_v
    return ap_arr[-1]


ap_crit_OL = ap_crit(rho_OL)
ap_crit_LQG = ap_crit(rho_LQG)
ap_crit_ADRC = ap_crit(rho_ADRC)

fig, ax = plt.subplots(figsize=(13, 7))
ax.contour(RPM_arr, ap_arr*1e3, rho_OL, levels=[1.0], colors='gray',
           linewidths=2.5)
ax.contour(RPM_arr, ap_arr*1e3, rho_LQG, levels=[1.0], colors=COL[LBL_LQG],
           linewidths=2.5)
ax.contour(RPM_arr, ap_arr*1e3, rho_ADRC, levels=[1.0], colors=COL[LBL_ADRC],
           linewidths=2.0, linestyles='--')
ax.plot(4900, 0.3, '*', color='gold', markersize=25, markeredgecolor='k',
        markeredgewidth=2, label='Point article (4900 RPM, 0.3mm)', zorder=10)
ax.axvline(4900, color='black', linestyle=':', alpha=0.4, linewidth=1)
ax.plot(4900, ap_crit_OL*1e3, 's', color='gray', markersize=10,
        markeredgecolor='k', label=f'$a_p^{{crit}}$ OL = {ap_crit_OL*1e3:.2f}mm')
ax.plot(4900, ap_crit_LQG*1e3, 's', color=COL[LBL_LQG], markersize=10,
        markeredgecolor='k', label=f'$a_p^{{crit}}$ LQG = {ap_crit_LQG*1e3:.2f}mm')
ax.plot(4900, ap_crit_ADRC*1e3, 's', color=COL[LBL_ADRC], markersize=10,
        markeredgecolor='k',
        label=f'$a_p^{{crit}}$ ESO-ADRC = {ap_crit_ADRC*1e3:.2f}mm')
custom_lines = [
    plt.Line2D([0], [0], color='gray', lw=2.5, label='Open-Loop'),
    plt.Line2D([0], [0], color=COL[LBL_LQG], lw=2.5, label='LQG'),
    plt.Line2D([0], [0], color=COL[LBL_ADRC], lw=2.0, ls='--',
               label='ESO-ADRC (certified rung; A-ESO-ADRC fallback boundary)'),
]
leg1 = ax.legend(handles=custom_lines, loc='upper left', fontsize=11,
                 title='Frontière stabilité', framealpha=0.95)
ax.add_artist(leg1)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.set_xlabel("Vitesse de rotation RPM", fontsize=13)
ax.set_ylabel("Profondeur de coupe $a_p$ (mm)", fontsize=13)
ax.set_title("SLD boucle fermée (monodromie rigoureuse, pire des 3 positions)",
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.5)
ax.set_xlim([RPM_arr.min(), RPM_arr.max()])
ax.set_ylim([0, ap_arr[-1]*1e3])
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig08_SLD_overlay.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig08_SLD_overlay.pdf", bbox_inches='tight')
plt.close()
print(f"  ✓ fig08_SLD_overlay")


# ============================================================
# FIGURE 9 : multi-metric grid
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
metrics_list = [
    ('y_max', 'y_max (µm)', 'Maximum vibration'),
    ('y_rms', 'y_RMS (µm)', 'RMS vibration'),
    ('y_p2p', 'y_pp (µm)', 'Peak-to-peak vibration'),
    ('u_max', 'u_max (V)', 'Tension piezo pic'),
    ('u_rms', 'u_RMS (V)', 'Tension piezo RMS'),
    ('du_max', 'Δu_max (V)', 'Variation max piezo'),
]
for idx, (metric, ylabel, title) in enumerate(metrics_list):
    ax = axes[idx // 3, idx % 3]
    for j, (lbl, _) in enumerate(CONTROLLERS):
        v = [s['metrics'][lbl][metric] for s in scenarios]
        bars = ax.bar(x + (j-1)*w, v, w, color=COL[lbl], alpha=0.85,
                      edgecolor='k', label=lbl, linewidth=1.0)
        for bar, vi in zip(bars, v):
            ax.text(bar.get_x() + bar.get_width()/2, vi*1.02, f'{vi:.2f}',
                    ha='center', fontsize=7, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, axis='y', alpha=0.5)
plt.suptitle("Comparaison multi-métriques : LQG vs ESO-ADRC vs A-ESO-ADRC",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig09_metrics_grid.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig09_metrics_grid.pdf", bbox_inches='tight')
plt.close()
print(f"  ✓ fig09_metrics_grid")


# ============================================================
# FINAL TEXT SUMMARY
# ============================================================
print(f"\n{'='*72}")
print(f" RÉSUMÉ FINAL : LQG vs ESO-ADRC vs A-ESO-ADRC ")
print(f"{'='*72}")

print(f"\n  PERFORMANCE VIBRATOIRE (y_RMS, µm) :")
hdr = "".join(f"{lbl:<14}" for lbl, _ in CONTROLLERS)
print(f"  {'Scénario':<28}{hdr}")
print(f"  {'-'*70}")
for s in scenarios:
    vals = "".join(f"{s['metrics'][lbl]['y_rms']:<14.4f}" for lbl, _ in CONTROLLERS)
    div = " ".join(lbl for lbl, _ in CONTROLLERS if s['metrics'][lbl]['diverged'])
    print(f"  {s['name']:<28}{vals}" + (f"  [DIV: {div}]" if div else ""))
means = {lbl: np.mean([s['metrics'][lbl]['y_rms'] for s in scenarios])
         for lbl, _ in CONTROLLERS}
print(f"  {'-'*70}")
print(f"  {'MOYENNE':<28}" + "".join(f"{means[lbl]:<14.4f}"
                                     for lbl, _ in CONTROLLERS))

print(f"\n  DESIGNS (grille sur le modèle nominal uniquement) :")
print(f"     LQG                  : w_q={LQG_SHARED.w_q:.0e}, "
      f"w_qd={LQG_SHARED.w_qd:.0e}")
print(f"     ESO-ADRC performance : (w_q={perf_row['w_q']:.0e}, "
      f"w_qd={perf_row['w_qd']:.0e}, sigma_d={perf_row['sd']:.0e}), "
      f"rms_nom={perf_row['rms']:.3f}µm, worst-ρ={perf_row['rho']:.3f}")
print(f"     ESO-ADRC certified   : (w_q={cert_row['w_q']:.0e}, "
      f"w_qd={cert_row['w_qd']:.0e}, sigma_d={cert_row['sd']:.0e}), "
      f"rms_nom={cert_row['rms']:.3f}µm, worst-ρ={cert_row['rho']:.3f}")

print(f"\n  STABILITÉ (SLD, monodromie rigoureuse) à RPM = 4900 :")
print(f"     a_p crit OPEN-LOOP : {ap_crit_OL*1e3:.3f} mm")
print(f"     a_p crit LQG       : {ap_crit_LQG*1e3:.3f} mm "
      f"({ap_crit_LQG/ap_crit_OL:.1f}x OL)")
print(f"     a_p crit ESO-ADRC  : {ap_crit_ADRC*1e3:.3f} mm "
      f"({ap_crit_ADRC/ap_crit_OL:.1f}x OL) [certified rung; A-ESO-ADRC "
      f"fallback boundary]")

s_n = scenarios[0]
print(f"\n  EFFORT DE COMMANDE (S1 nominal) :")
for lbl, _ in CONTROLLERS:
    print(f"     {lbl:<12}: u_max = {s_n['metrics'][lbl]['u_max']:6.2f}V, "
          f"u_RMS = {s_n['metrics'][lbl]['u_rms']:5.2f}V")

print(f"\nTemps total : {time.time()-t_global:.1f} s")
print(f"\n  >>> figures dans {OUT_DIR}/")
