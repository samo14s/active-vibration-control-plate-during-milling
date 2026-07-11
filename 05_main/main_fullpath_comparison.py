"""
main_fullpath_comparison.py
===========================
FULL-PATH fair three-way comparison of the controllers over the ENTIRE feed pass
(the tool travels the whole width L_P, T_path = L_P / v_feed ≈ 20.4 s):

    1. LQG       — reactive feedback (the shared base)
    2. DARC      — LQG base + anticipative inverse-model feedforward
    3. DARC-MPC  — LQG base + feedforward + neural-network nonlinear residual

The comparison is FAIR: all three share the identical LQG base (w_q=1e14); DARC
adds the model-based feedforward, DARC-MPC adds the learned NN residual on top.
Run on the nonlinear NDDE plant with a fine displacement sensor (0.1 µm noise).

Run:  python main_fullpath_comparison.py
"""
import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plate_model import PlateModel
from milling_force import precompute_nonlinear_periodic
from lqg_controller import LQGController
from darc_mpc_v3_controller import DARC_MPC_v3_Controller
from newmark_solver import NewmarkSimulator

OUT_DIR = "figs_fullpath"
os.makedirs(OUT_DIR, exist_ok=True)

# Physical parameters (preserved package values)
LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU = 2830, 69e9, 0.33
NT = 3; DT_TOOL = 0.010
ETA = np.deg2rad(35); GAM = np.deg2rad(15)
KT = 925e6; KN = 0.26; MU = 0.20
RPM = 4900; FT = 0.02e-3; AE = 0.1e-3; AP = 0.3e-3
D31 = 175e-12; H_PA = 0.7e-3; E_PE = 63e9; NU_PE = 0.35
ZETA = [0.0031, 0.0017, 0.0027]
DT = 5e-5
SENSOR_NOISE = 1.0e-7
FF_HARMONICS = 30
NN_ITERS = 20
RT = DT_TOOL / 2
PHI_ST = np.pi - np.arccos(1 - AE / RT); PHI_EX = np.pi
OMEGA = 2 * np.pi * RPM / 60
TAU = 60.0 / (NT * RPM); N_PER = int(round(TAU / DT))
K1 = KN * np.cos(ETA); K2 = 1 + MU * np.tan(ETA) * np.cos(GAM) - KN * np.sin(ETA)
V_FEED = FT * NT * RPM / 60
T_PATH = LP / V_FEED                                   # full feed pass duration


def build_plate():
    p = PlateModel(LP, HP, BP, RHO, E_AL, NU, n_modes=3, zeta_modes=ZETA, verbose=False)
    p.precompute_Dp(zp_pos=HP - AP / 2, n_pos=2001)
    p.set_observation(LP, HP)
    p.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    return p


def force(sim):
    return precompute_nonlinear_periodic(DT, N_PER, sim.nstep, OMEGA, NT, RT, ETA,
                                         PHI_ST, PHI_EX, HP - AP, HP, K1, K2, KT)


print("=" * 70)
print(f" FULL-PATH comparison  (T_path = {T_PATH:.1f} s, {int(T_PATH/DT):,} steps)")
print(" LQG  vs  DARC (FF)  vs  DARC-MPC (FF + NN)  — fair, same LQG base")
print("=" * 70)
t_glob = time.time()
plate = build_plate()

# full-path simulator + tool trajectory over the whole width
sim = NewmarkSimulator(plate, dt=DT, T_end=T_PATH, ft=FT, tau=TAU, verbose=False)
xp = np.minimum(V_FEED * sim.t_vec, LP)
kp = np.clip(np.round(xp / LP * 2000).astype(int), 0, 2000)
a3, a4, a42, a43 = force(sim)
rng_seed = 1

# short representative training segment (mid-path) for the NN
sim_tr = NewmarkSimulator(build_plate(), dt=DT, T_end=0.5, ft=FT, tau=TAU, verbose=False)
xptr = np.minimum(V_FEED * (sim_tr.t_vec + 0.5 * T_PATH), LP)
kptr = np.clip(np.round(xptr / LP * 2000).astype(int), 0, 2000)
a3t, a4t, a42t, a43t = force(sim_tr)
Dp_ff = plate.get_Dp_at(1000)[0]                       # mode shape at mid-path

def run(ctrl):
    if ctrl is not None and ctrl.__class__.__name__ == 'DARC_MPC_v3_Controller':
        for h in ('history_u_lqg', 'history_u_ff', 'history_u_total',
                  'history_phase', 'history_safety'):
            setattr(ctrl, h, [])
    return sim.simulate(a3, a4, kp, controller=ctrl, alpha4_2_t=a42, alpha4_3_t=a43,
                        sensor_noise=SENSOR_NOISE, sensor_rng=np.random.default_rng(rng_seed),
                        progress=False)

def stats(res):
    i = res['stop_idx']; y = res['y'][:i + 1]; u = res['u'][:i + 1]
    return dict(rms=float(np.sqrt(np.mean(y**2)) * 1e6),
                ymax=float(np.max(np.abs(y)) * 1e6),
                umax=float(np.max(np.abs(u))), res=res)

# 1) LQG
print("\n[1/3] LQG (full path) ...")
lqg = LQGController(plate, dt=DT, verbose=False)
lqg.optimize_weights(w_q_list=[1e14], w_qd_list=[1e8], w_r=1.0, gain_norm_max=1e10)
lqg.discretize_observer()
t0 = time.time(); S_lqg = stats(run(lqg)); print(f"      rms={S_lqg['rms']:.4f} um  ({time.time()-t0:.0f}s)")

# 2) DARC (feedforward only)
print("[2/3] DARC = LQG + inverse-model feedforward (full path) ...")
darc = DARC_MPC_v3_Controller(plate, dt=DT, base_w_q=1e14, base_w_qd=1e8, ff_alpha=1.0,
                              ff_max=20.0, alpha4_periodic=a4[:N_PER], n_per=N_PER,
                              enable_adaptation=True, u_max=150.0, verbose=False)
darc.design_periodic_feedforward(FT, a3[:N_PER], Dp_ff, n_harm=FF_HARMONICS)
t0 = time.time(); S_darc = stats(run(darc)); print(f"      rms={S_darc['rms']:.4f} um  ({time.time()-t0:.0f}s)")

# 3) DARC-MPC (feedforward + NN residual)
print("[3/3] DARC-MPC = LQG + feedforward + NN residual (training + full path) ...")
darc_mpc = DARC_MPC_v3_Controller(plate, dt=DT, base_w_q=1e14, base_w_qd=1e8, ff_alpha=1.0,
                                  ff_max=20.0, alpha4_periodic=a4[:N_PER], n_per=N_PER,
                                  enable_adaptation=True, u_max=150.0, verbose=False)
darc_mpc.design_periodic_feedforward(FT, a3[:N_PER], Dp_ff, n_harm=FF_HARMONICS)
darc_mpc.train_nn_residual(sim_tr, a3t, a4t, kptr, alpha4_2_t=a42t, alpha4_3_t=a43t,
                           n_iter=NN_ITERS, verbose=False)
t0 = time.time(); S_mpc = stats(run(darc_mpc)); print(f"      rms={S_mpc['rms']:.4f} um  ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------- summary
g_darc = (1 - S_darc['rms'] / S_lqg['rms']) * 100
g_mpc = (1 - S_mpc['rms'] / S_lqg['rms']) * 100
print("\n" + "=" * 70)
print(f"  FULL-PATH RMS (T = {T_PATH:.1f} s):")
print(f"    LQG       : {S_lqg['rms']:.4f} um   (u_max {S_lqg['umax']:.1f} V)")
print(f"    DARC      : {S_darc['rms']:.4f} um   ({g_darc:+.1f}% vs LQG)")
print(f"    DARC-MPC  : {S_mpc['rms']:.4f} um   ({g_mpc:+.1f}% vs LQG)")
print(f"  Total time: {time.time()-t_glob:.0f} s")

# ---------------------------------------------------------------- figure
ds = max(1, sim.nstep // 4000)                          # downsample for plotting
t = sim.t_vec[::ds]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                               gridspec_kw={'height_ratios': [3, 1]})
ax1.plot(t, S_lqg['res']['y'][::ds] * 1e6, color='#2E8B57', lw=0.6, alpha=0.9,
         label=f"LQG  (RMS {S_lqg['rms']:.3f} µm)")
ax1.plot(t, S_darc['res']['y'][::ds] * 1e6, color='#1f77b4', lw=0.6, alpha=0.9,
         label=f"DARC  (RMS {S_darc['rms']:.3f} µm, {g_darc:+.0f}%)")
ax1.plot(t, S_mpc['res']['y'][::ds] * 1e6, color='#DC143C', lw=0.6, alpha=0.9,
         label=f"DARC-MPC  (RMS {S_mpc['rms']:.3f} µm, {g_mpc:+.0f}%)")
ax1.set_ylabel("Free-corner vibration $y$ (µm)", fontsize=12)
ax1.set_title(f"Full-pass control over the entire feed path (T = {T_PATH:.1f} s) — fair comparison",
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right'); ax1.grid(True, alpha=0.4)

ax2.bar(['LQG', 'DARC', 'DARC-MPC'], [S_lqg['rms'], S_darc['rms'], S_mpc['rms']],
        color=['#2E8B57', '#1f77b4', '#DC143C'], edgecolor='k', alpha=0.85)
for i, v in enumerate([S_lqg['rms'], S_darc['rms'], S_mpc['rms']]):
    ax2.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
ax2.set_ylabel("RMS (µm)"); ax2.grid(True, axis='y', alpha=0.4)
ax2.set_xlabel("Tool position along path  →  (x: 0 → 100 mm,  t: 0 → %.0f s)" % T_PATH)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fullpath_comparison.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"  figure -> {OUT_DIR}/fullpath_comparison.png")
