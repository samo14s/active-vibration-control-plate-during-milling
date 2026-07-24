"""
probe_fairness.py
=================
Independent check of the headline claim "+19.3% RMS reduction of DARC-MPC over LQG".

Hypothesis under test: the gain is an artefact of an unfair baseline.
  - main_simulation.py gives the *baseline* LQG a grid search that lands on some w_q,
  - while the LQG *inside* DARC is hard-set to base_w_q = 1e14.
If the baseline is simply retuned to w_q = 1e14, does the DARC advantage survive?

Also tests whether the phase-indexed feedforward is state-dependent at all, by
evaluating the trained network on zero state vs a range of realistic states.
"""
import _bootstrap  # noqa: F401
import numpy as np

from plate_model import PlateModel
from milling_force import precompute_alpha_periodic
from lqg_controller import LQGController
from darc_mpc_v3_controller import DARC_MPC_v3_Controller
from newmark_solver import NewmarkSimulator

LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU_AL = 2830, 69e9, 0.33
NT = 3
DT_TOOL = 0.010
ETA_H = np.deg2rad(35)
GAMMA_N = np.deg2rad(15)
KT_NOMINAL = 925e6
KN = 0.26
MU_C = 0.20
RPM = 4900
FT = 0.02e-3
AE = 0.1e-3
D31, H_PA, E_PE, NU_PE = 175e-12, 0.7e-3, 63e9, 0.35
N1, N2, N_MODES = 30, 24, 3
ZETA = [0.0031, 0.0017, 0.0027]
DT = 5e-5
T_END = 0.5
AP = 0.3e-3


def build_plate(zp_pos):
    p = PlateModel(LP, HP, BP, RHO, E_AL, NU_AL, N1=N1, N2=N2,
                   n_modes=N_MODES, zeta_modes=ZETA, verbose=False)
    p.precompute_Dp(zp_pos=zp_pos, n_pos=2001)
    p.set_observation(x_obs=LP, z_obs=HP)
    p.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    return p


def rms(y):
    return float(np.sqrt(np.mean(y ** 2)) * 1e6)


plate = build_plate(zp_pos=HP - AP / 2)

phi_st = np.pi - np.arccos(1 - AE / (DT_TOOL / 2))
phi_ex = np.pi
Omega = 2 * np.pi * RPM / 60
RT = DT_TOOL / 2
tau = 60 / (NT * RPM)
k1 = KN * np.cos(ETA_H)
k2 = 1 + MU_C * np.tan(ETA_H) * np.cos(GAMMA_N) - KN * np.sin(ETA_H)

sim = NewmarkSimulator(plate, dt=DT, T_end=T_END, ft=FT, tau=tau, verbose=False)
v_feed = FT * NT * RPM / 60
xp_path = np.minimum(v_feed * sim.t_vec, LP)
kp_idx = np.clip(np.round(xp_path / LP * 2000).astype(int), 0, 2000)
n_per = int(np.round(tau / DT))
alpha3, alpha4 = precompute_alpha_periodic(
    DT, n_per, sim.nstep, Omega, NT, RT, ETA_H,
    phi_st, phi_ex, HP - AP, HP, k1, k2, KT_NOMINAL)

print("=" * 74)
print(" PROBE 1 : is the LQG baseline detuned relative to DARC's internal LQG?")
print("=" * 74)

res_ol = sim.simulate(alpha3, alpha4, kp_idx, controller=None, progress=False)
print(f"  open loop                      y_rms = {rms(res_ol['y']):8.4f} um   "
      f"diverged_at={res_ol['diverged_at']}")

lqg_results = {}
for tag, wq in [("grid (as in main_simulation)", None), ("w_q=1e13", 1e13), ("w_q=1e14", 1e14)]:
    lqg = LQGController(plate, dt=DT, verbose=False)
    if wq is None:
        lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                             w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
    else:
        lqg.optimize_weights(w_q_list=[wq], w_qd_list=[1e8], w_r=1.0)
    lqg.discretize_observer()
    r = sim.simulate(alpha3, alpha4, kp_idx, controller=lqg, progress=False)
    lqg_results[tag] = r
    print(f"  LQG {tag:<30} y_rms = {rms(r['y']):8.4f} um   "
          f"u_max = {np.max(np.abs(r['u'])):6.2f} V   "
          f"(selected w_q={lqg.w_q:.0e}, w_qd={lqg.w_qd:.0e})")

print()
print("=" * 74)
print(" PROBE 2 : DARC-MPC v3 with its default internal LQG base (w_q=1e14)")
print("=" * 74)

plate_d = build_plate(zp_pos=HP - AP / 2)
sim_d = NewmarkSimulator(plate_d, dt=DT, T_end=T_END, ft=FT, tau=tau, verbose=False)
darc = DARC_MPC_v3_Controller(plate_d, dt=DT,
                              ff_lr=0.005, ff_max=10.0, ff_alpha=1.0,
                              alpha4_periodic=alpha4[:n_per], n_per=n_per,
                              safety_alpha=5.0, enable_adaptation=True,
                              u_max=150.0, verbose=False)
darc.pretrain_iterative_simulation(sim_d, alpha3, alpha4, kp_idx,
                                   n_iterations=30, n_epochs_per_iter=15,
                                   verbose=False)
for attr in ('history_u_lqg', 'history_u_ff', 'history_u_total',
             'history_phase', 'history_safety'):
    setattr(darc, attr, [])
res_darc = sim.simulate(alpha3, alpha4, kp_idx, controller=darc, progress=False)
print(f"  DARC-MPC v3                    y_rms = {rms(res_darc['y']):8.4f} um   "
      f"u_max = {np.max(np.abs(res_darc['u'])):6.2f} V")

print()
print("=" * 74)
print(" VERDICT")
print("=" * 74)
r_grid = rms(lqg_results["grid (as in main_simulation)"]['y'])
r_1e14 = rms(lqg_results["w_q=1e14"]['y'])
r_darc = rms(res_darc['y'])
print(f"  DARC vs LQG(grid, as published)  : {(r_grid - r_darc) / r_grid * 100:+7.2f} %")
print(f"  DARC vs LQG(w_q=1e14, fair base) : {(r_1e14 - r_darc) / r_1e14 * 100:+7.2f} %")

print()
print("=" * 74)
print(" PROBE 3 : is the trained feedforward net actually state-dependent?")
print("=" * 74)
rng = np.random.default_rng(0)
phases = np.linspace(0, 2 * np.pi, 64, endpoint=False)
u_zero = np.array([darc.ff_nn.forward(np.zeros(darc.n_x), ph)[0] for ph in phases])
spread = []
for ph in phases:
    vals = []
    for _ in range(40):
        x = np.concatenate([1e-6 * rng.normal(size=N_MODES),
                            1e-3 * rng.normal(size=N_MODES)])
        vals.append(darc.ff_nn.forward(x, ph)[0])
    spread.append(np.std(vals))
print(f"  u_FF(phase, x=0)      range = [{u_zero.min():+.3f}, {u_zero.max():+.3f}] V"
      f"   peak-to-peak = {np.ptp(u_zero):.3f} V")
print(f"  std of u_FF over realistic states, averaged over phase = {np.mean(spread):.4f} V")
print(f"  ratio  state-sensitivity / phase-amplitude = "
      f"{np.mean(spread) / max(np.ptp(u_zero), 1e-12) * 100:.2f} %")

# Fourier content of the learned feedforward
F = np.fft.rfft(u_zero) / len(u_zero)
mag = 2 * np.abs(F)
mag[0] = abs(F[0])
print("  harmonic content of u_FF(phase):")
for h in range(0, 7):
    print(f"     h={h}: {mag[h]:.4f} V   ({mag[h] / max(mag[1:].max(), 1e-12) * 100:5.1f} % of largest AC harmonic)")
print(f"  energy in harmonics 1-2 / total AC energy = "
      f"{np.sum(mag[1:3] ** 2) / max(np.sum(mag[1:] ** 2), 1e-30) * 100:.1f} %")
