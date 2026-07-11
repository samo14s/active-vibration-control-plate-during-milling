"""
validate_phase_observer.py
===========================
Validation of the DARC-MPC v4 PLAD sensorless spindle-phase observer.

T1 : synthetic signal — fundamental at an off-nominal frequency buried
     under a stronger 2nd harmonic, a chatter-like tone at 521 Hz and
     broadband noise.  The PLL must lock, estimate the frequency and
     track the raw signal phase.
T2 : closed-loop simulation at NOMINAL speed — after the one-shot
     calibration the estimated disturbance-clock phase must match the
     true clock 2π(k mod n_per)/n_per to a few degrees.
T3 : closed-loop with a constant spindle-speed offset — same check
     against the ACTUAL clock (n_per_act), plus RMS comparison
     LQG / v3 / v4.

Pass criteria printed at the end (exit code 1 on failure).
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _d in ("01_core", "02_controllers", "03_analysis"):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from plate_model import PlateModel
from milling_force import precompute_alpha_periodic
from lqg_controller import LQGController
from darc_mpc_v3_controller import DARC_MPC_v3_Controller
from darc_mpc_v4_plad_controller import (DARC_MPC_v4_PLAD_Controller,
                                         SpindlePhaseObserver)
from newmark_solver import NewmarkSimulator

DT = 5e-5
FAILURES = []


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def circ_stats(err):
    z = np.mean(np.exp(1j*np.asarray(err)))
    mean = np.angle(z)
    std = np.sqrt(-2*np.log(max(abs(z), 1e-12)))
    return np.degrees(mean), np.degrees(std)


# ================= T1 : synthetic =================
print("=== T1: synthetic PLL test ===")
f_nom = 1/(82*DT)
for f_ratio in (1.0, 1.025, 0.975):
    pll = SpindlePhaseObserver(f_nom, DT)
    f_true = f_nom*f_ratio
    rng = np.random.default_rng(1)
    n = int(0.5/DT)
    errs = []
    for k in range(n):
        t = k*DT
        th = 2*np.pi*f_true*t
        y = (0.4e-6*np.cos(th)
             + 0.5e-6*np.cos(2*th + 1.0)
             + 0.15e-6*np.cos(2*np.pi*521*t + 0.5)
             + rng.normal(0, 0.05e-6))
        theta, om, c = pll.update(y)
        if t > 0.25:
            th_hat = theta - pll.bandpass_phase(om)
            errs.append(th_hat - th)
    mean_e, std_e = circ_stats(errs)
    f_err = abs(pll.omega/(2*np.pi) - f_true)
    check(f"T1 f_ratio={f_ratio:+.3f}",
          abs(mean_e) < 12 and std_e < 5 and f_err < 0.5,
          f"mean {mean_e:+.1f}°, std {std_e:.1f}°, "
          f"f̂ {pll.omega/2/np.pi:.1f} Hz (true {f_true:.1f})")

# ================= setup for T2/T3 =================
LP, HP, BP = 0.100, 0.080, 0.004
NT = 3; DT_TOOL = 0.010
ETA_H = np.deg2rad(35); GAMMA_N = np.deg2rad(15)
KT = 925e6; KN = 0.26; MU_C = 0.20
FT = 0.02e-3; AE = 0.1e-3
ZETA = [0.0031, 0.0017, 0.0027]
AP = 0.3e-3
PHI_ST = np.pi - np.arccos(1 - AE/(DT_TOOL/2)); PHI_EX = np.pi
RT = DT_TOOL/2
K1 = KN*np.cos(ETA_H)
K2 = 1 + MU_C*np.tan(ETA_H)*np.cos(GAMMA_N) - KN*np.sin(ETA_H)

t0 = time.time()
plate = PlateModel(LP, HP, BP, 2830, 69e9, 0.33, N1=30, N2=24, n_modes=3,
                   zeta_modes=ZETA, verbose=False)
plate.precompute_Dp(zp_pos=HP - AP/2, n_pos=2001)
plate.set_observation(x_obs=LP, z_obs=HP)
plate.add_piezo_patch(0, 0.020, 0, 0.060, 175e-12, 0.7e-3, 63e9, 0.35)


def setup(n_per_act, T_end=0.5):
    rpm = 60.0/(NT*n_per_act*DT)
    Omega = 2*np.pi*rpm/60
    tau = 60/(NT*rpm)
    sim = NewmarkSimulator(plate, dt=DT, T_end=T_end, ft=FT, tau=tau,
                           verbose=False)
    v_feed = FT*NT*rpm/60
    kp = np.clip(np.round(np.minimum(v_feed*sim.t_vec, LP)/LP*2000
                          ).astype(int), 0, 2000)
    a3, a4 = precompute_alpha_periodic(DT, n_per_act, sim.nstep, Omega, NT,
                                       RT, ETA_H, PHI_ST, PHI_EX,
                                       HP-AP, HP, K1, K2, KT)
    return sim, kp, a3, a4


N_PER = 82
sim_n, kp_n, a3_n, a4_n = setup(N_PER)

v3 = DARC_MPC_v3_Controller(plate, dt=DT, ff_lr=0.005, ff_max=10.0,
                            ff_alpha=1.0, alpha4_periodic=a4_n[:N_PER],
                            n_per=N_PER, safety_alpha=5.0,
                            enable_adaptation=True, u_max=150.0,
                            verbose=False)
v3.pretrain_iterative_simulation(sim_n, a3_n, a4_n, kp_n, n_iterations=30,
                                 n_epochs_per_iter=15, verbose=False)

v4 = DARC_MPC_v4_PLAD_Controller(plate, dt=DT, alpha3_periodic=a3_n[:N_PER],
                                 ff_lr=0.005, ff_max=10.0, ff_alpha=1.0,
                                 alpha4_periodic=a4_n[:N_PER], n_per=N_PER,
                                 safety_alpha=5.0, u_max=150.0,
                                 verbose=False)
v4.copy_feedforward_from(v3)
cal = v4.calibrate_phase_reference(sim_n, a3_n, a4_n, kp_n,
                                   T_cal=0.35, t_lock=0.15, verbose=True)
check("T2 calibration", cal is not None,
      f"φ_cal = {np.degrees(cal if cal is not None else 0):+.1f}°")

lqg = LQGController(plate, dt=DT, verbose=False)
lqg.optimize_weights(w_q_list=[1e10, 1e12, 1e14, 1e16],
                     w_qd_list=[1e4, 1e6, 1e8], w_r=1.0)
lqg.discretize_observer()

I0 = int(0.2/DT)


def reset3():
    v3.history_u_lqg = []; v3.history_u_ff = []; v3.history_u_total = []
    v3.history_phase = []; v3.history_safety = []


# ================= T2 : nominal =================
print("\n=== T2: nominal deployment ===")
v4.reset_runtime()
res4 = sim_n.simulate(a3_n, a4_n, kp_n, controller=v4, progress=False)
ph_est = np.array(v4.history_phase_est)
k_arr = np.arange(1, len(ph_est)+1)
ph_true = 2*np.pi*(k_arr % N_PER)/N_PER
mean_e, std_e = circ_stats(ph_est[I0:] - ph_true[I0:])
conf = np.array(v4.history_conf)[I0:].mean()
check("T2 phase accuracy", abs(mean_e) < 5 and std_e < 5,
      f"mean {mean_e:+.1f}°, std {std_e:.1f}°, conf {conf:.2f}")

reset3()
res3 = sim_n.simulate(a3_n, a4_n, kp_n, controller=v3, progress=False)
resL = sim_n.simulate(a3_n, a4_n, kp_n, controller=lqg, progress=False)
rL = np.sqrt(np.mean(resL['y'][I0:]**2))*1e6
r3 = np.sqrt(np.mean(res3['y'][I0:]**2))*1e6
r4 = np.sqrt(np.mean(res4['y'][I0:]**2))*1e6
check("T2 no regression at nominal", r4 < r3*1.02,
      f"y_rms LQG {rL:.4f} / v3 {r3:.4f} / v4 {r4:.4f} µm")

# ================= T3 : offsets =================
print("\n=== T3: constant speed offsets ===")
for n_act in (80, 83):
    sim_m, kp_m, a3_m, a4_m = setup(n_act)
    v4.reset_runtime()
    res4m = sim_m.simulate(a3_m, a4_m, kp_m, controller=v4, progress=False)
    ph_est = np.array(v4.history_phase_est)
    k_arr = np.arange(1, len(ph_est)+1)
    ph_true = 2*np.pi*(k_arr % n_act)/n_act
    mean_e, std_e = circ_stats(ph_est[I0:] - ph_true[I0:])
    f_hat = np.array(v4.history_omega_hat)[I0:].mean()/(2*np.pi)
    f_true = 1/(n_act*DT)
    check(f"T3 phase accuracy (n_per={n_act})",
          abs(mean_e) < 8 and std_e < 5 and abs(f_hat - f_true) < 0.5,
          f"mean {mean_e:+.1f}°, std {std_e:.1f}°, "
          f"f̂ {f_hat:.1f} Hz (true {f_true:.1f})")

    reset3()
    res3m = sim_m.simulate(a3_m, a4_m, kp_m, controller=v3, progress=False)
    resLm = sim_m.simulate(a3_m, a4_m, kp_m, controller=lqg, progress=False)
    rL = np.sqrt(np.mean(resLm['y'][I0:]**2))*1e6
    r3 = np.sqrt(np.mean(res3m['y'][I0:]**2))*1e6
    r4 = np.sqrt(np.mean(res4m['y'][I0:]**2))*1e6
    check(f"T3 v4 beats desynchronised v3 (n_per={n_act})", r4 < r3,
          f"y_rms LQG {rL:.4f} / v3 {r3:.4f} / v4 {r4:.4f} µm")

print(f"\nTotal {time.time()-t0:.1f} s — "
      f"{'ALL PASS' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
sys.exit(1 if FAILURES else 0)
