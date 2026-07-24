"""
run_benchmark.py
================
Fair comparison of active-damping laws for the same workpiece, at MATCHED
control effort, judged by the certified closed-loop stability boundary.

WHY IT IS SET UP THIS WAY
-------------------------
Two things make published controller comparisons in this area hard to trust,
and both are avoided here by construction.

  1. Unequal tuning effort. A proposed law is tuned carefully; the baseline
     gets whatever the author first tried. In the package this work started
     from, the baseline LQG was selected by maximising a surrogate
     (|min Re(lambda_CL)|) subject to an arbitrary gain-norm cap, and simply
     removing that cap produced a baseline that beat the proposed method.
     Here EVERY law, including the baselines, is tuned by the same scalar
     search against the SAME objective that is then reported.

  2. Unequal control effort. A law that uses three times the voltage is not
     better, it is louder. Here every law is tuned subject to the same cap on
     peak voltage, and the realised effort is reported alongside the result
     so the reader can see the comparison was actually matched.

The figure of merit is the certified critical depth of cut a_p,crit from
closed_loop_sld -- the real sampled loop inside the monodromy matrix -- rather
than an RMS reduction at one operating point, because it is the quantity that
decides whether the process can be run.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tests"))
import _bootstrap  # noqa: F401,E402

from scipy.linalg import solve_continuous_are, expm                # noqa: E402

from evolving_plate import EvolvingPlateModel                      # noqa: E402
from milling_force import precompute_alpha_periodic                # noqa: E402
from closed_loop_sld import (spectral_radius, critical_depth,      # noqa: E402
                             build_monodromy, discretize_controller)

LP, HP, H0 = 0.100, 0.080, 0.004
RHO, E_AL, NU = 2830, 69e9, 0.33
ZETA = np.array([0.0031, 0.0017, 0.0027])
N_MODES, N1, N2 = 3, 30, 24
PIEZO = dict(d31=175e-12, h_Pa=0.7e-3, E_Pe=63e9, nu_Pe=0.35)
PATCH = (0.0, 0.020, 0.0, 0.060)
OBS = (LP, 0.5 * HP)

NT, RPM, D_TOOL = 3, 4900, 0.010
ETA_H, GAMMA_N = np.deg2rad(35), np.deg2rad(15)
KT, KN, MU_C, AE, FT = 925e6, 0.26, 0.20, 0.1e-3, 0.02e-3
RT = D_TOOL / 2
PHI_ST = np.pi - np.arccos(1 - AE / RT)
PHI_EX = np.pi
K1 = KN * np.cos(ETA_H)
K2 = 1 + MU_C * np.tan(ETA_H) * np.cos(GAMMA_N) - KN * np.sin(ETA_H)
COMMON = dict(NT=NT, RT=RT, eta_h=ETA_H, phi_st=PHI_ST, phi_ex=PHI_EX,
              hp=HP, k1=K1, k2=K2, kt=KT)

U_MAX = 150.0          # actuator limit [V]
U_CAP = 60.0           # peak-voltage budget every law must respect
TAU = 60.0 / (NT * RPM)
M_CTRL = int(round(TAU / 5e-5))
TS = TAU / M_CTRL      # sample period chosen so that dt divides tau exactly
AP_EVAL = 0.3e-3       # depth at which the effort is measured

print("=" * 78)
print(" FAIR BENCHMARK AT MATCHED CONTROL EFFORT")
print("=" * 78)
print(f"  sample period Ts = {TS*1e6:.3f} us chosen so that tau/Ts = "
      f"{M_CTRL} exactly")
print(f"  effective spindle speed = {60.0/(NT*TAU):.1f} RPM (no quantisation)")
print(f"  peak-voltage budget = {U_CAP:.0f} V of {U_MAX:.0f} V available")

plate = EvolvingPlateModel(LP, HP, H0, RHO, E_AL, NU, N1=N1, N2=N2,
                           n_modes=N_MODES, zeta_modes=ZETA, verbose=False)
plate.precompute_Dp(zp_pos=HP - AP_EVAL / 2, n_pos=2001)
plate.set_observation(*OBS)
plate.add_piezo_patch(*PATCH, **PIEZO)
Dp = plate.Dp_array[:, 1000]
n = N_MODES
n2 = 2 * n

A = np.zeros((n2, n2))
A[:n, n:] = np.eye(n)
A[n:, :n] = -np.diag(plate.omega_n ** 2)
A[n:, n:] = -np.diag(2 * ZETA * plate.omega_n)
B = np.zeros((n2, 1))
B[n:, 0] = plate.H_Pe_modal
C = np.zeros((1, n2))
C[0, :n] = plate.D_obs

print(f"  plate f = {np.round(plate.freq_n,1)} Hz, "
      f"||H_Pe|| = {np.linalg.norm(plate.H_Pe_modal):.4f} N/V")

# ---------------------------------------------------------------- forcing
n_per = M_CTRL
a3, a4 = precompute_alpha_periodic(TS, n_per, 40 * n_per, 2 * np.pi * RPM / 60,
                                   NT, RT, ETA_H, PHI_ST, PHI_EX,
                                   HP - AP_EVAL, HP, K1, K2, KT)


def simulate(ctrl_design, ap=AP_EVAL, n_periods=60):
    """
    Sampled closed loop with the periodic cutting force, at the SAME timing
    the monodromy uses. Returns (y_rms [m], u_peak [V], u_rms [V], stable).
    """
    a3p, a4p = precompute_alpha_periodic(
        TS, n_per, n_periods * n_per, 2 * np.pi * RPM / 60, NT, RT, ETA_H,
        PHI_ST, PHI_EX, HP - ap, HP, K1, K2, KT)
    nstep = n_periods * n_per
    n_tau = n_per

    Ad = expm(A * TS)
    Bd = np.linalg.solve(A, (Ad - np.eye(n2)) @ B)
    if ctrl_design is not None:
        Ao, Gu, Gy = discretize_controller(ctrl_design['A'], ctrl_design['B'],
                                           ctrl_design['C'], ctrl_design['L'],
                                           TS)
        Kf = np.asarray(ctrl_design['K'], float).reshape(1, n2)

    DpDpT = np.outer(Dp, Dp)
    Om2 = np.diag(plate.omega_n ** 2)
    Cmod = np.diag(2 * ZETA * plate.omega_n)

    # a4 is periodic with period n_per, so only n_per distinct step matrices
    # exist; building them once turns the inner loop into two matrix-vector
    # products and keeps the timing identical to the monodromy computation.
    step_cache = []
    for j in range(n_per):
        Aq = np.zeros((n2, n2))
        Aq[:n, n:] = np.eye(n)
        Aq[n:, :n] = -(Om2 + a4p[j] * DpDpT)
        Aq[n:, n:] = -Cmod
        Maug = np.zeros((2 * n2, 2 * n2))
        Maug[:n2, :n2] = Aq
        Maug[:n2, n2:] = np.eye(n2)
        E = expm(Maug * TS)
        step_cache.append((E[:n2, :n2], E[:n2, n2:]))

    # The plant state is carried, not reconstructed: q history is kept only
    # for the regenerative delay term.
    st = np.zeros(n2)
    q_hist = np.zeros((n, nstep))
    xh = np.zeros(n2)
    u = 0.0
    ys, us = [], []

    for k in range(1, nstep):
        y = float(plate.D_obs @ st[:n])
        if ctrl_design is not None:
            xh = Ao @ xh + Gu.flatten() * u + Gy.flatten() * y
            u = float(np.clip((-Kf @ xh).item(), -U_MAX, U_MAX))
        qd = q_hist[:, k - n_tau] if k >= n_tau else np.zeros(n)

        Adk, Sk = step_cache[k % n_per]
        f = FT * a3p[k] * Dp + a4p[k] * (DpDpT @ qd)
        st = Adk @ st + Sk @ (B.flatten() * u
                              + np.concatenate([np.zeros(n), f]))
        q_hist[:, k] = st[:n]
        ys.append(y)
        us.append(u)
        if not np.isfinite(y) or abs(y) > 1e-3:
            return np.inf, np.inf, np.inf, False

    # discard the first third as transient
    ys = np.array(ys[len(ys) // 3:])
    us = np.array(us[len(us) // 3:])
    return (float(np.sqrt(np.mean(ys ** 2))), float(np.max(np.abs(us))),
            float(np.sqrt(np.mean(us ** 2))), True)


def lqg_design(w_q, w_qd=1e8, w_r=1.0):
    Q = np.block([[w_q * np.outer(plate.D_obs, plate.D_obs) + 1e-3 * np.eye(n),
                   np.zeros((n, n))],
                  [np.zeros((n, n)), w_qd * np.eye(n) + 1e-3 * np.eye(n)]])
    R = np.array([[w_r]])
    K = np.linalg.solve(R, B.T @ solve_continuous_are(A, B, Q, R))
    V = np.array([[1e-12]])
    L = solve_continuous_are(A.T, C.T, 1e-6 * np.eye(n2), V) @ C.T / 1e-12
    return dict(A=A, B=B, C=C, K=K, L=L)


def dvf_design(g):
    """Direct velocity feedback as a static gain on the velocity states."""
    K = np.zeros((1, n2))
    K[0, n:] = g * plate.H_Pe_modal / np.linalg.norm(plate.H_Pe_modal)
    V = np.array([[1e-12]])
    L = solve_continuous_are(A.T, C.T, 1e-6 * np.eye(n2), V) @ C.T / 1e-12
    return dict(A=A, B=B, C=C, K=K, L=L)


def ppf_design(g, mode=0, zf=0.3):
    """
    Positive position feedback on one mode, cast into the same observer-based
    form so the certified monodromy can consume it: the filter is appended as
    extra controller states via an equivalent output-feedback gain.
    Implemented here as a modal position gain on the target mode, which is
    the static limit of PPF and is what the lobe computation can certify
    without extending the state.
    """
    K = np.zeros((1, n2))
    K[0, mode] = g
    V = np.array([[1e-12]])
    L = solve_continuous_are(A.T, C.T, 1e-6 * np.eye(n2), V) @ C.T / 1e-12
    return dict(A=A, B=B, C=C, K=K, L=L)


LAWS = [
    ("open loop", None, None),
    ("DVF", dvf_design, np.geomspace(1e2, 1e7, 16)),
    ("modal position fb", ppf_design, np.geomspace(1e2, 1e8, 16)),
    ("LQG", lqg_design, np.geomspace(1e11, 1e17, 16)),
]

print()
print("  tuning each law against the SAME objective (max certified a_p,crit)")
print("  subject to the SAME peak-voltage budget")
print()
print(f"  {'law':<20} {'gain':>10} {'u_peak[V]':>10} {'u_rms[V]':>9} "
      f"{'y_rms[um]':>10} {'ap_crit[mm]':>12}")

results = []
t0 = time.time()
for name, builder, grid in LAWS:
    if builder is None:
        y, up, ur, okk = simulate(None)
        ap = critical_depth(lambda a: spectral_radius(
            plate.omega_n, ZETA, Dp, plate.H_Pe_modal, plate.D_obs,
            RPM=RPM, ap=a, Ts=TS, ctrl_design=None, **COMMON),
            lo=1e-7, hi=6e-3)
        print(f"  {name:<20} {'-':>10} {0.0:10.2f} {0.0:9.2f} "
              f"{y*1e6:10.4f} {ap*1e3:12.4f}")
        results.append(dict(law=name, gain=None, u_peak=0.0, u_rms=0.0,
                            y_rms=y, ap_crit=ap))
        continue

    best = None
    for g in grid:
        des = builder(float(g))
        y, up, ur, okk = simulate(des)
        if not okk or up > U_CAP:
            continue
        ap = critical_depth(lambda a: spectral_radius(
            plate.omega_n, ZETA, Dp, plate.H_Pe_modal, plate.D_obs,
            RPM=RPM, ap=a, Ts=TS, ctrl_design=des, **COMMON),
            lo=1e-7, hi=6e-3)
        if best is None or ap > best['ap_crit']:
            best = dict(law=name, gain=float(g), u_peak=up, u_rms=ur,
                        y_rms=y, ap_crit=ap)
    if best is None:
        print(f"  {name:<20} {'-':>10}  no stable gain within the budget")
        continue
    print(f"  {name:<20} {best['gain']:10.2e} {best['u_peak']:10.2f} "
          f"{best['u_rms']:9.2f} {best['y_rms']*1e6:10.4f} "
          f"{best['ap_crit']*1e3:12.4f}")
    results.append(best)

print()
print(f"  ({time.time()-t0:.0f} s)")
print()
print("  Every controlled row respects the same peak-voltage budget, so the")
print("  a_p,crit column compares control LAWS, not control effort.")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results_benchmark.json"), "w") as f:
    json.dump(dict(u_cap=U_CAP, Ts=TS, rpm_effective=60.0 / (NT * TAU),
                   results=results), f, indent=1)
print("  wrote results_benchmark.json")
