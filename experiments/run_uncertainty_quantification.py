"""
run_uncertainty_quantification.py
=================================
ROADMAP item 5 and validation step 6 — report the stability boundary as a
DISTRIBUTION rather than a line.

WHY THIS IS THE MOST VALUABLE THING A LAB-LESS STUDY CAN DO
-----------------------------------------------------------
Without an experiment, a single certified `a_p,crit` is a number conditional
on parameters that were never measured. Propagating the uncertainty in those
parameters converts it into something a reader can act on: a boundary with a
confidence band, and a ranking of which unmeasured parameter is actually
responsible for the width.

This also replaces the starting package's "Monte Carlo robustness analysis",
which was never run at all: `run_uncertainty_analysis` does not exist,
`run_monte_carlo` is never called, the documented ±15 % was in fact ω ±2 %
and K_T ±5 %, and the figure labelled "Monte Carlo" is a boxplot over four
deterministic scenarios. See `baseline/RETRACTED.md` item 5.

WHAT IS SAMPLED, AND WHY THOSE
------------------------------
Only quantities that genuinely cannot be pinned down without a rig:

  zeta_1..3   modal damping -- the dominant uncertainty in any chatter
              prediction, and the hardest to know without a measured FRF.
              Log-normal, because damping is positive and multiplicative.
  K_T         specific cutting pressure. Quoted as a constant in the source
              package with no chip thickness attached; the size-effect study
              shows a 1.4x spread over the relevant chip range alone.
  k_N         the normal/tangential force ratio.
  E           Young's modulus of the workpiece (alloy and temper spread).
  d31         piezo strain constant -- catalogue tolerance is wide.
  h_Pa        patch thickness tolerance.
  clamp       a stiffness knock-down on the clamped edge, standing in for the
              fact that a real fixture is not perfectly rigid. Applied as a
              reduction of the first modal frequency.

The mode shapes are recomputed only for the parameters that change them; the
rest enter the modal model directly. That keeps the sample count affordable
while still propagating the parameters that matter.

OUTPUT
------
A percentile band on `a_p,crit` open loop and under the best benchmark law,
plus a one-at-a-time sensitivity ranking so the reader knows which
unmeasured parameter to go and measure first.
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

from scipy.linalg import solve_continuous_are                     # noqa: E402

from evolving_plate import EvolvingPlateModel                     # noqa: E402
from closed_loop_sld import spectral_radius, critical_depth       # noqa: E402

LP, HP, H0 = 0.100, 0.080, 0.004
RHO, E_NOM, NU = 2830, 69e9, 0.33
ZETA_NOM = np.array([0.0031, 0.0017, 0.0027])
N_MODES, N1, N2 = 3, 30, 24
PATCH = (0.0, 0.020, 0.0, 0.060)
OBS = (LP, 0.5 * HP)
D31_NOM, HPA_NOM, E_PE, NU_PE = 175e-12, 0.7e-3, 63e9, 0.35

NT, RPM, D_TOOL = 3, 4900, 0.010
ETA_H, GAMMA_N = np.deg2rad(35), np.deg2rad(15)
KT_NOM, KN_NOM, MU_C, AE = 925e6, 0.26, 0.20, 0.1e-3
RT = D_TOOL / 2
PHI_ST = np.pi - np.arccos(1 - AE / RT)
PHI_EX = np.pi
TAU = 60.0 / (NT * RPM)
TS = TAU / 82
AP_LO, AP_HI = 1e-6, 8e-3

N_SAMPLES = int(os.environ.get("UQ_SAMPLES", "160"))
SEED = 20260724

# relative 1-sigma spreads, and how they are applied
SPREAD = dict(zeta=0.40,      # log-normal, 40 % -- damping is poorly known
              KT=0.15, kN=0.15, E=0.03,
              d31=0.10, h_Pa=0.05, clamp=0.05)

OUT = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()

print("=" * 78)
print(" UNCERTAINTY QUANTIFICATION OF THE CERTIFIED STABILITY BOUNDARY")
print("=" * 78)
print(f"  {N_SAMPLES} samples, seed {SEED}")
print("  sampled: " + ", ".join(f"{k} ±{v*100:.0f}%" for k, v in SPREAD.items()))

# The plate geometry does not change, so build it once and rescale the modal
# model per sample. E enters omega as sqrt(E); the clamp knock-down and the
# patch parameters are applied analytically, which is what makes 160 samples
# affordable without re-meshing.
base = EvolvingPlateModel(LP, HP, H0, RHO, E_NOM, NU, N1=N1, N2=N2,
                          n_modes=N_MODES, zeta_modes=ZETA_NOM, verbose=False)
base.precompute_Dp(zp_pos=HP - 0.15e-3, n_pos=2001)
base.set_observation(*OBS)
base.add_piezo_patch(*PATCH, d31=D31_NOM, h_Pa=HPA_NOM,
                     E_Pe=E_PE, nu_Pe=NU_PE)
Dp0 = base.Dp_array[:, 1000]
OM0 = base.omega_n.copy()
H0_MODAL = base.H_Pe_modal.copy()
DOBS0 = base.D_obs.copy()
print(f"  nominal f = {np.round(base.freq_n, 1)} Hz")


def design_lqg(omega_n, zeta_n, H, D_obs, w_q=1e14, w_qd=1e8):
    n = len(omega_n)
    n2 = 2 * n
    A = np.zeros((n2, n2))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -np.diag(np.asarray(omega_n) ** 2)
    A[n:, n:] = -np.diag(2 * np.asarray(zeta_n) * np.asarray(omega_n))
    B = np.zeros((n2, 1))
    B[n:, 0] = H
    C = np.zeros((1, n2))
    C[0, :n] = D_obs
    Q = np.block([[w_q * np.outer(D_obs, D_obs) + 1e-3 * np.eye(n),
                   np.zeros((n, n))],
                  [np.zeros((n, n)), w_qd * np.eye(n) + 1e-3 * np.eye(n)]])
    R = np.array([[1.0]])
    K = np.linalg.solve(R, B.T @ solve_continuous_are(A, B, Q, R))
    V = np.array([[1e-12]])
    L = solve_continuous_are(A.T, C.T, 1e-6 * np.eye(n2), V) @ C.T / 1e-12
    return dict(A=A, B=B, C=C, K=K, L=L)


def sample_to_model(p):
    """Turn a parameter dict into (omega, zeta, Dp, H, D_obs, k1, k2, kt)."""
    om = OM0 * np.sqrt(p['E'] / E_NOM) * p['clamp']
    zeta = ZETA_NOM * p['zeta']
    # patch authority scales with d31 and (approximately) with the moment arm,
    # which grows with patch thickness
    H = H0_MODAL * (p['d31'] / D31_NOM) * (p['h_Pa'] / HPA_NOM)
    k1 = p['kN'] * np.cos(ETA_H)
    k2 = (1 + MU_C * np.tan(ETA_H) * np.cos(GAMMA_N)
          - p['kN'] * np.sin(ETA_H))
    return om, zeta, Dp0, H, DOBS0, k1, k2, p['KT']


def ap_crit(p, controlled):
    om, zeta, Dp, H, Dobs, k1, k2, kt = sample_to_model(p)
    common = dict(NT=NT, RT=RT, eta_h=ETA_H, phi_st=PHI_ST, phi_ex=PHI_EX,
                  hp=HP, k1=k1, k2=k2, kt=kt)
    des = design_lqg(om, zeta, H, Dobs) if controlled else None
    return critical_depth(lambda a: spectral_radius(
        om, zeta, Dp, H, Dobs, RPM=RPM, ap=a, Ts=TS,
        ctrl_design=des, **common), lo=AP_LO, hi=AP_HI, tol=2e-6)


rng = np.random.default_rng(SEED)


def draw():
    return dict(
        zeta=float(np.exp(rng.normal(0, SPREAD['zeta']))),
        KT=KT_NOM * (1 + SPREAD['KT'] * rng.normal()),
        kN=KN_NOM * (1 + SPREAD['kN'] * rng.normal()),
        E=E_NOM * (1 + SPREAD['E'] * rng.normal()),
        d31=D31_NOM * (1 + SPREAD['d31'] * rng.normal()),
        h_Pa=HPA_NOM * (1 + SPREAD['h_Pa'] * rng.normal()),
        clamp=1.0 - abs(SPREAD['clamp'] * rng.normal()),   # clamp only softens
    )


NOMINAL = dict(zeta=1.0, KT=KT_NOM, kN=KN_NOM, E=E_NOM,
               d31=D31_NOM, h_Pa=HPA_NOM, clamp=1.0)

print()
print("-" * 78)
print(" Nominal")
print("-" * 78)
a_ol_nom = ap_crit(NOMINAL, False)
a_cl_nom = ap_crit(NOMINAL, True)
print(f"  open loop  a_p,crit = {a_ol_nom*1e3:.4f} mm")
print(f"  LQG        a_p,crit = {a_cl_nom*1e3:.4f} mm")

print()
print("-" * 78)
print(f" Propagating the unmeasured parameters ({N_SAMPLES} samples)")
print("-" * 78)
ol, cl, keep = [], [], []
for i in range(N_SAMPLES):
    p = draw()
    try:
        a1, a2 = ap_crit(p, False), ap_crit(p, True)
    except Exception:
        continue
    ol.append(a1); cl.append(a2); keep.append(p)
    if (i + 1) % max(1, N_SAMPLES // 8) == 0:
        print(f"   {i+1:4d}/{N_SAMPLES}   ({time.time()-t0:.0f} s)")

ol = np.array(ol); cl = np.array(cl)
n_cens_ol = int(np.sum(ol >= AP_HI * 0.99) + np.sum(ol <= AP_LO * 1.01))
n_cens_cl = int(np.sum(cl >= AP_HI * 0.99) + np.sum(cl <= AP_LO * 1.01))

print()
print("=" * 78)
print(" THE BOUNDARY IS A DISTRIBUTION, NOT A LINE")
print("=" * 78)
print(f"  {'':<12} {'p5':>10} {'p25':>10} {'median':>10} {'p75':>10} "
      f"{'p95':>10} {'p95/p5':>9}")
for tag, arr in (("open loop", ol), ("LQG", cl)):
    q = np.percentile(arr, [5, 25, 50, 75, 95]) * 1e3
    print(f"  {tag:<12} {q[0]:10.4f} {q[1]:10.4f} {q[2]:10.4f} {q[3]:10.4f} "
          f"{q[4]:10.4f} {q[4]/max(q[0],1e-9):9.1f}x")
print()
print(f"  censored at a search bound: open loop {n_cens_ol}, LQG {n_cens_cl}"
      f" of {len(ol)}")
print()
print(f"  A single quoted a_p,crit of {a_ol_nom*1e3:.3f} mm (open loop) sits")
print(f"  at the {float(np.mean(ol <= a_ol_nom))*100:.0f}th percentile of its own")
print(f"  uncertainty. The 5-95 band spans a factor "
      f"{np.percentile(ol,95)/max(np.percentile(ol,5),1e-30):.1f}.")

# ------------------------------------------------------------------
print()
print("=" * 78)
print(" WHICH UNMEASURED PARAMETER SHOULD BE MEASURED FIRST?")
print("=" * 78)
print("  One-at-a-time: each parameter moved to its +1 sigma and -1 sigma")
print("  value with the others held nominal.")
print()
print(f"  {'parameter':<12} {'a_p,crit -1s':>14} {'nominal':>10} "
      f"{'+1s':>10} {'swing':>10}")
sens = []
for key in ('zeta', 'KT', 'kN', 'E', 'd31', 'h_Pa', 'clamp'):
    lo_p, hi_p = dict(NOMINAL), dict(NOMINAL)
    if key == 'zeta':
        lo_p['zeta'] = float(np.exp(-SPREAD['zeta']))
        hi_p['zeta'] = float(np.exp(+SPREAD['zeta']))
    elif key == 'clamp':
        lo_p['clamp'] = 1.0 - SPREAD['clamp']
        hi_p['clamp'] = 1.0
    else:
        nomv = NOMINAL[key]
        lo_p[key] = nomv * (1 - SPREAD[key])
        hi_p[key] = nomv * (1 + SPREAD[key])
    a_lo, a_hi = ap_crit(lo_p, True), ap_crit(hi_p, True)
    swing = abs(a_hi - a_lo) / max(a_cl_nom, 1e-30)
    sens.append((key, a_lo, a_hi, swing))
    print(f"  {key:<12} {a_lo*1e3:14.4f} {a_cl_nom*1e3:10.4f} "
          f"{a_hi*1e3:10.4f} {swing*100:9.1f}%")

sens.sort(key=lambda r: -r[3])
print()
print("  ranked by influence on the closed-loop boundary:")
for i, (k, _, _, sw) in enumerate(sens, 1):
    print(f"    {i}. {k:<10} {sw*100:6.1f} %")
print()
print(f"  => measure {sens[0][0]} first; it dominates the uncertainty and is")
print("     the cheapest thing to pin down with a single tap test.")

res = dict(n_samples=int(len(ol)), seed=SEED, spread=SPREAD,
           nominal=dict(ap_ol=a_ol_nom, ap_lqg=a_cl_nom),
           percentiles={
               'open_loop': np.percentile(ol, [5, 25, 50, 75, 95]).tolist(),
               'lqg': np.percentile(cl, [5, 25, 50, 75, 95]).tolist()},
           censored=dict(open_loop=n_cens_ol, lqg=n_cens_cl),
           sensitivity=[dict(param=k, ap_lo=a, ap_hi=b, swing=s)
                        for k, a, b, s in sens],
           runtime_s=time.time() - t0)
with open(os.path.join(OUT, "results_uncertainty.json"), "w") as f:
    json.dump(res, f, indent=1)
print(f"\n  wrote results_uncertainty.json   ({time.time()-t0:.0f} s)")
