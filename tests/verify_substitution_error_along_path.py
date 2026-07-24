"""
verify_substitution_error_along_path.py
=======================================
How wrong is the "equivalent damping substitution" for closed-loop lobes, and
does the answer depend on where you evaluate it?

BACKGROUND
----------
The common shortcut for a closed-loop stability lobe diagram is:
read the closed-loop damping ratios off the poles, then substitute them into
the OPEN-LOOP single-mode lobe formula. `verify_closed_loop_sld.py` measured
that error as about 6 % at mid-path, using the path-averaged mode shape that
the baseline SLD scripts use.

That 6 % is not the whole story. The substitution replaces a dynamic
output-feedback controller by a scalar damping ratio, and how badly that
misrepresents the loop depends on the actuator/disturbance geometry at the
tool position being evaluated -- which sweeps along the pass. This script
evaluates the error with the LOCAL mode shape at a series of tool positions,
rather than with one path-averaged Dp.

If the error grows substantially where the actuator is poorly aligned with
the disturbance, then the shortcut is not merely imprecise: it is least
reliable exactly where the control problem is hardest, and reporting a single
path-averaged figure hides that.
"""
import numpy as np
import _bootstrap  # noqa: F401

from scipy.linalg import solve_continuous_are

from evolving_plate import EvolvingPlateModel
from closed_loop_sld import spectral_radius, critical_depth
from actuator_placement import matched_fraction_series

LP, HP, H0 = 0.100, 0.080, 0.004
RHO, E_AL, NU = 2830, 69e9, 0.33
ZETA = np.array([0.0031, 0.0017, 0.0027])
N_MODES, N1, N2 = 3, 30, 24
PIEZO = dict(d31=175e-12, h_Pa=0.7e-3, E_Pe=63e9, nu_Pe=0.35)
PATCH = (0.0, 0.020, 0.0, 0.060)
OBS = (LP, 0.5 * HP)
NT, RPM, D_TOOL = 3, 4900, 0.010
ETA_H, GAMMA_N = np.deg2rad(35), np.deg2rad(15)
KT, KN, MU_C, AE = 925e6, 0.26, 0.20, 0.1e-3
RT = D_TOOL / 2
PHI_ST = np.pi - np.arccos(1 - AE / RT)
PHI_EX = np.pi
K1 = KN * np.cos(ETA_H)
K2 = 1 + MU_C * np.tan(ETA_H) * np.cos(GAMMA_N) - KN * np.sin(ETA_H)
COMMON = dict(NT=NT, RT=RT, eta_h=ETA_H, phi_st=PHI_ST, phi_ex=PHI_EX,
              hp=HP, k1=K1, k2=K2, kt=KT)
TAU = 60.0 / (NT * RPM)
TS = TAU / int(round(TAU / 5e-5))

plate = EvolvingPlateModel(LP, HP, H0, RHO, E_AL, NU, N1=N1, N2=N2,
                           n_modes=N_MODES, zeta_modes=ZETA, verbose=False)
plate.precompute_Dp(zp_pos=HP - 0.15e-3, n_pos=2001)
plate.set_observation(*OBS)
plate.add_piezo_patch(*PATCH, **PIEZO)
n, n2 = N_MODES, 2 * N_MODES


def design(Dp_used):
    A = np.zeros((n2, n2))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -np.diag(plate.omega_n ** 2)
    A[n:, n:] = -np.diag(2 * ZETA * plate.omega_n)
    B = np.zeros((n2, 1))
    B[n:, 0] = plate.H_Pe_modal
    C = np.zeros((1, n2))
    C[0, :n] = plate.D_obs
    Q = np.block([[1e14 * np.outer(plate.D_obs, plate.D_obs) + 1e-3 * np.eye(n),
                   np.zeros((n, n))],
                  [np.zeros((n, n)), 1e8 * np.eye(n) + 1e-3 * np.eye(n)]])
    R = np.array([[1.0]])
    K = np.linalg.solve(R, B.T @ solve_continuous_are(A, B, Q, R))
    V = np.array([[1e-12]])
    L = solve_continuous_are(A.T, C.T, 1e-6 * np.eye(n2), V) @ C.T / 1e-12
    ev = np.linalg.eigvals(A - B @ K)
    pos = ev[np.imag(ev) > 1e-9]
    pos = pos[np.argsort(np.abs(pos))][:n]
    w_cl = np.real(np.abs(pos))
    z_cl = np.real(-np.real(pos) / np.abs(pos))
    return dict(A=A, B=B, C=C, K=K, L=L), w_cl, z_cl


DES, W_CL, Z_CL = design(None)

print("=" * 78)
print(" Equivalent-damping substitution error, evaluated ALONG the pass")
print("=" * 78)
print(f"  closed-loop poles -> zeta = {np.round(Z_CL*100, 2)} %  "
      f"(open loop {np.round(ZETA*100, 2)} %)")
print()
print("  The substitution uses these zeta in the OPEN-LOOP formula.")
print("  The certified value puts the real sampled controller in the")
print("  monodromy. Both are evaluated with the LOCAL mode shape Dp(x).")
print()
print(f"  {'x [mm]':>8} {'gamma':>7} {'ap subst [mm]':>14} "
      f"{'ap certified [mm]':>18} {'error':>10}")

rows = []
for xm in (0, 10, 20, 30, 40, 50, 60, 70, 85, 100):
    idx = int(np.clip(round(xm / 1000.0 / LP * 2000), 0, 2000))
    Dp = plate.Dp_array[:, idx]
    g = matched_fraction_series([plate.H_Pe_modal], Dp.reshape(-1, 1))[0]

    ap_sub = critical_depth(lambda a: spectral_radius(
        W_CL, Z_CL, Dp, plate.H_Pe_modal, plate.D_obs,
        RPM=RPM, ap=a, Ts=TS, ctrl_design=None, **COMMON), lo=1e-7, hi=8e-3)
    ap_cert = critical_depth(lambda a: spectral_radius(
        plate.omega_n, ZETA, Dp, plate.H_Pe_modal, plate.D_obs,
        RPM=RPM, ap=a, Ts=TS, ctrl_design=DES, **COMMON), lo=1e-7, hi=8e-3)

    censored = ap_sub >= 8e-3 * 0.99 or ap_cert >= 8e-3 * 0.99
    err = (ap_sub - ap_cert) / ap_cert * 100
    rows.append((xm, g, ap_sub, ap_cert, err, censored))
    mark = "  (censored)" if censored else ""
    print(f"  {xm:8.0f} {g:7.3f} {ap_sub*1e3:14.4f} {ap_cert*1e3:18.4f} "
          f"{err:+9.1f}%{mark}")

print()
valid = [r for r in rows if not r[5]]
if valid:
    errs = np.array([r[4] for r in valid])
    print(f"  over {len(valid)} uncensored stations: "
          f"error from {errs.min():+.1f}% to {errs.max():+.1f}%, "
          f"mean {errs.mean():+.1f}%")
    opt = errs[errs > 0]
    if len(opt):
        print(f"  OPTIMISTIC (over-promises stability) at "
              f"{len(opt)} of {len(valid)} stations, worst {opt.max():+.1f}%")

print()
print("=" * 78)
print(" Why the path-averaged evaluation looks harmless: it is degenerate")
print("=" * 78)
print("  The baseline SLD scripts average the SIGNED mode shape along the path")
print("  (gen_SLD_academic_style.py: Dp_avg = np.mean(Dp_sample, axis=0)).")
print("  Mode 2 of a cantilever plate is antisymmetric in x, so that average")
print("  cancels it exactly -- and the regenerative gain depends on Dp^2.")
print()
D = plate.Dp_array
Dp_avg = D.mean(axis=1)                  # signed average, as the baseline does
Dp_rms = np.sqrt((D ** 2).mean(axis=1))  # magnitude-preserving alternative
print(f"  {'mode':>6} {'signed mean':>15} {'rms along path':>16} {'retained':>10}")
for k in range(N_MODES):
    print(f"  {k+1:6d} {Dp_avg[k]:+15.4e} {Dp_rms[k]:16.4f} "
          f"{abs(Dp_avg[k])/Dp_rms[k]*100:9.2f}%")
print()
print(f"  mode 2 at x=0 {D[1,0]:+.3f}, at x=100 mm {D[1,-1]:+.3f}"
      f"  ->  signed average {Dp_avg[1]:+.2e}")
print(f"  in the regenerative gain that is a factor "
      f"{(D[1]**2).mean()/max(Dp_avg[1]**2,1e-300):.2e} error on mode 2")
print()

for tag, Dp_u in (("signed average (baseline)", Dp_avg),
                  ("rms magnitude", Dp_rms)):
    ap_s = critical_depth(lambda a: spectral_radius(
        W_CL, Z_CL, Dp_u, plate.H_Pe_modal, plate.D_obs,
        RPM=RPM, ap=a, Ts=TS, ctrl_design=None, **COMMON), lo=1e-7, hi=8e-3)
    ap_c = critical_depth(lambda a: spectral_radius(
        plate.omega_n, ZETA, Dp_u, plate.H_Pe_modal, plate.D_obs,
        RPM=RPM, ap=a, Ts=TS, ctrl_design=DES, **COMMON), lo=1e-7, hi=8e-3)
    print(f"  {tag:<28} substitution {ap_s*1e3:7.4f} mm  "
          f"certified {ap_c*1e3:7.4f} mm  -> {(ap_s-ap_c)/ap_c*100:+6.1f}%")

print()
print("  So the reassuring path-averaged number is not a fair summary of the")
print("  per-station errors above: it is computed on a mode shape from which")
print("  mode 2 has been removed entirely. Averaging a signed mode shape does")
print("  not merely hide the substitution error, it destroys the model being")
print("  averaged. Evaluate the lobe position by position.")
