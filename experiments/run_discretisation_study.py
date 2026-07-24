"""
run_discretisation_study.py
===========================
ROADMAP items 10, 11 and 13 — settle the discretisation for the whole package
and publish the convergence table.

Three separate discretisation parameters currently disagree with each other
in the starting package, and each of them moves a headline number:

  10. THE DELAY QUANTISATION. `n_tau = round(tau/dt)` with no check. At
      tau = 4081.63 us and dt = 50 us, tau/dt = 81.633 rounds to 82, so the
      simulated delay is 4100 us: the solver runs at an effective 4878 RPM
      while its own stability analysis uses the exact tau at 4900 RPM, and
      every FFT harmonic marker is 0.45 % off per harmonic.

  13. THE INTEGRATION STEP. Average-acceleration Newmark is unconditionally
      stable but not accurate: period elongation is (omega dt)^2/12. At
      dt = 50 us the highest retained mode gets 7.5 samples per cycle.

  11. THE SEMI-DISCRETISATION RESOLUTION. `gen_SLD_academic_style.py` uses
      m_div = 40 and `main_simulation.py` uses 30, and the open-loop critical
      depth moves between them -- a spread on the headline number produced by
      a parameter nobody chose deliberately.

This script quantifies all three and recommends one setting.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tests"))
import _bootstrap  # noqa: F401,E402

from evolving_plate import EvolvingPlateModel                     # noqa: E402
from closed_loop_sld import spectral_radius, critical_depth       # noqa: E402

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

plate = EvolvingPlateModel(LP, HP, H0, RHO, E_AL, NU, N1=N1, N2=N2,
                           n_modes=N_MODES, zeta_modes=ZETA, verbose=False)
plate.precompute_Dp(zp_pos=HP - 0.15e-3, n_pos=2001)
plate.set_observation(*OBS)
plate.add_piezo_patch(*PATCH, **PIEZO)
Dp = plate.Dp_array[:, 1000]

print("=" * 78)
print(" DISCRETISATION STUDY")
print("=" * 78)
print(f"  tooth period tau = {TAU*1e6:.3f} us at {RPM} RPM, {NT} teeth")
print(f"  retained modes   = {np.round(plate.freq_n, 1)} Hz")

# ------------------------------------------------------------------ item 10
print()
print("-" * 78)
print(" 10. Delay quantisation: dt must divide tau exactly")
print("-" * 78)
print(f"  {'dt [us]':>10} {'tau/dt':>10} {'n_tau':>7} {'delay [us]':>12} "
      f"{'eff. RPM':>10} {'error':>9}")
for dt in (50e-6, 49.776e-6, 25e-6, 20e-6, 12.444e-6, 10e-6):
    r = TAU / dt
    n_tau = int(round(r))
    delay = n_tau * dt
    rpm_eff = 60.0 / (NT * delay)
    err = (rpm_eff - RPM) / RPM * 100
    exact = abs(r - n_tau) < 1e-9
    print(f"  {dt*1e6:10.3f} {r:10.4f} {n_tau:7d} {delay*1e6:12.3f} "
          f"{rpm_eff:10.1f} {err:+8.3f}%{'   exact' if exact else ''}")
print()
print("  The baseline uses dt = 50 us -> 4878.0 RPM in the time domain while")
print("  its stability analysis uses 4900 RPM. Choose dt = tau/N for integer")
print("  N; tau/82 = 49.776 us is the nearest to the original step.")

# ------------------------------------------------------------------ item 13
print()
print("-" * 78)
print(" 13. Newmark period elongation, (omega dt)^2 / 12")
print("-" * 78)
print(f"  {'dt [us]':>10} " +
      " ".join(f"{'mode '+str(k+1):>12}" for k in range(N_MODES)) +
      f" {'worst s/cycle':>14}")
for N in (82, 164, 328, 656):
    dt = TAU / N
    el = (plate.omega_n * dt) ** 2 / 12 * 100
    spc = 1.0 / (plate.freq_n * dt)
    print(f"  {dt*1e6:10.3f} " +
          " ".join(f"{e:11.3f}%" for e in el) +
          f" {spc.min():14.1f}")
bw = 2 * ZETA * 100
print()
print(f"  half-power bandwidths: {np.round(bw, 3)} %")
print("  A mode is numerically detuned by more than its own bandwidth when")
print("  the elongation above exceeds these. At dt = tau/82 (49.8 us) mode 3")
print(f"  is detuned {(plate.omega_n[2]*TAU/82)**2/12*100:.2f} % against a "
      f"{bw[2]:.3f} % bandwidth -- about "
      f"{((plate.omega_n[2]*TAU/82)**2/12*100)/bw[2]:.0f}x.")
print("  dt = tau/328 (12.4 us) brings every mode inside its own bandwidth.")

# ------------------------------------------------------------------ item 11
print()
print("-" * 78)
print(" 11. Semi-discretisation resolution: one m_div for the package")
print("-" * 78)
print("  Open-loop critical depth vs the plant refinement, at fixed")
print("  controller rate. `refine` multiplies the plant sub-steps only.")
print()
print(f"  {'refine':>8} {'m_div':>8} {'rho at ap=0.3mm':>18} "
      f"{'ap_crit [mm]':>14} {'drift':>9}")
prev = None
for refine in (1, 2, 4, 8):
    rho = spectral_radius(plate.omega_n, ZETA, Dp, plate.H_Pe_modal,
                          plate.D_obs, RPM=RPM, ap=0.3e-3, Ts=TAU / 82,
                          refine=refine, ctrl_design=None, **COMMON)
    apc = critical_depth(lambda a: spectral_radius(
        plate.omega_n, ZETA, Dp, plate.H_Pe_modal, plate.D_obs,
        RPM=RPM, ap=a, Ts=TAU / 82, refine=refine,
        ctrl_design=None, **COMMON), lo=1e-7, hi=8e-3)
    d = "" if prev is None else f"{(apc-prev)/prev*100:+8.2f}%"
    print(f"  {refine:8d} {82*refine:8d} {rho:18.6f} {apc*1e3:14.5f} {d:>9}")
    prev = apc

print()
print("  The baseline's own two scripts disagree (m_div = 30 vs 40). Fix one")
print("  value package-wide and publish this table; the converged value is")
print("  the only one that should appear in a ratio.")

# ------------------------------------------------------------------
print()
print("=" * 78)
print(" RECOMMENDED SETTINGS")
print("=" * 78)
print(f"  controller sample period  Ts    = tau/82  = {TAU/82*1e6:.3f} us")
print(f"    -> tau/Ts is exactly 82, so the effective spindle speed is")
print(f"       {RPM} RPM with no quantisation error")
print(f"  time-integration step     dt    = tau/328 = {TAU/328*1e6:.3f} us")
print(f"    -> every retained mode detuned less than its own bandwidth")
print(f"  semi-discretisation       refine >= 4 (m_div >= 328)")
print(f"    -> quote a_p,crit from the converged value, and publish the table")
print()
print("  Assert `abs(tau/dt - round(tau/dt)) < 1e-9` wherever a delay line is")
print("  built, and print the effective RPM, so the mismatch cannot recur.")
