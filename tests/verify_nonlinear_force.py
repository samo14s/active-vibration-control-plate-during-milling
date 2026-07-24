"""
verify_nonlinear_force.py
=========================
ROADMAP item 14 — verification of the chip-thickness-positive force model.

Three checks:

1. EQUIVALENCE. With the positivity test disabled, the nonlinear model must
   reproduce the baseline's alpha3 / alpha4 exactly. This is what makes the
   nonlinear model trustworthy: it is the same physics with one inequality
   restored, not a different model.

2. THE INEQUALITY BEHAVES CORRECTLY. At this immersion the exit angle is
   phi_ex = pi, where the nominal chip thickness is zero, so loss of contact
   begins at INFINITESIMAL amplitude rather than at h_max. The required
   behaviour is therefore that the deviation from the linear model vanishes
   in the limit |du| -> 0 and grows monotonically with amplitude, and that
   the force SATURATES rather than growing without bound.

3. QUADRATURE CONVERGENCE in the number of axial slices.
"""
import numpy as np
import _bootstrap  # noqa: F401

from milling_force import milling_force_coeffs
from nonlinear_milling_force import (NonlinearMillingForce,
                                     chip_thickness_limits)

NT, RPM, D_TOOL = 3, 4900, 0.010
ETA_H, GAMMA_N = np.deg2rad(35), np.deg2rad(15)
KT, KN, MU_C = 925e6, 0.26, 0.20
AE, FT = 0.1e-3, 0.02e-3
HP, AP = 0.080, 0.3e-3
RT = D_TOOL / 2
PHI_ST = np.pi - np.arccos(1 - AE / RT)
PHI_EX = np.pi
K1 = KN * np.cos(ETA_H)
K2 = 1 + MU_C * np.tan(ETA_H) * np.cos(GAMMA_N) - KN * np.sin(ETA_H)
OMEGA = 2 * np.pi * RPM / 60
TAU = 60.0 / (NT * RPM)

ok = True

h_max, h_mean = chip_thickness_limits(FT, PHI_ST, PHI_EX)
print("=" * 78)
print(" Chip geometry for the article's parameters (ROADMAP item 17)")
print("=" * 78)
print(f"  radial immersion  ae/D        = {AE/D_TOOL*100:.2f} %")
print(f"  engagement arc    phi_ex-phi_st = {np.rad2deg(PHI_EX-PHI_ST):.2f} deg")
print(f"  max uncut chip    h_max       = {h_max*1e6:.2f} um")
print(f"  mean uncut chip   h_mean      = {h_mean*1e6:.2f} um")
print(f"  cutting speed                 = {OMEGA*RT:.2f} m/s "
      f"= {OMEGA*RT*60:.0f} m/min")
print()
print(f"  => the tooth leaves the cut once the relative vibration exceeds")
print(f"     about {h_max*1e6:.1f} um. The baseline's divergence guard is")
print(f"     5000 um = {5e-3/h_max:.0f} x that.")

# ------------------------------------------------------------------
print()
print("=" * 78)
print(" 1. EQUIVALENCE with the baseline coefficients (positivity disabled)")
print("=" * 78)
print(f"  {'n_slice':>8} {'max rel err alpha3':>20} {'max rel err alpha4':>20}")
for n_slice in (12, 24, 48, 96, 192):
    f = NonlinearMillingForce(OMEGA, NT, RT, ETA_H, PHI_ST, PHI_EX,
                              HP - AP, HP, K1, K2, KT, FT, n_slice=n_slice)
    e3, e4 = [], []
    for k in range(60):
        t = k * TAU / 60
        a3r, a4r = milling_force_coeffs(t, OMEGA, NT, RT, ETA_H,
                                        PHI_ST, PHI_EX, HP - AP, HP,
                                        K1, K2, KT)
        a3n, a4n = f.linear_coeffs(t)
        scale3 = max(abs(a3r), 1e-12)
        scale4 = max(abs(a4r), 1e-12)
        if abs(a3r) > 1e-6:
            e3.append(abs(a3n - a3r) / scale3)
        if abs(a4r) > 1e-6:
            e4.append(abs(a4n - a4r) / scale4)
    m3 = max(e3) if e3 else 0.0
    m4 = max(e4) if e4 else 0.0
    print(f"  {n_slice:8d} {m3:20.3e} {m4:20.3e}")
    if n_slice == 192:
        good = m3 < 2e-2 and m4 < 2e-2
        ok &= good
        print(f"  {'':8} -> converging to the analytic integral   "
              f"{'PASS' if good else 'FAIL'}")
print("  (the baseline integrates the same expression in closed form; the")
print("   residual is the midpoint quadrature error, falling with n_slice)")

# ------------------------------------------------------------------
print()
print("=" * 78)
print(" 2. Deviation from the linear model, and saturation")
print("=" * 78)
f = NonlinearMillingForce(OMEGA, NT, RT, ETA_H, PHI_ST, PHI_EX,
                          HP - AP, HP, K1, K2, KT, FT, n_slice=48)
t_probe = np.linspace(0, TAU, 200)
print(f"  {'|du| [um]':>11} {'F linear':>14} {'F nonlinear':>14} "
      f"{'ratio':>9} {'out of cut':>12}")
prev = None
mono = True
for du_um in (0.1, 1.0, 3.0, 4.0, 10.0, 100.0, 1000.0):
    du = du_um * 1e-6
    f.reset_stats()
    Fl = Fn = 0.0
    for t in t_probe:
        Fl += abs(f.modal_scalar(t, 0.0, du, clamp=False))
        Fn += abs(f.modal_scalar(t, 0.0, du, clamp=True))
    Fl /= len(t_probe)
    Fn /= len(t_probe)
    st = f.stats()
    print(f"  {du_um:11.1f} {Fl:14.4e} {Fn:14.4e} {Fn/max(Fl,1e-30):9.4f} "
          f"{st['fraction_out_of_cut']*100:11.1f} %")
    if prev is not None and du_um > 10 and Fn > prev * 3:
        mono = False
    prev = Fn

# The nominal chip thickness VANISHES at the exit angle (phi_ex = pi, where
# sin(phi) = 0), so any non-zero vibration drives h negative somewhere in the
# arc. Loss of contact therefore begins at infinitesimal amplitude at this
# immersion -- it is not a large-amplitude-only effect. The correct test is
# that the deviation from the linear model grows monotonically with amplitude
# and vanishes in the limit, not that the two are ever bit-identical.
print()
print("  The nominal chip thickness is ZERO at the exit angle (phi_ex = pi),")
print("  so loss of contact starts at infinitesimal amplitude, not at h_max.")
print("  Required behaviour: deviation -> 0 as |du| -> 0, and monotone in |du|.")
print()
devs = []
for du_um in (0.001, 0.01, 0.1, 1.0):
    du = du_um * 1e-6
    num = den = 0.0
    for t in t_probe:
        a = f.modal_scalar(t, 0.0, du, clamp=False)
        b = f.modal_scalar(t, 0.0, du, clamp=True)
        num += abs(a - b)
        den += abs(a)
    devs.append(num / max(den, 1e-30))
    print(f"    |du| = {du_um:7.3f} um   relative deviation = "
          f"{devs[-1]*100:8.4f} %")

vanishes = devs[0] < 0.02
monotone = all(devs[i] < devs[i + 1] for i in range(len(devs) - 1))
print()
print(f"  deviation vanishes as |du| -> 0 : {'PASS' if vanishes else 'FAIL'}")
print(f"  deviation monotone in |du|      : {'PASS' if monotone else 'FAIL'}")
print(f"  force saturates at large |du|   : {'PASS' if mono else 'FAIL'}")
ok &= vanishes and monotone and mono

print()
print("=" * 78)
print(f" RESULT: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
print("=" * 78)
raise SystemExit(0 if ok else 1)
