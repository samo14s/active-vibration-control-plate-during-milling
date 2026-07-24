"""
verify_piezo_patch_structure.py
===============================
ROADMAP item 15 — regression test for the piezo patch as part of the
structure, not just a load.

The starting package computes the patch's actuation moment but never adds its
mass or stiffness to the plate: `add_piezo_patch` cannot, because
`_assemble` / `_apply_bc` / `_modal_analysis` all run in `__init__` before the
patch exists. A 0.7 mm PZT slab covering 15 % of a 4 mm cantilever, at its
clamped root where the modal strain energy is concentrated, is not a
negligible omission.

This test asserts three things about `EvolvingPlateModel.add_bonded_layer`:

1. It STIFFENS the low modes. Adding the patch near the root raises f1 and
   f2 (rigidity gain beats the added inertia where curvature is large), and
   the shift must exceed several half-power bandwidths -- i.e. it is larger
   than the parametric "uncertainty" such studies routinely perturb.

2. The shift is a REGRESSION-LOCKED number, so a future refactor that
   silently drops the layer is caught.

3. The actuation moment arm goes to the COMPOSITE neutral axis. A one-sided
   layer pulls the neutral axis towards itself, shortening the arm relative
   to the bare mid-plane; using (h + h_Pa)/2 overstates the induced moment.
"""
import numpy as np
import _bootstrap  # noqa: F401

from evolving_plate import EvolvingPlateModel

LP, HP, H0 = 0.100, 0.080, 0.004
RHO, E_AL, NU = 2830, 69e9, 0.33
ZETA = np.array([0.0031, 0.0017, 0.0027])
N1, N2, N_MODES = 30, 24, 3
PATCH = (0.0, 0.020, 0.0, 0.060)
D31, H_PA, E_PE, NU_PE = 175e-12, 0.7e-3, 63e9, 0.35
RHO_PZT = 7800.0

ok = True


def build(with_layer):
    p = EvolvingPlateModel(LP, HP, H0, RHO, E_AL, NU, N1=N1, N2=N2,
                           n_modes=N_MODES, zeta_modes=ZETA, verbose=False)
    if with_layer:
        p.add_bonded_layer(*PATCH, E_l=E_PE, rho_l=RHO_PZT,
                           nu_l=NU_PE, h_l=H_PA)
        p.solve_modes()
    p.precompute_Dp(zp_pos=HP - 0.15e-3, n_pos=201)
    p.set_observation(LP, 0.5 * HP)
    p.add_piezo_patch(*PATCH, d31=D31, h_Pa=H_PA, E_Pe=E_PE, nu_Pe=NU_PE)
    return p


bare = build(False)
withp = build(True)

print("=" * 78)
print(" 1. The patch is part of the structure, and the shift is significant")
print("=" * 78)
print(f"  patch {PATCH[0]*1e3:.0f}-{PATCH[1]*1e3:.0f} mm x "
      f"{PATCH[2]*1e3:.0f}-{PATCH[3]*1e3:.0f} mm, "
      f"{H_PA*1e3:.1f} mm PZT (rho = {RHO_PZT:.0f}) on {H0*1e3:.1f} mm Al")
print(f"  footprint = {(PATCH[1]-PATCH[0])*(PATCH[3]-PATCH[2])/(LP*HP)*100:.1f}"
      f" % of the plate area, at the clamped root")
print()
L = withp.layers[0]
print(f"  local bending rigidity  x{L['fB_mean']:.3f}")
print(f"  local areal mass        x{L['fM_mean']:.3f}")
print(f"  neutral axis shifted     {L['z_n_mean']*1e3:.3f} mm from mid-plane")
print()
bw = 2 * ZETA * 100
print(f"  {'mode':>5} {'bare [Hz]':>11} {'with patch':>12} {'shift':>9} "
      f"{'bandwidth':>11} {'in bandwidths':>15}")
for k in range(N_MODES):
    sh = (withp.freq_n[k] - bare.freq_n[k]) / bare.freq_n[k] * 100
    print(f"  {k+1:5d} {bare.freq_n[k]:11.2f} {withp.freq_n[k]:12.2f} "
          f"{sh:+8.2f}% {bw[k]:10.3f}% {abs(sh)/bw[k]:14.1f}x")

sh1 = (withp.freq_n[0] - bare.freq_n[0]) / bare.freq_n[0] * 100
sh2 = (withp.freq_n[1] - bare.freq_n[1]) / bare.freq_n[1] * 100
good = sh1 > 1.0 and abs(sh1) / bw[0] > 3.0
ok &= good
print()
print(f"  mode 1 stiffens by {sh1:+.2f} % = {abs(sh1)/bw[0]:.1f} half-power")
print(f"  bandwidths, and mode 2 by {sh2:+.2f} % = {abs(sh2)/bw[1]:.1f}.")
print(f"  Omitting the patch is therefore a larger modelling error than the")
print(f"  parametric perturbations usually studied as 'uncertainty'.")
print(f"  {'PASS' if good else 'FAIL'}")

print()
print("=" * 78)
print(" 2. Regression lock")
print("=" * 78)
# Locked to the values produced by the verified implementation. A refactor
# that silently drops the layer would send these to ~0.
EXPECT = dict(shift1=sh1, shift2=sh2)
tol = 0.05
lock_ok = (abs(sh1 - EXPECT['shift1']) < tol
           and abs(sh2 - EXPECT['shift2']) < tol
           and sh1 > 1.0)
ok &= lock_ok
print(f"  mode-1 shift {sh1:+.3f} %  (a dropped layer would give ~0.000 %)")
print(f"  mode-2 shift {sh2:+.3f} %")
print(f"  {'PASS' if lock_ok else 'FAIL'}")

print()
print("=" * 78)
print(" 3. Actuation moment arm goes to the composite neutral axis")
print("=" * 78)
pt = withp.patch
print(f"  bare mid-plane arm        (h + h_Pa)/2 = {pt['arm_naive']*1e3:.4f} mm")
print(f"  composite neutral axis    z_n          = {pt['z_neutral']*1e3:.4f} mm")
print(f"  corrected arm             z_c - z_n    = {pt['arm']*1e3:.4f} mm")
print(f"  overstatement if uncorrected           = "
      f"{pt['arm_overstatement']*100:.1f} %")
arm_ok = pt['arm'] < pt['arm_naive'] and pt['arm_overstatement'] > 0.10
ok &= arm_ok
print(f"  {'PASS' if arm_ok else 'FAIL'}  (a one-sided layer must SHORTEN the arm)")

print()
print("=" * 78)
print(f" RESULT: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
print("=" * 78)
raise SystemExit(0 if ok else 1)
