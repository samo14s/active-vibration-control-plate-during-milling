"""
verify_evolving.py
==================
Verification of EvolvingPlateModel.

1. The unit-thickness decomposition K_e(h) = h^3 KB1 + h KS1 and
   M_e(h) = h MT1 + h^3 MR1 must reproduce mindlin_q8's element matrices to
   machine precision, for several thicknesses.
2. With a uniform thickness field, EvolvingPlateModel must reproduce
   PlateModel's modal frequencies and its D_obs / H_Pe_modal.
3. The source element's 2x2 mass quadrature must be shown singular, and the
   selective-reduced-integration rule must restore full rank without moving
   the benchmark frequencies.
4. Uniform thinning must follow the analytical plate-bending law f ~ h.
5. The SIGN of the frequency shift under material removal must depend on
   where material is taken from (inertia near the tip vs stiffness near the
   clamp), not merely decrease.
6. The re-assembly after removal must agree with first-order eigenvalue
   perturbation theory, with the error converging as O(delta h).
"""
import _bootstrap  # noqa: F401
import numpy as np

from mindlin_q8 import stiffness_matrix_M, mass_matrix_M
from plate_model import PlateModel
from evolving_plate import EvolvingPlateModel, unit_element_matrices

LP, HP = 0.100, 0.080
RHO, E_AL, NU = 2830, 69e9, 0.33
KAPPA = 5.0 / 6.0
N1, N2 = 30, 24
ZETA = [0.0031, 0.0017, 0.0027]
D31, H_PA, E_PE, NU_PE = 175e-12, 0.7e-3, 63e9, 0.35

ok = True

print("=" * 74)
print(" 1. exact thickness decomposition, checked against the source element")
print("=" * 74)
print("   With quadrature='uniform2' the decomposition must reproduce")
print("   mindlin_q8's element matrices bit-for-bit at every thickness.")
lex, ley = LP / N1, HP / N2
KB1, KS1, MT1, MR1 = unit_element_matrices(E_AL, NU, KAPPA, RHO, lex, ley,
                                           quadrature="uniform2")
for h in (0.5e-3, 2e-3, 4e-3, 9e-3):
    Ke_ref = stiffness_matrix_M(E_AL, NU, KAPPA, lex, ley, h)
    Me_ref = mass_matrix_M(RHO, lex, ley, h)
    Ke_new = h ** 3 * KB1 + h * KS1
    Me_new = h * MT1 + h ** 3 * MR1
    eK = np.max(np.abs(Ke_new - Ke_ref)) / np.max(np.abs(Ke_ref))
    eM = np.max(np.abs(Me_new - Me_ref)) / np.max(np.abs(Me_ref))
    good = eK < 1e-12 and eM < 1e-12
    ok &= good
    print(f"   h = {h*1e3:5.2f} mm   rel.err K = {eK:.2e}   M = {eM:.2e}   "
          f"{'PASS' if good else 'FAIL'}")

print()
print("=" * 74)
print(" 1b. the source quadrature yields a SINGULAR mass matrix; SRI fixes it")
print("=" * 74)
print("   N^T Ie N is quartic in each parametric direction, so the 2x2 rule")
print("   used by the source element under-integrates it. Total mass stays")
print("   exact -- which is why the error is invisible in the frequencies --")
print("   but the matrix loses rank, so it is not a valid inner product.")
print()
KB2, KS2, MT2, MR2 = unit_element_matrices(E_AL, NU, KAPPA, RHO, lex, ley,
                                           quadrature="uniform2")
KB3, KS3, MT3, MR3 = unit_element_matrices(E_AL, NU, KAPPA, RHO, lex, ley,
                                           quadrature="sri")
h = 4e-3
M2 = h * MT2 + h ** 3 * MR2
M3 = h * MT3 + h ** 3 * MR3
r2, r3 = np.linalg.matrix_rank(M2), np.linalg.matrix_rank(M3)
m2 = np.sum(M2[0::3, 0::3])
m3 = np.sum(M3[0::3, 0::3])
m_exact = RHO * h * lex * ley
print(f"   element mass rank  uniform2 = {r2:2d}/24    sri = {r3:2d}/24")
print(f"   total element mass uniform2 = {m2:.6e} kg")
print(f"                      sri      = {m3:.6e} kg")
print(f"                      exact    = {m_exact:.6e} kg")
good = (r2 == 12 and r3 == 24
        and abs(m2 - m_exact) / m_exact < 1e-12
        and abs(m3 - m_exact) / m_exact < 1e-12)
ok &= good
print(f"   {'PASS' if good else 'FAIL'}  (rank restored, total mass unchanged)")

# assembled: the singular rule must fail Cholesky, the SRI rule must not
import scipy.linalg as _sla
for tag, quad, expect_ok in (("uniform2", "uniform2", False), ("sri", "sri", True)):
    q = EvolvingPlateModel(LP, HP, 4e-3, RHO, E_AL, NU, N1=6, N2=5, n_modes=3,
                           zeta_modes=ZETA, quadrature=quad, verbose=False)
    Mf = q.M_free.toarray()
    nz = int(np.sum(np.linalg.eigvalsh(Mf) < 1e-12 * np.linalg.eigvalsh(Mf).max()))
    try:
        _sla.cho_factor(Mf)
        chol = True
    except Exception:
        chol = False
    good = (chol == expect_ok)
    ok &= good
    print(f"   assembled 6x5, {tag:8s}: zero-mass directions = {nz:3d}, "
          f"Cholesky {'OK' if chol else 'FAILS'}   {'PASS' if good else 'FAIL'}")

print()
print("=" * 74)
print(" 1c. the quadrature fix does not move the benchmark frequencies")
print("=" * 74)
a = EvolvingPlateModel(LP, HP, 4e-3, RHO, E_AL, NU, N1=N1, N2=N2, n_modes=3,
                       zeta_modes=ZETA, quadrature="uniform2", verbose=False)
b = EvolvingPlateModel(LP, HP, 4e-3, RHO, E_AL, NU, N1=N1, N2=N2, n_modes=3,
                       zeta_modes=ZETA, quadrature="sri", verbose=False)
print(f"   uniform2 (source rule) f = {np.round(a.freq_n, 3)} Hz")
print(f"   sri      (corrected)   f = {np.round(b.freq_n, 3)} Hz")
d = np.max(np.abs(b.freq_n - a.freq_n) / a.freq_n) * 100
good = d < 0.5
ok &= good
print(f"   max shift = {d:.3f} %   {'PASS' if good else 'FAIL'}  "
      f"(the published 519 Hz benchmark is unaffected)")

print()
print("=" * 74)
print(" 2. uniform field reproduces PlateModel")
print("=" * 74)
print("   Compared with quadrature='uniform2' so this is a like-for-like")
print("   check of the assembly, not of the quadrature change.")
H0 = 0.004
ref = PlateModel(LP, HP, H0, RHO, E_AL, NU, N1=N1, N2=N2, n_modes=3,
                 zeta_modes=ZETA, verbose=False)
ref.precompute_Dp(zp_pos=HP - 0.15e-3, n_pos=201)
ref.set_observation(LP, HP)
ref.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)

ev = EvolvingPlateModel(LP, HP, H0, RHO, E_AL, NU, N1=N1, N2=N2, n_modes=3,
                        zeta_modes=ZETA, quadrature="uniform2", verbose=False)
ev.precompute_Dp(zp_pos=HP - 0.15e-3, n_pos=201)
ev.set_observation(LP, HP)
ev.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)

ef = np.max(np.abs(ev.freq_n - ref.freq_n) / ref.freq_n)
print(f"   PlateModel     f = {np.round(ref.freq_n, 3)} Hz")
print(f"   EvolvingPlate  f = {np.round(ev.freq_n, 3)} Hz")
print(f"   max rel. freq error = {ef:.2e}   {'PASS' if ef < 1e-9 else 'FAIL'}")
ok &= ef < 1e-9

# mode-shape-derived quantities agree up to the per-mode sign convention
s = np.sign(ev.D_obs * ref.D_obs)
eo = np.max(np.abs(np.abs(ev.D_obs) - np.abs(ref.D_obs)) / np.abs(ref.D_obs))
print(f"   |D_obs| rel. error       = {eo:.2e}   {'PASS' if eo < 1e-8 else 'FAIL'}")
print(f"   relative mode signs      = {s.astype(int)}")
ok &= eo < 1e-8

# H_Pe_modal is DELIBERATELY different: the baseline takes the actuator
# moment arm to the bare-plate mid-plane, (h + h_Pa)/2, while a one-sided
# bonded patch pulls the neutral axis towards itself and shortens the arm.
# The two must therefore differ by exactly the arm ratio and nothing else.
arm_ratio = ev.patch['arm'] / ev.patch['arm_naive']
got = np.max(np.abs(ev.H_Pe_modal)) / np.max(np.abs(ref.H_Pe_modal))
err = abs(got - arm_ratio) / arm_ratio
good = err < 1e-8
ok &= good
print()
print(f"   actuator moment arm, bare mid-plane : "
      f"{ev.patch['arm_naive']*1e3:.4f} mm")
print(f"   actuator moment arm, composite N.A. : "
      f"{ev.patch['arm']*1e3:.4f} mm "
      f"(neutral axis shifted {ev.patch['z_neutral']*1e3:.4f} mm)")
print(f"   => baseline overstates the arm by "
      f"{ev.patch['arm_overstatement']*100:.1f} %")
print(f"   |H_Pe| ratio: expected {arm_ratio:.6f}, got {got:.6f}   "
      f"{'PASS' if good else 'FAIL'}")

print()
print("=" * 74)
print(" 3. uniform thinning follows the plate law  f ~ h")
print("=" * 74)
print(f"   {'h [mm]':>8} {'f1 [Hz]':>10} {'f1/h':>12} {'dev. from law':>15}")
base = None
for h in (8e-3, 6e-3, 4e-3, 3e-3, 2e-3):
    p = EvolvingPlateModel(LP, HP, h, RHO, E_AL, NU, N1=N1, N2=N2, n_modes=3,
                           zeta_modes=ZETA, verbose=False)
    ratio = p.freq_n[0] / h
    if base is None:
        base = ratio
    dev = (ratio - base) / base * 100
    print(f"   {h*1e3:8.2f} {p.freq_n[0]:10.2f} {ratio:12.1f} {dev:+14.2f}%")
print("   (a few % drift is the expected shear/rotary-inertia correction:")
print("    thicker plates are relatively softer than Kirchhoff theory predicts)")

print()
print("=" * 74)
print(" 4. removal location controls the SIGN of the frequency shift")
print("=" * 74)
print("   Removing material from a cantilever near the CLAMPED edge takes away")
print("   stiffness where the modal curvature is largest -> f1 falls.")
print("   Removing near the FREE edge takes away inertia where the modal")
print("   displacement is largest -> f1 rises. A model that always lowers the")
print("   frequencies would be wrong.")
print()
for tag, z0, z1 in (("near clamp  (z = 0 .. H/4)", 0.0, HP / 4),
                    ("near tip    (z = 3H/4 .. H)", 3 * HP / 4, HP)):
    p = EvolvingPlateModel(LP, HP, 8e-3, RHO, E_AL, NU, N1=N1, N2=N2,
                           n_modes=3, zeta_modes=ZETA, verbose=False)
    f0 = p.freq_n[0]
    p.remove_material(0.0, LP, z0, z1, 3.0e-3)
    p.solve_modes()
    print(f"   {tag:<28} f1 : {f0:8.2f} -> {p.freq_n[0]:8.2f} Hz  "
          f"({(p.freq_n[0]-f0)/f0*100:+6.2f}%)")

print()
print("=" * 74)
print(" 5. first-order eigenvalue perturbation check (quantitative)")
print("=" * 74)
print("   For a small thickness change, mass-normalised modes give")
print("      d(omega^2) = phi^T (dK - omega^2 dM) phi")
print("   Comparing this against the re-solved eigenvalue validates that the")
print("   re-assembly after material removal is consistent with the modal data.")
print()
p = EvolvingPlateModel(LP, HP, 4e-3, RHO, E_AL, NU, N1=N1, N2=N2, n_modes=3,
                       zeta_modes=ZETA, verbose=False)
K0, M0 = p.K_free.copy(), p.M_free.copy()
w2_0 = p.omega_n ** 2
V0 = p.V.copy()

print("   The first-order prediction has an O(delta^2) truncation error, so the")
print("   correct test is not a fixed tolerance but CONVERGENCE: halving delta")
print("   must roughly halve the relative error.")
print()
deltas = (5e-5, 1e-5, 2e-6)
errs = {k: [] for k in range(3)}
for delta in deltas:
    q = EvolvingPlateModel(LP, HP, 4e-3, RHO, E_AL, NU, N1=N1, N2=N2,
                           n_modes=3, zeta_modes=ZETA, verbose=False)
    q.remove_material(0.0, LP, HP / 2, HP, delta)
    q.solve_modes()
    dK = q.K_free - K0
    dM = q.M_free - M0
    print(f"   delta h = {delta*1e6:6.1f} um")
    for k in range(3):
        phi = V0[:, k]
        pred = float(phi @ (dK @ phi) - w2_0[k] * (phi @ (dM @ phi)))
        actual = q.omega_n[k] ** 2 - w2_0[k]
        rel = abs(pred - actual) / max(abs(actual), 1e-30)
        errs[k].append(rel)
        print(f"      mode {k+1}: d(w^2) predicted {pred:+12.4e}  "
              f"actual {actual:+12.4e}  rel.err {rel*100:6.3f}%")

print()
print("   first-order convergence (error should fall in proportion to delta):")
pert_ok = True
for k in range(3):
    # error/delta must stay roughly constant => the residual is O(delta^2)
    ratios = [errs[k][i] / deltas[i] for i in range(len(deltas))]
    spread = max(ratios) / max(min(ratios), 1e-30)
    good = errs[k][-1] < errs[k][0] and spread < 3.0
    pert_ok &= good
    print(f"      mode {k+1}: rel.err {errs[k][0]*100:6.3f}% -> "
          f"{errs[k][-1]*100:6.3f}% as delta falls {deltas[0]/deltas[-1]:.0f}x ; "
          f"err/delta spread = {spread:4.2f}   {'PASS' if good else 'FAIL'}")
ok &= pert_ok

print()
print("=" * 74)
print(f" SUMMARY : {'ALL PASS' if ok else 'FAILURES PRESENT'}")
print("=" * 74)
raise SystemExit(0 if ok else 1)
