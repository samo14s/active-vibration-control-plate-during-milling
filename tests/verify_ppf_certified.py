"""
verify_ppf_certified.py
=======================
Certify REAL positive position feedback, not its static limit.

WHY THIS MATTERS
----------------
`run_benchmark.py` reported that "static modal position feedback achieved
nothing", which is true and physically correct -- a stiffness shift adds no
damping, and chatter is a damping problem. But it is emphatically NOT a
verdict on PPF, and saying so would be wrong: PPF is the origin of the
workpiece-side active-damping literature, and Zhang & Sims (2005) report a
7x increase in limiting depth of cut with it.

The difference is the filter. PPF is

    eta_ddot + 2 zf wf eta_dot + wf^2 eta = wf^2 y
    u = -g eta

whose second-order dynamics supply the phase lag that turns a position
measurement into damping near the targeted mode. Dropping the filter and
feeding position straight back is a different law entirely.

Certifying it needs a monodromy that carries controller states of its own,
which `build_monodromy` now accepts through the generic (Ac, Bc, Cc) form.

CHECKS
------
1. PPF with its filter must beat the static limit decisively.
2. The improvement must come from the FILTER, i.e. it must depend on the
   filter frequency being near the targeted mode, and degrade when detuned.
3. Sign matters: PPF is *positive* position feedback and the wrong sign
   should destabilise rather than damp.
"""
import numpy as np
import _bootstrap  # noqa: F401

from scipy.linalg import expm

from evolving_plate import EvolvingPlateModel
from closed_loop_sld import spectral_radius, critical_depth

LP, HP, H0 = 0.100, 0.080, 0.004
RHO, E_AL, NU = 2830, 69e9, 0.33
ZETA = np.array([0.0031, 0.0017, 0.0027])
N_MODES, N1, N2 = 3, 30, 24
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
TS = TAU / 82

plate = EvolvingPlateModel(LP, HP, H0, RHO, E_AL, NU, N1=N1, N2=N2,
                           n_modes=N_MODES, zeta_modes=ZETA, verbose=False)
plate.precompute_Dp(zp_pos=HP - 0.15e-3, n_pos=2001)
plate.set_observation(*OBS)
plate.add_piezo_patch(*PATCH, d31=175e-12, h_Pa=0.7e-3,
                      E_Pe=63e9, nu_Pe=0.35)
Dp = plate.Dp_array[:, 1000]
ok = True


# build_monodromy applies u = -Cc x_c. Positive position feedback wants
# u = +g eta, so Cc = -g -- and the plant's own DC gain from volts to sensor
# displacement flips it again if it is negative. Neither sign can be assumed:
# the eigenvector signs an eigensolver returns are arbitrary, so the feedback
# sign has to be derived from the plant, which is what PLANT_SIGN does.
PLANT_SIGN = float(np.sign(plate.D_obs @ np.linalg.solve(
    np.diag(plate.omega_n ** 2), plate.H_Pe_modal))) or 1.0


def ppf_ctrl(gain, wf, zf=0.3, sign=+1.0):
    """
    PPF as a generic discrete controller (Ac, Bc, Cc).

        eta_ddot + 2 zf wf eta_dot + wf^2 eta = wf^2 y,   u = +g eta

    expressed in build_monodromy's u = -Cc x_c convention and corrected for
    the plant's DC gain sign. `sign=-1` deliberately inverts it, to check that
    the choice is not arbitrary.
    """
    Ac_c = np.array([[0.0, 1.0], [-wf ** 2, -2 * zf * wf]])
    Bc_c = np.array([[0.0], [wf ** 2]])
    M = np.zeros((3, 3))
    M[:2, :2] = Ac_c
    M[:2, 2:] = Bc_c
    E = expm(M * TS)
    Cc0 = -sign * PLANT_SIGN * gain      # u = -Cc x_c  ->  u = +g eta
    return dict(Ac=E[:2, :2], Bc=E[:2, 2:].reshape(2, 1),
                Cc=np.array([[Cc0, 0.0]]))


def static_ctrl(gain, mode=0):
    """
    Static position feedback: one state, no filter -- the stiffness-shift
    limit of PPF, with the same plant-sign correction so the comparison is
    like for like.
    """
    return dict(Ac=np.array([[0.0]]), Bc=np.array([[1.0]]),
                Cc=np.array([[-PLANT_SIGN * gain]]))


def ap_of(ctrl):
    return critical_depth(lambda a: spectral_radius(
        plate.omega_n, ZETA, Dp, plate.H_Pe_modal, plate.D_obs,
        RPM=RPM, ap=a, Ts=TS, ctrl_design=None, **COMMON)
        if ctrl is None else spectral_radius(
        plate.omega_n, ZETA, Dp, plate.H_Pe_modal, plate.D_obs,
        RPM=RPM, ap=a, Ts=TS, ctrl_design=ctrl, **COMMON),
        lo=1e-7, hi=8e-3, tol=1e-6)


# spectral_radius takes a continuous design dict; feed the generic form
# straight to build_monodromy instead
from closed_loop_sld import build_monodromy                       # noqa: E402
from milling_force import milling_force_coeffs                    # noqa: E402


def rho_generic(ctrl, ap):
    m = 82
    dt = TAU / m
    a4 = np.empty(m)
    for i in range(m):
        _, v = milling_force_coeffs(i * dt, 2 * np.pi * RPM / 60, NT, RT,
                                    ETA_H, PHI_ST, PHI_EX,
                                    HP - ap, HP, K1, K2, KT)
        a4[i] = v
    P = build_monodromy(plate.omega_n, ZETA, Dp, plate.H_Pe_modal,
                        plate.D_obs, a4, dt, m, ctrl=ctrl)
    return float(np.max(np.abs(np.linalg.eigvals(P))))


def ap_generic(ctrl):
    return critical_depth(lambda a: rho_generic(ctrl, a),
                          lo=1e-7, hi=8e-3, tol=1e-6)


w1 = float(plate.omega_n[0])
ap_ol = ap_generic(None)

print("=" * 78)
print(" Real PPF, certified with its filter inside the monodromy")
print("=" * 78)
print(f"  targeted mode 1 at {plate.freq_n[0]:.1f} Hz,  open-loop "
      f"a_p,crit = {ap_ol*1e3:.4f} mm")
print(f"  plant DC gain sign (volts -> sensor) = {PLANT_SIGN:+.0f}, "
      f"applied to both laws")

print()
print("-" * 78)
print(" 1. PPF with its filter vs the static limit, both gain-swept")
print("-" * 78)
print(f"  {'gain':>10} {'static a_p,crit':>17} {'PPF a_p,crit':>15}")
best_static = best_ppf = 0.0
g_ppf_best = None
for g in np.geomspace(1e2, 1e7, 6):
    a_s = ap_generic(static_ctrl(float(g)))
    a_p = ap_generic(ppf_ctrl(float(g), w1))
    best_static = max(best_static, a_s)
    if a_p > best_ppf:
        best_ppf, g_ppf_best = a_p, float(g)
    print(f"  {g:10.2e} {a_s*1e3:17.4f} {a_p*1e3:15.4f}")

good = best_ppf > 2 * max(best_static, ap_ol)
ok &= good
print()
print(f"  best static = {best_static*1e3:.4f} mm "
      f"({best_static/ap_ol:.2f}x open loop)")
print(f"  best PPF    = {best_ppf*1e3:.4f} mm "
      f"({best_ppf/ap_ol:.2f}x open loop)")
print(f"  {'PASS' if good else 'FAIL'} -- the filter, not the feedback, is "
      f"what does the work")

print()
print("-" * 78)
print(" 2. The benefit is tied to the filter frequency (detuning degrades it)")
print("-" * 78)
# use the gain the sweep actually selected, so the detuning and sign
# checks are run where the effect is large rather than at an arbitrary
# weak gain where everything looks marginal
g_best = g_ppf_best
print(f"  {'wf / w1':>9} {'a_p,crit [mm]':>15}")
vals = []
for r in (0.25, 0.5, 1.0, 2.0, 4.0):
    a = ap_generic(ppf_ctrl(g_best, w1 * r))
    vals.append((r, a))
    print(f"  {r:9.2f} {a*1e3:15.4f}")
tuned = [a for r, a in vals if r == 1.0][0]
detuned = max(a for r, a in vals if r != 1.0)
good = tuned >= detuned
ok &= good
print()
print(f"  tuned ({tuned*1e3:.4f} mm) is the best of the sweep: "
      f"{'PASS' if good else 'FAIL'}")

print()
print("-" * 78)
print(" 3. Sign matters")
print("-" * 78)
a_pos = ap_generic(ppf_ctrl(g_best, w1, sign=+1.0))
a_neg = ap_generic(ppf_ctrl(g_best, w1, sign=-1.0))
print(f"  correct sign : {a_pos*1e3:.4f} mm")
print(f"  wrong sign   : {a_neg*1e3:.4f} mm")
good = a_pos > a_neg
ok &= good
print(f"  {'PASS' if good else 'FAIL'}")

print()
print("=" * 78)
print(" CONCLUSION")
print("=" * 78)
print("  The benchmark's 'static modal position feedback achieves nothing' is")
print("  correct and is NOT a statement about PPF. With its second-order")
print(f"  filter, PPF reaches {best_ppf/ap_ol:.1f}x the open-loop critical depth")
print(f"  where the static limit reaches {best_static/ap_ol:.2f}x. Any claim")
print("  about PPF must be made with the filter in the loop.")
print()
print("=" * 78)
print(f" RESULT: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
print("=" * 78)
raise SystemExit(0 if ok else 1)
