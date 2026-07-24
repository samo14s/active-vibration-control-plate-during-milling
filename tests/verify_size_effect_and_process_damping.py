"""
verify_size_effect_and_process_damping.py
=========================================
ROADMAP item 17 — the size effect, edge forces and process damping.

Four checks:

1. INERT BY DEFAULT. Switching the machinery in must be a visible decision:
   with the default settings the force is bit-identical to the constant-K_t
   law, so no existing result can change silently.

2. THE SIZE EFFECT ACTS IN THE RIGHT DIRECTION AND AT THE RIGHT PLACE. The
   power law must agree with the constant law at the calibration chip
   thickness and raise the specific pressure below it. At h_mean = 2.0 um the
   difference is what decides whether a constant K_t is defensible here.

3. EDGE FORCES DOMINATE AS h -> 0. That is the whole point of carrying them:
   a purely proportional law predicts zero force from a vanishing chip, which
   is wrong, and it is why the linear model mis-states the force exactly in
   the regime this process operates in.

4. PROCESS DAMPING HAS THE RIGHT PHYSICAL SIGNATURE. C_pd = K_pd a_p / V_c
   must raise the stability boundary, scale with axial depth, and FALL with
   cutting speed -- the speed dependence is the recognisable fingerprint, and
   its absence is why speed-independent models under-predict low-speed lobes.
"""
import numpy as np
import _bootstrap  # noqa: F401

from evolving_plate import EvolvingPlateModel
from closed_loop_sld import spectral_radius, critical_depth
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
BASE = dict(Omega=OMEGA, NT=NT, RT=RT, eta=ETA_H, phi_st=PHI_ST,
            phi_ex=PHI_EX, za_low=HP - AP, za_high=HP,
            k1=K1, k2=K2, kt=KT, ft=FT, n_slice=48)
t_probe = np.linspace(0, TAU, 120)

h_max, h_mean = chip_thickness_limits(FT, PHI_ST, PHI_EX)

print("=" * 78)
print(" 1. Inert by default")
print("=" * 78)
f_def = NonlinearMillingForce(**BASE)
f_con = NonlinearMillingForce(**BASE, kt_model="constant",
                              K_te=0.0, K_re=0.0, c_pd=0.0)
d = max(abs(f_def.modal_scalar(t, 0.0, 1e-7) - f_con.modal_scalar(t, 0.0, 1e-7))
        for t in t_probe)
good = d == 0.0
ok &= good
print(f"  max |difference| = {d:.3e}   {'PASS' if good else 'FAIL'}")
print("  (so enabling any refinement is a visible decision, never a silent one)")

print()
print("=" * 78)
print(" 2. Size effect: agrees at the calibration point, rises below it")
print("=" * 78)
f_pow = NonlinearMillingForce(**BASE, kt_model="power", kt_exponent=0.25)
print(f"  calibration chip h_ref = h_mean = {f_pow.h_ref*1e6:.3f} um")
print()
print(f"  {'h [um]':>9} {'K_t constant':>15} {'K_t power law':>16} {'ratio':>8}")
for h_um in (0.2, 0.5, 1.0, 2.0, 4.0, 10.0):
    h = h_um * 1e-6
    kc = f_con.kt_of_h(h)
    kp = f_pow.kt_of_h(h)
    print(f"  {h_um:9.2f} {kc/1e6:14.1f}M {kp/1e6:15.1f}M {kp/kc:8.3f}")
at_ref = abs(f_pow.kt_of_h(f_pow.h_ref) - KT) / KT
rises = f_pow.kt_of_h(0.5e-6) > f_pow.kt_of_h(4e-6)
good = at_ref < 1e-12 and rises
ok &= good
print()
print(f"  exact at h_ref: {at_ref:.2e};  rises as h falls: {rises}   "
      f"{'PASS' if good else 'FAIL'}")
print(f"  At this process's mean chip ({h_mean*1e6:.2f} um) the two agree by")
print(f"  construction, but at 0.5 um the power law is "
      f"{f_pow.kt_of_h(0.5e-6)/KT:.2f}x the constant value -- which is why a")
print(f"  single K_t must be quoted WITH the chip thickness it belongs to.")

print()
print("=" * 78)
print(" 3. Edge forces dominate as the chip vanishes")
print("=" * 78)
# representative magnitudes; the real values must come from the literature
K_TE, K_RE = 20e3, 12e3       # N/m of cutting edge -- PLACEHOLDER
f_edge = NonlinearMillingForce(**BASE, K_te=K_TE, K_re=K_RE)
print(f"  edge coefficients (PLACEHOLDER, not calibrated): "
      f"K_te = {K_TE/1e3:.0f} kN/m, K_re = {K_RE/1e3:.0f} kN/m")
print()
print(f"  {'|du| [um]':>11} {'chip only':>14} {'chip + edge':>14} "
      f"{'edge share':>12}")
shares = []
for du_um in (0.0, 0.5, 2.0, 3.5):
    du = du_um * 1e-6
    a = np.mean([abs(f_con.modal_scalar(t, 0.0, du)) for t in t_probe])
    b = np.mean([abs(f_edge.modal_scalar(t, 0.0, du)) for t in t_probe])
    share = abs(b - a) / max(abs(b), 1e-30)
    shares.append(share)
    print(f"  {du_um:11.2f} {a:14.4e} {b:14.4e} {share*100:11.1f} %")
good = shares[-1] > shares[0]
ok &= good
print()
print(f"  the edge share grows as vibration thins the chip: "
      f"{'PASS' if good else 'FAIL'}")

print()
print("=" * 78)
print(" 4. Process damping: raises the lobe, and FALLS with cutting speed")
print("=" * 78)
plate = EvolvingPlateModel(0.100, HP, 0.004, 2830, 69e9, 0.33, N1=30, N2=24,
                           n_modes=3, zeta_modes=[0.0031, 0.0017, 0.0027],
                           verbose=False)
plate.precompute_Dp(zp_pos=HP - AP / 2, n_pos=2001)
plate.set_observation(0.100, 0.5 * HP)
plate.add_piezo_patch(0, 0.020, 0, 0.060, d31=175e-12, h_Pa=0.7e-3,
                      E_Pe=63e9, nu_Pe=0.35)
Dp = plate.Dp_array[:, 1000]
ZETA = np.array([0.0031, 0.0017, 0.0027])

# C_pd = K_pd * a_p / V_c enters as damping along Dp. Folding it into the
# modal damping is exact for a rank-one contribution along Dp, and is how it
# is represented in the equivalent-viscous form.
# Scaled so the added damping is a few times the 0.31 % structural value at
# the nominal operating point -- large enough to be visible, small enough that
# the critical depth is not censored at the search ceiling. A value that
# saturates the search makes the comparison meaningless: the reported "gain"
# then just tracks the OPEN-LOOP variation, which is what a first attempt at
# this test did.
K_PD = 5.0e4          # N.s/m^2 -- PLACEHOLDER, must come from calibration


def zeta_eff(rpm, ap):
    """Modal damping augmented by the equivalent process damping."""
    V_c = 2 * np.pi * rpm / 60 * RT
    c_pd = K_PD * ap / V_c
    # rank-one along Dp, projected onto each mode, normalised by 2 omega
    add = c_pd * Dp ** 2 / (2 * plate.omega_n)
    return ZETA + add


COMMON = dict(NT=NT, RT=RT, eta_h=ETA_H, phi_st=PHI_ST, phi_ex=PHI_EX,
              hp=HP, k1=K1, k2=K2, kt=KT)
print(f"  K_pd = {K_PD:.1e} N.s/m^2 (PLACEHOLDER)")
print()
AP_HI = 4e-3
print(f"  {'RPM':>7} {'V_c [m/min]':>13} {'added zeta_1':>14} "
      f"{'ap_crit no pd':>15} {'ap_crit with pd':>17} {'gain':>9}")
gains, added = [], []
censored = False
for rpm in (1200, 2400, 4900, 9800):
    a_no = critical_depth(lambda a: spectral_radius(
        plate.omega_n, ZETA, Dp, plate.H_Pe_modal, plate.D_obs,
        RPM=rpm, ap=a, Ts=60.0 / (NT * rpm) / 82,
        ctrl_design=None, **COMMON), lo=1e-6, hi=AP_HI, tol=1e-6)
    a_pd = critical_depth(lambda a: spectral_radius(
        plate.omega_n, zeta_eff(rpm, a), Dp, plate.H_Pe_modal, plate.D_obs,
        RPM=rpm, ap=a, Ts=60.0 / (NT * rpm) / 82,
        ctrl_design=None, **COMMON), lo=1e-6, hi=AP_HI, tol=1e-6)
    dz = (zeta_eff(rpm, a_no) - ZETA)[0]
    added.append((rpm, dz))
    g = a_pd / max(a_no, 1e-30)
    hit = a_pd >= AP_HI * 0.99
    censored |= hit
    gains.append((rpm, g, hit))
    print(f"  {rpm:7d} {2*np.pi*rpm/60*RT*60:13.1f} {dz*100:13.3f}% "
          f"{a_no*1e3:14.4f} {a_pd*1e3:16.4f} {g:8.2f}x"
          f"{'  CENSORED' if hit else ''}")

if censored:
    print()
    print("  !! at least one point is censored at the search ceiling, so the")
    print("     gain column there tracks the open-loop value, not the damping")

# the definitional test, immune to censoring: c_pd = K_pd a_p / V_c ~ 1/V_c
falls_by_construction = added[0][1] > added[-1][1]
raises_lobe = all(g >= 0.999 for _, g, _ in gains)
uncens = [(r, g) for r, g, h in gains if not h]
falls_with_speed = (falls_by_construction and
                    (len(uncens) < 2 or uncens[0][1] >= uncens[-1][1]))
ok &= raises_lobe and falls_with_speed
print()
print(f"  never lowers the boundary        : {'PASS' if raises_lobe else 'FAIL'}")
print(f"  added damping falls with speed   : "
      f"{'PASS' if falls_with_speed else 'FAIL'} "
      f"({added[0][1]*100:.3f}% at {added[0][0]} RPM -> "
      f"{added[-1][1]*100:.3f}% at {added[-1][0]} RPM)")
print()
print("  That speed dependence is the recognisable fingerprint of process")
print("  damping, and its absence is why a speed-independent model")
print("  under-predicts the low-speed lobes.")

print()
print("=" * 78)
print(" NOTE ON CALIBRATION")
print("=" * 78)
print("  K_te, K_re and K_pd above are PLACEHOLDERS chosen to exercise the")
print("  machinery, not calibrated values. Real coefficients for AL6061-T6")
print("  must be read from Wang, Hao, Wang, Hou & Lallart, Shock and")
print("  Vibration 2018, Art. 3831825 (same alloy, same tool geometry, and it")
print("  publishes a chip-thickness power law rather than a constant).")
print("  No result in this repository currently depends on these numbers:")
print("  every refinement is off by default.")

print()
print("=" * 78)
print(f" RESULT: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
print("=" * 78)
raise SystemExit(0 if ok else 1)
