"""
analyze_actuator_alignment.py
=============================
How much of the milling disturbance can a given piezo patch actually cancel,
as a function of tool position along the pass?

The regenerative/forcing disturbance enters the modal equations along the
direction Dp(x_p) (the mode shapes at the tool contact point). The actuator
pushes along H_Pe (the modal force per volt). Only the component of the
disturbance that lies along H_Pe is input-matched and therefore cancellable
by feedforward; the orthogonal component is not reachable by that actuator at
all. The matched fraction is

    cos(theta(x_p)) = |<H_Pe, Dp(x_p)>| / (||H_Pe|| ||Dp(x_p)||)

This script evaluates it along the whole 100 mm pass for the baseline patch
placement, and asks whether a different single placement, or a second patch,
would keep the alignment usable over the whole path.

A study that only simulates the first few millimetres of the pass never sees
this, because the tool starts directly over the patch.
"""
import numpy as np

import _bootstrap  # noqa: F401
from plate_model import PlateModel

LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU = 2830, 69e9, 0.33
D31, H_PA, E_PE, NU_PE = 175e-12, 0.7e-3, 63e9, 0.35
ZETA = [0.0031, 0.0017, 0.0027]
AP = 0.3e-3
RPM, NT, FT = 4900, 3, 0.02e-3


def alignment(plate):
    H = plate.H_Pe_modal / np.linalg.norm(plate.H_Pe_modal)
    Dp = plate.Dp_array
    Dn = Dp / np.linalg.norm(Dp, axis=0, keepdims=True)
    return np.abs(H @ Dn)


base = PlateModel(LP, HP, BP, RHO, E_AL, NU, N1=30, N2=24, n_modes=3,
                  zeta_modes=ZETA, verbose=False)
base.precompute_Dp(zp_pos=HP - AP / 2, n_pos=2001)
base.set_observation(x_obs=LP, z_obs=HP)
base.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)

x = base.xp_array
cos_base = alignment(base)

v_feed = FT * NT * RPM / 60
print("=" * 78)
print(" Input-matched fraction of the disturbance along the tool pass")
print("=" * 78)
print(f"  feed = {v_feed*1e3:.2f} mm/s, so a 0.5 s simulation covers "
      f"{v_feed*0.5*1e3:.2f} mm of the {LP*1e3:.0f} mm pass "
      f"({v_feed*0.5/LP*100:.1f} %)")
print(f"  full pass takes {LP/v_feed:.1f} s")
print()
print(f"  baseline patch: x in [0, 20] mm, z in [0, 60] mm")
print(f"  {'tool x [mm]':>12} {'cos(theta)':>12} {'cancellable':>13}")
for xm in (0, 2.45, 10, 20, 30, 50, 70, 100):
    i = int(np.clip(xm / 1000 / LP * 2000, 0, 2000))
    print(f"  {x[i]*1e3:12.2f} {cos_base[i]:12.3f} {cos_base[i]*100:12.1f} %")

print()
in_sim = cos_base[:int(2.45e-3 / LP * 2000) + 1]
print(f"  mean over the 2.45 mm actually simulated : {in_sim.mean():.3f}")
print(f"  mean over the full 100 mm pass          : {cos_base.mean():.3f}")
print(f"  minimum over the full pass              : {cos_base.min():.3f} "
      f"at x = {x[np.argmin(cos_base)]*1e3:.1f} mm")
print()
print("  => a feedforward branch tuned in the first 2.45 mm operates where")
print("     the actuator is nearly aligned with the disturbance; over the")
print("     rest of the pass most of the disturbance is simply not reachable")
print("     by this actuator.")

print()
print("=" * 78)
print(" Does a different SINGLE patch placement fix it?")
print("=" * 78)
print(f"  {'patch x-window [mm]':>22} {'mean cos':>10} {'min cos':>9} "
      f"{'worst x [mm]':>13}")
cands = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (0, 100),
         (30, 70)]
best = None
for x1, x2 in cands:
    p = PlateModel(LP, HP, BP, RHO, E_AL, NU, N1=30, N2=24, n_modes=3,
                   zeta_modes=ZETA, verbose=False)
    p.precompute_Dp(zp_pos=HP - AP / 2, n_pos=2001)
    p.set_observation(x_obs=LP, z_obs=HP)
    p.add_piezo_patch(x1 / 1000, x2 / 1000, 0, 0.060, D31, H_PA, E_PE, NU_PE)
    c = alignment(p)
    print(f"  {f'[{x1}, {x2}]':>22} {c.mean():10.3f} {c.min():9.3f} "
          f"{x[np.argmin(c)]*1e3:13.1f}")
    if best is None or c.min() > best[1]:
        best = ((x1, x2), c.min())
print(f"  best single placement by worst-case: {best[0]} mm, "
      f"min cos = {best[1]:.3f}")

print()
print("=" * 78)
print(" Two patches: can a pair keep the whole pass reachable?")
print("=" * 78)
print("  With two independent inputs the reachable subspace is span(H1, H2),")
print("  so the matched fraction is the norm of the projection of Dp onto")
print("  that plane rather than onto a single line.")
print()


def two_patch_alignment(w1, w2):
    ps = []
    for x1, x2 in (w1, w2):
        p = PlateModel(LP, HP, BP, RHO, E_AL, NU, N1=30, N2=24, n_modes=3,
                       zeta_modes=ZETA, verbose=False)
        p.precompute_Dp(zp_pos=HP - AP / 2, n_pos=2001)
        p.set_observation(x_obs=LP, z_obs=HP)
        p.add_piezo_patch(x1 / 1000, x2 / 1000, 0, 0.060,
                          D31, H_PA, E_PE, NU_PE)
        ps.append(p)
    Hm = np.column_stack([p.H_Pe_modal for p in ps])
    Q, _ = np.linalg.qr(Hm)                      # orthonormal basis of span
    Dn = ps[0].Dp_array / np.linalg.norm(ps[0].Dp_array, axis=0, keepdims=True)
    return np.linalg.norm(Q.T @ Dn, axis=0)


print(f"  {'patch pair [mm]':>30} {'mean':>8} {'min':>8}")
for w1, w2 in (((0, 20), (80, 100)), ((0, 25), (75, 100)),
               ((0, 30), (70, 100)), ((10, 30), (70, 90))):
    c2 = two_patch_alignment(w1, w2)
    print(f"  {f'{list(w1)} + {list(w2)}':>30} {c2.mean():8.3f} {c2.min():8.3f}")

print()
print("  Reference: single baseline patch  mean = "
      f"{cos_base.mean():.3f}  min = {cos_base.min():.3f}")
