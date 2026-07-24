"""
verify_leissa_cantilever.py
===========================
ROADMAP item 24 — verify the boundary condition that is actually used.

The starting package verifies the CCCC (fully clamped) plate against Leissa
to 0.02 %, which is a good check of the element. But the workpiece in this
study is a CANTILEVER (CFFF: one edge clamped, three free), and CFFF is the
harder case for a plate element: the free edges carry the shear-locking and
corner-condition difficulties that a fully clamped boundary hides. A reviewer
will ask for the benchmark in the configuration actually used.

BENCHMARK
---------
Leissa, "Vibration of Plates", NASA SP-160 (1969), square cantilever plate,
nu = 0.3. The non-dimensional frequency parameter is

    lambda = omega a^2 sqrt(rho h / D),    D = E h^3 / (12 (1 - nu^2))

CONFIDENCE, stated honestly: lambda_1 = 3.492 and lambda_2 = 8.525 were
independently confirmed during the literature survey. The higher values
(21.429, 27.331, 31.111, 54.443) are the commonly quoted set but could NOT be
independently confirmed, so they are marked UNVERIFIED here and must be
checked against the original before being tabulated in a manuscript.

Thin-plate values are the Kirchhoff limit, so a Mindlin element must be run
at a small thickness ratio to approach them; the convergence of the error
with h/a is itself part of the check.
"""
import numpy as np
import _bootstrap  # noqa: F401

from evolving_plate import EvolvingPlateModel

# Leissa SP-160, square cantilever (CFFF), nu = 0.3
LEISSA = [
    (3.492, True),    # confirmed
    (8.525, True),    # confirmed
    (21.429, False),  # commonly quoted, NOT independently confirmed
    (27.331, False),
    (31.111, False),
    (54.443, False),
]

A = 1.0            # square plate side [m]
E = 200e9
NU = 0.3           # Leissa's value, not the AL6061 0.33
RHO = 8000.0

ok = True

print("=" * 78)
print(" Leissa CFFF (cantilever) benchmark — the boundary condition used here")
print("=" * 78)
print("  lambda = omega a^2 sqrt(rho h / D),  D = E h^3 / (12 (1 - nu^2))")
print("  Leissa's values are the thin-plate (Kirchhoff) limit, so a Mindlin")
print("  element must approach them from below as h/a -> 0.")
print()

n_modes = 6
print(f"  {'h/a':>8} {'mesh':>8} " +
      " ".join(f"{'l'+str(i+1):>9}" for i in range(n_modes)))

results = {}
for ratio, N in ((0.02, 24), (0.01, 24), (0.005, 24), (0.002, 24)):
    h = ratio * A
    D = E * h ** 3 / (12 * (1 - NU ** 2))
    p = EvolvingPlateModel(A, A, h, RHO, E, NU, N1=N, N2=N, n_modes=n_modes,
                           zeta_modes=[0.0] * n_modes, verbose=False)
    lam = p.omega_n * A ** 2 * np.sqrt(RHO * h / D)
    results[ratio] = lam
    print(f"  {ratio:8.3f} {f'{N}x{N}':>8} " +
          " ".join(f"{v:9.3f}" for v in lam))

print()
print("  mesh convergence at h/a = 0.005:")
for N in (12, 18, 24, 32):
    h = 0.005 * A
    D = E * h ** 3 / (12 * (1 - NU ** 2))
    p = EvolvingPlateModel(A, A, h, RHO, E, NU, N1=N, N2=N, n_modes=n_modes,
                           zeta_modes=[0.0] * n_modes, verbose=False)
    lam = p.omega_n * A ** 2 * np.sqrt(RHO * h / D)
    print(f"  {'':8} {f'{N}x{N}':>8} " + " ".join(f"{v:9.3f}" for v in lam))

print()
print("=" * 78)
print(" Error against Leissa at the thinnest plate (h/a = 0.002, 24x24)")
print("=" * 78)
lam = results[0.002]
print(f"  {'mode':>5} {'FEM':>10} {'Leissa':>10} {'error':>9}  status")
for i in range(n_modes):
    ref, confirmed = LEISSA[i]
    err = (lam[i] - ref) / ref * 100
    tag = "confirmed" if confirmed else "UNVERIFIED reference"
    # only the confirmed references are allowed to gate the test
    good = abs(err) < 2.0
    if confirmed:
        ok &= good
    print(f"  {i+1:5d} {lam[i]:10.3f} {ref:10.3f} {err:+8.2f}%  "
          f"{tag}{'' if good else '   <-- CHECK'}")

print()
print("  Only modes 1-2 gate this test; the higher Leissa values could not be")
print("  independently confirmed and must be checked in the original before")
print("  being quoted. They are printed so the comparison is visible, not")
print("  hidden.")

print()
print("=" * 78)
print(f" RESULT: {'PASS' if ok else 'FAIL'} on the confirmed references")
print("=" * 78)
raise SystemExit(0 if ok else 1)
