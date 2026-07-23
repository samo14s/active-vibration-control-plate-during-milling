"""
verify_inprocess.py
==================
Checks for the in-process material-removal model (InProcessPlate):

  1. Uniform consistency  : with no material removed it equals PlateModel.
  2. Pass idempotency     : each element is thinned at most once per pass, so
                            re-calling machine_to at the same tool position does
                            not double-thin.
  3. Mass/thickness       : removal reduces the mean thickness monotonically.
  4. Physical sign        : thinning the top band (near the free tip) RAISES the
                            cantilever fundamental frequency (tip-mass removal).
  5. Fixed-basis ROM      : reduced_on_basis(V0) tracks the exact instantaneous
                            fundamental frequency to good accuracy.

Run:  python verify_inprocess.py
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "01_core"))

from plate_model import PlateModel
from inprocess_plate import InProcessPlate

LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU = 2830, 69e9, 0.33
ZE = [0.0031, 0.0017, 0.0027]


def banner(t):
    print("\n" + "=" * 68 + f"\n {t}\n" + "=" * 68)


def fresh(n=3):
    zeta = (ZE + [0.003] * n)[:n]     # pad damping if more modes than defaults
    return InProcessPlate(LP, HP, BP, RHO, E_AL, NU, N1=30, N2=24,
                          n_modes=n, zeta_modes=zeta, verbose=False)


def main():
    ok = True

    banner("1. Uniform consistency: InProcessPlate == PlateModel")
    base = PlateModel(LP, HP, BP, RHO, E_AL, NU, N1=30, N2=24, n_modes=3,
                      zeta_modes=ZE, verbose=False)
    ip = fresh()
    c1 = np.allclose(base.freq_n, ip.freq_n, rtol=1e-9)
    print(f"  base = {np.round(base.freq_n,3)}")
    print(f"  ip   = {np.round(ip.freq_n,3)}   match={c1}")
    print(f"  [1] {'PASS' if c1 else 'FAIL'}"); ok &= c1

    banner("2. Pass idempotency (machine_to thins each element once per pass)")
    p = fresh(); p.new_pass()
    p.machine_to(LP, a_p=0.030, a_e=0.5e-3); h1 = p.h_elem.copy()
    p.machine_to(LP, a_p=0.030, a_e=0.5e-3); h2 = p.h_elem.copy()   # same pass again
    c2 = np.allclose(h1, h2)
    print(f"  thickness unchanged on 2nd machine_to in same pass: {c2}")
    p.new_pass(); p.machine_to(LP, a_p=0.030, a_e=0.5e-3); h3 = p.h_elem.copy()
    c2b = (h3.min() < h1.min() - 1e-9)     # a NEW pass does thin further
    print(f"  a NEW pass thins further: {c2b}  (min h {h1.min()*1e3:.3f}->{h3.min()*1e3:.3f} mm)")
    print(f"  [2] {'PASS' if (c2 and c2b) else 'FAIL'}"); ok &= (c2 and c2b)

    banner("3. Removal reduces mean thickness monotonically")
    p = fresh(); p.new_pass()
    means = [p.h_elem.mean()]
    for x in np.linspace(LP/5, LP, 5):
        p.machine_to(x, a_p=0.030, a_e=0.4e-3)
        means.append(p.h_elem.mean())
    means = np.array(means)
    c3 = np.all(np.diff(means) <= 1e-15) and means[-1] < means[0]
    print(f"  mean thickness (mm): {np.round(means*1e3,4)}")
    print(f"  [3] {'PASS' if c3 else 'FAIL'}"); ok &= c3

    banner("4. Physical sign: thinning the free-tip band raises mode 1")
    p = fresh(); f0 = p.freq_n[0]
    p.new_pass(); p.machine_to(LP, a_p=0.030, a_e=0.8e-3); p.rebuild()
    f1 = p.instantaneous_modes(3)[0][0]
    c4 = f1 > f0
    print(f"  mode 1: {f0:.1f} -> {f1:.1f} Hz ({(f1-f0)/f0*100:+.2f}%), rises={c4}")
    print(f"  [4] {'PASS' if c4 else 'FAIL'}"); ok &= c4

    banner("5. Fixed-basis ROM tracks the exact fundamental frequency")
    p = fresh(4); V0 = p.V.copy()
    p.new_pass(); p.machine_to(LP, a_p=0.030, a_e=0.5e-3); p.rebuild()
    fe = p.instantaneous_modes(4)[0]
    Mr, Kr = p.reduced_on_basis(V0)
    fr = np.sort(np.sqrt(np.abs(np.linalg.eigvals(np.linalg.solve(Mr, Kr))))) / (2*np.pi)
    err1 = abs(fr[0]-fe[0])/fe[0]*100
    c5 = err1 < 1.0
    print(f"  exact f1={fe[0]:.2f} Hz  ROM f1={fr[0]:.2f} Hz  err={err1:.3f}%")
    print(f"  [5] {'PASS' if c5 else 'FAIL'}"); ok &= c5

    banner("SUMMARY")
    print(f"  InProcessPlate material-removal model: "
          f"{'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
