"""
verify_mindlin.py
=================
Structural validation of the literal Mindlin Q8 port (mindlin_q8.py):

  1. Element-level sanity (partition of unity, symmetry, rigid-body / hourglass
     mode count, mass consistency).
  2. CCCC fully-clamped thin square plate vs Leissa frequency parameter.
  3. Cantilever AL6061 plate (article geometry) vs beam-strip estimate / article.
  4. Mesh convergence of the cantilever fundamental frequency.
  5. Thermal-load smoke test (Thermal_stress_M port).

No MATLAB required. Run:  python verify_mindlin.py
"""
import os
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

# --- resolve package imports regardless of CWD ---
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "01_core"))

import mindlin_q8 as M
from plate_model import PlateModel


def banner(t):
    print("\n" + "=" * 68 + f"\n {t}\n" + "=" * 68)


# ------------------------------------------------------------------
# 1. Element-level sanity
# ------------------------------------------------------------------
def test_element():
    banner("1. Element-level checks (literal port sanity)")
    ok = True
    for (xi, eta) in [(-0.3, 0.6), (0.1, -0.9), (0.5, 0.5)]:
        _, Nr, dN = M.shape_function_M(xi, eta)
        s, sdx, sdy = Nr.sum(), dN[0].sum(), dN[1].sum()
        print(f"  (xi,eta)=({xi:+.1f},{eta:+.1f})  sum(N)={s:.6f}  "
              f"sum(dN/dxi)={sdx:+.1e}  sum(dN/deta)={sdy:+.1e}")
        ok &= abs(s - 1) < 1e-10 and abs(sdx) < 1e-10 and abs(sdy) < 1e-10

    Ke = M.stiffness_matrix_M(2.1e11, 0.3, 5/6, 0.02, 0.02, 0.001)
    Me = M.mass_matrix_M(7800, 0.02, 0.02, 0.001)
    sym = np.allclose(Ke, Ke.T) and np.allclose(Me, Me.T)
    print(f"\n  Ke, Me symmetric: {sym}")

    evK = np.sort(np.linalg.eigvalsh(Ke))
    # true zero modes: absolute floor (element stiffness scale ~ evK[4])
    scale = evK[4] if evK.size > 4 else evK.max()
    n_zero = int(np.sum(np.abs(evK) < 1e-6 * scale))
    print(f"  Ke smallest 6 eigenvalues: {np.array2string(evK[:6], precision=3)}")
    print(f"  -> zero-energy modes of single free element: {n_zero} "
          f"(expected 4 = 3 rigid-body + 1 hourglass, uniform-reduced Q8)")
    ok &= sym and (n_zero == 4)
    print(f"\n  [1] {'PASS' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------
# Standalone assembler with selectable BC (analytical validation)
# ------------------------------------------------------------------
def assemble(Lx, Ly, h, E, nu, rho, N1, N2, kappa=5/6):
    lex, ley = Lx / N1, Ly / N2
    node_id, ntot = M.build_node_map(N1, N2)
    ndof = 3 * ntot
    Ke = M.stiffness_matrix_M(E, nu, kappa, lex, ley, h)
    Me = M.mass_matrix_M(rho, lex, ley, h)
    nn = N1 * N2 * 576
    rows = np.zeros(nn, np.int64); cols = np.zeros(nn, np.int64)
    Kd = np.zeros(nn); Md = np.zeros(nn); idx = 0
    for J in range(1, N2 + 1):
        for I in range(1, N1 + 1):
            D = M.elem_dofs_M(I, J, node_id)
            for a in range(24):
                for b in range(24):
                    rows[idx] = D[a]; cols[idx] = D[b]
                    Kd[idx] = Ke[a, b]; Md[idx] = Me[a, b]; idx += 1
    Kg = csr_matrix((Kd, (rows, cols)), shape=(ndof, ndof))
    Mg = csr_matrix((Md, (rows, cols)), shape=(ndof, ndof))
    return Kg, Mg, node_id


def solve_modes(Kg, Mg, DOFb, nmodes=6):
    DOFf = np.setdiff1d(np.arange(Kg.shape[0]), DOFb)
    Kf = Kg[DOFf, :][:, DOFf].tocsc()
    Mf = Mg[DOFf, :][:, DOFf].tocsc()
    w2, _ = eigsh(Kf, k=nmodes, M=Mf, sigma=0, which='LM')
    return np.sqrt(np.abs(np.sort(np.real(w2))))


# ------------------------------------------------------------------
# 2. CCCC thin square plate
# ------------------------------------------------------------------
def test_cccc():
    banner("2. CCCC clamped square plate vs Leissa lambda (thin-plate limit)")
    a, h, E, nu, rho = 1.0, 0.005, 2.0e11, 0.3, 8000.0
    N1 = N2 = 16
    Kg, Mg, nid = assemble(a, a, h, E, nu, rho, N1, N2)
    db = np.unique(np.concatenate([M.clamped_edge_dofs(nid, e)
                   for e in ("bottom", "top", "left", "right")]))
    om = solve_modes(Kg, Mg, db, 6)
    D = E * h**3 / (12 * (1 - nu**2))
    lam = om * a**2 * np.sqrt(rho * h / D)
    ref = np.array([35.99, 73.41, 73.41, 108.27, 131.64, 132.24])
    print(f"  mesh {N1}x{N2}, h/a={h/a}")
    print(f"  lambda  : {np.array2string(lam, precision=2)}")
    print(f"  Leissa  : {np.array2string(ref, precision=2)}")
    err = abs(lam[0] - ref[0]) / ref[0] * 100
    print(f"  lambda_1 err = {err:+.2f}%")
    ok = err < 1.0 and np.all(np.abs(lam - ref) / ref < 0.02)
    print(f"\n  [2] {'PASS' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------
# 3. Cantilever AL6061 (article geometry)
# ------------------------------------------------------------------
def test_cantilever():
    banner("3. Cantilever AL6061 plate (article geometry) — mode 1 ~ 521 Hz")
    LP, HP, BP = 0.100, 0.080, 0.004
    RHO, E_AL, NU = 2830, 69e9, 0.33
    plate = PlateModel(LP, HP, BP, RHO, E_AL, NU, N1=30, N2=24,
                       n_modes=6,
                       zeta_modes=[0.0031, 0.0017, 0.0027, 0, 0, 0],
                       verbose=False)
    Dp = E_AL * BP**3 / (12 * (1 - NU**2))
    f1_beam = (1.875**2 / (2*np.pi)) * np.sqrt(Dp / (RHO*BP)) / HP**2
    print(f"  FEM frequencies (Hz): {np.array2string(plate.freq_n, precision=1)}")
    print(f"  beam-strip f1 = {f1_beam:.1f} Hz, article ~521 Hz, "
          f"FEM f1 = {plate.freq_n[0]:.1f} Hz")
    ok = abs(plate.freq_n[0] - 521) / 521 < 0.05
    print(f"\n  [3] {'PASS' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------
# 4. Mesh convergence
# ------------------------------------------------------------------
def test_convergence():
    banner("4. Mesh convergence of cantilever f1")
    LP, HP, BP = 0.100, 0.080, 0.004
    RHO, E_AL, NU = 2830, 69e9, 0.33
    f1s = []
    for (n1, n2) in [(10, 8), (20, 16), (30, 24), (40, 32)]:
        p = PlateModel(LP, HP, BP, RHO, E_AL, NU, N1=n1, N2=n2,
                       n_modes=3, verbose=False)
        f1s.append(p.freq_n[0])
        print(f"  mesh {n1:2d}x{n2:2d}: f = {np.array2string(p.freq_n, precision=1)}")
    ok = abs(f1s[-1] - f1s[-2]) / f1s[-1] < 1e-3
    print(f"\n  [4] {'PASS' if ok else 'FAIL'} (last-two-mesh drift "
          f"{abs(f1s[-1]-f1s[-2])/f1s[-1]*100:.3f}%)")
    return ok


# ------------------------------------------------------------------
# 5. Thermal load smoke test
# ------------------------------------------------------------------
def test_thermal():
    banner("5. Thermal load (Thermal_stress_M port) smoke test")
    F = M.thermal_stress_M(1.3e11, 0.27, 2.8e-6, 0.02, 0.02, 500e-6, 100.0)
    finite = np.all(np.isfinite(F))
    # thermal moment load acts on rotational DOFs; net transverse force = 0
    net_w = F[0::3].sum()
    print(f"  F shape={F.shape}, finite={finite}, ||F||={np.linalg.norm(F):.3e}, "
          f"sum(F_w)={net_w:+.2e}")
    ok = finite and abs(net_w) < 1e-9
    print(f"\n  [5] {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = [test_element(), test_cccc(), test_cantilever(),
               test_convergence(), test_thermal()]
    banner("SUMMARY")
    names = ["element", "CCCC", "cantilever", "convergence", "thermal"]
    for n, r in zip(names, results):
        print(f"  {n:12s} : {'PASS' if r else 'FAIL'}")
    print(f"\n  {'ALL PASS' if all(results) else 'SOME FAILED'}")
    sys.exit(0 if all(results) else 1)
