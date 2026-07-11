"""
validate_von_karman.py
======================
Validation of the von Kármán reduced-order model (01_core/von_karman_rom.py).

V1 : membrane stiffness is symmetric positive-definite after BC.
V2 : the modal nonlinear force is EXACTLY cubic (fit residual → 0).
V3 : linear reduction — F_nl(q) → 0 faster than linearly as q → 0
     (cubic scaling F_nl(εq) = ε³ F_nl(q)).
V4 : LITERATURE benchmark — a fully clamped, in-plane immovable SQUARE
     plate must reproduce the classical hardening backbone
        ω_nl/ω_l ≈ 1.1–1.22 at A/h = 1
     (single-mode harmonic balance; Chia 1980, Yamaki).
V5 : the cantilever milling plate hardens (ω_nl/ω_l > 1) but weakly.

Exit code 1 on any failure.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _d in ("01_core",):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from plate_model import PlateModel
from von_karman_rom import VonKarmanROM

FAIL = []


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAIL.append(name)


# ================= V4 : clamped square plate benchmark =================
print("=== V4: clamped square plate backbone vs literature ===")
# square plate, all edges clamped, in-plane immovable
Lsq = 0.20
h_sq = 0.002
plate_sq = PlateModel(Lsq, Lsq, h_sq, 2830, 69e9, 0.33,
                      N1=24, N2=24, n_modes=3, verbose=False)
# clamp ALL four edges for the transverse problem (override cantilever BC)
n1, n2, ndof = plate_sq.n1, plate_sq.n2, plate_sq.ndof


def edge_dofs_all(plate):
    n1, n2 = plate.n1, plate.n2
    dofs = []
    for I in range(1, n1+1):
        for J in (1, n2):
            base = 3*(J-1)*n1 + 3*(I-1)
            dofs += [base, base+1, base+2]
    for J in range(1, n2+1):
        for I in (1, n1):
            base = 3*(J-1)*n1 + 3*(I-1)
            dofs += [base, base+1, base+2]
    return np.unique(dofs)


# rebuild modal basis with all-edges-clamped BC
DOFb = edge_dofs_all(plate_sq)
DOFf = np.setdiff1d(np.arange(ndof), DOFb)
from scipy.sparse.linalg import eigsh
Kff = plate_sq.Kg[DOFf, :][:, DOFf].tocsc()
Mff = plate_sq.Mg[DOFf, :][:, DOFf].tocsc()
v0 = np.ones(Kff.shape[0])
w2, V = eigsh(Kff, k=3, M=Mff, sigma=0, which='LM', v0=v0)
order = np.argsort(np.real(w2))
w2 = np.real(w2[order]); V = np.real(V[:, order])
for k in range(3):
    V[:, k] /= np.sqrt(V[:, k] @ (Mff @ V[:, k]))
plate_sq.DOFf = DOFf; plate_sq.DOFb = DOFb; plate_sq.V = V
plate_sq.omega_n = np.sqrt(w2); plate_sq.freq_n = plate_sq.omega_n/(2*np.pi)

rom_sq = VonKarmanROM(plate_sq, membrane_bc="immovable", verbose=True)
check("V2 cubic exactness (clamped)", rom_sq.cubic_residual < 1e-8,
      f"relative fit residual = {rom_sq.cubic_residual:.2e}")

amp, ratio, g = rom_sq.backbone_single_mode(mode=0, amp_ratio=[0.2, 0.5, 1.0])
print(f"    ω_nl/ω_l : A/h=0.2 → {ratio[0]:.4f}, "
      f"0.5 → {ratio[1]:.4f}, 1.0 → {ratio[2]:.4f}  (Γ_1111={g:.3e})")
check("V4 hardening at A/h=1 in literature range [1.08,1.30]",
      1.08 <= ratio[2] <= 1.30,
      f"ω_nl/ω_l(A/h=1) = {ratio[2]:.3f}")
check("V4 monotonic hardening", ratio[0] < ratio[1] < ratio[2],
      f"{ratio[0]:.3f} < {ratio[1]:.3f} < {ratio[2]:.3f}")

# ================= V1/V3 : membrane SPD + cubic scaling =================
print("\n=== V1/V3: membrane SPD and cubic scaling ===")
from scipy.linalg import lu_solve
Kmff = rom_sq.Km_full[np.ix_(rom_sq.m_free, rom_sq.m_free)]
eigmin = np.linalg.eigvalsh(Kmff).min()
check("V1 membrane K_m SPD", eigmin > 0,
      f"min eig(K_m,free) = {eigmin:.3e}")

q0 = np.array([1.0, 0.5, -0.3]) * (h_sq/np.max(np.abs(rom_sq.Phi_full)))*0.3
F1 = rom_sq.force(q0)
F2 = rom_sq.force(0.5*q0)
scale = np.linalg.norm(F2)/max(np.linalg.norm(F1), 1e-30)
check("V3 cubic scaling F(q/2)=F(q)/8", abs(scale - 0.125) < 1e-6,
      f"||F(q/2)||/||F(q)|| = {scale:.6f} (expect 0.125)")

# analytic jacobian vs finite difference
J = rom_sq.jacobian(q0)
Jfd = np.zeros_like(J)
eps = 1e-9 * (h_sq/np.max(np.abs(rom_sq.Phi_full)))
for s in range(3):
    dq = np.zeros(3); dq[s] = eps
    Jfd[:, s] = (rom_sq.force(q0+dq) - rom_sq.force(q0-dq))/(2*eps)
jerr = np.linalg.norm(J - Jfd)/max(np.linalg.norm(J), 1e-30)
check("V3 analytic jacobian vs FD", jerr < 1e-5,
      f"relative error = {jerr:.2e}")

# ================= V5 : cantilever milling plate =================
print("\n=== V5: cantilever milling plate (AL6061) ===")
for h_mm in (4.0, 1.0):
    plate = PlateModel(0.100, 0.080, h_mm*1e-3, 2830, 69e9, 0.33,
                       N1=30, N2=24, n_modes=3,
                       zeta_modes=[0.0031, 0.0017, 0.0027], verbose=False)
    rom = VonKarmanROM(plate, membrane_bc="cantilever", verbose=False)
    amp, ratio, g = rom.backbone_single_mode(mode=0, amp_ratio=[0.5, 1.0])
    check(f"V5 hardening h={h_mm}mm (Γ>0, ω_nl/ω_l>1)",
          g > 0 and ratio[1] > 1.0,
          f"f1={plate.freq_n[0]:.0f}Hz, Γ_1111={g:.3e}, "
          f"ω_nl/ω_l(A/h=1)={ratio[1]:.4f}, resid={rom.cubic_residual:.1e}")

print(f"\n{'ALL PASS' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
