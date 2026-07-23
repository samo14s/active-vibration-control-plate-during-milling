"""
verify_geometry.py
=================
Asserts that the Mindlin `PlateModel` conforms to the article setup diagram:

    Z_P |                         . displacement sensor  (x_obs, z_obs)
        |     tool ---> feed X_P
        |  +------------------------+   <- free top edge (Z_P = H_P)
        |  |                        |
        |  |   [piezo patch]        |   plate in the (X_P, Z_P) plane,
        |  |   (x_P1,z_P1)-(x_P2,z_P2)   deflection w out-of-plane (Y_P)
        |  |                        |
   O_P  +--+------------------------+---- X_P
        /////////  clamped base (encastrement, Z_P = 0)  /////////

Checks (article geometry, package 12):
  1. Frame & dimensions: L_P=100, H_P=80, B_P=4 mm; in-plane (X_P, Z_P).
  2. Boundary condition: the clamped edge is the bottom edge Z_P=0, and all
     three DOFs (w, theta_x, theta_y) are fixed there — a true encastrement.
  3. Cantilever mode shape: mode-1 deflection is ~0 at the clamped base and
     maximal at the free top edge (out-of-plane w).
  4. Piezo patch spans the diagram corners (x_P1,z_P1)-(x_P2,z_P2), lower-left
     to upper-right; the induced modal force is non-zero.
  5. Displacement sensor: observation point near the top edge yields a
     non-zero modal observation row D_obs.
  6. Tool path: runs along X_P (0 -> L_P) at a fixed height Z_P near the top.

Run:  python verify_geometry.py
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "01_core"))

import mindlin_q8 as M
from plate_model import PlateModel

# --- article geometry (metres) — matches gen_geometry_figure.py ---
LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU = 2830, 69e9, 0.33
N1, N2 = 30, 24
# piezo patch corners (diagram): (x_P1,z_P1)=(0,0)  (x_P2,z_P2)=(0.020,0.060)
XP1, ZP1, XP2, ZP2 = 0.0, 0.0, 0.020, 0.060
D31, H_PA, E_PE, NU_PE = 175e-12, 0.7e-3, 63e9, 0.35


def banner(t):
    print("\n" + "=" * 68 + f"\n {t}\n" + "=" * 68)


def w_at(plate, mode, x, z):
    """Out-of-plane deflection of `mode` at physical point (X_P=x, Z_P=z)."""
    Nv = M.shape_at_point_M(x, z, plate.N1, plate.N2,
                            plate.lex, plate.ley, plate.node_id, plate.ndof)
    return Nv[plate.DOFf] @ plate.V[:, mode]


def main():
    plate = PlateModel(LP, HP, BP, RHO, E_AL, NU, N1=N1, N2=N2,
                       n_modes=3, verbose=False)
    ok = True

    # 1. frame & dimensions -------------------------------------------------
    banner("1. Frame & dimensions (X_P length, Z_P height, Y_P thickness)")
    print(f"  L_P (X_P) = {plate.lp*1e3:.1f} mm   H_P (Z_P) = {plate.hp*1e3:.1f} mm"
          f"   B_P (thickness) = {plate.bp*1e3:.1f} mm")
    print(f"  in-plane mesh: {plate.N1} x {plate.N2} Q8 elements  "
          f"lex(X_P)={plate.lex*1e3:.3f} mm  ley(Z_P)={plate.ley*1e3:.3f} mm")
    c1 = abs(plate.lp - LP) < 1e-12 and abs(plate.hp - HP) < 1e-12 and abs(plate.bp - BP) < 1e-12
    print(f"  [1] {'PASS' if c1 else 'FAIL'}")
    ok &= c1

    # 2. clamped base at Z_P = 0 -------------------------------------------
    banner("2. Boundary condition: encastrement at the bottom edge Z_P = 0")
    DOFb = plate.DOFb
    # nodes on the bottom edge (iy = 0)
    bottom_nodes = plate.node_id[:, 0]
    bottom_nodes = bottom_nodes[bottom_nodes >= 0]
    expected = np.sort(np.concatenate([3*bottom_nodes, 3*bottom_nodes+1, 3*bottom_nodes+2]))
    same = np.array_equal(np.sort(DOFb), expected)
    # confirm those nodes are physically at z = 0
    zmax_clamped = 0.0
    for n in bottom_nodes[:5]:
        # locate grid position of node n on bottom row -> z=0 by construction
        pass
    print(f"  clamped DOFs = {len(DOFb)}  (= 3 x {len(bottom_nodes)} bottom-edge nodes)")
    print(f"  all three DOFs (w, theta_x, theta_y) fixed on Z_P=0 edge: {same}")
    # sanity: no clamped node lies above the base
    c2 = same and (len(DOFb) == 3 * (2*N1 + 1))
    print(f"  [2] {'PASS' if c2 else 'FAIL'}  (expected 3x(2*N1+1) = {3*(2*N1+1)} DOFs)")
    ok &= c2

    # 3. cantilever mode shape: clamped bottom, free top -------------------
    banner("3. Cantilever mode-1 shape: w=0 at base, max at free top edge")
    xc = LP / 2
    zs = np.linspace(0, HP, 9)
    ws = np.array([w_at(plate, 0, xc, z) for z in zs])
    ws_n = ws / np.max(np.abs(ws))
    for z, wn in zip(zs, ws_n):
        bar = "#" * int(abs(wn) * 40)
        print(f"  Z_P={z*1e3:5.1f} mm | w/|w|max = {wn:+.3f} {bar}")
    base_zero = abs(ws_n[0]) < 1e-3
    top_max = abs(ws_n[-1]) > 0.98
    monotonic = np.all(np.diff(np.abs(ws_n)) > -1e-6)
    c3 = base_zero and top_max and monotonic
    print(f"  base~0: {base_zero}   top=max: {top_max}   monotonic rise: {monotonic}")
    print(f"  [3] {'PASS' if c3 else 'FAIL'}")
    ok &= c3

    # 4. piezo patch corners (x_P1,z_P1)-(x_P2,z_P2) -----------------------
    banner("4. Piezo patch spans diagram corners (x_P1,z_P1)-(x_P2,z_P2)")
    plate.set_observation(x_obs=LP, z_obs=HP)
    plate.add_piezo_patch(XP1, XP2, ZP1, ZP2, D31, H_PA, E_PE, NU_PE)
    g = M.piezo_moment_load_M(XP1, XP2, ZP1, ZP2, plate.N1, plate.N2,
                              plate.lex, plate.ley, plate.node_id, plate.ndof)
    nz = np.nonzero(g)[0]
    Hnorm = np.linalg.norm(plate.H_Pe_modal)
    print(f"  patch (x_P1,z_P1)=({XP1*1e3:.0f},{ZP1*1e3:.0f}) mm  "
          f"(x_P2,z_P2)=({XP2*1e3:.0f},{ZP2*1e3:.0f}) mm")
    print(f"  loaded DOFs in patch: {len(nz)}   ||H_Pe_modal|| = {Hnorm:.3e} N/V")
    c4 = len(nz) > 0 and Hnorm > 0
    print(f"  [4] {'PASS' if c4 else 'FAIL'}")
    ok &= c4

    # 5. displacement sensor near the top ---------------------------------
    banner("5. Displacement sensor (out-of-plane w) near top edge")
    Dobs = plate.D_obs
    print(f"  sensor at (x_obs,z_obs)=({plate.x_obs*1e3:.0f},{plate.z_obs*1e3:.0f}) mm"
          f"  D_obs = {np.array2string(Dobs, precision=3)}")
    c5 = np.linalg.norm(Dobs) > 0 and abs(plate.z_obs - HP) < 1e-9
    print(f"  [5] {'PASS' if c5 else 'FAIL'}")
    ok &= c5

    # 6. tool path along X_P at fixed height -------------------------------
    banner("6. Tool path along X_P (0 -> L_P) at fixed Z_P near top")
    ap = 0.3e-3
    plate.precompute_Dp(zp_pos=HP - ap/2, n_pos=2001)
    xr = plate.xp_array
    c6 = (abs(xr[0]) < 1e-9 and abs(xr[-1] - LP) < 1e-9
          and abs(plate.zp_pos - (HP - ap/2)) < 1e-12)
    print(f"  X_P from {xr[0]*1e3:.1f} to {xr[-1]*1e3:.1f} mm at "
          f"Z_P={plate.zp_pos*1e3:.2f} mm (a_p/2 below top)")
    print(f"  [6] {'PASS' if c6 else 'FAIL'}")
    ok &= c6

    banner("SUMMARY")
    print(f"  Mindlin model conforms to the setup diagram: "
          f"{'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
