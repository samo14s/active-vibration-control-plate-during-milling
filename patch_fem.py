"""
Crawley-de Luis piezoelectric patch on the Mindlin plate.

Reconstructed (again -- the file keeps falling out of the archive, see
RESULTS.md).  Geometry and every free parameter are pinned to the documented
plant anchors rather than guessed; the reconstruction reproduces

    open loop, corner tool : peak 409.6 um/N, static 7.045 um/N
    PPF g = [0.092, 0.055] : a_lim x12.25, 148.1 V/N, zeta1 24.33 %,
                             zeta2 4.35 %, static ratio 0.861

on the 40 x 32 mesh with 6 modes and zeta = 0.5 %.

MODEL
  * one PZT patch, 60 x 20 x 0.7 mm, bonded at x in [5, 25] mm with its long
    side running UP the cantilever, y in [0, 60] mm from the clamp (this
    orientation -- not the spanwise strip -- is what reproduces the anchors:
    it integrates mode-1 root curvature over 3/4 of the free length while the
    mode-4 curvature partly cancels, giving the documented modal coupling
    pattern);
  * covered elements get the patch bending stiffness about the plate
    midplane, E_p/(1-nu_p^2) * ((h/2+hp)^3 - (h/2)^3)/3, its shear stiffness
    and its translational + rotary mass (perfectly bonded, no bond-layer
    compliance);
  * per volt the patch applies the Crawley-de Luis equi-biaxial bending
    moment  m_p = E_p/(1-nu_p) * d31 * (h+hp)/2  per unit length of the
    patch boundary, realised as the consistent element load
    f = int Bb^T [m_p, m_p, 0]^T dA  (interior contributions cancel, the
    boundary line moment remains);
  * the collocated sensor is the same patch pair read through the reciprocal
    channel, cS = bmod (see avc_plant.py).

d31 is calibrated (-181.3 pm/V, inside the PZT-5A datasheet spread) so that
the PPF baseline lands on the documented 148.1 V/N; E_p = 61 GPa = 1/s11E.
"""
import numpy as np
from plate_fem import (Lx, Ly, h, mesh, assemble, solve_modes, edofs,
                       _Bb, _GAUSS, Db, Ds, rho, kappa)

# ---------------------------------------------------------------- patch data
AP, BP, HP = 0.020, 0.060, 0.0007     # m  footprint (x, y) and thickness
XPATCH = (0.005, 0.025)               # documentation only; modal() takes xp0
E_P = 61.0e9                          # Pa   PZT-5A  (1 / s11^E)
NU_P = 0.35
RHO_P = 7750.0                        # kg/m^3
D31 = -181.3e-12                      # m/V  calibrated to the 148.1 V/N anchor
Vmax = 200.0                          # V    patch voltage limit


def _patch_props():
    """Element section: plate + perfectly bonded patch about the midplane."""
    z0, z1 = h / 2.0, h / 2.0 + HP
    Dp = E_P / (1.0 - NU_P ** 2) * (z1 ** 3 - z0 ** 3) / 3.0
    Db_e = Db + Dp * np.array([[1.0, NU_P, 0.0],
                               [NU_P, 1.0, 0.0],
                               [0.0, 0.0, (1.0 - NU_P) / 2.0]])
    Gp = E_P / (2.0 * (1.0 + NU_P))
    Ds_e = Ds + kappa * Gp * HP * np.eye(2)
    mh = rho * h + RHO_P * HP
    mI = rho * h ** 3 / 12.0 + RHO_P * (z1 ** 3 - z0 ** 3) / 3.0
    return Db_e, Ds_e, mh, mI


def _covered(els, nodes, xp0, yp0):
    """Indices of elements whose centroid lies inside the patch footprint."""
    xc = nodes[els, 0].mean(axis=1)
    yc = nodes[els, 1].mean(axis=1)
    return np.where((xc > xp0) & (xc < xp0 + AP)
                    & (yc > yp0) & (yc < yp0 + BP))[0]


def modal(nx, ny, xp0, yp0, n_modes):
    """Patched-plate modal model.

    Returns f [Hz], Phi (full-size, mass-normalised), bmod = Phi^T bvec
    (modal patch force per volt), bvec (full-size nodal load per volt),
    nodes, K, M (sparse), free-DOF index.
    """
    nodes, els, xs, ys = mesh(nx, ny)
    cov = _covered(els, nodes, xp0, yp0)
    props = {int(e): _patch_props() for e in cov}
    K, M, nodes, els, free = assemble(nx, ny, elem_props=props)
    f, Phi = solve_modes(K, M, free, n_modes)

    # Crawley-de Luis consistent moment load per volt
    a, b = Lx / nx, Ly / ny
    zbar = (h + HP) / 2.0
    m_p = E_P / (1.0 - NU_P) * D31 * zbar          # N (moment / unit length / V)
    fe = np.zeros(12)
    detJ = a * b / 4.0
    for xi, eta in _GAUSS:
        Bb = _Bb(a, b, xi, eta)
        fe += Bb.T @ np.array([m_p, m_p, 0.0]) * detJ
    bvec = np.zeros(K.shape[0])
    for e in cov:
        bvec[edofs(els[e])] += fe

    bmod = Phi.T @ bvec
    return f, Phi, bmod, bvec, nodes, K, M, free


if __name__ == '__main__':
    f, Phi, bmod, bvec, nodes, K, M, free = modal(40, 32, 0.005, 0.0, 6)
    print('patched plate [Hz]:', '  '.join('%.1f' % x for x in f))
    print('bmod:', np.array2string(bmod, precision=4))
