"""Surface-bonded piezoelectric patch actuator coupling.

Induced-moment model for a thin patch bonded on one face of the plate:
a drive voltage u produces the free in-plane strain  L = d31 * u / h_pa,
equibiaxial patch stress  s = E_pe/(1-nu_pe) * L, and, acting at the offset
z_off = (b_p + h_pa)/2 from the plate mid-plane, the distributed bending
moment intensity

    m_ind = E_pe/(1-nu_pe) * d31 * z_off * u        [N]

uniform over the patch footprint. The consistent nodal load is

    f_pe = Theta * u,   Theta = sum_e  int_{A_e ∩ patch} Bb^T [1,1,0]^T m1 dA

with m1 = E_pe/(1-nu_pe)*d31*z_off. This is the standard thin-patch limit
and is equivalent to the pin-force/moment model used in Du et al. (2024),
Eq. (14)-(15), up to their two-sided correction factor C_P0; the achievable
voltage->displacement gain is validated against their measured swept
response (ref. Fig. 12b) in tests/test_fem.py.
"""
from __future__ import annotations

import numpy as np

from .fem_plate import NDOF, PlateModel, _shape_deriv
from .params import Params


def coupling_vector(model: PlateModel) -> np.ndarray:
    """Theta (n_free,) such that f_free = Theta * u  [N per V]."""
    p: Params = model.params
    pz = p.piezo
    x1, z1 = pz.x0, pz.z0
    x2, z2 = pz.x0 + pz.length, pz.z0 + pz.width
    m1 = pz.E / (1.0 - pz.nu) * pz.d31 * 0.5 * (p.plate.bp + pz.h_pa)

    ex = p.plate.lp / model.nx
    ez = model.height / model.nz
    Jinv = np.diag([2.0 / ex, 2.0 / ez])
    detJ = ex * ez / 4.0
    gp2 = [(-1 / np.sqrt(3), -1 / np.sqrt(3)), (1 / np.sqrt(3), -1 / np.sqrt(3)),
           (1 / np.sqrt(3), 1 / np.sqrt(3)), (-1 / np.sqrt(3), 1 / np.sqrt(3))]

    # element consistent load (12,) for a fully covered element
    fe = np.zeros(12)
    for xi, eta in gp2:
        dN = Jinv @ _shape_deriv(xi, eta)
        Bb = np.zeros((3, 12))
        for a in range(4):
            c = 3 * a
            Bb[0, c + 1] = dN[0, a]
            Bb[1, c + 2] = dN[1, a]
            Bb[2, c + 1] = dN[1, a]
            Bb[2, c + 2] = dN[0, a]
        fe += Bb.T @ np.array([m1, m1, 0.0]) * detJ

    ndof_full = NDOF * len(model.nodes)
    f_full = np.zeros(ndof_full)
    for e, conn in enumerate(model.elems):
        xc = model.nodes[conn].mean(axis=0)
        inside = (x1 - 1e-9 <= xc[0] <= x2 + 1e-9) and \
                 (z1 - 1e-9 <= xc[1] <= z2 + 1e-9)
        if not inside:
            continue
        dofs = np.concatenate([[NDOF * n, NDOF * n + 1, NDOF * n + 2]
                               for n in conn])
        f_full[dofs] += fe

    return f_full[model.free]
