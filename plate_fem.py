"""
100 x 80 x 4 mm AL6061 cantilever plate, clamped along y = 0.

Mindlin / Reissner plate, 4-node bilinear quadrilateral with MITC4 shear
interpolation (shear-locking free).  Nodal DOF ordering: (w, beta_x, beta_y)
with the convention  gamma = grad(w) + beta,  so beta = -grad(w) in the thin
limit.  Element node ordering is CCW starting from the lower-left corner:

        3 ---- 2          xi_i  = [-1, +1, +1, -1]
        |      |          eta_i = [-1, -1, +1, +1]
        0 ---- 1

E is calibrated (65.85 GPa) so that the bare-plate spectrum reproduces the
validated five-mode basis of the Du et al. (2024) benchmark to < 0.3 %.
"""
import numpy as np

# ---------------------------------------------------------------- geometry
Lx, Ly, h = 0.100, 0.080, 0.004          # m  (span, free length, thickness)

# ---------------------------------------------------------------- material
E = 65.85e9          # Pa   calibrated to the validated modal basis
nu = 0.33
rho = 2700.0         # kg/m^3
kappa = 5.0 / 6.0    # shear correction factor

G = E / (2.0 * (1.0 + nu))
D = E * h ** 3 / (12.0 * (1.0 - nu ** 2))

Db = D * np.array([[1.0, nu, 0.0],
                   [nu, 1.0, 0.0],
                   [0.0, 0.0, (1.0 - nu) / 2.0]])
Ds = kappa * G * h * np.eye(2)

_XI = np.array([-1.0, 1.0, 1.0, -1.0])
_ET = np.array([-1.0, -1.0, 1.0, 1.0])


def shape(xi, eta):
    """Bilinear Q4 shape functions and their natural derivatives."""
    N = 0.25 * (1.0 + _XI * xi) * (1.0 + _ET * eta)
    dNxi = 0.25 * _XI * (1.0 + _ET * eta)
    dNet = 0.25 * _ET * (1.0 + _XI * xi)
    return N, dNxi, dNet


def mesh(nx, ny):
    """Structured rectangular mesh. Returns nodes (nn,2), els (nel,4), xs, ys."""
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    nodes = np.column_stack([X.ravel(), Y.ravel()])   # node id = i*(ny+1) + j

    els = np.empty((nx * ny, 4), dtype=int)
    e = 0
    for i in range(nx):
        for j in range(ny):
            n0 = i * (ny + 1) + j
            n1 = (i + 1) * (ny + 1) + j
            n2 = (i + 1) * (ny + 1) + j + 1
            n3 = i * (ny + 1) + j + 1
            els[e] = (n0, n1, n2, n3)
            e += 1
    return nodes, els, xs, ys


# ------------------------------------------------------------- element level
_GP = 1.0 / np.sqrt(3.0)
_GAUSS = [(-_GP, -_GP), (_GP, -_GP), (_GP, _GP), (-_GP, _GP)]


def _Bb(a, b, xi, eta):
    """Bending strain-displacement matrix (3 x 12), kappa = L beta."""
    N, dNxi, dNet = shape(xi, eta)
    dNx, dNy = dNxi * 2.0 / a, dNet * 2.0 / b
    B = np.zeros((3, 12))
    for i in range(4):
        B[0, 3 * i + 1] = dNx[i]
        B[1, 3 * i + 2] = dNy[i]
        B[2, 3 * i + 1] = dNy[i]
        B[2, 3 * i + 2] = dNx[i]
    return B


def _Bs_mitc4(a, b, xi, eta):
    """MITC4 assumed covariant shear (2 x 12), gamma = grad(w) + beta.

    Covariant strains  g_xi = w,xi + (a/2) beta_x  and  g_eta = w,eta +
    (b/2) beta_y  are sampled at the edge midpoints A(0,-1), C(0,+1) and
    D(-1,0), B(+1,0) and interpolated linearly across the element -- the
    standard tying scheme that removes shear locking on the Q4 plate.
    """
    B = np.zeros((2, 12))

    def _tie(row, pt, pref, comp, jfac):
        N, dNxi, dNet = shape(*pt)
        dW = dNxi if comp == 0 else dNet
        for i in range(4):
            B[row, 3 * i] += pref * dW[i]
            B[row, 3 * i + 1 + comp] += pref * jfac * N[i]

    _tie(0, (0.0, -1.0), 0.5 * (1.0 - eta), 0, a / 2.0)   # A
    _tie(0, (0.0, +1.0), 0.5 * (1.0 + eta), 0, a / 2.0)   # C
    _tie(1, (-1.0, 0.0), 0.5 * (1.0 - xi), 1, b / 2.0)    # D
    _tie(1, (+1.0, 0.0), 0.5 * (1.0 + xi), 1, b / 2.0)    # B
    B[0, :] *= 2.0 / a          # cartesian gamma_x = (2/a) g_xi
    B[1, :] *= 2.0 / b
    return B


def elem_km(a, b, Db_e=None, Ds_e=None, mh=None, mI=None):
    """MITC4 stiffness and consistent mass of an a x b rectangle.

    Db_e, Ds_e, mh (translational rho*h) and mI (rotary rho*h^3/12) default
    to the bare plate; patch_fem passes augmented values on covered elements.
    """
    Db_e = Db if Db_e is None else Db_e
    Ds_e = Ds if Ds_e is None else Ds_e
    mh = rho * h if mh is None else mh
    mI = rho * h ** 3 / 12.0 if mI is None else mI

    detJ = a * b / 4.0
    Ke = np.zeros((12, 12))
    Me = np.zeros((12, 12))
    for xi, eta in _GAUSS:
        Bb = _Bb(a, b, xi, eta)
        Bs = _Bs_mitc4(a, b, xi, eta)
        Ke += (Bb.T @ Db_e @ Bb + Bs.T @ Ds_e @ Bs) * detJ
        N = shape(xi, eta)[0]
        Nw = np.zeros((1, 12)); Nw[0, 0::3] = N
        Nx = np.zeros((1, 12)); Nx[0, 1::3] = N
        Ny = np.zeros((1, 12)); Ny[0, 2::3] = N
        Me += (mh * Nw.T @ Nw + mI * (Nx.T @ Nx + Ny.T @ Ny)) * detJ
    return Ke, Me


# -------------------------------------------------------------- assembly
def edofs(conn):
    """Global DOF indices of one element (12,)."""
    return np.concatenate([[3 * n, 3 * n + 1, 3 * n + 2] for n in conn])


def assemble(nx, ny, elem_props=None):
    """Assemble K, M on the structured mesh.

    elem_props : optional dict {element index: (Db_e, Ds_e, mh, mI)} for
    elements whose section differs from the bare plate (piezo patch).
    Returns K, M (sparse CSR), nodes, els, free-DOF index (clamp at y = 0).
    """
    from scipy import sparse
    nodes, els, xs, ys = mesh(nx, ny)
    a, b = Lx / nx, Ly / ny
    ndof = 3 * len(nodes)

    Ke0, Me0 = elem_km(a, b)
    rows, cols, kv, mv = [], [], [], []
    for e, conn in enumerate(els):
        if elem_props is not None and e in elem_props:
            Ke, Me = elem_km(a, b, *elem_props[e])
        else:
            Ke, Me = Ke0, Me0
        ed = edofs(conn)
        II, JJ = np.meshgrid(ed, ed, indexing='ij')
        rows.append(II.ravel()); cols.append(JJ.ravel())
        kv.append(Ke.ravel()); mv.append(Me.ravel())
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    K = sparse.coo_matrix((np.concatenate(kv), (rows, cols)),
                          shape=(ndof, ndof)).tocsr()
    M = sparse.coo_matrix((np.concatenate(mv), (rows, cols)),
                          shape=(ndof, ndof)).tocsr()

    clamped = np.where(nodes[:, 1] < 1e-12)[0]
    fixed = np.concatenate([[3 * n, 3 * n + 1, 3 * n + 2] for n in clamped])
    free = np.setdiff1d(np.arange(ndof), fixed)
    return K, M, nodes, els, free


def solve_modes(K, M, free, n_modes, ndof=None):
    """First n_modes of the clamped problem; mass-normalised full-size shapes."""
    from scipy.sparse.linalg import eigsh
    Kf = K[np.ix_(free, free)]
    Mf = M[np.ix_(free, free)]
    w2, V = eigsh(Kf.tocsc(), k=n_modes, M=Mf.tocsc(), sigma=0.0, which='LM')
    idx = np.argsort(w2)
    w2, V = w2[idx], V[:, idx]
    for k in range(n_modes):                       # mass-normalise
        V[:, k] /= np.sqrt(V[:, k] @ (Mf @ V[:, k]))
    f = np.sqrt(np.abs(w2)) / (2.0 * np.pi)
    n = K.shape[0] if ndof is None else ndof
    Phi = np.zeros((n, n_modes))
    Phi[free] = V
    return f, Phi


if __name__ == '__main__':
    K, M, nodes, els, free = assemble(40, 32)
    f, Phi = solve_modes(K, M, free, 6)
    print('bare plate, 40 x 32 mesh [Hz]:',
          '  '.join('%.1f' % x for x in f))
    print('reference                 :  519.6  1054.0  2687.5  3290.9  4047.5')
