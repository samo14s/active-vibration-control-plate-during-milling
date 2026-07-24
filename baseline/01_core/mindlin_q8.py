"""
mindlin_q8.py
=============
Élément fini de plaque épaisse (Reissner-Mindlin) Serendip à 8 nœuds (Q8),
3 ddl/nœud : (w, theta_x, theta_y)  ->  24 ddl/élément.

Ce module est un **portage littéral** de l'implémentation MATLAB du dépôt
`Plate-FEM/Mindlin_plate`. Chaque fonction Python reproduit exactement la
formulation de la fonction `.m` correspondante :

    Python                         MATLAB (Plate-FEM/Mindlin_plate)
    ---------------------------    --------------------------------
    shape_function_M(...)      <-  Shape_function_M.m
    jacobian_M(...)            <-  Jacobian_M.m
    matrix_der_M(...)          <-  Matrix_der_M.m        (Bf, Bs)
    stiffness_matrix_M(...)    <-  Stiffness_matrix_M.m  (K = h*(Kf + Ks))
    mass_matrix_M(...)         <-  Mass_matrix_M.m       (Ie = diag[1, h^2/12, h^2/12])
    thermal_stress_M(...)      <-  Thermal_stress_M.m

Théorie (Plate-FEM/Mindlin_plate/Mindlin.markdown) :
    eps_f = z * Df * u          (flexion)      Hf* = E/(1-nu^2) [...]
    eps_s = Ds * u              (cisaillement) Hs* = E*kappa/(2(1+nu)) I2
    Ke = Kf + Ks , Kf = h^3/12 int Pf^T Hf* Pf dS , Ks = kappa*h int Ps^T Hs* Ps dS

Intégration de Gauss 2x2 (réduite uniforme) pour Kf, Ks et M — identique au
code MATLAB source (`xi = [-1/sqrt(3), 1/sqrt(3)]`).

Numérotation du maillage Serendip et fonctions d'aide (`build_node_map`,
`elem_dofs_M`, `shape_at_point_M`, `piezo_moment_load_M`) : ajoutées pour
l'intégration dans le package de l'article (assemblage, encastrement,
observation, couplage piézoélectrique). Elles produisent une numérotation
propre topologiquement équivalente à celle du MATLAB (mêmes matrices
élémentaires, même physique).

Ordre des nœuds de l'élément (identique à `nodeCoor` du MATLAB) :
      4 ------7------ 3          (xi, eta)
      |               |          n1 = (-1,-1)   n5 = ( 0,-1)
      8       .       6          n2 = (+1,-1)   n6 = (+1, 0)
      |               |          n3 = (+1,+1)   n7 = ( 0,+1)
      1 ------5------ 2          n4 = (-1,+1)   n8 = (-1, 0)
"""
import numpy as np


# =====================================================================
#  Fonctions de forme Serendip Q8 et leurs dérivées paramétriques
#  <- Shape_function_M.m
# =====================================================================
def shape_function_M(xi: float, eta: float):
    """
    Retourne (N, N_row, derN) :
      - N     : (3, 24) matrice d'interpolation (blocs diag [Ni Ni Ni])
      - N_row : (8,)     fonctions de forme [N1 ... N8]
      - derN  : (2, 8)   dérivées paramétriques
                    ligne 0 : dN/dxi
                    ligne 1 : dN/deta
    """
    N1e = -((1 - xi) * (1 - eta) * (1 + xi + eta)) / 4
    N2e = -((1 + xi) * (1 - eta) * (1 - xi + eta)) / 4
    N3e = -((1 + xi) * (1 + eta) * (1 - xi - eta)) / 4
    N4e = -((1 - xi) * (1 + eta) * (1 + xi - eta)) / 4
    N5e = ((1 - xi) * (1 + xi) * (1 - eta)) / 2
    N6e = ((1 + xi) * (1 - eta) * (1 + eta)) / 2
    N7e = ((1 - xi) * (1 + xi) * (1 + eta)) / 2
    N8e = ((1 - xi) * (1 - eta) * (1 + eta)) / 2

    N_row = np.array([N1e, N2e, N3e, N4e, N5e, N6e, N7e, N8e])

    # N (3 x 24) : diag([Ni Ni Ni]) juxtaposés
    N = np.zeros((3, 24))
    for i in range(8):
        N[0, 3 * i] = N_row[i]
        N[1, 3 * i + 1] = N_row[i]
        N[2, 3 * i + 2] = N_row[i]

    derN = np.zeros((2, 8))
    # dN/dxi
    derN[0, :] = [
        ((2 * xi + eta) * (1 - eta)) / 4,
        -((-2 * xi + eta) * (1 - eta)) / 4,
        ((2 * xi + eta) * (1 + eta)) / 4,
        -((-2 * xi + eta) * (1 + eta)) / 4,
        -xi * (1 - eta),
        ((1 - eta) * (1 + eta) / 2),
        -xi * (1 + eta),
        -((1 - eta) * (1 + eta)) / 2,
    ]
    # dN/deta
    derN[1, :] = [
        ((1 - xi) * (xi + 2 * eta)) / 4,
        -((1 + xi) * (xi - 2 * eta)) / 4,
        ((1 + xi) * (xi + 2 * eta)) / 4,
        -((1 - xi) * (xi - 2 * eta)) / 4,
        -((1 - xi) * (1 + xi)) / 2,
        -(1 + xi) * eta,
        ((1 - xi) * (1 + xi)) / 2,
        -(1 - xi) * eta,
    ]
    return N, N_row, derN


# =====================================================================
#  Jacobienne isoparamétrique  <- Jacobian_M.m
# =====================================================================
def jacobian_M(node_coor: np.ndarray, xi: float, eta: float):
    """
    node_coor : (8, 2) coordonnées physiques des 8 nœuds.
    Retourne (Jac, XYder) :
      - Jac   : (2, 2)  J = [[dx/dxi, dx/deta], [dy/dxi, dy/deta]]
      - XYder : (2, 8)  dérivées physiques
                    ligne 0 : dN/dx
                    ligne 1 : dN/dy
                    XYder = inv(J).T @ derN
    """
    _, _, derN = shape_function_M(xi, eta)

    Jac = np.zeros((2, 2))
    Jac[0, 0] = derN[0, :] @ node_coor[:, 0]   # dx/dxi
    Jac[0, 1] = derN[1, :] @ node_coor[:, 0]   # dx/deta
    Jac[1, 0] = derN[0, :] @ node_coor[:, 1]   # dy/dxi
    Jac[1, 1] = derN[1, :] @ node_coor[:, 1]   # dy/deta

    XYder = np.linalg.inv(Jac).T @ derN
    return Jac, XYder


# =====================================================================
#  Matrices déformation-déplacement  <- Matrix_der_M.m
# =====================================================================
def matrix_der_M(node_coor: np.ndarray, xi: float, eta: float):
    """
    Retourne (Bf, Bs) :
      - Bf : (3, 24) flexion   (eps_f = Bf u)
      - Bs : (2, 24) cisaillement (eps_s = Bs u)
    ddl par nœud : [w, theta_x, theta_y] -> colonnes (0::3, 1::3, 2::3).
    """
    _, N_row, _ = shape_function_M(xi, eta)
    _, XYder = jacobian_M(node_coor, xi, eta)

    Bf = np.zeros((3, 24))
    Bf[0, 2::3] = XYder[0, :]     #  d theta_y / dx
    Bf[1, 1::3] = -XYder[1, :]    # -d theta_x / dy
    Bf[2, 1::3] = -XYder[0, :]    # -d theta_x / dx
    Bf[2, 2::3] = XYder[1, :]     #  d theta_y / dy

    Bs = np.zeros((2, 24))
    Bs[0, 0::3] = XYder[0, :]     #  dw/dx
    Bs[0, 2::3] = N_row           #  + theta_y
    Bs[1, 0::3] = XYder[1, :]     #  dw/dy
    Bs[1, 1::3] = -N_row          #  - theta_x

    return Bf, Bs


# Coordonnées nodales locales de l'élément rectangulaire (ordre 1..8)
def _node_coor(lex: float, ley: float) -> np.ndarray:
    return np.array([
        [0.0,      0.0],       # 1
        [lex,      0.0],       # 2
        [lex,      ley],       # 3
        [0.0,      ley],       # 4
        [lex / 2,  0.0],       # 5
        [lex,      ley / 2],   # 6
        [lex / 2,  ley],       # 7
        [0.0,      ley / 2],   # 8
    ])


# Points de Gauss 2x2 (réduite uniforme) — identique au MATLAB
_G2 = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])


# =====================================================================
#  Matrice de raideur élémentaire  <- Stiffness_matrix_M.m
# =====================================================================
def stiffness_matrix_M(E: float, nu: float, k: float,
                       lex: float, ley: float, h: float) -> np.ndarray:
    """
    K (24x24) = h * (Kf + Ks), Gauss 2x2 pour flexion ET cisaillement.
      Hf = E*h^2/(12(1-nu^2)) [[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]]
      Hs = E*k/(2(1+nu)) I2         (k = facteur de correction de cisaillement)
    """
    node_coor = _node_coor(lex, ley)

    Hf = E * h**2 / (12 * (1 - nu**2)) * np.array([[1.0, nu, 0.0],
                                                   [nu, 1.0, 0.0],
                                                   [0.0, 0.0, (1 - nu) / 2]])
    Hs = E * k / (2 * (1 + nu)) * np.eye(2)

    Kf = np.zeros((24, 24))
    Ks = np.zeros((24, 24))
    for I in range(2):
        for J in range(2):
            Bf, Bs = matrix_der_M(node_coor, _G2[I], _G2[J])
            Jac, _ = jacobian_M(node_coor, _G2[I], _G2[J])
            detJ = np.linalg.det(Jac)
            Kf += Bf.T @ Hf @ Bf * detJ
            Ks += Bs.T @ Hs @ Bs * detJ

    return h * (Kf + Ks)


# =====================================================================
#  Matrice de masse élémentaire  <- Mass_matrix_M.m
# =====================================================================
def mass_matrix_M(rho: float, lex: float, ley: float, h: float) -> np.ndarray:
    """
    M (24x24) = rho * h * int N^T Ie N dS, Gauss 2x2.
      Ie = diag([1, h^2/12, h^2/12])   (masse + inertie rotatoire)
    """
    node_coor = _node_coor(lex, ley)
    Ie = np.diag([1.0, h**2 / 12, h**2 / 12])

    M = np.zeros((24, 24))
    for I in range(2):
        for J in range(2):
            N, _, _ = shape_function_M(_G2[I], _G2[J])
            Jac, _ = jacobian_M(node_coor, _G2[I], _G2[J])
            detJ = np.linalg.det(Jac)
            M += N.T @ Ie @ N * detJ

    return rho * h * M


# =====================================================================
#  Chargement thermique élémentaire  <- Thermal_stress_M.m
# =====================================================================
def thermal_stress_M(E: float, nu: float, alp: float,
                     lex: float, ley: float, h: float, dT: float) -> np.ndarray:
    """
    F (24,) = int Bf^T Hf eps_ther dS, Gauss 2x2.
      eps_ther = alp * dT * [1, 1, 0]^T
    (Portage littéral — le facteur d'épaisseur est déjà porté par Hf.)
    """
    node_coor = _node_coor(lex, ley)
    Hf = E * h**2 / (12 * (1 - nu**2)) * np.array([[1.0, nu, 0.0],
                                                   [nu, 1.0, 0.0],
                                                   [0.0, 0.0, (1 - nu) / 2]])
    eps_ther = alp * dT * np.array([1.0, 1.0, 0.0])

    F = np.zeros(24)
    for I in range(2):
        for J in range(2):
            Bf, _ = matrix_der_M(node_coor, _G2[I], _G2[J])
            Jac, _ = jacobian_M(node_coor, _G2[I], _G2[J])
            detJ = np.linalg.det(Jac)
            F += Bf.T @ Hf @ eps_ther * detJ
    return F


# =====================================================================
#  Numérotation du maillage Serendip 8 nœuds (N1 x N2 éléments)
# =====================================================================
def build_node_map(N1: int, N2: int):
    """
    Construit la table de numérotation des nœuds du maillage Serendip.

    Grille (2*N1+1) x (2*N2+1) : un nœud existe à (ix, iy) SAUF si ix et iy
    sont tous deux impairs (centres d'éléments, absents en Serendip).
    Numérotation par lignes (ix rapide, iy lent).

    Retourne (node_id, ntot) :
      - node_id : (2*N1+1, 2*N2+1) int, index global du nœud ou -1
      - ntot    : nombre total de nœuds = (2N1+1)(N2+1) + (N1+1)N2
    """
    nx = 2 * N1 + 1
    ny = 2 * N2 + 1
    node_id = -np.ones((nx, ny), dtype=np.int64)
    cnt = 0
    for iy in range(ny):
        for ix in range(nx):
            if (ix % 2 == 1) and (iy % 2 == 1):
                continue          # centre d'élément : pas de nœud (Serendip)
            node_id[ix, iy] = cnt
            cnt += 1
    return node_id, cnt


# Décalages (ix, iy) des 8 nœuds locaux dans la cellule 2x2 (ordre 1..8)
_ELEM_OFFSETS = np.array([
    [0, 0], [2, 0], [2, 2], [0, 2],      # coins  1-4
    [1, 0], [2, 1], [1, 2], [0, 1],      # milieux 5-8
])


def elem_dofs_M(I: int, J: int, node_id: np.ndarray) -> np.ndarray:
    """
    ddl globaux (24,) de l'élément (I, J), I in [1,N1], J in [1,N2],
    dans l'ordre local des nœuds 1..8 et ddl [w, theta_x, theta_y].
    """
    bx = 2 * (I - 1)
    by = 2 * (J - 1)
    dofs = np.empty(24, dtype=np.int64)
    for a in range(8):
        n = node_id[bx + _ELEM_OFFSETS[a, 0], by + _ELEM_OFFSETS[a, 1]]
        dofs[3 * a] = 3 * n
        dofs[3 * a + 1] = 3 * n + 1
        dofs[3 * a + 2] = 3 * n + 2
    return dofs


def clamped_edge_dofs(node_id: np.ndarray, edge: str = "bottom") -> np.ndarray:
    """
    ddl à bloquer pour un bord encastré (w = theta_x = theta_y = 0).
    edge : 'bottom' (iy=0), 'top' (iy=max), 'left' (ix=0), 'right' (ix=max).
    """
    nx, ny = node_id.shape
    if edge == "bottom":
        nodes = node_id[:, 0]
    elif edge == "top":
        nodes = node_id[:, ny - 1]
    elif edge == "left":
        nodes = node_id[0, :]
    elif edge == "right":
        nodes = node_id[nx - 1, :]
    else:
        raise ValueError(f"edge inconnu : {edge}")
    nodes = nodes[nodes >= 0]
    dofs = np.concatenate([3 * nodes, 3 * nodes + 1, 3 * nodes + 2])
    return np.sort(dofs)


# =====================================================================
#  Évaluation de w(x, y) : vecteur ligne de sélection modale
# =====================================================================
def shape_at_point_M(x_g: float, y_g: float,
                     N1: int, N2: int, lex: float, ley: float,
                     node_id: np.ndarray, ndof: int) -> np.ndarray:
    """
    Retourne N_glob (ndof,) tel que  w(x_g, y_g) = N_glob @ U_global.
    Seuls les ddl transverses (w) des 8 nœuds de l'élément contenant le
    point portent les valeurs des fonctions de forme Serendip.
    """
    N_glob = np.zeros(ndof)

    I = int(np.floor(x_g / lex)) + 1
    J = int(np.floor(y_g / ley)) + 1
    I = max(1, min(I, N1))
    J = max(1, min(J, N2))

    x0 = (I - 1) * lex
    y0 = (J - 1) * ley
    xi = max(-1.0, min(1.0, 2 * (x_g - x0) / lex - 1))
    eta = max(-1.0, min(1.0, 2 * (y_g - y0) / ley - 1))

    _, N_row, _ = shape_function_M(xi, eta)
    DofE = elem_dofs_M(I, J, node_id)

    for a in range(8):
        N_glob[DofE[3 * a]] = N_row[a]   # ddl transverse w du nœud a
    return N_glob


# =====================================================================
#  Couplage piézoélectrique : charge de moment (analogie thermique)
# =====================================================================
def piezo_moment_load_M(xP1: float, xP2: float, zP1: float, zP2: float,
                        N1: int, N2: int, lex: float, ley: float,
                        node_id: np.ndarray, ndof: int) -> np.ndarray:
    """
    Calcule g (ndof,) = int_patch Bf^T [1, 1, 0]^T dA.

    Un patch piézoélectrique impose un champ de moment de flexion isotrope
    (M_xx = M_yy = m_piezo) sur sa surface. La charge nodale cohérente en
    théorie de Mindlin est  F = int Bf^T [m; m; 0] dA  (travail virtuel du
    moment appliqué). Cette routine renvoie l'intégrale pour une intensité
    unité ; l'appelant multiplie par m_piezo.

    C'est l'analogue Mindlin exact du terme Kirchhoff `m_piezo * ∫∇²N dA`
    (même moment physique appliqué, formulation cohérente avec Bf).
    Intégration de Gauss 2x2 sur l'intersection patch ∩ élément.
    """
    g = np.zeros(ndof)
    m_vec = np.array([1.0, 1.0, 0.0])

    I_min = max(1, int(np.floor(xP1 / lex)) + 1)
    I_max = min(N1, int(np.ceil(xP2 / lex)))
    J_min = max(1, int(np.floor(zP1 / ley)) + 1)
    J_max = min(N2, int(np.ceil(zP2 / ley)))

    node_coor = _node_coor(lex, ley)
    xg2 = _G2
    wg2 = np.array([1.0, 1.0])

    for J in range(J_min, J_max + 1):
        for I in range(I_min, I_max + 1):
            xe_lo, xe_hi = (I - 1) * lex, I * lex
            ye_lo, ye_hi = (J - 1) * ley, J * ley

            x_lo = max(xe_lo, xP1); x_hi = min(xe_hi, xP2)
            y_lo = max(ye_lo, zP1); y_hi = min(ye_hi, zP2)
            if x_hi <= x_lo or y_hi <= y_lo:
                continue

            xm, xh = (x_lo + x_hi) / 2, (x_hi - x_lo) / 2
            ym, yh = (y_lo + y_hi) / 2, (y_hi - y_lo) / 2

            DofE = elem_dofs_M(I, J, node_id)

            for ig in range(2):
                for jg in range(2):
                    xp_g = xm + xh * xg2[ig]
                    yp_g = ym + yh * xg2[jg]
                    xi_loc = 2 * (xp_g - xe_lo) / lex - 1
                    eta_loc = 2 * (yp_g - ye_lo) / ley - 1

                    Bf, _ = matrix_der_M(node_coor, xi_loc, eta_loc)
                    contrib = Bf.T @ m_vec           # (24,)
                    g[DofE] += wg2[ig] * wg2[jg] * contrib * (xh * yh)

    return g
