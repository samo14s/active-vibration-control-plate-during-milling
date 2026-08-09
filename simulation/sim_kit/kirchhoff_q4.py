"""
kirchhoff_q4.py
================
Éléments finis de plaque mince (Kirchhoff) à 4 nœuds Q4 avec 12 ddl :
chaque nœud porte (w, dw/dx, dw/dy).

Fonctions de forme : polynômes d'Hermite 2D (12 fonctions cubiques).
Intégration par quadrature de Gauss.
"""
import numpy as np


# =====================================================================
#  Fonctions de forme Hermite Q4 (12 ddl) et leurs dérivées analytiques
# =====================================================================
def shape_function_K(xi: float, eta: float, lex: float, ley: float):
    """
    Retourne (N, derN) où :
      - N    : (12,)   fonctions de forme aux ddl (w, dw/dx, dw/dy) x 4 nœuds
      - derN : (5, 12) dérivées :
          ligne 0 : ∂N/∂xi
          ligne 1 : ∂N/∂eta
          ligne 2 : ∂²N/∂xi²
          ligne 3 : ∂²N/∂eta²
          ligne 4 : ∂²N/(∂xi ∂eta)
    """
    # Fonctions de forme
    N = np.array([
        -((xi-1)*(eta-1)*(xi**2 + xi + eta**2 + eta - 2))/8,
        -ley*((xi-1)*(eta-1)**2*(eta+1))/16,
         lex*((xi-1)**2*(xi+1)*(eta-1))/16,
         ((xi+1)*(eta-1)*(xi**2 - xi + eta**2 + eta - 2))/8,
         ley*((xi+1)*(eta-1)**2*(eta+1))/16,
         lex*((xi-1)*(xi+1)**2*(eta-1))/16,
         ((xi+1)*(eta+1)*(-xi**2 + xi - eta**2 + eta + 2))/8,
         ley*((xi+1)*(eta-1)*(eta+1)**2)/16,
        -lex*((xi-1)*(xi+1)**2*(eta+1))/16,
         ((xi-1)*(eta+1)*(xi**2 + xi + eta**2 - eta - 2))/8,
        -ley*((xi-1)*(eta-1)*(eta+1)**2)/16,
        -lex*((xi-1)**2*(xi+1)*(eta+1))/16,
    ])

    derN = np.zeros((5, 12))

    # ∂N/∂xi
    derN[0, :] = [
        -((eta-1)*(eta**2 + eta + 3*xi**2 - 3))/8,
        -ley*((eta-1)**2*(eta+1))/16,
         lex*((eta-1)*(3*xi**2 - 2*xi - 1))/16,
         ((eta-1)*(eta**2 + eta + 3*xi**2 - 3))/8,
         ley*((eta-1)**2*(eta+1))/16,
         lex*((eta-1)*(3*xi**2 + 2*xi - 1))/16,
         ((eta+1)*(-eta**2 + eta - 3*xi**2 + 3))/8,
         ley*((eta-1)*(eta+1)**2)/16,
        -lex*((eta+1)*(3*xi**2 + 2*xi - 1))/16,
         ((eta+1)*(eta**2 - eta + 3*xi**2 - 3))/8,
        -ley*((eta-1)*(eta+1)**2)/16,
        -lex*((eta+1)*(3*xi**2 - 2*xi - 1))/16,
    ]

    # ∂N/∂eta
    derN[1, :] = [
        -((xi-1)*(xi**2 + xi + 3*eta**2 - 3))/8,
        -ley*((xi-1)*(3*eta**2 - 2*eta - 1))/16,
         lex*((xi-1)**2*(xi+1))/16,
         ((xi+1)*(xi**2 - xi + 3*eta**2 - 3))/8,
         ley*((xi+1)*(3*eta**2 - 2*eta - 1))/16,
         lex*((xi-1)*(xi+1)**2)/16,
         ((xi+1)*(-xi**2 + xi - 3*eta**2 + 3))/8,
         ley*((xi+1)*(3*eta**2 + 2*eta - 1))/16,
        -lex*((xi+1)**2*(xi-1))/16,
         ((xi-1)*(xi**2 + xi + 3*eta**2 - 3))/8,
        -ley*((xi-1)*(3*eta**2 + 2*eta - 1))/16,
        -lex*((xi-1)**2*(xi+1))/16,
    ]

    # ∂²N/∂xi²
    derN[2, :] = [
        -(3*xi*(eta-1))/4,
         0.0,
         lex*((3*xi-1)*(eta-1))/8,
         (3*xi*(eta-1))/4,
         0.0,
         lex*((3*xi+1)*(eta-1))/8,
        -(3*xi*(eta+1))/4,
         0.0,
        -lex*((3*xi+1)*(eta+1))/8,
         (3*xi*(eta+1))/4,
         0.0,
        -lex*((3*xi-1)*(eta+1))/8,
    ]

    # ∂²N/∂eta²
    derN[3, :] = [
        -(3*eta*(xi-1))/4,
        -ley*((3*eta-1)*(xi-1))/8,
         0.0,
         (3*eta*(xi+1))/4,
         ley*((3*eta-1)*(xi+1))/8,
         0.0,
        -(3*eta*(xi+1))/4,
         ley*((3*eta+1)*(xi+1))/8,
         0.0,
         (3*eta*(xi-1))/4,
        -ley*((3*eta+1)*(xi-1))/8,
         0.0,
    ]

    # ∂²N/(∂xi ∂eta)
    derN[4, :] = [
        -(3*xi**2 + 3*eta**2 - 4)/8,
        -ley*(3*eta**2 - 2*eta - 1)/16,
         lex*(3*xi**2 - 2*xi - 1)/16,
         (3*xi**2 + 3*eta**2 - 4)/8,
         ley*(3*eta**2 - 2*eta - 1)/16,
         lex*(3*xi**2 + 2*xi - 1)/16,
         (-3*xi**2 - 3*eta**2 + 4)/8,
         ley*(3*eta**2 + 2*eta - 1)/16,
        -lex*(3*xi**2 + 2*xi - 1)/16,
         (3*xi**2 + 3*eta**2 - 4)/8,
        -ley*(3*eta**2 + 2*eta - 1)/16,
        -lex*(3*xi**2 - 2*xi - 1)/16,
    ]

    return N, derN


def jacobian_K(xi: float, eta: float, lex: float, ley: float):
    """Jacobienne d'un Q4 rectangulaire (analytique)."""
    Jac = np.array([[lex/2, 0.0],
                    [0.0,   ley/2]])
    return Jac


def matrix_der_K(xi: float, eta: float, lex: float, ley: float):
    """
    Matrice B (3x12) des dérivées secondes physiques :
        B[0, :] = -∂²N/∂x²
        B[1, :] = -∂²N/∂y²
        B[2, :] = -2 ∂²N/∂x∂y
    """
    _, derN = shape_function_K(xi, eta, lex, ley)
    Jac = jacobian_K(xi, eta, lex, ley)
    J_invT = np.linalg.inv(Jac).T

    B = np.zeros((3, 12))
    B[0, :] = -J_invT[0, 0]**2 * derN[2, :]
    B[1, :] = -J_invT[1, 1]**2 * derN[3, :]
    B[2, :] = -2 * J_invT[0, 0] * J_invT[1, 1] * derN[4, :]

    return B


def stiffness_matrix_K(E: float, nu: float, lex: float, ley: float, h: float):
    """Matrice de raideur élémentaire 12x12 — Gauss 3x3."""
    Hf = E*h**2 / (12*(1 - nu**2)) * np.array([[1.0, nu,  0.0],
                                                [nu,  1.0, 0.0],
                                                [0.0, 0.0, (1-nu)/2]])
    xi  = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
    eta = xi.copy()
    w   = np.array([5/9, 8/9, 5/9])

    K = np.zeros((12, 12))
    for i in range(3):
        for j in range(3):
            B  = matrix_der_K(xi[i], eta[j], lex, ley)
            Ja = jacobian_K(xi[i], eta[j], lex, ley)
            K += h * w[i] * w[j] * B.T @ Hf @ B * np.linalg.det(Ja)
    return K


def mass_matrix_K(rho: float, lex: float, ley: float, h: float,
                  rotary: bool = True):
    """
    Matrice de masse élémentaire 12x12 — Gauss 5x5 exacte.

        M = I0 ∫ N Nᵀ dA + I2 ∫ (N,x N,xᵀ + N,y N,yᵀ) dA
        I0 = rho*h ,  I2 = rho*h³/12

    Le second terme est l'inertie de rotation : ce sont les termes I2 de
    l'Eq. (A.1) de l'article, que la version précédente omettait. Son effet est
    de -0.8 % environ sur le cinquième mode.

    Les abscisses et poids de Gauss étaient auparavant arrondis à trois
    décimales (somme des poids = 2.001), ce qui surestimait la masse de
    exactement +0.100 %. Ils viennent maintenant de numpy.

    rotary=False restitue le modèle de Kirchhoff pur (sans inertie de rotation).
    """
    xi, w = np.polynomial.legendre.leggauss(5)
    I0 = rho * h
    I2 = rho * h**3 / 12.0 if rotary else 0.0

    M = np.zeros((12, 12))
    for i in range(5):
        for j in range(5):
            N, derN = shape_function_K(xi[i], xi[j], lex, ley)
            Ja = jacobian_K(xi[i], xi[j], lex, ley)
            dJ = np.linalg.det(Ja)
            M += w[i] * w[j] * dJ * I0 * np.outer(N, N)
            if I2:
                Nx = derN[0, :] * (2.0/lex)
                Ny = derN[1, :] * (2.0/ley)
                M += w[i] * w[j] * dJ * I2 * (np.outer(Nx, Nx) + np.outer(Ny, Ny))
    return M


def shape_at_point(x_g: float, y_g: float,
                   N1: int, N2: int,
                   lex: float, ley: float, n1: int):
    """
    Retourne le vecteur ligne N_glob (1 x ndof) tel que
        w(x_g, y_g) = N_glob @ U_global
    """
    ndof = 3 * n1 * (N2 + 1)
    N_glob = np.zeros(ndof)

    I = int(np.floor(x_g/lex)) + 1
    J = int(np.floor(y_g/ley)) + 1
    I = max(1, min(I, N1))
    J = max(1, min(J, N2))

    x0 = (I - 1) * lex
    y0 = (J - 1) * ley
    xi  = max(-1.0, min(1.0, 2*(x_g - x0)/lex - 1))
    eta = max(-1.0, min(1.0, 2*(y_g - y0)/ley - 1))

    N_loc, _ = shape_function_K(xi, eta, lex, ley)

    DofE = np.r_[
        3*(J-1)*n1 + 3*(I-1) + np.arange(6),
        3*J*n1     + 3*(I-1) + np.arange(3, 6),
        3*J*n1     + 3*(I-1) + np.arange(3),
    ]

    N_glob[DofE] = N_loc
    return N_glob


def laplace_n_patch(xP1: float, xP2: float, zP1: float, zP2: float,
                    N1: int, N2: int, lex: float, ley: float,
                    n1: int, ndof: int):
    """
    Calcule g (1 x ndof) = ∫∫_patch ∇²N dA
    pour le couplage piézoélectrique (Eq. 15 article).
    Intégration de Gauss 2x2 sur l'intersection patch/élément.
    """
    g = np.zeros(ndof)

    I_min = max(1,  int(np.floor(xP1/lex)) + 1)
    I_max = min(N1, int(np.ceil (xP2/lex)))
    J_min = max(1,  int(np.floor(zP1/ley)) + 1)
    J_max = min(N2, int(np.ceil (zP2/ley)))

    xg2 = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    wg2 = np.array([1.0, 1.0])

    for J in range(J_min, J_max + 1):
        for I in range(I_min, I_max + 1):
            xe_lo, xe_hi = (I-1)*lex, I*lex
            ye_lo, ye_hi = (J-1)*ley, J*ley

            x_lo = max(xe_lo, xP1);  x_hi = min(xe_hi, xP2)
            y_lo = max(ye_lo, zP1);  y_hi = min(ye_hi, zP2)
            if x_hi <= x_lo or y_hi <= y_lo:
                continue

            xm, xh = (x_lo + x_hi)/2, (x_hi - x_lo)/2
            ym, yh = (y_lo + y_hi)/2, (y_hi - y_lo)/2

            DofE = np.r_[
                3*(J-1)*n1 + 3*(I-1) + np.arange(6),
                3*J*n1     + 3*(I-1) + np.arange(3, 6),
                3*J*n1     + 3*(I-1) + np.arange(3),
            ]

            for ig in range(2):
                for jg in range(2):
                    xp_g = xm + xh * xg2[ig]
                    yp_g = ym + yh * xg2[jg]

                    xi_loc  = 2*(xp_g - xe_lo)/lex - 1
                    eta_loc = 2*(yp_g - ye_lo)/ley - 1

                    B_loc = matrix_der_K(xi_loc, eta_loc, lex, ley)
                    Lap_N = -(B_loc[0, :] + B_loc[1, :])

                    g[DofE] += wg2[ig] * wg2[jg] * Lap_N * (xh * yh)

    return g
