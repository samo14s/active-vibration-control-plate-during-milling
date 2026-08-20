"""
adrc.py — ADRC-FOPID : rejet actif de perturbation + loi FOPID
===============================================================
Le procede est vu comme un double integrateur perturbe

    y'' = f(t) + b0 u ,

ou la "perturbation totale" f rassemble TOUT ce qui n'est pas le canal
nominal b0 u : dynamique modale, couplage regeneratif retarde alpha4 D^T D
y(t - tau), variation de D(x) le long de la passe, et le contenu spectral aux
harmoniques de passage de dent. C'est precisement le point : dans le fraisage
d'une paroi mince, la source d'instabilite EST une perturbation retardee que
l'observateur peut estimer et compenser sans jamais la modeliser.

Observateur d'etat etendu lineaire (LESO) d'ordre 3, parametre en bande
passante (Gao, ACC 2003) :

    z1' = z2 + 3 w_o (y - z1)
    z2' = z3 + b0 u + 3 w_o^2 (y - z1)
    z3' =              w_o^3 (y - z1)          -> z3 estime f

Loi de commande :

    u = (u0 - z3) / b0 ,   u0 = FOPID(e),  e = -z1   (consigne nulle)

Le terme -z3/b0 annule la perturbation estimee ; le FOPID ne pilote plus que
le double integrateur restant. Le correcteur complet reste LINEAIRE et
INVARIANT : il s'ecrit exactement comme un (A, B, C, D) de y vers u, donc il
passe dans la meme machinerie de Floquet et de simulation que le FOPID seul —
condition necessaire a une comparaison equitable.

Il n'y a pas de boucle algebrique : u ne depend que des etats (z1 alimente le
FOPID, z3 la compensation), et le terme b0 u de z2' se simplifie exactement
avec la division par b0, d'ou z2' = u0 + 3 w_o^2 (y - z1).

Parametres : K_p, K_i, K_d, lambda, mu (comme le FOPID) + w_o, b0.
"""
import numpy as np

from fopid import frac_ss


def adrc_fopid_ss(Kp, Ki, Kd, lam, mu, wo, b0, wb, wh, N, sign_loop=1.0):
    """ADRC-FOPID en representation d'etat : y -> u.

    Etats : [z1, z2, z3, x_I, x_D].
    """
    Ai, Bi, Ci, Di = frac_ss(-abs(lam), wb, wh, N)      # s^(-lambda)
    Ad, Bd, Cd, Dd = frac_ss(abs(mu), wb, wh, N)        # s^(mu)
    ni, nd = Ai.shape[0], Ad.shape[0]
    n = 3 + ni + nd
    b1, b2, b3 = 3.0 * wo, 3.0 * wo ** 2, wo ** 3

    # --- loi FOPID sur e = -z1 :  u0 = sign_loop * (Kp e + Ki I e + Kd D e)
    d_e = sign_loop * (Kp + Ki * Di.item() + Kd * Dd.item())   # part directe
    c_I = sign_loop * Ki * Ci                                   # sur x_I
    c_D = sign_loop * Kd * Cd                                   # sur x_D
    # u0 = -d_e * z1 + c_I x_I + c_D x_D
    u0_row = np.zeros(n)
    u0_row[0] = -d_e
    u0_row[3:3 + ni] = c_I
    u0_row[3 + ni:] = c_D

    A = np.zeros((n, n))
    B = np.zeros((n, 1))
    # z1' = z2 + b1 (y - z1)
    A[0, 0] = -b1
    A[0, 1] = 1.0
    B[0, 0] = b1
    # z2' = u0 + b2 (y - z1)          (le terme b0*u annule -z3 exactement)
    A[1, :] = u0_row
    A[1, 0] += -b2
    B[1, 0] = b2
    # z3' = b3 (y - z1)
    A[2, 0] = -b3
    B[2, 0] = b3
    # etats fractionnaires, entree e = -z1
    A[3:3 + ni, 3:3 + ni] = Ai
    A[3:3 + ni, 0] = -Bi[:, 0]
    A[3 + ni:, 3 + ni:] = Ad
    A[3 + ni:, 0] = -Bd[:, 0]

    # u = (u0 - z3) / b0
    C = (u0_row.copy() / b0).reshape(1, -1)
    C[0, 2] -= 1.0 / b0
    D = np.zeros((1, 1))
    return A, B, C, D


def eso_bandwidth_guess(plate, n_modes=2, factor=4.0):
    """Repere pour w_o : quelques fois la pulsation du mode le plus haut
    retenu (l'observateur doit etre plus rapide que ce qu'il estime)."""
    return factor * float(plate.omega_n[n_modes - 1])


def b0_nominal(plate, n_modes=2):
    """b0 nominal : gain haute frequence de y'' par volt, soit D_obs . H_Pe.

    C'est la definition de l'ADRC (le gain du canal y'' <- u quand s -> inf) et
    on la garde telle quelle. Mais il faut savoir ce qu'elle vaut ICI : avec la
    geometrie de pastille imposee par la Fig. 12(b) les residus valent
    [-0.655, -0.980, -0.880, +3.294, +2.616], donc la somme (+3.395) est une
    compensation a 60 % de somme|r_i| = 8.42, dominee par les modes 4-5 (3351
    et 4122 Hz). Le gain effectif b(w) = -w^2 P_u(jw) mesure sur ce modele vaut
    -4.4 a 1300 Hz, -3.8 a 2000 Hz, +11.2 a 5000 Hz : la valeur asymptotique
    +3.4 n'est atteinte qu'au-dela de ~15 kHz, soit tres au-dessus du filtre
    d'anti-repliement a 8 kHz. Autrement dit, dans la bande ou vit le
    broutement, le canal a le SIGNE OPPOSE a cette limite, et une amplitude
    d'ordre 1 a 4 au lieu de 3.4.

    On ne corrige donc pas la definition — on s'assure que le boitier de
    recherche COUVRE largement la valeur utile : sign_variant explore les deux
    signes et b0_scale va de 0.05 a 50, soit |b0| de 0.17 a 170. C'est au PSO
    de trouver, pas a cette fonction de deviner.
    """
    H = np.asarray(plate.H_Pe_modal, float)[:n_modes]
    D_obs = plate.D_row(plate.lp, plate.hp)[:n_modes]
    return float(D_obs @ H)
