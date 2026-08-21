"""ss_balance.py — equilibrage diagonal d'une realisation d'etat.

POURQUOI. Une similitude diagonale ne change RIEN a la fonction de transfert,
mais elle change tout a ce que Floquet en fait. Les verifications de ce depot
qui portent sur la reponse frequentielle sont aveugles a la difference :
`ss_frf` resout (jw I - A) x = B, et un solve LAPACK equilibre en interne. Ce
qui n'equilibre rien, en revanche, c'est `expm` — et c'est expm que
`step_integrals` applique a la matrice augmentee plaque + correcteur.

MESURE, conditionnement de la matrice d'etat des optima STOCKES avant cet
equilibrage, et apres une simple similitude diagonale :

    fopid 3.345e10 -> 4.6e7      fdob  5.533e19 -> 3.1e4
    fdob12345 5.306e20 -> 3.1e4  hinf  2.282e22 -> 4.8e2
    lqg   8.282e20 -> 4.8e1      vpa   7.367e14 -> 2.5e1
    adrc  1.358e28 -> voir plus bas

La source commune est `fopid.rolloff_ss` : sa forme compagne de Butterworth
porte C = [0, 2.5266e9], elle est greffee sur TOUS les correcteurs par
`fopid.series`, et series ne rehomogeneise jamais. Le 2.5266e9 se lit
verbatim dans la ligne C de chaque optimum stocke.

POURQUOI PAS `scipy.linalg.matrix_balance`. Elle n'equilibre que A et laisse
B et C intacts. Sur l'ADRC c'est pire qu'insuffisant : la coordonnee z3 de
l'ESO n'entre dans AUCUNE derivee d'etat (le terme b0*u l'annule exactement
dans la loi de commande), donc la colonne z3 de A est identiquement nulle, A
est structurellement singuliere, et matrix_balance fait diverger le facteur
d'echelle de cette coordonnee vers l'infini. Or z3 n'est pas un etat mort : il
apparait dans C. En equilibrant le SYSTEME [A B ; C] la colonne n'est plus
nulle et la coordonnee retrouve une echelle finie.

Les facteurs sont arrondis a des puissances de deux : la mise a l'echelle est
alors EXACTE en binaire et n'ajoute aucune erreur d'arrondi a elle seule.
"""
import numpy as np


def balance(A, Bs, Cs, n_sweep=8):
    """Egalise, etat par etat, la norme de la LIGNE i de [A B] et celle de la
    COLONNE i de [A ; C] par la similitude diagonale x = T x~.

    Bs, Cs : listes de matrices d'entree / de sortie (un probleme H-infini en
    a deux de chaque). Retourne (A, Bs, Cs, t) avec t = diag(T).
    """
    A = np.array(A, float)
    Bs = [np.array(b, float) for b in Bs]
    Cs = [np.array(c, float) for c in Cs]
    n = A.shape[0]
    t = np.ones(n)
    for _ in range(n_sweep):
        moved = False
        for i in range(n):
            r = np.linalg.norm(np.concatenate(
                [np.delete(A[i, :], i)] + [b[i, :] for b in Bs]))
            c = np.linalg.norm(np.concatenate(
                [np.delete(A[:, i], i)] + [cc[:, i] for cc in Cs]))
            if r <= 0.0 or c <= 0.0:
                continue
            f = 2.0 ** np.round(np.log2(np.sqrt(c / r)))
            if f == 1.0:
                continue
            A[i, :] *= f
            A[:, i] /= f
            for b in Bs:
                b[i, :] *= f
            for cc in Cs:
                cc[:, i] /= f
            t[i] /= f
            moved = True
        if not moved:
            break
    return A, Bs, Cs, t


def balance_ss(ss):
    """(A, B, C, D) equilibre. Fonction de transfert inchangee A L'ARRONDI
    PRES, et meme exactement quand les facteurs sont des puissances de deux
    representables — ce qu'ils sont par construction."""
    A, B, C, D = [np.atleast_2d(np.asarray(m, float)) for m in ss]
    if A.shape[0] == 0:
        return A, B, C, D
    A2, (B2,), (C2,), _ = balance(A, [B], [C])
    if not (np.all(np.isfinite(A2)) and np.all(np.isfinite(B2))
            and np.all(np.isfinite(C2))):
        return A, B, C, D
    return A2, B2, C2, D
