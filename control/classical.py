"""
classical.py — DVF, VPA et LQG : les trois references du domaine
=================================================================
Trois structures que toute comparaison serieuse en controle actif de vibration
doit contenir, et qui manquaient. Elles ne sont pas la pour gagner : elles sont
la pour donner une ECHELLE. Un ecart entre deux structures riches ne veut rien
dire tant qu'on ne sait pas ce que rend un retour de vitesse a un parametre.

DVF — RETOUR DIRECT DE VITESSE. La structure la plus simple qui existe en
amortissement actif : u = -g ydot. Un derivateur pur n'est pas realisable, et
surtout il amplifierait le bruit de mesure sans borne ; on prend donc la forme
lavee (« washout »)

    K(s) = g s / (1 + s/w_d)

qui derive dans la bande utile et sature au-dela. DEUX parametres. C'est peu,
et c'est le propos : si une structure a dix-huit etats ne bat pas nettement
celle-ci, la question de savoir a quoi servent les seize autres est legitime.

Sa faiblesse est connue et attendue ICI en particulier : le DVF n'est
inconditionnellement stable que pour un couple COLOCALISE. Sur cette plaque
l'actionneur et le capteur ne le sont pas — c'est toute l'origine du zero
instable a 2459 Hz — donc rien ne garantit sa stabilite, et l'optimiseur devra
la trouver plutot que d'en heriter.

VPA — ABSORBEUR PASSIF VIRTUEL. Le correcteur imite un absorbeur accorde,
masse-ressort-amortisseur, mais en logiciel :

    K(s) = g w_a^2 / (s^2 + 2 z_a w_a s + w_a^2)

Un absorbeur PAR MODE de broutement, donc deux, soit SIX parametres. C'est la
forme sous laquelle la litterature du fraisage traite le plus souvent le
broutement, et elle a un avantage structurel reel sur le DVF : etant
passe-bande, elle ne demande aucune autorite hors de la bande visee — donc
elle ne reveille pas les modes hauts ni le zero instable.

LQG — REGULATEUR QUADRATIQUE + FILTRE DE KALMAN. La reference optimale au sens
quadratique. Ponderation d'etat ecrite sur la SORTIE mesuree, Q = q C'C, et
bruits pris proportionnels a la voie de commande, W = w B B'. Seuls les
RAPPORTS q/r et w/v pilotent le resultat — c'est une propriete du probleme,
pas une approximation — mais on garde les quatre poignees separees pour ne pas
imposer a l'optimiseur une reparametrisation qu'il n'a pas demandee, plus une
frequence de ponderation qui met l'accent la ou le broutement vit. Cinq
parametres, comme le FOPID, le H-infini et la mu-synthese.

CE QUE LE LQG N'A PAS, ET QUI COMPTE. Il n'offre AUCUNE garantie de robustesse
— les marges de retour d'etat du LQR sont detruites par l'observateur (Doyle
1978, « Guaranteed margins for LQG regulators: there are none »). C'est
precisement la raison historique pour laquelle H-infini et mu existent, et
c'est donc une comparaison qui a un contenu : les trois sont ici, sur la meme
plaque et sous les memes contraintes.
"""
import numpy as np
from scipy.linalg import solve_continuous_are


def dvf_ss(g, f_d, sign_loop=1.0):
    """K(s) = sign . g s / (1 + s/w_d) — derivation lavee."""
    wd = 2.0 * np.pi * float(f_d)
    # g s/(1+s/wd) = g wd s/(s+wd) = g wd (1 - wd/(s+wd))
    A = np.array([[-wd]])
    B = np.array([[1.0]])
    C = np.array([[-float(sign_loop) * g * wd * wd]])
    D = np.array([[float(sign_loop) * g * wd]])
    return A, B, C, D


def vpa_ss(gains, freqs, zetas, sign_loop=1.0):
    """Somme d'absorbeurs du second ordre, un par mode vise.

    K(s) = sign . sum_k  g_k w_k^2 / (s^2 + 2 z_k w_k s + w_k^2)
    """
    gains = np.atleast_1d(np.asarray(gains, float))
    freqs = np.atleast_1d(np.asarray(freqs, float))
    zetas = np.atleast_1d(np.asarray(zetas, float))
    n = len(gains)
    A = np.zeros((2 * n, 2 * n))
    B = np.zeros((2 * n, 1))
    C = np.zeros((1, 2 * n))
    for k in range(n):
        w = 2.0 * np.pi * freqs[k]
        i = 2 * k
        A[i, i + 1] = 1.0
        A[i + 1, i] = -w * w
        A[i + 1, i + 1] = -2.0 * zetas[k] * w
        B[i + 1, 0] = 1.0
        C[0, i] = float(sign_loop) * gains[k] * w * w
    return A, B, C, np.zeros((1, 1))


def _bandpass(f0, zeta):
    """Meme forme que hinf.bandpass_weight, gain unite a la resonance."""
    w0 = 2.0 * np.pi * float(f0)
    a = 2.0 * float(zeta) * w0
    return (np.array([[0.0, 1.0], [-w0 * w0, -a]]),
            np.array([[0.0], [1.0]]),
            np.array([[0.0, a]]),
            np.array([[0.0]]))


class LqgFailure(RuntimeError):
    """Riccati du LQR ou du filtre insoluble pour ces ponderations."""


def lqg_ss(plant, q, r, w_proc, v_meas, f_w, zeta_w=0.5):
    """LQG a ponderation MISE EN FORME EN FREQUENCE.

    Le procede est augmente d'un passe-bande sur la sortie ; c'est CETTE
    sortie filtree qui est penalisee. Sans cela le LQG ponderait l'erreur
    uniformement, y compris la ou il n'y a rien a corriger, et la comparaison
    avec le H-infini — dont la ponderation est justement un passe-bande —
    porterait autant sur le choix de ponderation que sur la structure.
    """
    Ap, Bp, Cp, _ = [np.atleast_2d(np.asarray(m, float)) for m in plant]
    Aw, Bw, Cw, _ = _bandpass(f_w, zeta_w)
    npp, nw = Ap.shape[0], Aw.shape[0]
    n = npp + nw
    A = np.zeros((n, n))
    A[:npp, :npp] = Ap
    A[npp:, npp:] = Aw
    A[npp:, :npp] = Bw @ Cp
    B = np.zeros((n, 1))
    B[:npp, 0:1] = Bp
    C = np.zeros((1, n))
    C[0, :npp] = Cp
    Cz = np.zeros((1, n))
    Cz[0, npp:] = Cw                       # sortie PONDEREE (bande visee)
    try:
        X = solve_continuous_are(A, B, float(q) * (Cz.T @ Cz),
                                 float(r) * np.eye(1))
        Y = solve_continuous_are(A.T, C.T, float(w_proc) * (B @ B.T),
                                 float(v_meas) * np.eye(1))
    except (np.linalg.LinAlgError, ValueError) as e:
        raise LqgFailure(str(e))
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y))):
        raise LqgFailure('Riccati non finie')
    K = (B.T @ X) / float(r)               # u = -K x
    L = (Y @ C.T) / float(v_meas)          # injection
    Ak = A - B @ K - L @ C
    # u = -K x_hat, et l'observateur est pilote par y : le signe du retour est
    # DECOUVERT par la synthese, comme en H-infini, pas impose de l'exterieur.
    return Ak, L, -K, np.zeros((1, 1))
