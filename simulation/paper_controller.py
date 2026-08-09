"""
================================================================================
LA COMMANDE DE L'ARTICLE — Du, Liu, Dai & Long, IJMS 274 (2024) 109257
================================================================================
"Robust combined time delay control" : commande robuste par synthese mu,
COMBINEE a une commande a retard actif. C'est le sujet meme de l'article, et
c'est ce qui manquait a ce depot -- qui ne comparait jusqu'ici qu'un LQG modal
et un ADRC/ESO, aucun des deux n'etant la methode de l'article.

Ce module suit les equations de l'article partout ou elles sont chiffrees, et
DIT explicitement ou elles ne le sont pas.

--------------------------------------------------------------------------------
CE QUE L'ARTICLE PUBLIE, ET QUI EST REPRIS ICI TEL QUEL
--------------------------------------------------------------------------------
  Eq. (16)      forme d'etat  x = [q ; q'],
                A = [[0, I], [-M^-1(K + a4 D^T D), -M^-1 C]], B = [0 ; M^-1 H],
                C = [D, 0].
  Eq. (17)      G(s) = G_r(s) + Delta_a(s) W_a(s), modele reduit aux DEUX
                premiers modes, modes 3+ couverts par incertitude additive.
  Eqs.(18)-(19) les ponderations d'incertitude additive, AVEC leurs valeurs :
                W(s) = r (s^2 + 2 z1 w1 s + w1^2)/(s^2 + 2 z2 w2 s + w2^2)
                  force   : r = 14e-6, z1 = 0.56, z2 = 0.12,
                            w1 = 2 pi 1400, w2 = 2 pi 2800
                  tension : r = 4e-7,  z1 = 0.58, z2 = 0.22,
                            w1 = 2 pi 1100, w2 = 2 pi 3500
  Eq. (23)      a4_0 = 1.6 a4_moyen, L_a = 1.3 a4_moyen, c.-a-d. a4 parcourt
                0.3 a 2.9 fois sa valeur moyenne.
  Eqs.(24)-(25) les elements de D_r^T D_r sont pris a leur moyenne (max+min)/2
                le long de l'arete, la demi-amplitude servant de perturbation.
  Sec. 3.2      masse et raideur modales a +/- 10 %, amortissement a +/- 20 %.
  Eq. (30)      COMMANDE A RETARD ACTIF :  u_d(t) = Kp y(t-tau) + Kd y'(t-tau).
  Sec. 3.4      la commande combinee est la SOMME des deux.

--------------------------------------------------------------------------------
CE QUE L'ARTICLE NE PUBLIE PAS — ET QUI EST DONC UN CHOIX, PAS UNE REPRODUCTION
--------------------------------------------------------------------------------
  * W_f(s), W_u(s), W_n : l'article dit seulement "designed as low-pass filters"
    et "a constant", sans aucune valeur. Elles fixent pourtant entierement le
    compromis performance / tension du correcteur robuste ;
  * Kp et Kd de l'Eq. (30) : aucune valeur donnee ;
  * le correcteur K_mu(s) lui-meme : ni ordre, ni coefficients, ni gamma atteint.

Ces quatre manques suffisent a rendre le correcteur de l'article NON
REPRODUCTIBLE au sens strict. Ce module implemente sa METHODE avec des
ponderations choisies ici, et tout resultat obtenu doit etre lu ainsi.

--------------------------------------------------------------------------------
DEVIATION ASSUMEE : mu STRUCTURE
--------------------------------------------------------------------------------
La synthese mu de l'article porte une incertitude STRUCTUREE a dix parametres
reels (Eq. 22). Une iteration D-K sur une telle structure demande un calcul de
borne superieure de mu que python-control n'offre pas. On fait donc :

  * incertitude ADDITIVE (Eqs. 17-19, publiee) : incluse dans le probleme H-inf
    et dans l'iteration D-K, avec des echelles D scalaires par bloc ;
  * incertitude PARAMETRIQUE (Eq. 22, publiee elle aussi) : NON incluse dans la
    synthese ; elle est verifiee A POSTERIORI en simulation, sur les memes
    plages, par le tableau de perturbations de run_demo --full.

C'est plus faible que ce que fait l'article. Le dire est preferable a laisser
croire a une synthese mu complete.
--------------------------------------------------------------------------------
CE QUE CETTE IMPLEMENTATION DONNE, ET CE QU'ELLE NE DONNE PAS ENCORE
--------------------------------------------------------------------------------
1. RETARD ACTIF (Eq. 30) : IMPLEMENTE ET EFFICACE, et il reproduit l'affirmation
   centrale de l'article a son sujet -- "a lower voltage cost".

   L'Eq. (31) suggere elle-meme l'ordre de grandeur de Kp : pour annuler le
   terme regeneratif il faut H Kp ~ -a4 D^2/D_obs, soit Kp ~ 2.44e6 V/m. Le
   balayage donne l'optimum a 4.88e6, c.-a-d. DEUX fois cette valeur, et avec
   le SIGNE que predit l'Eq. (31) -- Kp negatif echoue a toutes les valeurs
   essayees. Le raisonnement d'annulation de l'article donne donc le bon signe
   et le bon ordre de grandeur.

       a 4900 tr/min :  libre 0.0501 mm  ->  retard actif 0.3241 mm
                        pour 18.9 V de crete seulement.

   RECTIFICATION. J'ai d'abord ecrit que c'etait "la MEME limite que le
   meilleur LQG" parce que les deux affichaient 0.3241. C'etait un ARTEFACT DE
   QUANTIFICATION : la bissection par defaut a un pas de 0.001943 mm, et deux
   limites plus proches que cela tombent sur la meme valeur imprimee. En
   resserrant la tolerance d'un facteur 30 :

       LQG modal        0.324789 mm
       Eq. (30)         0.324363 mm

   soit 0.13 % d'ecart -- ni identiques, ni distinguables au reglage publie, et
   AUCUNE des deux n'est egale au 0.3241 affiche. Ce qui reste vrai et n'est pas
   affecte est la TENSION : 18.9 V de crete contre 33 V, et 5.3 V a 0.10 mm.

2. MAIS UN SEUL COUPLE (Kp, Kd) NE TIENT PAS SUR LA PLAGE DE VITESSES :

       tr/min    3000    4200    4900    6000    7200
       libre   0.0676  0.0734  0.0501  0.1026  0.2153
       Eq.(30) 0.0000  0.0000  0.3241  0.0000  0.0000

   Regle sur 4900 tr/min, il est PIRE que la boucle ouverte aux quatre autres
   vitesses. C'est logique : tau change avec la vitesse, et un gain fixe sur un
   retard fixe n'est accorde qu'a une seule vitesse.

   ET C'EST EXACTEMENT CE QUE DIT L'ARTICLE : "Milling under delayed PD control
   is still unstable at start and end positions... Single active control is not
   sufficient to suppress milling chatter of flexible workpieces." Sa Fig. 14
   montre la meme chose. Ce resultat-la est donc REPRODUIT, y compris dans son
   echec.

3. ROBUSTE PAR SYNTHESE mu : IMPLEMENTE MAIS PAS ENCORE EFFICACE. Le pipeline
   tourne (plant generalise, hinfsyn, iterations D-K, mise a l'echelle), mais
   les ponderations de performance ne sont pas accordees : le correcteur obtenu
   sature l'amplificateur et n'ameliore pas la limite (0.0482 contre 0.0501 mm
   en boucle ouverte). La cause n'est pas mysterieuse -- W_f, W_u et W_n ne
   sont pas publiees, et ce sont elles qui fixent entierement ce compromis.
   Tant qu'elles ne sont pas accordees, la commande COMBINEE de l'article n'a
   pas de sens a evaluer, puisque l'une de ses deux moities ne fonctionne pas.

--------------------------------------------------------------------------------
UN SOUS-PRODUIT : LES POIDS PUBLIES VALIDENT LE NIVEAU DU MODELE
--------------------------------------------------------------------------------
Les Eqs. (18)-(19) sont censees MAJORER la reponse des modes 3 et plus. En les
comparant a ce que donne le modele :

    canal tension (m/V) : max|W_au| = 9.08e-7  contre  3.81e-7  -> marge 2.39
    canal effort  (m/N) : max|W_af| = 5.51e-5  contre  4.02e-5  -> marge 1.37

Les deux MAJORENT bien, sur 100 % de la bande 1-10 kHz, avec des marges de
concepteur ordinaires. Or le canal tension porte le produit H_Pe * D_obs :
c'est donc une contrainte sur le NIVEAU de H_Pe -- precisement la grandeur que
le defaut F9 declare non validee, et ici elle vient d'une EQUATION de l'article,
pas de l'echelle en dB de ses figures, qui est inexploitable.

Consequence immediate : g <= 2.39, sans quoi le poids publie ne couvrirait plus
rien. Cela AMPUTE le haut de l'encadrement [0.44, 2.92] etabli au script 17 --
et c'est le cote qui compte, celui qui aneantit l'ESO. Voir verification/20.
================================================================================
"""
import numpy as np
import scipy.signal as sig
import scipy.linalg as la

# --- Eqs. (18)-(19), valeurs de l'article ------------------------------------
W_AF = dict(r=14e-6, z1=0.56, z2=0.12, w1=2*np.pi*1400, w2=2*np.pi*2800)
W_AU = dict(r=4e-7, z1=0.58, z2=0.22, w1=2*np.pi*1100, w2=2*np.pi*3500)

# --- Eq. (23) ----------------------------------------------------------------
A4_NOM = 1.6          # a4_0 = 1.6 * a4_moyen
A4_PERT = 1.3         # L_alpha = 1.3 * a4_moyen

# --- Sec. 3.2 ----------------------------------------------------------------
PERT_MK = 0.10        # masse et raideur modales
PERT_Z = 0.20         # amortissement

N_RED = 2             # modele reduit aux deux premiers modes (Sec. 3.1)

# Kp et Kd de l'Eq. (30). NON DONNES PAR L'ARTICLE : trouves ici par balayage a
# 4900 tr/min, la condition que l'article simule (Figs. 14-15). Kp vaut deux
# fois l'estimation d'annulation de l'Eq. (31), et en porte le signe.
KP_DELAY = 4.8761e6   # V/m
KD_DELAY = 0.0        # V/(m/s)


def _w_additive(par):
    """(num, den) de l'Eq. (18)/(19) : r (s^2+2z1w1 s+w1^2)/(s^2+2z2w2 s+w2^2)."""
    num = par['r']*np.array([1.0, 2*par['z1']*par['w1'], par['w1']**2])
    den = np.array([1.0, 2*par['z2']*par['w2'], par['w2']**2])
    return num, den


def reduced_plant(plate, alpha4_mean, DD_nom):
    """Modele reduit de l'Eq. (16), deux modes, au point de fonctionnement
    nominal de l'Eq. (25).

    Etat x = [q1, q2, q1', q2'] ; entrees (f, u) ; sortie y = deplacement au
    capteur. f est un effort SCALAIRE au point de contact outil, comme dans les
    FRF des Figs. 4 et 5.
    """
    n = N_RED
    om = np.asarray(plate.omega_n, float)[:n]
    ze = np.asarray(plate.zeta_modes, float)[:n]
    H = np.asarray(plate.H_Pe_modal, float)[:n]
    Dobs = np.asarray(plate.D_obs, float)[:n]
    Dtool = np.asarray(plate.Dp_array, float)[:n, 0]

    K0 = np.diag(om**2)
    C0 = np.diag(2*ze*om)
    Keff = K0 + A4_NOM*alpha4_mean*DD_nom          # Eq. (25)

    A = np.block([[np.zeros((n, n)), np.eye(n)],
                  [-Keff, -C0]])
    Bf = np.concatenate([np.zeros(n), Dtool])[:, None]
    Bu = np.concatenate([np.zeros(n), H])[:, None]
    Cy = np.concatenate([Dobs, np.zeros(n)])[None, :]
    return A, Bf, Bu, Cy


def _ss_of_tf(num, den):
    A, B, C, D = sig.tf2ss(num, den)
    return np.atleast_2d(A), np.atleast_2d(B), np.atleast_2d(C), np.atleast_2d(D)


# Echelles de mise en forme du probleme de synthese. Sans elles, hinfsyn rend
# K = 0 : la plaque vaut ~1e-9 m/V, la mesure ~1e-6 m et les ponderations
# additives ~1e-7, si bien que le canal de commande disparait numeriquement
# devant les autres. Verifie : gamma croissait proportionnellement a wp_gain
# tandis que le gain du correcteur restait a 5e-10 V par micrometre, c.-a-d.
# que la solution optimale etait "ne rien faire". On synthetise donc en
# variables normalisees u/U0 et y/Y0, et on defait l'echelle a la fin.
U0 = 150.0            # V, pleine echelle de l'amplificateur
Y0 = 1e-6             # m, micrometre


def generalized_plant(plate, alpha4_mean, DD_nom, wu_gain, wp_gain,
                      wf_pole=2*np.pi*2000.0, wn_gain=1e-3):
    """Plant generalise P pour la synthese, structure de la Fig. 9.

    Entrees   : [w_a (Delta additive) ; d (effort) ; n (bruit) ; u]
    Sorties   : [z_a1, z_a2 (vers Delta) ; z_p (performance) ; z_u (tension) ;
                 y (mesure)]

    wu_gain, wp_gain : LES PONDERATIONS QUE L'ARTICLE NE DONNE PAS. wp_gain
        pese le deplacement au capteur, wu_gain la tension. Leur RAPPORT fixe
        le compromis ; c'est le seul reglage libre de cette synthese.
    wf_pole : coupure du passe-bas W_f(s) sur l'effort de coupe. L'article dit
        "low-pass" sans valeur ; 2 kHz couvre les deux modes conserves.
    """
    A, Bf, Bu, Cy = reduced_plant(plate, alpha4_mean, DD_nom)
    Bu = Bu*U0                      # entree normalisee u/U0
    Cy = Cy/Y0                      # sortie normalisee y/Y0
    n = A.shape[0]

    Aaf, Baf, Caf, Daf = _ss_of_tf(*_w_additive(W_AF))
    Aau, Bau, Cau, Dau = _ss_of_tf(*_w_additive(W_AU))
    naf, nau = Aaf.shape[0], Aau.shape[0]

    # W_f(s) : passe-bas d'ordre 1 sur l'effort
    Af = np.array([[-wf_pole]])
    Bf_w = np.array([[wf_pole]])
    Cf = np.array([[1.0]])
    Df = np.array([[0.0]])
    nf = 1

    N = n + naf + nau + nf
    Ag = np.zeros((N, N))
    Ag[:n, :n] = A
    Ag[n:n+naf, n:n+naf] = Aaf
    Ag[n+naf:n+naf+nau, n+naf:n+naf+nau] = Aau
    Ag[n+naf+nau:, n+naf+nau:] = Af

    # l'effort filtre entre dans la plaque ; W_af est pilote par le MEME effort
    # filtre, W_au par la tension
    def col(*blocks):
        return np.concatenate(blocks, axis=0)

    z = lambda r, c=1: np.zeros((r, c))
    B_wa = col(z(n), z(naf), z(nau), z(nf))                 # Delta -> sortie y
    B_d = col(z(n), z(naf), z(nau), Bf_w)                   # d -> W_f
    B_n = col(z(n), z(naf), z(nau), z(nf))                  # bruit -> mesure
    B_u = col(Bu, z(naf), Bau*U0, z(nf))                    # u -> plaque et W_au
    # sortie de W_f : effort reel applique a la plaque et a W_af
    Ag[:n, n+naf+nau:] += Bf @ Cf
    Ag[n:n+naf, n+naf+nau:] += Baf @ Cf
    B_d[:n] += Bf @ Df
    B_d[n:n+naf] += Baf @ Df

    Bg = np.hstack([B_wa, B_d, B_n, B_u])

    Cy_g = np.zeros((1, N))
    Cy_g[0, :n] = Cy
    # les sorties vers Delta sont en metres ; on les normalise par Y0 comme y,
    # ce qui laisse Delta adimensionnel et borne par 1
    Cza1 = np.zeros((1, N))
    Cza1[0, n:n+naf] = Caf/Y0
    Cza2 = np.zeros((1, N))
    Cza2[0, n+naf:n+naf+nau] = Cau/Y0
    Czp = wp_gain*Cy_g
    Czu = np.zeros((1, N))

    Cg = np.vstack([Cza1, Cza2, Czp, Czu, Cy_g])
    Dg = np.zeros((5, 4))
    Dg[0, 1] = float(np.asarray(Daf @ Df).ravel()[0])/Y0   # d -> W_f -> W_af
    Dg[1, 3] = float(np.asarray(Dau).ravel()[0])*U0/Y0     # u -> z_a2
    Dg[3, 3] = wu_gain                  # u -> z_u
    Dg[4, 0] = 1.0                      # Delta -> y (incertitude ADDITIVE)
    Dg[4, 2] = wn_gain                  # bruit -> y
    return Ag, Bg, Cg, Dg


def synthesize(plate, alpha4_mean, DD_nom, wu_gain=2e-3, wp_gain=3e5,
               dk_iters=3, verbose=True):
    """Synthese H-inf, puis quelques iterations D-K a echelles SCALAIRES.

    Retourne (Ak, Bk, Ck, Dk) continu et gamma atteint.
    """
    import control

    best = None
    d = np.ones(2)                    # echelles D des deux blocs additifs
    for it in range(max(1, dk_iters)):
        Ag, Bg, Cg, Dg = generalized_plant(plate, alpha4_mean, DD_nom,
                                           wu_gain, wp_gain)
        # mise a l'echelle D : z_a <- d z_a, w_a <- d^-1 w_a
        Cg = Cg.copy()
        Bg = Bg.copy()
        Dg = Dg.copy()
        Cg[0, :] *= d[0]
        Dg[0, :] *= d[0]
        Cg[1, :] *= d[1]
        Dg[1, :] *= d[1]
        dm = 0.5*(d[0] + d[1])        # Delta_a est 1x2 : une seule colonne w_a
        Bg[:, 0] /= dm
        Dg[:, 0] /= dm
        P = control.ss(Ag, Bg, Cg, Dg)
        try:
            K, CL, gam, _ = control.hinfsyn(P, 1, 1)
        except Exception as e:
            if verbose:
                print(f'   [D-K {it}] hinfsyn a echoue : {e}')
            break
        if verbose:
            print(f'   [D-K {it}] gamma = {gam:.4f}   (D = {d.round(3)})')
        if best is None or gam < best[1]:
            # K synthetise sur (y/Y0) -> (u/U0) ; on revient aux volts et
            # aux metres : u = (U0/Y0) K_s y
            sc = U0/Y0
            best = ((np.asarray(K.A), np.asarray(K.B),
                     np.asarray(K.C)*sc, np.asarray(K.D)*sc), gam)
        # remise a l'echelle : on egalise la contribution des deux blocs
        Tw = control.ss(Ag, Bg, Cg, Dg)
        w = np.logspace(1.5, 4.5, 200)*2*np.pi
        try:
            CLs = control.ss(CL)
            mag = np.array([np.abs(CLs(1j*wi)) for wi in w])
            g1 = mag[:, 0, :].max()
            g2 = mag[:, 1, :].max()
            r = np.sqrt(max(g1, 1e-12)/max(g2, 1e-12))
            d = np.array([d[0]/np.sqrt(r), d[1]*np.sqrt(r)])
            d /= d.mean()
        except Exception:
            break
    if best is None:
        raise RuntimeError('synthese H-inf impossible')
    return best[0], best[1]


# ============================================================================
# CORRECTEURS UTILISABLES PAR LE SIMULATEUR
# ============================================================================
class DelayedPD:
    """Eq. (30) :  u(t) = Kp y(t-tau) + Kd y'(t-tau).

    L'article appelle cela "active time delay control", et le compare seul
    (courbe "Delayed PD" de sa Fig. 14) puis combine au robuste.

    dt = tau/82 exactement, donc y(t-tau) est le 82e echantillon en arriere :
    aucune interpolation n'est necessaire, ce qui est le seul cas ou un retard
    pur se simule sans erreur.

    Kp et Kd ne sont PAS donnes par l'article.
    """

    def __init__(self, dt, tau, Kp, Kd):
        self.dt = float(dt)
        self.n_tau = int(round(tau/dt))
        self.Kp = float(Kp)
        self.Kd = float(Kd)
        self.reset()

    def reset(self):
        self.buf = np.zeros(self.n_tau + 2)
        self.i = 0

    def _push(self, y):
        self.buf[self.i] = y
        self.i = (self.i + 1) % len(self.buf)

    def _delayed(self, k):
        return self.buf[(self.i - 1 - k) % len(self.buf)]

    def step(self, x_hat_prev, u_prev, y_meas):
        self._push(y_meas)
        yd = self._delayed(self.n_tau)
        yd1 = self._delayed(self.n_tau + 1)
        u = self.Kp*yd + self.Kd*(yd - yd1)/self.dt
        return x_hat_prev, u


class RobustMu:
    """Correcteur robuste K_mu(s), discretise a dt. u = Ck x + Dk y."""

    def __init__(self, dt, Ak, Bk, Ck, Dk):
        Ad, Bd, Cd, Dd, _ = sig.cont2discrete((Ak, Bk, Ck, Dk), dt,
                                              method='zoh')
        self.Ad, self.Bd, self.Cd, self.Dd = Ad, Bd, Cd, Dd
        self.n = Ad.shape[0]
        self.reset()

    def reset(self):
        self.x = np.zeros(self.n)

    def step(self, x_hat_prev, u_prev, y_meas):
        u = float(np.asarray(self.Cd @ self.x).ravel()[0]
                  + self.Dd[0, 0]*y_meas)
        self.x = self.Ad @ self.x + self.Bd[:, 0]*y_meas
        return x_hat_prev, u


class Combined:
    """La methode de l'article : robuste + retard actif, Sec. 3.4."""

    def __init__(self, robust, delayed):
        self.r = robust
        self.d = delayed

    def reset(self):
        self.r.reset()
        self.d.reset()

    def step(self, x_hat_prev, u_prev, y_meas):
        _, ur = self.r.step(x_hat_prev, u_prev, y_meas)
        _, ud = self.d.step(x_hat_prev, u_prev, y_meas)
        return x_hat_prev, ur + ud


# ============================================================================
def alpha4_statistics(sim, rpm, ap):
    """a4 moyen sur une periode, et les elements de D_r^T D_r le long de
    l'arete : nominal (max+min)/2 et demi-amplitude, comme l'Eq. (24)."""
    import simulation_base as SB
    from milling_force import precompute_alpha_periodic
    tau = 60.0/(SB.N_TEETH*rpm)
    dt = tau/SB.STEPS_PER_TOOTH
    phi = np.pi - np.arccos(1 - SB.AE_NOM/(SB.D_TOOL/2))
    _, a4 = precompute_alpha_periodic(
        dt, SB.STEPS_PER_TOOTH, SB.STEPS_PER_TOOTH, 2*np.pi*rpm/60,
        SB.N_TEETH, SB.D_TOOL/2, SB.HELIX, phi, np.pi,
        SB.PLATE_H - ap, SB.PLATE_H, sim.k1c, sim.k2c, SB.KT)
    Dp = np.asarray(sim.plate.Dp_array, float)[:N_RED, :]
    E = np.einsum('ik,jk->ijk', Dp, Dp)
    DD_nom = 0.5*(E.max(axis=2) + E.min(axis=2))
    DD_pert = 0.5*(E.max(axis=2) - E.min(axis=2))
    return float(a4.mean()), DD_nom, DD_pert
