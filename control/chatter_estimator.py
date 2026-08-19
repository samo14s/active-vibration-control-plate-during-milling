"""
chatter_estimator.py — estimation EN LIGNE du broutement (frequence + niveau)
=============================================================================
Le maillon faible mesure de l'observateur modal (`OBSERVATEUR_MODAL.md` §6.5)
est qu'il suppose connaitre la frequence du mode : a la derive modale
reellement constatee dans le papier (+17 % / +9 %), sa limite tombe de 0.222 a
0.117 mm, parce qu'un passe-bande de largeur relative 0.7 % a le mode
entierement hors de sa fenetre. Ce module fournit ce qu'il faut pour
recentrer la fenetre en ligne — et pour ne le faire QUE quand il y a
quelque chose a recentrer.

LA DIFFICULTE, ET ELLE EST REELLE. Le signal de coupe est domine par la
reponse FORCEE aux harmoniques de passage de dent, pas par le broutement.
A 4900 tr/min et 3 dents, f_dent = 245 Hz, donc les harmoniques tombent a
490, 735, 980 et 1225 Hz dans la bande protegee 400-1300 Hz. Or les modes de
broutement sont a 540 et 1068 Hz, soit a 10 % et 8 % des harmoniques 2 et 4.
Un estimateur naif verrouille sur l'harmonique, pas sur le mode — et le
recentrage ferait alors exactement le contraire de ce qu'on veut.

D'ou la chaine, dans cet ordre :

  1. DECIMATION d'un facteur 8. A n_sub = 656 la simulation echantillonne a
     161 kHz ; le mode le plus haut est a 4122 Hz, donc Nyquist apres
     decimation (10 kHz) reste largement au-dessus et il n'y a pas de
     repliement a redouter.
  2. PEIGNE DE REJECTION aux harmoniques de passage de dent. Leurs frequences
     ne sont pas estimees : elles se DEDUISENT de la vitesse de broche et du
     nombre de dents, qui sont connus de la commande numerique. Les encoches
     sont etroites (zeta_n = 0.005) pour ne pas toucher un mode situe a 8 %.
  3. PASSE-BANDE sur la bande protegee, pour ignorer le continu, la reponse
     forcee basse frequence et les modes hauts.
  4. ESTIMATION DE FREQUENCE par moindres carres recursifs sur un AR(2)
     contraint : pour une sinusoide pure, y[k] + y[k-2] = a y[k-1] avec
     a = 2 cos(w T). Un seul parametre, donc une convergence rapide et un
     seul reglage (le facteur d'oubli).
  5. NIVEAU DE BROUTEMENT : rapport de la valeur efficace du signal filtre a
     celle du signal brut, lisse exponentiellement. C'est un nombre sans
     dimension qui monte quand du contenu NON harmonique apparait, donc
     exactement l'evenement qu'on veut detecter.

Le verrou de frequence n'est mis a jour que lorsque le niveau depasse un
seuil : sous ce seuil il n'y a rien a estimer et l'estimateur ne ferait que
suivre du bruit.

REGLAGES, CHOISIS PAR MESURE. Sur signaux synthetiques (harmoniques de dent
plus une sinusoide a 20 % d'amplitude), l'erreur d'estimation finale vaut :

    oubli 0.9990 -> 0.07 % (540 Hz), 0.12 % (1068), 0.08 % (632 = derive +17 %)
    oubli 0.9995 -> 0.52 %, 0.76 %, 0.52 %
    oubli 0.9998 -> 2.94 %, 4.31 %, 3.07 %

et le niveau du cas SANS broutement vaut 0.0036. D'ou lam = 0.999.

La PORTE, elle, se regle sur le signal de coupe reel et non sur des
sinusoides. Fraction du temps ou la porte s'ouvre a tort en passe stable
(a_p = 0.15 mm), et frequence finale sur une passe qui broute (0.25 mm) puis
sur une qui diverge (0.35 mm) :

    tau_g = 0.010 s, n_warm = 2 :   9 % a tort   579.6 Hz   638.3 Hz
    tau_g = 0.020 s, n_warm = 2 :   0 % a tort   579.6 Hz   639.0 Hz
    tau_g = 0.050 s, n_warm = 2 :   0 % a tort   579.6 Hz   — porte JAMAIS
                                    ouverte : la passe a diverge avant la fin
                                    du temps de chauffe

D'ou tau_gate = 20 ms et n_warm = 2. Le dernier cas est la raison d'etre du
reglage : une porte trop lente arrive apres la bataille.
"""
import numpy as np


class Biquad:
    """Section du second ordre en forme directe II transposee."""

    __slots__ = ('b', 'a', 'z1', 'z2')

    def __init__(self, b, a):
        self.set(b, a)
        self.z1 = 0.0
        self.z2 = 0.0

    def set(self, b, a):
        self.b = np.asarray(b, float) / a[0]
        self.a = np.asarray(a, float) / a[0]

    def reset(self):
        self.z1 = self.z2 = 0.0

    def __call__(self, x):
        y = self.b[0] * x + self.z1
        self.z1 = self.b[1] * x - self.a[1] * y + self.z2
        self.z2 = self.b[2] * x - self.a[2] * y
        return y


def _prewarp(f, fs):
    """Pulsation analogique qui, apres bilineaire, retombe EXACTEMENT sur f.

    Sans cette pre-distorsion l'encoche a 490 Hz atterrit a 488.8 Hz pour
    fs = 20 kHz : 0.24 % d'ecart, alors que sa demi-largeur a zeta = 0.005
    ne vaut que 0.5 %. Mesure : l'attenuation a l'harmonique tombe alors de
    ~0.001 a 0.30, et le peigne ne sert plus a rien. Ce n'est donc pas un
    raffinement, c'est la condition pour que le module fonctionne.
    """
    return 2.0 * fs * np.tan(np.pi * f / fs)


def _bilinear(num, den, fs):
    from scipy.signal import bilinear
    return bilinear(num, den, fs)


def notch(f0, zeta, fs):
    """Encoche (s^2 + w0^2) / (s^2 + 2 zeta w0 s + w0^2), centree sur f0."""
    w0 = _prewarp(f0, fs)
    return _bilinear([1.0, 0.0, w0 ** 2], [1.0, 2 * zeta * w0, w0 ** 2], fs)


def bandpass(f1, f2, fs):
    """Passe-bande du second ordre entre f1 et f2."""
    w1, w2 = _prewarp(f1, fs), _prewarp(f2, fs)
    bw, w0 = w2 - w1, np.sqrt(w1 * w2)
    return _bilinear([bw, 0.0], [1.0, bw, w0 ** 2], fs)


class ChatterEstimator:
    """Frequence dominante et niveau de broutement, en ligne.

    Parametres (tous fixes, aucun n'est ajuste par l'optimiseur) :
      f_tooth   frequence de passage de dent [Hz], connue de la commande
      band      bande protegee (f1, f2) [Hz]
      decim     facteur de decimation
      lam       facteur d'oubli des moindres carres recursifs
      tau       constante de temps du lissage des valeurs efficaces [s]
      level_on  seuil de niveau au-dessus duquel le verrou de frequence suit
    """

    def __init__(self, fs, f_tooth, band=(400.0, 1300.0), decim=8,
                 lam=0.999, tau=5e-3, tau_gate=20e-3, level_on=0.25, n_warm=2.0,
                 f_nom=None, f_init=None):
        self.decim = int(decim)
        self.fs = fs / self.decim
        self.band = band
        self.lam = float(lam)
        self.level_on = float(level_on)
        # Toutes les harmoniques jusqu'au haut de la bande, la fondamentale
        # comprise : le passe-bande seul ne l'attenue qu'a 0.43, ce qui
        # suffirait a fausser l'indicateur de niveau.
        self.harmonics = [k * f_tooth for k in range(1, 60)
                          if k * f_tooth < band[1] * 1.3]
        self.combs = [Biquad(*notch(f, 0.005, self.fs))
                      for f in self.harmonics]
        self.bp = Biquad(*bandpass(band[0], band[1], self.fs))
        # Encoches d'EXCLUSION, mises a jour en ligne : elles servent a
        # retirer les modes que d'AUTRES estimateurs suivent. Sans elles, un
        # mode voisin qui a derive vers le bas de la sous-bande tire les
        # moindres carres vers le bord : mesure sur la plaque derivee, le
        # second estimateur se verrouillait a 763 Hz — le bas de sa bande —
        # au lieu de 1164 Hz, et le recentrage placait alors le passe-bande
        # de l'observateur la ou il n'y a aucun mode.
        self.excl = []
        self.excl_f = []
        self.beta = float(np.exp(-1.0 / (tau * self.fs)))
        # Une SECONDE mesure de niveau, dix fois plus lente, sert de porte.
        # Mesure sur signal de coupe simule : en passe STABLE le niveau rapide
        # atteint 0.13 a 0.35 en transitoire alors que sa valeur etablie reste
        # sous 0.05 ; en broutement il monte a 0.86-0.98 et y reste. Une porte
        # sur le niveau rapide se laisserait donc tromper par les transitoires
        # d'entree en coupe. La porte lente les ignore.
        self.beta_g = float(np.exp(-1.0 / (tau_gate * self.fs)))
        self.n_warm = float(n_warm) / (1 - self.beta_g)
        self._first = True
        self.p_raw_g = self.p_flt_g = 0.0
        self.level_slow = 0.0
        f0 = np.sqrt(band[0] * band[1]) if f_init is None else f_init
        self.f_nom = f0 if f_nom is None else float(f_nom)
        self.a = 2.0 * np.cos(2 * np.pi * f0 / self.fs)
        self.P = 1.0e3
        self.y1 = self.y2 = 0.0
        self.p_raw = self.p_flt = 0.0
        self.f_hat = f0
        self.level = 0.0
        # Suivi de la DISPERSION de l'estimee. Un estimateur dont la
        # sous-bande ne contient aucun broutement suit du bruit : sa sortie
        # erre au lieu de se poser. Mesure sur la plaque derivee, ou seul le
        # mode 1 broute : l'estimateur du mode 2 se posait a 760-910 Hz au
        # lieu de 1164 Hz, avec une dispersion relative dix fois superieure a
        # celle de l'estimateur du mode 1. On ne declare donc "verrouille"
        # qu'une estimee dont l'ecart-type relatif est sous `lock_tol`.
        self.f_bar = f0
        self.f_var = 0.0
        self.locked = False
        self.conf = 0.0
        self.lock_tol = 0.01
        # CONFIANCE continue, entre 0 et 1, deduite de la meme dispersion que
        # le verrou binaire. Le verrou repond a "faut-il recentrer ?" ; la
        # confiance repond a "de combien faut-il y croire ?", et c'est elle
        # qui module l'autorite donnee a l'observateur.
        self.conf_lo = 0.004
        self.conf_hi = 0.020
        self.conf = 0.0
        self._i = 0

    def set_exclude(self, freqs):
        """Frequences a retirer du signal avant estimation (celles que les
        autres estimateurs suivent). Les encoches ne sont refabriquees que si
        la frequence a bouge de plus de 2 %."""
        fs = [f for f in freqs if 0.5 * self.band[0] < f < 2 * self.band[1]]
        if len(fs) != len(self.excl_f):
            self.excl = [Biquad(*notch(f, 0.02, self.fs)) for f in fs]
            self.excl_f = list(fs)
            return
        for i, f in enumerate(fs):
            if abs(f / max(self.excl_f[i], 1e-9) - 1.0) > 0.02:
                self.excl[i].set(*notch(f, 0.02, self.fs))
                self.excl_f[i] = f

    def reset(self):
        for b in self.combs + self.excl:
            b.reset()
        self.bp.reset()
        self.y1 = self.y2 = 0.0
        self.p_raw = self.p_flt = 0.0
        self.p_raw_g = self.p_flt_g = 0.0
        self.level_slow = 0.0
        self._first = True
        self.f_hat = self.f_bar = self.f_nom
        self.f_var = 0.0
        self.locked = False
        self.conf = 0.0
        self.P = 1.0e3
        self._i = 0

    def __call__(self, y):
        """Un echantillon a la cadence PLEINE ; ne travaille qu'un sur `decim`."""
        self._i += 1
        if self._i % self.decim:
            return self.f_hat, self.level
        v = float(y)
        for b in self.combs:
            v = b(v)
        for b in self.excl:
            v = b(v)
        v = self.bp(v)
        # niveaux efficaces lisses
        self.p_raw = self.beta * self.p_raw + (1 - self.beta) * float(y) ** 2
        self.p_flt = self.beta * self.p_flt + (1 - self.beta) * v * v
        self.level = float(np.sqrt(self.p_flt / max(self.p_raw, 1e-30)))
        if self._first:
            # Amorcage sur le premier echantillon traite : sans lui le rapport
            # part de 0/0 et met plusieurs constantes de temps a devenir
            # interpretable, ce qui retarde la porte au-dela du temps que met
            # une passe instable a diverger.
            self.p_raw = self.p_raw_g = float(y) ** 2
            self.p_flt = self.p_flt_g = v * v
            self._first = False
        self.p_raw_g = self.beta_g * self.p_raw_g + (1 - self.beta_g) * float(y) ** 2
        self.p_flt_g = self.beta_g * self.p_flt_g + (1 - self.beta_g) * v * v
        self.level_slow = float(np.sqrt(self.p_flt_g
                                        / max(self.p_raw_g, 1e-30)))
        # moindres carres recursifs sur  y[k] + y[k-2] = a y[k-1]
        # NORMALISATION. Le signal est en metres, donc de l'ordre de 1e-6 :
        # avec P = 1e3 le gain des moindres carres vaut phi P / (lam + phi^2 P)
        # ~ 1e-4 et la correction ~ 1e-11 par pas, c'est-a-dire rien. On
        # divise donc par la valeur efficace courante du signal filtre, ce qui
        # rend le regresseur d'ordre 1 sans changer la frequence estimee.
        sc = np.sqrt(max(self.p_flt, 1e-30))
        # Temps de chauffe : tant que les valeurs efficaces lentes n'ont pas
        # vu trois constantes de temps, leur rapport n'a pas de sens et vaut
        # transitoirement jusqu'a 0.83 meme en passe parfaitement stable.
        warm = self._i * (1.0 / self.decim) > self.n_warm
        if warm and self.level_slow > self.level_on \
                and abs(self.y1) > 1e-12 * sc:
            phi = self.y1 / sc
            z = (v + self.y2) / sc
            g = self.P * phi / (self.lam + phi * self.P * phi)
            self.a += g * (z - self.a * phi)
            self.P = (self.P - g * phi * self.P) / self.lam
            self.P = min(self.P, 1.0e12)
            c = np.clip(0.5 * self.a, -1.0, 1.0)
            f = float(np.arccos(c)) * self.fs / (2 * np.pi)
            if self.band[0] <= f <= self.band[1]:
                self.f_hat = f
                b = self.beta_g
                self.f_bar = b * self.f_bar + (1 - b) * f
                self.f_var = b * self.f_var + (1 - b) * (f - self.f_bar) ** 2
                sd = np.sqrt(self.f_var) / max(self.f_bar, 1e-9)
                self.locked = sd < self.lock_tol
                self.conf = float(np.clip(
                    (self.conf_hi - sd) / (self.conf_hi - self.conf_lo),
                    0.0, 1.0))
        else:
            # Pas de broutement : on RAMENE lentement le verrou vers la valeur
            # nominale au lieu de le laisser ou un transitoire l'avait mene.
            # Sans cela, une passe stable finit avec f_hat sur un bord de bande
            # et le superviseur recentrerait l'observateur LOIN des modes.
            self.f_hat += (1 - self.beta_g) * (self.f_nom - self.f_hat)
            self.f_bar, self.f_var = self.f_hat, 0.0
            self.locked = False
            self.conf = 0.0
            self.P = 1.0e3
        self.y2, self.y1 = self.y1, v
        return self.f_hat, self.level
