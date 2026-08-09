"""
fopid_controller.py
===================
Contrôleur FOPID (PI^lambda D^mu, fractional-order PID) pour la suppression
du broutement, compatible avec l'interface de NewmarkSimulator :

    x_hat, u = controller.step(x_hat_prev, u_prev, y_meas)

Loi de commande (sortie unique y = déplacement observé, entrée = tension patch) :

    e(t)  = y_meas / y_ref                     (erreur normalisée, cible 0)
    u(t)  = sat( Kp e + Ki I^lambda[e] + Kd D^mu[e] ,  +/- V_max )

Les opérateurs fractionnaires (s/w_c)^(+mu) et (s/w_c)^(-lambda) sont
approximés par le filtre récursif d'Oustaloup à bande limitée [w_b, w_h],
converti en espace d'état et discrétisé par transformation bilinéaire (Tustin)
au pas dt du simulateur. La normalisation par w_c rend les gains Kp, Ki, Kd
du même ordre de grandeur (des volts par y_ref) autour de la bande critique.

Références : Oustaloup et al. (2000) ; Podlubny (1999), "Fractional-order
systems and PI^lambda D^mu controllers".
"""
import numpy as np
# (scipy.signal non requis : réalisation en cascade maison)


# ---------------------------------------------------------------------------
def oustaloup_zpk(alpha: float, w_b: float, w_h: float, N: int = 4):
    """
    Approximation d'Oustaloup de s^alpha sur [w_b, w_h] (rad/s), 2N+1 cellules.
    Retourne (zeros, poles) SANS le gain (à normaliser par l'appelant).
    Valide pour alpha dans [-1, 1].
    """
    k = np.arange(-N, N + 1)
    r = w_h / w_b
    w_z = w_b * r ** ((k + N + 0.5 * (1 - alpha)) / (2 * N + 1))
    w_p = w_b * r ** ((k + N + 0.5 * (1 + alpha)) / (2 * N + 1))
    return -w_z, -w_p


class FractionalOperator:
    """
    Filtre discret réalisant (s/w_c)^alpha, alpha dans [-1, 1].

    Réalisation en CASCADE de sections du premier ordre (s+w_z)/(s+w_p),
    chacune discrétisée par Tustin — parfaitement conditionnée, contrairement
    à une forme compagnon globale (les pôles d'Oustaloup s'étalent sur
    plusieurs décades et rendent zpk2ss/cont2discrete singulières).

    Section continue (s+w_z)/(s+w_p), Tustin s = (2/dt)(1-z^-1)/(1+z^-1) :
        b0 = (K2+w_z)/(K2+w_p), b1 = (w_z-K2)/(K2+w_p), a1 = (w_p-K2)/(K2+w_p)
    avec K2 = 2/dt. Le gain global est normalisé pour |G(e^{j w_c dt})| = 1.
    """

    def __init__(self, alpha: float, dt: float,
                 w_c: float, w_b: float, w_h: float, N: int = 4):
        self.alpha = float(alpha)
        if abs(self.alpha) < 1e-9:                      # s^0 = identité
            self._identity = True
            return
        self._identity = False
        z, p = oustaloup_zpk(self.alpha, w_b, w_h, N)
        w_z, w_p = -z, -p
        K2 = 2.0 / dt
        self.b0 = (K2 + w_z) / (K2 + w_p)
        self.b1 = (w_z - K2) / (K2 + w_p)
        self.a1 = (w_p - K2) / (K2 + w_p)
        self.ns = len(w_z)
        # normalisation : |G(e^{j w_c dt})| = 1
        zc = np.exp(1j * w_c * dt)
        G = np.prod((self.b0 + self.b1 / zc) / (1.0 + self.a1 / zc))
        self.gain = 1.0 / max(abs(G), 1e-300)
        self.x_in = np.zeros(self.ns)     # entrée précédente de chaque section
        self.y_out = np.zeros(self.ns)    # sortie précédente de chaque section

    def reset(self):
        if not self._identity:
            self.x_in[:] = 0.0
            self.y_out[:] = 0.0

    def step(self, e: float) -> float:
        if self._identity:
            return e
        v = e
        for i in range(self.ns):
            y = self.b0[i] * v + self.b1[i] * self.x_in[i] \
                - self.a1[i] * self.y_out[i]
            self.x_in[i] = v
            self.y_out[i] = y
            v = y
        return self.gain * v


# ---------------------------------------------------------------------------
class FOPIDController:
    """
    FOPID : u = sat(Kp e + Ki I^lambda e + Kd D^mu e), e = y_meas / y_ref.

    Paramètres (params) : (Kp, Ki, Kd, lam, mu)
        Kp, Ki, Kd : gains signés, en volts par unité de y_ref
        lam, mu    : ordres fractionnaires, dans (0, 1]

    Interface NewmarkSimulator : step(x_hat_prev, u_prev, y_meas) -> (x_hat, u)
    (x_hat est renvoyé inchangé : le FOPID n'utilise pas d'observateur).
    """

    def __init__(self, dt: float, params,
                 y_ref: float = 1e-5,       # 10 um
                 V_max: float = 150.0,      # saturation ampli (specs E420 / patch)
                 w_c: float = 2 * np.pi * 537.0,   # mode 1 (Table 4)
                 w_b: float = 2 * np.pi * 20.0,
                 w_h: float = 2 * np.pi * 5000.0,
                 N_oust: int = 4):
        self.Kp, self.Ki, self.Kd, lam, mu = [float(v) for v in params]
        self.lam = float(np.clip(lam, 0.01, 1.0))
        self.mu = float(np.clip(mu, 0.01, 1.0))
        self.y_ref = y_ref
        self.V_max = V_max
        self.dt = dt
        self.I_op = FractionalOperator(-self.lam, dt, w_c, w_b, w_h, N_oust)
        self.D_op = FractionalOperator(+self.mu, dt, w_c, w_b, w_h, N_oust)
        self.params = (self.Kp, self.Ki, self.Kd, self.lam, self.mu)

    # -- interface NewmarkSimulator ------------------------------------
    def step(self, x_hat_prev, u_prev, y_meas):
        e = y_meas / self.y_ref
        u = (self.Kp * e
             + self.Ki * self.I_op.step(e)
             + self.Kd * self.D_op.step(e))
        u = float(np.clip(u, -self.V_max, self.V_max))
        return x_hat_prev, u

    def reset(self):
        self.I_op.reset()
        self.D_op.reset()

    def __repr__(self):
        return (f"FOPID(Kp={self.Kp:+.2f}, Ki={self.Ki:+.2f}, "
                f"Kd={self.Kd:+.2f}, lam={self.lam:.3f}, mu={self.mu:.3f})")
