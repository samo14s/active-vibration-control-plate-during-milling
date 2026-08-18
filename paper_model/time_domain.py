"""time_domain.py — integration temporelle (Newmark moyenne) du modele
remodelise, avec POSITION D'OUTIL MOBILE et retard regeneratif.

Equation resolue (modes normalises en masse, M = I) :

    q" + C q' + (K + s a4(t) D(x)^T D(x)) q(t)
        = s a4(t) D(x)^T D(x) q(t - tau) + f_z s a3(t) D(x)^T + H_Pe u(t)

ou x(t) = min(v t, l_P) est la position de fraisage sur le bord superieur,
v = f_z N_T (rpm/60) l'avance, et s le drapeau de convention de signe
(-1 = signe derive de (1)(2)(5)(10)(A.4), celui qui reproduit les figures de
stabilite du papier ; +1 = Eq. (13) telle que publiee, celui sur lequel
milling_plant.py est calibre).

La duree d'une passe complete vaut l_P / v : a 4900 tr/min avec
f_z = 0.02 mm/dent et 3 dents, v = 4.9 mm/s et la passe dure 20.408 s —
exactement l'axe des temps des Figs. 14 et 15 du papier."""
import numpy as np

from milling_dynamics import alpha34, N_TEETH, AE_NOM, FZ_NOM


def pass_duration(rpm, fz=FZ_NOM, lp=0.100, NT=N_TEETH):
    """Duree d'une passe complete [s] et vitesse d'avance [m/s]."""
    v = fz * NT * rpm / 60.0
    return lp / v, v


class TimeSim:
    """Integrateur Newmark (gamma=1/2, beta=1/4) a pas fixe dt = tau/n_sub."""

    def __init__(self, plate, rpm, ap, sign=-1.0, ae=AE_NOM, fz=FZ_NOM,
                 n_sub=82, n_modes=None, n_pos=401):
        self.p = plate
        self.rpm, self.ap, self.ae, self.fz = rpm, ap, ae, fz
        self.sign = float(sign)
        self.n = plate.omega_n.size if n_modes is None else int(n_modes)
        self.tau = 60.0 / (N_TEETH * rpm)
        self.dt = self.tau / n_sub
        self.n_tau = n_sub
        self.Om = 2 * np.pi * rpm / 60.0
        self.K = np.diag(plate.omega_n[:self.n]**2)
        self.C = np.diag(2 * plate.zeta_modes[:self.n] * plate.omega_n[:self.n])
        self.H = np.asarray(plate.H_Pe_modal, float)[:self.n]
        self.D_obs = plate.D_row(plate.lp, plate.hp)[:self.n]
        # tabulation de D(x) sur le bord superieur
        self.xs = np.linspace(0.0, plate.lp, n_pos)
        self.Dtab = np.array([plate.D_row(x, plate.hp)[:self.n] for x in self.xs])

    def _D_at(self, x):
        i = int(round(x / self.p.lp * (len(self.xs) - 1)))
        return self.Dtab[min(max(i, 0), len(self.xs) - 1)]

    def run(self, T=None, controller=None, moving=True, x0=0.0,
            stop_um=5000.0, taper_mm=0.0, ap_ramp_s=0.0, ap_start=0.02e-3):
        """taper_mm : longueur des rampes d'ENGAGEMENT en entree et en sortie.
        L'outil reel ne passe pas instantanement de zero a la pleine profondeur
        au bord de la piece ; imposer l'engagement plein jusqu'a l'arete est un
        artefact de modelisation qui produit un transitoire terminal."""
        """Integre sur T secondes (par defaut la passe complete si moving).

        controller(y, t, k) -> tension [V], ou None pour la boucle ouverte.
        Retour : dict t, y_obs, y_mill, u, x_tool, diverged, t_div."""
        Tpass, v = pass_duration(self.rpm, self.fz, self.p.lp)
        if T is None:
            T = Tpass if moving else 0.6
        nstep = int(round(T / self.dt)) + 1
        t = np.arange(nstep) * self.dt
        n, dt = self.n, self.dt
        q = np.zeros((n, nstep))
        qd = np.zeros((n, nstep))
        qdd = np.zeros((n, nstep))
        y_o = np.zeros(nstep)
        y_m = np.zeros(nstep)
        u_s = np.zeros(nstep)
        xt = np.minimum(v * t, self.p.lp) if moving else np.full(nstep, x0)
        za_hi = self.p.hp
        if taper_mm > 0.0:
            tp = taper_mm * 1e-3
            ramp = np.clip(np.minimum(xt, self.p.lp - xt) / tp, 0.0, 1.0)
        else:
            ramp = np.ones_like(xt)
        if ap_ramp_s > 0.0:
            f0 = min(ap_start / self.ap, 1.0)
            ramp = ramp * np.clip(f0 + (1 - f0) * t / ap_ramp_s, f0, 1.0)
        g, b = 0.5, 0.25
        diverged = 0
        for k in range(1, nstep):
            D = self._D_at(xt[k])
            DtD = np.outer(D, D)
            a3, a4 = alpha34(t[k], self.Om, za_hi - self.ap * ramp[k], za_hi,
                             self.ae)
            a4 *= self.sign
            a3 *= self.sign
            y_prev = float(self.D_obs @ q[:, k - 1])
            y_o[k] = y_prev
            u = 0.0 if controller is None else float(
                controller(y=y_prev, t=t[k], k=k))
            u_s[k] = u
            qd_del = q[:, k - self.n_tau] if k - self.n_tau > 0 else np.zeros(n)
            Keff = self.K + a4 * DtD
            F = self.fz * a3 * D + a4 * (DtD @ qd_del) + self.H * u
            qdp = qd[:, k - 1] + (1 - g) * dt * qdd[:, k - 1]
            qp = q[:, k - 1] + dt * qd[:, k - 1] + (0.5 - b) * dt**2 * qdd[:, k - 1]
            S = np.eye(n) + g * dt * self.C + b * dt**2 * Keff
            qdd[:, k] = np.linalg.solve(S, F - self.C @ qdp - Keff @ qp)
            qd[:, k] = qdp + g * dt * qdd[:, k]
            q[:, k] = qp + b * dt**2 * qdd[:, k]
            y_m[k] = float(D @ q[:, k])
            if abs(y_m[k]) > stop_um * 1e-6:
                diverged = k
                break
        end = diverged if diverged else nstep - 1
        sl = slice(0, end + 1)
        return dict(t=t[sl], y_obs=y_o[sl], y_mill=y_m[sl], u=u_s[sl],
                    x_tool=xt[sl], diverged=bool(diverged),
                    t_div=float(t[end]) if diverged else None,
                    dt=dt, tau=self.tau, T_pass=Tpass, feed_mm_s=v * 1e3)


def amplitude_spectrum(t, y, fmax=1600.0):
    """Spectre d'amplitude unilateral (um) — meme convention que les
    panneaux (b), (d), (f) de la Fig. 14."""
    y = np.asarray(y, float) - np.mean(y)
    N = len(y)
    Y = np.fft.rfft(y) * 2.0 / N
    f = np.fft.rfftfreq(N, t[1] - t[0])
    m = f <= fmax
    return f[m], np.abs(Y[m]) * 1e6
