"""
simulate.py — Integration temporelle de la coupe avec commande
===============================================================
Meme equation, meme schema (Newmark acceleration moyenne, gamma = 1/2,
beta = 1/4), meme convention de signe et meme pas dt = tau/n_sub que
`plant/time_domain.py`, avec trois ajouts necessaires au papier :

  * la commande peut etre un objet a etat (correcteur mu discretise) et
    utiliser des valeurs RETARDEES de la mesure (Eq. 30) ;
  * les coefficients de coupe alpha3, alpha4 sont TABULES sur une periode de
    dent (ils y sont exactement periodiques a engagement constant, et le pas
    verifie tau = n_sub dt : la tabulation est donc exacte, pas approchee) ;
  * la resolution de Newmark utilise Sherman-Morrison, puisque
        S = S0 + beta dt^2 alpha4(t) D(x) D(x)^T
    ne differe de la constante S0 que par une correction de rang 1. Le cout
    par pas devient O(n^2) au lieu d'une factorisation.

Equation integree (modes normalises en masse) :

    q" + C q' + (K + s a4(t) D(x)^T D(x)) q(t)
        = s a4(t) D(x)^T D(x) q(t - tau) + f_z s a3(t) D(x)^T + H_Pe u(t)

avec x(t) = min(v t, l_P), v = f_z N_T rpm/60.
"""
import numpy as np

from milling_dynamics import alpha34, N_TEETH, AE_NOM, FZ_NOM


def pass_duration(rpm, fz=FZ_NOM, lp=0.100, NT=N_TEETH):
    v = fz * NT * rpm / 60.0
    return lp / v, v


class MillingSimulation:
    """Simulation temporelle d'une passe, avec ou sans commande."""

    def __init__(self, plate, rpm, ap, ae=AE_NOM, fz=FZ_NOM, sign=1.0,
                 n_modes=5, n_sub=164, n_pos=401, mode_scale=None,
                 zeta_scale=1.0, v_max=150.0):
        self.p = plate
        self.rpm, self.ap, self.ae, self.fz = rpm, ap, ae, fz
        self.sign = float(sign)
        self.v_max = None if v_max is None else abs(float(v_max))
        self.n = int(n_modes)
        self.tau = 60.0 / (N_TEETH * rpm)
        self.n_sub = int(n_sub)
        self.dt = self.tau / self.n_sub
        self.Om = 2 * np.pi * rpm / 60.0

        w = plate.omega_n[:self.n].copy()
        z = plate.zeta_modes[:self.n].copy() * zeta_scale
        if mode_scale is not None:            # perturbation masse/raideur
            w = w * np.asarray(mode_scale, float)[:self.n]
        self.omega, self.zeta = w, z
        self.K = np.diag(w**2)
        self.C = np.diag(2 * z * w)
        self.H = np.asarray(plate.H_Pe_modal, float)[:self.n]
        self.D_obs = plate.D_row(plate.lp, plate.hp)[:self.n]

        # tables : coefficients de coupe sur une periode de dent
        a3 = np.empty(self.n_sub)
        a4 = np.empty(self.n_sub)
        for k in range(self.n_sub):
            a3[k], a4[k] = alpha34(k * self.dt, self.Om,
                                   plate.hp - ap, plate.hp, ae)
        self.a3, self.a4 = self.sign * a3, self.sign * a4

        # table : deformee modale le long du bord superieur
        self.xs = np.linspace(0.0, plate.lp, n_pos)
        self.Dtab = np.array([plate.D_row(x, plate.hp)[:self.n]
                              for x in self.xs])

        # Newmark : S0 constante, corrections de rang 1 par Sherman-Morrison
        g, b = 0.5, 0.25
        self.g, self.b = g, b
        S0 = np.eye(self.n) + g * self.dt * self.C + b * self.dt**2 * self.K
        self.S0inv = np.linalg.inv(S0)
        self.vtab = self.Dtab @ self.S0inv.T          # v_i = S0^-1 D_i
        self.dvtab = np.einsum('ij,ij->i', self.Dtab, self.vtab)

    # ------------------------------------------------------------------
    def run(self, controller=None, T=None, moving=True, x0=0.0,
            stop_um=5000.0, record_split=False):
        """Integre la passe. `controller(y=..., t=..., k=...) -> u [V]`."""
        Tpass, v = pass_duration(self.rpm, self.fz, self.p.lp)
        if T is None:
            T = Tpass if moving else 0.6
        nstep = int(round(T / self.dt)) + 1
        n, dt, g, b = self.n, self.dt, self.g, self.b
        t = np.arange(nstep) * dt
        xt = np.minimum(v * t, self.p.lp) if moving else np.full(nstep, x0)
        idx = np.clip(np.round(xt / self.p.lp * (len(self.xs) - 1)
                               ).astype(int), 0, len(self.xs) - 1)

        q = np.zeros((n, nstep))
        qdot = np.zeros((n, nstep))
        qd = np.zeros(n)
        qdd = np.zeros(n)
        y_o = np.zeros(nstep)
        yd_o = np.zeros(nstep)
        y_m = np.zeros(nstep)
        u_s = np.zeros(nstep)          # tension REELLEMENT appliquee
        u_c = np.zeros(nstep)          # tension DEMANDEE par le correcteur
        u_rob = np.zeros(nstep) if record_split else None
        u_pd = np.zeros(nstep) if record_split else None
        S0inv, ntau = self.S0inv, self.n_sub
        diverged = 0
        for k in range(1, nstep):
            i = idx[k]
            D = self.Dtab[i]
            a3 = self.a3[k % ntau]
            a4 = self.a4[k % ntau]
            y_prev = float(self.D_obs @ q[:, k - 1])
            yd_prev = float(self.D_obs @ qdot[:, k - 1])
            y_o[k], yd_o[k] = y_prev, yd_prev
            u = 0.0 if controller is None else float(
                controller(y=y_prev, yd=yd_prev, t=t[k], k=k))
            u_c[k] = u
            # saturation de l'amplificateur, identique pour tout correcteur
            if self.v_max is not None:
                u = float(np.clip(u, -self.v_max, self.v_max))
            u_s[k] = u
            if record_split and controller is not None:
                u_rob[k] = getattr(controller, 'u_rob_last', u)
                u_pd[k] = getattr(controller, 'u_pd_last', 0.0)
            q_del = q[:, k - ntau] if k - ntau > 0 else np.zeros(n)
            # second membre de Newmark
            qdp = qd + (1 - g) * dt * qdd
            qp = q[:, k - 1] + dt * qd + (0.5 - b) * dt**2 * qdd
            F = self.fz * a3 * D + a4 * (D * float(D @ q_del)) + self.H * u
            rhs = F - self.C @ qdp - self.K @ qp - a4 * D * float(D @ qp)
            # (S0 + b dt^2 a4 D D^T)^-1 rhs  par Sherman-Morrison
            c = b * dt**2 * a4
            wv = S0inv @ rhs
            if c != 0.0:
                wv = wv - (c * float(self.vtab[i] @ rhs)
                           / (1.0 + c * self.dvtab[i])) * self.vtab[i]
            qdd = wv
            qd = qdp + g * dt * qdd
            q[:, k] = qp + b * dt**2 * qdd
            qdot[:, k] = qd
            y_m[k] = float(D @ q[:, k])
            if not np.isfinite(y_m[k]) or abs(y_m[k]) > stop_um * 1e-6:
                diverged = k
                break
        end = diverged if diverged else nstep - 1
        sl = slice(0, end + 1)
        n_sat = int(np.count_nonzero(np.abs(u_c[sl] - u_s[sl]) > 1e-12))
        out = dict(t=t[sl], y_obs=y_o[sl], yd_obs=yd_o[sl], y_mill=y_m[sl],
                   u=u_s[sl], u_cmd=u_c[sl], n_saturated=n_sat,
                   v_max=self.v_max,
                   x_tool=xt[sl], diverged=bool(diverged),
                   t_div=float(t[end]) if diverged else None,
                   dt=dt, tau=self.tau, T_pass=Tpass, feed_mm_s=v * 1e3)
        if record_split:
            out['u_robust'] = u_rob[sl]
            out['u_delay'] = u_pd[sl]
        return out


def amplitude_spectrum(t, y, fmax=1600.0, scale=1e6):
    """Spectre d'amplitude unilateral (convention des Figs. 14, 20, 21)."""
    y = np.asarray(y, float) - np.mean(y)
    N = len(y)
    Y = np.fft.rfft(y) * 2.0 / N
    f = np.fft.rfftfreq(N, t[1] - t[0])
    m = f <= fmax
    return f[m], np.abs(Y[m]) * scale


def mean_abs_amplitude(u, t=None, skip=0.05):
    """Amplitude moyenne du signal (les 'control voltages' compares par le
    papier), en ecartant le transitoire initial."""
    u = np.asarray(u, float)
    i0 = int(skip * u.size)
    return float(np.mean(np.abs(u[i0:])))
