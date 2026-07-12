"""adaptive_lqg_controller.py
===========================
Traitement, sur la plaque de CET article, de la lacune identifiée dans
l'article ASME (Wang, Sun & Sun, ASME OJE 2026, 10.1115/1.4070743) :

   « les implémentations existantes reposent sur des modèles / lobes
     pré-calculés qui deviennent invalides quand la dynamique de la pièce
     évolue avec l'enlèvement de matière »

Ici les contrôleurs LQG / DARC-MPC sont MODEL-BASED et conçus UNE FOIS
sur le modèle nominal.  Quand les fréquences propres dérivent pendant
l'usinage (enlèvement de matière — l'équivalent de l'Eq. (6) de l'article
ASME), le gain de Kalman, le gain LQR et le modèle inverse du feedforward
deviennent faux.

Solution implémentée, calquée sur la stratégie de l'article ASME :

  1. IDENTIFICATION EN LIGNE  — moindres carrés récursifs (RLS AR(2)) à
     facteur d'oubli sur la mesure de déplacement, après réjection des
     harmoniques synchrones de la broche (le module d'identification du
     framework `milling_sim` de ce dépôt est réutilisé tel quel) ;
  2. ADAPTATION COORDONNÉE    — re-conception périodique (Riccati) du
     LQR + Kalman sur le modèle identifié (échelle fréquentielle s =
     f̂₁/f₁ⁿᵒᵐ appliquée aux 3 modes), avec continuité de l'état
     estimé (même base modale) ;
  3. (option DARC)            — re-conception du feedforward inverse sur
     le modèle identifié à chaque adaptation.

Interface `step(x_hat_prev, u_prev, y_meas)` identique à LQGController :
le solveur Newmark n'est pas modifié.
"""

import os
import sys
import types

import numpy as np

from lqg_controller import LQGController

# reuse the RLS identifier of the repository's adaptive-machining framework
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from milling_sim.identification import RLSModalIdentifier  # noqa: E402


def _scaled_plate(plate, s: float):
    """Lightweight view of `plate` with natural frequencies scaled by s
    (K ∝ s², C ∝ s; modal basis, masses, coupling vectors unchanged)."""
    p = types.SimpleNamespace()
    p.n_modes = plate.n_modes
    p.Mp = plate.Mp
    p.Kp = plate.Kp * (s * s)
    p.Cp = plate.Cp * s
    p.H_Pe_modal = plate.H_Pe_modal
    p.D_obs = plate.D_obs
    return p


class _Shadow:
    """One hypothesis observer of the multi-model bank.

    The KNOWN periodic cutting excitation w_k = ft * a3(phase) enters the
    observer through G_w (same spindle-phase information the DARC
    feedforward uses); the innovation then measures MODEL error rather
    than the unmodelled forcing, so the hypothesis comparison is fair.
    """

    def __init__(self, ctl: LQGController, s: float):
        self.ctl = ctl
        self.s = s
        self.x = np.zeros(ctl.n_x)
        self.ewma = 0.0

    def update(self, u_prev: float, y: float, w_k: float,
               lam: float) -> None:
        c = self.ctl
        x_pred = (c.A_obs_d @ self.x + c.G_u.flatten() * u_prev
                  + c.G_w.flatten() * w_k)
        e = y - float(np.squeeze(c.C @ x_pred))   # innovation (pre-update)
        self.x = x_pred + c.G_y.flatten() * y
        self.ewma = lam * self.ewma + (1.0 - lam) * e * e


class AdaptiveLQGController:
    """LQG with online drift identification and scheduled Riccati redesign.

    Identification = adaptive MULTI-MODEL estimation (bank of shadow
    Kalman observers at frequency-scale hypotheses s - delta, s, s + delta;
    the smallest innovation variance designates the true scale).  The
    RLS AR(2) of the ASME article is kept as a diagnostic, but in THIS
    closed loop the nonsynchronous residual is dominated by the
    regenerative quasi-periodic root (mode reflections at f +/- k f_tooth),
    which sits outside the structural-mode band - the observer-bank
    approach identifies the drift from the model-fit itself and is the
    appropriate tool here.
    """

    name = "adaptive-LQG"

    def __init__(self, plate, dt: float, rpm: float,
                 w_q: float = 1e14, w_qd: float = 1e8, w_r: float = 1.0,
                 gain_norm_max: float = 1e10,
                 window: int = 256,
                 redesign_period_s: float = 0.10,
                 s_tol: float = 0.005,
                 s_min: float = 0.70, s_max: float = 1.05,
                 bank_delta: float = 0.012,
                 bank_period_s: float = 0.10,
                 ewma_lambda: float = 0.998,
                 ident_ft: float = 0.0,
                 ident_a3_period=None,
                 ident_Dp=None,
                 verbose: bool = False):
        self.plate = plate
        self.dt = dt
        self.rpm = rpm
        # known periodic excitation for the shadow observers
        self._ident_ft = float(ident_ft)
        self._ident_a3 = (np.asarray(ident_a3_period, float)
                          if ident_a3_period is not None else None)
        self._Dp_ff = (np.asarray(ident_Dp, float)
                       if ident_Dp is not None
                       else np.zeros(plate.n_modes))
        self.weights = (w_q, w_qd, w_r, gain_norm_max)
        self.window = window
        self.redesign_period = max(1, int(round(redesign_period_s / dt)))
        self.s_tol = s_tol
        self.s_min, self.s_max = s_min, s_max
        self.bank_delta = bank_delta
        self.bank_period = max(1, int(round(bank_period_s / dt)))
        self.lam = ewma_lambda
        self.verbose = verbose

        self.f1_nom = float(plate.freq_n[0])
        self.ident = RLSModalIdentifier(fs=1.0 / dt,
                                        f_lo=0.5 * self.f1_nom,
                                        f_hi=3.0 * self.f1_nom)
        self.s_hat = 1.0
        self.s_designed = 1.0
        self._buf = []
        self._k = 0
        self._since_design = 0
        self._since_bank = 0

        self.base = self._design(1.0)
        self._make_bank(1.0)
        self.n_modes = plate.n_modes
        self.n_x = 2 * plate.n_modes
        self.log_t, self.log_s, self.log_f1 = [], [], []

    # ------------------------------------------------------------------
    def _design(self, s: float) -> LQGController:
        w_q, w_qd, w_r, gmax = self.weights
        ctl = LQGController(_scaled_plate(self.plate, s), dt=self.dt,
                            verbose=False)
        ctl.optimize_weights(w_q_list=[w_q], w_qd_list=[w_qd], w_r=w_r,
                             gain_norm_max=gmax)
        ctl.discretize_observer()
        self.s_designed = s
        self._since_design = 0
        if self.verbose:
            print(f"   [adaptive-LQG] redesign at s = {s:.4f} "
                  f"(f1 = {s * self.f1_nom:.1f} Hz)")
        return ctl

    def _observer_only(self, s: float) -> LQGController:
        """Kalman observer at hypothesis s (LQR part reused from base)."""
        ctl = LQGController(_scaled_plate(self.plate, s), dt=self.dt,
                            verbose=False)
        ctl.K_lqr = self.base.K_lqr          # only the observer matters
        ctl.discretize_observer()
        # ZOH feedthrough of the known periodic modal excitation Dp*w
        n = self.plate.n_modes
        B_w = np.zeros((ctl.n_x, 1))
        B_w[n:, 0] = self._Dp_ff
        A_obs = ctl.A - ctl.L_kal @ ctl.C
        try:
            ctl.G_w = np.linalg.solve(
                A_obs, (ctl.A_obs_d - np.eye(ctl.n_x)) @ B_w)
        except np.linalg.LinAlgError:
            ctl.G_w = B_w * self.dt
        return ctl

    def _make_bank(self, s_center: float) -> None:
        self.bank = []
        for s in (s_center - self.bank_delta, s_center,
                  s_center + self.bank_delta):
            s = float(np.clip(s, self.s_min, self.s_max))
            sh = _Shadow(self._observer_only(s), s)
            self.bank.append(sh)

    # ------------------------------------------------------------------
    def _bank_decision(self) -> None:
        i_best = int(np.argmin([sh.ewma for sh in self.bank]))
        s_best = self.bank[i_best].s
        # move the smoothed estimate toward the winning hypothesis
        self.s_hat = float(np.clip(0.5 * self.s_hat + 0.5 * s_best,
                                   self.s_min, self.s_max))
        if i_best != 1:                      # recentre the bank
            self._make_bank(self.s_hat)
        else:
            for sh in self.bank:
                sh.ewma *= 0.5               # forget, keep hypotheses

    # ------------------------------------------------------------------
    def step(self, x_hat_prev, u_prev, y_meas):
        self._k += 1
        self._since_design += 1
        self._since_bank += 1

        if self._ident_a3 is not None:
            w_k = self._ident_ft * self._ident_a3[
                self._k % len(self._ident_a3)]
        else:
            w_k = 0.0
        for sh in self.bank:
            sh.update(u_prev, y_meas, w_k, self.lam)
        if self._since_bank >= self.bank_period:
            self._since_bank = 0
            self._bank_decision()
            self.log_t.append(self._k * self.dt)
            self.log_s.append(self.s_hat)
            self.log_f1.append(self.ident.fn_hz
                               if self.ident.fn_hz else np.nan)

        # diagnostic RLS on the nonsynchronous residual
        self._buf.append(y_meas)
        if len(self._buf) >= self.window:
            self.ident.update(np.asarray(self._buf, float),
                              self.rpm / 60.0)
            self._buf = []

        if (self._since_design >= self.redesign_period
                and abs(self.s_hat - self.s_designed) > self.s_tol):
            self.base = self._design(self.s_hat)
            self.on_redesign()

        return self.base.step(x_hat_prev, u_prev, y_meas)

    # hook for subclasses (adaptive DARC feedforward re-design)
    def on_redesign(self) -> None:
        pass


class AdaptiveDARCFFController(AdaptiveLQGController):
    """Adaptive LQG base + anticipative feedforward re-designed on the
    identified model at every adaptation (the DARC-MPC architecture with
    its model-based parts kept current — the ASME-gap treatment applied
    to the full article controller)."""

    name = "adaptive-DARC(FF)"

    def __init__(self, plate, dt, rpm, ft, a3_period, Dp_ff,
                 n_harm: int = 30, **kw):
        self.ft = ft
        self.a3_period = np.asarray(a3_period, float)
        self.Dp_ff = np.asarray(Dp_ff, float)
        self.n_harm = n_harm
        self.n_per = len(self.a3_period)
        super().__init__(plate, dt, rpm, **kw)
        self._design_ff()
        self._phase = 0

    # -- inverse-model feedforward on the CURRENT design -----------------
    def _design_ff(self) -> None:
        ctl = self.base
        n = self.plate.n_modes
        A_cl = ctl.A - ctl.B @ ctl.K_lqr
        B_d = np.zeros((ctl.n_x, 1))
        B_d[n:, 0] = self.Dp_ff
        w = self.ft * self.a3_period
        W = np.fft.rfft(w)
        Om = 2 * np.pi / (self.n_per * self.dt)
        Im = np.eye(ctl.n_x)
        U = np.zeros(len(W), dtype=complex)
        for h in range(1, min(self.n_harm + 1, len(W))):
            jw = 1j * h * Om
            R = np.linalg.solve(jw * Im - A_cl, np.hstack([B_d, ctl.B]))
            G_wy = (ctl.C @ R[:, :1]).item()
            G_uy = (ctl.C @ R[:, 1:]).item()
            if abs(G_uy) > 1e-18:
                U[h] = -G_wy * W[h] / G_uy
        self.u_ff = np.fft.irfft(U, n=self.n_per)
        self.u_ff = np.clip(self.u_ff, -150.0, 150.0)

    def on_redesign(self) -> None:
        self._design_ff()

    def step(self, x_hat_prev, u_prev, y_meas):
        x_hat, u_fb = super().step(x_hat_prev, u_prev, y_meas)
        u = u_fb + float(self.u_ff[self._phase % self.n_per])
        self._phase += 1
        return x_hat, float(np.clip(u, -150.0, 150.0))
