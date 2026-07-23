"""
internal_model_control.py
=========================
FAIR FEEDBACK BASELINE — Repetitive / Internal-Model LQG (RC-LQG)
================================================================

Research gap addressed (#1, the most important one)
---------------------------------------------------
The article's central argument is::

    "a feedback controller cannot fully reject a periodic disturbance
     (phase lag / waterbed limit) — only a feedforward can."

This is **not true in general**.  The *Internal Model Principle*
(Francis & Wonham, 1976) states that a feedback loop achieves asymptotic
rejection of a disturbance **iff a model of the disturbance-generating
dynamics is included in the loop**.  For the periodic tooth-passing
excitation (fundamental ω_TPF = N_T·Ω/60 and its harmonics) the internal
model is a bank of undamped resonators 1/(s²+(hω_TPF)²) — exactly the
mechanism used by *repetitive control* (Hara et al., 1988) in machining.

Comparing the article's DARC (LQG + feedforward + NN) only against a
**feedback-only** LQG therefore compares against a deliberately handicapped
baseline.  A reviewer at IEEE TCST / Automatica / MSSP will demand the fair
feedback baseline implemented here.

What this controller is
-----------------------
``RepetitiveLQG`` keeps the *identical* LQG DNA of the article (same modal
plant, same base state/velocity/control weights, same Kalman observer) and
**only adds** a multi-harmonic internal model of the tooth-passing period —
the feedback analogue of the DARC feedforward.  It is a proper LQG-optimal
servo-compensator:

    z = [x ; ξ]                         augmented state
    ẋ = A x + B u + E d(t)              modal plant (Mp = I)
    ξ̇ = S ξ + L (C x)                  resonator bank driven by output error
    u  = -K_x x̂  -  K_ξ ξ              LQR over the augmented plant

with S = blkdiag_h [[0, 1], [-(hω_TPF)², 0]], and (K_x, K_ξ) from the
augmented Riccati equation.  The plant state x is Kalman-estimated (as in the
article's LQG); the resonator states ξ are computed exactly by integrating the
measured output, so no extra observer is needed.

Honest positioning (crossover discovered empirically by ``main_gap_study.py``)
-----------------------------------------------------------------------------
On the nominal scenario, sweeping the displacement-sensor noise gives a clean
crossover (steady-state y_RMS, µm)::

    sensor noise   LQG      RC-LQG    DARC        winner
      0.0 µm       0.599    0.162     0.290       RC-LQG  (feedback wins!)
      0.1 µm       0.600    0.173     0.293       RC-LQG
      0.6 µm       0.643    0.398     0.371       DARC
      1.0 µm       0.714    0.627     0.483       DARC
      2.0 µm       0.981    1.220     0.824       DARC
      4.0 µm       1.666    2.426     1.568       DARC

Two genuine, defensible facts replace the article's incorrect
"feedback cannot reject a periodic disturbance":
  1. **With a good sensor a pure-feedback repetitive/internal-model LQG BEATS
     the DARC feedforward** (0.16 vs 0.29 µm) — the internal-model principle
     rejects the tooth-passing excitation, and better.  The article's "55 %"
     headline is measured against a deliberately handicapped feedback-only LQG.
  2. **DARC's real, defensible edge is sensor-independence.** Its feedforward is
     synchronized to the spindle encoder (phase), so it is immune to sensor
     degradation, whereas RC-LQG is feedback and *amplifies* sensor noise at the
     rejected harmonics (the classic repetitive-control waterbed penalty — note
     RC-LQG becomes *worse than plain LQG* beyond ≈2 µm sensor noise).  DARC
     therefore dominates precisely in the industrially relevant regime of
     coarse / noisy displacement sensing.
This module lets the paper *quantify* the trade-off instead of asserting an
incorrect impossibility — a far stronger and more credible contribution.

References
----------
- B.A. Francis, W.M. Wonham, "The internal model principle of control theory",
  Automatica 12 (1976) 457-465.
- S. Hara, Y. Yamamoto, T. Omata, M. Nakano, "Repetitive control system: a new
  type servo system for periodic exogenous signals", IEEE TAC 33 (1988) 659-668.
- E.J. Davison, "The robust control of a servomechanism problem for linear
  time-invariant multivariable systems", IEEE TAC 21 (1976) 25-34.
"""
import numpy as np
from scipy.linalg import solve_continuous_are, expm


class RepetitiveLQG:
    """
    Repetitive / internal-model LQG controller (fair feedback baseline).

    Parameters
    ----------
    plate : PlateModel
        Must already have ``Mp, Kp, Cp, D_obs, H_Pe_modal`` (same requirement
        as the article's ``LQGController``).
    dt : float
        Sample period [s] (matches the Newmark integrator).
    f_tpf : float
        Tooth-passing frequency [Hz] = N_T · RPM / 60 (fundamental of the
        periodic excitation to be rejected).
    n_harm : int
        Number of tooth-passing harmonics embedded in the internal model.
    base_w_q, base_w_qd, base_w_r : float
        The *identical* LQG base weights used by the article's controller
        (fair comparison — same reactive DNA).
    w_im : float
        Penalty on the internal-model (resonator) states.  Larger → more
        aggressive periodic-disturbance rejection (and more control effort).
        Tuned by default to keep |u| within the ±150 V piezo range.
    u_max : float
        Voltage saturation [V].
    """

    def __init__(self, plate, dt, f_tpf, n_harm=6,
                 base_w_q=1e14, base_w_qd=1e8, base_w_r=1.0,
                 w_im=1e12, u_max=150.0, verbose=True):
        self.plate = plate
        self.dt = dt
        self.n_modes = plate.n_modes
        self.n_x = 2 * plate.n_modes
        self.u_max = u_max
        self.verbose = verbose
        self.f_tpf = f_tpf
        self.w0 = 2.0 * np.pi * f_tpf

        self._build_plant()
        self._build_internal_model(n_harm)
        self._design_augmented_lqr(base_w_q, base_w_qd, base_w_r, w_im)
        self._design_kalman()
        self._discretize()

        # runtime state
        self.xi = np.zeros(self.n_im)     # resonator states (known exactly)

        if verbose:
            self._summary()

    # ------------------------------------------------------------------
    def _build_plant(self):
        n = self.n_modes
        A = np.zeros((self.n_x, self.n_x))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -np.linalg.solve(self.plate.Mp, self.plate.Kp)
        A[n:, n:] = -np.linalg.solve(self.plate.Mp, self.plate.Cp)
        B = np.zeros((self.n_x, 1))
        B[n:, 0] = np.linalg.solve(self.plate.Mp, self.plate.H_Pe_modal)
        C = np.zeros((1, self.n_x))
        C[0, :n] = self.plate.D_obs
        self.A, self.B, self.C = A, B, C

    # ------------------------------------------------------------------
    def _build_internal_model(self, n_harm):
        """
        Resonator bank S (2H×2H), input map L (2H×1) driven by the output.

        A **rotational (balanced) realization** is used for each harmonic::

            S_h = [[0, -ω_h], [ω_h, 0]]   (eigenvalues ±jω_h),  L_h = [ω_h, 0]^T

        rather than the controllable-canonical form ``[[0,1],[-ω_h²,0]]``.  The
        canonical form makes the two resonator states span ~ω_h² in magnitude
        (10⁻¹³ vs 10⁻⁶ here), so the augmented Riccati solver — dominated by the
        w_q≈10¹⁴ output penalty — leaves the internal-model modes essentially
        undamped (≈0 % closed-loop damping) and the rejection never engages
        within a realistic horizon.  The rotational form keeps both states O(y),
        so the LQR actually damps and uses the resonators (verified: the
        canonical form gives y_RMS≈y_RMS(LQG); the rotational form gives a
        genuine ~70 % reduction).
        """
        n_im = 2 * n_harm
        S = np.zeros((n_im, n_im))
        L = np.zeros((n_im, 1))
        self.harm_freqs = []
        for i in range(n_harm):
            wh = (i + 1) * self.w0
            S[2*i:2*i+2, 2*i:2*i+2] = [[0.0, -wh], [wh, 0.0]]
            L[2*i:2*i+2, 0] = [wh, 0.0]
            self.harm_freqs.append(wh / (2 * np.pi))
        self.S, self.L_im, self.n_im, self.n_harm = S, L, n_im, n_harm

    # ------------------------------------------------------------------
    def _design_augmented_lqr(self, w_q, w_qd, w_r, w_im):
        n = self.n_modes
        A, B, C = self.A, self.B, self.C
        S, L = self.S, self.L_im
        nx, nim = self.n_x, self.n_im

        # Augmented plant:  z = [x; xi],  xi_dot = S xi + L (C x)
        A_aug = np.zeros((nx + nim, nx + nim))
        A_aug[:nx, :nx] = A
        A_aug[nx:, :nx] = L @ C
        A_aug[nx:, nx:] = S
        B_aug = np.zeros((nx + nim, 1))
        B_aug[:nx, :] = B

        # Cost — identical LQG DNA on x, plus internal-model penalty on xi.
        M_yp = np.outer(self.plate.D_obs, self.plate.D_obs)
        Q_top = w_q * M_yp + 1e-3 * np.eye(n)
        Q_bot = w_qd * np.eye(n) + 1e-3 * np.eye(n)
        Q_x = np.block([[Q_top, np.zeros((n, n))],
                        [np.zeros((n, n)), Q_bot]])
        Q_aug = np.zeros((nx + nim, nx + nim))
        Q_aug[:nx, :nx] = Q_x
        Q_aug[nx:, nx:] = w_im * np.eye(nim)
        R = np.array([[w_r]])

        P = solve_continuous_are(A_aug, B_aug, Q_aug, R)
        K = np.linalg.solve(R, B_aug.T @ P)
        self.K_x = K[:, :nx]
        self.K_xi = K[:, nx:]
        self.A_aug, self.B_aug = A_aug, B_aug
        self.w_q, self.w_qd, self.w_r, self.w_im = w_q, w_qd, w_r, w_im

        # closed-loop eigenvalues (augmented state feedback) for diagnostics
        self.ev_cl_aug = np.linalg.eigvals(A_aug - B_aug @ K)

    # ------------------------------------------------------------------
    def _design_kalman(self, W_kal=1e-6, V_kal=1e-12):
        """Kalman observer for the plant state x (same design as the LQG)."""
        W = W_kal * np.eye(self.n_x)
        V = np.array([[V_kal]])
        P = solve_continuous_are(self.A.T, self.C.T, W, V)
        self.L_kal = P @ self.C.T @ np.linalg.inv(V)

    # ------------------------------------------------------------------
    def _discretize(self):
        # observer (ZOH)
        A_obs = self.A - self.L_kal @ self.C
        self.A_obs_d = expm(A_obs * self.dt)
        try:
            self.G_u = np.linalg.solve(A_obs,
                                       (self.A_obs_d - np.eye(self.n_x)) @ self.B)
            self.G_y = np.linalg.solve(A_obs,
                                       (self.A_obs_d - np.eye(self.n_x)) @ self.L_kal)
        except np.linalg.LinAlgError:
            self.A_obs_d = np.eye(self.n_x) + A_obs * self.dt
            self.G_u = self.B * self.dt
            self.G_y = self.L_kal * self.dt

        # resonator bank (ZOH):  xi[k] = S_d xi[k-1] + L_d y[k]
        self.S_d = expm(self.S * self.dt)
        try:
            self.L_d = np.linalg.solve(self.S,
                                       (self.S_d - np.eye(self.n_im)) @ self.L_im)
        except np.linalg.LinAlgError:
            self.L_d = self.L_im * self.dt

    # ------------------------------------------------------------------
    def _summary(self):
        zmin = np.min(-np.real(self.ev_cl_aug) /
                      np.maximum(np.abs(self.ev_cl_aug), 1e-30)) * 100
        print(f"[RC-LQG] internal model: {self.n_harm} harmonics of "
              f"f_TPF = {self.f_tpf:.1f} Hz "
              f"({self.harm_freqs[0]:.0f}..{self.harm_freqs[-1]:.0f} Hz)")
        print(f"[RC-LQG] augmented states = {self.n_x}+{self.n_im} = "
              f"{self.n_x + self.n_im}")
        print(f"[RC-LQG] ||K_x|| = {np.linalg.norm(self.K_x):.2e}, "
              f"||K_xi|| = {np.linalg.norm(self.K_xi):.2e}")
        print(f"[RC-LQG] min closed-loop damping (augmented) = {zmin:.2f}%")

    # ------------------------------------------------------------------
    def reset(self):
        self.xi = np.zeros(self.n_im)

    def discretize_observer(self):
        """Kept for interface compatibility with LQGController."""
        pass

    # ------------------------------------------------------------------
    def step(self, x_hat_prev, u_prev, y_meas):
        """
        One control step (plain 3-arg interface, like ``LQGController.step``).

        1. Kalman update of the plant state estimate x̂.
        2. Resonator bank update driven by the *measured* output y.
        3. Augmented LQR control  u = -K_x x̂ - K_ξ ξ.
        """
        x_hat = (self.A_obs_d @ x_hat_prev
                 + self.G_u.flatten() * u_prev
                 + self.G_y.flatten() * y_meas)

        # internal model integrates the measured output (exact, no estimation)
        self.xi = self.S_d @ self.xi + self.L_d.flatten() * y_meas

        u = float(-(self.K_x @ x_hat)[0] - (self.K_xi @ self.xi)[0])
        u = float(np.clip(u, -self.u_max, self.u_max))
        return x_hat, u
