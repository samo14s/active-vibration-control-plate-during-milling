"""
adrc_controller.py
==================
ESO-ADRC (modal Extended-State-Observer Active Disturbance Rejection Control)
for flexible-workpiece milling chatter, and its ADAPTIVE development (A-ESO-ADRC).

WHY AN ESO FOR THIS PLANT. The regenerative cutting force
alpha4(t)*Dp*Dp^T*(q(t-tau) - q(t)) is, from the controller's viewpoint, a
matched-in-modal-space internal disturbance with unknown periodic + delayed
structure, and material removal drifts the plant during the pass. ADRC's premise
fits: lump EVERYTHING unmodelled — regenerative force, feed forcing, spillover of
unmodelled modes, stiffness drift — into a per-mode "total disturbance" d(t) in R^n,
estimated online by an extended state observer and available to the control law at
every step. No cutting-force model, no delay model, and no parameter identification
are needed at run time.

DESIGN FINDING #1 — canonical output LADRC FAILS here (kept reproducible in
`CanonicalLADRC_Controller` below). The piezo-patch -> tip-sensor transfer of this
plate is NON-COLLOCATED with alternating modal residues (D_i*H_i = -0.40 / +0.65 /
-0.19 for modes 1-3): its DC gain (-2.4e-8) and high-frequency gain (+0.057) have
opposite signs, which proves an odd number of real right-half-plane zeros. The
canonical form  y_ddot = f + b0*u  assumes ONE consistent input-gain sign, so the
LESO/LADRC law injects positive feedback at mode 1 and destabilizes the plate for
EVERY bandwidth pair (wc, wo) — verified over wc = 2*pi*(50..800) Hz,
wo = 2*pi*(800..2500) Hz, even without cutting. The correct ADRC formulation for a
multi-modal non-collocated structure is in MODAL SPACE, where the design model is
diagonal and collocation is irrelevant:

    q_ddot = -K q - C q_dot + kappa*H u + d(t),    y = D_obs q            (1)
    ESO state z = [q_hat; q_hat_dot; d_hat]  in R^{3n},   d_dot ~ 0        (2)
    u = ( -K_fb [q_hat; q_hat_dot] - gamma * H^T d_hat / (H^T H) ) / kappa (3)

K_fb is the same output-weighted LQR gain construction as the LQG baseline, so
LQG vs ESO-ADRC isolates exactly the ADRC ingredient: replace the plain Kalman
filter with a disturbance-estimating ESO (+ optional matched cancellation gamma).

DESIGN FINDING #2 — the ESO gain via a scaled Riccati equation. Bandwidth-
parametrised pole placement of the 3n-state observer from ONE output is numerically
hopeless (|L| ~ 1e17); the robust equivalent is the filter Riccati equation with the
disturbance-state process-noise intensity sigma_d^2 as the single "observer
bandwidth" knob (plus a small leakage -beta_d on d_hat that bounds the estimate and
regularises the ARE). States/output are rescaled before the solve.

DESIGN FINDING #3 — matched cancellation (gamma > 0) does not pay off here: the
actuator direction H is only ~19 % aligned with the tool-force direction Dp, so the
grid search selects gamma = 0 and the tuned controller acts through the
disturbance-aware state estimate alone. The gamma channel is kept as an option and
the finding is documented (see docs/).

ADAPTIVE DEVELOPMENT (AdaptiveESO_ADRC_Controller). No fixed tuning covers the
whole uncertainty range: the closed-loop Floquet map over a frequency-mismatch ball
shows COMPLEMENTARY instability holes for aggressive vs robust tunings (waterbed).
A-ESO-ADRC therefore runs a LADDER of pre-designed rungs (design tuples
(w_q, w_qd, sigma_d)) that share the one PHYSICAL observer state z = [q; q_dot; d]
— switching is bumpless by construction — under a measurement-driven supervisor:

  (a) RUNG SUPERVISION — the slow EMA of y^2 is compared to a quiet level
      calibrated at cut-in; a sustained cost rise (dwell-limited, hysteretic)
      toggles to the alternative rung, and a fast-EMA PANIC test jumps straight to
      the CERTIFIED-ROBUST rung (the rung with the smallest worst-case monodromy
      radius over the design mismatch ball — certified at design time with
      `fdm_stability.closed_loop_rho_generic`). Quiet costs relax back to the
      performance rung. No plant parameter is identified — the earlier
      probe-identification result showed identification under stable periodic
      cutting needs persistent excitation; the supervisor needs only the measured
      cost, which is exactly what control is about.
  (b) DESIGN FINDING #4 — actuator-effectiveness (kappa) self-identification was
      TESTED AND REJECTED: the natural estimator (EMA regression of the H-projected
      d_hat on u) is biased in closed loop, because d_hat contains the periodic
      cutting force, which correlates with u THROUGH THE FEEDBACK PATH and swamps
      the (kappa_true - kappa_hat)*H*u signal — the same identifiability obstruction
      documented for innovation-based schemes in this repository (persistent
      excitation would be required). Empirically the mechanism ran to its projection
      bounds in the wrong direction (e.g. kappa_hat -> 3.0 with kappa_true = 0.4)
      and destabilised otherwise-recoverable scenarios. It was removed. The loop
      itself tolerates effectiveness loss to at least kappa = 0.25 (verified in
      main_adaptive_removal.py), so no adaptation is needed on this axis within the
      envelope studied.

The supervisor updates from measured signals only — no probe, no cutting-force
knowledge, no model identification — and is verified in main_adaptive_removal.py.
"""
import numpy as np
from scipy.linalg import expm, solve_continuous_are


# =====================================================================
# Shared design helpers
# =====================================================================
def _zoh(A, Bcols, dt):
    """Exact ZOH discretisation of  x_dot = A x + Bcols * v  (van Loan)."""
    nn = A.shape[0]
    nb = Bcols.shape[1]
    M = np.zeros((nn + nb, nn + nb))
    M[:nn, :nn] = A
    M[:nn, nn:] = Bcols
    E = expm(M * dt)
    return E[:nn, :nn], E[:nn, nn:]


def modal_matrices(design):
    """(Km, Cm, H, D_obs, n) from a (truncated) design view (mass-normalised)."""
    n = design.n_modes
    Km = np.linalg.solve(design.Mp, design.Kp)
    Cm = np.linalg.solve(design.Mp, design.Cp)
    H = np.linalg.solve(design.Mp, np.asarray(design.H_Pe_modal, float))
    D_obs = np.asarray(design.D_obs, float)
    return Km, Cm, H, D_obs, n


def lqr_output_weighted(design, w_q, w_qd, w_r=1.0):
    """State-feedback gain with the SAME output-weighted LQR construction as
    LQGController.optimize_weights:  Q = blkdiag(w_q*D_obs D_obs^T + 1e-3 I,
    w_qd*I + 1e-3 I),  R = w_r.  Returns K_fb (1 x 2n as flat vector)."""
    Km, Cm, H, D_obs, n = modal_matrices(design)
    A2 = np.zeros((2 * n, 2 * n))
    A2[0:n, n:2 * n] = np.eye(n)
    A2[n:2 * n, 0:n] = -Km
    A2[n:2 * n, n:2 * n] = -Cm
    B2 = np.zeros((2 * n, 1))
    B2[n:2 * n, 0] = H
    Q = np.block([[w_q * np.outer(D_obs, D_obs) + 1e-3 * np.eye(n),
                   np.zeros((n, n))],
                  [np.zeros((n, n)), w_qd * np.eye(n) + 1e-3 * np.eye(n)]])
    P = solve_continuous_are(A2, B2, Q, np.array([[w_r]]))
    return (B2.T @ P / w_r).ravel()


def eso_matrices(design, beta_d=10.0):
    """Augmented ESO design model (A_e, B_e, C_e) with per-mode disturbance states
    and a small leakage -beta_d on d (bounds d_hat; regularises the Riccati)."""
    Km, Cm, H, D_obs, n = modal_matrices(design)
    m = 3 * n
    A_e = np.zeros((m, m))
    A_e[0:n, n:2 * n] = np.eye(n)
    A_e[n:2 * n, 0:n] = -Km
    A_e[n:2 * n, n:2 * n] = -Cm
    A_e[n:2 * n, 2 * n:3 * n] = np.eye(n)
    A_e[2 * n:3 * n, 2 * n:3 * n] = -beta_d * np.eye(n)
    B_e = np.zeros(m)
    B_e[n:2 * n] = H
    C_e = np.concatenate([D_obs, np.zeros(2 * n)])
    return A_e, B_e, C_e


def eso_gain(design, sigma_d, V=1e-12, eps_q=1e-10, eps_qd=1e-2, beta_d=10.0):
    """Continuous ESO (filter) gain L from the scaled Riccati equation.
    sigma_d^2 is the process-noise intensity of the disturbance states — the single
    'disturbance-tracking bandwidth' knob. States/output are rescaled to O(1)
    before the solve (q ~ 1e-7 m, q_dot ~ 1e-3 m/s, d ~ 10 m/s^2), which is what
    makes the ARE solvable at V = 1e-12."""
    A_e, B_e, C_e = eso_matrices(design, beta_d=beta_d)
    n = len(B_e) // 3
    W = np.diag(np.r_[eps_q * np.ones(n), eps_qd * np.ones(n),
                      sigma_d**2 * np.ones(n)])
    S = np.diag(np.r_[1e-7 * np.ones(n), 1e-3 * np.ones(n), 1e1 * np.ones(n)])
    Si = np.linalg.inv(S)
    At = Si @ A_e @ S
    Ct = (C_e @ S) / np.sqrt(V)
    Wt = Si @ W @ Si.T
    P = solve_continuous_are(At.T, Ct.reshape(3 * n, 1), Wt, np.array([[1.0]]))
    return (S @ (P @ Ct)) / np.sqrt(V)


# =====================================================================
# Fixed controller
# =====================================================================
class ESO_ADRC_Controller:
    """Fixed modal ESO-ADRC (Eqs. 1-3 of the module docstring).
    n_x = 3n (ESO state [q_hat; q_hat_dot; d_hat], allocated by the harness)."""

    def __init__(self, design, dt, w_q, w_qd, sigma_d, gamma=0.0, w_r=1.0,
                 kalman_V=1e-12, beta_d=10.0, u_max=150.0, verbose=True):
        self.design = design
        self.dt = float(dt)
        self.w_q, self.w_qd, self.sigma_d = float(w_q), float(w_qd), float(sigma_d)
        self.gamma = float(gamma)
        self.u_max = float(u_max)
        Km, Cm, H, D_obs, n = modal_matrices(design)
        self.n = n
        self.n_x = 3 * n
        self.H = H
        self.A_e, self.B_e, self.C_e = eso_matrices(design, beta_d=beta_d)
        self.L = eso_gain(design, sigma_d, V=kalman_V, beta_d=beta_d)
        self.K_fb = lqr_output_weighted(design, w_q, w_qd, w_r)
        g_d = gamma * H / (H @ H)
        self.K_con = np.concatenate([self.K_fb, g_d])
        A_obs = self.A_e - np.outer(self.L, self.C_e)
        Ad, G = _zoh(A_obs, np.column_stack([self.B_e, self.L]), self.dt)
        self.A_d, self.G_u, self.G_y = Ad, G[:, 0], G[:, 1]
        if verbose:
            print(f"[ESO-ADRC] w_q={w_q:.0e}, w_qd={w_qd:.0e}, "
                  f"sigma_d={sigma_d:.0e}, gamma={gamma:.2f} "
                  f"({n}-mode ESO, {self.n_x} states)")

    # Continuous closed-loop realization  z_dot = A_con z + B_con_y*y, u = -K_con z
    # for the generic monodromy SLD (fdm_stability.closed_loop_rho_generic).
    def controller_realization(self):
        A_con = self.A_e - np.outer(self.B_e, self.K_con) \
              - np.outer(self.L, self.C_e)
        return A_con, self.L, self.K_con

    def step(self, z_prev, u_prev, y_meas):
        z = self.A_d @ z_prev + self.G_u * u_prev + self.G_y * y_meas
        u = float(np.clip(-self.K_con @ z, -self.u_max, self.u_max))
        return z, u


# =====================================================================
# Adaptive controller
# =====================================================================
class AdaptiveESO_ADRC_Controller:
    """
    A-ESO-ADRC = a ladder of pre-designed ESO-ADRC rungs sharing ONE physical
    observer state z = [q_hat; q_hat_dot; d_hat] (bumpless switching), under a
    measurement-cost supervisor:
      * slow-EMA cost vs a RUNNING-MIN quiet level (self-calibrating, floored);
      * dwell-limited, hysteretic toggling perf <-> robust rung;
      * fast-EMA PANIC jump to the certified-robust rung, with an ABSOLUTE panic
        floor active from the first step (catches divergence during calibration)
        and ESCALATING post-panic locks (2x per panic) that suppress limit-cycle
        toggling between a rung's instability hole and the robust rung.
    See the module docstring for the rationale (and for the REJECTED kappa
    self-identification mechanism — DESIGN FINDING #4).

    Parameters
    ----------
    rungs : sequence of (w_q, w_qd, sigma_d). rungs[perf_rung] is the performance
        design, rungs[robust_rung] the certified-robust fallback (certify at design
        time with the generic monodromy over the mismatch ball — see
        main_simulation.py).
    """

    def __init__(self, design, dt, rungs=((1e16, 1e8, 3e3), (1e14, 1e8, 1e4)),
                 perf_rung=0, robust_rung=None, gamma=0.0,
                 kalman_V=1e-12, beta_d=10.0, u_max=150.0,
                 # supervisor
                 cost_tau=10e-3, fast_tau=1.5e-3, calib_steps=4 * 82,
                 ratio_up=6.0, ratio_back=1.5, panic_ratio=30.0,
                 dwell_steps=250, panic_lock_steps=2000,
                 quiet_floor=0.3e-6, panic_abs=2.5e-6,
                 verbose=True):
        self.design = design
        self.dt = float(dt)
        self.u_max = float(u_max)
        self.gamma = float(gamma)
        Km, Cm, H, D_obs, n = modal_matrices(design)
        self.n = n
        self.n_x = 3 * n
        self.H = H
        self.A_e, self.B_e, self.C_e = eso_matrices(design, beta_d=beta_d)
        self.rung_defs = [tuple(map(float, r)) for r in rungs]
        self.perf_rung = int(perf_rung)
        self.robust_rung = int(robust_rung) if robust_rung is not None \
            else len(self.rung_defs) - 1
        # Precompute per-rung gains and discrete matrices (shared physical state).
        self._tables = []
        for (w_q, w_qd, sigma_d) in self.rung_defs:
            L = eso_gain(design, sigma_d, V=kalman_V, beta_d=beta_d)
            K_fb = lqr_output_weighted(design, w_q, w_qd)
            K_con = np.concatenate([K_fb, gamma * H / (H @ H)])
            A_obs = self.A_e - np.outer(L, self.C_e)
            Ad, G = _zoh(A_obs, np.column_stack([self.B_e, L]), self.dt)
            self._tables.append(dict(L=L, K_con=K_con, A_d=Ad,
                                     G_u=G[:, 0], G_y=G[:, 1]))
        # supervisor parameters
        self.cost_tau = float(cost_tau)
        self.fast_tau = float(fast_tau)
        self.calib = int(calib_steps)
        self.r_up = float(ratio_up)
        self.r_back = float(ratio_back)
        self.panic_ratio = float(panic_ratio)
        self.dwell = int(dwell_steps)
        self.panic_lock0 = int(panic_lock_steps)
        self.quiet_floor = float(quiet_floor)**2
        self.panic_abs = float(panic_abs)**2
        self.reset_adaptation()
        if verbose:
            names = ", ".join(f"({w:.0e},{wd:.0e},{sd:.0e})"
                              for (w, wd, sd) in self.rung_defs)
            print(f"[A-ESO-ADRC] rungs [{names}], perf={self.perf_rung}, "
                  f"robust={self.robust_rung} (cost-supervised, panic + "
                  f"escalating locks)")

    # ------------------------------------------------------------------
    def reset_adaptation(self):
        self.i_rung = self.perf_rung
        self._E_slow = 0.0
        self._E_fast = 0.0
        self._E_quiet = None       # running min of E_slow after calibration
        self._k = 0
        self._k_last_sw = 0
        self._k_panic = -10**9
        self._n_panic = 0
        self.history_rung = []
        self.history_cost = []

    # Frozen-rung continuous realization for the monodromy SLD.
    def controller_realization(self, rung=None):
        t = self._tables[self.i_rung if rung is None else int(rung)]
        A_con = self.A_e - np.outer(self.B_e, t['K_con']) \
              - np.outer(t['L'], self.C_e)
        return A_con, t['L'], t['K_con']

    # ------------------------------------------------------------------
    def step(self, z_prev, u_prev, y_meas):
        t = self._tables[self.i_rung]
        z = t['A_d'] @ z_prev + t['G_u'] * u_prev + t['G_y'] * y_meas
        self._k += 1

        # --- cost monitors (EMAs of measured y^2) --------------------------
        af = self.dt / self.fast_tau
        as_ = self.dt / self.cost_tau
        y2 = y_meas * y_meas
        self._E_fast = (1 - af) * self._E_fast + af * y2
        self._E_slow = (1 - as_) * self._E_slow + as_ * y2
        # Quiet level: running MIN of the slow cost after the calibration window
        # (floored) — robust to a cut that is already elevated at calibration time.
        if self._k >= self.calib:
            cand = max(self._E_slow, self.quiet_floor)
            self._E_quiet = cand if self._E_quiet is None \
                else min(self._E_quiet, cand)

        # --- PANIC: jump to the certified-robust rung ----------------------
        # Ratio test once calibrated; absolute floor active from step one.
        panic_level = self.panic_abs if self._E_quiet is None \
            else max(self.panic_ratio * self._E_quiet, self.panic_abs)
        if self.i_rung != self.robust_rung and self._E_fast > panic_level:
            self.i_rung = self.robust_rung
            self._k_last_sw = self._k
            self._k_panic = self._k
            self._n_panic += 1

        # --- slow supervisor (dwell-limited, hysteretic, escalating locks) --
        elif (self._E_quiet is not None
                and self._k - self._k_last_sw >= self.dwell):
            lock = self.panic_lock0 * 2**min(max(self._n_panic - 1, 0), 5)
            r = self._E_slow / self._E_quiet
            if self.i_rung == self.perf_rung and r > self.r_up:
                self.i_rung = self.robust_rung        # sustained elevated cost
                self._k_last_sw = self._k
            elif (self.i_rung == self.robust_rung and r < self.r_back
                    and self._k - self._k_panic >= lock):
                self.i_rung = self.perf_rung          # quiet again -> performance
                self._k_last_sw = self._k

        # --- control law at the current rung --------------------------------
        u = float(np.clip(-(t['K_con'] @ z), -self.u_max, self.u_max))
        self.history_rung.append(self.i_rung)
        self.history_cost.append(self._E_slow)
        return z, u


# =====================================================================
# Canonical output LADRC — kept ONLY to reproduce the documented negative
# result (module docstring, DESIGN FINDING #1): on this non-collocated,
# non-minimum-phase plant it destabilizes for every (wc, wo).
# =====================================================================
def _leso_discrete(b0, wo, dt):
    """ZOH-discretised canonical 3rd-order LESO: z+ = A_d z + G_u u + G_y y."""
    A = np.array([[0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0]])
    B = np.array([0.0, b0, 0.0])
    L = np.array([3.0 * wo, 3.0 * wo**2, wo**3])
    C = np.array([1.0, 0.0, 0.0])
    A_obs = A - np.outer(L, C)
    A_d = expm(A_obs * dt)
    M = np.linalg.solve(A_obs, A_d - np.eye(3))
    return A_d, M @ B, M @ L


class CanonicalLADRC_Controller:
    """Textbook LADRC on y_ddot = f + b0*u (Gao bandwidth parametrisation).
    DO NOT use on this plant — see DESIGN FINDING #1. n_x = 3."""

    def __init__(self, plate, dt, wc, wo, b0=None, u_max=150.0, verbose=True):
        self.plate = plate
        self.dt = dt
        self.wc = float(wc)
        self.wo = float(wo)
        self.u_max = float(u_max)
        self.n_x = 3
        self.b0_nom = float(np.dot(plate.D_obs,
                                   np.linalg.solve(plate.Mp, plate.H_Pe_modal)))
        self.b0 = float(b0) if b0 is not None else self.b0_nom
        self.kp = self.wc**2
        self.kd = 2.0 * self.wc
        self.A_d, self.G_u, self.G_y = _leso_discrete(self.b0, self.wo, dt)
        if verbose:
            print(f"[LADRC-canonical] wc = {self.wc/2/np.pi:.0f} Hz, "
                  f"wo = {self.wo/2/np.pi:.0f} Hz, b0 = {self.b0:.4g} (m/s^2)/V "
                  f"— unstable by design on this plant (kept for the negative "
                  f"result)")

    def controller_realization(self):
        A = np.array([[0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, 0.0, 0.0]])
        B = np.array([0.0, self.b0, 0.0])
        L = np.array([3.0 * self.wo, 3.0 * self.wo**2, self.wo**3])
        C = np.array([1.0, 0.0, 0.0])
        K_con = np.array([self.kp, self.kd, 1.0]) / self.b0
        A_con = A - np.outer(B, K_con) - np.outer(L, C)
        return A_con, L, K_con

    def step(self, z_prev, u_prev, y_meas):
        z = self.A_d @ z_prev + self.G_u * u_prev + self.G_y * y_meas
        u = (-self.kp * z[0] - self.kd * z[1] - z[2]) / self.b0
        u = float(np.clip(u, -self.u_max, self.u_max))
        return z, u
