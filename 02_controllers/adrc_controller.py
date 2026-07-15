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

P5 EXTENSION — HARMONIC RESONANT CANCELLATION (ESO_ADRC_HRC_Controller) and the
4-RUNG LADDER. The cutting force is periodic at the KNOWN tooth frequency
(spindle-synchronous, model-independent): per-line LTI resonant compensators with
inverse-closed-loop-FRF phase (the online causal counterpart of model-inverse ILC)
cut the nominal RMS ~50 % BELOW the LQG baseline. DESIGN FINDING #5: the harmonic
states must live in the CONTROLLER OUTPUT PATH driven by y — putting them inside
the ESO (with or without LQ-optimal disturbance feedthrough) destabilises the true
5-mode plant via estimator spillover, verified over every intensity tested. The
HRC rung is nominal-brilliant but fragile off-nominal, which the supervised ladder
absorbs: A-ESO-ADRC runs [HRC, performance-ESO, quasi-Kalman-ESO, certified-ESO]
with the v3 supervisor (rising-energy cascade panic with severity-based target,
recovery-trend holds, sticky per-pass failure flags, probe aborts, desperation
probing from the robust end, escalating locks) — the only controller in the study
that never diverges across all 9 stress scenarios while beating LQG by ~50 % on
the nominal plant.
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
# Harmonic resonant cancellation (HRC) extension
# =====================================================================
class ESO_ADRC_HRC_Controller(ESO_ADRC_Controller):
    """
    ESO-ADRC + tooth-harmonic resonant internal model (HRC).

    The cutting force is periodic at the KNOWN tooth-passing frequency (spindle-
    synchronous — independent of the plant model). For each harmonic h = 1..H a
    small LTI resonant compensator

        C_h(s) = g_h (c1 s + c0) / (s^2 + 2 lam s + wh^2),   wh = h*w_tooth

    driven by the measured y is added to the ESO-ADRC law. The phase (c1, c0) is
    set from the DESIGN closed-loop FRF G_cl(j wh) (plant design model + the base
    ESO-ADRC in the loop) so that C_h(j wh) G_cl(j wh) is negative-real, and the
    per-line gain is normalised: g_h = g_base / |G_cl(j wh)|. This is the exact
    LTI equivalent of phase-compensated adaptive feedforward cancellation (AFC),
    and the ONLINE, causal counterpart of the earlier trial-based model-inverse
    ILC. IMPORTANT design finding: putting the harmonic states INSIDE the ESO
    (with or without optimal DAC feedthrough) destabilises the true 5-mode plant
    through estimator spillover — the resonant layer must live in the CONTROLLER
    OUTPUT PATH, driven by y directly, with the ESO left untouched. The pointwise
    inverse-FRF phase at the lines is essentially exact (design-vs-5-mode phase
    error ~0.0 deg at every line).

    Everything stays LTI: controller_realization() returns the augmented
    (A_con, B_con_y, K_con) with state [z_eso; x_res], so the generic monodromy
    SLD and the design-ball certification apply unchanged.
    """

    def __init__(self, design, dt, w_q, w_qd, sigma_d, n_harm, g_base, lam=5.0,
                 w_tooth=None, gamma=0.0, w_r=1.0, kalman_V=1e-12, beta_d=10.0,
                 u_max=150.0, verbose=True):
        super().__init__(design, dt, w_q, w_qd, sigma_d, gamma=gamma, w_r=w_r,
                         kalman_V=kalman_V, beta_d=beta_d, u_max=u_max,
                         verbose=False)
        if w_tooth is None:
            raise ValueError("w_tooth (tooth-passing angular frequency) required")
        self.n_harm = int(n_harm)
        self.g_base = float(g_base)
        self.lam = float(lam)
        self.w_tooth = float(w_tooth)
        me = 3 * self.n
        mr = 2 * self.n_harm
        # design closed-loop FRF from additive u to y (base ESO in the loop)
        A_cl, B_cl, C_cl = self._design_closed_loop()
        A_r = np.zeros((mr, mr))
        B_r = np.zeros(mr)
        K_r = np.zeros(mr)
        for h in range(self.n_harm):
            wh = (h + 1) * self.w_tooth
            G = complex(C_cl @ np.linalg.solve(
                1j * wh * np.eye(A_cl.shape[0]) - A_cl, B_cl))
            psi = np.pi - np.angle(G)          # C(jwh)*G_cl(jwh) negative-real
            gh = self.g_base / abs(G)
            i0 = 2 * h
            A_r[i0, i0 + 1] = 1.0
            A_r[i0 + 1, i0] = -wh**2
            A_r[i0 + 1, i0 + 1] = -2.0 * self.lam
            B_r[i0 + 1] = 1.0
            K_r[i0] = -gh * (-wh * np.sin(psi))     # u = -K_con z_total
            K_r[i0 + 1] = -gh * np.cos(psi)
        self.A_r, self.B_r = A_r, B_r
        # augmented implementation matrices: state zt = [z_eso; x_res]
        self.n_x = me + mr
        K_e = self.K_con                            # base ESO law
        self.K_con = np.concatenate([K_e, K_r])
        A_impl = np.zeros((self.n_x, self.n_x))
        A_impl[:me, :me] = self.A_e - np.outer(self.L, self.C_e)
        A_impl[me:, me:] = A_r
        B_u = np.concatenate([self.B_e, np.zeros(mr)])
        B_y = np.concatenate([self.L, B_r])
        self.A_d, G2 = _zoh(A_impl, np.column_stack([B_u, B_y]), self.dt)
        self.G_u, self.G_y = G2[:, 0], G2[:, 1]
        if verbose:
            print(f"[ESO-ADRC+HRC] base (w_q={w_q:.0e}, w_qd={w_qd:.0e}, "
                  f"sigma_d={sigma_d:.0e}), H={self.n_harm} tooth harmonics, "
                  f"g_base={g_base:g}, lam={lam:g} rad/s")

    def _design_closed_loop(self):
        """Closed loop of the DESIGN model with the base ESO-ADRC: additive u->y."""
        Km, Cm, H, D_obs, n = modal_matrices(self.design)
        # K_con is still the base (3n) ESO law when this is called in __init__
        K_e = self.K_con[:3 * n]
        m = 3 * n
        A = np.zeros((2 * n + m, 2 * n + m))
        A[0:n, n:2 * n] = np.eye(n)
        A[n:2 * n, 0:n] = -Km
        A[n:2 * n, n:2 * n] = -Cm
        A[n:2 * n, 2 * n:] = -np.outer(H, K_e)
        A[2 * n:, 0:n] = np.outer(self.L, D_obs)
        A[2 * n:, 2 * n:] = self.A_e - np.outer(self.B_e, K_e)             - np.outer(self.L, self.C_e)
        B = np.zeros(2 * n + m)
        B[n:2 * n] = H
        B[2 * n:] = self.B_e                    # additive u also seen by the ESO
        C = np.concatenate([D_obs, np.zeros(n + m)])
        return A, B, C

    # controller_realization: u = -K_con zt,  zt_dot = A_con zt + B_y y
    def controller_realization(self):
        me = 3 * self.n
        mr = 2 * self.n_harm
        m = me + mr
        A = np.zeros((m, m))
        A[:me, :me] = self.A_e - np.outer(self.B_e, self.K_con[:me])             - np.outer(self.L, self.C_e)
        A[:me, me:] = np.outer(self.B_e, -self.K_con[me:])
        A[me:, me:] = self.A_r
        B_y = np.concatenate([self.L, self.B_r])
        return A, B_y, self.K_con


# =====================================================================
# Adaptive controller
# =====================================================================
class AdaptiveESO_ADRC_Controller:
    """
    A-ESO-ADRC v2 = an ORDERED LADDER of pre-designed rungs (index 0 = highest
    performance ... last = certified-robust fallback), sharing one physical state
    [z_eso (3n); x_res (2*H_max)] (bumpless switching; resonant states are zeroed
    on every switch), under a measurement-cost supervisor:

      * slow-EMA cost vs a RUNNING-MIN quiet level, floored AND CAPPED — the cap
        (~1 um RMS) encodes an absolute performance expectation so the supervisor
        also reacts when the initial rung is *bounded but bad* (without it, a rung
        ringing at, say, 5 um calibrates its own misery as "quiet");
      * dwell-limited moves ONE rung at a time: toward robust when the cost is
        elevated, toward performance when quiet; each rung's last measured cost is
        remembered (with a freshness window), so the supervisor prefers the
        neighbour with the better fresh record — this resolves the case where the
        ROBUST end is itself in a sensitivity hole and the middle rung is right;
      * fast-EMA PANIC jump straight to the last (certified) rung, with an
        absolute floor active from step one and ESCALATING post-panic locks
        (2x per panic) on performance-ward moves.

    Rungs are given as dicts: {'kind': 'eso', 'w_q':..., 'w_qd':..., 'sigma_d':...}
    or {'kind': 'hrc', ..., 'n_harm':..., 'g_base':..., 'lam':...} (requires
    w_tooth). No plant parameter is identified anywhere — see the module
    docstring for why (identifiability under stable periodic cutting).
    """

    def __init__(self, design, dt, rungs, w_tooth=None,
                 kalman_V=1e-12, beta_d=10.0, u_max=150.0,
                 # supervisor
                 cost_tau=10e-3, fast_tau=1.5e-3, calib_steps=4 * 82,
                 ratio_up=6.0, ratio_back=1.5, panic_ratio=30.0,
                 dwell_steps=1000, dwell_up_steps=600, panic_lock_steps=2000,
                 quiet_floor=0.3e-6, quiet_cap=1.0e-6, panic_abs=2.5e-6,
                 mem_fresh_steps=3000,
                 verbose=True):
        self.design = design
        self.dt = float(dt)
        self.u_max = float(u_max)
        Km, Cm, H, D_obs, n = modal_matrices(design)
        self.n = n
        me = 3 * n
        # Build each rung as a standalone controller, then embed into the shared
        # state layout [z_eso; x_res(2*H_max)]. ESO-only rungs leave the resonant
        # substate decaying (no drive, no output).
        self.rung_defs = list(rungs)
        objs = []
        for r in self.rung_defs:
            if r['kind'] == 'hrc':
                objs.append(ESO_ADRC_HRC_Controller(
                    design, dt, r['w_q'], r['w_qd'], r['sigma_d'],
                    n_harm=r['n_harm'], g_base=r['g_base'],
                    lam=r.get('lam', 5.0), w_tooth=w_tooth,
                    kalman_V=kalman_V, beta_d=beta_d, u_max=u_max,
                    verbose=False))
            else:
                objs.append(ESO_ADRC_Controller(
                    design, dt, r['w_q'], r['w_qd'], r['sigma_d'],
                    kalman_V=kalman_V, beta_d=beta_d, u_max=u_max,
                    verbose=False))
        H_max = max((o.n_harm if hasattr(o, 'n_harm') else 0) for o in objs)
        mr = 2 * H_max
        self.n_x = me + mr
        self.n_rungs = len(objs)
        self.robust_rung = self.n_rungs - 1
        LAM_DECAY = 30.0                      # unused resonant states decay
        self._tables = []
        for o in objs:
            mo = o.n_x
            A_impl = np.zeros((self.n_x, self.n_x))
            B_u = np.zeros(self.n_x)
            B_y = np.zeros(self.n_x)
            K_con = np.zeros(self.n_x)
            # ESO block (common physical meaning across rungs)
            A_impl[:me, :me] = o.A_e - np.outer(o.L, o.C_e)
            B_u[:me] = o.B_e
            B_y[:me] = o.L
            K_con[:me] = o.K_con[:me]
            # resonant block
            if mo > me:                        # HRC rung
                mr_o = mo - me
                A_impl[me:me + mr_o, me:me + mr_o] = o.A_r
                B_y[me:me + mr_o] = o.B_r
                K_con[me:me + mr_o] = o.K_con[me:]
                if self.n_x > mo:
                    A_impl[mo:, mo:] = -LAM_DECAY * np.eye(self.n_x - mo)
            elif mr > 0:
                A_impl[me:, me:] = -LAM_DECAY * np.eye(mr)
            Ad, G = _zoh(A_impl, np.column_stack([B_u, B_y]), self.dt)
            self._tables.append(dict(obj=o, A_d=Ad, G_u=G[:, 0], G_y=G[:, 1],
                                     K_con=K_con))
        # supervisor parameters
        self.cost_tau = float(cost_tau)
        self.fast_tau = float(fast_tau)
        self.calib = int(calib_steps)
        self.r_up = float(ratio_up)
        self.r_back = float(ratio_back)
        self.panic_ratio = float(panic_ratio)
        self.dwell = int(dwell_steps)
        self.dwell_up = int(dwell_up_steps)
        self.panic_lock0 = int(panic_lock_steps)
        self.quiet_floor = float(quiet_floor)**2
        self.quiet_cap = float(quiet_cap)**2
        self.panic_abs = float(panic_abs)**2
        self.mem_fresh = int(mem_fresh_steps)
        self.reset_adaptation()
        if verbose:
            kinds = "/".join(r['kind'] for r in self.rung_defs)
            print(f"[A-ESO-ADRC] {self.n_rungs}-rung ladder ({kinds}), "
                  f"robust rung = {self.robust_rung} "
                  f"(cost-supervised, memory + panic + escalating locks)")

    # ------------------------------------------------------------------
    def reset_adaptation(self):
        self.i_rung = 0
        self._E_slow = 0.0
        self._E_fast = 0.0
        self._E_quiet = None
        self._k = 0
        self._k_last_sw = 0
        self._k_panic = -10**9
        self._n_panic = 0
        self._E_ff = 0.0           # 25 ms EMA of E_fast (trend ref)
        self._E_ss = 0.0           # 50 ms EMA of E_slow (trend ref)
        self._cost_mem = [None] * self.n_rungs
        self._cost_k = [-10**9] * self.n_rungs
        self._failed = [False] * self.n_rungs
        self._perfward = False
        self._Ef_sw = 0.0
        self.history_rung = []
        self.history_cost = []

    # Frozen-rung continuous realization for the monodromy SLD.
    def controller_realization(self, rung=None):
        o = self._tables[self.i_rung if rung is None else int(rung)]['obj']
        return o.controller_realization()

    # ------------------------------------------------------------------
    def _switch(self, target):
        self._cost_mem[self.i_rung] = self._E_slow
        self._cost_k[self.i_rung] = self._k
        if int(target) > self.i_rung:
            # left robust-ward => this rung failed here; E_slow lags a divergence
            # caught early, so a cost snapshot alone would under-report it. The
            # flag lasts the whole pass: never RELAX onto a rung that failed.
            self._failed[self.i_rung] = True
        self._perfward = int(target) < self.i_rung
        self.i_rung = int(target)
        self._k_last_sw = self._k
        self._Ef_sw = max(self._E_fast, self.quiet_floor)
        self._zero_res = True                 # zero resonant substate at next step

    def _fresh(self, i):
        return (self._cost_mem[i] is not None
                and self._k - self._cost_k[i] <= self.mem_fresh)

    def step(self, z_prev, u_prev, y_meas):
        if getattr(self, '_zero_res', False):
            z_prev = z_prev.copy()
            z_prev[3 * self.n:] = 0.0
            self._zero_res = False
        t = self._tables[self.i_rung]
        z = t['A_d'] @ z_prev + t['G_u'] * u_prev + t['G_y'] * y_meas
        self._k += 1

        # --- cost monitors -----------------------------------------------------
        af = self.dt / self.fast_tau
        as_ = self.dt / self.cost_tau
        y2 = y_meas * y_meas
        self._E_fast = (1 - af) * self._E_fast + af * y2
        self._E_slow = (1 - as_) * self._E_slow + as_ * y2
        aff = self.dt / (self.fast_tau * 16)
        ass = self.dt / (self.cost_tau * 5)
        self._E_ff = (1 - aff) * self._E_ff + aff * self._E_fast
        self._E_ss = (1 - ass) * self._E_ss + ass * self._E_slow
        if self._k >= self.calib:
            cand = min(max(self._E_slow, self.quiet_floor), self.quiet_cap)
            self._E_quiet = cand if self._E_quiet is None \
                else min(self._E_quiet, cand)

        # --- PANIC = DIVERGENCE detector: energy high AND RISING ---------------
        # (a ringing-DOWN plate inherited from the previous rung has high but
        # decaying energy and must not trigger it). Cascade one rung toward the
        # robust end — jumping straight to the last rung would fly past a
        # middle rung whose basin still contains the current amplitude.
        panic_level = self.panic_abs if self._E_quiet is None \
            else max(self.panic_ratio * self._E_quiet, self.panic_abs)
        # Re-arm delay: right after ANY switch the trend EMAs still reflect the
        # previous rung — give the new rung ~15 ms to bend the energy trend.
        # Trend margin x2: at a FLAT (bounded) ring the fast EMA oscillates about
        # its 25 ms average within a tooth period, so only a sustained doubling
        # counts as growth (true divergence reaches x20+ within the EMA lag).
        ksw = self._k - self._k_last_sw
        # PROBE ABORT: a performance-ward probe from a quiet robust rung must be
        # aborted at the first sign of growth (x3 the energy at the switch) —
        # beyond the LQG margin no rung can ring down from a large excursion
        # (every Floquet radius is >= 1 there), so the escape must fire while
        # the amplitude is still inside the fallback's bounded regime.
        if (self._perfward and 60 <= ksw < self.dwell
                and self._E_fast > 3.0 * self._Ef_sw
                and self._E_fast > self.quiet_floor):
            self._switch(self.i_rung + 1)
            self._k_panic = self._k
            self._n_panic += 1
            ksw = 0
        hard = ksw >= 60 and self._E_fast > 8.0 * self._E_ff
        soft = ksw >= 300 and self._E_fast > 2.0 * self._E_ff
        if (self.i_rung != self.robust_rung
                and self._E_fast > panic_level and (hard or soft)):
            # Severity picks the target: EXPLOSIVE growth jumps straight to the
            # certified end (no time to audition intermediate rungs); moderate
            # growth cascades one rung, so a middle rung whose basin still
            # contains the current amplitude gets its chance (the -8 % hole).
            self._switch(self.robust_rung if hard else self.i_rung + 1)
            self._k_panic = self._k
            self._n_panic += 1

        # --- slow supervisor -----------------------------------------------------
        # Escalate quickly (dwell_up) when the cost is elevated AND not in
        # sustained decline (a slowly recovering rung — ring-down can take
        # hundreds of ms near a marginal Floquet radius — is left to finish the
        # job). Performance-ward moves use the long dwell, the post-panic locks
        # (escalating 2x per panic) and the per-rung cost memory.
        elif (self._E_quiet is not None
                and self._k - self._k_last_sw >= self.dwell_up):
            lock = self.panic_lock0 * 2**min(max(self._n_panic - 1, 0), 5)
            perfward_ok = self._k - self._k_panic >= lock
            long_ok = self._k - self._k_last_sw >= self.dwell
            r = self._E_slow / self._E_quiet
            declining = self._E_slow < 0.95 * self._E_ss
            if r > self.r_up and not declining:
                up = self.i_rung + 1          # toward robust
                dn = self.i_rung - 1          # toward performance
                if up < self.n_rungs and not (self._fresh(up)
                        and self._cost_mem[up] >= self._E_slow):
                    self._switch(up)
                elif (dn >= 0 and perfward_ok and long_ok
                        and (not self._failed[dn]
                             or (self.i_rung == self.robust_rung
                                 and self._k - self._k_last_sw >= 2000))
                        and not (self._fresh(dn)
                                 and self._cost_mem[dn] >= self._E_slow)):
                    # desperation probe: at the robust end with a persistently
                    # elevated cost (a sensitivity hole), a previously-failed
                    # rung is the only remaining option — panic guards the try.
                    self._switch(dn)
            elif (r < self.r_back and self.i_rung > 0 and perfward_ok
                    and long_ok):
                dn = self.i_rung - 1
                # never relax onto a rung that failed this pass, nor onto one
                # freshly measured as bad
                if not self._failed[dn] and not (self._fresh(dn)
                        and self._cost_mem[dn] > self.r_up * self._E_quiet):
                    self._switch(dn)

        # --- control law at the current rung -----------------------------------
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
