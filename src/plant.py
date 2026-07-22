"""
Modal delay-differential model of thin-walled cantilever-plate milling.

Governing equation (paper Eq. 13, reduced to N modes, mass-normalised so M = I):

    q'' + C q' + K q + g(t) a4 Phi Phi^T [q(t) - q(t-tau)]
                                   = g(t) Phi f_static  +  Hpe u(t)

    y(t) = Phi^T q(t)                       (plate displacement at the sensor)

where
    q                modal coordinates (N,)
    C = diag(2 zeta_i w_i),  K = diag(w_i^2)
    Phi              mass-normalised mode shape at the milling/sensor point
    a4               absolute milling-force coefficient (N/m)        [regenerative]
    g(t) in {0,1}    tooth-engagement window  (non-smooth intermittent milling)
    tau              regenerative time delay = tooth-passing period
    f_static         quasi-static cutting force (feed related)       [forced vib.]
    Hpe u            piezo modal actuation                           [control]

Two views of the plant are provided:

  * design_state_space()  -> the *nominal linear* plant  u -> y  (and d -> y),
    a stable but extremely lightly damped 3-mode system.  This is what every
    controller is designed on.  The milling regeneration is deliberately NOT
    part of it: following the paper, the time-delay cutting force is treated as
    a disturbance / structured uncertainty that the controller must reject.

  * MillingPlant  -> the *true* nonlinear simulated plant: the design plant plus
    the intermittent regenerative delay feedback, actuator saturation,
    measurement noise, and optional parameter perturbations for robustness tests.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

from . import config as C


# --------------------------------------------------------------------------- #
#  Nominal linear modal matrices                                              #
# --------------------------------------------------------------------------- #
def modal_matrices(omega=None, zeta=None):
    """Return (K, Cd) modal stiffness/damping (mass-normalised, M = I)."""
    omega = C.OMEGA_N if omega is None else np.asarray(omega, float)
    zeta  = C.MODE_ZETA if zeta is None else np.asarray(zeta, float)
    K  = np.diag(omega ** 2)
    Cd = np.diag(2.0 * zeta * omega)
    return K, Cd


def piezo_input_vector(gain=None, phi=None):
    """
    Modal input vector Hpe (N,) mapping control voltage u [V] -> modal force.
    Modelled proportional to the mode shape at the actuator location and scaled
    by a calibrated authority PIEZO_GAIN * H0.  H0 sets the volt->modal-force
    scale so that ~O(10-100 V) yields meaningful control (calibrated in plant).
    """
    gain = C.PIEZO_GAIN if gain is None else gain
    phi  = C.PHI if phi is None else np.asarray(phi, float)
    H0 = 3.0e-2          # base authority [modal-force / V], calibrated
    return gain * H0 * phi


def design_state_space(gain_piezo=None, n_modes=None, zeta_design=None):
    """
    Nominal linear design plant (continuous-time state space):

        x = [q ; q'] in R^{2N}
        x' = A x + Bu u + Bd d
        y  = Cy x  (+ measurement noise added in simulation)

    `n_modes` selects how many modes the *design* model keeps.  The paper designs
    on a REDUCED 2-mode model and represents modes 3+ as additive uncertainty;
    passing n_modes=2 reproduces that (and yields lower-order, gentler, easily
    discretisable robust controllers), while the true simulated plant always
    keeps all C.N_MODES modes.

    `zeta_design` overrides the modal damping used FOR DESIGN ONLY.  Designing the
    H-infinity / mu controllers against a well-damped model (e.g. 15 %) instead of
    the true ~0.2 % stops them from relying on a sharp pole/zero cancellation of
    the lightly damped resonances; the resulting controllers are broadband
    dampers that stay stable when the true modal frequencies drift (e.g. from
    material removal).  The true simulated plant always uses the measured damping.

    Returns dict with A, Bu, Bd, Cy and the mode-shape / matrices used.
    """
    n = C.N_MODES if n_modes is None else int(n_modes)
    omega = C.OMEGA_N[:n]
    zeta = C.MODE_ZETA[:n] if zeta_design is None else np.full(n, float(zeta_design))
    K, Cd = modal_matrices(omega, zeta)
    Hpe = piezo_input_vector(gain_piezo)[:n]
    phi = C.PHI[:n]

    A = np.block([[np.zeros((n, n)), np.eye(n)],
                  [-K,               -Cd]])
    Bu = np.vstack([np.zeros((n, 1)), Hpe.reshape(-1, 1)])          # u -> states
    Bd = np.vstack([np.zeros((n, 1)), phi.reshape(-1, 1)])          # force -> states
    Cy = np.hstack([phi.reshape(1, -1), np.zeros((1, n))])          # states -> y

    return dict(A=A, Bu=Bu, Bd=Bd, Cy=Cy, n=n, K=K, Cd=Cd, Hpe=Hpe, phi=phi)


# --------------------------------------------------------------------------- #
#  Tooth-engagement window (intermittent / non-smooth milling force)          #
# --------------------------------------------------------------------------- #
def engagement(t, tau, duty=None, smooth=True):
    """
    Periodic tooth-engagement window g(t) in [0,1], period = tau, duty = DUTY.
    `smooth=True` uses a raised-cosine ramp at entry/exit so the RHS is
    Lipschitz (kinder to the fixed-step integrator) while keeping the same
    engaged fraction; `smooth=False` is a hard rectangular pulse.
    """
    duty = C.DUTY if duty is None else duty
    ph = (t / tau) % 1.0                    # phase in [0,1)
    if not smooth:
        return 1.0 if ph < duty else 0.0
    # raised-cosine window of width `duty`, centred in the engaged part
    if ph >= duty:
        return 0.0
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * ph / duty))


# --------------------------------------------------------------------------- #
#  Spatial variation of the milling-point mode shape along the free upper edge #
# --------------------------------------------------------------------------- #
def edge_mode_profile(xi):
    """
    Multipliers (one per mode) applied to the reference mode shape as the tool
    advances along the free upper edge, xi in [0,1] (0 = pass start, 1 = end).

    These reproduce the paper's "varying dynamic characteristics" (its Fig. 7,
    D_Pr^T D_Pr varying with milling position): the first mode is almost constant
    along the edge, while the higher modes vary strongly (their spatial profiles
    have nodes/antinodes along the edge).  Kept strictly positive so the modal
    coupling never changes sign spuriously.
    """
    xi = np.clip(xi, 0.0, 1.0)
    s1 = 1.0 + 0.20 * np.sin(np.pi * xi)           # mode 1: nearly constant
    s2 = 1.0 + 0.50 * np.cos(np.pi * xi)           # mode 2: strong variation (1.5 -> 0.5)
    s3 = 1.0 + 0.45 * np.cos(2.0 * np.pi * xi)     # mode 3: two antinodes
    return np.array([s1, s2, s3])[:C.N_MODES]


# --------------------------------------------------------------------------- #
#  True (nonlinear) milling plant used for simulation                         #
# --------------------------------------------------------------------------- #
@dataclass
class MillingPlant:
    """
    True simulated plant: design plant + intermittent regenerative delay term.

    Parameter perturbations (for robustness tests, paper Section 4.2/Fig.16):
        dmass, dstiff : fractional perturbation of modal mass / stiffness
        dzeta         : fractional perturbation of damping ratio
        alpha4_factor : milling-force-coefficient multiplier (0.3 .. 2.9)

    Full-pass options (peripheral milling of the whole upper edge, ~20.4 s):
        moving        : if True the milling contact point advances along the edge
                        over `t_pass`, so the milling-point mode shape (and hence
                        the regenerative coupling and the cutting excitation) vary
                        with time -- the paper's varying dynamic characteristics.
                        The displacement SENSOR and the piezo ACTUATOR stay fixed.
        freq_drift    : if True the modal frequencies drift upward across the pass
                        (material removal shaving the free-edge thickness).
        t_pass        : pass duration [s] (defaults to C.T_PASS = 20.4 s).
    """
    cut_gain: float = None
    piezo_gain: float = None
    tau: float = None
    dmass: float = 0.0
    dstiff: float = 0.0
    dzeta: float = -0.20        # Fig.16 uses 80% of nominal damping
    alpha4_factor: float = None
    smooth_engage: bool = True
    moving: bool = False
    freq_drift: bool = False
    t_pass: float = None
    seed: int = C.SEED

    # filled in __post_init__
    A: np.ndarray = field(default=None, repr=False)
    Areg: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        self.cut_gain   = C.CUT_GAIN if self.cut_gain is None else self.cut_gain
        self.piezo_gain = C.PIEZO_GAIN if self.piezo_gain is None else self.piezo_gain
        self.tau        = C.TAU_S if self.tau is None else self.tau
        self.alpha4_factor = (C.ALPHA4_NOMINAL_FACTOR if self.alpha4_factor is None
                              else self.alpha4_factor)

        n = C.N_MODES
        # perturbed modal data.  In a mass-normalised model (M = I) a modal-mass
        # perturbation of factor (1+dmass) scales omega by 1/sqrt(1+dmass) AND the
        # mass-normalised mode shape phi by 1/sqrt(1+dmass) (since phi^T M phi = 1).
        # Rescaling phi is essential: it drives the regenerative gain G ~ phi phi^T,
        # the disturbance/sensor maps Bd, Cy and the piezo input Hpe.  Omitting it
        # would make a mass perturbation a no-op whenever dmass == dstiff (omega
        # cancels), silently voiding the mass/stiffness part of the robustness study.
        self.omega0 = C.OMEGA_N * np.sqrt((1.0 + self.dstiff) / (1.0 + self.dmass))
        self.zeta0  = C.MODE_ZETA * (1.0 + self.dzeta)
        K, Cd = modal_matrices(self.omega0, self.zeta0)
        # FIXED sensor / actuator mode shape (right-upper corner / piezo patch)
        self.phi = C.PHI / np.sqrt(1.0 + self.dmass)
        self.Hpe = piezo_input_vector(self.piezo_gain, phi=self.phi)

        self.A = np.block([[np.zeros((n, n)), np.eye(n)],
                           [-K,               -Cd]])
        self.Bu = np.vstack([np.zeros((n, 1)), self.Hpe.reshape(-1, 1)])
        self.Bd = np.vstack([np.zeros((n, 1)), self.phi.reshape(-1, 1)])
        self.Cy = np.hstack([self.phi.reshape(1, -1), np.zeros((1, n))])

        # absolute milling-force coefficient and regenerative matrix (static case)
        self.alpha4 = self.cut_gain * C.ALPHA4_BASE * self.alpha4_factor
        self.G = self.alpha4 * np.outer(self.phi, self.phi)     # (n,n)

        # static (feed) cutting force amplitude   [modal force units]
        self.f_static = C.CONDITION_S["feed_per_tooth"] * C.ALPHA4_BASE
        self.t_pass = C.T_PASS if self.t_pass is None else self.t_pass
        self.drift = C.FREQ_DRIFT_PASS[:n] if self.freq_drift else np.zeros(n)
        self._time_varying = self.moving or self.freq_drift
        self.n = n
        self._rng = np.random.default_rng(self.seed)

    # ---- continuous-time RHS -------------------------------------------- #
    def deriv(self, t, x, x_tau, u):
        """
        State derivative of the true plant.
            x     : current state [q; qdot] (2n,)
            x_tau : delayed state [q(t-tau); qdot(t-tau)] (2n,)
            u     : scalar control voltage (already saturated)
        """
        n = self.n
        q     = x[:n]
        q_tau = x_tau[:n]
        g = engagement(t, self.tau, smooth=self.smooth_engage)

        if not self._time_varying:
            # ---- fixed-position fast path (single-point milling snapshot) ----
            xdot = self.A @ x
            xdot[n:] -= g * (self.G @ (q - q_tau))          # regenerative (destab.)
            xdot[n:] += g * self.f_static * self.phi        # feed excitation
            xdot[n:] += self.Hpe * u                        # control
            return xdot

        # ---- full-pass path: milling point moves, frequencies may drift ----
        prog = np.clip(t / self.t_pass, 0.0, 1.0)
        omega = self.omega0 * (1.0 + self.drift * prog)     # material removal
        Kdiag = omega ** 2
        Cdiag = 2.0 * self.zeta0 * omega
        # milling-point mode shape (sensor/actuator remain the fixed self.phi)
        phi_mill = self.phi * edge_mode_profile(prog) if self.moving else self.phi

        qd = x[n:]
        qdd = -Kdiag * q - Cdiag * qd                       # diagonal modal dynamics
        proj = float(phi_mill @ (q - q_tau))                # milling-point rel. displ.
        qdd -= g * self.alpha4 * proj * phi_mill            # regenerative (destab.)
        qdd += g * self.f_static * phi_mill                 # feed excitation
        qdd += self.Hpe * u                                 # control (fixed actuator)
        return np.concatenate([qd, qdd])

    def measure(self, x, noise=True):
        y = float((self.Cy @ x).item())
        if noise:
            y += self._rng.normal(0.0, C.MEAS_NOISE_STD)
        return y

    def displacement(self, x):
        return float(self.phi @ x[:self.n])
