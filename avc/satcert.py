"""SatCERT: regional stability certificates for the SATURATED,
time-periodic, delayed milling closed loop.

Formulation (declared scope, cf. docs/new_strategy_satcert_outline.md):

1. LIFT. The closed loop (plant + deployed controller + regenerative
   delay) is lifted per tooth-passing period into m interval maps by the
   scalar-history semi-discretization of avc.sld:

       z_{k+1} = G_k z_k + Gam_k * psi_k + g_k,      k = 0..m-1

   with z = [x_p | x_K | w-history], psi = dz_{Vmax}(v) the deadzone of
   the commanded voltage v_k = K_row z_k, Gam_k the exact ZOH map of the
   (negative) actuation path (plus optional anti-windup rows), and g_k
   the affine static feed-ripple force. The certificates hold for this
   semi-discretized model at declared resolution m (convergence checked
   numerically); the deadzone is held per interval, an approximation
   cross-validated against the 50 kHz nonlinear simulator.

2. FORCED ORBIT + HEADROOM. The periodic forced orbit z*_k solves the
   affine periodic system; the certificates apply to the VARIATIONAL
   dynamics about z*, which is required to be strictly unsaturated:
   the residual headroom  u_bar = V_max - max_k |v*_k| > 0  becomes the
   deadzone bound of the variational nonlinearity (standard shift
   argument: dz_{Vmax}(v*+dv) satisfies the generalized-sector
   inequality with the REDUCED threshold u_bar).

3. CERTIFICATE (region of guaranteed linear behavior). Because the
   forced orbit is strictly unsaturated with headroom u_bar, any
   variational trajectory whose commanded voltage never exceeds u_bar
   keeps the deadzone inactive, evolves exactly linearly, and decays
   (linear Floquet radius < 1 is checked). The maximal such region is
   the maximal output-admissible set of Gilbert & Tan (1991) for the
   period map:

       O_inf = { dz : |v*_k + R_k Phi^j dz| <= V_max  for all j >= 0, k },
       R_k = K_row G_{k-1} ... G_0,   Phi = G_{m-1} ... G_0,

   with the PHASE-RESOLVED headroom V_max - |v*_k| (v* is the periodic
   forced voltage). O_inf is invariant by the shift property (the
   constraint set of Phi*dz is the j+1 tail of the constraint set of
   dz, and v* is periodic) and EXACT for the lifted model at
   resolution m (voltages are certified at the m interval boundaries;
   convergence in m is checked numerically). Its extent along the
   surface-step direction e_h — the physical perturbation of interest,
   a step of height h left in the stored surface by the previous pass,
   certified for BOTH signs of h — is closed-form:

       h_max = min_{j,k} (V_max - |v*_k|) / |dv_{j,k}|,
       dv_{j,k} = R_k Phi^j e_h   (voltage response to a unit step);

   the phase resolution matters: the forced peak and the step-response
   peak generally occur at different phases of the tooth period, so
   the uniform-headroom bound  u_bar / max|dv|  (also reported, meta
   ['h_uniform']) is conservative exactly by the phase misalignment.
   Sound, solver-free, and tight in the reported direction; the price
   is that certified trajectories never clip — the certificate refuses
   (rather than exploits) the saturated regime, which is exactly the
   guarantee a process planner needs.

All SI units. No SDP solver required (the earlier LMI formulations are
documented in the repository history; at N ~ 61 they exceed memory or
lose feasibility to conditioning, while this closed form is exact in
the reported direction).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

from . import milling
from .controller import Controller
from .modal import ModalModel


# ---------------------------------------------------------------------------
# 1. lifted saturated model
# ---------------------------------------------------------------------------
@dataclass
class LiftedSat:
    """Per-interval maps of the lifted saturated closed loop."""

    G: list                      # m matrices (N, N)
    Gam: list                    # m vectors (N,)  (psi input column, signed)
    g: list                      # m vectors (N,)  (affine forced term)
    K_row: np.ndarray            # (N,) commanded voltage v = K_row @ z
    e_h: np.ndarray              # (N,) surface-step perturbation direction
    N: int
    m: int
    tau: float
    meta: dict = field(default_factory=dict)

    # ---- forced periodic orbit ---------------------------------------
    def forced_orbit(self):
        """Periodic z*_k (m+1 points, z*_m = z*_0) and voltages v*_k."""
        N, m = self.N, self.m
        Phi = np.eye(N)
        acc = np.zeros(N)
        for k in range(m):
            acc = self.G[k] @ acc + self.g[k]
            Phi = self.G[k] @ Phi
        z0 = np.linalg.solve(np.eye(N) - Phi, acc)
        zs = [z0]
        for k in range(m):
            zs.append(self.G[k] @ zs[-1] + self.g[k])
        v = np.array([float(self.K_row @ z) for z in zs[:-1]])
        return np.array(zs), v

    def headroom(self, v_max: float) -> float:
        """u_bar = V_max - max_k |v*_k| (must be > 0 to certify)."""
        _, v = self.forced_orbit()
        return float(v_max - np.max(np.abs(v)))

    def monodromy(self) -> np.ndarray:
        Phi = np.eye(self.N)
        for k in range(self.m):
            Phi = self.G[k] @ Phi
        return Phi


def lift_saturated(mm: ModalModel, x_T: float, ap: float, rpm: float,
                   controller: Controller, m: int = 24,
                   aw_gain: np.ndarray | None = None) -> LiftedSat:
    """Build the lifted saturated closed loop (no delayed taps supported;
    the campaign controllers have k_r = 0).

    aw_gain: optional (nk,) anti-windup vector: xK_dot += aw_gain*(sat(v)-v)
    = -aw_gain * psi.
    """
    if controller is None or controller.gd1 is not None:
        raise ValueError("lift_saturated needs a tap-free controller")
    p = mm.params
    tau = p.tooth_passing_delay(rpm)
    dt = tau / m
    t_sub = (np.arange(m * 8) + 0.5) * (tau / (m * 8))
    a4_fine = milling.alpha4_time(p, ap, rpm, t_sub)
    a4 = a4_fine.reshape(m, 8).mean(axis=1)
    ft = p.process.ft

    A0, Bu, Bf, Cs, Cw = mm.abcd(x_T, alpha4=0.0)
    Ady = mm.abcd(x_T, alpha4=1.0)[0] - A0
    n2 = A0.shape[0]
    bu = np.asarray(Bu, float).reshape(n2)
    bf_col = np.asarray(Bf, float).reshape(n2)
    cs = np.asarray(Cs, float).reshape(n2)
    cw = np.asarray(Cw, float).reshape(n2)

    nk = controller.nk
    Ak = np.asarray(controller.Ak, float).reshape(nk, nk)
    Bk = np.asarray(controller.Bk, float).reshape(nk)
    Ck = np.asarray(controller.Ck, float).reshape(nk)
    Dk = float(np.asarray(controller.Dk, float).ravel()[0])
    ck_eff = Ck.copy()
    if controller.gd0 is not None:
        ck_eff += np.asarray(controller.gd0, float).reshape(nk)

    na = n2 + nk
    Abase = np.zeros((na, na))
    Abase[:n2, :n2] = A0 + Dk * np.outer(bu, cs)
    Abase[:n2, n2:] = np.outer(bu, ck_eff)
    Abase[n2:, :n2] = np.outer(Bk, cs)
    Abase[n2:, n2:] = Ak
    Dy = np.zeros((na, na))
    Dy[:n2, :n2] = Ady

    Bw = np.zeros(na)
    Bw[:n2] = bf_col
    cw_a = np.zeros(na)
    cw_a[:n2] = cw

    # psi input (continuous): plant gets bu*(v - psi) -> -bu * psi;
    # anti-windup: xK_dot += -aw_gain * psi
    Bpsi = np.zeros(na)
    Bpsi[:n2] = -bu
    if aw_gain is not None:
        Bpsi[n2:] = -np.asarray(aw_gain, float).reshape(nk)
    # affine static ripple force: Fy_static = -a4(t)*ft through bf
    # (delayed & current-time regenerative parts are in the homogeneous map)

    nh = m + 1
    N = na + nh
    iw = na
    shift = np.eye(m)

    K_row = np.zeros(N)
    K_row[:n2] = Dk * cs
    K_row[n2:na] = ck_eff

    e_h = np.zeros(N)
    e_h[iw:] = 1.0               # surface step: whole stored history offset

    Gs, Gams, gs = [], [], []
    for i in range(m):
        Ai = Abase - a4[i] * Dy
        B = np.column_stack([a4[i] * Bw, Bpsi, -a4[i] * ft * Bw])
        nc = 3
        Maug = np.zeros((na + nc, na + nc))
        Maug[:na, :na] = Ai * dt
        Maug[:na, na:] = B * dt
        E = expm(Maug)

        G = np.zeros((N, N))
        G[:na, :na] = E[:na, :na]
        q_w = E[:na, na]
        G[:na, iw + m] += 0.5 * q_w
        G[:na, iw + m - 1] += 0.5 * q_w
        G[iw + 1: iw + nh, iw: iw + m] = shift
        G[iw, :] = cw_a @ G[:na, :]

        Gam = np.zeros(N)
        Gam[:na] = E[:na, na + 1]
        Gam[iw] = cw_a @ Gam[:na]

        g = np.zeros(N)
        g[:na] = E[:na, na + 2]
        g[iw] = cw_a @ g[:na]

        Gs.append(G)
        Gams.append(Gam)
        gs.append(g)

    return LiftedSat(G=Gs, Gam=Gams, g=gs, K_row=K_row, e_h=e_h, N=N, m=m,
                     tau=tau, meta={"x_T": x_T, "ap": ap, "rpm": rpm,
                                    "aw": aw_gain is not None})


# ---------------------------------------------------------------------------
# 2. regional certificate (common quadratic, per-interval multipliers)
# ---------------------------------------------------------------------------
@dataclass
class CertResult:
    feasible: bool
    h_max: float                 # largest certified surface step [m]
    u_bar: float                 # voltage headroom used [V]
    lin_rho: float               # linear spectral radius (must be < 1)
    status: str
    meta: dict = field(default_factory=dict)


def certify(lift: LiftedSat, v_max: float, horizon: int = 4000,
            decay_tol: float = 1e-10, h_cap: float = 1.0e-3) -> CertResult:
    """Maximal-output-admissible (Gilbert-Tan) region-of-linearity
    certificate for the saturated periodic delayed loop.

    Delivered guarantee (at declared resolution m): for every surface
    step h*e_h with |h| <= h_max, the total commanded voltage
    v*_k + h*dv_{j,k} never leaves the amplifier's linear range, so the
    loop evolves exactly linearly and decays (the linear Floquet radius
    is checked < 1). h_max uses the PHASE-RESOLVED headroom and is
    EXACT for the lifted model in the surface-step direction:

        h_max = min_{j,k : dv != 0} (V_max - |v*_k|) / |dv_{j,k}|

    computed by direct simulation of the unit step over `horizon`
    periods with early stop once the state norm decays below decay_tol
    relative to its peak (the binding constraint occurs in the
    transient; once |dv| decays below (V_max-|v*|)/h_cap the tail
    cannot bind). h_max is capped at h_cap (a step taller than that is
    outside the model's validity anyway). meta reports 'h_uniform'
    (worst-phase bound u_bar/max|dv|), 'h_l2' (inner voltage-energy-
    Gramian ellipsoid, uniform headroom), and 'v_peak_per_m'
    (max|dv| per meter of step).
    """
    rho = float(np.max(np.abs(np.linalg.eigvals(lift.monodromy()))))
    _, v_star = lift.forced_orbit()
    head = v_max - np.abs(v_star)          # (m,) phase-resolved headroom
    u_bar = float(np.min(head))
    if rho >= 1.0 or u_bar <= 0.0:
        return CertResult(False, 0.0, u_bar, rho,
                          "lin-unstable" if rho >= 1.0 else "forced-saturated",
                          dict(lift.meta))

    z = lift.e_h.copy()
    z_pk = float(np.linalg.norm(z))
    h_max = float(h_cap)
    v_pk = 0.0
    v_sq = 0.0
    for _ in range(horizon):
        for k in range(lift.m):
            dv = abs(float(lift.K_row @ z))
            if dv > 0.0:
                v_pk = max(v_pk, dv)
                v_sq += dv * dv
                h_max = min(h_max, head[k] / dv)
            z = lift.G[k] @ z
        zn = float(np.linalg.norm(z))
        z_pk = max(z_pk, zn)
        if zn < decay_tol * z_pk:
            break
    if v_pk <= 0.0:
        return CertResult(False, 0.0, u_bar, rho, "degenerate",
                          dict(lift.meta))
    return CertResult(True, float(h_max), u_bar, rho, "O-infinity",
                      {**lift.meta, "h_uniform": u_bar / v_pk,
                       "h_l2": u_bar / np.sqrt(v_sq),
                       "v_peak_per_m": v_pk})


# ---------------------------------------------------------------------------
# 3. implementation-exact sampled-data lift (control-rate resolution)
# ---------------------------------------------------------------------------
@dataclass
class SampledSatLift:
    """Period-lifted SAMPLED-DATA saturated closed loop at the real-time
    control rate: continuous plant, ZOH-discretized controller, one
    control period of computation delay (explicit pending-voltage
    state), amplifier deadzone acting on the discrete commanded voltage
    exactly where the real DAC clips. This removes the Pade-latency /
    continuous-controller modeling gap of `lift_saturated` (which
    underestimates the forced-orbit voltage by 5-13 % as ap grows and
    is therefore unsound for headroom certification near the envelope
    boundary — measured against the 100 kHz nonlinear simulator).

    Phi   : (N, N) homogeneous period map (tick 0 -> tick m)
    acc   : (N,)  affine period accumulation (forced ripple)
    V     : (m, N) commanded voltage at each tick as a functional of
            the period-start state;  v_aff the affine offsets
    e_h   : surface-step direction (all stored history slots)
    """

    Phi: np.ndarray
    acc: np.ndarray
    V: np.ndarray
    v_aff: np.ndarray
    e_h: np.ndarray
    N: int
    m: int
    tau: float
    dt: float
    meta: dict = field(default_factory=dict)

    def forced_voltages(self) -> np.ndarray:
        z0 = np.linalg.solve(np.eye(self.N) - self.Phi, self.acc)
        return self.V @ z0 + self.v_aff

    def rho(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.Phi))))


def lift_sampled(mm: ModalModel, x_T: float, ap: float, rpm: float,
                 controller: Controller, m: int | None = None,
                 n_sub: int = 8) -> SampledSatLift:
    """Build the sampled-data lift. `controller` must be the deployed
    controller WITHOUT the Pade latency augmentation (latency=False):
    the one-control-period computation delay is modeled exactly by the
    pending-voltage state, mirroring avc.simulate. m defaults to
    round(tau / Ts) at Ts = 1/ctrl_rate_hz, so dt = tau/m deviates from
    the true control period by < 0.2 % (declared); the regenerative
    delay is then exactly m ticks."""
    if controller is None or controller.gd1 is not None:
        raise ValueError("lift_sampled needs a tap-free controller")
    p = mm.params
    tau = p.tooth_passing_delay(rpm)
    rate = getattr(p.design, "ctrl_rate_hz", 50e3)
    if m is None:
        m = int(round(tau * rate))
    dt = tau / m
    t_sub = (np.arange(m * n_sub) + 0.5) * (tau / (m * n_sub))
    a4 = milling.alpha4_time(p, ap, rpm, t_sub).reshape(m, n_sub).mean(axis=1)
    ft = p.process.ft

    A0, Bu, Bf, Cs, Cw = mm.abcd(x_T, alpha4=0.0)
    Ady = mm.abcd(x_T, alpha4=1.0)[0] - A0
    n2 = A0.shape[0]
    bu = np.asarray(Bu, float).reshape(n2)
    bf_col = np.asarray(Bf, float).reshape(n2)
    cs = np.asarray(Cs, float).reshape(n2)
    cw = np.asarray(Cw, float).reshape(n2)

    nk = controller.nk
    Ak = np.asarray(controller.Ak, float).reshape(nk, nk)
    Bk = np.asarray(controller.Bk, float).reshape(nk)
    Ck = np.asarray(controller.Ck, float).reshape(nk)
    Dk = float(np.asarray(controller.Dk, float).ravel()[0])
    ck_eff = Ck.copy()
    if controller.gd0 is not None:
        ck_eff += np.asarray(controller.gd0, float).reshape(nk)
    # ZOH discretization of the controller at dt (as in avc.simulate)
    Mk = np.zeros((nk + 1, nk + 1))
    Mk[:nk, :nk] = Ak * dt
    Mk[:nk, nk] = Bk * dt
    Ek = expm(Mk)
    Akd, Bkd = Ek[:nk, :nk], Ek[:nk, nk]

    # per-interval plant ZOH maps: inputs [u_applied, w_tau, ripple]
    Ea = np.empty((m, n2, n2))
    Bcols = np.empty((m, n2, 3))
    for i in range(m):
        Ai = A0 - a4[i] * Ady
        B = np.column_stack([bu, a4[i] * bf_col, -a4[i] * ft * bf_col])
        Maug = np.zeros((n2 + 3, n2 + 3))
        Maug[:n2, :n2] = Ai * dt
        Maug[:n2, n2:] = B * dt
        E = expm(Maug)
        Ea[i] = E[:n2, :n2]
        Bcols[i] = E[:n2, n2:]

    N = n2 + nk + 1 + m                    # x_p | x_K | u_pend | hist ring
    iu = n2 + nk
    ih = iu + 1
    # propagate identity + affine column through one period, recording
    # the commanded-voltage rows (ring buffer: slot k%m holds w_{k-m})
    Z = np.zeros((N, N + 1))
    Z[:N, :N] = np.eye(N)
    Vrows = np.empty((m, N + 1))
    for k in range(m):
        Xp = Z[:n2]
        Xk = Z[n2:iu]
        y = cs @ Xp
        v = ck_eff @ Xk + Dk * y           # commanded at tick k
        Vrows[k] = v
        u_apply = Z[iu].copy()             # pending from tick k-1
        w_now = cw @ Xp
        w_tau = Z[ih + (k % m)].copy()
        Z[ih + (k % m)] = w_now
        Z[iu] = v                          # deadzone inactive: linear maps
        aff = np.zeros(N + 1)
        aff[N] = 1.0
        Z[:n2] = (Ea[k] @ Xp + np.outer(Bcols[k][:, 0], u_apply)
                  + np.outer(Bcols[k][:, 1], w_tau)
                  + np.outer(Bcols[k][:, 2], aff))
        Z[n2:iu] = Akd @ Xk + np.outer(Bkd, y)
    e_h = np.zeros(N)
    e_h[ih:] = 1.0
    return SampledSatLift(Phi=Z[:, :N].copy(), acc=Z[:, N].copy(),
                          V=Vrows[:, :N].copy(), v_aff=Vrows[:, N].copy(),
                          e_h=e_h, N=N, m=m, tau=tau, dt=dt,
                          meta={"x_T": x_T, "ap": ap, "rpm": rpm,
                                "rate": rate, "nk": nk})


def certify_sampled(lift: SampledSatLift, v_max: float,
                    horizon: int = 6000, decay_tol: float = 1e-10,
                    h_cap: float = 1.0e-3) -> CertResult:
    """Gilbert-Tan region-of-linearity certificate on the sampled-data
    lift (implementation-exact: the deadzone acts on the discrete
    commanded voltage, which is certified at EVERY control tick, and
    the forced orbit is the exact periodic response of the digital
    loop). Phase-resolved headroom:

        h_max = min_{j,k} (V_max - |v*_k|) / |dv_{j,k}(e_h)|.
    """
    rho = lift.rho()
    v_star = lift.forced_voltages()
    head = v_max - np.abs(v_star)
    u_bar = float(np.min(head))
    if rho >= 1.0 or u_bar <= 0.0:
        return CertResult(False, 0.0, u_bar, rho,
                          "lin-unstable" if rho >= 1.0 else "forced-saturated",
                          dict(lift.meta))
    z = lift.e_h.copy()
    z_pk = float(np.linalg.norm(z))
    h_plus = h_minus = float(h_cap)     # one-sided exact onsets (+h / -h)
    v_pk = 0.0
    for _ in range(horizon):
        dv = lift.V @ z
        up, dn = dv > 0.0, dv < 0.0
        if np.any(up):
            v_pk = max(v_pk, float(np.max(dv[up])))
            h_plus = min(h_plus,
                         float(np.min((v_max - v_star[up]) / dv[up])))
            h_minus = min(h_minus,
                          float(np.min((v_max + v_star[up]) / dv[up])))
        if np.any(dn):
            v_pk = max(v_pk, float(np.max(-dv[dn])))
            h_plus = min(h_plus,
                         float(np.min((v_max + v_star[dn]) / (-dv[dn]))))
            h_minus = min(h_minus,
                          float(np.min((v_max - v_star[dn]) / (-dv[dn]))))
        z = lift.Phi @ z
        zn = float(np.linalg.norm(z))
        z_pk = max(z_pk, zn)
        if zn < decay_tol * z_pk:
            break
    if v_pk <= 0.0:
        return CertResult(False, 0.0, u_bar, rho, "degenerate",
                          dict(lift.meta))
    h_max = min(h_plus, h_minus)        # certified for BOTH step signs
    return CertResult(True, float(h_max), u_bar, rho, "O-infinity-sampled",
                      {**lift.meta, "h_plus": h_plus, "h_minus": h_minus,
                       "h_uniform": u_bar / v_pk, "v_peak_per_m": v_pk,
                       "forced_peak": float(np.max(np.abs(v_star)))})


def certified_depth(mm: ModalModel, x_T: float, rpm: float,
                    controller: Controller, v_max: float,
                    h_req: float, ap_hi: float = 5e-3,
                    tol: float = 2e-5) -> tuple:
    """Largest ap whose implementation-exact (sampled-data) certificate
    tolerates a surface step >= h_req in BOTH signs. `controller` must
    be deployed WITHOUT Pade latency (latency=False). Returns
    (ap_cert [m], CertResult at ap_cert)."""
    def ok(ap):
        if ap <= 0:
            return None
        return certify_sampled(lift_sampled(mm, x_T, ap, rpm, controller),
                               v_max)

    lo, hi = 0.0, float(ap_hi)
    r_lo = None
    r = ok(hi)
    if r is not None and r.feasible and r.h_max >= h_req:
        return hi, r
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        r = ok(mid)
        if r is not None and r.feasible and r.h_max >= h_req:
            lo, r_lo = mid, r
        else:
            hi = mid
    return lo, r_lo
