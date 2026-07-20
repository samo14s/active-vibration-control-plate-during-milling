"""Controller synthesis: H-infinity (DGKF two-Riccati), LQG, delayed-PD
baseline, position-scheduled gain grid, and the robust point design.

Design plant (weighted generalized plant, frozen scheduling parameter):

    states  x  = [eta, eta_dot | x_Wf (| x_Wa)]
    exog.   w  = [f_regen residual (1 N scale), sensor noise n (| Delta_a out)]
    control u  = piezo voltage [V]
    perf.   z  = [Wf * w_mill, Wu * u (| Wa * u)]
    meas.   y  = y_s + Wn * n (| + Delta_a output)

Weights (from params.design):
    Wf(s) = wf_dc_gain * wb^2 / (s^2 + 2*zf*wb*s + wb^2)   [1/m], zf = 0.707
    Wu    = wu_gain  (static, ~1/V_max)                    [1/V]
    Wn    = wn_gain  (static sensor-noise scaling)         [m]
    Wa(s) = r*(s^2+2*z1*w1*s+w1^2)/(s^2+2*z2*w2*s+w2^2)    [m/V]
            overbounds the u->y_s truncation residual (ref. Eq. (18) form).

All controllers follow the avc.controller.Controller contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import scipy.linalg as sla

from .controller import Controller
from .milling import mean_alpha4
from .modal import ModalModel, reduce_model
from .params import Params

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# ---------------------------------------------------------------------------
# cached FEM -> modal reductions (the FEM eigensolve is the expensive step)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=32)
def cached_reduce_model(params: Params, n_modes: int,
                        height_removed: float) -> ModalModel:
    """reduce_model with memoization (Params is frozen and hashable)."""
    return reduce_model(params, n_modes, height_removed)


# ---------------------------------------------------------------------------
# small linear-algebra / FRF helpers
# ---------------------------------------------------------------------------
def spectral_abscissa(A: np.ndarray) -> float:
    """max Re(eig(A)) [1/s]; < 0 means asymptotically stable."""
    return float(np.max(np.linalg.eigvals(A).real))


def _siso_frf(A: np.ndarray, B: np.ndarray, C: np.ndarray,
              freqs_hz: np.ndarray, D=None) -> np.ndarray:
    """C (jwI - A)^-1 B (+ D) for a SISO channel (B (n,1), C (1,n))."""
    w = 2.0 * np.pi * np.asarray(freqs_hz, dtype=float)
    d0 = 0.0 if D is None else float(np.asarray(D).reshape(-1)[0]) \
        if np.asarray(D).size else 0.0
    if A.shape[0] == 0:
        return np.full(w.shape, d0, dtype=complex)
    lam, V = np.linalg.eig(A)
    if np.linalg.cond(V) < 1e10:
        Bt = np.linalg.solve(V, B[:, 0].astype(complex))
        Ct = (C.astype(complex) @ V)[0]
        return (Ct * Bt) @ (1.0 / (1j * w[None, :] - lam[:, None])) + d0
    n = A.shape[0]
    return np.array([(C @ np.linalg.solve(1j * wi * np.eye(n) - A, B))[0, 0]
                     for wi in w]) + d0


def _frf_maxsv(A, B, C, D, freqs_hz) -> float:
    """max over the grid of the largest singular value of the transfer
    matrix C (jwI - A)^-1 B + D  (grid lower bound of the H-inf norm)."""
    w = 2.0 * np.pi * np.asarray(freqs_hz, dtype=float)
    n = A.shape[0]
    use_eig = False
    try:
        lam, V = np.linalg.eig(A)
        if np.linalg.cond(V) < 1e10:
            Bt = np.linalg.solve(V, B.astype(complex))
            Ct = C.astype(complex) @ V
            use_eig = True
    except np.linalg.LinAlgError:
        pass
    out = 0.0
    for wi in w:
        if use_eig:
            G = Ct @ (Bt / (1j * wi - lam)[:, None]) + D
        else:
            G = C @ np.linalg.solve(1j * wi * np.eye(n) - A, B) + D
        out = max(out, np.linalg.svd(G, compute_uv=False)[0])
    return float(out)


def _care(a, b, q, r, s=None):
    """Stabilizing solution X of  a'X + Xa - (Xb+s) r^-1 (b'X+s') + q = 0.

    Tries scipy.linalg.solve_continuous_are (handles the sign-indefinite r
    of the H-infinity game Riccati); falls back to an ordered-Schur
    Hamiltonian solver. Raises numpy.linalg.LinAlgError if no stabilizing
    solution exists.
    """
    n = a.shape[0]
    if s is None:
        s = np.zeros((n, r.shape[0]))
    try:
        x = sla.solve_continuous_are(a, b, q, r, s=s, balanced=True)
        return 0.5 * (x + x.T)
    except Exception:
        pass
    rinv = np.linalg.inv(r)
    a_h = a - b @ rinv @ s.T
    q_h = q - s @ rinv @ s.T
    H = np.block([[a_h, -b @ rinv @ b.T], [-q_h, -a_h.T]])
    try:
        _T, Z, sdim = sla.schur(H, output="real", sort="lhp")
    except Exception as exc:  # LinAlgError or convergence failure
        raise np.linalg.LinAlgError(str(exc))
    if sdim != n:
        raise np.linalg.LinAlgError("no n-dim stable invariant subspace")
    U1, U2 = Z[:n, :n], Z[n:, :n]
    if np.linalg.cond(U1) > 1e12:
        raise np.linalg.LinAlgError("ill-conditioned invariant subspace")
    x = U2 @ np.linalg.inv(U1)
    return 0.5 * (x + x.T)


# ---------------------------------------------------------------------------
# 1. weights
# ---------------------------------------------------------------------------
def make_weights(params: Params) -> dict:
    """State-space (A, B, C, D) tuples of the synthesis weights.

    Wf: 2nd-order low-pass on the milling-point deflection, DC gain
        wf_dc_gain [1/m], corner wf_bandwidth_hz, damping 0.707 (rolls off
        above the ~2.5 kHz chatter band). Balanced modal realization for
        good ARE conditioning.
    Wu: lead cascade on the voltage: DC value wu_gain [1/V], rising by
        (wu_pole_hz/wu_zero_hz)^wu_order between zero and pole. The
        high-frequency penalty discourages controller gain above the
        retained-mode band.
    Wn: lead cascade sensor-noise weight: wn_gain [m] at low frequency,
        rising by (wn_pole_hz/wn_zero_hz)^wn_order. Eddy-current probes
        are noisier at high frequency, and the rising Wn is what actually
        bounds the observer (hence controller) gain in the truncated-mode
        band; certified a posteriori by rs_margin.
    """
    d = params.design
    wb = 2.0 * np.pi * d.wf_bandwidth_hz
    zf = 1.0 / np.sqrt(2.0)
    beta = np.sqrt(d.wf_dc_gain * wb)     # split the gain over B and C
    Af = np.array([[-2.0 * zf * wb, wb], [-wb, 0.0]])
    Bf = np.array([[0.0], [beta]])
    Cf = np.array([[beta, 0.0]])          # -> Wf(s) = g*wb^2 / den(s)
    Df = np.zeros((1, 1))
    return {
        "Wf": (Af, Bf, Cf, Df),
        "Wu": _lead_cascade(d.wu_gain, d.wu_zero_hz, d.wu_pole_hz,
                            d.wu_order),
        "Wn": _lead_cascade(d.wn_gain, d.wn_zero_hz, d.wn_pole_hz,
                            d.wn_order),
    }


def _series(s1, s2):
    """Series composition u -> s1 -> s2 -> y of (A, B, C, D) systems."""
    A1, B1, C1, D1 = s1
    A2, B2, C2, D2 = s2
    n1, n2 = A1.shape[0], A2.shape[0]
    A = np.zeros((n1 + n2, n1 + n2))
    A[:n1, :n1] = A1
    A[n1:, :n1] = B2 @ C1
    A[n1:, n1:] = A2
    B = np.vstack([B1, B2 @ D1])
    C = np.hstack([D2 @ C1, C2])
    D = D2 @ D1
    return A, B, C, D


def _lead_cascade(dc_gain: float, z_hz: float, p_hz: float, order: int):
    """(A,B,C,D) of  dc_gain * [(1+s/wz)/(1+s/wp)]^order  (lead if p>z)."""
    wz, wp = 2.0 * np.pi * z_hz, 2.0 * np.pi * p_hz
    k = wp / wz                            # section HF gain
    sys = (np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((1, 0)),
           np.array([[dc_gain]]))
    for _ in range(int(order)):
        sec = (np.array([[-wp]]), np.array([[1.0]]),
               np.array([[k * (wz - wp)]]), np.array([[k]]))
        sys = _series(sys, sec)
    return sys


# ---------------------------------------------------------------------------
# 2. additive uncertainty weight
# ---------------------------------------------------------------------------
def _sensor_frf(mm: ModalModel, freqs_hz: np.ndarray) -> np.ndarray:
    """Voltage -> sensor-displacement FRF [m/V] by modal summation."""
    w = 2.0 * np.pi * np.asarray(freqs_hz, dtype=float)
    H = np.zeros(w.shape, dtype=complex)
    for i in range(mm.n_modes):
        H += mm.cs[i] * mm.bu[i] / (mm.omega[i] ** 2 - w ** 2
                                    + 2j * mm.zeta[i] * mm.omega[i] * w)
    return H


def _truncation_residual(params: Params, n_red: int, n_full: int,
                         theta_grid, freqs_hz: np.ndarray) -> np.ndarray:
    """max over theta of |G_full - G_red| for the u -> y_s channel [m/V].

    The channel does not depend on x_T, but the removal states in
    theta_grid = [(x_T, height_removed), ...] are all included.
    """
    removals = sorted({float(th[1]) for th in theta_grid}) or [0.0]
    E = np.zeros(len(freqs_hz))
    for hr in removals:
        mf = cached_reduce_model(params, int(n_full), hr)
        mr = cached_reduce_model(params, int(n_red), hr)
        E = np.maximum(E, np.abs(_sensor_frf(mf, freqs_hz)
                                 - _sensor_frf(mr, freqs_hz)))
    return E


def _wa_ss(r, w1, z1, w2, z2):
    """Biproper realization of r*(s^2+2 z1 w1 s+w1^2)/(s^2+2 z2 w2 s+w2^2)."""
    A = np.array([[0.0, 1.0], [-w2 ** 2, -2.0 * z2 * w2]])
    B = np.array([[0.0], [w2]])
    C = (r / w2) * np.array([[w1 ** 2 - w2 ** 2,
                              2.0 * (z1 * w1 - z2 * w2)]])
    D = np.array([[r]])
    return A, B, C, D


def _wa_mag(freqs_hz, r, w1, z1, w2, z2):
    w = 2.0 * np.pi * np.asarray(freqs_hz, dtype=float)
    num = np.abs(-w ** 2 + 2j * z1 * w1 * w + w1 ** 2)
    den = np.abs(-w ** 2 + 2j * z2 * w2 * w + w2 ** 2)
    return r * num / den


def _fit_overbound(freqs_hz: np.ndarray, E: np.ndarray):
    """Coarse grid search for (r, w1, z1, w2, z2) of the ref. Eq. (18) form
    subject to |Wa(j w)| >= E(w) on the grid; minimizes the mean log
    magnitude (least conservative overbound within the candidate set)."""
    if np.max(E) <= 0.0:
        wb = 2.0 * np.pi * float(freqs_hz[-1])
        return 1e-15, 0.5 * wb, 0.7, wb, 0.7
    w = 2.0 * np.pi * np.asarray(freqs_hz, dtype=float)
    w_pk = 2.0 * np.pi * float(freqs_hz[int(np.argmax(E))])
    lw = np.log10(w)
    best = None
    for f2 in (0.85, 0.95, 1.0, 1.05, 1.15):
        w2 = f2 * w_pk
        for z2 in (0.05, 0.1, 0.2, 0.35, 0.6):
            den = np.abs(-w ** 2 + 2j * z2 * w2 * w + w2 ** 2)
            for f1 in (0.15, 0.3, 0.5, 0.7):
                w1 = f1 * w2
                for z1 in (0.5, 0.7, 1.0):
                    num = np.abs(-w ** 2 + 2j * z1 * w1 * w + w1 ** 2)
                    S = num / den
                    r = 1.02 * float(np.max(E / S))   # 2 % safety margin
                    cost = _trapz(np.log10(r * S), lw)
                    if best is None or cost < best[0]:
                        best = (cost, r, w1, z1, w2, z2)
    return best[1:]


def additive_uncertainty_weight(params: Params, n_red: int, n_full: int,
                                theta_grid, n_freq: int = 400,
                                f_range=(50.0, 2.0e4)):
    """Overbounding 2nd-order additive uncertainty weight Wa [m/V].

    Computes max over theta of |G_full(jw) - G_red(jw)| for the
    voltage -> sensor channel from modal sums at the removal states of
    theta_grid, then fits the ref. Eq. (18) form
    r*(s^2+2 z1 w1 s+w1^2)/(s^2+2 z2 w2 s+w2^2) by coarse grid search
    subject to overbounding on a log grid 50 Hz - 20 kHz.

    Returns a state-space tuple (A, B, C, D).
    """
    freqs = np.geomspace(f_range[0], f_range[1], n_freq)
    E = _truncation_residual(params, n_red, n_full, theta_grid, freqs)
    return _wa_ss(*_fit_overbound(freqs, E))


# ---------------------------------------------------------------------------
# 3. generalized design plant
# ---------------------------------------------------------------------------
@dataclass
class GenPlant:
    """Weighted generalized plant with named channel bookkeeping.

    x = [design model (2n) | Wf states (2) | Wa states (2, optional)]
    w = [f_regen (1 N scale), noise n (unit)] (+ Delta_a output, unit)
    z = [Wf*w_mill, Wu*u] (+ [Wa*u])
    y = y_s + Wn*n (+ Delta_a output)
    """
    A: np.ndarray
    B1: np.ndarray
    B2: np.ndarray
    C1: np.ndarray
    C2: np.ndarray
    D11: np.ndarray
    D12: np.ndarray
    D21: np.ndarray
    D22: np.ndarray
    n_plant: int                      # design-model states (2n)
    sl_x: dict                        # state-block name -> slice
    iw: dict                          # exogenous input name -> column
    iz: dict                          # performance output name -> row
    c_wmill: np.ndarray               # (nx,) x -> w_mill map [m]
    meta: dict = field(default_factory=dict)

    @property
    def nx(self) -> int:
        return self.A.shape[0]

    @property
    def nw(self) -> int:
        return self.B1.shape[1]

    @property
    def nz(self) -> int:
        return self.C1.shape[0]


def pade_delay_ss(T: float):
    """(A,B,C,D) first-order Pade of a pure delay e^{-sT}:
    (1 - sT/2)/(1 + sT/2)."""
    return (np.array([[-2.0 / T]]), np.array([[2.0 / T]]),
            np.array([[2.0]]), np.array([[-1.0]]))


def rolloff_ss(params: Params):
    """Mandatory actuator roll-off filter F(s) (anti-spillover).

    rolloff_order/2 cascaded sections wc^2/(s^2 + 2 zeta wc s + wc^2);
    composed in series with every synthesized controller at deployment
    (see deploy()).
    """
    wc = 2.0 * np.pi * params.design.rolloff_hz
    zc = params.design.rolloff_zeta
    n_sec = max(1, int(params.design.rolloff_order) // 2)
    sec = (np.array([[0.0, 1.0], [-wc ** 2, -2.0 * zc * wc]]),
           np.array([[0.0], [1.0]]),
           np.array([[wc ** 2, 0.0]]),
           np.zeros((1, 1)))
    sys = (np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((1, 0)),
           np.eye(1))
    for _ in range(n_sec):
        sys = _series(sys, sec)
    return sys[0], sys[1], sys[2]


def sampled_rho(params: Params, K_dep: Controller, mm: ModalModel) -> float:
    """Exact spectral radius of the single-rate sampled closed loop.

    Structural plant (alpha4 = 0) discretized exactly (ZOH) at the
    real-time period 1/ctrl_rate_hz, deployed controller discretized
    exactly, one control period of computation delay (u applied during
    interval k is the value computed at tick k-1) - the same timing the
    simulator implements. < 1 certifies the discrete implementation.
    """
    dt_c = 1.0 / params.design.ctrl_rate_hz
    n = mm.n_modes
    A_p = np.block([[np.zeros((n, n)), np.eye(n)],
                    [-np.diag(mm.omega ** 2),
                     -np.diag(2.0 * mm.zeta * mm.omega)]])
    Bu_p = np.concatenate([np.zeros(n), mm.bu])[:, None]
    Cs_p = np.concatenate([mm.cs, np.zeros(n)])[None, :]

    def zoh(A, B):
        nn = A.shape[0]
        M = np.zeros((nn + 1, nn + 1))
        M[:nn, :nn] = A * dt_c
        M[:nn, nn] = B[:, 0] * dt_c
        E = sla.expm(M)
        return E[:nn, :nn], E[:nn, nn:nn + 1]

    Ad, Bd = zoh(A_p, Bu_p)
    Akd, Bkd = zoh(K_dep.Ak, K_dep.Bk)
    npz, nk = Ad.shape[0], Akd.shape[0]
    # states [x_p ; x_K ; u_prev]
    Az = np.zeros((npz + nk + 1, npz + nk + 1))
    Az[:npz, :npz] = Ad
    Az[:npz, -1:] = Bd
    Az[npz:npz + nk, :npz] = Bkd @ Cs_p
    Az[npz:npz + nk, npz:npz + nk] = Akd
    Az[-1, :npz] = (K_dep.Dk @ Cs_p)[0]
    Az[-1, npz:npz + nk] = K_dep.Ck[0]
    return float(np.abs(np.linalg.eigvals(Az)).max())


def reduce_controller(K: Controller, w_cap: float,
                      n_keep: int | None = None) -> Controller:
    """Residualize controller dynamics faster than |lambda| = w_cap.

    Ordered real Schur decomposition puts the slow modes in the leading
    block; the fast block (a high-gain artifact of the central DGKF
    controller, unrepresentable at the real-time rate) is replaced by its
    static (DC) contribution. Standard controller order reduction; every
    deployment certificate (rs_margin, sampled_rho, closed-loop SLD,
    time-domain simulation) is evaluated on the REDUCED controller, so
    the reduction error is covered by the verification, not assumed away.

    n_keep, when given, pins the retained order (needed for a scheduled
    grid so the deployed order is identical at every scheduling point and
    the controller state survives gain updates): the n_keep slowest modes
    are kept regardless of w_cap.
    """
    A = np.asarray(K.Ak, float)
    if n_keep is not None:
        lam = np.sort(np.abs(np.linalg.eigvals(A)))
        if n_keep >= len(lam):
            return K
        w_cap = 0.5 * (lam[n_keep - 1] + lam[n_keep])
    T, Z, sdim = sla.schur(A, output="real",
                           sort=lambda x, y: (x * x + y * y) < w_cap ** 2)
    n1 = int(sdim)
    if n1 == K.nk:
        return K
    if n1 == 0:
        raise RuntimeError("controller has no dynamics below w_cap")
    Bt = Z.T @ K.Bk
    Ct = K.Ck @ Z
    A11, A12 = T[:n1, :n1], T[:n1, n1:]
    A22 = T[n1:, n1:]
    B1s, B2s = Bt[:n1], Bt[n1:]
    C1s, C2s = Ct[:, :n1], Ct[:, n1:]
    A22iB2 = np.linalg.solve(A22, B2s)
    Ak = A11
    Bk = B1s - A12 @ A22iB2
    Ck = C1s
    Dk = K.Dk - C2s @ A22iB2
    cw = None
    if K.c_wmill is not None:
        cw = (K.c_wmill @ Z)[:n1]        # static u-feedthrough part dropped
    gd0 = None if K.gd0 is None else (K.gd0 @ Z)[:n1]
    gd1 = None if K.gd1 is None else (K.gd1 @ Z)[:n1]
    return Controller(Ak=Ak, Bk=Bk, Ck=Ck, Dk=Dk, gd0=gd0, gd1=gd1,
                      c_wmill=cw, gamma=K.gamma, label=K.label,
                      meta={**K.meta, "reduced_from": K.nk,
                            "w_cap": float(w_cap)})


def deploy(K: Controller, params: Params,
           latency: bool = False, n_keep: int | None = None) -> Controller:
    """Compose the actuator roll-off filter in series AFTER the raw
    synthesized controller: the deployable hardware controller is
    u = F(s) * K(s) * y_s. Design tools (close_loop, tzw_norm, the grid
    validation) work with the RAW controller against plants that already
    contain the filter; SLD and time-domain simulation receive the
    deployed controller (their plant models do not contain the filter).

    latency=True additionally composes the first-order Pade model of the
    real-time implementation latency (the same block the synthesis saw):
    use it for CONTINUOUS-time closed-loop analysis (SLD), so the
    analyzed loop matches the designed one. The simulator implements the
    sampling and computation delay physically and must receive the
    latency=False controller.

    Delayed taps (gd0/gd1), when later attached via
    with_delayed_feedback, act on the composed state and add to the
    output voltage directly (narrowband low-frequency action; certified
    as-implemented by the closed-loop semi-discretization lobes).
    """
    # residualize central-controller dynamics that the real-time target
    # cannot represent (cap at 0.3 * ctrl_rate_hz)
    w_cap = 2.0 * np.pi * 0.3 * params.design.ctrl_rate_hz
    K = reduce_controller(K, w_cap, n_keep=n_keep)
    chain = rolloff_ss(params) + (np.zeros((1, 1)),)
    if latency:
        T_lat = params.design.latency_factor / params.design.ctrl_rate_hz
        chain = _series(chain, pade_delay_ss(T_lat))
    Afl, Bfl, Cfl, Dfl = chain
    nk, nf = K.nk, Afl.shape[0]
    Ak = np.zeros((nk + nf, nk + nf))
    Ak[:nk, :nk] = K.Ak
    Ak[nk:, :nk] = Bfl @ K.Ck
    Ak[nk:, nk:] = Afl
    Bk = np.vstack([K.Bk, Bfl @ K.Dk])
    Ck = np.hstack([Dfl @ K.Ck, Cfl])
    Dk = Dfl @ K.Dk
    cw = None if K.c_wmill is None else np.concatenate(
        [K.c_wmill, np.zeros(nf)])
    # delayed state taps survive composition (zero-extended over the
    # filter states; taps add to the output voltage directly)
    gd0 = None if K.gd0 is None else np.concatenate([K.gd0, np.zeros(nf)])
    gd1 = None if K.gd1 is None else np.concatenate([K.gd1, np.zeros(nf)])
    return Controller(Ak=Ak, Bk=Bk, Ck=Ck, Dk=Dk, gd0=gd0, gd1=gd1,
                      c_wmill=cw, gamma=K.gamma, label=K.label,
                      meta={**K.meta, "deployed": True,
                            "latency": bool(latency)})


def build_design_plant(mm3: ModalModel, x_T: float, ap_design: float,
                       weights: dict, Wa=None) -> GenPlant:
    """Assemble the weighted generalized plant at frozen (x_T, removal).

    The design model is mm3.abcd(x_T, alpha4=mean_alpha4(params, ap_design))
    (signed alpha4, negative for down-milling), driven through the
    mandatory actuator roll-off filter F(s) (rolloff_ss). The
    regenerative-residual exogenous force enters through the Bf column
    with scale 1.0 N (Wf does the performance shaping). D11 = 0; D12 (Wu
    direct term) has full column rank; D21 (Wn) has full row rank.
    """
    p = mm3.params
    a4 = mean_alpha4(p, ap_design)
    # abcd's alpha4 argument is the coefficient of the CURRENT-time force
    # term Fy_now = alpha4*w(t); the physical regenerative force
    # Fy = a4*(w_tau - w) has current-time part -a4*w(t), so the design
    # plant must be assembled with alpha4 = -a4 (softening for the signed
    # down-milling a4 < 0), consistent with docs section 2.5 and with the
    # SLD/simulation layers. meta records the physical signed a4.
    Ap, Bu, Bff, Cs, Cw = mm3.abcd(x_T, alpha4=-a4)
    # actuator chain seen by the synthesis: implementation-latency Pade
    # (sampled-data model, NOT deployed) -> roll-off filter (deployed)
    T_lat = p.design.latency_factor / p.design.ctrl_rate_hz
    Ar, Br, Cr = rolloff_ss(p)
    Afl, Bfl, Cfl, _Dfl = _series(pade_delay_ss(T_lat),
                                  (Ar, Br, Cr, np.zeros((1, 1))))
    n_p = Ap.shape[0]
    nfl = Afl.shape[0]
    Af, Bfw, Cf, _Df = weights["Wf"]
    Au, Buw, Cu, Du = weights["Wu"]
    An, Bn, Cn, Dn = weights["Wn"]
    nf = Af.shape[0]
    nu_s = Au.shape[0]
    nn_s = An.shape[0]
    na = 0 if Wa is None else Wa[0].shape[0]
    o_fl = n_p                               # offset of filter block
    o_wf = o_fl + nfl                        # offset of Wf block
    o_wu = o_wf + nf                         # offset of Wu block
    o_wn = o_wu + nu_s                       # offset of Wn block
    o_wa = o_wn + nn_s                       # offset of Wa block
    nx = o_wa + na
    nw = 3 if Wa is not None else 2
    nz = 3 if Wa is not None else 2

    A = np.zeros((nx, nx))
    A[:n_p, :n_p] = Ap
    A[:n_p, o_fl:o_wf] = Bu @ Cfl            # plant driven by filter output
    A[o_fl:o_wf, o_fl:o_wf] = Afl
    A[o_wf:o_wu, :n_p] = Bfw @ Cw
    A[o_wf:o_wu, o_wf:o_wu] = Af
    B1 = np.zeros((nx, nw))
    B1[:n_p, 0] = Bff[:, 0] * 1.0            # 1 N regen-residual scale
    B2 = np.zeros((nx, 1))
    B2[o_fl:o_wf, 0] = Bfl[:, 0]             # u -> filter input
    C1 = np.zeros((nz, nx))
    C1[0, o_wf:o_wu] = Cf[0]
    C2 = np.zeros((1, nx))
    C2[0, :n_p] = Cs[0]
    D11 = np.zeros((nz, nw))
    D12 = np.zeros((nz, 1))
    D12[1, 0] = float(Du[0, 0])
    D21 = np.zeros((1, nw))
    D21[0, 1] = float(Dn[0, 0])
    D22 = np.zeros((1, 1))
    if nu_s:                                 # dynamic Wu (lead cascade)
        A[o_wu:o_wn, o_wu:o_wn] = Au
        B2[o_wu:o_wn, 0] += Buw[:, 0]
        C1[1, o_wu:o_wn] = Cu[0]
    if nn_s:                                 # dynamic Wn (lead cascade)
        A[o_wn:o_wa, o_wn:o_wa] = An
        B1[o_wn:o_wa, 1] = Bn[:, 0]          # noise input drives Wn
        C2[0, o_wn:o_wa] = Cn[0]             # y = Cs x + Wn(s) n

    sl_x = {"plant": slice(0, n_p), "filt": slice(o_fl, o_wf),
            "wf": slice(o_wf, o_wu), "wu": slice(o_wu, o_wn),
            "wn": slice(o_wn, o_wa)}
    iw = {"f_regen": 0, "noise": 1}
    iz = {"wmill": 0, "u": 1}
    if Wa is not None:
        Aa, Ba, Ca, Da = Wa
        A[o_wa:, o_wa:] = Aa
        B2[o_wa:, 0] += Ba[:, 0]
        C1[2, o_wa:] = Ca[0]
        D12[2, 0] = Da[0, 0]
        D21[0, 2] = 1.0                      # Delta_a output enters y
        sl_x["wa"] = slice(o_wa, nx)
        iw["delta_a"] = 2
        iz["wa"] = 2

    c_wmill = np.zeros(nx)
    c_wmill[:n_p] = Cw[0]                    # [bf(x_T)^T, 0] on plant block
    return GenPlant(A=A, B1=B1, B2=B2, C1=C1, C2=C2, D11=D11, D12=D12,
                    D21=D21, D22=D22, n_plant=n_p, sl_x=sl_x, iw=iw, iz=iz,
                    c_wmill=c_wmill,
                    meta={"x_T": x_T, "ap_design": ap_design, "alpha4": a4,
                          "height_removed": mm3.height_removed})


def close_loop(plant: GenPlant, K: Controller):
    """(Acl, Bcl, Ccl, Dcl) of the w -> z closed loop (D22 = 0 assumed;
    delayed taps gd0/gd1, if any, are NOT included - frozen LTI part only)."""
    if np.any(plant.D22):
        raise ValueError("close_loop assumes D22 = 0")
    A, B1, B2 = plant.A, plant.B1, plant.B2
    C1, C2 = plant.C1, plant.C2
    D11, D12, D21 = plant.D11, plant.D12, plant.D21
    Acl = np.block([[A + B2 @ K.Dk @ C2, B2 @ K.Ck],
                    [K.Bk @ C2, K.Ak]])
    Bcl = np.vstack([B1 + B2 @ K.Dk @ D21, K.Bk @ D21])
    Ccl = np.hstack([C1 + D12 @ K.Dk @ C2, D12 @ K.Ck])
    Dcl = D11 + D12 @ K.Dk @ D21
    return Acl, Bcl, Ccl, Dcl


def tzw_norm(plant: GenPlant, K: Controller, freqs_hz) -> float:
    """Grid H-infinity norm ||Tzw||_inf of the closed weighted plant."""
    return _frf_maxsv(*close_loop(plant, K), freqs_hz)


# ---------------------------------------------------------------------------
# 4. H-infinity synthesis (DGKF central controller, gamma bisection)
# ---------------------------------------------------------------------------
_F_CHECK = (10.0, 5.0e4)
_N_FREQ_CHECK = 150


def _dgkf_central(plant: GenPlant, g: float, freqs_hz=None):
    """DGKF central controller at fixed level gamma = g, or None.

    Feasibility: both game Riccati solutions exist and are PSD,
    rho(X Y) < g^2, the closed-loop interconnection is Hurwitz, and the
    grid norm of Tzw does not exceed g. Any numerical failure counts as
    infeasible (returns None).
    """
    A, B1, B2 = plant.A, plant.B1, plant.B2
    C1, C2 = plant.C1, plant.C2
    D11, D12, D21 = plant.D11, plant.D12, plant.D21
    n, nw = A.shape[0], B1.shape[1]
    nz, ny, nu = C1.shape[0], C2.shape[0], B2.shape[1]
    if np.any(D11) or np.any(plant.D22):
        raise ValueError("DGKF synthesis assumes D11 = 0 and D22 = 0")
    if np.linalg.matrix_rank(D12) < nu or np.linalg.matrix_rank(D21) < ny:
        raise ValueError("D12 / D21 rank conditions violated")
    if freqs_hz is None:
        freqs_hz = np.geomspace(_F_CHECK[0], _F_CHECK[1], _N_FREQ_CHECK)

    # gamma-invariant channel normalization (ARE conditioning)
    su = np.linalg.norm(D12, 2)
    sy = np.linalg.norm(D21, 2)
    B2n, D12n = B2 / su, D12 / su
    C2n, D21n = C2 / sy, D21 / sy
    R2 = D12n.T @ D12n
    S2 = D21n @ D21n.T

    try:
        X = _care(A, np.hstack([B1, B2n]), C1.T @ C1,
                  sla.block_diag(-g ** 2 * np.eye(nw), R2),
                  s=np.hstack([np.zeros((n, nw)), C1.T @ D12n]))
        Y = _care(A.T, np.hstack([C1.T, C2n.T]), B1 @ B1.T,
                  sla.block_diag(-g ** 2 * np.eye(nz), S2),
                  s=np.hstack([np.zeros((n, nz)), B1 @ D21n.T]))
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y))):
        return None
    if np.linalg.eigvalsh(X).min() < -1e-8 * max(1.0, np.abs(X).max()):
        return None
    if np.linalg.eigvalsh(Y).min() < -1e-8 * max(1.0, np.abs(Y).max()):
        return None
    if np.abs(np.linalg.eigvals(X @ Y)).max() >= g ** 2 * (1.0 - 1e-9):
        return None
    try:
        F = -np.linalg.solve(R2, B2n.T @ X + D12n.T @ C1)
        L = -np.linalg.solve(S2.T, (Y @ C2n.T + B1 @ D21n.T).T).T
        Z = np.linalg.solve(np.eye(n) - (Y @ X) / g ** 2, np.eye(n))
    except np.linalg.LinAlgError:
        return None
    Ak = (A + (B1 @ (B1.T @ X)) / g ** 2 + B2n @ F
          + Z @ L @ (C2n + (D21n @ (B1.T @ X)) / g ** 2))
    K = Controller(Ak=Ak, Bk=(-Z @ L) / sy, Ck=F / su,
                   Dk=np.zeros((nu, ny)),
                   c_wmill=plant.c_wmill.copy(), gamma=g, label="hinf",
                   meta=dict(plant.meta))
    Acl, Bcl, Ccl, Dcl = close_loop(plant, K)
    if spectral_abscissa(Acl) >= 0.0:
        return None
    if _frf_maxsv(Acl, Bcl, Ccl, Dcl, freqs_hz) > g:
        return None
    return K


def hinf_synthesis(plant: GenPlant, gamma_max: float = 1e6,
                   tol: float = 0.01, max_bisect: int = 40,
                   backoff: float = 1.3) -> Controller:
    """Standard two-Riccati (DGKF) H-inf controller with bounded gamma
    bisection (see _dgkf_central for the per-gamma feasibility checks).

    The returned controller is the central controller RE-SOLVED at
    backoff * gamma_opt: at the feasibility edge the game Riccati
    solutions blow up and the central controller develops pathological
    high-gain fast poles (observed: a 69 kHz pole with near-70 % FRF
    residue) that no real-time target can implement; a 1.3x back-off
    yields a benign controller at almost no performance cost (verified:
    closed-loop critical depths change < 2 % for back-off 1.3-3.0).
    Controller.gamma is the norm bound actually certified for the
    returned controller (backoff * gamma_opt); meta['gamma_opt'] records
    the bisection optimum. Controller.c_wmill is the [bf^T, 0] map on the
    design-model block of the (observer-structured) controller state.
    """
    freqs = np.geomspace(_F_CHECK[0], _F_CHECK[1], _N_FREQ_CHECK)
    # find a feasible upper bound
    K_best, g_hi = None, 1.0
    while g_hi <= gamma_max:
        K = _dgkf_central(plant, g_hi, freqs)
        if K is not None:
            K_best = K
            break
        g_hi *= 4.0
    if K_best is None:
        raise RuntimeError(f"hinf_synthesis: no feasible gamma <= {gamma_max:g}")
    # bisection (bounded)
    g_lo = 0.0 if g_hi <= 1.0 else g_hi / 4.0
    for _ in range(max_bisect):
        if (g_hi - g_lo) <= tol * g_hi:
            break
        g_mid = 0.5 * (g_lo + g_hi)
        K = _dgkf_central(plant, g_mid, freqs)
        if K is None:
            g_lo = g_mid
        else:
            K_best, g_hi = K, g_mid
    g_opt = g_hi
    if backoff > 1.0:
        K = _dgkf_central(plant, backoff * g_opt, freqs)
        if K is not None:                # monotone feasibility: expected
            K_best = K
            K_best.gamma = backoff * g_opt
        else:                            # extremely unlikely; keep optimum
            K_best.gamma = g_opt
    else:
        K_best.gamma = g_opt
    K_best.meta["gamma"] = K_best.gamma
    K_best.meta["gamma_opt"] = g_opt
    K_best.meta["backoff"] = backoff
    return K_best


# ---------------------------------------------------------------------------
# 5. LQG baseline
# ---------------------------------------------------------------------------
def lqg_design(mm3: ModalModel, x_T: float, ap_design: float,
               q_w: float, r_u: float, w_noise: float,
               v_noise: float) -> Controller:
    """LQR on the w_mill cost + Kalman filter (observer structure).

    Cost: integral of q_w*w_mill^2 [1/m^2] + r_u*u^2 [1/V^2]; process
    noise of intensity w_noise [N^2] through the milling-force column,
    measurement noise v_noise [m^2].
    """
    a4 = mean_alpha4(mm3.params, ap_design)
    # current-time regenerative part enters with alpha4 = -a4 (see
    # build_design_plant)
    Ap0, Bu0, Bff0, Cs0, Cw0 = mm3.abcd(x_T, alpha4=-a4)
    # include the same actuator chain as the H-infinity design plants:
    # implementation-latency Pade + mandatory roll-off filter
    T_lat = mm3.params.design.latency_factor / mm3.params.design.ctrl_rate_hz
    Ar, Br, Cr = rolloff_ss(mm3.params)
    Afl, Bfl, Cfl, _D = _series(pade_delay_ss(T_lat),
                                (Ar, Br, Cr, np.zeros((1, 1))))
    n_p, nfl = Ap0.shape[0], Afl.shape[0]
    Ap = np.block([[Ap0, Bu0 @ Cfl], [np.zeros((nfl, n_p)), Afl]])
    Bu = np.vstack([np.zeros((n_p, 1)), Bfl])
    Bff = np.vstack([Bff0, np.zeros((nfl, 1))])
    Cs = np.hstack([Cs0, np.zeros((1, nfl))])
    Cw = np.hstack([Cw0, np.zeros((1, nfl))])
    X = sla.solve_continuous_are(Ap, Bu, q_w * (Cw.T @ Cw)
                                 + 1e-12 * np.eye(Ap.shape[0]),
                                 np.array([[r_u]]))
    F = (Bu.T @ X) / r_u                     # u = -F x
    Y = sla.solve_continuous_are(Ap.T, Cs.T, w_noise * (Bff @ Bff.T)
                                 + 1e-12 * np.eye(Ap.shape[0]),
                                 np.array([[v_noise]]))
    L = (Y @ Cs.T) / v_noise                 # (2n+nfl, 1)
    return Controller(Ak=Ap - Bu @ F - L @ Cs, Bk=L, Ck=-F,
                      Dk=np.zeros((1, 1)), c_wmill=Cw[0].copy(),
                      label="lqg",
                      meta={"x_T": x_T, "ap_design": ap_design, "alpha4": a4,
                            "q_w": q_w, "r_u": r_u, "w_noise": w_noise,
                            "v_noise": v_noise})


# ---------------------------------------------------------------------------
# 6. delayed-PD baseline (single active time-delay control, ref. Eq. (30))
# ---------------------------------------------------------------------------
def delayed_pd_controller(kp: float, kd: float,
                          f_filt_hz: float = 8000.0) -> Controller:
    """Delayed PD on the sensor signal through a critically damped 2-state
    low-pass: x1 = filtered y_s [m], x2 = filtered dy_s/dt [m/s];
    u(t) = kp*x1(t-tau) + kd*x2(t-tau)  (gd1 = [kp, kd] [V/m, V s/m],
    gd0 = 0, Ck = Dk = 0)."""
    wf = 2.0 * np.pi * f_filt_hz
    return Controller(Ak=np.array([[0.0, 1.0], [-wf * wf, -2.0 * wf]]),
                      Bk=np.array([[0.0], [wf * wf]]),
                      Ck=np.zeros((1, 2)), Dk=np.zeros((1, 1)),
                      gd0=np.zeros(2), gd1=np.array([kp, kd], dtype=float),
                      label="dpd",
                      meta={"kp": kp, "kd": kd, "f_filt_hz": f_filt_hz})


# ---------------------------------------------------------------------------
# 7. position-scheduled gain grid
# ---------------------------------------------------------------------------
def _bracket(grid: np.ndarray, v: float):
    """Piecewise-linear bracket: (index i, weight t) with v between
    grid[i] and grid[i+1]; clipped to the grid range."""
    if len(grid) == 1:
        return 0, 0.0
    v = float(np.clip(v, grid[0], grid[-1]))
    i = int(np.clip(np.searchsorted(grid, v, side="right") - 1,
                    0, len(grid) - 2))
    return i, (v - grid[i]) / (grid[i + 1] - grid[i])


@dataclass
class ScheduledController:
    """Grid of DGKF controllers, all assembled in the SAME state basis
    (design-model modal states + weight states), with piecewise-linear
    (bilinear) interpolation of (Ak, Bk, Ck, Dk, c_wmill) over
    theta = (x_T, height_removed)."""
    params: Params
    x_grid: np.ndarray
    removal_grid: np.ndarray
    ap_design: float
    n_modes: int
    controllers: list                 # controllers[i_x][i_removal]
    label: str = "pslpv"

    @property
    def gammas(self) -> np.ndarray:
        return np.array([[k.gamma for k in row] for row in self.controllers])

    def deployed_order(self) -> int:
        """Common retained order for deployment: the minimum slow-mode
        count across the grid at the real-time pole cap, so the deployed
        controller order (hence its state) is identical at every
        scheduling point."""
        if not hasattr(self, "_n_keep"):
            w_cap = 2.0 * np.pi * 0.3 * self.params.design.ctrl_rate_hz
            n_min = min(
                int(np.sum(np.abs(np.linalg.eigvals(k.Ak)) < w_cap))
                for row in self.controllers for k in row)
            object.__setattr__(self, "_n_keep", max(n_min, 2))
        return self._n_keep

    def at_theta(self, x_T: float, removal: float,
                 deployed: bool = True, latency: bool = False) -> Controller:
        """Interpolated controller at (x_T, height_removed).

        deployed=True (default) composes the actuator roll-off filter for
        use in SLD / time-domain simulation; deployed=False returns the
        raw controller for evaluation against design plants that already
        contain the filter.
        """
        i, tx = _bracket(self.x_grid, x_T)
        j, tr = _bracket(self.removal_grid, removal)
        corners = [(self.controllers[i][j], (1 - tx) * (1 - tr))]
        if tx > 0.0:
            corners.append((self.controllers[i + 1][j], tx * (1 - tr)))
        if tr > 0.0:
            corners.append((self.controllers[i][j + 1], (1 - tx) * tr))
        if tx > 0.0 and tr > 0.0:
            corners.append((self.controllers[i + 1][j + 1], tx * tr))

        def mix(attr):
            return sum(wgt * getattr(k, attr) for k, wgt in corners)

        K = Controller(Ak=mix("Ak"), Bk=mix("Bk"), Ck=mix("Ck"),
                       Dk=mix("Dk"), c_wmill=mix("c_wmill"),
                       gamma=max(k.gamma for k, _ in corners),
                       label=self.label,
                       meta={"x_T": x_T, "removal": removal})
        if not deployed:
            return K
        return deploy(K, self.params, latency=latency,
                      n_keep=self.deployed_order())

    def validate_midpoints(self) -> float:
        """Close every interpolated mid-grid controller on its frozen
        weighted design plant; return the max spectral abscissa [1/s]
        (< 0 certifies the interpolation on the midpoint set)."""
        weights = make_weights(self.params)
        xm = 0.5 * (self.x_grid[1:] + self.x_grid[:-1])
        rm = 0.5 * (self.removal_grid[1:] + self.removal_grid[:-1])
        pts = [(x, r) for x in xm for r in self.removal_grid]
        pts += [(x, r) for x in self.x_grid for r in rm]
        pts += [(x, r) for x in xm for r in rm]
        if not pts:
            return -np.inf
        worst = -np.inf
        for x, r in pts:
            mm = cached_reduce_model(self.params, self.n_modes, float(r))
            plant = build_design_plant(mm, float(x), self.ap_design, weights)
            Acl = close_loop(plant, self.at_theta(float(x), float(r),
                                                  deployed=False))[0]
            worst = max(worst, spectral_abscissa(Acl))
        return worst


def gain_schedule(params: Params, x_grid, removal_grid, ap_design: float,
                  n_modes: int = 3,
                  gamma_margin: float = 1.10) -> ScheduledController:
    """Grid-based gain-scheduled DGKF synthesis.

    Pass 1 re-solves the two Riccati equations (gamma bisection) at every
    grid point; pass 2 re-assembles every central controller at the COMMON
    level gamma_c = gamma_margin * max gamma(theta) in the SAME state basis
    (design-model + weight states). The common gamma makes the entrywise
    interpolation of at_theta well-posed: the grid controllers vary
    smoothly with theta instead of jumping with the local optimal gamma.
    Interpolated midpoints must still be certified via validate_midpoints.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    removal_grid = np.asarray(removal_grid, dtype=float)
    weights = make_weights(params)
    plants, ctrls = [], []
    for x in x_grid:
        prow, row = [], []
        for hr in removal_grid:
            mm = cached_reduce_model(params, int(n_modes), float(hr))
            plant = build_design_plant(mm, float(x), ap_design, weights)
            K = hinf_synthesis(plant)
            prow.append(plant)
            row.append(K)
        plants.append(prow)
        ctrls.append(row)
    g_common = gamma_margin * max(k.gamma for row in ctrls for k in row)
    for i, prow in enumerate(plants):
        for j, plant in enumerate(prow):
            K = _dgkf_central(plant, g_common)
            if K is not None:               # feasible by construction
                ctrls[i][j] = K
            ctrls[i][j].label = "pslpv-grid"
            ctrls[i][j].meta["gamma_common"] = g_common
    return ScheduledController(params=params, x_grid=x_grid,
                               removal_grid=removal_grid,
                               ap_design=ap_design, n_modes=int(n_modes),
                               controllers=ctrls)


# ---------------------------------------------------------------------------
# 8. robust point design (R-HINF baseline)
# ---------------------------------------------------------------------------
def robust_point_design(params: Params, x_grid, ap_design: float,
                        removal_grid=(0.0,), n_modes: int | None = None,
                        n_full: int | None = None, n_freq: int = 400,
                        f_range=(50.0, 2.0e4), gamma_max: float = 1e6,
                        tol: float = 0.01) -> Controller:
    """Single H-inf design at the path midpoint whose Wa is inflated to
    additionally overbound max_theta |G_red(theta) - G_red(mid)| on the
    u -> y_s channel: position/removal variation treated as norm-bounded
    uncertainty (the conservatism of the mu-synthesis reference approach).
    """
    d = params.design
    n_modes = d.n_modes_design if n_modes is None else int(n_modes)
    n_full = d.n_modes_full if n_full is None else int(n_full)
    x_grid = np.asarray(x_grid, dtype=float)
    removal_grid = np.asarray(removal_grid, dtype=float)
    x_mid = 0.5 * (x_grid.min() + x_grid.max())
    hr_nom = float(removal_grid[len(removal_grid) // 2])
    theta = [(float(x), float(r)) for x in x_grid for r in removal_grid]

    freqs = np.geomspace(f_range[0], f_range[1], n_freq)
    E = _truncation_residual(params, n_modes, n_full, theta, freqs)
    a4 = mean_alpha4(params, ap_design)
    mm_nom = cached_reduce_model(params, n_modes, hr_nom)
    Ap, Bu, _bf, Cs, _cw = mm_nom.abcd(x_mid, alpha4=-a4)
    G_mid = _siso_frf(Ap, Bu, Cs, freqs)
    E_pos = np.zeros_like(E)
    for x, hr in theta:
        mm = cached_reduce_model(params, n_modes, hr)
        Ax, Bx, _b, Cx, _c = mm.abcd(x, alpha4=-a4)
        E_pos = np.maximum(E_pos, np.abs(_siso_frf(Ax, Bx, Cx, freqs) - G_mid))

    Wa = _wa_ss(*_fit_overbound(freqs, E + E_pos))
    weights = make_weights(params)
    plant = build_design_plant(mm_nom, x_mid, ap_design, weights, Wa=Wa)
    K = hinf_synthesis(plant, gamma_max=gamma_max, tol=tol)
    K.label = "rhinf"
    K.meta.update({"x_mid": x_mid, "hr_nom": hr_nom,
                   "x_grid": x_grid.tolist(),
                   "removal_grid": removal_grid.tolist()})
    return K


# ---------------------------------------------------------------------------
# 9. spillover certificate and best-frozen baseline
# ---------------------------------------------------------------------------
def _ctrl_frf(K: Controller, freqs_hz) -> np.ndarray:
    """Controller y_s -> u FRF (LTI part only, delayed taps excluded)."""
    return _siso_frf(K.Ak, K.Bk, K.Ck, freqs_hz, D=K.Dk)


def rs_margin(params: Params, K: Controller, Wa, theta_grid,
              n_modes: int | None = None, n_freq: int = 600,
              f_range=(50.0, 2.0e4)) -> float:
    """Small-gain spillover certificate.

    Returns  max_theta  || Wa * K / (1 + G_red(theta) K) ||_inf  computed on
    a frequency grid; < 1 certifies robust stability against every additive
    perturbation bounded by the fitted truncation weight Wa (structural
    u -> y_s channel, alpha4 = 0). Delayed taps are excluded from K here;
    the full delayed loop is certified separately by the closed-loop
    semi-discretization lobes on the full-order model.
    """
    d = params.design
    n_modes = d.n_modes_design if n_modes is None else int(n_modes)
    freqs = np.geomspace(f_range[0], f_range[1], n_freq)
    Kf = _ctrl_frf(K, freqs)
    Wm = np.abs(_siso_frf(Wa[0], Wa[1], Wa[2], freqs, D=Wa[3]))
    worst = 0.0
    for hr in sorted({float(th[1]) for th in theta_grid}):
        mm = cached_reduce_model(params, n_modes, hr)
        G = _sensor_frf(mm, freqs)
        # loop convention is u = +K y everywhere in this codebase, so an
        # additive plant perturbation sees K/(1 - G K), NOT K/(1 + G K)
        worst = max(worst, float(np.max(np.abs(Wm * Kf / (1.0 - G * Kf)))))
    return worst


def best_frozen_design(params: Params, x_grid, removal_grid, ap_design: float,
                       n_modes: int = 3, n_freq: int = 300) -> Controller:
    """Best single frozen H-infinity controller over the scheduling grid.

    The strongest fair non-scheduled baseline from the same design family:
    synthesize a DGKF controller at every grid point with the same weights
    as the scheduled design, evaluate every candidate's worst-case weighted
    norm max_theta ||Tzw(theta)||_inf across ALL grid plants (discarding
    candidates that destabilize any plant), and return the min-max winner.
    The advantage of PS-LPV over this baseline is therefore attributable to
    scheduling alone.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    removal_grid = np.asarray(removal_grid, dtype=float)
    weights = make_weights(params)
    freqs = np.geomspace(*_F_CHECK, n_freq)
    plants, cands = [], []
    for x in x_grid:
        for hr in removal_grid:
            mm = cached_reduce_model(params, int(n_modes), float(hr))
            plant = build_design_plant(mm, float(x), ap_design, weights)
            plants.append(plant)
            cands.append(hinf_synthesis(plant))
    best, best_wc = None, np.inf
    for K in cands:
        wc = 0.0
        for plant in plants:
            Acl = close_loop(plant, K)[0]
            if spectral_abscissa(Acl) >= 0.0:
                wc = np.inf
                break
            wc = max(wc, tzw_norm(plant, K, freqs))
        if wc < best_wc:
            best, best_wc = K, wc
    if best is None:
        raise RuntimeError("no frozen candidate stabilizes the whole grid")
    best.label = "frozen-hinf"
    best.meta.update({"worst_case_gamma": float(best_wc),
                      "design_point": {"x_T": best.meta.get("x_T"),
                                       "height_removed":
                                       best.meta.get("height_removed")},
                      "x_grid": x_grid.tolist(),
                      "removal_grid": removal_grid.tolist()})
    return best        # RAW: callers deploy(K, params[, latency=...])
