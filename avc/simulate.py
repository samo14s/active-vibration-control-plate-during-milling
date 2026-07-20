"""Nonlinear time-domain simulation of the plate-milling closed loop.

Fixed-step LTV engine used as the EVALUATION model of the paper: full
helical multi-tooth engagement geometry with per-slice unilateral
(loss-of-contact) chip nonlinearity, moving tool position, sampled-data
controller (ZOH at fs/5, like the real NI target) with the
spindle-synchronized delayed state tap of avc.controller, measurement
noise, voltage saturation, and optional gain scheduling.

Plant (avc.modal.ModalModel, mass-normalized modal coordinates eta):

    eta_dd + diag(2 zeta_i w_i) eta_d + diag(w_i^2) eta
        = bu u(t) + bf(x_T(t)) Fy(t)
    w_mill = bf(x_T)^T eta,     y_s = cs^T eta + noise

Milling force, per tooth j and axial slice k (helix lag, down-milling
engagement window; geometry identical to avc.milling.alpha4_time):

    phi_jk(t) = Omega t + 2 pi j / N - z_k tan(helix) / R
    h_jk      = sin(phi_jk) * [ ft + w(t) - w(t - tau) ]   (chip thickness)
    dFy_jk    = -( kt cos(phi_jk) + kr kt sin(phi_jk) ) * h_jk * dz

summed over engaged slices; with loss_of_contact=True a slice with
h_jk <= 0 (tooth jumped out of the cut) produces zero force. When no
slice loses contact the sum reduces EXACTLY (same axial slices) to the
signed linear evaluation model of avc.milling,

    Fy(t) = a4(t) [ w(t-tau) - w(t) ] - a4(t) ft,     a4(t) <= 0,

which is the model avc.sld validates against the analytic turning
benchmark. The dynamic chip-thickness sign is +w(t) - w(t-tau): the
current deflection thickens the chip, the previously machined surface
thins it, and with this orientation the elemental force reproduces the
operative linear model exactly (consistent with the avc.milling
docstring).

Integration: the free-plate modal form is diagonal, so the exact ZOH
propagator over one step dt is n independent 2x2 blocks (precomputed
once via 3x3 matrix exponentials, force held constant over the step) -
fast and unconditionally stable. Regenerative delay: circular history of
the milling-point deflection at fs with linear interpolation. Controller
delayed tap: second history of the scalar gd1 @ xk at the controller
rate, linear interpolation. All SI units.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm
from scipy.signal import welch

from .controller import Controller
from .milling import engagement_angles
from .params import Params

CTRL_DIV = 5           # legacy default divisor (used when ctrl_rate_hz absent)
SCHED_PERIOD = 0.02    # gain-scheduling update interval [s]
ETA_STRIDE = 20        # modal-state subsampling stride [physics steps]
_W_RUNAWAY = 1e150     # [m] numerical-runaway guard (truncate the run)


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------
@dataclass
class SimResult:
    """Time histories of one simulated milling pass (all SI units).

    t : (ns,) time [s]; w_mill : milling-point deflection [m];
    y_s : noisy sensor output [m]; u : applied (saturated) voltage [V];
    x_tool : tool position [m]; eta : (ns//ETA_STRIDE, n) subsampled
    modal displacements [m kg^0.5] (times in extras['t_eta']);
    extras : run metadata (fs, fs_ctrl, tau, sat_frac, ...).
    """

    t: np.ndarray
    w_mill: np.ndarray
    y_s: np.ndarray
    u: np.ndarray
    x_tool: np.ndarray
    eta: np.ndarray | None = None
    extras: dict = field(default_factory=dict)

    # ---- scalar metrics -------------------------------------------------
    def _sel(self, x: np.ndarray, t_from: float) -> np.ndarray:
        return x[self.t >= t_from]

    def rms_w(self, t_from: float = 0.0) -> float:
        """RMS milling-point deflection [m] for t >= t_from."""
        return float(np.sqrt(np.mean(self._sel(self.w_mill, t_from) ** 2)))

    def peak_w(self, t_from: float = 0.0) -> float:
        """Peak |w_mill| [m] for t >= t_from."""
        return float(np.max(np.abs(self._sel(self.w_mill, t_from))))

    def rms_u(self, t_from: float = 0.0) -> float:
        """RMS actuator voltage [V] for t >= t_from."""
        return float(np.sqrt(np.mean(self._sel(self.u, t_from) ** 2)))

    def peak_u(self, t_from: float = 0.0) -> float:
        """Peak |u| [V] for t >= t_from."""
        return float(np.max(np.abs(self._sel(self.u, t_from))))

    def psd_w(self, nperseg: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """Welch PSD of w_mill: (freqs [Hz], PSD [m^2/Hz])."""
        nper = int(min(nperseg, len(self.w_mill)))
        return welch(self.w_mill, fs=self.extras["fs"], nperseg=nper)


# ---------------------------------------------------------------------------
# milling force: helical per-slice engagement with loss of contact
# ---------------------------------------------------------------------------
class CutGeometry:
    """Per-slice helical engagement geometry, precomputed once per run.

    Slice angular offsets and axial discretization are identical to
    avc.milling.alpha4_time(..., n_axial), so with loss_of_contact off
    force(t, w, w_tau) == a4(t) (w_tau - w) - a4(t) ft exactly (same
    slice quadrature of a4).
    """

    def __init__(self, params: Params, ap: float, rpm: float,
                 n_axial: int = 16):
        tool = params.tool
        z = (np.arange(n_axial) + 0.5) * (ap / n_axial)
        lag = z * np.tan(np.deg2rad(tool.helix_deg)) / tool.radius
        j = 2.0 * np.pi * np.arange(tool.n_teeth) / tool.n_teeth
        self.offs = (j[:, None] - lag[None, :]).ravel()   # (N*n_axial,)
        self.dz = ap / n_axial
        self.Omega = 2.0 * np.pi * rpm / 60.0
        self.kt = tool.kt
        self.kr = tool.kn
        self.phi_in, self.phi_out = engagement_angles(params)
        self.ft = params.process.ft

    def force(self, t: float, w: float, w_tau: float,
              loss_of_contact: bool = True) -> float:
        """Total y-force on the plate [N] at time t.

        w, w_tau: milling-point deflection now and one tooth period ago
        [m]. h_jk = sin(phi) (ft + w - w_tau); engaged slices with
        h_jk <= 0 produce zero force when loss_of_contact is True.
        """
        phi = np.mod(self.Omega * t + self.offs, 2.0 * np.pi)
        eng = (phi >= self.phi_in) & (phi <= self.phi_out)
        if not eng.any():
            return 0.0
        phi = phi[eng]
        s = np.sin(phi)
        h = s * (self.ft + w - w_tau)
        if loss_of_contact:
            h = np.maximum(h, 0.0)
        return float(-self.dz * np.sum(
            (self.kt * np.cos(phi) + self.kr * self.kt * s) * h))


# ---------------------------------------------------------------------------
# discretization helpers
# ---------------------------------------------------------------------------
def _modal_zoh(omega: np.ndarray, zeta: np.ndarray, dt: float):
    """Exact per-mode ZOH propagator of the free plate over one step.

    Mode i: [eta, eta_d]' = [[0,1],[-w^2,-2 z w]] [eta, eta_d] + [0,1] f,
    f held constant over dt. Returns (a11,a12,a21,a22,b1,b2), each (n,):
      eta+   = a11 eta + a12 eta_d + b1 f
      eta_d+ = a21 eta + a22 eta_d + b2 f
    """
    n = len(omega)
    coef = np.zeros((6, n))
    for i in range(n):
        M = np.zeros((3, 3))
        M[0, 1] = 1.0
        M[1, 0] = -omega[i] ** 2
        M[1, 1] = -2.0 * zeta[i] * omega[i]
        M[1, 2] = 1.0
        E = expm(M * dt)
        coef[:, i] = (E[0, 0], E[0, 1], E[1, 0], E[1, 1], E[0, 2], E[1, 2])
    return coef


def _ctrl_discretize(ctrl: Controller, dt_c: float) -> dict:
    """ZOH discretization + gain extraction of a Controller at dt_c."""
    Ak = np.atleast_2d(np.asarray(ctrl.Ak, float))
    nk = Ak.shape[0]
    Bk = np.asarray(ctrl.Bk, float).reshape(nk)
    M = np.zeros((nk + 1, nk + 1))
    M[:nk, :nk] = Ak * dt_c
    M[:nk, nk] = Bk * dt_c
    E = expm(M)
    d = {
        "nk": nk,
        "Akd": E[:nk, :nk],
        "Bkd": E[:nk, nk],
        "Ck": np.asarray(ctrl.Ck, float).reshape(nk),
        "Dk": float(np.asarray(ctrl.Dk, float).ravel()[0]),
        "gd0": None if ctrl.gd0 is None
        else np.asarray(ctrl.gd0, float).reshape(nk),
        "gd1": None if ctrl.gd1 is None
        else np.asarray(ctrl.gd1, float).reshape(nk),
    }
    return d


def _delayed(hist: np.ndarray, pos: float) -> float:
    """Linear interpolation of a history array at fractional index pos.

    hist[k] holds the sample at index k; pos < 0 (before the cut
    started: undeflected surface / controller at rest) returns 0.
    """
    if pos < 0.0:
        return 0.0
    i0 = int(pos)
    fr = pos - i0
    return (1.0 - fr) * hist[i0] + fr * hist[i0 + 1]


# ---------------------------------------------------------------------------
# main engine
# ---------------------------------------------------------------------------
def simulate(mm, rpm: float, ap: float, controller: Controller | None = None,
             T: float | None = None, x0: float = 0.005,
             x1: float | None = None, moving: bool = True, fs: float = 100e3,
             noise_rms: float | None = None, seed: int = 0,
             loss_of_contact: bool = True, v_max: float = 150.0,
             scheduled=None, n_axial: int = 40) -> SimResult:
    """Simulate a milling pass of the full nonlinear LTV closed loop.

    Parameters
    ----------
    mm : avc.modal.ModalModel - evaluation plant (n modes as given).
    rpm : spindle speed [rev/min];  ap : axial depth of cut [m].
    controller : avc.controller.Controller or None (open loop). Run ZOH-
        discretized at the real-time target rate params.design.ctrl_rate_hz
        (fs must be an integer multiple), with ONE control period of
        computation delay (matching the latency model used in synthesis),
        delayed state tap included, output saturated to +-v_max before
        application to the plant.
    T : simulated time [s]. Default (None): time for the tool to feed
        from x0 to x1 (moving=True required).
    x0, x1 : start / end tool position [m] (x1 default: 0.995*lp).
    moving : if True, x_tool = x0 + feed_speed*t (bf(x_tool) refreshed
        every control period from a precomputed fine position grid).
    fs : physics sample rate [Hz] (fixed step dt = 1/fs).
    noise_rms : sensor noise RMS [m]; None -> params.sensor.noise_rms;
        0 disables noise.
    seed : RNG seed for the noise realization.
    loss_of_contact : per-slice unilateral chip nonlinearity on/off.
    v_max : voltage saturation [V].
    scheduled : optional gain schedule with .at_theta(x_T, removal) ->
        Controller; when given it overrides `controller` and the gains
        are re-extracted and re-discretized every SCHED_PERIOD (20 ms)
        at the current tool position (controller state is kept).
    n_axial : axial slices for the engagement integration.

    Returns
    -------
    SimResult with histories at fs and metadata in .extras.
    """
    p: Params = mm.params
    dt = 1.0 / fs
    tau = p.tooth_passing_delay(rpm)
    if tau <= 3.0 * dt:
        raise ValueError("fs too low to resolve the regenerative delay")

    feed = p.feed_speed(rpm) if moving else 0.0
    lp = p.plate.lp
    x_end = float(x1) if x1 is not None else 0.995 * lp
    if moving and x_end <= x0:
        raise ValueError("x1 must exceed x0 for a moving pass")
    if T is None:
        if not moving:
            raise ValueError("T must be given when moving=False")
        T = (x_end - x0) / feed
    n_steps = int(round(T * fs))
    if n_steps < 2:
        raise ValueError("simulation horizon shorter than 2 steps")

    if noise_rms is None:
        noise_rms = p.sensor.noise_rms
    removal = getattr(mm, "height_removed", 0.0)
    n = mm.n_modes

    # ---- plant propagator and I/O vectors -------------------------------
    a11, a12, a21, a22, b1, b2 = _modal_zoh(mm.omega, mm.zeta, dt)
    bu = np.asarray(mm.bu, float)
    cs = np.asarray(mm.cs, float)

    # ---- bf(x_tool) cache on a fine position grid -----------------------
    if moving:
        x_lo, x_hi = x0, min(x_end, 0.9999 * lp)
        ng = max(16, int(np.ceil((x_hi - x_lo) / 0.25e-3)) + 1)
        xg = np.linspace(x_lo, x_hi, ng)
        Bg = np.array([mm.bf(x) for x in xg])          # (ng, n)

        def bf_at(x: float) -> np.ndarray:
            i = int(np.clip(np.searchsorted(xg, x), 1, ng - 1))
            fr = (x - xg[i - 1]) / (xg[i] - xg[i - 1])
            fr = min(max(fr, 0.0), 1.0)
            return (1.0 - fr) * Bg[i - 1] + fr * Bg[i]

        bf = bf_at(x0)
    else:
        bf = np.asarray(mm.bf(x0), float)

    # ---- controller setup ----------------------------------------------
    rate = getattr(p.design, "ctrl_rate_hz", None)
    ctrl_div = max(1, int(round(fs / rate))) if rate else CTRL_DIV
    if rate and abs(ctrl_div * rate - fs) > 1e-6 * fs:
        raise ValueError(
            f"fs={fs:g} must be an integer multiple of the real-time "
            f"controller rate ctrl_rate_hz={rate:g}")
    dt_c = ctrl_div * dt
    ctrl = scheduled.at_theta(float(x0), removal) if scheduled is not None \
        else controller
    have_ctrl = ctrl is not None
    if have_ctrl:
        cd = _ctrl_discretize(ctrl, dt_c)
        xk = np.zeros(cd["nk"])
    n_ctrl = n_steps // ctrl_div + 1
    s_hist = np.zeros(n_ctrl)          # delayed-tap scalars gd1 @ xk
    Dc = tau / dt_c                    # tap delay in controller steps

    # ---- histories and buffers ------------------------------------------
    t = np.arange(n_steps) * dt
    w_hist = np.zeros(n_steps)         # milling-point deflection = record
    ys = np.zeros(n_steps)
    uu = np.zeros(n_steps)
    xt = np.zeros(n_steps)
    n_eta = (n_steps - 1) // ETA_STRIDE + 1
    eta_hist = np.zeros((n_eta, n))

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n_steps) * noise_rms if noise_rms > 0.0 \
        else np.zeros(n_steps)

    geom = CutGeometry(p, ap, rpm, n_axial=n_axial)
    D = tau * fs                       # regenerative delay in physics steps
    sched_steps = max(1, int(round(SCHED_PERIOD * fs)))

    eta = np.zeros(n)
    etad = np.zeros(n)
    u_hold = 0.0
    u_next = 0.0                       # pending output (computation delay)
    jc = 0                             # controller tick counter
    n_sat = 0
    diverged_at = None

    for k in range(n_steps):
        tk = t[k]
        x_now = min(x0 + feed * tk, x_end) if moving else x0
        xt[k] = x_now

        if moving and k % ctrl_div == 0:
            bf = bf_at(x_now)
        if scheduled is not None and k > 0 and k % sched_steps == 0:
            ctrl = scheduled.at_theta(x_now, removal)
            new = _ctrl_discretize(ctrl, dt_c)
            if new["nk"] != cd["nk"]:
                xk = np.zeros(new["nk"])   # defensive: structure changed
            cd = new

        w_k = float(bf @ eta)
        if not np.isfinite(w_k) or abs(w_k) > _W_RUNAWAY:
            diverged_at = tk
            n_steps = k
            break
        w_hist[k] = w_k
        y_k = float(cs @ eta) + noise[k]
        ys[k] = y_k

        if have_ctrl and k % ctrl_div == 0:
            u_hold = u_next               # one control period of latency
            u_raw = cd["Ck"] @ xk + cd["Dk"] * y_k
            if cd["gd0"] is not None:
                u_raw += cd["gd0"] @ xk
            if cd["gd1"] is not None:
                s_hist[jc] = cd["gd1"] @ xk
                u_raw += _delayed(s_hist, jc - Dc)
            xk = cd["Akd"] @ xk + cd["Bkd"] * y_k
            if abs(u_raw) > v_max:
                n_sat += 1
                u_raw = np.sign(u_raw) * v_max
            u_next = float(u_raw)
            jc += 1
        uu[k] = u_hold

        w_tau = _delayed(w_hist, k - D)
        Fy = geom.force(tk, w_k, w_tau, loss_of_contact)
        fm = bu * u_hold + bf * Fy
        eta, etad = (a11 * eta + a12 * etad + b1 * fm,
                     a21 * eta + a22 * etad + b2 * fm)
        if k % ETA_STRIDE == 0:
            eta_hist[k // ETA_STRIDE] = eta

    if diverged_at is not None:        # truncate to the computed samples
        sl = slice(0, n_steps)
        t, w_hist, ys, uu, xt = (t[sl], w_hist[sl], ys[sl], uu[sl], xt[sl])
        eta_hist = eta_hist[: (max(n_steps - 1, 0)) // ETA_STRIDE + 1]

    extras = {
        "fs": fs, "fs_ctrl": fs / ctrl_div, "tau": tau, "rpm": rpm,
        "ap": ap, "feed": feed, "removal": removal, "seed": seed,
        "noise_rms": noise_rms, "n_axial": n_axial, "v_max": v_max,
        "eta_stride": ETA_STRIDE, "t_eta": t[::ETA_STRIDE].copy(),
        "sat_frac": n_sat / max(jc, 1), "diverged_at": diverged_at,
        "label": (ctrl.label if have_ctrl else "OL"),
        "loss_of_contact": loss_of_contact,
    }
    return SimResult(t=t, w_mill=w_hist, y_s=ys, u=uu, x_tool=xt,
                     eta=eta_hist, extras=extras)
