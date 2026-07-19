"""Closed-loop time-domain simulation of the plate-actuator milling system.

Physics: modal evaluation model (high order), exact ZOH propagation per
mode at the physics substep rate (default 50 kHz); regenerative milling
force evaluated at every substep with the delayed cutting-point deflection
from a ring buffer.  Control: zero-order-hold at the controller rate
(10 kHz) with actuator saturation and a one-sample computation delay, and
a discrete Butterworth low-pass on the acceleration measurement, matching
the hardware chain of the reference experiment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import scipy.signal as sig

from .modal import ModalModel
from .milling import MillingForce
from .params import SystemParams


def zoh_blocks(omega: np.ndarray, zeta: np.ndarray, dt: float):
    """Exact ZOH propagation coefficients for each modal 2nd-order system."""
    wd = omega * np.sqrt(1.0 - zeta ** 2)
    E = np.exp(-zeta * omega * dt)
    c, s = np.cos(wd * dt), np.sin(wd * dt)
    A11 = E * (c + zeta * omega / wd * s)
    A12 = E * s / wd
    A21 = -omega ** 2 * E * s / wd
    A22 = E * (c - zeta * omega / wd * s)
    B1 = (1.0 - A11) / omega ** 2
    B2 = -A21 / omega ** 2
    return A11, A12, A21, A22, B1, B2


@dataclass
class SimResult:
    t: np.ndarray                # control-rate time stamps
    x_cut: np.ndarray            # cutter position [m]
    w_cut: np.ndarray            # deflection at the moving cutting point [m]
    w_points: np.ndarray         # (n_pts, N) deflection at fixed monitor points
    acc: np.ndarray              # filtered accelerometer signal [m/s^2]
    V: np.ndarray                # applied actuator voltage [V]
    F: np.ndarray                # plate-normal milling force [N]
    surf_t: np.ndarray = field(default=None)   # surface-generation instants
    surf_x: np.ndarray = field(default=None)   # cutter position at those instants
    surf_w: np.ndarray = field(default=None)   # deflection imprinted on the surface
    meta: dict = field(default_factory=dict)


class Saturation:
    def __init__(self, lo: float, hi: float):
        self.lo, self.hi = lo, hi

    def __call__(self, v: float) -> float:
        return min(max(v, self.lo), self.hi)


def simulate(sys: SystemParams, model: ModalModel, controller=None,
             T: float | None = None, seed: int = 0,
             regeneration: bool = True,
             monitor_points: np.ndarray | None = None,
             freq_scale_end: np.ndarray | None = None,
             force_scale: float = 1.0) -> SimResult:
    """Run one closed-loop (or open-loop if controller is None) pass.

    freq_scale_end : optional per-mode frequency scale factor reached at the
        end of the pass (linear drift), representing in-process material
        removal or model perturbations of the *plant*.
    """
    rng = np.random.default_rng(seed)
    ctl, mil, sen = sys.control, sys.milling, sys.sensor
    Ts = ctl.Ts
    sub = sys.config.phys_substeps
    dtp = Ts / sub
    T = mil.pass_time + 1.0 if T is None else T
    N = int(round(T / Ts))

    n = model.n
    omega0 = model.omega.copy()
    zeta = model.zeta
    ba = model.ba
    phi_acc = model.phi_acc

    drift = freq_scale_end is not None
    if not drift:
        A11, A12, A21, A22, B1, B2 = zoh_blocks(omega0, zeta, dtp)

    force = MillingForce(mil)
    sat = Saturation(sys.piezo.V_min, sys.piezo.V_max)

    # anti-aliasing measurement filter: analog Butterworth (charge-amplifier
    # chain), simulated at the physics substep rate BEFORE the 10 kHz
    # sampling, so above-Nyquist plate modes alias only as far as the
    # analog filter lets them through — as in the real hardware.
    ba_f, aa_f = sig.butter(sen.lp_order, 2.0 * np.pi * sen.lp_cutoff,
                            btype="low", analog=True)
    bz, az = sig.bilinear(ba_f, aa_f, fs=1.0 / dtp)
    bz = bz / az[0]
    az = az / az[0]
    fb0, fb1, fb2 = bz[0], bz[1], bz[2]
    fa1, fa2 = az[1], az[2]
    fs1 = fs2 = 0.0   # direct-form-II-transposed biquad states

    # regeneration ring buffer at substep rate (covers the multi-delay
    # surface memory up to max_miss tooth periods)
    tau_steps = mil.tau / dtp
    max_miss = 8
    buf_len = int(np.ceil(tau_steps * (max_miss + 1))) + 2
    wbuf = np.zeros(buf_len)
    bptr = 0
    force.reset_regen(max_miss)

    if monitor_points is None:
        monitor_points = np.linspace(mil.x_start, mil.x_end, 11)
    Phi_mon = model.phi_at_many(monitor_points)      # (n_pts, n)

    eta = np.zeros(n)
    etad = np.zeros(n)

    # delayed command queue (computation delay)
    if ctl.delay_steps < 1:
        raise ValueError("delay_steps must be >= 1: the loop structure "
                         "senses at the interval start, so a zero-delay "
                         "controller cannot be represented")
    cmd_queue = [0.0] * ctl.delay_steps

    t_out = np.arange(N) * Ts
    x_out = np.empty(N)
    wc_out = np.empty(N)
    wp_out = np.empty((len(monitor_points), N))
    acc_out = np.empty(N)
    V_out = np.empty(N)
    F_out = np.empty(N)
    surf_t, surf_x, surf_w = [], [], []

    # surface generation: tooth crossing phi = 0 (mod 2 pi) for up-milling
    def surface_crossings(t0, t1):
        m = mil
        out = []
        # angles of all teeth pass 0 mod 2pi at times (2 pi k - j*pitch)/Omega
        k0 = int(np.floor(m.omega_spindle * t0 / m.phi_pitch)) - 1
        k1 = int(np.ceil(m.omega_spindle * t1 / m.phi_pitch)) + 1
        for k in range(k0, k1 + 1):
            tc = k * m.phi_pitch / m.omega_spindle
            if t0 <= tc < t1:
                out.append(tc)
        return out

    noise_sd = sen.noise_rms

    V_apply = 0.0
    for k in range(N):
        tk = k * Ts
        # apply the (delayed, saturated) command over this interval
        V_apply = sat(cmd_queue.pop(0))
        xc = mil.cutter_x(tk)
        in_pass = 0.0 <= tk <= mil.pass_time
        phi_c = model.phi_at(xc)

        if drift:
            s = np.clip(tk / mil.pass_time, 0.0, 1.0)
            om = omega0 * (1.0 + s * (freq_scale_end - 1.0))
            A11, A12, A21, A22, B1, B2 = zoh_blocks(om, zeta, dtp)
        else:
            om = omega0

        Fk = 0.0
        y = 0.0
        for ss in range(sub):
            t = tk + ss * dtp
            w_c = float(phi_c @ eta)
            if regeneration:
                # surface-memory lookup: w at time t - m*tau, linearly
                # interpolated in the ring buffer; the most recent written
                # sample (bptr - 1) holds w at time t
                def w_lookup(mm: int) -> float:
                    idx = bptr - 1 - mm * tau_steps
                    i0 = int(np.floor(idx)) % buf_len
                    i1 = (i0 + 1) % buf_len
                    frac = idx - np.floor(idx)
                    return (1.0 - frac) * wbuf[i0] + frac * wbuf[i1]
                F = (force.normal_force_regen(t, w_c, w_lookup)
                     * force_scale if in_pass else 0.0)
            else:
                F = (force.normal_force(t, 0.0, 0.0) * force_scale
                     if in_pass else 0.0)

            # sensing BEFORE propagation: acceleration at time t from the
            # state at time t and the inputs active on [t, t+dt); the
            # analog anti-aliasing filter advances at the substep rate
            etadd = (-om ** 2 * eta - 2.0 * zeta * om * etad
                     + ba * V_apply + phi_c * F)
            acc_true = float(phi_acc @ etadd)
            acc_f = fb0 * acc_true + fs1
            fs1 = fb1 * acc_true - fa1 * acc_f + fs2
            fs2 = fb2 * acc_true - fa2 * acc_f

            if ss == 0:
                Fk = F
                # sample at the control rate, add sensor noise, run control
                y = acc_f + rng.normal(0.0, noise_sd)
                x_out[k] = xc
                wc_out[k] = w_c
                wp_out[:, k] = Phi_mon @ eta
                acc_out[k] = y
                V_out[k] = V_apply
                F_out[k] = Fk
                cmd = (controller.update(k, tk, y, xc)
                       if controller is not None else 0.0)
                cmd_queue.append(cmd)

            u_modal = ba * V_apply + phi_c * F
            e0, ed0 = eta, etad
            eta = A11 * e0 + A12 * ed0 + B1 * u_modal
            etad = A21 * e0 + A22 * ed0 + B2 * u_modal
            wbuf[bptr % buf_len] = float(phi_c @ eta)
            bptr += 1

        if in_pass:
            # surface imprint at the tooth-crossing instant tc: interpolate
            # the substep-resolution deflection history (wbuf holds w at
            # every substep; the last write corresponds to time tk + Ts)
            t_end = tk + Ts
            for tc in surface_crossings(tk, tk + Ts):
                idx = bptr - 1 - (t_end - tc) / dtp
                i0 = int(np.floor(idx)) % buf_len
                i1 = (i0 + 1) % buf_len
                frac = idx - np.floor(idx)
                surf_t.append(tc)
                surf_x.append(mil.cutter_x(tc))
                surf_w.append((1.0 - frac) * wbuf[i0] + frac * wbuf[i1])

    return SimResult(t=t_out, x_cut=x_out, w_cut=wc_out, w_points=wp_out,
                     acc=acc_out, V=V_out, F=F_out,
                     surf_t=np.array(surf_t), surf_x=np.array(surf_x),
                     surf_w=np.array(surf_w),
                     meta={"monitor_points": monitor_points})
