"""Time-domain simulation of regenerative thin-wall milling.

Implements the coupled tool-workpiece model of Sec. 2:

* Eq. (1):  Ft,j = Kt h_j(phi_j) ap            (plus Fr,j = Kr Ft,j)
* Eq. (2):  h_j = ft sin phi + (x(t)-x(t-tau)) sin phi
                             + (y(t)-y(t-tau)) cos phi
* Eqs. (3)-(4): summation over engaged teeth with window g(phi_j)
* Eq. (5):  modal equations of motion (symplectic-Euler integrated)
* Eqs. (6)-(7): modal parameters updated continuously with the removed
  volume fraction Vr(t) (the Eq. (15) stiffness-update concept collapsed
  onto the modal description, as the paper's "efficient modal parameter
  updating without complete re-analysis")
* Eq. (13): process damping force proportional to Kp V0 / V

The regenerative delay tau = 60 / (N n) is realised with a full
displacement history buffer and linear interpolation, which remains exact
under the spindle-speed variation commanded by the controllers.

Nonlinearities: the tooth leaves the cut when h_j < 0 (jump-out) and the
engagement window g(phi_j) switches teeth in and out of contact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .parameters import PaperSetup

try:                                     # optional JIT acceleration
    from numba import njit
    NUMBA = True
except ImportError:                      # pragma: no cover
    NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def deco(f):
            return f
        return deco


@njit(cache=True)
def _integrate_interval(i0, i1, dt,
                        omega, ft, tau, ap,
                        phi0,
                        phi_st, phi_ex, Kt, Kr, n_teeth,
                        cp_damping,
                        runout,           # per-tooth radial offsets [m]
                        mass, cc, kk, is_x, is_tool,
                        q, qd,
                        xh, yh,           # relative-displacement history
                        id_buf, id_dec,   # workpiece-y samples for RLS
                        f_buf, f_dec):    # Fy samples for spectra
    """Advance the coupled system from step i0 to i1 (exclusive).

    Returns (phi_end, rms_yrel, mean_fx, mean_fy, mean_ft_sum, max_f,
             rms_ywp, n_id, n_f).
    """
    n_modes = mass.shape[0]
    delay = tau / dt
    two_pi = 2.0 * math.pi
    pitch = two_pi / n_teeth

    sum_y2 = 0.0
    sum_yw2 = 0.0
    sum_fx = 0.0
    sum_fy = 0.0
    sum_ftan = 0.0
    max_f = 0.0
    n_id = 0
    n_f = 0
    phi = phi0

    for i in range(i0, i1):
        # relative tool-workpiece displacement from the modal states
        x_rel = 0.0
        y_rel = 0.0
        y_wp = 0.0
        yd_rel = 0.0
        for m in range(n_modes):
            if is_x[m] == 1:
                x_rel += q[m]            # tool x (wall rigid in x)
            else:
                if is_tool[m] == 1:
                    y_rel += q[m]
                    yd_rel += qd[m]
                else:
                    y_rel -= q[m]        # wall motion reduces relative y
                    yd_rel -= qd[m]
                    y_wp += q[m]
        xh[i] = x_rel
        yh[i] = y_rel

        # delayed surface position (previous tooth pass), Eq. (2)
        jf = i - delay
        if jf <= 0.0:
            x_d = 0.0
            y_d = 0.0
        else:
            j = int(jf)
            frac = jf - j
            x_d = xh[j] * (1.0 - frac) + xh[j + 1] * frac
            y_d = yh[j] * (1.0 - frac) + yh[j + 1] * frac

        fx = 0.0
        fy = 0.0
        ftan_sum = 0.0
        engaged = 0
        for jt in range(n_teeth):
            pj = (phi + pitch * jt) % two_pi
            if phi_st <= pj <= phi_ex:   # window function g(phi_j)
                s = math.sin(pj)
                c = math.cos(pj)
                h = (ft * s
                     + (x_rel - x_d) * s
                     + (y_rel - y_d) * c
                     + runout[jt])
                if h > 0.0:
                    ftj = Kt * h * ap            # Eq. (1)
                    frj = Kr * ftj
                    fx += -ftj * c - frj * s     # Eq. (3)
                    fy += ftj * s - frj * c      # Eq. (4)
                    ftan_sum += ftj
                    engaged += 1

        # process damping, Eq. (13): opposes relative y velocity while cutting
        if engaged > 0:
            fy += -cp_damping * yd_rel

        # modal integration (symplectic Euler), Eq. (5)
        for m in range(n_modes):
            if is_x[m] == 1:
                fm = fx
            else:
                fm = fy if is_tool[m] == 1 else -fy
            acc = (fm - cc[m] * qd[m] - kk[m] * q[m]) / mass[m]
            qd[m] += acc * dt
            q[m] += qd[m] * dt

        phi += omega * dt

        sum_y2 += y_rel * y_rel
        sum_yw2 += y_wp * y_wp
        sum_fx += fx
        sum_fy += fy
        sum_ftan += ftan_sum
        fmag = math.sqrt(fx * fx + fy * fy)
        if fmag > max_f:
            max_f = fmag

        if i % id_dec == 0 and n_id < id_buf.shape[0]:
            id_buf[n_id] = y_wp
            n_id += 1
        if i % f_dec == 0 and n_f < f_buf.shape[0]:
            f_buf[n_f] = fy
            n_f += 1

    n_steps = i1 - i0
    rms_y = math.sqrt(sum_y2 / n_steps)
    rms_yw = math.sqrt(sum_yw2 / n_steps)
    return (phi % two_pi, rms_y, sum_fx / n_steps, sum_fy / n_steps,
            sum_ftan / n_steps, max_f, rms_yw, n_id, n_f)


@dataclass
class IntervalMeasurement:
    """Sensor picture handed to a controller once per control period."""
    t: float
    rms_disp_um: float          # relative vibration RMS (accelerometer-derived)
    rms_wall_um: float          # wall-mounted sensor RMS
    mean_fx_N: float
    mean_fy_N: float
    mean_ft_N: float            # summed tangential force (for power)
    max_f_N: float
    id_samples: np.ndarray      # wall displacement @ 10 kHz (noisy)
    force_samples: np.ndarray   # Fy @ 5 kHz
    Vr: float                   # removed-volume fraction (ground truth)
    mrr_cm3_min: float
    n_rpm: float
    vf_mm_min: float
    ap_mm: float
    ft_mm: float                # feed per tooth


class MillingSimulator:
    """Owns the physical state; controllers only see IntervalMeasurement."""

    def __init__(self, setup: PaperSetup, seed: int = 0,
                 batch_scatter: float = 0.02):
        self.setup = setup
        self.rng = np.random.default_rng(seed)
        sim = setup.sim
        self.dt = sim.dt
        self.n_total = sim.n_steps
        self.ctrl_dt = 1.0 / setup.controller.update_rate_hz
        self.steps_per_ctrl = int(round(self.ctrl_dt / self.dt))

        # workpiece-to-workpiece scatter ("same material batch", Sec. 4.1)
        self.modes = []
        for m in setup.modes:
            mm = type(m)(**{**m.__dict__})
            mm.f0_hz *= 1.0 + batch_scatter * self.rng.standard_normal() * 0.5
            mm.zeta0 *= 1.0 + batch_scatter * self.rng.standard_normal()
            self.modes.append(mm)

        n_modes = len(self.modes)
        self.mass = np.array([m.mass_kg for m in self.modes])
        self.cc = np.zeros(n_modes)
        self.kk = np.zeros(n_modes)
        self.is_x = np.array(
            [1 if m.direction == "x" else 0 for m in self.modes],
            dtype=np.int64)
        self.is_tool = np.array(
            [1 if m.structure == "tool" else 0 for m in self.modes],
            dtype=np.int64)
        self.q = np.zeros(n_modes)
        self.qd = np.zeros(n_modes)
        self.xh = np.zeros(self.n_total + 2)
        self.yh = np.zeros(self.n_total + 2)

        # tool runout: ~5 um radial, random phase per run
        r0 = 5e-6 * (0.5 + self.rng.random())
        ph = 2.0 * math.pi * self.rng.random()
        nt = setup.tool.n_teeth
        radial = np.array([r0 * math.cos(2.0 * math.pi * j / nt + ph)
                           for j in range(nt)])
        # chip-thickness perturbation = own offset minus preceding tooth's
        self.runout = radial - np.roll(radial, 1)

        self.id_dec = max(1, int(round(1.0 / (sim.accel_sample_hz * self.dt))))
        self.f_dec = max(1, int(round(1.0 / (sim.force_sample_hz * self.dt))))

        self.i = 0
        self.phi = 0.0
        self.Vr = 0.0
        self.t = 0.0
        self._update_modal_arrays()

    # -- ground-truth structural evolution, Eqs. (6)-(7) ------------------
    def _update_modal_arrays(self) -> None:
        for m_idx, m in enumerate(self.modes):
            self.kk[m_idx] = m.k(self.Vr)
            self.cc[m_idx] = m.c(self.Vr)

    def current_modes(self):
        """Snapshot of the true instantaneous modal parameters."""
        out = []
        for m in self.modes:
            out.append((m.wn(self.Vr) / (2 * math.pi), m.zeta(self.Vr)))
        return out

    @property
    def finished(self) -> bool:
        # a pass ends when the programmed stock is removed (Vr = 1) or at
        # the hard time ceiling
        return (self.Vr >= 1.0
                or self.i + self.steps_per_ctrl > self.n_total)

    def advance(self, n_rpm: float, vf_mm_min: float,
                ap_mm: float) -> IntervalMeasurement:
        """Run one control period with the commanded parameters."""
        setup = self.setup
        tool = setup.tool
        omega = 2.0 * math.pi * n_rpm / 60.0
        tau = 60.0 / (n_rpm * tool.n_teeth)
        vf_m_s = vf_mm_min / 1000.0 / 60.0
        ft_m = vf_m_s / (tool.n_teeth * n_rpm / 60.0)
        ap_m = ap_mm / 1000.0

        # process damping coefficient, Eq. (13), scaled with axial depth
        Vc = math.pi * tool.diameter_m * n_rpm / 60.0
        cp = tool.Kp * tool.V0 / max(Vc, 0.5) * (ap_m / 0.004)

        i1 = self.i + self.steps_per_ctrl
        n_id_max = self.steps_per_ctrl // self.id_dec + 2
        n_f_max = self.steps_per_ctrl // self.f_dec + 2
        id_buf = np.zeros(n_id_max)
        f_buf = np.zeros(n_f_max)

        (self.phi, rms_y, mfx, mfy, mftan, maxf, rms_yw, n_id, n_f) = \
            _integrate_interval(
                self.i, i1, self.dt,
                omega, ft_m, tau, ap_m, self.phi,
                *setup.engagement.entry_exit_angles(tool),
                tool.Kt, tool.Kr, tool.n_teeth,
                cp, self.runout,
                self.mass, self.cc, self.kk, self.is_x, self.is_tool,
                self.q, self.qd, self.xh, self.yh,
                id_buf, self.id_dec, f_buf, self.f_dec)

        self.i = i1
        self.t = self.i * self.dt

        # material removal bookkeeping -> Vr -> modal update (Eqs. (6)-(7))
        mrr_m3_s = vf_m_s * ap_m * setup.engagement.ae_m
        mrr_cm3_min = mrr_m3_s * 1e6 * 60.0
        self.Vr = min(1.0, self.Vr + mrr_m3_s * self.ctrl_dt
                      / (setup.sim.removal_volume_cm3 * 1e-6))
        self._update_modal_arrays()

        noise = setup.sim.sensor_noise_um * 1e-6
        id_samples = (id_buf[:n_id]
                      + noise * self.rng.standard_normal(n_id))

        return IntervalMeasurement(
            t=self.t,
            rms_disp_um=rms_y * 1e6,
            rms_wall_um=rms_yw * 1e6,
            mean_fx_N=mfx, mean_fy_N=mfy, mean_ft_N=mftan, max_f_N=maxf,
            id_samples=id_samples,
            force_samples=f_buf[:n_f].copy(),
            Vr=self.Vr,
            mrr_cm3_min=mrr_cm3_min,
            n_rpm=n_rpm, vf_mm_min=vf_mm_min, ap_mm=ap_mm,
            ft_mm=ft_m * 1000.0)
