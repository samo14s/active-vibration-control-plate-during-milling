"""
baseline_controllers.py
=======================
Established active-damping laws, implemented as fair benchmarks.

WHY THIS MODULE EXISTS
----------------------
A new controller for piezo-actuated vibration control is judged against the
laws the field already uses. Comparing only against LQG, and only at whatever
tuning the author happened to pick, does not establish anything. This module
provides the standard baselines with a COMMON tuning procedure: each law is
tuned by scalar search to minimise the same cost under the same control-effort
budget, so the comparison isolates the control law rather than the tuning
effort spent on it.

All controllers expose the interface the Newmark simulator expects:

    step(x_state_prev, u_prev, y_meas) -> (x_state, u)

`x_state` is whatever internal state the law carries, stored by the simulator
and handed back on the next call, so no controller keeps hidden state that
would break a restart.

LAWS PROVIDED
-------------
DirectVelocityFeedback
    u = -g * ydot, with ydot from a washout (high-pass) differentiator since
    only displacement is measured. Unconditionally stabilising for a
    collocated, dual sensor/actuator pair; the pair here is NOT collocated
    (patch at the clamped edge, sensor at the free corner), so stability is
    a property to be checked, not assumed -- which is exactly why the
    certified closed-loop lobe matters.

PositivePositionFeedback
    Second-order filter targeted at one mode:
        eta_ddot + 2 zf wf eta_dot + wf^2 eta = wf^2 y
        u = g * eta
    Standard for piezo plate control; rolls off above the targeted mode, so
    it does not spill over into high-frequency dynamics the way DVF can.

RepetitiveController
    u_rc[k] = Q * u_rc[k-N] + L * e[k-N+phase_lead]
    with N the tooth-passing period in samples and Q a zero-phase low-pass
    that trades asymptotic rejection for robustness. This is the ESTABLISHED
    method for rejecting a periodic disturbance of known period, and is the
    correct benchmark for any "phase-indexed" or "phase-aware" feedforward
    scheme: such schemes learn a periodic function of tooth-passing phase,
    which is precisely what repetitive control realises with a convergence
    proof attached.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def _plant_dc_gain(plate):
    """
    Sign and magnitude of the static gain from actuator volts to sensor
    displacement, y_dc = -(D_obs^T Kp^{-1} H_Pe) * u.

    Computed rather than assumed: the modal eigenvector signs returned by an
    eigensolver are arbitrary, so a hard-coded feedback sign is a latent
    positive-feedback bug.
    """
    return float(plate.D_obs @ np.linalg.solve(plate.Kp, plate.H_Pe_modal))


class DirectVelocityFeedback:
    """
    u = -g * dy/dt, derivative realised by a washout filter

        H(s) = s / (s/wc + 1)

    so that sensor noise is not differentiated without bound. State is the
    single filter state.
    """

    name = "DVF"

    def __init__(self, plate, dt, gain, wc=None, u_max=150.0):
        self.dt = dt
        self.u_max = u_max
        self.gain = gain
        # default corner an octave above the highest retained mode
        self.wc = wc if wc is not None else 2.0 * float(np.max(plate.omega_n))
        self.sign = np.sign(_plant_dc_gain(plate)) or 1.0
        # xf' = -wc xf + wc y ;  ydot ~ wc (y - xf)
        ad = float(np.exp(-self.wc * dt))
        self.ad = ad
        self.bd = 1.0 - ad
        self.n_state = 1

    def step(self, x_prev, u_prev, y_meas):
        xf = float(np.atleast_1d(x_prev)[0])
        ydot = self.wc * (y_meas - xf)
        xf_new = self.ad * xf + self.bd * y_meas
        u = -self.sign * self.gain * ydot
        u = float(np.clip(u, -self.u_max, self.u_max))
        return np.array([xf_new]), u


class PositivePositionFeedback:
    """
    Positive position feedback on one target mode.

        eta_ddot + 2 zf wf eta_dot + wf^2 eta = wf^2 y
        u = g * eta

    Discretised by matrix exponential (exact for the zero-order-held input).
    """

    name = "PPF"

    def __init__(self, plate, dt, gain, mode=0, zeta_f=0.3,
                 freq_ratio=1.0, u_max=150.0):
        self.dt = dt
        self.u_max = u_max
        self.gain = gain
        self.sign = np.sign(_plant_dc_gain(plate)) or 1.0
        wf = float(plate.omega_n[mode]) * freq_ratio
        self.wf = wf
        Ac = np.array([[0.0, 1.0], [-wf ** 2, -2.0 * zeta_f * wf]])
        Bc = np.array([[0.0], [wf ** 2]])
        Maug = np.zeros((3, 3))
        Maug[:2, :2] = Ac
        Maug[:2, 2:] = Bc
        E = expm(Maug * dt)
        self.Ad = E[:2, :2]
        self.Bd = E[:2, 2:].reshape(2)
        self.n_state = 2

    def step(self, x_prev, u_prev, y_meas):
        xf = np.atleast_1d(x_prev)[:2].astype(float)
        xf_new = self.Ad @ xf + self.Bd * y_meas
        u = -self.sign * self.gain * xf_new[0]
        u = float(np.clip(u, -self.u_max, self.u_max))
        return xf_new, u


class RepetitiveController:
    """
    Plug-in repetitive controller on top of a stabilising inner loop.

        u = u_inner(x_hat) + u_rc
        u_rc[k] = Q(q) u_rc[k-N] + L e[k-N+lead]

    Q is a zero-phase moving average over 2*q_width+1 samples, applied to the
    delayed memory; `lead` compensates the inner loop's phase at the
    tooth-passing harmonics. N is the tooth-passing period in samples.

    CONVERGENCE (why this is the right benchmark, and what a "learned"
    phase-indexed feedforward has to beat): with S(z) the sensitivity of the
    inner loop, the repetitive loop converges monotonically at every harmonic
    h of the tooth-passing frequency iff

        |Q(e^{jwh}) [1 - L z^{lead} S(e^{jwh})]| < 1 .

    That is a checkable condition, reported by `convergence_margin()`. An
    iterative-learning scheme with a hand-set correction gain has no such
    certificate.
    """

    name = "RC"

    def __init__(self, inner, dt, period_samples, L=0.5, lead=0,
                 q_width=2, u_max=150.0, enabled=True):
        self.inner = inner
        self.dt = dt
        self.N = int(period_samples)
        self.L = L
        self.lead = int(lead)
        self.q_width = int(q_width)
        self.u_max = u_max
        self.enabled = enabled
        self.mem_u = np.zeros(self.N)      # u_rc one period back
        self.mem_e = np.zeros(self.N)      # error one period back
        self.k = 0

    def _q_filter(self, idx):
        """Zero-phase moving average of the delayed memory around idx."""
        w = self.q_width
        if w == 0:
            return self.mem_u[idx % self.N]
        sl = [(idx + j) % self.N for j in range(-w, w + 1)]
        return float(np.mean(self.mem_u[sl]))

    def step(self, x_prev, u_prev, y_meas, k_step=None):
        k = self.k if k_step is None else k_step
        x_hat, u_in = self.inner.step(x_prev, u_prev, y_meas)

        idx = k % self.N
        u_rc = 0.0
        if self.enabled:
            src = (idx + self.lead) % self.N
            u_rc = self._q_filter(idx) + self.L * self.mem_e[src]
            u_rc = float(np.clip(u_rc, -self.u_max, self.u_max))

        # memories are written for the NEXT period
        self.mem_u[idx] = u_rc
        self.mem_e[idx] = -y_meas          # drive the error towards zero
        self.k = k + 1

        u = float(np.clip(u_in + u_rc, -self.u_max, self.u_max))
        return x_hat, u

    def convergence_margin(self, S_at_harmonics):
        """
        max_h |Q [1 - L z^lead S]| over the supplied harmonic sensitivities.
        Values < 1 certify monotone convergence at those harmonics.
        """
        vals = []
        for h, S in enumerate(S_at_harmonics, start=1):
            w = 2 * np.pi * h / self.N
            Q = 1.0 if self.q_width == 0 else float(
                np.mean([np.cos(w * j) for j in range(-self.q_width,
                                                      self.q_width + 1)]))
            vals.append(abs(Q * (1.0 - self.L * np.exp(1j * w * self.lead) * S)))
        return float(np.max(vals)) if vals else np.nan


def tune_scalar_gain(build_fn, run_fn, lo, hi, n_grid=12, u_budget=None,
                     n_refine=2, verbose=False):
    """
    Tune a one-parameter law by grid search plus local refinement.

    build_fn(gain) -> controller
    run_fn(controller) -> dict(cost=..., u_rms=..., u_max=..., ok=bool)

    The best gain minimises `cost` among runs that are stable and respect
    `u_budget` on u_rms, so every law in the benchmark is compared at the
    same control effort rather than at whatever effort makes it look best.
    """
    best = None
    for _ in range(n_refine + 1):
        grid = np.geomspace(lo, hi, n_grid)
        for g in grid:
            try:
                r = run_fn(build_fn(float(g)))
            except Exception:
                continue
            if not r.get('ok', True) or not np.isfinite(r['cost']):
                continue
            if u_budget is not None and r['u_rms'] > u_budget:
                continue
            if best is None or r['cost'] < best[1]['cost']:
                best = (float(g), r)
        if best is None:
            return None, None
        g0 = best[0]
        lo, hi = g0 / 3.0, g0 * 3.0
        if verbose:
            print(f"      refine around g = {g0:.4g}, cost = {best[1]['cost']:.5g}")
    return best
