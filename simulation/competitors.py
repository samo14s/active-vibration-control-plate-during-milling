"""
competitors.py — literature baselines re-implemented on the frozen plate base.

Two reference architectures, both driven by the SAME single non-collocated
displacement measurement and the SAME +/-150 V saturation as the proposed
controller, so that only the control architecture differs:

  * ModalLQG (alpha = 0)
        Kalman observer on the 10-state structural modal model (NO extended
        disturbance states) + LQR modal state feedback. The regenerative
        delay is left unmodelled, i.e. treated as an exogenous disturbance.
        This is the architecture of Parus et al. (J. Vib. Control 2013) and
        the LQG branch of Du et al. (Int. J. Mech. Sci. 2022).

  * ModalLQG (alpha != 0) — "memory-based" / delayed state feedback
        u = -K x_hat(t) - alpha * K x_hat(t - tau)
        i.e. the control law uses both the current and the one-period-delayed
        state, as in Dong et al. (J. Vib. Control 2023) and Ahvazi et al.
        (ISA Trans. 2025). Those works synthesise an independent delayed gain
        through LMI / gradient methods assuming the full state is measured;
        here the delayed gain is constrained to be collinear with K and its
        scale alpha is optimised by the same PSO used for every other
        controller, with the state supplied by the observer (same sensing).

Interface: NewmarkSimulator convention step(x_hat_prev, u_prev, y_meas).
"""
import numpy as np
from scipy.linalg import expm


class ModalLQG:
    def __init__(self, dt, plate, log10_rho=-6.0, log10_qw=0.0,
                 log10_R=-7.0, alpha=0.0, tau=None, w_hi=0.01,
                 V_max=150.0, t_ramp=0.02, g_reg=0.0, g_reg_v=0.0):
        self.dt = float(dt)
        self.V_max = float(V_max)
        self.t_ramp = float(t_ramp)
        # anti-spillover drive filter (2nd-order LP, fc = 1.8 kHz, Tustin) —
        # models the band-limited amplifier path; same filter in every
        # architecture so the comparison stays architecture-only.
        import numpy as _np
        _wc = 2 * _np.pi * 1800.0
        _K = 2.0 / self.dt
        _a0 = _K**2 + _np.sqrt(2.0) * _wc * _K + _wc**2
        self._lpf_b = _np.array([_wc**2, 2*_wc**2, _wc**2]) / _a0
        self._lpf_a = _np.array([2*(_wc**2 - _K**2) / _a0,
                                 (_K**2 - _np.sqrt(2.0)*_wc*_K + _wc**2) / _a0])
        self._lpf_x = _np.zeros(2)
        self._lpf_y = _np.zeros(2)
        self.alpha = float(alpha)
        self.g_reg = float(g_reg)
        self.g_reg_v = float(g_reg_v)
        self.Dp0 = np.asarray(plate.Dp_array[:, 0]).copy()

        n = plate.n_modes
        K = np.diag(plate.Kp) if plate.Kp.ndim == 2 else np.asarray(plate.Kp)
        C = np.diag(plate.Cp) if plate.Cp.ndim == 2 else np.asarray(plate.Cp)
        H = np.asarray(plate.H_Pe_modal).ravel()
        D = np.asarray(plate.D_obs).ravel()
        self.n = n
        nx = 2 * n

        A = np.zeros((nx, nx))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -np.diag(K)
        A[n:, n:] = -np.diag(C)
        B = np.zeros(nx)
        B[n:] = H
        Cx = np.zeros(nx)
        Cx[:n] = D
        self.Cx = Cx

        M = np.zeros((nx + 1, nx + 1))
        M[:nx, :nx] = A
        M[:nx, nx] = B
        Md = expm(M * self.dt)
        Ad, Bd = Md[:nx, :nx], Md[:nx, nx]
        self.Ad, self.Bd = Ad, Bd

        # --- Kalman observer (force-like process noise on the modal
        #     accelerations; the unmodelled regenerative force is what this
        #     noise level implicitly stands for)
        q_amp = 1e-8
        Qd = np.diag(np.concatenate([
            np.full(n, q_amp ** 2),
            np.full(n, (10.0 ** log10_qw) ** 2)])) * self.dt
        Rval = (10.0 ** log10_R) ** 2
        P = Qd.copy()
        c = Cx
        for _ in range(8000):
            s = float(c @ P @ c) + Rval
            Kg = (P @ c) / s
            Pc = P - np.outer(Kg, c @ P)
            Pn = Ad @ Pc @ Ad.T + Qd
            if np.max(np.abs(Pn - P)) <= 1e-8 * max(np.max(np.abs(Pn)), 1e-30):
                P = Pn
                break
            P = Pn
        s = float(c @ P @ c) + Rval
        self.Ld = (P @ c) / s

        # --- LQR modal state feedback (energy weighting, modes 1-2)
        wmode = np.array([1.0, 1.0] + [w_hi] * (n - 2))
        Qx = np.diag(np.concatenate([K * wmode, wmode]))
        Ru = 10.0 ** log10_rho
        P = Qx.copy()
        for _ in range(30000):
            s = float(Bd @ P @ Bd) + Ru
            Kg = (Bd @ P @ Ad) / s
            Pn = Ad.T @ P @ Ad - np.outer(Ad.T @ P @ Bd, Kg) + Qx
            if np.max(np.abs(Pn - P)) <= 1e-8 * max(np.max(np.abs(Pn)), 1e-30):
                P = Pn
                break
            P = Pn
        s = float(Bd @ P @ Bd) + Ru
        self.K_lqr = (Bd @ P @ Ad) / s

        self.nx = nx
        self.xh = np.zeros(nx)
        self.n_tau = int(round(tau / dt)) if tau else 0
        self._buf = np.zeros((max(self.n_tau, 1), nx))
        self._bi = 0
        self._k = 0
        self._lpf_x[:] = 0.0
        self._lpf_y[:] = 0.0

    # ------------------------------------------------------------------
    def step(self, x_hat_prev, u_prev, y_meas, x_p_now=None):
        innov = float(y_meas) - float(self.Cx @ self.xh)
        xc = self.xh + self.Ld * innov
        self.xh = self.Ad @ xc + self.Bd * float(u_prev)

        u = -float(self.K_lqr @ self.xh)
        if self.n_tau > 0:
            if self.alpha != 0.0:
                u -= self.alpha * float(self.K_lqr @ self._buf[self._bi])
            n = self.n
            if self.g_reg != 0.0:
                u += self.g_reg * float(
                    self.Dp0 @ (self.xh[:n] - self._buf[self._bi][:n]))
            if self.g_reg_v != 0.0:
                u += self.g_reg_v * float(
                    self.Dp0 @ (self.xh[n:2 * n]
                                - self._buf[self._bi][n:2 * n]))
            self._buf[self._bi] = self.xh
            self._bi = (self._bi + 1) % self.n_tau

        self._k += 1
        uf = (self._lpf_b[0]*u + self._lpf_b[1]*self._lpf_x[0]
              + self._lpf_b[2]*self._lpf_x[1]
              - self._lpf_a[0]*self._lpf_y[0] - self._lpf_a[1]*self._lpf_y[1])
        self._lpf_x[1] = self._lpf_x[0]; self._lpf_x[0] = u
        self._lpf_y[1] = self._lpf_y[0]; self._lpf_y[0] = uf
        ramp = min(self._k * self.dt / self.t_ramp, 1.0)
        u = float(np.clip(ramp * uf, -self.V_max, self.V_max))
        return self.xh, u

    def reset(self):
        self.xh = np.zeros(self.nx)
        self._buf[:] = 0.0
        self._bi = 0
        self._k = 0
        self._lpf_x[:] = 0.0
        self._lpf_y[:] = 0.0
