"""
modal_adrc.py — Proposed controller (article 2 / thesis axis 3).

Modal-ESO ADRC-FOPID for the non-collocated thin-walled plate:

  * model-aided extended state observer built on the calibrated 5-mode modal
    model, augmented with one total-disturbance state per dominant chatter
    mode (modes 1 and 2). The regenerative delayed force, the intermittent
    cutting force, modal coupling and model mismatch are all absorbed into
    the two disturbance states — the article-1 ADRC idea lifted to modal
    space.
  * disturbance cancellation through the single piezo channel in the
    least-squares sense over the dominant modes (single input cannot cancel
    two modal forces exactly):  u_dc = -beta * (H1 d1 + H2 d2)/(H1^2 + H2^2)
  * FOPID feedback shaped on the observer-reconstructed dominant-mode output
    y12 = D1 q1 + D2 q2 (a model-based modal filter: modes 3-5 are excluded
    from the feedback path, killing observation spillover).

Observer gains: steady-state Kalman (CARE), parameterised by a single
disturbance-intensity decade log10(sig_d) and measurement decade log10(R).
Discretisation: exact ZOH of the closed observer via a single expm.

Interface: NewmarkSimulator convention step(x_hat_prev, u_prev, y_meas).
"""
import numpy as np
from scipy.linalg import expm


class ModalADRCFOPID:
    n_dist = 2                    # disturbance states on modes 1 and 2

    def __init__(self, dt, plate, fopid_params,
                 log10_sigd=9.0, log10_R=-15.0, beta=1.0, tau=None,
                 log10_qw=None, log10_fhp=None,
                 y_ref=1e-5, V_max=150.0,
                 w_c=2 * np.pi * 537.0, w_b=2 * np.pi * 20.0,
                 w_h=2 * np.pi * 5000.0, N_oust=4):
        from fopid_controller import FractionalOperator

        self.dt = float(dt)
        self.V_max = float(V_max)
        self.y_ref = float(y_ref)
        self.beta = float(beta)

        n = plate.n_modes
        K = np.diag(plate.Kp) if plate.Kp.ndim == 2 else np.asarray(plate.Kp)
        C = np.diag(plate.Cp) if plate.Cp.ndim == 2 else np.asarray(plate.Cp)
        H = np.asarray(plate.H_Pe_modal).ravel()
        D = np.asarray(plate.D_obs).ravel()
        self.H, self.D = H, D
        self._Kstiff = np.asarray(K, float).copy()

        nd = self.n_dist
        nx = 2 * n + nd
        A = np.zeros((nx, nx))
        A[:n, n:2 * n] = np.eye(n)
        A[n:2 * n, :n] = -np.diag(K)
        A[n:2 * n, n:2 * n] = -np.diag(C)
        S = np.zeros((n, nd))
        S[0, 0] = 1.0
        S[1, 1] = 1.0
        A[n:2 * n, 2 * n:] = S
        B = np.zeros((nx, 1))
        B[n:2 * n, 0] = H
        Cx = np.zeros((1, nx))
        Cx[0, :n] = D
        self.A, self.B, self.Cx = A, B, Cx

        # discrete steady-state Kalman observer (physically scaled DARE)
        M = np.zeros((nx + 1, nx + 1))
        M[:nx, :nx] = A
        M[:nx, nx] = B[:, 0]
        Md = expm(M * self.dt)
        Ad = Md[:nx, :nx]
        Bd = Md[:nx, nx]
        q_amp = 1e-8                          # modal-coordinate noise floor
        if log10_qw is None:
            vel_q = (q_amp * np.sqrt(K)) ** 2   # legacy (model-trusting)
        else:
            # force-like process noise on the modal accelerations: this is
            # what lets the filter keep up with the regenerative excitation
            vel_q = np.full(n, (10.0 ** log10_qw) ** 2)
        Qd = np.diag(np.concatenate([
            np.full(n, q_amp ** 2),
            vel_q,
            np.full(nd, (10.0 ** log10_sigd) ** 2)])) * self.dt
        Rval = (10.0 ** log10_R) ** 2
        # fixed-point iteration of the filter Riccati recursion (robust for
        # near-unit-circle modes where the QZ-based DARE solver fails)
        P = Qd.copy()
        c = Cx[0]
        for _ in range(8000):
            s = float(c @ P @ c) + Rval
            K = (P @ c) / s
            Pc = P - np.outer(K, c @ P)
            Pn = Ad @ Pc @ Ad.T + Qd
            if np.max(np.abs(Pn - P)) <= 1e-8 * max(np.max(np.abs(Pn)), 1e-30):
                P = Pn
                break
            P = Pn
        s = float(c @ P @ c) + Rval
        Ld = (P @ c) / s                      # filter gain (nx,)
        Ld = Ld.reshape(-1, 1)
        self.Ad, self.Bd, self.Ld = Ad, Bd, Ld.ravel()

        self.nx, self.n = nx, n
        self.xh = np.zeros(nx)

        # FOPID on the reconstructed dominant-mode output
        Kp, Ki, Kd, lam, mu = [float(v) for v in fopid_params]
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.lam = float(np.clip(lam, 0.01, 1.0))
        self.mu = float(np.clip(mu, 0.01, 1.0))
        self.I_op = FractionalOperator(-self.lam, dt, w_c, w_b, w_h, N_oust)
        self.D_op = FractionalOperator(+self.mu, dt, w_c, w_b, w_h, N_oust)

        # --- modal-state feedback layer (discrete LQR via robust recursion)
        n10 = 2 * n
        A10 = A[:n10, :n10]
        M2 = np.zeros((n10 + 1, n10 + 1))
        M2[:n10, :n10] = A10
        M2[n:n10, n10] = H            # voltage enters velocity rows
        M2d = expm(M2 * self.dt)
        Ad10, Bd10 = M2d[:n10, :n10], M2d[:n10, n10]
        self.K_lqr = np.zeros(n10)
        self._lqr_ready = False
        self._Ad10, self._Bd10 = Ad10, Bd10
        B10c = np.zeros(n10)
        B10c[n:] = H
        self._A10c, self._B10c = A10.copy(), B10c
        self.K_fs = None
        self._fs_ready = False
        self._h = np.zeros(0)

        h12 = H[0] ** 2 + H[1] ** 2
        self._cancel = np.zeros(nx)
        self._cancel[2 * n] = H[0] / h12
        self._cancel[2 * n + 1] = H[1] / h12
        self.g_rate = 0.0                      # modal-rate damping gain
        self.t_ramp = 0.05                     # soft-start ramp [s]
        _wc = 2 * np.pi * 1800.0               # anti-spillover drive filter
        _K = 2.0 / self.dt
        _a0 = _K**2 + np.sqrt(2.0) * _wc * _K + _wc**2
        self._lpf_b = np.array([_wc**2, 2*_wc**2, _wc**2]) / _a0
        self._lpf_a = np.array([2*(_wc**2 - _K**2) / _a0,
                                (_K**2 - np.sqrt(2.0)*_wc*_K + _wc**2) / _a0])
        self._lpf_x = np.zeros(2)
        self._lpf_y = np.zeros(2)
        self._k = 0
        # regenerative-kernel reshaping layer (delayed estimated states)
        self.g_reg = 0.0
        self.g_reg_v = 0.0          # phase-lead companion of g_reg
        self.Dp0 = np.asarray(plate.Dp_array[:, 0]).copy()
        self._Dp_all = np.asarray(plate.Dp_array)
        self._L_plate = float(plate.xp_array[-1])
        self.n_tau = int(round(tau / dt)) if tau else 0
        self._qbuf = np.zeros((max(self.n_tau, 1), n))
        self._vbuf = np.zeros((max(self.n_tau, 1), n))
        self._bi = 0
        self.f_hp = 400.0 if log10_fhp is None else 10.0 ** log10_fhp
        a = np.exp(-2 * np.pi * self.f_hp * self.dt)
        self._hp_a = a
        self._d_lp = 0.0                       # LP state of projected dist.


    def build_fslqr(self, log10_rho=-6.0, w_hi=0.01, shapes=None):
        """Frequency-shaped LQR (Li et al., MSSP 2023, extended to modal
        space and to a delay-driven plant).

        A resonant band-pass filter  G_L(s) = p*xi0*w0*s / (s^2 + xi0*w0*s
        + w0^2)  is placed on each targeted modal coordinate; its states
        augment the 10-state structural model and the augmented quadratic
        cost is solved by the same robust Riccati recursion. The result is a
        control gain that is large only in a narrow band around each chatter
        frequency, buying authority without spending voltage budget
        elsewhere.

        shapes: list of (mode_index, f0_Hz, xi0, p)
        """
        n = self.n
        n10 = 2 * n
        if not shapes:
            return self.build_lqr(log10_rho=log10_rho, w_hi=w_hi)

        nf = 2 * len(shapes)
        Aa = np.zeros((n10 + nf, n10 + nf))
        Aa[:n10, :n10] = self._A10c
        Ba = np.zeros(n10 + nf)
        Ba[:n10] = self._B10c
        Cq = np.zeros((len(shapes), n10 + nf))
        for j, (mi, f0, xi0, p) in enumerate(shapes):
            w0 = 2 * np.pi * float(f0)
            r = n10 + 2 * j
            # filter state-space: h1' = h2 ; h2' = -w0^2 h1 - xi0 w0 h2 + q_mi
            Aa[r, r + 1] = 1.0
            Aa[r + 1, r] = -w0 ** 2
            Aa[r + 1, r + 1] = -xi0 * w0
            Aa[r + 1, mi] = 1.0                  # driven by modal coord q_mi
            Cq[j, r + 1] = p * xi0 * w0          # filtered output

        # exact ZOH of the augmented plant
        M = np.zeros((n10 + nf + 1, n10 + nf + 1))
        M[:n10 + nf, :n10 + nf] = Aa
        M[:n10 + nf, n10 + nf] = Ba
        Md = expm(M * self.dt)
        Ada, Bda = Md[:n10 + nf, :n10 + nf], Md[:n10 + nf, n10 + nf]

        wmode = np.array([1.0, 1.0] + [w_hi] * (n - 2))
        Qx = np.zeros((n10 + nf, n10 + nf))
        Qx[:n10, :n10] = np.diag(np.concatenate([self._Kstiff * wmode, wmode]))
        Qx += Cq.T @ Cq                          # frequency-shaped term
        Ru = 10.0 ** log10_rho

        P = Qx.copy()
        for _ in range(30000):
            s = float(Bda @ P @ Bda) + Ru
            Kg = (Bda @ P @ Ada) / s
            Pn = Ada.T @ P @ Ada - np.outer(Ada.T @ P @ Bda, Kg) + Qx
            if np.max(np.abs(Pn - P)) <= 1e-8 * max(np.max(np.abs(Pn)), 1e-30):
                P = Pn
                break
            P = Pn
        s = float(Bda @ P @ Bda) + Ru
        Kfull = (Bda @ P @ Ada) / s
        self.K_lqr = Kfull[:n10]
        self.K_fs = Kfull[n10:]

        # discrete filter dynamics driven by the estimated modal coordinates
        Af = Aa[n10:, :]
        Mf = np.zeros((n10 + nf, n10 + nf))
        Mf[n10:, :] = Af
        Mfd = expm(Mf * self.dt)
        self._Ffs = Mfd[n10:, n10:]              # filter-state transition
        self._Gfs = Mfd[n10:, :n10]              # input from modal states
        self._h = np.zeros(nf)
        self._fs_ready = True
        self._lqr_ready = True

    def build_lqr(self, log10_rho=-6.0, w_hi=0.01):
        """Discrete LQR on the 10-state structural model, energy-weighted on
        the two dominant modes; high modes de-weighted (spillover care)."""
        n = self.n
        wmode = np.array([1.0, 1.0] + [w_hi] * (n - 2))
        Qx = np.diag(np.concatenate([self._Kstiff * wmode, wmode]))
        Ru = 10.0 ** log10_rho
        Ad, Bd = self._Ad10, self._Bd10
        P = Qx.copy()
        for _ in range(30000):
            s = float(Bd @ P @ Bd) + Ru
            Kg = (Bd @ P @ Ad) / s
            Pn = Ad.T @ P @ Ad - np.outer(Ad.T @ P @ Bd, Kg) + Qx
            if np.max(np.abs(Pn - P)) <= 1e-9 * max(np.max(np.abs(Pn)), 1e-30):
                P = Pn
                break
            P = Pn
        s = float(Bd @ P @ Bd) + Ru
        self.K_lqr = (Bd @ P @ Ad) / s
        self._lqr_ready = True

    # ------------------------------------------------------------------
    def step(self, x_hat_prev, u_prev, y_meas, x_p_now=None):
        if x_p_now is not None:
            kp = int(round(np.clip(x_p_now / self._L_plate, 0.0, 1.0)
                           * (self._Dp_all.shape[1] - 1)))
            self.Dp0 = self._Dp_all[:, kp]
        # measurement update (y corresponds to the previous plant state),
        # then time update with the actually applied voltage
        innov = float(y_meas) - float(self.Cx[0] @ self.xh)
        xc = self.xh + self.Ld * innov
        self.xh = self.Ad @ xc + self.Bd * float(u_prev)

        # model-based modal filter: dominant-mode output for the FOPID
        y12 = self.D[0] * self.xh[0] + self.D[1] * self.xh[1]
        e = y12 / self.y_ref
        u_fb = (self.Kp * e + self.Ki * self.I_op.step(e)
                + self.Kd * self.D_op.step(e))

        d_proj = float(self._cancel @ self.xh)
        self._d_lp = self._hp_a * self._d_lp + (1 - self._hp_a) * d_proj
        u_dc = -self.beta * (d_proj - self._d_lp)
        n = self.n
        u_lqr = -float(self.K_lqr @ self.xh[:2 * n]) if self._lqr_ready else 0.0
        if self._fs_ready:
            u_lqr -= float(self.K_fs @ self._h)
            self._h = self._Ffs @ self._h + self._Gfs @ self.xh[:2 * n]
        u_reg = 0.0
        if (self.g_reg != 0.0 or self.g_reg_v != 0.0) and self.n_tau > 0:
            q_now = self.xh[:n]
            q_tau = self._qbuf[self._bi]
            v_now = self.xh[n:2 * n]
            v_tau = self._vbuf[self._bi]
            u_reg = (self.g_reg * float(self.Dp0 @ (q_now - q_tau))
                     + self.g_reg_v * float(self.Dp0 @ (v_now - v_tau)))
            self._qbuf[self._bi] = q_now
            self._vbuf[self._bi] = v_now
            self._bi = (self._bi + 1) % self.n_tau
        elif self.n_tau > 0:
            self._qbuf[self._bi] = self.xh[:n]
            self._vbuf[self._bi] = self.xh[n:2 * n]
            self._bi = (self._bi + 1) % self.n_tau
        u_rate = -self.g_rate * (self.H[0] * self.xh[n]
                                 + self.H[1] * self.xh[n + 1])
        self._k += 1
        _u0 = u_fb + u_dc + u_rate + u_lqr + u_reg
        _uf = (self._lpf_b[0]*_u0 + self._lpf_b[1]*self._lpf_x[0]
               + self._lpf_b[2]*self._lpf_x[1]
               - self._lpf_a[0]*self._lpf_y[0] - self._lpf_a[1]*self._lpf_y[1])
        self._lpf_x[1] = self._lpf_x[0]; self._lpf_x[0] = _u0
        self._lpf_y[1] = self._lpf_y[0]; self._lpf_y[0] = _uf
        ramp = min(self._k * self.dt / self.t_ramp, 1.0)
        u = float(np.clip(ramp * _uf, -self.V_max, self.V_max))

        return self.xh[:2 * self.n], u

    def reset(self):
        self.xh = np.zeros(self.nx)
        self._d_lp = 0.0
        self._k = 0
        self._qbuf[:] = 0.0
        self._vbuf[:] = 0.0
        self._bi = 0
        self._lpf_x[:] = 0.0
        self._lpf_y[:] = 0.0
        if self._fs_ready:
            self._h = np.zeros(len(self._h))
        self.I_op.reset()
        self.D_op.reset()

    def __repr__(self):
        return (f"ModalADRCFOPID(Kp={self.Kp:+.1f}, Ki={self.Ki:+.1f}, "
                f"Kd={self.Kd:+.1f}, lam={self.lam:.2f}, mu={self.mu:.2f}, "
                f"beta={self.beta:.2f})")
