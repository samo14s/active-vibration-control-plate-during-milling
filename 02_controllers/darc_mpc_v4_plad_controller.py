"""
darc_mpc_v4_plad_controller.py
===============================
   ╔══════════════════════════════════════════════════════════════════╗
   ║   DARC-MPC v4 : PLAD — Phase-Locked Adaptive DARC                 ║
   ║                                                                  ║
   ║   Addresses the research gap left open by v3:                    ║
   ║                                                                  ║
   ║   v3 :  u = u_LQG(x̂) + α · NN_FF(φ_clock, x̂)                     ║
   ║         with  φ_clock = 2π (k mod n_per) / n_per                 ║
   ║         → OPEN-LOOP clock. Assumes the tooth-passing period      ║
   ║           is EXACTLY known and constant. Any real spindle-speed  ║
   ║           deviation (1–3 % droop under load, speed fluctuation,  ║
   ║           deliberate spindle-speed variation) desynchronises     ║
   ║           the learned feedforward, which then injects a          ║
   ║           periodic voltage at the WRONG phase — performance      ║
   ║           degrades toward, and below, the LQG baseline.          ║
   ║         → v3 also computed an adaptation factor (lambda_robust)  ║
   ║           that was never applied to the control signal: the      ║
   ║           advertised adaptation layer was inert.                 ║
   ║                                                                  ║
   ║   v4 :  u = u_LQG(x̂) + α·c_lock(t) · NN_FF(φ̂(t), x̂)              ║
   ║         with  φ̂(t) estimated ONLINE, sensorless, from the        ║
   ║         displacement measurement itself:                        ║
   ║                                                                  ║
   ║         1. Band-pass filter y around the nominal tooth-passing   ║
   ║            frequency f_t = N_T·RPM/60.                           ║
   ║         2. Digital PLL (product phase detector + PI loop) locks  ║
   ║            onto the forced fundamental → (ω̂, θ̂).                 ║
   ║         3. Model-based phase referencing maps the measured       ║
   ║            vibration phase back to the disturbance clock:        ║
   ║               φ̂ = θ̂ − ∠H_bp(ω̂) − ∠G_cl(ω̂, x_p) + φ_cal          ║
   ║            where G_cl is the closed-loop FRF from the modal      ║
   ║            cutting-force fundamental to the sensor, scheduled    ║
   ║            over tool position x_p (grid + interpolation), and    ║
   ║            φ_cal is a one-shot offline calibration constant      ║
   ║            identified at nominal conditions (absorbs neglected   ║
   ║            regenerative coupling).                               ║
   ║         4. The PLL lock quality gives a confidence c_lock ∈      ║
   ║            [0,1] which GATES the feedforward: when lock is       ║
   ║            lost or not yet acquired the controller degrades      ║
   ║            GRACEFULLY to pure LQG — the adaptation layer is now  ║
   ║            actually wired into the control law.                  ║
   ║                                                                  ║
   ║   The Lyapunov safety filter of v3 is retained unchanged.        ║
   ╚══════════════════════════════════════════════════════════════════╝

Interface notes
---------------
* The class exposes ``enable_gs`` so that ``NewmarkSimulator`` passes the
  current tool position ``x_p_now`` (metres) to ``step()`` — the hook that
  existed in the solver but was never used by any controller.
* ``training_mode = True`` makes ``step()`` fall back to the internal
  step-counter clock (valid because training/calibration are performed at
  nominal, exactly-known spindle speed).  The PLL still runs in shadow so
  that ``calibrate_phase_reference()`` can identify φ_cal.
* Runtime state (PLL, counters, histories) auto-resets whenever the
  history buffers have been cleared, so existing driver scripts that
  reset ``history_*`` between runs keep working unmodified.
"""

import numpy as np

from darc_mpc_v3_controller import DARC_MPC_v3_Controller


# ====================================================================
# Sensorless spindle-phase observer : band-pass + digital PLL
# ====================================================================
class SpindlePhaseObserver:
    """
    Tracks the tooth-passing fundamental in the displacement signal.

    y_bp(t) ≈ A·cos(θ_y(t)) after the band-pass.  A product phase
    detector demodulates y_bp against the internal oscillator; a PI
    loop steers (ω̂, θ̂) so that θ̂ → θ_y.  The in-phase correlation
    I ≈ ½·cos(θ_y − θ̂) provides the lock-quality metric.
    """

    def __init__(self, f_nom, dt,
                 bp_Q=4.0,
                 pll_fn=8.0, pll_zeta=1.0,
                 lpf_fc=50.0, env_fc=20.0, conf_tau=0.040,
                 omega_pull=0.07,
                 amp_min=0.01e-6):
        self.f_nom = f_nom
        self.dt = dt
        self.omega_nom = 2*np.pi*f_nom

        # --- band-pass biquad (RBJ, constant 0 dB peak gain) ---
        w0 = 2*np.pi*f_nom*dt
        alpha = np.sin(w0)/(2*bp_Q)
        a0 = 1 + alpha
        self.b0 = alpha/a0
        self.b2 = -alpha/a0
        self.a1 = -2*np.cos(w0)/a0
        self.a2 = (1 - alpha)/a0

        # --- PI loop gains (phase-detector gain k_d = 1/2) ---
        kd = 0.5
        wn = 2*np.pi*pll_fn
        self.Kp = 2*pll_zeta*wn/kd
        self.Ki = wn**2/kd

        # --- one-pole low-pass coefficients ---
        self.beta_lpf = 1 - np.exp(-2*np.pi*lpf_fc*dt)
        self.beta_env = 1 - np.exp(-2*np.pi*env_fc*dt)
        self.beta_cnf = 1 - np.exp(-dt/conf_tau)

        self.omega_pull = omega_pull      # relative pull-in range
        self.amp_min = amp_min            # amplitude validity floor [m]
        self.warmup_steps = int(0.060/dt)

        self.reset()

    # ---------------------------------------------------------------
    def reset(self):
        self.x1 = 0.0; self.x2 = 0.0      # biquad input delay line
        self.y1 = 0.0; self.y2 = 0.0      # biquad output delay line
        self.theta = 0.0
        self.omega = self.omega_nom
        self.e_lpf = 0.0                  # quadrature branch (phase error)
        self.i_lpf = 0.0                  # in-phase branch (lock metric)
        self.env = 0.0                    # rectified-mean envelope
        self.conf = 0.0
        self.n_steps = 0

    # ---------------------------------------------------------------
    def bandpass_phase(self, omega=None):
        """Phase of the band-pass biquad at angular frequency omega."""
        if omega is None:
            omega = self.omega
        z = np.exp(1j*omega*self.dt)
        H = (self.b0 + self.b2*z**-2) / (1 + self.a1*z**-1 + self.a2*z**-2)
        return float(np.angle(H))

    # ---------------------------------------------------------------
    def update(self, y_meas):
        """One sample. Returns (theta, omega, confidence)."""
        # 1. band-pass
        y_bp = (self.b0*y_meas + self.b2*self.x2
                - self.a1*self.y1 - self.a2*self.y2)
        self.x2 = self.x1;  self.x1 = y_meas
        self.y2 = self.y1;  self.y1 = y_bp

        # 2. envelope (rectified mean → amplitude of a sine × 2/π)
        self.env += self.beta_env*(abs(y_bp) - self.env)
        A = max(self.env*np.pi/2, 1e-12)

        # 3. demodulation
        s = y_bp/A
        self.e_lpf += self.beta_lpf*(-s*np.sin(self.theta) - self.e_lpf)
        self.i_lpf += self.beta_lpf*( s*np.cos(self.theta) - self.i_lpf)

        # 4. PI loop
        self.omega += self.Ki*self.e_lpf*self.dt
        lo = self.omega_nom*(1 - self.omega_pull)
        hi = self.omega_nom*(1 + self.omega_pull)
        self.omega = min(max(self.omega, lo), hi)
        self.theta += (self.omega + self.Kp*self.e_lpf)*self.dt
        self.theta %= 2*np.pi

        # 5. lock confidence
        #    direction of the (I, Q) correlation vector = cos(Δθ) exactly,
        #    independent of the envelope normalisation (which is inflated
        #    by harmonics/chatter leaking through the band-pass);
        #    its magnitude (≈ ½ when locked) gates against pure noise.
        r = np.hypot(self.i_lpf, self.e_lpf)
        lock_dir = self.i_lpf/max(r, 1e-12)             # ≈ cos(Δθ)
        mag_ok = min(max((2*r - 0.30)/0.30, 0.0), 1.0)
        c_raw = min(max((lock_dir - 0.50)/0.40, 0.0), 1.0)*mag_ok
        self.n_steps += 1
        if self.n_steps < self.warmup_steps or self.env*np.pi/2 < self.amp_min:
            c_raw = 0.0
        self.conf += self.beta_cnf*(c_raw - self.conf)

        return self.theta, self.omega, self.conf


# ====================================================================
# DARC-MPC v4 : Phase-Locked Adaptive DARC (PLAD)
# ====================================================================
class DARC_MPC_v4_PLAD_Controller(DARC_MPC_v3_Controller):
    """
    v3 residual-correction architecture + sensorless phase-locked
    synchronisation of the learned feedforward.

    u(t) = u_LQG(x̂) + ff_alpha·c_lock(t) · NN_FF(φ̂(t), x̂)
    """

    enable_gs = True        # NewmarkSimulator passes x_p_now to step()

    def __init__(self, plate, dt,
                 alpha3_periodic=None,
                 n_ref_pos=21,
                 pll_kwargs=None,
                 **v3_kwargs):
        # The PLL confidence gate replaces the inert v3 RLS adaptation.
        v3_kwargs.setdefault('enable_adaptation', False)
        super().__init__(plate, dt, **v3_kwargs)

        self.f_nom = 1.0/(self.n_per*dt)          # tooth-passing frequency
        self.pll = SpindlePhaseObserver(self.f_nom, dt,
                                        **(pll_kwargs or {}))

        self.alpha3_periodic = alpha3_periodic
        self.phase_cal = 0.0
        self.training_mode = False
        self._build_phase_reference(n_ref_pos)
        self._reset_runtime()

        if self.verbose:
            print(f"  [v4-PLAD] f_nom = {self.f_nom:.1f} Hz, "
                  f"phase-ref grid: {n_ref_pos} positions x 3 freqs")

    # ---------------------------------------------------------------
    #  Model-based phase reference  ∠G_cl(ω, x_p)
    # ---------------------------------------------------------------
    def _build_phase_reference(self, n_ref_pos):
        """
        Closed-loop FRF phase from the modal cutting-force fundamental to
        the sensor, on a (frequency × tool-position) grid.  The cutting
        force fundamental is  f(t) ≈ Re{C_a3 · e^{iφ_clock}} · ft · Dp(x_p),
        so the expected sensor fundamental is
            y_f(t) = Re{ G_cl(iω, x_p) · e^{iφ_clock} },
            G_cl = D_obs (iωI − A_cl)^{-1} [0 ; Dp(x_p)] · C_a3 .
        Only the PHASE is needed (positive scalars ft drop out).
        """
        n = self.n_modes
        A_cl = self.A - self.B @ self.K_lqr

        # fundamental Fourier coefficient of alpha3 over the clock phase
        if self.alpha3_periodic is not None:
            a3 = np.asarray(self.alpha3_periodic, dtype=float)
            k = np.arange(len(a3))
            C_a3 = (2.0/len(a3))*np.sum(a3*np.exp(-1j*2*np.pi*k/len(a3)))
        else:
            C_a3 = 1.0 + 0j
        self.C_a3 = C_a3

        n_pos_total = self.plate.Dp_array.shape[1]
        kp_grid = np.linspace(0, n_pos_total - 1, n_ref_pos).astype(int)
        self.ref_xp = self.plate.xp_array[kp_grid]
        self.ref_freq_ratio = np.array([0.95, 1.00, 1.05])

        phase_grid = np.zeros((3, n_ref_pos))
        mag_grid = np.zeros((3, n_ref_pos))
        I2n = np.eye(2*n)
        for i_f, r in enumerate(self.ref_freq_ratio):
            s = 1j*2*np.pi*self.f_nom*r
            M_inv = np.linalg.inv(s*I2n - A_cl)
            for i_p, kp in enumerate(kp_grid):
                Bw = np.zeros(2*n, dtype=complex)
                Bw[n:] = self.plate.Dp_array[:, kp]
                y_ph = (self.plate.D_obs @ (M_inv @ Bw)[:n]) * C_a3
                phase_grid[i_f, i_p] = np.angle(y_ph)
                mag_grid[i_f, i_p] = np.abs(y_ph)

        # unwrap along both axes so bilinear interpolation is safe
        phase_grid = np.unwrap(phase_grid, axis=1)
        phase_grid = np.unwrap(phase_grid, axis=0)
        self.ref_phase_grid = phase_grid
        self.ref_mag_grid = mag_grid

    # ---------------------------------------------------------------
    def _ref_phase(self, omega, x_p):
        """Bilinear interpolation of ∠G_cl at (omega, x_p)."""
        r = omega/(2*np.pi*self.f_nom)
        fr = self.ref_freq_ratio
        r = min(max(r, fr[0]), fr[-1])
        i = 0 if r <= fr[1] else 1
        tr = (r - fr[i])/(fr[i+1] - fr[i])

        xp = min(max(x_p, self.ref_xp[0]), self.ref_xp[-1])
        j = int(np.searchsorted(self.ref_xp, xp) - 1)
        j = min(max(j, 0), len(self.ref_xp) - 2)
        tp = (xp - self.ref_xp[j])/(self.ref_xp[j+1] - self.ref_xp[j])

        g = self.ref_phase_grid
        return ((1-tr)*(1-tp)*g[i, j] + (1-tr)*tp*g[i, j+1]
                + tr*(1-tp)*g[i+1, j] + tr*tp*g[i+1, j+1])

    # ---------------------------------------------------------------
    def _reset_runtime(self):
        self.pll.reset()
        self.k_int = 1
        self.x_prev = np.zeros(self.n_x)
        self.u_prev = 0.0
        # diagnostics
        self.history_conf = []
        self.history_omega_hat = []
        self.history_phase_est = []
        self.history_phase_clock = []
        self.history_alpha_eff = []
        # shadow-calibration buffer
        self._cal_buf = []

    # ---------------------------------------------------------------
    def reset_runtime(self):
        """Public reset: call between simulation runs."""
        self.history_u_lqg = []
        self.history_u_ff = []
        self.history_u_total = []
        self.history_phase = []
        self.history_safety = []
        self._reset_runtime()

    # ---------------------------------------------------------------
    def _estimate_clock_phase(self, x_p_now):
        """Map PLL oscillator phase back to the disturbance clock phase."""
        th_bp = self.pll.bandpass_phase(self.pll.omega)
        th_ref = self._ref_phase(self.pll.omega, x_p_now)
        return (self.pll.theta - th_bp - th_ref + self.phase_cal) % (2*np.pi)

    # ---------------------------------------------------------------
    def step(self, x_hat_prev, u_prev, y_meas, x_p_now=0.0, k_step=0):
        """
        Pipeline v4 :
        1. Kalman observer                     (unchanged from v3)
        2. PLL spindle-phase observer          (new, sensorless)
        3. LQG action                          (unchanged)
        4. NN FF at estimated phase, gated by lock confidence (new)
        5. Lyapunov safety filter              (unchanged)
        """
        if not self.history_phase and self.k_int != 1:
            self._reset_runtime()       # histories were cleared externally

        # 1. Kalman
        x_hat = (self.A_obs_d @ x_hat_prev
                 + self.G_u.flatten()*u_prev
                 + self.G_y.flatten()*y_meas)

        # 2. spindle-phase observer (always runs; shadow in training mode)
        _, omega_hat, conf = self.pll.update(y_meas)
        phase_clock = 2*np.pi*(self.k_int % self.n_per)/self.n_per
        phase_est = self._estimate_clock_phase(x_p_now)

        if self.training_mode:
            phase = phase_clock          # nominal conditions: clock is exact
            alpha_eff = self.ff_alpha
            if conf > 0.6:               # collect samples for calibration
                self._cal_buf.append((self.k_int, phase_clock - phase_est))
        else:
            phase = phase_est
            alpha_eff = self.ff_alpha*conf

        self.history_phase.append(phase)
        self.history_conf.append(conf)
        self.history_omega_hat.append(omega_hat)
        self.history_phase_est.append(phase_est)
        self.history_phase_clock.append(phase_clock)
        self.history_alpha_eff.append(alpha_eff)

        # 3. LQG action
        u_lqg = float(np.squeeze(-self.K_lqr @ x_hat))
        u_lqg = np.clip(u_lqg, -self.u_max, self.u_max)
        self.history_u_lqg.append(u_lqg)

        # 4. gated feedforward
        u_ff, _, _ = self.ff_nn.forward(x_hat, phase)
        u_ff = alpha_eff*u_ff
        self.history_u_ff.append(u_ff)

        # 5. combine + safety
        u_proposed = np.clip(u_lqg + u_ff, -self.u_max, self.u_max)
        u_safe, violated = self.safety.filter_action(x_hat, u_proposed, u_lqg)
        u_safe = float(np.clip(u_safe, -self.u_max, self.u_max))

        self.history_u_total.append(u_safe)
        self.history_safety.append(1 if violated else 0)

        self.x_prev = x_hat.copy()
        self.u_prev = u_safe
        self.k_int += 1

        return x_hat, u_safe

    # ---------------------------------------------------------------
    def pretrain_iterative_simulation(self, *args, **kwargs):
        """Training runs at nominal speed: use the exact clock phase."""
        self.training_mode = True
        try:
            return super().pretrain_iterative_simulation(*args, **kwargs)
        finally:
            self.training_mode = False

    # ---------------------------------------------------------------
    def copy_feedforward_from(self, other):
        """Import trained NN weights (e.g. from a v3 controller)."""
        self.ff_nn.W1 = other.ff_nn.W1.copy()
        self.ff_nn.b1 = other.ff_nn.b1.copy()
        self.ff_nn.W2 = other.ff_nn.W2.copy()
        self.ff_nn.b2 = other.ff_nn.b2.copy()

    # ---------------------------------------------------------------
    def calibrate_phase_reference(self, simulator, alpha3_t, alpha4_t,
                                  kp_idx, T_cal=0.35, t_lock=0.15,
                                  verbose=True):
        """
        One-shot offline calibration at NOMINAL conditions.

        Runs a closed-loop simulation with the clock-driven feedforward
        (training mode) while the PLL shadows the measurement, then sets
        φ_cal to the circular mean of (φ_clock − φ̂_raw) over the locked
        portion.  Absorbs every systematic bias of the phase-reference
        model (neglected regenerative coupling, filter transients, …).
        """
        self.training_mode = True
        self.reset_runtime()
        self.phase_cal = 0.0
        try:
            simulator.simulate(alpha3_t, alpha4_t, kp_idx, controller=self,
                               progress=False, stop_at_time=T_cal)
        finally:
            self.training_mode = False

        k_lock = int(t_lock/self.dt)
        buf = np.array([d for k, d in self._cal_buf if k >= k_lock])
        if len(buf) == 0:
            if verbose:
                print("  [v4-PLAD] calibration FAILED: PLL never locked "
                      f"({len(self._cal_buf)} locked samples, none after "
                      f"t = {t_lock:.2f} s)")
            return None
        cal = float(np.angle(np.mean(np.exp(1j*buf))))
        spread = float(np.sqrt(-2*np.log(
            max(np.abs(np.mean(np.exp(1j*buf))), 1e-12))))
        self.phase_cal = cal
        self.reset_runtime()
        if verbose:
            print(f"  [v4-PLAD] phase calibration: φ_cal = {np.degrees(cal):+.1f}°"
                  f"  (circular std {np.degrees(spread):.1f}°, "
                  f"{len(buf)} samples)")
        return cal
