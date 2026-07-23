"""
rcsac_controller.py
===================
RC-SAC — Regeneration-Cancelling Saturation-Aware Control.

Novel control strategy for regenerative chatter in thin-wall milling with a
single piezo actuator under hard voltage saturation. Three pillars:

P1. EXACT REGENERATION-CHANNEL DECOUPLING FILTER  C(s)
    The chatter loop closes through ONE scalar channel: the tool-point
    deflection y_tool = Dp^T q, driven by the regenerative force
    F_reg = alpha4(t) * Dp * s(t),  s = y_tool(t-tau) - y_tool(t).
    The filter
        C(s) = [Dp^T G(s) Dp] / [Dp^T G(s) H_Pe],   G(s) = modal receptance
    inverts the actuator-to-tool-channel dynamics AT EVERY FREQUENCY, so the
    feedforward u_ff = -C(s) * [alpha4(t) s(t)] cancels the regenerative force
    exactly in the loop channel. Key properties (this plate):
      - the two modes in the loop demand OPPOSITE static gains
        (Dp1/H1 = +85, Dp3/H3 = -257) -> impossible statically, resolved
        dynamically: |C| = 85 @ mode 1 (phase 0), 361 @ mode 3 (phase 179 deg);
      - C(0) ~= 92: cheap where the energy is, large gain only at HF;
      - C is stable and biproper (denominator zeros at ~1052 Hz and ~3816 Hz,
        all in the left half-plane) -> realizable as cascaded biquads (SOS).
    Because the delayed difference s of the tau-PERIODIC forced response is
    ~zero, u_ff only 'sees' the chatter component: the cancellation voltage
    scales with the SUPPRESSED vibration level, not with the chatter growth.

P2. CUTTING-FORCE-AWARE KALMAN PREDICTOR
    The standard LQG observer treats the cutting force as noise; during heavy
    regeneration its estimate of s is too poor to drive P1 (verified: with
    perfect states P1 stabilizes a_p = 4 mm unsaturated; with the blind
    observer it fails at 2.5 mm). The predictor injects the KNOWN force model
        F_hat = ft*alpha3(k)*Dp + alpha4(k)*Dp*(Dp^T(q_hat_delay - q_hat))
    into the observer propagation (matrix G_F), turning it into a predictor of
    the cutting plant.

P3. SATURATION-AWARE PRIORITY ALLOCATION
    The stability-critical u_ff is granted the voltage budget first; the LQG
    damping command only uses the remaining headroom:
        u = clip(u_ff) + clip(u_lqg, [-U..+U] - u_ff).
    The observer is always fed the APPLIED (saturated) voltage (anti-windup).

Baseline LQG design (gains, Kalman) is reused from lqg_controller.LQGController.
"""
import numpy as np
from scipy import signal

from lqg_controller import LQGController


class RCSACController:
    """
    RC-SAC controller. Interface compatible with the package harnesses:
        x_hat, u = ctrl.step(x_hat_prev, u_prev, y_meas)
    (x_hat_prev is accepted for compatibility; internal state is used. Call
    reset() before every new simulation run.)

    Parameters
    ----------
    plate : PlateModel (nominal design model, with D_obs / H_Pe_modal set)
    dt : sample time
    Dp_tool : (n_modes,) modal deflection row at the cutting point
    alpha3_t, alpha4_t : (nstep,) nominal force coefficient histories
    n_tau : regenerative delay in steps (tooth period / dt)
    u_sat : voltage saturation (V)
    ff_scale : cancellation scale factor theta (1.0 nominal)
    w_q_list, w_qd_list, w_r : LQR weight grid (as in LQGController)
    """

    def __init__(self, plate, dt, Dp_tool, alpha3_t, alpha4_t, n_tau,
                 u_sat=150.0, ff_scale=1.0, ft=0.02e-3,
                 w_q_list=None, w_qd_list=None, w_r=1.0,
                 verbose=True):
        self.plate = plate
        self.dt = dt
        self.n_modes = plate.n_modes
        self.Dp = np.asarray(Dp_tool, dtype=float)
        self.a3 = np.asarray(alpha3_t, dtype=float)
        self.a4 = np.asarray(alpha4_t, dtype=float)
        self.n_tau = int(n_tau)
        self.u_sat = float(u_sat)
        self.theta = float(ff_scale)
        self.ft = float(ft)

        # --- P0: baseline LQG (Kalman + LQR grid search) ---
        self.lqg = LQGController(plate, dt=dt, verbose=False)
        self.lqg.optimize_weights(
            w_q_list=w_q_list or [1e10, 1e12, 1e14, 1e16],
            w_qd_list=w_qd_list or [1e4, 1e6, 1e8], w_r=w_r)
        self.lqg.discretize_observer()
        self.ev_cl = self.lqg.ev_cl

        # --- P2: force-aware observer input matrix G_F ---
        n_x = 2 * self.n_modes
        A_obs = self.lqg.A - self.lqg.L_kal @ self.lqg.C
        E = np.zeros((n_x, self.n_modes)); E[self.n_modes:, :] = np.eye(self.n_modes)
        self.G_F = np.linalg.solve(A_obs, (self.lqg.A_obs_d - np.eye(n_x)) @ E)

        # --- P1: exact decoupling filter C(s) = N1/N2, SOS discretization ---
        om = plate.omega_n; zt = plate.zeta_modes
        H = plate.H_Pe_modal

        def pmode(m):
            return np.array([1.0, 2*zt[m]*om[m], om[m]**2])

        N1p = np.zeros(1); N2p = np.zeros(1)
        for m in range(self.n_modes):
            pr = np.array([1.0])
            for j in range(self.n_modes):
                if j != m:
                    pr = np.polymul(pr, pmode(j))
            N1p = np.polyadd(N1p, self.Dp[m]*self.Dp[m]*pr)
            N2p = np.polyadd(N2p, self.Dp[m]*H[m]*pr)
        zc = np.roots(N1p); pc = np.roots(N2p)
        if np.max(pc.real) >= 0:
            raise RuntimeError("decoupling filter unstable (non-minimum-phase "
                               "actuator/tool channel) — RC-SAC not applicable")
        kc = N1p[0] / N2p[0]
        zd, pd_, kd = signal.bilinear_zpk(zc, pc, kc, 1.0/dt)
        self.sos = signal.zpk2sos(zd, pd_, kd)
        self.C_dc = float(np.real(kc*np.prod(-zc)/np.prod(-pc)))
        self.C_hf = float(np.real(kc))

        self.reset()
        if verbose:
            print(f"[RC-SAC] C(0)={self.C_dc:.1f}, C(inf)={self.C_hf:.1f}, "
                  f"filter poles stable, u_sat=±{self.u_sat:.0f} V, "
                  f"theta={self.theta}")

    # ------------------------------------------------------------------
    def reset(self):
        """Reset all internal state (call before each simulation)."""
        self.k = 0
        self.x_hat = np.zeros(2*self.n_modes)
        self.q_buf = np.zeros((self.n_modes, len(self.a4) + 1))
        self.zi = np.zeros((self.sos.shape[0], 2))

    # ------------------------------------------------------------------
    def step(self, x_hat_prev=None, u_prev=0.0, y_meas=0.0, k_step=None):
        """One control step. Returns (x_hat, u)."""
        k = self.k = (k_step if k_step is not None else self.k + 1)
        n = self.n_modes
        km1 = min(k-1, len(self.a4)-1); kk = min(k, len(self.a4)-1)

        # P2: force-aware Kalman prediction (uses its own delayed estimate)
        qd_hat = (self.q_buf[:, k-1-self.n_tau]
                  if k-1-self.n_tau > 0 else np.zeros(n))
        s_prev = self.Dp @ (qd_hat - self.x_hat[:n])
        F_hat = self.ft*self.a3[km1]*self.Dp + self.a4[km1]*self.Dp*s_prev
        self.x_hat = (self.lqg.A_obs_d @ self.x_hat
                      + self.lqg.G_u.flatten()*u_prev
                      + self.G_F @ F_hat
                      + self.lqg.G_y.flatten()*y_meas)
        self.q_buf[:, k] = self.x_hat[:n]

        # baseline damping command
        u_lqg = float(np.squeeze(-self.lqg.K_lqr @ self.x_hat))

        # P1: regeneration-cancelling feedforward through C(z)
        q_del = (self.q_buf[:, k-self.n_tau]
                 if k-self.n_tau > 0 else np.zeros(n))
        s = self.Dp @ (q_del - self.x_hat[:n])
        w_out, self.zi = signal.sosfilt(
            self.sos, np.array([self.theta*self.a4[kk]*s]), zi=self.zi)
        u_ff = float(np.clip(-w_out[0], -self.u_sat, self.u_sat))

        # P3: priority allocation (ff first, damping in the headroom)
        u = u_ff + float(np.clip(u_lqg, -self.u_sat-u_ff, self.u_sat-u_ff))
        return self.x_hat, u
