"""
newmark_nonlinear_solver.py
===========================
Intégration temporelle GÉOMÉTRIQUEMENT NON-LINÉAIRE (von Kármán) du modèle
de fraisage avec retard régénératif :

    Mp q̈ + Cp q̇ + (Kp + a4·Dp²) q + G_vK(q) − a4·Dp²·q(t−τ)
        = ft·a3·Dp + Hpe·u

où G_vK(q) = Σ_ijk Γ_rijk q_i q_j q_k est la force modale cubique de
von Kármán fournie par VonKarmanROM.

Schéma de Newmark implicite (γ=1/2, β=1/4) avec itération de
Newton-Raphson interne sur la raideur non linéaire.  Lorsque le tenseur
Γ est nul (ou rom=None) le schéma se réduit EXACTEMENT au solveur
linéaire d'origine (une seule "itération" triviale).

L'interface (controller.step, piezo, kp_idx …) est identique à
NewmarkSimulator : le contrôleur — conçu sur le modèle LINÉAIRE — est
ainsi appliqué tel quel à la dynamique non linéaire.
"""
import numpy as np


class NewmarkNonlinearSimulator:

    def __init__(self, plate, dt, T_end, ft, tau, rom=None,
                 newton_tol=1e-10, newton_max=12, verbose=True):
        self.plate = plate
        self.dt = dt
        self.T_end = T_end
        self.ft = ft
        self.tau = tau
        self.rom = rom
        self.newton_tol = newton_tol
        self.newton_max = newton_max
        self.t_vec = np.arange(0, T_end + dt/2, dt)
        self.nstep = len(self.t_vec)
        self.n_tau = int(np.round(tau/dt))
        self.n_modes = plate.n_modes
        self.verbose = verbose
        if verbose:
            print(f"[Newmark-NL] dt={dt:.2e}, nstep={self.nstep}, "
                  f"n_tau={self.n_tau}, "
                  f"vK={'on' if rom is not None else 'off (linear)'}")

    # ----------------------------------------------------------------
    def simulate(self, alpha3_t, alpha4_t, kp_idx, controller=None,
                 piezo=None, rng=None, stop_threshold=5e-3,
                 stop_at_time=None, progress=True):
        n = self.n_modes
        n_x = 2*n
        nstep = self.nstep
        dt = self.dt
        gNM, bNM = 0.5, 0.25

        qm = np.zeros((n, nstep))
        qmd = np.zeros((n, nstep))
        qmdd = np.zeros((n, nstep))
        x_hat = np.zeros((n_x, nstep))
        y_mill = np.zeros(nstep)
        y_meas = np.zeros(nstep)
        u_cmd = np.zeros(nstep)
        u_real = np.zeros(nstep)
        n_newton = np.zeros(nstep, dtype=int)

        Mp = self.plate.Mp
        Kp = self.plate.Kp
        Cp = self.plate.Cp
        H_Pe = (self.plate.H_Pe_modal if controller is not None
                else np.zeros(n))
        D_obs = self.plate.D_obs
        rom = self.rom

        if piezo is not None:
            piezo.reset()
        if rng is None:
            rng = np.random.default_rng(0)

        u_real_prev = 0.0
        diverged_at = 0
        progress_step = max(1, nstep//10)

        for k in range(1, nstep):
            kp = int(kp_idx[k])
            Dp_now, DpT_Dp_now = self.plate.get_Dp_at(kp)
            a3 = alpha3_t[k]
            a4 = alpha4_t[k]

            y_true = D_obs @ qm[:, k-1]
            y_obs_now = (piezo.measure(y_true, rng=rng)
                         if piezo is not None else y_true)
            y_meas[k] = y_obs_now

            if controller is not None:
                cls = controller.__class__.__name__
                if hasattr(controller, 'enable_gs'):
                    xp = self.plate.xp_array[kp]
                    step_out = controller.step(x_hat[:, k-1], u_real_prev,
                                               y_obs_now, x_p_now=xp)
                elif cls in ('CALOR_FF_Controller', 'NRACC_Controller',
                             'NRACC_v2_Controller', 'NRACC_v3_Controller',
                             'NRACC_Enhanced_Controller', 'NRACC_RU_Controller',
                             'DARC_MPC_Controller', 'DARC_MPC_v2_Controller',
                             'DARC_MPC_v3_Controller'):
                    step_out = controller.step(x_hat[:, k-1], u_real_prev,
                                               y_obs_now, k_step=k)
                else:
                    step_out = controller.step(x_hat[:, k-1], u_real_prev,
                                               y_obs_now)
                if len(step_out) == 3:
                    x_hat[:, k], u, _ = step_out
                else:
                    x_hat[:, k], u = step_out
            else:
                u = 0.0
            u_cmd[k] = u

            u_actual = (piezo.apply(u) if (piezo is not None
                        and controller is not None) else u)
            u_real[k] = u_actual

            q_delay = (qm[:, k - self.n_tau] if k - self.n_tau > 0
                       else np.zeros(n))

            K_eff = Kp + a4*DpT_Dp_now
            F_now = self.ft*a3*Dp_now + a4*DpT_Dp_now @ q_delay + H_Pe*u_actual

            # Newmark predictors
            qd_pred = qmd[:, k-1] + (1-gNM)*dt*qmdd[:, k-1]
            q_pred = qm[:, k-1] + dt*qmd[:, k-1] + (0.5-bNM)*dt**2*qmdd[:, k-1]

            # Newton-Raphson on acceleration a_new
            a_new = qmdd[:, k-1].copy()
            S_lin = Mp + gNM*dt*Cp + bNM*dt**2*K_eff
            n_it = 0
            for it in range(self.newton_max):
                q_k = q_pred + bNM*dt**2 * a_new
                qd_k = qd_pred + gNM*dt * a_new
                g_vk = rom.force(q_k) if rom is not None else 0.0
                R = (Mp @ a_new + Cp @ qd_k + K_eff @ q_k
                     + (g_vk if rom is not None else 0.0) - F_now)
                if rom is not None:
                    Jt = S_lin + bNM*dt**2 * rom.jacobian(q_k)
                else:
                    Jt = S_lin
                da = np.linalg.solve(Jt, -R)
                a_new += da
                n_it = it + 1
                if np.linalg.norm(da) <= self.newton_tol*(1 + np.linalg.norm(a_new)):
                    break

            qmdd[:, k] = a_new
            qmd[:, k] = qd_pred + gNM*dt*a_new
            qm[:, k] = q_pred + bNM*dt**2*a_new
            n_newton[k] = n_it

            y_mill[k] = Dp_now @ qm[:, k]
            u_real_prev = u_actual

            if progress and (k % progress_step == 0):
                print(f"   {round(100*k/nstep):3d}%  t={self.t_vec[k]:.3f}s  "
                      f"|y|={abs(y_mill[k])*1e6:8.1f}um  "
                      f"|u|={abs(u_real[k]):6.1f}V  nit={n_it}")

            if stop_at_time is not None and self.t_vec[k] >= stop_at_time:
                diverged_at = k
                break
            if abs(y_mill[k]) > stop_threshold:
                if self.verbose:
                    print(f"   ! divergence a t={self.t_vec[k]:.4f}s, "
                          f"|y|={abs(y_mill[k])*1e6:.0f}um")
                diverged_at = k
                break

        return dict(t=self.t_vec, y=y_mill, y_meas=y_meas,
                    u=u_real, u_cmd=u_cmd, qm=qm, qmd=qmd, qmdd=qmdd,
                    x_hat=x_hat, diverged_at=diverged_at,
                    n_newton=n_newton,
                    stop_idx=(diverged_at if diverged_at > 0 else nstep-1))
