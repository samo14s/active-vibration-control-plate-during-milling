"""
newmark_solver.py
=================
Intégration temporelle du modèle de fraisage avec retard régénératif :
    Mp*qpdd + Cp*qpd + (Kp + a4*Dp²)*qp - a4*Dp²*qp(t-τ) = ft*a3*Dp + Hpe*u

Schéma de Newmark implicite (gamma=1/2, beta=1/4).
Le terme régénératif utilise les états retardés stockés dans l'historique.

La commande est saturée par le simulateur lui-même à +/- v_max avant d'être
appliquée à la plaque ET avant d'être renvoyée au correcteur au pas suivant
(u_prev), de sorte que l'observateur d'un correcteur voit toujours la tension
réellement appliquée.
"""
import numpy as np


class NewmarkSimulator:
    """
    Simulateur Newmark adapté au fraisage avec retard et outil mobile.

    Parameters
    ----------
    v_max : borne de l'amplificateur, en volts. La commande rendue par le
        correcteur y est écrêtée avant application. `SimBase` passe
        `simulation_base.V_MAX` ; la valeur par défaut n'est qu'un garde-fou
        pour un usage direct de cette classe. `None` désactive la saturation
        (à n'utiliser que pour un diagnostic, jamais pour comparer des lois
        de commande).
    """

    def __init__(self, plate, dt: float, T_end: float,
                 ft: float, tau: float,
                 verbose: bool = True,
                 v_max: float = 150.0):
        self.plate = plate
        self.dt = dt
        self.T_end = T_end
        self.ft = ft
        self.tau = tau
        self.v_max = v_max
        self.t_vec = np.arange(0, T_end + dt/2, dt)
        self.nstep = len(self.t_vec)
        self.n_tau = int(np.round(tau / dt))
        self.n_modes = plate.n_modes
        self.verbose = verbose

        if verbose:
            print(f"[Newmark] dt = {dt:.2e}, T_end = {T_end:.3f}, "
                  f"nstep = {self.nstep}")
            print(f"[Newmark] tau = {tau*1e3:.3f} ms = {self.n_tau} pas")

    # ---------------------------------------------------------------
    def simulate(self, alpha3_t, alpha4_t,
                 kp_idx, controller=None,
                 piezo=None, rng=None,
                 stop_threshold: float = 5e-3,
                 stop_at_time: float = None,
                 progress: bool = True):
        """
        Lance la simulation.

        Parameters
        ----------
        alpha3_t, alpha4_t : (nstep,) coefficients de force
        kp_idx : (nstep,) indice de position outil dans plate.Dp_array
        controller : LQGController ou None (pas de contrôle)
        piezo : PiezoActuator ou None (modèle idéal)
        rng : np.random.Generator pour bruit reproductible
        stop_threshold : interrompt si |y_p| > ce seuil (sécurité)
        stop_at_time : interrompt à ce temps (utilisé pour Sim 1 sans contrôle)
        """
        n = self.n_modes
        n_x = 2 * n
        nstep = self.nstep
        dt = self.dt

        qm   = np.zeros((n, nstep))
        qmd  = np.zeros((n, nstep))
        qmdd = np.zeros((n, nstep))
        x_hat = np.zeros((n_x, nstep))
        y_mill = np.zeros(nstep)
        y_meas = np.zeros(nstep)            # mesure bruitée + retardée
        u_cmd  = np.zeros(nstep)            # commande LQR (avant piezo)
        u_real = np.zeros(nstep)            # tension réellement appliquée

        Mp = self.plate.Mp;  Kp_modal = self.plate.Kp;  Cp_modal = self.plate.Cp
        H_Pe_modal = (self.plate.H_Pe_modal
                      if controller is not None
                      else np.zeros(n))
        D_obs = self.plate.D_obs

        gNM, bNM = 0.5, 0.25
        u_real_prev = 0.0
        diverged_at = 0

        if piezo is not None:
            piezo.reset()
        if rng is None:
            rng = np.random.default_rng(0)

        progress_step = max(1, nstep // 10)

        for k in range(1, nstep):
            kp = int(kp_idx[k])
            Dp_now, DpT_Dp_now = self.plate.get_Dp_at(kp)
            a3 = alpha3_t[k]
            a4 = alpha4_t[k]

            # Mesure : idéale ou réaliste (bruit + retard)
            y_true = D_obs @ qm[:, k-1]
            if piezo is not None:
                y_obs_now = piezo.measure(y_true, rng=rng)
            else:
                y_obs_now = y_true
            y_meas[k] = y_obs_now

            # Observateur + commande LQR
            if controller is not None:
                # APP-LQG accepte position outil pour gain scheduling
                if hasattr(controller, 'enable_gs'):
                    x_p_now_phys = self.plate.xp_array[kp]
                    step_out = controller.step(x_hat[:, k-1],
                                                u_real_prev, y_obs_now,
                                                x_p_now=x_p_now_phys)
                # CALOR-FF nécessite k_step pour le feedforward périodique
                elif controller.__class__.__name__ == 'CALOR_FF_Controller':
                    step_out = controller.step(x_hat[:, k-1],
                                                u_real_prev, y_obs_now,
                                                k_step=k)
                # NRACC nécessite aussi k_step
                elif controller.__class__.__name__ in ('NRACC_Controller', 'NRACC_v2_Controller', 'NRACC_v3_Controller', 'NRACC_Enhanced_Controller', 'DARC_MPC_Controller', 'DARC_MPC_v2_Controller', 'DARC_MPC_v3_Controller'):
                    step_out = controller.step(x_hat[:, k-1],
                                                u_real_prev, y_obs_now,
                                                k_step=k)
                # NRACC-RU nécessite aussi k_step
                elif controller.__class__.__name__ == 'NRACC_RU_Controller':
                    step_out = controller.step(x_hat[:, k-1],
                                                u_real_prev, y_obs_now,
                                                k_step=k)
                else:
                    step_out = controller.step(x_hat[:, k-1],
                                                u_real_prev, y_obs_now)
                # SMCController retourne (x_hat, u, s) ; LQG retourne (x_hat, u)
                if len(step_out) == 3:
                    x_hat[:, k], u, _ = step_out
                else:
                    x_hat[:, k], u = step_out
            else:
                u = 0.0
            u_cmd[k] = u

            # Saturation amplificateur : la borne du banc s'applique a TOUT
            # correcteur, qu'il sature en interne ou non. u_cmd garde la
            # commande brute, u_real la tension effectivement appliquee.
            if self.v_max is not None:
                u = float(np.clip(u, -self.v_max, self.v_max))

            # Application au piezo (saturation, slew, ampli, hystérésis)
            if piezo is not None and controller is not None:
                u_actual = piezo.apply(u)
            else:
                u_actual = u
            u_real[k] = u_actual

            # Etat retardé
            if k - self.n_tau > 0:
                q_delay = qm[:, k - self.n_tau]
            else:
                q_delay = np.zeros(n)

            # Forces (la tension RÉELLE est utilisée)
            K_eff = Kp_modal + a4 * DpT_Dp_now
            F_now = self.ft * a3 * Dp_now \
                  + a4 * DpT_Dp_now @ q_delay \
                  + H_Pe_modal * u_actual

            # Newmark implicite
            qd_pred = qmd[:, k-1] + (1-gNM)*dt * qmdd[:, k-1]
            q_pred  = qm [:, k-1] + dt*qmd[:, k-1] + (0.5-bNM)*dt**2 * qmdd[:, k-1]
            S_eff = Mp + gNM*dt*Cp_modal + bNM*dt**2 * K_eff
            rhs = F_now - Cp_modal @ qd_pred - K_eff @ q_pred

            qmdd[:, k] = np.linalg.solve(S_eff, rhs)
            qmd [:, k] = qd_pred + gNM*dt * qmdd[:, k]
            qm  [:, k] = q_pred  + bNM*dt**2 * qmdd[:, k]

            y_mill[k] = Dp_now @ qm[:, k]
            u_real_prev = u_actual

            if progress and (k % progress_step == 0):
                pct = round(100*k/nstep)
                msg = (f"   {pct:3d}%  t={self.t_vec[k]:.3f}s  "
                       f"|y|={abs(y_mill[k])*1e6:7.1f}um  "
                       f"|u|={abs(u_real[k]):6.1f}V")
                if piezo is not None:
                    msg += f"  sat={piezo.n_saturations}, slew={piezo.n_slew_limits}"
                print(msg)

            # Conditions d'arret
            if stop_at_time is not None and self.t_vec[k] >= stop_at_time:
                if self.verbose:
                    print(f"   Arret programme a t={self.t_vec[k]:.4f}s, "
                          f"|y|={abs(y_mill[k])*1e6:.0f}um")
                diverged_at = k
                break
            if abs(y_mill[k]) > stop_threshold:
                if self.verbose:
                    print(f"   ! Divergence a t={self.t_vec[k]:.4f}s, "
                          f"|y|={abs(y_mill[k])*1e6:.0f}um")
                diverged_at = k
                break

        return dict(t=self.t_vec, y=y_mill, y_meas=y_meas,
                    u=u_real, u_cmd=u_cmd,
                    qm=qm, qmd=qmd, qmdd=qmdd,
                    x_hat=x_hat, diverged_at=diverged_at,
                    stop_idx=(diverged_at if diverged_at > 0 else nstep - 1),
                    piezo_stats=(piezo.stats() if piezo is not None else None))
