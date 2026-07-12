"""
newmark_solver.py
=================
Intégration temporelle de l'équation modale NON LINÉAIRE à retard (NDDE)
du fraisage régénératif (Nasiri & Moradi, MSSP 2025, Eqs. 36-37, 44-45) :

    η̈_i + 2(ζ_i + ζ_p,i) ω_i η̇_i + ω_i² η_i + λ_i η_i³
        = (1/M)[ F_cut(t, Δ) · Φ_i(x_outil) + Σ_k H_Pe,i · V_k ]

avec la force de coupe régénérative non linéaire (article Eq. 4/16) et la
séparation outil-pièce (Eq. 6) :

    Δ        = w(outil,t) - w(outil,t-τ) = Dpᵀ(η - η_retard)
    F_cut    = g · [ f_t·a3  - ( a4·Δ + a4_2·Δ² + a4_3·Δ³ ) ]   (+ bord/usure)
    g        = facteur de séparation ∈ {0,1}

Schéma de Newmark (γ=1/2, β=1/4).  L'ossature LINÉAIRE (a4, terme retardé) est
traitée implicitement — IDENTIQUE à la version d'origine du package, donc le
comportement nominal (valeurs principales) est conservé.  Les corrections NON
LINÉAIRES (durcissement λ·η³, coupe a4_2·Δ²+a4_3·Δ³, séparation) sont évaluées
explicitement sur le prédicteur (négligeables aux amplitudes µm, actives au
chatter mm).  L'interface des contrôleurs LQG / DARC est inchangée.

AVERTISSEMENT (grandes amplitudes) : en boucle ouverte au-delà de la limite de
stabilité, l'amplitude du « cycle limite » dépend du paramètre numérique
``chip_sat`` (borne du Δ régénératif, défaut 10·f_t) et du seuil de séparation
``sep_chip``/``sep_amp`` — PAS d'une physique de séparation complète (pas de
régénération multiple 2τ, 3τ...).  Ne tirer AUCUNE conclusion quantitative des
amplitudes de chatter en boucle ouverte ; seuls le seuil de stabilité et les
régimes contrôlés (µm, non-linéarités dormantes) sont exploitables.
"""
import numpy as np


class NewmarkSimulator:
    """
    Simulateur Newmark de la NDDE non linéaire du fraisage (outil mobile,
    retard régénératif, durcissement de Von Kármán, coupe non linéaire,
    séparation, process damping).
    """

    def __init__(self, plate, dt: float, T_end: float,
                 ft: float, tau: float,
                 verbose: bool = True):
        self.plate = plate
        self.dt = dt
        self.T_end = T_end
        self.ft = ft
        self.tau = tau
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
                 alpha4_2_t=None, alpha4_3_t=None,
                 lam_modal=None,
                 sep_chip=None, sep_amp=None, chip_sat=None, edge_force=0.0,
                 sensor_floor: float = 0.0, sensor_noise: float = 0.0,
                 sensor_rng=None,
                 stop_threshold: float = 5e-3,
                 stop_at_time: float = None,
                 progress: bool = True):
        """
        Lance la simulation de la NDDE non linéaire.

        Parameters
        ----------
        alpha3_t, alpha4_t : (nstep,) coefficients linéaires (forçage / régén.)
        kp_idx : (nstep,) indice de position outil dans plate.Dp_array
        controller : LQG / DARC ou None (pas de contrôle)
        piezo : PiezoActuator ou None (modèle idéal)
        alpha4_2_t, alpha4_3_t : (nstep,) coeff. coupe quadratique / cubique
            (article Eq. 4/16).  None → coupe linéaire (ossature du package).
        lam_modal : (n_modes,) coeff. cubiques de Von Kármán.  None → lus depuis
            plate.lam_modal (sinon 0).
        sep_chip : seuil d'épaisseur de copeau pour la séparation [m].  None →
            avance par dent (self.ft).  Aux amplitudes µm la séparation ne se
            déclenche jamais (g=1) → ossature linéaire conservée.
        sep_amp : borne d'amplitude normale pour la séparation [m] (None=off).
        chip_sat : saturation physique de l'épaisseur de copeau régénérative
            utilisée dans le polynôme non linéaire [m].  None → 10·avance.
            Borne la force de coupe au domaine d'engagement physique
            (réduction du copeau / engagement fini, §4.1) : le chatter sature
            en cycle limite borné au lieu de diverger numériquement.
        edge_force : force de bord constante (usure/bleu, Eq. 39) [N] (0=off).
        sensor_floor : RÉSOLUTION du CAPTEUR [m] (zone morte de mesure).  Le
            capteur ne résout pas la vibration sous ce seuil : il renvoie 0 tant
            que |y| < sensor_floor.  Le contrôleur ne « voit » donc pas la
            vibration sous ce niveau et ne peut pas la réduire en dessous -> la
            vibration RÉELLE ne descend pas sous ~sensor_floor (limite physique
            réaliste du capteur).  Appliqué même en capteur idéal.  0 = parfait.
        sensor_noise : bruit additif du capteur [m] (écart-type), 0=off.
        sensor_rng : générateur aléatoire pour le bruit capteur (reproductible).
        """
        n = self.n_modes
        n_x = 2 * n
        nstep = self.nstep
        dt = self.dt
        if sensor_rng is None:
            sensor_rng = np.random.default_rng(0)

        # Coefficients non linéaires (par défaut : ossature linéaire)
        if alpha4_2_t is None:
            alpha4_2_t = np.zeros(nstep)
        if alpha4_3_t is None:
            alpha4_3_t = np.zeros(nstep)
        if lam_modal is None:
            lam_modal = np.asarray(getattr(self.plate, 'lam_modal',
                                           np.zeros(n)), dtype=float)
        else:
            lam_modal = np.asarray(lam_modal, dtype=float)
        if sep_chip is None:
            sep_chip = self.ft                      # séparation ≈ avance/dent
        if chip_sat is None:
            chip_sat = 10.0 * self.ft               # engagement physique ~ 10·avance

        qm   = np.zeros((n, nstep))
        qmd  = np.zeros((n, nstep))
        qmdd = np.zeros((n, nstep))
        x_hat = np.zeros((n_x, nstep))
        y_mill = np.zeros(nstep)
        y_meas = np.zeros(nstep)            # mesure bruitée + retardée
        u_cmd  = np.zeros(nstep)            # commande LQR (avant piezo)
        u_real = np.zeros(nstep)            # tension réellement appliquée
        sep_flag = np.zeros(nstep)          # séparation (1 = hors coupe)

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
            a4_2 = alpha4_2_t[k]
            a4_3 = alpha4_3_t[k]

            # Mesure : idéale ou réaliste (bruit + retard), puis plancher capteur
            y_true = D_obs @ qm[:, k-1]
            if piezo is not None:
                y_obs_now = piezo.measure(y_true, rng=rng)
            else:
                y_obs_now = y_true
            # Réglage du CAPTEUR : bruit additif éventuel, puis résolution finie
            # (zone morte). Le contrôleur ne voit pas (et ne corrige pas) la
            # vibration sous la résolution -> la vibration réelle ne descend pas
            # sous ~sensor_floor.
            if sensor_noise > 0.0:
                y_obs_now = y_obs_now + sensor_rng.normal(0.0, sensor_noise)
            if sensor_floor > 0.0 and abs(y_obs_now) < sensor_floor:
                y_obs_now = 0.0
            y_meas[k] = y_obs_now

            # Observateur + commande (interface contrôleur INCHANGÉE)
            if controller is not None:
                if hasattr(controller, 'enable_gs'):
                    x_p_now_phys = self.plate.xp_array[kp]
                    step_out = controller.step(x_hat[:, k-1],
                                                u_real_prev, y_obs_now,
                                                x_p_now=x_p_now_phys)
                elif controller.__class__.__name__ == 'CALOR_FF_Controller':
                    step_out = controller.step(x_hat[:, k-1],
                                                u_real_prev, y_obs_now,
                                                k_step=k)
                elif controller.__class__.__name__ in ('NRACC_Controller', 'NRACC_v2_Controller', 'NRACC_v3_Controller', 'NRACC_Enhanced_Controller', 'DARC_MPC_Controller', 'DARC_MPC_v2_Controller', 'DARC_MPC_v3_Controller', 'DARCController'):
                    step_out = controller.step(x_hat[:, k-1],
                                                u_real_prev, y_obs_now,
                                                k_step=k)
                elif controller.__class__.__name__ == 'NRACC_RU_Controller':
                    step_out = controller.step(x_hat[:, k-1],
                                                u_real_prev, y_obs_now,
                                                k_step=k)
                else:
                    step_out = controller.step(x_hat[:, k-1],
                                                u_real_prev, y_obs_now)
                if len(step_out) == 3:
                    x_hat[:, k], u, _ = step_out
                else:
                    x_hat[:, k], u = step_out
            else:
                u = 0.0
            u_cmd[k] = u

            # Application au piezo (saturation, slew, ampli, hystérésis)
            if piezo is not None and controller is not None:
                u_actual = piezo.apply(u)
            else:
                u_actual = u
            u_real[k] = u_actual

            # Etat retardé (régénératif)
            if k - self.n_tau > 0:
                q_delay = qm[:, k - self.n_tau]
            else:
                q_delay = np.zeros(n)

            # --- Prédicteurs de Newmark ---
            qd_pred = qmd[:, k-1] + (1-gNM)*dt * qmdd[:, k-1]
            q_pred  = qm [:, k-1] + dt*qmd[:, k-1] + (0.5-bNM)*dt**2 * qmdd[:, k-1]

            # --- Déplacement régénératif Δ et séparation (sur prédicteur) ---
            delta = float(Dp_now @ (q_pred - q_delay))    # Δ = Dpᵀ(η - η_retard)
            w_tool = float(Dp_now @ q_pred)               # déflexion à l'outil
            g_sep = 1.0
            if delta < -sep_chip:                          # copeau négatif → sortie
                g_sep = 0.0
            elif sep_amp is not None and abs(w_tool) > sep_amp:
                g_sep = 0.0
            sep_flag[k] = 1.0 - g_sep

            # --- Raideur effective (ossature linéaire implicite, gée par g) ---
            K_eff = Kp_modal + g_sep * a4 * DpT_Dp_now

            # --- Forces ---
            # Coupe linéaire (forçage moyen + régénératif retardé) — IDENTIQUE
            F_lin = g_sep * (self.ft * a3 * Dp_now + a4 * DpT_Dp_now @ q_delay)
            # Coupe NON LINÉAIRE (quadratique + cubique en Δ) — explicite.
            # Δ borné à l'engagement physique (chip_sat) : la non-linéarité
            # sature le chatter en cycle limite (article §4.1) au lieu de
            # diverger.  Aux amplitudes µm, delta_nl = delta (effet nul).
            delta_nl = max(-chip_sat, min(chip_sat, delta))
            F_cut_nl = -g_sep * Dp_now * (a4_2 * delta_nl**2 + a4_3 * delta_nl**3)
            # Bord / usure (Eq. 39), force statique de coupe
            F_edge = g_sep * edge_force * Dp_now
            # Durcissement structurel de Von Kármán -λ·η³ — explicite
            F_vk = -lam_modal * q_pred**3
            # Actionneur piézo
            F_pe = H_Pe_modal * u_actual

            F_now = F_lin + F_cut_nl + F_edge + F_vk + F_pe

            # --- Résolution de Newmark implicite ---
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
                    x_hat=x_hat, sep_flag=sep_flag, diverged_at=diverged_at,
                    stop_idx=(diverged_at if diverged_at > 0 else nstep - 1),
                    piezo_stats=(piezo.stats() if piezo is not None else None))
