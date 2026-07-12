"""
piezo_actuator.py
==================
Modèle réaliste d'actionneur piézoélectrique QDA60-200.7 incluant :
   - saturation de tension : ±V_max
   - limitation de slew-rate (vitesse de variation max)
   - dynamique de l'amplificateur de puissance (passe-bas 1er ordre)
   - retard de phase linéaire supplémentaire (approximation 1er ordre du
     déphasage matériau ; ce N'EST PAS un modèle d'hystérésis Bouc-Wen —
     le bloc est exactement linéaire, sans boucle d'hystérésis)
   - retard de phase de l'amplificateur
"""
import numpy as np


class PiezoActuator:
    """
    Modèle réaliste d'un actionneur piézoélectrique.

    Specs typiques pour QDA60-200.7 (source datasheets piezo standards) :
       - Tension max     : ±150 V (mode bipolaire) ou 0-200 V (unipolaire)
       - Slew rate       : ~200-500 V/ms selon amplificateur
       - Bande passante  : 3-10 kHz
       - Hystérésis      : 10-15% à pleine échelle
       - Linéarité       : ±2% sur plage utile
    """

    def __init__(self,
                 V_max: float = 150.0,           # tension max [V]
                 V_min: float = -150.0,          # tension min [V]
                 slew_rate: float = 300e3,       # V/s (300 V/ms)
                 omega_amp: float = 2*np.pi*5e3, # bande passante amplificateur [rad/s]
                 hysteresis_pct: float = 0.10,   # % hystérésis
                 sensor_noise_rms: float = 1e-7, # bruit capteur eddy [m] RMS
                 sensor_delay: float = 50e-6,    # retard capteur [s]
                 dt: float = 5e-5,
                 verbose: bool = True):
        self.V_max = V_max
        self.V_min = V_min
        self.slew_rate = slew_rate
        self.omega_amp = omega_amp
        self.hyst_pct = hysteresis_pct
        self.sensor_noise_rms = sensor_noise_rms
        self.sensor_delay = sensor_delay
        self.dt = dt

        # Discrétisation amplificateur (passe-bas 1er ordre)
        # H(s) = omega / (s + omega) -> ZOH discret
        self.alpha_amp = np.exp(-omega_amp * dt)

        # États internes
        self.u_amp_prev = 0.0       # sortie filtrée précédente
        self.u_actuator_prev = 0.0  # sortie réelle précédente
        self.hyst_state = 0.0       # variable d'état Bouc-Wen
        self.delay_buffer = []      # tampon retard capteur

        # Statistiques
        self.n_saturations = 0
        self.n_slew_limits = 0

        if verbose:
            print(f"[PiezoActuator] Modèle réaliste activé :")
            print(f"   V_max          = ±{V_max} V")
            print(f"   Slew-rate      = {slew_rate*1e-3:.0f} V/ms")
            print(f"   BW amplificateur = {omega_amp/(2*np.pi)*1e-3:.1f} kHz")
            print(f"   Hystérésis     = {hysteresis_pct*100:.0f}%")
            print(f"   Bruit capteur  = {sensor_noise_rms*1e6:.2f} µm RMS")
            print(f"   Retard capteur = {sensor_delay*1e6:.0f} µs")

    def reset(self):
        """Réinitialise les états internes."""
        self.u_amp_prev = 0.0
        self.u_actuator_prev = 0.0
        self.hyst_state = 0.0
        self.delay_buffer = []
        self.n_saturations = 0
        self.n_slew_limits = 0

    # ------------------------------------------------------------------
    def saturate(self, u_cmd: float) -> float:
        """Applique saturation ±V_max."""
        if u_cmd > self.V_max:
            self.n_saturations += 1
            return self.V_max
        elif u_cmd < self.V_min:
            self.n_saturations += 1
            return self.V_min
        return u_cmd

    # ------------------------------------------------------------------
    def slew_limit(self, u_cmd: float) -> float:
        """Limite la vitesse de variation."""
        du_max = self.slew_rate * self.dt
        du = u_cmd - self.u_actuator_prev
        if du > du_max:
            self.n_slew_limits += 1
            return self.u_actuator_prev + du_max
        elif du < -du_max:
            self.n_slew_limits += 1
            return self.u_actuator_prev - du_max
        return u_cmd

    # ------------------------------------------------------------------
    def amplifier_dynamics(self, u_cmd: float) -> float:
        """
        Filtre passe-bas 1er ordre (modèle simplifié de l'amplificateur).
        u_amp[k] = alpha * u_amp[k-1] + (1 - alpha) * u_cmd
        """
        u_amp = self.alpha_amp * self.u_amp_prev + (1 - self.alpha_amp) * u_cmd
        self.u_amp_prev = u_amp
        return u_amp

    # ------------------------------------------------------------------
    def hysteresis(self, u_cmd: float) -> float:
        """
        Retard de phase matériau — approximation LINÉAIRE 1er ordre.

        ATTENTION : sign(x)*|x| = x, donc ce bloc se réduit exactement à
            hyst_state += 0.5*(1-hyst_pct)*(u_cmd - hyst_state)   (lag 1er ordre)
            u_out      = (1-hyst_pct)*u_cmd + hyst_pct*hyst_state
        C'est un filtre linéaire (déphasage ~1° aux modes contrôlés), PAS une
        hystérésis : aucune boucle multivaluée, aucune dépendance à l'amplitude.
        Aucune conclusion de robustesse à l'hystérésis ne peut en être tirée ;
        un vrai modèle Bouc-Wen (ż = ẋ[A - |z|^n(γ + β·sgn(ẋz))]) serait requis.
        """
        self.hyst_state += 0.5 * (1 - self.hyst_pct) * (u_cmd - self.hyst_state)
        u_hyst = (1 - self.hyst_pct) * u_cmd + self.hyst_pct * self.hyst_state
        return u_hyst

    # ------------------------------------------------------------------
    def apply(self, u_cmd: float) -> float:
        """
        Pipeline complet :
            commande -> saturation -> slew-rate -> amplificateur -> hystérésis
        """
        # 1. Saturation
        u = self.saturate(u_cmd)
        # 2. Slew-rate
        u = self.slew_limit(u)
        # 3. Filtre amplificateur
        u = self.amplifier_dynamics(u)
        # 4. Hystérésis (effet matériau piezo)
        u = self.hysteresis(u)

        self.u_actuator_prev = u
        return u

    # ------------------------------------------------------------------
    def measure(self, y_true: float, rng=None) -> float:
        """
        Simule le capteur : ajoute bruit + retard.
        Le retard est implémenté par tampon FIFO.
        """
        n_delay = int(round(self.sensor_delay / self.dt))

        # Ajouter à la fin du tampon
        self.delay_buffer.append(y_true)
        # Sortie retardée de exactement n_delay échantillons :
        # après l'append au pas k, le tampon contient [y[k-n_delay], ..., y[k]] ;
        # on dépile dès que len > n_delay pour renvoyer y[k-n_delay]
        # (n_delay = 0 -> aucune latence ; n_delay = 1 -> un échantillon).
        if len(self.delay_buffer) > n_delay:
            y_delayed = self.delay_buffer.pop(0)
        else:
            y_delayed = self.delay_buffer[0]

        # Ajouter bruit gaussien
        if rng is not None:
            noise = rng.normal(0, self.sensor_noise_rms)
        else:
            noise = np.random.normal(0, self.sensor_noise_rms)

        return y_delayed + noise

    # ------------------------------------------------------------------
    def stats(self):
        """Retourne les statistiques d'utilisation."""
        return dict(
            n_saturations = self.n_saturations,
            n_slew_limits = self.n_slew_limits,
        )
