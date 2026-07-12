"""Model and process parameters for the thin-walled milling simulation.

All values follow Wang, Sun & Sun, "Adaptive Control Strategy for High-Speed
Milling of Thin-Walled Components", ASME Open Journal of Engineering, Vol. 5,
051006 (2026), DOI 10.1115/1.4070743 — in particular Table 1 (cutting
parameter ranges) and Sec. 4.1 (experimental setup).

Units are SI internally (m, s, N, rad).  Human-facing values (mm, rpm,
mm/min) are converted at the boundaries and marked in the field names.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Tool / cutting mechanics (Sec. 4.1: 4-flute, 12 mm carbide end mill, TiAlN)
# ---------------------------------------------------------------------------
@dataclass
class ToolParams:
    n_teeth: int = 4                 # N in Eqs. (3)-(4)
    diameter_m: float = 12e-3        # 12 mm end mill
    helix_deg: float = 30.0          # 30 deg helix (not modelled axially)

    # Specific cutting-force coefficients for Al 7075-T6 (Altintas-type
    # mechanistic model, Eq. (1)):  Ft = Kt * h * ap,  Fr = Kr * Ft.
    Kt: float = 796e6                # tangential coefficient [N/m^2]
    Kr: float = 0.212                # radial/tangential force ratio [-]

    # Process damping, Eq. (13): Cp = Kp * V0 / V
    Kp: float = 42.0                 # process damping constant [N s/m]
    V0: float = 2.0                  # reference cutting velocity [m/s]

    @property
    def radius_m(self) -> float:
        return 0.5 * self.diameter_m


# ---------------------------------------------------------------------------
# Radial engagement (Sec. 4.1: radial immersion fixed at 25 % of diameter,
# down-milling of the wall flank).
# ---------------------------------------------------------------------------
@dataclass
class EngagementParams:
    ae_m: float = 3.0e-3             # radial depth of cut, 3.0 mm (25 %)
    down_milling: bool = True

    def entry_exit_angles(self, tool: ToolParams) -> tuple[float, float]:
        """Return (phi_start, phi_exit) in radians for the window g(phi_j)."""
        ratio = self.ae_m / tool.diameter_m
        ratio = min(max(ratio, 1e-6), 1.0)
        if self.down_milling:
            phi_st = math.acos(2.0 * ratio - 1.0)
            phi_ex = math.pi
        else:
            phi_st = 0.0
            phi_ex = math.acos(1.0 - 2.0 * ratio)
        return phi_st, phi_ex


# ---------------------------------------------------------------------------
# Structural dynamics (Sec. 2).  The coupled tool-workpiece system is
# described by modal parameters; the thin wall is compliant normal to its
# face (y) while the tool contributes symmetric x / y modes.
#
# Time variation, Eqs. (6)-(7):
#   w_n,i(t) = w0_n,i * sqrt(1 - alpha_i * Vr(t))
#   zeta_i(t) = zeta0_i * (1 + beta_i * Vr(t))
# where Vr is the removed-volume fraction of the wall-thinning pass.
# ---------------------------------------------------------------------------
@dataclass
class Mode:
    f0_hz: float                     # initial natural frequency
    zeta0: float                     # initial damping ratio
    mass_kg: float                   # modal mass (assumed constant)
    alpha: float = 0.0               # frequency sensitivity to Vr, Eq. (6)
    beta: float = 0.0                # damping sensitivity to Vr, Eq. (7)
    direction: str = "y"             # 'x' or 'y'
    structure: str = "tool"          # 'tool' or 'workpiece'

    def wn(self, Vr: float) -> float:
        s = 1.0 - self.alpha * Vr
        return 2.0 * math.pi * self.f0_hz * math.sqrt(max(s, 0.05))

    def zeta(self, Vr: float) -> float:
        return self.zeta0 * (1.0 + self.beta * Vr)

    def k(self, Vr: float) -> float:
        w = self.wn(Vr)
        return self.mass_kg * w * w

    def c(self, Vr: float) -> float:
        return 2.0 * self.zeta(Vr) * self.mass_kg * self.wn(Vr)


def default_modes() -> list[Mode]:
    """Modal set calibrated so that

    * the dominant wall mode falls through the 800-1200 Hz chatter band
      reported in Sec. 4.2 as material is removed, and
    * the offline-optimised baseline point (12 000 rpm, ap = 4 mm) is stable
      at the initial geometry but crosses the shrinking stability boundary
      mid-pass (the central failure mode of fixed-parameter machining).
    """
    return [
        # Thin-wall bending modes (compliant in y only).
        Mode(f0_hz=1580.0, zeta0=0.028, mass_kg=0.085, alpha=0.55, beta=0.90,
             direction="y", structure="workpiece"),
        Mode(f0_hz=2600.0, zeta0=0.022, mass_kg=0.240, alpha=0.35, beta=0.50,
             direction="y", structure="workpiece"),
        # Tool/spindle modes (geometry-independent).
        Mode(f0_hz=950.0, zeta0=0.035, mass_kg=0.420, alpha=0.0, beta=0.0,
             direction="x", structure="tool"),
        Mode(f0_hz=980.0, zeta0=0.035, mass_kg=0.400, alpha=0.0, beta=0.0,
             direction="y", structure="tool"),
    ]


# ---------------------------------------------------------------------------
# Cutting-parameter envelope, Table 1.
# ---------------------------------------------------------------------------
@dataclass
class ParameterLimits:
    n_min_rpm: float = 6000.0
    n_max_rpm: float = 20000.0
    vf_min_mm_min: float = 800.0
    vf_max_mm_min: float = 3600.0
    ap_min_mm: float = 1.0
    ap_max_mm: float = 10.0
    n_dot_max_rpm_s: float = 20000.0  # spindle acceleration limit, Eq. (26)
                                      # (Sec. 4.1: rapid accel/decel spindle
                                      #  sized for SSV chatter suppression)
    vf_dot_max_mm_min_s: float = 2400.0


# ---------------------------------------------------------------------------
# Adaptive controller settings (Table 1 control parameters + Sec. 3.3).
# ---------------------------------------------------------------------------
@dataclass
class ControllerParams:
    update_rate_hz: float = 100.0    # adaptation update rate (Table 1)
    eta: float = 0.80                # stability margin factor, Eq. (17)
    alpha_n0: float = 0.30           # nominal spindle learning rate, Eq. (18)
    beta1: float = 3.0               # gain-scheduling wrt instability, Eq. (25)
    beta2: float = 40.0              # gain-scheduling wrt uncertainty, Eq. (25)
    Q_ref_cm3_min: float = 27.0      # reference MRR for the feed MPC, Eq. (20)
    n_slew_rpm_s: float = 3000.0     # controller's own speed-ramp policy
                                     # (smooth strategic transitions)
    ft_min_mm: float = 0.030         # feed/tooth floor (chip-thinning /
                                     # rubbing avoidance)
    Np: int = 10                     # MPC prediction horizon, Eq. (20)
    lambda_f: float = 0.15           # feed-move penalty, Eq. (20)
    gamma_a: float = 0.35            # depth-of-cut adaptation gain, Eq. (21)
    V_target_um: float = 22.0        # target RMS vibration [um], Eq. (21)
                                     # (keeps operation under the 25 um
                                     #  stability threshold of Fig. 3)
    w1: float = 1.0                  # objective weights, Eq. (16)
    w2: float = 40.0
    w3: float = 5.0
    w4: float = 8.0
    ft_max_mm: float = 0.055         # feed/tooth cap (surface term D_surf)
    Q_cap_cm3_min: float = 31.0      # productivity ceiling
    dither_frac: float = 0.002       # identification speed perturbation
    dither_hz: float = 1.3           # (Sec. 3.1: "brief perturbations in
                                     #  spindle speed to excite the system")


# ---------------------------------------------------------------------------
# Baseline strategies (Sec. 4.1).
# ---------------------------------------------------------------------------
@dataclass
class BaselineParams:
    # Offline-optimised fixed parameters (stability lobes at initial geometry).
    n_rpm: float = 12000.0
    vf_mm_min: float = 1800.0
    ap_mm: float = 4.0
    # Spindle-speed-variation baseline: +/-10 % sinusoid at 5 Hz.
    ssv_amplitude: float = 0.10
    ssv_freq_hz: float = 5.0
    # "Manual conservative reduction when visible chatter occurred":
    # operator reacts with a delay, cutting feed/depth in steps.
    operator_rms_threshold_um: float = 90.0
    operator_reaction_s: float = 2.0
    operator_feed_factor: float = 0.82
    operator_ap_step_mm: float = 0.5
    operator_ap_floor_mm: float = 1.5
    operator_vf_floor_mm_min: float = 1200.0


# ---------------------------------------------------------------------------
# Simulation configuration.
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    dt: float = 1.0e-5               # integration step (100 kHz)
    duration_s: float = 75.0         # hard ceiling on the (compressed) pass;
                                     # a run normally ends when Vr reaches 1
    # Volume to be removed in the (compressed) pass.  Scaled so that the
    # nominal baseline MRR (~21.6 cm^3/min) completes the pass in ~42 s,
    # sweeping the modal parameters over the same excursion the physical
    # 1.0-3.0 mm final-thickness walls experience (Sec. 4.1).
    removal_volume_cm3: float = 15.0
    sensor_noise_um: float = 0.4     # displacement measurement noise (1 sigma)
    accel_sample_hz: float = 10000.0 # vibration channel (Sec. 3.1: 10 kHz)
    force_sample_hz: float = 5000.0  # force channel (Sec. 3.1: 5 kHz)
    rms_window_s: float = 0.1        # 0.1 s sliding RMS (Sec. 4.1)
    seed: int = 42

    @property
    def n_steps(self) -> int:
        return int(round(self.duration_s / self.dt))


# ---------------------------------------------------------------------------
# Surface-quality correlation (Sec. 3.2: "surface roughness indicators are
# derived from the high-frequency components of the cutting force signals,
# utilizing empirical correlations calibrated for the specific tool-workpiece
# combination").  Ra combines a kinematic feed-mark term with a vibration
# term active above a floor amplitude.
# ---------------------------------------------------------------------------
@dataclass
class RoughnessModel:
    ra_base_um: float = 0.55
    feed_coeff_um_per_mm: float = 8.5    # contribution of feed per tooth
    vib_coeff: float = 0.035             # um Ra per um RMS above the floor
    vib_floor_um: float = 16.0
    ra_cap_um: float = 5.0

    def ra_um(self, ft_mm: float, rms_um: float) -> float:
        ra = (self.ra_base_um
              + self.feed_coeff_um_per_mm * ft_mm
              + self.vib_coeff * max(0.0, rms_um - self.vib_floor_um))
        return min(ra, self.ra_cap_um)


@dataclass
class PaperSetup:
    """Aggregate of every parameter group used by the study."""
    tool: ToolParams = field(default_factory=ToolParams)
    engagement: EngagementParams = field(default_factory=EngagementParams)
    modes: list[Mode] = field(default_factory=default_modes)
    limits: ParameterLimits = field(default_factory=ParameterLimits)
    controller: ControllerParams = field(default_factory=ControllerParams)
    baseline: BaselineParams = field(default_factory=BaselineParams)
    sim: SimConfig = field(default_factory=SimConfig)
    roughness: RoughnessModel = field(default_factory=RoughnessModel)
