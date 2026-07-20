"""Physical and process parameters.

All values follow the experimental setup of Du, Liu, Dai, Long,
"Robust combined time delay control for milling chatter suppression of
flexible workpieces", Int. J. Mech. Sci. 274 (2024) 109257, so that
simulation results are directly comparable with their published
measurements. SI units throughout unless noted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PlateParams:
    """Thin-walled cantilever plate, AL6061 (ref. Table 1)."""

    lp: float = 0.100          # length in feed direction x [m]
    hp: float = 0.080          # height (cantilever span) z [m]
    bp: float = 0.004          # thickness [m]
    E: float = 69.0e9          # Young's modulus [Pa]
    nu: float = 0.33           # Poisson ratio
    rho: float = 2830.0        # density [kg/m^3]
    kappa: float = 5.0 / 6.0   # Mindlin shear correction factor
    # Measured modal damping ratios, modes 1..5 (ref. Table 4)
    zeta: tuple = (0.0031, 0.0017, 0.0027, 0.0056, 0.0035)
    # Damping ratio applied to modes beyond the measured five
    zeta_high: float = 0.0035

    # Reference values for model validation (ref. Table 4)
    f_measured: tuple = (540.0, 1068.0, 2787.0, 3351.0, 4122.0)   # [Hz]
    f_theoretical: tuple = (537.0, 1101.0, 2805.0, 3423.0, 4254.0)  # [Hz]


@dataclass(frozen=True)
class PiezoParams:
    """Surface-bonded patch actuator QDA60-20-0.7 (ref. Table 2).

    Placed with its lower-left corner at the plate's clamped lower-left
    corner region, per the placement optimization cited in the reference.
    """

    length: float = 0.060      # along x [m]
    width: float = 0.020       # along z [m]
    h_pa: float = 0.0007       # thickness [m]
    E: float = 63.0e9          # Young's modulus [Pa]
    nu: float = 0.35           # Poisson ratio
    d31: float = -175.0e-12    # strain constant [m/V]
    x0: float = 0.0            # patch lower-left corner x [m]
    z0: float = 0.0            # patch lower-left corner z [m]
    v_max: float = 150.0       # amplifier voltage limit (PI E-420, gain 100) [V]


@dataclass(frozen=True)
class ToolParams:
    """Milling tool and cutting coefficients (ref. Table 3)."""

    n_teeth: int = 3
    diameter: float = 0.010    # [m]
    helix_deg: float = 35.0
    rake_deg: float = 15.0
    kt: float = 925.0e6        # tangential cutting force coefficient [Pa]
    kn: float = 0.26           # proportionality constant k_n
    mu_c: float = 0.2          # friction coefficient

    @property
    def radius(self) -> float:
        return self.diameter / 2.0


@dataclass(frozen=True)
class ProcessParams:
    """Milling process (down-milling along the top free edge)."""

    ae: float = 0.1e-3         # radial depth of cut [m]
    ft: float = 0.02e-3        # feed per tooth [m]
    down_milling: bool = True
    rpm_ref: float = 4900.0    # reference spindle speed used in ref. Fig. 14
    ap_ref: float = 0.3e-3     # reference axial depth used in ref. Fig. 14


@dataclass(frozen=True)
class SensorParams:
    """Eddy-current displacement probe, upper corner of the plate."""

    xs: float = 0.095          # [m]
    zs: float = 0.075          # [m]
    noise_rms: float = 0.2e-6  # displacement noise RMS [m]


@dataclass(frozen=True)
class MeshParams:
    """FEM mesh. Element size 2.5 mm aligns patch edges with the mesh."""

    nx: int = 40
    nz: int = 32


@dataclass(frozen=True)
class DesignParams:
    """Controller design configuration."""

    # Reduced design model: the five measured modes of the reference rig.
    # Truncation after mode 5 leaves a 1.55x frequency gap to mode 6
    # (4.04 -> 6.28 kHz), which the actuator roll-off filter can separate;
    # truncating after mode 3 would leave only a 1.23x gap to mode 4 and
    # no realizable filter can separate that (verified: catastrophic
    # spillover of 3-mode designs on the 12-mode evaluation model).
    n_modes_design: int = 5
    n_modes_full: int = 12     # evaluation model (spillover check)
    # Mandatory actuator roll-off filter (anti-spillover), composed in
    # series with every synthesized controller at deployment;
    # rolloff_order/2 cascaded 2nd-order sections at rolloff_hz.
    rolloff_hz: float = 4800.0
    rolloff_zeta: float = 0.7
    rolloff_order: int = 4
    # Discrete implementation model: the controller runs on a real-time
    # target at ctrl_rate_hz; the design plant includes a first-order Pade
    # approximation of the total implementation latency
    # latency_factor / ctrl_rate_hz (0.5 sample ZOH reconstruction + 1
    # sample computation), so the synthesized controller carries its
    # sampled-data margins by construction.
    # 50 kHz is comfortable for an ~11-state SISO controller on the
    # reference rig's NI PXIe target and keeps the truncated-mode band
    # (6.3-12.2 kHz) below half the Nyquist rate.
    ctrl_rate_hz: float = 50000.0
    latency_factor: float = 1.5
    n_pos_grid: int = 9        # scheduling grid: tool positions
    n_removal_grid: int = 3    # scheduling grid: removal states
    removal_total: float = 1.0e-3   # multi-pass demo: total height recession [m]
    # H-infinity weights are constructed in avc/synthesis.py from these:
    wf_dc_gain: float = 2.0e6      # displacement performance weight DC gain [1/m]
    wf_bandwidth_hz: float = 2500.0
    wu_gain: float = 1.0 / 150.0   # voltage weight DC value ~ 1/Vmax [1/V]
    # control-effort roll-off (spillover safety): Wu rises tenfold between
    # the zero and the pole, forcing the controller to roll off above the
    # retained-mode band; certified a posteriori by the small-gain check
    # against the fitted additive-uncertainty weight (synthesis.rs_margin).
    # Weight poles are kept at or below ~6 kHz: the weight states become
    # controller states, and every controller pole must stay well below
    # the real-time Nyquist rate (ctrl_rate_hz/2) or the discrete
    # implementation aliases it (certified by sampled_rho).
    wu_zero_hz: float = 1500.0
    wu_pole_hz: float = 6000.0
    wu_order: int = 2              # lead sections cascaded in Wu
    wn_gain: float = 0.2e-6        # sensor noise level at low frequency [m]
    # sensor-noise weight lead (eddy-current probes get noisier at high
    # frequency; the rising Wn also forces the observer gains to roll off,
    # which is what actually bounds the controller gain in the
    # truncated-mode band)
    wn_zero_hz: float = 1200.0
    wn_pole_hz: float = 6000.0
    wn_order: int = 2              # lead sections cascaded in Wn


@dataclass(frozen=True)
class Params:
    plate: PlateParams = field(default_factory=PlateParams)
    piezo: PiezoParams = field(default_factory=PiezoParams)
    tool: ToolParams = field(default_factory=ToolParams)
    process: ProcessParams = field(default_factory=ProcessParams)
    sensor: SensorParams = field(default_factory=SensorParams)
    mesh: MeshParams = field(default_factory=MeshParams)
    design: DesignParams = field(default_factory=DesignParams)

    def tooth_passing_delay(self, rpm: float) -> float:
        """Regenerative delay tau = 60/(N_t * rpm) [s]."""
        return 60.0 / (self.tool.n_teeth * rpm)

    def feed_speed(self, rpm: float) -> float:
        """Table feed speed [m/s] = ft * N_t * rpm / 60."""
        return self.process.ft * self.tool.n_teeth * rpm / 60.0


DEFAULT = Params()


def milling_point(params: Params, height_removed: float = 0.0) -> float:
    """Height coordinate z of the milling point: the (receded) top edge."""
    return params.plate.hp - height_removed


if __name__ == "__main__":
    p = DEFAULT
    print("tau @4900rpm  :", p.tooth_passing_delay(4900.0) * 1e3, "ms")
    print("feed  @4900rpm:", p.feed_speed(4900.0) * 1e3, "mm/s")
    print("pass duration :", p.plate.lp / p.feed_speed(4900.0), "s")
