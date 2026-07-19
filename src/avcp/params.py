"""Physical and process parameters of the plate–actuator milling system.

All values trace to Wang, Song & Liu (2019), Int J Adv Manuf Technol,
doi:10.1007/s00170-019-04493-5 (their Table 1 and Sect. 4.1), except where
noted.  SI units throughout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class PlateParams:
    """Cantilever thin plate (Al-Mg alloy 5052), clamped along y = 0."""
    Lx: float = 0.120           # length along x [m]
    Ly: float = 0.120           # width along y [m]
    h: float = 0.004            # thickness [m]
    E: float = 70.3e9           # Young's modulus [Pa]; the reference paper prints
                                # 700 GPa, an evident typo for Al-Mg 5052
    nu: float = 0.33            # Poisson ratio
    rho: float = 2680.0         # density [kg/m^3]
    zeta: float = 0.0308        # modal damping ratio, identified from the paper's
                                # Table 3 (c = 2 m zeta omega gives 3.08 % at both
                                # measured points)

    @property
    def D(self) -> float:
        """Flexural rigidity [N m]."""
        return self.E * self.h ** 3 / (12.0 * (1.0 - self.nu ** 2))


@dataclass
class PiezoParams:
    """Surface-bonded piezoelectric patch actuator (PZT type).

    The patch occupies [u1, u2] x [v1, v2] on the plate surface.  Its bending
    stiffness and inertia are neglected (thin-patch assumption of the
    reference paper).  The electromechanical constant q maps the applied
    voltage to a uniform biaxial bending moment per unit length along the
    patch boundary:  m_p = q V,  q = Ep d31 (h + ha) / (2 (1 - nup)).
    ``authority`` is an identified dimensionless factor calibrating the
    actuator authority to the static deflections measured in the reference
    paper (Fig. 11); it absorbs bonding-layer and effective-constant
    uncertainty.
    """
    u1: float = 0.050
    u2: float = 0.070
    v1: float = 0.020
    v2: float = 0.058
    ha: float = 0.0004          # patch thickness [m]
    Ep: float = 61.0e9          # patch Young's modulus [Pa] (PZT-5H class)
    nup: float = 0.35
    d31: float = -274e-12       # piezoelectric constant [m/V]
    authority: float = 6.8      # identified from the reference paper's measured
                                # static deflections (Fig. 11: ~5.6e-4 m at
                                # 1200 V at the free edge); absorbs bonding and
                                # effective-constant uncertainty
    V_min: float = -500.0       # high-voltage amplifier range [V]
    V_max: float = 1500.0

    @property
    def q(self) -> float:
        """Moment per unit length per volt [N/V], negative for d31 < 0."""
        h_plate = 0.004  # overwritten by Model, kept for standalone use
        return self.moment_constant(h_plate)

    def moment_constant(self, h_plate: float) -> float:
        return (self.authority * self.Ep * self.d31 * (h_plate + self.ha)
                / (2.0 * (1.0 - self.nup)))


@dataclass
class SensorParams:
    """Accelerometer location (fixed during the pass)."""
    x: float = 0.060
    y: float = 0.100
    lp_cutoff: float = 2500.0   # anti-spillover low-pass cut-off [Hz]
    lp_order: int = 2           # Butterworth order
    noise_rms: float = 0.05     # measurement noise, m/s^2 RMS (charge-amp chain)


@dataclass
class MillingParams:
    """Side milling of the near-free-edge strip, per the reference paper.

    Cutter: 4-flute carbide end mill, D = 12 mm, feed 720 mm/min at
    7000 rpm; radial depth of cut 0.2 mm; the cutter path is a straight
    line parallel to, and 3 mm below, the free top edge.  The axial depth
    (band height along the tool axis / plate y) is ap.  Cutting-force
    coefficients are representative of Al alloys (Altintas, Manufacturing
    Automation, 2012) and are configurable.
    """
    n_teeth: int = 4
    diameter: float = 0.012
    rpm: float = 7000.0
    feed_mm_min: float = 720.0
    ae: float = 0.0002          # radial depth of cut [m]
    ap: float = 0.003           # axial depth of cut (band height) [m]
    Kt: float = 796.0e6         # tangential cutting coefficient [Pa]
    Kr: float = 0.212           # radial/tangential ratio
    up_milling: bool = True
    y_path: float = 0.117       # cutter path height (3 mm from the free edge)
    x_start: float = 0.0
    x_end: float = 0.120
    pass_time: float = 10.0     # s, "10 s to move from point 0 to point 10"
    runout: float = 0.0         # optional per-tooth radial runout [m]

    @property
    def omega_spindle(self) -> float:
        return self.rpm * 2.0 * np.pi / 60.0

    @property
    def f_tpf(self) -> float:
        """Tooth-passing frequency [Hz]."""
        return self.rpm / 60.0 * self.n_teeth

    @property
    def tau(self) -> float:
        """Regenerative (tooth-passing) delay [s]."""
        return 1.0 / self.f_tpf

    @property
    def fz(self) -> float:
        """Feed per tooth [m]."""
        return self.feed_mm_min / 1000.0 / 60.0 / (self.rpm / 60.0 * self.n_teeth)

    @property
    def phi_pitch(self) -> float:
        return 2.0 * np.pi / self.n_teeth

    def engagement_angles(self) -> tuple[float, float]:
        """Entry/exit angle [rad], measured from the surface-normal axis.

        Up-milling: phi in [0, phi_ex]; down-milling: [pi - phi_ex, pi].
        """
        ratio = min(1.0, 2.0 * self.ae / self.diameter)
        phi_ex = float(np.arccos(1.0 - ratio))
        if self.up_milling:
            return 0.0, phi_ex
        return np.pi - phi_ex, np.pi

    def cutter_x(self, t: float | np.ndarray) -> float | np.ndarray:
        """Cutter position along the path at time t (clipped to the pass)."""
        s = np.clip(np.asarray(t, dtype=float) / self.pass_time, 0.0, 1.0)
        out = self.x_start + (self.x_end - self.x_start) * s
        return float(out) if np.isscalar(t) else out


@dataclass
class ControlParams:
    """Shared control-loop hardware description (from the reference paper)."""
    fs: float = 10000.0         # controller sampling rate [Hz]
    delay_steps: int = 1        # one-sample computation/DA delay
    hv_gain: float = 200.0      # high-voltage amplifier gain (-2.5..7.5 V -> -500..1500 V)

    @property
    def Ts(self) -> float:
        return 1.0 / self.fs


@dataclass
class ModelConfig:
    """Numerical configuration."""
    nx: int = 60                # elements along x (2 mm mesh, patch edges on nodes)
    ny: int = 60
    n_modes_eval: int = 20      # evaluation (truth) model order
    n_modes_design: int = 6     # controller design model order
    phys_substeps: int = 5      # physics substeps per control sample (50 kHz)
    n_path_grid: int = 121      # path discretization for scheduling (1 mm)


@dataclass
class SystemParams:
    plate: PlateParams = field(default_factory=PlateParams)
    piezo: PiezoParams = field(default_factory=PiezoParams)
    sensor: SensorParams = field(default_factory=SensorParams)
    milling: MillingParams = field(default_factory=MillingParams)
    control: ControlParams = field(default_factory=ControlParams)
    config: ModelConfig = field(default_factory=ModelConfig)
