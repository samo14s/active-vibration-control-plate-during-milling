"""
Physical and numerical parameters for the milling-chatter active-control study.

All values are taken from (or estimated consistently with) the reference paper:

    J. Du, X. Liu, H. Dai, X. Long,
    "Robust combined time delay control for milling chatter suppression of
     flexible workpieces", Int. J. Mech. Sci. 274 (2024) 109257.
    https://doi.org/10.1016/j.ijmecsci.2024.109257

The paper studies the milling of a thin-walled AL6061 cantilever plate whose
transverse vibration is dominated by a few very lightly damped modes.  Milling
introduces a regenerative (time-delay) cutting force that turns those modes
unstable -> "chatter".  A piezoelectric patch bonded to the plate is used as the
actuator; a displacement sensor on the plate provides the measurement.

This module gathers every number in one place so that plant.py stays readable
and the controller-comparison study is fully reproducible.

Notes on values that the paper does not publish numerically
-----------------------------------------------------------
The paper reports the modal frequencies / damping ratios (Table 4), the plate
and tool geometry (Tables 1-3) and the milling condition "S", but it does *not*
publish the mass-normalised mode-shape amplitudes at the milling point, the
absolute milling-force coefficient alpha4, or the piezo modal-input gain.  Those
three quantities set the *loop gain* of the regenerative feedback and the
control authority.  We therefore:

  * use physically plausible mass-normalised mode shapes (O(1/sqrt(m_eff))),
  * expose a single dimensionless cutting-strength multiplier `CUT_GAIN`, and a
    single actuator-authority gain `PIEZO_GAIN`,

and calibrate those two scalars (see plant.calibrate_open_loop) so that the
UNCONTROLLED plate at condition "S" is unstable with the second mode chattering
near ~1.1 kHz, exactly as reported in Fig. 14(a) of the paper.  Everything else
is fixed by the measured modal data.
"""

from __future__ import annotations
import numpy as np

# --------------------------------------------------------------------------- #
# 1. Plate modal data  (Table 4 : measured natural frequencies & damping)     #
# --------------------------------------------------------------------------- #
# We keep the first three transverse modes.  Modes 1 and 2 dominate the milling
# response at condition S (2nd-mode chatter ~1135 Hz); mode 3 is retained to
# show broadband behaviour and to make the reduced-order controllers work
# against un-modelled high-frequency dynamics.
MODE_FREQ_HZ = np.array([540.0, 1068.0, 2787.0])          # measured f_n  (Hz)
MODE_ZETA    = np.array([0.0031, 0.0017, 0.0027])          # measured damping ratios
N_MODES      = len(MODE_FREQ_HZ)

OMEGA_N = 2.0 * np.pi * MODE_FREQ_HZ                        # rad/s

# Mass-normalised transverse mode shape amplitude scale (per mode) at a free-edge
# antinode.  Plate mass m_p = rho * l * h * b = 2830*0.1*0.08*0.004 = 0.0906 kg;
# the mass-normalised modal displacement is ~1/sqrt(m_eff), O(5..7) [1/sqrt(kg)].
PHI = np.array([6.6, 6.1, 5.2])                            # 1/sqrt(kg)

# --------------------------------------------------------------------------- #
#  Spatial mode shapes of the CANTILEVER plate (clamped bottom, 3 free edges)  #
# --------------------------------------------------------------------------- #
# The sensor, the piezo actuator and the milling point sit at DIFFERENT places
# (paper Fig. 2 / Section 5), so the plant is genuinely NON-COLLOCATED.  We build
# representative separable mode shapes  phi_i(x,z) = X_i(x) * Z_i(z),  x,z in [0,1]
# (x along the length l_p, z along the height h_p; z=0 clamped, z=1 free top edge):
#   mode 1 = 1st bending  : X_0(x)=1                * Z_1(z)   (uniform along edge)
#   mode 2 = 1st torsion  : X_rot(x)=2x-1           * Z_1(z)   (antisymmetric in x)
#   mode 3 = 2nd bending  : X_0(x)=1                * Z_2(z)
# Z_n are clamped-free (cantilever) beam mode shapes; the piezo bending coupling
# uses the z-curvature Z_n'' (paper Eq. 14 integrates mode curvature over the patch).
_B1, _S1 = 1.8751, 0.7341        # cantilever beam eigen-data (modes 1 and 2 in z)
_B2, _S2 = 4.6941, 1.0185

def _Zc(zt, b, s):               # clamped-free beam mode shape
    return (np.cosh(b*zt) - np.cos(b*zt)) - s*(np.sinh(b*zt) - np.sin(b*zt))

def _Zc2(zt, b, s):              # its 2nd derivative (curvature ~ piezo coupling)
    return b*b*((np.cosh(b*zt) + np.cos(b*zt)) - s*(np.sinh(b*zt) + np.sin(b*zt)))

def plate_mode_shape(xt, zt):
    """Displacement mode shape [phi1,phi2,phi3] at (xt,zt), normalised, in [0,1]."""
    z1 = _Zc(zt, _B1, _S1); z2 = _Zc(zt, _B2, _S2)
    return np.array([z1 * 1.0, z1 * (2.0*xt - 1.0), z2 * 1.0])[:N_MODES]

def plate_mode_curv(xt, zt):
    """z-curvature of the mode shapes at (xt,zt) — piezo bending coupling."""
    z1 = _Zc2(zt, _B1, _S1); z2 = _Zc2(zt, _B2, _S2)
    return np.array([z1 * 1.0, z1 * (2.0*xt - 1.0), z2 * 1.0])[:N_MODES]

# Locations (normalised x,z).  Paper Section 5 (experiments): displacement sensor
# at the RIGHT-UPPER corner of the plate back, piezo patch at the RIGHT-LOWER
# corner.  Both on the right side -> the torsion mode has the same sign at both,
# so the non-collocated plant stays MINIMUM-PHASE (controllable).  Milling runs
# along the free upper edge (z=1), x advancing with the tool.
SENSOR_POS = (1.00, 1.00)        # right-upper corner
ACT_POS    = (0.80, 0.25)        # right-lower patch centroid

# per-mode scale so |sensor mode shape| matches the calibrated PHI magnitudes
_phi_s_raw = plate_mode_shape(*SENSOR_POS)
_MS_SCALE  = PHI[:N_MODES] / np.abs(_phi_s_raw)
PHI_SENSOR = _phi_s_raw * _MS_SCALE                        # sensor output vector

def phi_mill(xt):
    """Milling-point mode shape at fractional edge position xt in [0,1]."""
    return plate_mode_shape(xt, 1.0) * _MS_SCALE

# piezo actuator modal input direction (curvature-based), scaled so the strongest
# (1st-mode) entry matches the reference amplitude; the calibrated PIEZO_GAIN then
# sets the absolute authority.
_hpe_raw = plate_mode_curv(*ACT_POS)
PHI_ACT  = _hpe_raw / np.abs(_hpe_raw[0]) * PHI[0]         # actuator input vector

# Reference milling position for the fixed-position (snapshot) study: near the
# right end of the free edge (close to the sensor), where the milling excites the
# 2nd (torsion) mode strongly and produces the 2nd-mode chatter of the paper's
# condition S (f_c2 ~ 1.1 kHz).  Still non-collocated: phi_mill(0.9) != sensor.
MILL_POS_STATIC = 0.9

# --------------------------------------------------------------------------- #
# 2. Plate geometry / material  (Table 1)                                     #
# --------------------------------------------------------------------------- #
PLATE = dict(
    material="AL6061",
    lp=0.100, hp=0.080, bp=0.004,     # m  (length, height, thickness)
    rho=2830.0,                        # kg/m^3
    E=69e9,                            # Pa
    mu=0.33,
)
PLATE_MASS = PLATE["rho"] * PLATE["lp"] * PLATE["hp"] * PLATE["bp"]   # ~0.0906 kg

# --------------------------------------------------------------------------- #
# 3. Piezoelectric patch actuator  (Table 2)                                  #
# --------------------------------------------------------------------------- #
PIEZO = dict(
    d31=175e-12,     # m/V   piezo strain constant
    h_pa=0.7e-3,     # m     actuator thickness
    E_pe=63e9,       # Pa
    mu_pe=0.35,
    length=0.060,    # m
    width=0.020,     # m
)

# --------------------------------------------------------------------------- #
# 4. Milling tool & cutting coefficients  (Table 3)                           #
# --------------------------------------------------------------------------- #
TOOL = dict(
    n_teeth=3,
    diameter=0.010,      # m
    radius=0.005,        # m
    helix_deg=35.0,
    rake_deg=15.0,
    kt=925e6,            # Pa   cutting force coefficient
    kn=0.26,             # proportionality constant
    mu_c=0.2,            # friction coefficient
)

# --------------------------------------------------------------------------- #
# 5. Milling condition "S"  (Section 4.2)                                      #
# --------------------------------------------------------------------------- #
CONDITION_S = dict(
    spindle_rpm=4900.0,
    a_radial=0.1e-3,     # m   radial cutting depth
    a_axial=0.3e-3,      # m   axial cutting depth
    feed_per_tooth=0.02e-3,   # m/tooth
    down_milling=True,
)

def tooth_passing_frequency(rpm: float, n_teeth: int) -> float:
    """Tooth passing frequency f_t [Hz]."""
    return rpm / 60.0 * n_teeth

def time_delay(rpm: float, n_teeth: int) -> float:
    """Regenerative time delay tau [s] = one tooth-passing period."""
    return 1.0 / tooth_passing_frequency(rpm, n_teeth)

# For condition S:  f_t = 4900/60*3 = 245 Hz  ->  tau = 4.082 ms
FT_S  = tooth_passing_frequency(CONDITION_S["spindle_rpm"], TOOL["n_teeth"])
TAU_S = time_delay(CONDITION_S["spindle_rpm"], TOOL["n_teeth"])

# --- Full milling PASS along the free upper edge (peripheral / side milling) --- #
# The tool advances along the edge at the feed velocity
#   v_feed = feed_per_tooth * n_teeth * (rpm/60)
# and traverses the whole plate length l_p, so the pass lasts T_PASS seconds.
FEED_VELOCITY = (CONDITION_S["feed_per_tooth"] * TOOL["n_teeth"]
                 * CONDITION_S["spindle_rpm"] / 60.0)          # m/s  (= 4.9 mm/s)
T_PASS = PLATE["lp"] / FEED_VELOCITY                            # s    (= 20.4 s)

# Material removal over the pass: peripheral milling shaves the plate thickness at
# the free upper edge, so the modal frequencies drift upward as the tool advances.
# The paper reports the 1st/2nd natural frequencies rising ~17 %/9 % after a whole
# SERIES of passes; a single pass removes far less, so we drift by a small fraction
# (scaled by FREQ_DRIFT_PASS) linearly from pass start to end.  Set to 0 to disable.
FREQ_DRIFT_PASS = np.array([0.03, 0.02, 0.015])   # fractional rise of each omega over one pass

# --------------------------------------------------------------------------- #
# 6. Cutting-engagement geometry (non-smooth / intermittent milling)          #
# --------------------------------------------------------------------------- #
# Low radial immersion -> each tooth cuts for only a short fraction of the
# tooth period.  Engagement (swept) angle for the given radial depth:
#   phi_eng = acos(1 - a_e / R)
_phi_eng = np.arccos(np.clip(1.0 - CONDITION_S["a_radial"] / TOOL["radius"], -1.0, 1.0))
# With N teeth the pitch angle is 2*pi/N; the engaged duty ratio is:
DUTY = float(_phi_eng / (2.0 * np.pi / TOOL["n_teeth"]))     # ~0.096  (9.6 %)

# --------------------------------------------------------------------------- #
# 7. Loop-gain calibration scalars  (see module docstring)                    #
# --------------------------------------------------------------------------- #
# Nominal absolute milling-force coefficient alpha4 [N/m] ~ kt * a_axial * (geo).
# alpha4 enters the regenerative stiffness  alpha4 * PHI PHI^T.  We fold the
# geometry / averaging factors and the mode-shape scaling into two calibrated
# dimensionless knobs, tuned in plant.py so that the open loop matches Fig.14(a).
ALPHA4_BASE = TOOL["kt"] * CONDITION_S["a_axial"]           # ~2.78e5 N/m (order)
# --- Calibrated (with the non-collocated geometry and the reference milling
#     position MILL_POS_STATIC) so that the UNCONTROLLED plate at condition S is
#     unstable with 2nd-mode chatter ~1.1 kHz (matches Fig. 14a), stabilisable
#     with tens of volts (matches the paper's control-voltage scale).
CUT_GAIN    = 2.2     # dimensionless regenerative-loop-gain multiplier
PIEZO_GAIN  = 8.0     # dimensionless actuator-authority multiplier

# Non-smooth milling-force coefficient bounds (Section 3.2, Eq. 23):
#   alpha4(t) varies in 0.3 alpha4 .. 2.9 alpha4 ; nominal 1.6 alpha4
ALPHA4_NOMINAL_FACTOR = 1.6
ALPHA4_MIN_FACTOR     = 0.3
ALPHA4_MAX_FACTOR     = 2.9

# --------------------------------------------------------------------------- #
# 8. Numerical / simulation settings                                          #
# --------------------------------------------------------------------------- #
FS       = 25600.0          # Hz   simulation & control sample rate (Nyquist 12.8 kHz)
DT       = 1.0 / FS         # s
T_SIM    = 0.30             # s    simulation horizon
U_MAX    = 150.0            # V    actuator (control-voltage) saturation limit

# Reproducibility
SEED = 20240403             # = paper acceptance date, for the measurement noise RNG

# Measurement noise (displacement sensor) standard deviation [m].
# Plate chatter amplitudes are O(10-100 um); use a small sensor noise.
MEAS_NOISE_STD = 2.0e-7     # 0.2 um RMS
