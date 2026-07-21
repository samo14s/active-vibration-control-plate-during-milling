"""SDOF regenerative milling with saturated inertial-actuator DVF —
the configuration CLASS of Ozsoy, Sims & Ozturk, MSSP 224 (2025)
111942 ("Actuator saturation during active vibration control of
milling": robotic-milling SDOF mode + direct velocity feedback behind
a force-limited inertial actuator, measured saturation islands).

PARAMETER PROVENANCE (read before quoting any result):
- `REPRESENTATIVE` below is a clearly-labeled stand-in configuration
  typical of the published rig class (low-frequency dominant mode,
  aluminium cutting, force-limited inertial actuator). It is NOT the
  published parameter set: this session's execution environment
  blocks every full-text source of the paper (whiterose eprints,
  ScienceDirect S0888327024008409, SSRN 4853204, ResearchGate — all
  HTTP 403 under the network policy; the paper postdates the local
  scholarly corpus). Results obtained with it demonstrate the island
  MECHANISM and the certificate/protocol machinery on the SDOF class,
  and must be labeled "class-representative", never "Ozsoy's rig".
- `OZSOY_PUBLISHED` is the drop-in slot: fill it from Table(s) of the
  OA PDF (https://eprints.whiterose.ac.uk/id/eprint/233535/) and
  every routine here runs unchanged on the published values.

MATCHED-AUTHORITY PROTOCOL (the comparability rule promised in the
foundational text §6): configurations are compared at equal authority
ratio  A = F_max / F_ref,  F_ref = a4_max * f_t  (the peak static
chip-force scale of the operating point, computed by the same slice
quadrature as the force model). Equal A means the actuator bound sits
at the same fraction of the process force scale, so saturation-island
exposure is compared like for like across rigs.

The lift below returns an avc.satcert.SampledSatLift, so EVERY
certificate of avc.satcert (region-of-linearity O-infinity, circle-
criterion global-recovery margins, clip-recovery diagnostics) applies
to the SDOF loop unchanged. The DVF is modeled sampled at `rate` with
one tick of computation delay (declared implementation assumption;
the published analysis is continuous describing-function).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import expm

from .satcert import SampledSatLift


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SDOFConfig:
    m_kg: float            # modal mass
    fn_hz: float           # natural frequency
    zeta: float            # damping ratio
    kt: float              # tangential cutting coefficient [Pa]
    kr: float              # radial/tangential ratio [-]
    n_teeth: int
    phi_in: float          # engagement entry angle [rad]
    phi_out: float         # engagement exit angle [rad]
    ft: float              # feed per tooth [m]
    g_v: float             # DVF gain [N s / m] (force = -g_v * xdot)
    f_max: float           # actuator force bound [N]
    rate: float = 10e3     # DVF implementation rate [Hz]
    fa_hz: float | None = None   # inertial-actuator resonance [Hz];
    #                              None = ideal force source. The
    #                              transmitted force is the classic
    #                              proof-mass high-pass s^2/(s^2 +
    #                              2 za wa s + wa^2) * F_cmd — the
    #                              phase-distortion ingredient of the
    #                              published rig class
    za: float = 0.25       # actuator damping ratio
    radius: float = 8e-3   # tool radius [m]
    helix_deg: float = 0.0  # helix angle [deg] (slice-lagged force)
    n_axial: int = 16      # axial slices for the helix quadrature
    force_conv: str = "plate-y"  # directional projection convention:
    #   "plate-y": coef = (Kt cos + Kn sin) sin (Du-rig lineage, as in
    #              avc.simulate.CutGeometry)
    #   "ozsoy-y": coef = -(Kt sin - Kn cos) sin — the y-force of
    #              Ozsoy Eq. (4) (down-milling switching, chip
    #              h = sin(theta)(ft + w - w_tau)); the overall sign
    #              is fixed by matching the published uncontrolled
    #              stability boundary (declared calibration)
    conv_sign: float = 1.0
    label: str = "sdof"


# class-representative stand-in (NOT the published values — see header):
# 25 Hz / 3 % robot-class mode, aluminium half-immersion down-milling,
# strong DVF (adds ~16x the structural damping: g_v / (2 zeta m wn)
# ~ 15.9) behind a 60 N inertial-actuator bound (saturates at chatter
# velocities ~ 2 cm/s, i.e. mode amplitudes ~ 0.13 mm)
REPRESENTATIVE = SDOFConfig(
    m_kg=20.0, fn_hz=25.0, zeta=0.03,
    kt=700e6, kr=0.30, n_teeth=3,
    phi_in=np.pi / 2, phi_out=np.pi,
    ft=0.10e-3, g_v=3.0e3, f_max=60.0,
    label="class-representative (stand-in)")

# PUBLISHED parameter set — Ozsoy, Sims & Ozturk, MSSP 224 (2025)
# 111942, saturation-island configuration (DVF, 1700-2250 rpm band).
# Provenance: the island-test flexure parameters come from the
# paper's Section 3.2 TEXT (its Table 2 lists the earlier preloaded
# configuration, 129.3 Hz / 1.34 % / 1.34e7 N/m); tool, cutting
# coefficients, actuator and DVF gain from Tables 1-2 and Eq. (2):
#   flexure workpiece for the island tests: fn = 128.1 Hz,
#   zeta = 1.48 %, k = 1.13e7 N/m  =>  m = k/wn^2 = 17.44 kg;
#   tool D16 mm, 4 teeth, 45 deg helix, Al 7075-T6, DOWN milling,
#   half immersion (phi in [pi/2, pi]), ft = 0.05 mm;
#   Kt = 660e6 N/m^2, Kn = 180e6 N/m^2 (=> kr = 180/660; edge
#   coefficients Kte/Kne are not tabulated in the paper — set to zero,
#   declared);
#   DVF gain g_dvf = 253 (V per m/s) through the actuator
#   f_act/V_in = 3 s^2/(s^2 + 15.834 s + 2785.6)  =>  proof-mass
#   fa = 8.4 Hz, za = 0.15, mid-band force gain 3*253 = 759 N s/m;
#   saturation threshold of the control signal: 27 N.
# Implementation rate 10 kHz is OUR declared assumption (the paper's
# Simulink Desktop Real-Time rate is not stated).
# force_conv/conv_sign: the y-projection of the paper's Eq. (4) with
# the overall regenerative sign CALIBRATED against the published
# uncontrolled stability boundary (reproduces its shape: high left
# flank ~14-18 mm at 1800-1921 rpm, collapse to ~2 mm above 2000 rpm,
# 0.9 mm at 2800 rpm vs the published 1.2 mm) — declared calibration.
OZSOY_PUBLISHED: SDOFConfig | None = SDOFConfig(
    m_kg=1.13e7 / (2.0 * np.pi * 128.1) ** 2, fn_hz=128.1, zeta=0.0148,
    kt=660e6, kr=180.0 / 660.0, n_teeth=4,
    phi_in=np.pi / 2, phi_out=np.pi, ft=0.05e-3,
    g_v=3.0 * 253.0, f_max=27.0, rate=10e3,
    fa_hz=8.4, za=0.15, radius=8e-3, helix_deg=45.0,
    force_conv="ozsoy-y", conv_sign=-1.0,
    label="Ozsoy-Sims-Ozturk MSSP 224 (2025) 111942, Sec. 3.2 flexure "
          "+ Tables 1-2 + Eq. (2) (DVF saturation-island config)")


def tooth_period(cfg: SDOFConfig, rpm: float) -> float:
    return 60.0 / (cfg.n_teeth * rpm)


# ---------------------------------------------------------------------------
# cutting-force directional coefficient (same slice quadrature as
# avc.simulate.CutGeometry, x-direction SDOF)
# ---------------------------------------------------------------------------
def _offsets(cfg: SDOFConfig, ap: float):
    """Per (tooth, axial-slice) angular offsets with helix lag, and the
    slice height dz (same quadrature as avc.simulate.CutGeometry)."""
    na = cfg.n_axial if cfg.helix_deg != 0.0 else 1
    z = (np.arange(na) + 0.5) * (ap / na)
    lag = z * np.tan(np.deg2rad(cfg.helix_deg)) / cfg.radius
    j = 2.0 * np.pi * np.arange(cfg.n_teeth) / cfg.n_teeth
    return (j[:, None] - lag[None, :]).ravel(), ap / na


def _dir_coef(cfg: SDOFConfig, phi: np.ndarray) -> np.ndarray:
    """Directional force coefficient WITHOUT the chip's sin factor
    (multiply by the chip h = sin(phi)*(ft + w - w_tau))."""
    s = np.sin(phi)
    kn = cfg.kr * cfg.kt
    if cfg.force_conv == "ozsoy-y":
        coef = -(cfg.kt * s - kn * np.cos(phi))
    else:
        coef = cfg.kt * np.cos(phi) + kn * s
    return cfg.conv_sign * coef


def alpha4_time(cfg: SDOFConfig, ap: float, rpm: float,
                t: np.ndarray) -> np.ndarray:
    """a4(t) such that F_cut = -a4(t) * (ft + x - x_tau)."""
    offs, dz = _offsets(cfg, ap)
    Omega = 2.0 * np.pi * rpm / 60.0
    phi = np.mod(Omega * np.asarray(t)[..., None] + offs, 2.0 * np.pi)
    eng = (phi >= cfg.phi_in) & (phi <= cfg.phi_out)
    coef = _dir_coef(cfg, phi) * np.sin(phi)
    return dz * np.sum(np.where(eng, coef, 0.0), axis=-1)


def authority_ratio(cfg: SDOFConfig, ap: float, rpm: float,
                    n_t: int = 2048) -> float:
    """A = F_max / (a4_max * ft) — the matched-authority metric."""
    t = np.linspace(0.0, tooth_period(cfg, rpm), n_t, endpoint=False)
    f_ref = float(np.max(np.abs(alpha4_time(cfg, ap, rpm, t)))) * cfg.ft
    return cfg.f_max / max(f_ref, 1e-300)


# ---------------------------------------------------------------------------
# sampled-data lift -> SampledSatLift (all satcert certificates apply)
# ---------------------------------------------------------------------------
def lift_sdof(cfg: SDOFConfig, ap: float, rpm: float,
              m: int | None = None, n_sub: int = 8) -> SampledSatLift:
    tau = tooth_period(cfg, rpm)
    if m is None:
        m = int(round(tau * cfg.rate))
    dt = tau / m
    t_sub = (np.arange(m * n_sub) + 0.5) * (tau / (m * n_sub))
    a4 = alpha4_time(cfg, ap, rpm, t_sub).reshape(m, n_sub).mean(axis=1)

    wn = 2.0 * np.pi * cfg.fn_hz
    A0m = np.array([[0.0, 1.0], [-wn * wn, -2.0 * cfg.zeta * wn]])
    if cfg.fa_hz is None:
        n2 = 2
        A0 = A0m
        bu = np.array([0.0, 1.0 / cfg.m_kg])               # ideal force
    else:
        # proof-mass high-pass: F_out = F_cmd - ca @ xa,
        # xa' = Aa xa + ba F_cmd
        wa = 2.0 * np.pi * cfg.fa_hz
        ca = np.array([wa * wa, 2.0 * cfg.za * wa])
        n2 = 4
        A0 = np.zeros((4, 4))
        A0[:2, :2] = A0m
        A0[1, 2:] = -ca / cfg.m_kg
        A0[2, 3] = 1.0
        A0[3, 2:] = np.array([-wa * wa, -2.0 * cfg.za * wa])
        bu = np.array([0.0, 1.0 / cfg.m_kg, 0.0, 1.0])
    Ady = np.zeros((n2, n2))
    Ady[1, 0] = -1.0 / cfg.m_kg                            # per unit a4
    bf = np.zeros(n2)
    bf[1] = 1.0 / cfg.m_kg                                 # cut force path
    cs = np.zeros(n2)
    cs[1] = 1.0                                            # velocity sensor
    cw = np.zeros(n2)
    cw[0] = 1.0                                            # surface record
    Dk = -cfg.g_v                                          # static DVF
    Ea = np.empty((m, n2, n2))
    Bcols = np.empty((m, n2, 3))
    for i in range(m):
        Ai = A0 + a4[i] * Ady
        B = np.column_stack([bu, a4[i] * bf, -a4[i] * cfg.ft * bf])
        Maug = np.zeros((n2 + 3, n2 + 3))
        Maug[:n2, :n2] = Ai * dt
        Maug[:n2, n2:] = B * dt
        E = expm(Maug)
        Ea[i] = E[:n2, :n2]
        Bcols[i] = E[:n2, n2:]

    N = n2 + 1 + m                       # [x, xdot | f_pend | hist ring]
    iu = n2
    ih = iu + 1
    nc = N + 1 + m
    Z = np.zeros((N, nc))
    Z[:N, :N] = np.eye(N)
    Vrows = np.empty((m, nc))
    for k in range(m):
        Xp = Z[:n2]
        y = cs @ Xp
        v = Dk * y                        # commanded FORCE at tick k
        Vrows[k] = v
        u_apply = Z[iu].copy()
        w_now = cw @ Xp
        w_tau = Z[ih + (k % m)].copy()
        Z[ih + (k % m)] = w_now
        Z[iu] = v
        aff = np.zeros(nc)
        aff[N] = 1.0
        Z[:n2] = (Ea[k] @ Xp + np.outer(Bcols[k][:, 0], u_apply)
                  + np.outer(Bcols[k][:, 1], w_tau)
                  + np.outer(Bcols[k][:, 2], aff))
        Z[iu, N + 1 + k] -= 1.0           # unit psi at tick k
    e_h = np.zeros(N)
    e_h[ih:] = 1.0
    return SampledSatLift(Phi=Z[:, :N].copy(), acc=Z[:, N].copy(),
                          V=Vrows[:, :N].copy(), v_aff=Vrows[:, N].copy(),
                          e_h=e_h, N=N, m=m, tau=tau, dt=dt,
                          Gpsi=Z[:, N + 1:].copy(),
                          Vpsi=Vrows[:, N + 1:].copy(),
                          meta={"cfg": cfg.label, "ap": ap, "rpm": rpm,
                                "rate": cfg.rate,
                                "authority": authority_ratio(cfg, ap, rpm)})


# ---------------------------------------------------------------------------
# nonlinear time-domain simulator (mirrors avc.simulate at SDOF scale)
# ---------------------------------------------------------------------------
def simulate_sdof(cfg: SDOFConfig, ap: float, rpm: float, T: float,
                  fs: float = 50e3, f_max: float | None = None,
                  loss_of_contact: bool = True,
                  surf_step: tuple[float, float] | None = None,
                  n_phi: int = 1) -> dict:
    """Fixed-step simulation. Returns dict with t, x, f_cmd, f_act,
    clip_frac, diverged."""
    if f_max is None:
        f_max = cfg.f_max
    dt = 1.0 / fs
    tau = tooth_period(cfg, rpm)
    ctrl_div = max(1, int(round(fs / cfg.rate)))
    if abs(ctrl_div * cfg.rate - fs) > 1e-6 * fs:
        raise ValueError("fs must be an integer multiple of cfg.rate")
    wn = 2.0 * np.pi * cfg.fn_hz
    A0m = np.array([[0.0, 1.0], [-wn * wn, -2.0 * cfg.zeta * wn]])
    if cfg.fa_hz is None:
        n2 = 2
        A0 = A0m
        b_u = np.array([0.0, 1.0 / cfg.m_kg])
    else:
        wa = 2.0 * np.pi * cfg.fa_hz
        ca = np.array([wa * wa, 2.0 * cfg.za * wa])
        n2 = 4
        A0 = np.zeros((4, 4))
        A0[:2, :2] = A0m
        A0[1, 2:] = -ca / cfg.m_kg
        A0[2, 3] = 1.0
        A0[3, 2:] = np.array([-wa * wa, -2.0 * cfg.za * wa])
        b_u = np.array([0.0, 1.0 / cfg.m_kg, 0.0, 1.0])
    b_f = np.zeros(n2)
    b_f[1] = 1.0 / cfg.m_kg
    M = np.zeros((n2 + 2, n2 + 2))
    M[:n2, :n2] = A0 * dt
    M[:n2, n2] = b_u * dt
    M[:n2, n2 + 1] = b_f * dt
    E = expm(M)
    Ad, bd_u, bd_f = E[:n2, :n2], E[:n2, n2], E[:n2, n2 + 1]

    n = int(round(T * fs))
    D = tau * fs
    t = np.arange(n) * dt
    Omega = 2.0 * np.pi * rpm / 60.0
    offs, dz_sl = _offsets(cfg, ap)

    x = np.zeros(n2)
    xh = np.zeros(n)
    fc = np.zeros(n)
    u_hold = 0.0
    u_next = 0.0
    n_clip = 0
    n_tick = 0
    diverged = None
    for k in range(n):
        w = x[0]
        if not np.isfinite(w) or abs(w) > 0.05:
            diverged = t[k]
            n = k
            break
        xh[k] = w
        if k % ctrl_div == 0:
            u_hold = u_next
            v_cmd = -cfg.g_v * x[1]
            if abs(v_cmd) > f_max:
                n_clip += 1
                v_cmd = np.sign(v_cmd) * f_max
            u_next = float(v_cmd)
            n_tick += 1
        pos = k - D
        if pos < 0.0:
            w_tau = 0.0
        else:
            i0 = int(pos)
            fr = pos - i0
            w_tau = (1.0 - fr) * xh[i0] + fr * xh[i0 + 1]
        if surf_step is not None and \
                surf_step[0] <= t[k] < surf_step[0] + tau:
            w_tau += surf_step[1]
        phi = np.mod(Omega * t[k] + offs, 2.0 * np.pi)
        eng = (phi >= cfg.phi_in) & (phi <= cfg.phi_out)
        F = 0.0
        if eng.any():
            s = np.sin(phi[eng])
            h = s * (cfg.ft + w - w_tau)
            if loss_of_contact:
                h = np.maximum(h, 0.0)
            F = float(-dz_sl * np.sum(_dir_coef(cfg, phi[eng]) * h))
        fc[k] = u_hold
        x = Ad @ x + bd_u * u_hold + bd_f * F
    return {"t": t[:n], "x": xh[:n], "f_act": fc[:n],
            "clip_frac": n_clip / max(n_tick, 1), "diverged": diverged,
            "tau": tau}
