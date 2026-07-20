"""Milling force model: helical multi-tooth engagement, linearized
regenerative coefficients.

Classical linear-edge force model (Altintas), tangential/radial:

    dFt = kt * h * dz,      dFr = kr * kt * h * dz

with instantaneous chip thickness h(phi) = [ft + w(t) - w(t-tau)] *
sin(phi), where w is the plate deflection along its normal y: the feed
contributes ft per tooth, the CURRENT deflection w(t) thickens the chip
and the surface left one tooth period ago, w(t-tau), thins it (this
orientation is what makes the operative force below carry
a4*(w(t-tau) - w(t)); the simulator implements exactly this h).
Projecting the elemental force on y and integrating over the engaged
helical flute segments gives the y-direction force

    Fy(t) = -a3(t) * ft - a4(t) * [ w(t) - w(t-tau) ]

where a4(t) [N/m] is the periodic regenerative (cutting-stiffness)
coefficient and a3(t) [N/m] the static feed-ripple coefficient. With the
down-milling engagement window [pi - phi_e, pi] the directional factor is
negative, so a4(t) <= 0 here; all downstream code (SLD, synthesis,
simulation) uses the signed value consistently:

    Fy_regen(t) = a4(t) * [ w(t-tau) - w(t) ]  (+ static ripple).

Down-milling engagement: phi in [pi - phi_e, pi], phi_e = acos(1 - 2 ae/D).

Angle convention: phi measured from the +y axis (plate normal, pointing
away from the plate) rotating toward +x (feed); dz along the tool axis.
Helix lag: phi_j(t, z) = Omega t + 2 pi j / N - z * tan(eta) / R.

The y-projection of the elemental force for a tooth at angle phi:

    dFy = [ -kt cos(phi) - kr kt sin(phi) ] * h(phi) dz     (down-milling)

so the specific directional factor is

    g(phi) = sin(phi) * ( kt cos(phi) + kr kt sin(phi) )

and  a4(t) = sum_j int_{engaged} g(phi_j(t,z)) dz,
     a3(t) = a4(t) in this single-direction projection (the same factor
     multiplies the ft term), which mirrors alpha3/alpha4 of the reference
     up to their oblique-cutting constants.
"""
from __future__ import annotations

import numpy as np

from .params import Params


def engagement_angles(params: Params) -> tuple[float, float]:
    """(phi_in, phi_out) for down-milling, radial depth ae."""
    ae = params.process.ae
    D = params.tool.diameter
    phi_e = np.arccos(1.0 - 2.0 * ae / D)
    return np.pi - phi_e, np.pi


def _g(phi: np.ndarray, params: Params) -> np.ndarray:
    """Directional cutting factor g(phi) [Pa]; zero outside engagement."""
    kt = params.tool.kt
    kr = params.tool.kn
    phi_in, phi_out = engagement_angles(params)
    phim = np.mod(phi, 2.0 * np.pi)
    engaged = (phim >= phi_in) & (phim <= phi_out)
    return np.where(engaged,
                    np.sin(phim) * (kt * np.cos(phim) + kr * kt * np.sin(phim)),
                    0.0)


def alpha4_time(params: Params, ap: float, rpm: float,
                t: np.ndarray, n_axial: int = 40) -> np.ndarray:
    """Periodic regenerative coefficient a4(t) [N/m].

    Axial integration over the helical flutes (n_axial slices over the
    axial depth ap).
    """
    R = params.tool.radius
    eta = np.deg2rad(params.tool.helix_deg)
    N = params.tool.n_teeth
    Omega = 2.0 * np.pi * rpm / 60.0
    z = (np.arange(n_axial) + 0.5) * (ap / n_axial)
    dz = ap / n_axial
    a4 = np.zeros_like(t, dtype=float)
    for j in range(N):
        for zk in z:
            phi = Omega * t + 2.0 * np.pi * j / N - zk * np.tan(eta) / R
            a4 += _g(phi, params) * dz
    return a4


def mean_alpha4(params: Params, ap: float) -> float:
    """Zero-order (time-averaged) regenerative coefficient [N/m].

    Analytic average: N/(2 pi) * ap * int_{phi_in}^{phi_out} g(phi) dphi
    (helix does not change the average).
    """
    kt = params.tool.kt
    kr = params.tool.kn
    N = params.tool.n_teeth
    phi_in, phi_out = engagement_angles(params)
    # int sin*cos = sin^2/2 ; int sin^2 = phi/2 - sin(2phi)/4
    def F(phi):
        return kt * 0.5 * np.sin(phi) ** 2 \
            + kr * kt * (phi / 2.0 - np.sin(2.0 * phi) / 4.0)
    return N / (2.0 * np.pi) * ap * (F(phi_out) - F(phi_in))


def static_force_time(params: Params, ap: float, rpm: float,
                      t: np.ndarray, n_axial: int = 40) -> np.ndarray:
    """Static (feed-copy) force ripple  -a3(t)*ft  [N] on the plate."""
    return -alpha4_time(params, ap, rpm, t, n_axial) * params.process.ft
