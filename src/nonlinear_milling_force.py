"""
nonlinear_milling_force.py
==========================
ROADMAP item 14 — the cutting force with the tooth allowed to leave the cut.

WHY THIS EXISTS
---------------
The baseline force model is linear in the displacement at every amplitude.
For the article's parameters the maximum uncut chip is

    h_max = f_t sin(phi_st) = 0.02 mm x sin(168.5 deg) = 3.98 um

at 1 % radial immersion, so the tooth physically separates from the workpiece
once the relative vibration exceeds about 4 um. The baseline nonetheless
integrates its linear law up to a divergence guard of 5 mm -- 1256 x h_max,
50 x the radial engagement, and 1.25 x the plate thickness. Loss of contact is
THE amplitude-limiting nonlinearity in low-immersion milling, so every
large-amplitude trace produced that way is an exponential, not a chatter
simulation.

WHAT IS MODELLED
----------------
For tooth j at axial height z, with Omega the spindle rate and eta the helix
angle,

    theta_j(t, z) = Omega t + 2 pi j / N_T - z tan(eta) / R

The tooth cuts only when it is inside the angular engagement window
[phi_st, phi_ex] AND the instantaneous chip thickness is positive:

    h_j = f_t sin(theta) + (u(t - tau) - u(t)) cos(theta)
    contributes only where h_j > 0

with u = Dp^T q the modal displacement at the contact point. The modal force
is then

    F = K_t Dp  sum_j sum_slices (k2 sin theta - k1 cos theta) max(h_j, 0) dz

CONSISTENCY WITH THE LINEAR MODEL
---------------------------------
Expanding max(h,0) -> h (i.e. assuming the tooth never leaves) and comparing
with the baseline's coefficients gives exactly

    F = f_t alpha3 Dp + alpha4 Dp (u_del - u_now)
    alpha3 = K_t (k2 ss - k1 sc),   alpha4 = K_t (k2 sc - k1 cc)

with ss, sc, cc the integrals of sin^2, sin cos, cos^2 over the engaged
region -- which is precisely `milling_force.milling_force_coeffs`. So this
module is a strict generalisation, and `tests/verify_nonlinear_force.py`
checks that the two agree to numerical precision whenever the chip stays
positive. That equivalence is what makes the nonlinear model trustworthy:
it is the same physics with one inequality restored.

WHEN THE INEQUALITY BITES
-------------------------
Not only at large amplitude. At this immersion the exit angle is phi_ex = pi,
where sin(phi) = 0, so the NOMINAL chip thickness vanishes at the end of the
arc. Any non-zero vibration therefore drives h negative somewhere in the
engagement, and the tooth is partly out of cut at infinitesimal amplitude:
measured here, 2.6 % of the engaged (tooth, slice) samples at 0.1 um of
relative vibration, 24.9 % at 1 um, 74.4 % at 3 um, and 100 % beyond 10 um.
The linear model is thus not merely wrong far from equilibrium at 1 % radial
immersion -- it is approximate everywhere, and exact only in the limit.

The force consequently SATURATES instead of growing: at 1000 um of relative
vibration the linear model returns a force 8.1e0 while the true force is
identically zero, because the tooth is never in contact.

SCOPE
-----
This restores loss of contact. It does NOT add process damping or an
edge/ploughing term, both of which matter at 2-4 um chip thickness and
154 m/min, and both of which remain stated limitations.
"""
from __future__ import annotations

import numpy as np


class NonlinearMillingForce:
    """
    Instantaneous modal cutting force with chip-thickness positivity.

    Parameters
    ----------
    Omega   : spindle angular rate [rad/s]
    NT      : number of teeth
    RT      : tool radius [m]
    eta     : helix angle [rad]
    phi_st, phi_ex : angular engagement window [rad]
    za_low, za_high : axial engagement band [m]
    k1, k2, kt : the baseline's force-direction coefficients
    ft      : feed per tooth [m]
    n_slice : axial slices used for the integration
    """

    def __init__(self, Omega, NT, RT, eta, phi_st, phi_ex,
                 za_low, za_high, k1, k2, kt, ft, n_slice=24):
        self.Omega = float(Omega)
        self.NT = int(NT)
        self.k1, self.k2, self.kt = float(k1), float(k2), float(kt)
        self.ft = float(ft)
        self.phi_st, self.phi_ex = float(phi_st), float(phi_ex)

        # midpoint rule over the axial band; the tooth offsets are constant
        # in time, so the (tooth, slice) grid is built once
        z_edges = np.linspace(za_low, za_high, n_slice + 1)
        self.z = 0.5 * (z_edges[:-1] + z_edges[1:])
        self.dz = float(z_edges[1] - z_edges[0])

        lag_helix = self.z * np.tan(eta) / RT           # (n_slice,)
        lag_tooth = 2 * np.pi * np.arange(NT) / NT      # (NT,)
        # theta_offset[j, s] = 2 pi j / NT - z_s tan(eta)/R
        self.theta_offset = lag_tooth[:, None] - lag_helix[None, :]

        self.n_out_of_cut = 0
        self.n_engaged = 0

    # -----------------------------------------------------------------
    def _engaged_mask(self, theta):
        """True where the tooth is inside the angular engagement window."""
        # wrap into [phi_st, phi_st + 2pi) then test against the window
        w = np.mod(theta - self.phi_st, 2 * np.pi)
        return w <= (self.phi_ex - self.phi_st)

    def modal_scalar(self, t, u_now, u_del, clamp=True):
        """
        Return the scalar s such that the modal force is s * Dp.

        clamp=False disables the positivity test, which recovers the linear
        model exactly and is used by the verification script.
        """
        theta = self.Omega * t + self.theta_offset       # (NT, n_slice)
        eng = self._engaged_mask(theta)
        if not eng.any():
            return 0.0

        th = theta[eng]
        st, ct = np.sin(th), np.cos(th)
        h = self.ft * st + (u_del - u_now) * ct

        if clamp:
            live = h > 0.0
            self.n_engaged += th.size
            self.n_out_of_cut += int(np.count_nonzero(~live))
            if not live.any():
                return 0.0
            st, ct, h = st[live], ct[live], h[live]

        return float(self.kt * np.sum((self.k2 * st - self.k1 * ct) * h)
                     * self.dz)

    # -----------------------------------------------------------------
    def linear_coeffs(self, t):
        """
        (alpha3, alpha4) from the same quadrature, for cross-checking against
        milling_force.milling_force_coeffs.

            F_linear = ft * alpha3 + alpha4 * (u_del - u_now)
        """
        theta = self.Omega * t + self.theta_offset
        eng = self._engaged_mask(theta)
        if not eng.any():
            return 0.0, 0.0
        th = theta[eng]
        st, ct = np.sin(th), np.cos(th)
        w = self.k2 * st - self.k1 * ct
        a3 = self.kt * np.sum(w * st) * self.dz
        a4 = self.kt * np.sum(w * ct) * self.dz
        return float(a3), float(a4)

    def stats(self):
        frac = (self.n_out_of_cut / self.n_engaged) if self.n_engaged else 0.0
        return dict(n_engaged=self.n_engaged,
                    n_out_of_cut=self.n_out_of_cut,
                    fraction_out_of_cut=frac)

    def reset_stats(self):
        self.n_engaged = 0
        self.n_out_of_cut = 0


def chip_thickness_limits(ft, phi_st, phi_ex):
    """
    Maximum and mean uncut chip thickness over the engagement arc.

    Reporting these is ROADMAP item 17: a constant K_t is only defensible if
    the chip thickness it was calibrated at is stated, and at a few microns
    the size effect makes a single K_t questionable in the first place.
    """
    phi = np.linspace(phi_st, phi_ex, 512)
    h = ft * np.sin(phi)
    return float(np.max(h)), float(np.mean(h))
