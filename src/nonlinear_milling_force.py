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

SIZE EFFECT, EDGE FORCES AND PROCESS DAMPING (ROADMAP item 17)
--------------------------------------------------------------
A single constant K_t is only defensible if the chip thickness it was
calibrated at is stated. This model runs at h_mean = 2.0 um, deep in the
size-effect regime where the specific cutting pressure rises steeply as the
chip thins, so three optional refinements are provided. All default OFF, so
the model reduces exactly to the constant-K_t law unless they are switched on
and the switch is visible in the results.

  kt_model="power"   K_t(h) = K_tc * (h / h_ref)^(-x)
      The standard size-effect form. x ~ 0.2-0.3 for aluminium. Setting
      h_ref to the mean uncut chip makes K_tc the value the constant-law
      calibration corresponds to, so the two agree at the calibration point
      and diverge only where the size effect actually bites.

  K_te, K_re         edge (ploughing) coefficients, force per unit length of
      cutting edge independent of chip thickness. They dominate as h -> 0 and
      are why a purely proportional law under-predicts force at microns.

  c_pd               process damping. The flank rubs the wavy surface, and
      the standard equivalent-viscous representation is a force opposing the
      surface-normal velocity with coefficient

          C_pd = K_pd * a_p / V_c

      i.e. it scales with axial depth and falls with cutting speed. It is the
      reason low-speed lobes are far higher than a speed-independent model
      predicts, and it is absent from the baseline entirely.

CALIBRATION IS NOT SUPPLIED, DELIBERATELY
-----------------------------------------
The numerical coefficients for AL6061-T6 must be taken from a source that
publishes them, not invented here. The survey identified

    Wang, Hao, Wang, Hou & Lallart, Shock and Vibration 2018, Art. 3831825
    (open access; AL6061-T6, peripheral milling, 10 mm 4-tooth 35 deg helix
    -- the same process as this study), which publishes the coefficients as a
    chip-thickness POWER LAW rather than a constant.

Until those numbers are read off that paper, `kt_model` stays "constant" and
the size-effect machinery is exercised only in sensitivity studies. The
defaults are chosen so that switching a refinement on cannot silently change
a result: with the defaults every added term is exactly zero.

SCOPE
-----
Loss of contact, the size effect, edge forces and process damping are
representable. Tool runout, flank wear and thermal effects are not.
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
                 za_low, za_high, k1, k2, kt, ft, n_slice=24,
                 kt_model="constant", kt_exponent=0.25, h_ref=None,
                 K_te=0.0, K_re=0.0, c_pd=0.0):
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

        # size effect / edge / process damping (all inert by default)
        if kt_model not in ("constant", "power"):
            raise ValueError(f"unknown kt_model {kt_model!r}")
        self.kt_model = kt_model
        self.kt_exponent = float(kt_exponent)
        # default reference chip = mean over the engaged arc, so that a power
        # law and the constant law agree at the calibration point
        if h_ref is None:
            phi = np.linspace(phi_st, phi_ex, 256)
            h_ref = float(np.mean(ft * np.sin(phi)))
        self.h_ref = max(float(h_ref), 1e-12)
        self.K_te = float(K_te)
        self.K_re = float(K_re)
        self.c_pd = float(c_pd)

        self.n_out_of_cut = 0
        self.n_engaged = 0

    # -----------------------------------------------------------------
    def kt_of_h(self, h):
        """
        Specific cutting pressure at chip thickness h.

        "constant" returns kt unchanged. "power" applies the size-effect law
        K_t(h) = kt * (h/h_ref)^(-x), clipped at h -> 0 so a vanishing chip
        cannot produce an unbounded pressure; the edge coefficients are the
        physically correct way to carry the force as h -> 0.
        """
        if self.kt_model == "constant":
            return self.kt
        hh = np.maximum(h, 1e-4 * self.h_ref)
        return self.kt * (hh / self.h_ref) ** (-self.kt_exponent)

    # -----------------------------------------------------------------
    def process_damping_coeff(self):
        """
        Equivalent viscous process-damping coefficient, C_pd = K_pd a_p / V_c,
        expressed here directly as `c_pd` (N.s/m per unit modal displacement).

        Returned so a caller can fold it into the modal damping matrix; it
        opposes the surface-normal velocity and therefore acts exactly like
        added damping along Dp.
        """
        return self.c_pd

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

        w = self.k2 * st - self.k1 * ct
        # chip-proportional term, with the size effect if enabled
        f = np.sum(w * self.kt_of_h(h) * h)
        # edge (ploughing) term: independent of h, present whenever the edge
        # is engaged, and dominant as h -> 0
        if self.K_te or self.K_re:
            f += np.sum(self.K_te * st - self.K_re * ct)
        return float(f * self.dz)

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
