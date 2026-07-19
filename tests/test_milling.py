"""Milling force and regenerative-stability validation."""
import copy

import numpy as np
import pytest

import avcp.milling as mm
from avcp import SystemParams, build_modal_model, simulate, MillingForce


@pytest.fixture(scope="module")
def sysp():
    return SystemParams()


def test_mean_force_pushes_wall_away(sysp):
    mf = MillingForce(sysp.milling)
    t = np.linspace(0.0, 5 * sysp.milling.tau, 800)
    F = mf.normal_force_series(t)
    # the tangential term can pull toward the tool above phi = atan(Kr),
    # but the engagement-window mean must push the wall away
    assert F.mean() < 0.0


def test_force_periodicity_at_tpf(sysp):
    mf = MillingForce(sysp.milling)
    t = np.linspace(0.0, 4 * sysp.milling.tau, 1600, endpoint=False)
    F = mf.normal_force_series(t)
    n4 = len(t) // 4
    assert np.allclose(F[:n4], F[n4:2 * n4], atol=1e-9)


def test_directional_factor_value(sysp):
    mil = copy.deepcopy(sysp.milling)
    mil.ae = mil.diameter / 2.0    # half immersion, phi in [0, pi/2]
    mf = MillingForce(mil)
    expected = mil.n_teeth * (0.5 - mil.Kr * np.pi / 4.0)
    assert mf.zoa_alpha_nn() == pytest.approx(expected, rel=1e-9)


def test_directional_factor_negative_at_low_immersion(sysp):
    mf = MillingForce(sysp.milling)
    assert mf.zoa_alpha_nn() < 0.0


def _linear_force_regen(self, t, w_now, w_lookup, ap=None):
    """Single-delay linear regeneration (no jump-out, no surface memory):
    the regime where the ZOA/Nyquist analysis is exact."""
    m = self.m
    ap = m.ap if ap is None else ap
    phis = np.mod(self.tooth_angles(t), 2.0 * np.pi)
    F = 0.0
    for phi in phis:
        if not (self.phi_st <= phi <= self.phi_ex):
            continue
        h = m.fz * np.sin(phi) + (w_now - w_lookup(1)) * np.cos(phi)
        dFt = m.Kt * ap * h
        F += dFt * (np.sin(phi) - m.Kr * np.cos(phi))
    return F


@pytest.mark.slow
def test_nyquist_a_crit_matches_linear_time_domain(sysp):
    """The direct-Nyquist critical depth agrees with the growth threshold
    of the linearized (no tooth jump-out) time-domain simulation."""
    sp = copy.deepcopy(sysp)
    sp.milling.ae = 0.006
    sp.milling.x_start, sp.milling.x_end = 0.06, 0.06001
    sp.milling.pass_time = 3.0
    model, fem = build_modal_model(sp)
    mf = MillingForce(sp.milling)
    frf = lambda w: model.frf_force_to_disp(0.06, 0.06, w)
    a_crit = mf.a_crit_nyquist(frf, sp.milling.rpm)

    orig = mm.MillingForce.normal_force_regen
    mm.MillingForce.normal_force_regen = _linear_force_regen
    try:
        def grows(ap_test):
            spx = copy.deepcopy(sp)
            spx.milling.ap = ap_test
            mx, _ = build_modal_model(spx, fem=fem)
            with np.errstate(over="ignore", invalid="ignore"):
                r = simulate(spx, mx, controller=None, T=3.0)
            s1 = np.nanstd(r.w_cut[(r.t > 1.0) & (r.t < 1.5)])
            s2 = np.nanstd(r.w_cut[r.t > 2.5])
            return (not np.isfinite(s2)) or s2 > 3.0 * s1

        assert not grows(0.7 * a_crit)
        assert grows(1.4 * a_crit)
    finally:
        mm.MillingForce.normal_force_regen = orig


@pytest.mark.slow
def test_supercritical_chatter_is_bounded(sysp):
    """At mid-path the nominal cut is beyond the linear regenerative
    limit; the surface-memory (multi-delay) mechanism and the chip cap
    must bound the chatter at physically plausible amplitudes instead of
    diverging."""
    sp = copy.deepcopy(sysp)
    sp.milling.x_start, sp.milling.x_end = 0.06, 0.060001
    sp.milling.pass_time = 5.0
    model, _ = build_modal_model(sp)
    r = simulate(sp, model, controller=None, T=5.0)
    late = r.w_cut[r.t > 2.0]
    assert np.all(np.isfinite(late))
    assert np.max(np.abs(late)) < 2e-3      # bounded (sub-mm scale)
    assert np.std(late) > 2e-6              # but clearly vibrating
