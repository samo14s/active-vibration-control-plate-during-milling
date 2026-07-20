"""Validation of avc/milling.py (engagement integrals, regenerative
coefficients).

1. Down-milling engagement window geometry for ae = 0.1 mm, D = 10 mm.
2. mean_alpha4 is linear in the axial depth ap and NEGATIVE
   (down-milling directional factor -- the sign convention every
   downstream module relies on).
3. The time average of the periodic coefficient alpha4_time over one
   tooth-passing period matches the analytic mean_alpha4 within 1%.
4. Duty cycle (fraction of time in cut) at the reference operating point
   is consistent with the narrow radial immersion smeared by the helix.
5. alpha4_time is tau-periodic with tau = 60/(N_t * rpm).
"""
import numpy as np
import pytest

from avc.milling import alpha4_time, engagement_angles, mean_alpha4
from avc.params import DEFAULT

RPM = DEFAULT.process.rpm_ref            # 4900 rpm
AP = DEFAULT.process.ap_ref              # 0.3 mm
TAU = DEFAULT.tooth_passing_delay(RPM)


# --------------------------------------------------------------------------
# 1. engagement window
# --------------------------------------------------------------------------
def test_engagement_window_down_milling():
    # ae = 0.1 mm, D = 10 mm  ->  phi_e = acos(1 - 2 ae/D) = acos(0.98)
    phi_e = np.arccos(0.98)
    phi_in, phi_out = engagement_angles(DEFAULT)
    assert phi_in == pytest.approx(np.pi - phi_e, rel=1e-12)
    assert phi_out == pytest.approx(np.pi, rel=1e-12)
    assert 0.0 < phi_out - phi_in < 0.5 * np.pi  # narrow immersion window


# --------------------------------------------------------------------------
# 2. mean_alpha4: linear in ap, negative sign
# --------------------------------------------------------------------------
def test_mean_alpha4_linear_in_ap_and_negative():
    a1 = mean_alpha4(DEFAULT, AP)
    a2 = mean_alpha4(DEFAULT, 2.0 * AP)
    assert a1 < 0.0, f"down-milling mean alpha4 must be negative, got {a1}"
    assert a2 < 0.0
    assert a2 / a1 == pytest.approx(2.0, abs=1e-9)
    # magnitude sanity at the reference depth (verified hand value ~ -2532 N/m)
    assert 1.0e3 < abs(a1) < 1.0e4


# --------------------------------------------------------------------------
# 3. time average of alpha4_time matches mean_alpha4 within 1%
# --------------------------------------------------------------------------
def test_alpha4_time_average_matches_mean():
    n_t = 40000
    # midpoint sampling of one full tooth-passing period
    t = (np.arange(n_t) + 0.5) * (TAU / n_t)
    a4 = alpha4_time(DEFAULT, AP, RPM, t, n_axial=60)
    a4_bar = float(np.mean(a4))
    a4_mean = mean_alpha4(DEFAULT, AP)
    assert a4_bar < 0.0
    assert a4_bar == pytest.approx(a4_mean, rel=0.01), (
        f"time avg {a4_bar:.2f} vs analytic {a4_mean:.2f} N/m")


# --------------------------------------------------------------------------
# 4. duty cycle at the reference point: helix-smeared narrow immersion
# --------------------------------------------------------------------------
def test_duty_cycle_reference_point():
    n_t = 20000
    t = (np.arange(n_t) + 0.5) * (TAU / n_t)
    a4 = alpha4_time(DEFAULT, AP, RPM, t)
    duty = np.count_nonzero(np.abs(a4) > 1e-9 * np.max(np.abs(a4))) / n_t
    assert 0.05 < duty < 0.35, f"duty cycle {duty:.3f} outside (0.05, 0.35)"


# --------------------------------------------------------------------------
# 5. tau-periodicity
# --------------------------------------------------------------------------
def test_alpha4_time_tau_periodic():
    # irrational-ish offset keeps samples off the engagement discontinuities
    t = (np.arange(600) + 0.317) * (TAU / 600.0)
    a4_a = alpha4_time(DEFAULT, AP, RPM, t)
    a4_b = alpha4_time(DEFAULT, AP, RPM, t + TAU)
    scale = np.max(np.abs(a4_a))
    assert scale > 0.0
    assert np.max(np.abs(a4_a - a4_b)) <= 1e-9 * scale, (
        "alpha4_time is not tau-periodic")
