"""Validation of avc/simulate.py (nonlinear time-domain engine).

1. Loss-of-contact equivalence: at small amplitude no slice ever leaves
   the cut, so the per-slice unilateral model must reproduce the linear
   regenerative model exactly (run-level w_mill comparison plus a direct
   CutGeometry vs. milling.alpha4_time force identity).
2. Stability consistency at rpm=4900, x_T=0.03 (SLD critical depth
   ~0.25 mm there): ap=0.05 mm bounded, ap=1.5 mm growing by >5x
   between the first and last quarter of a 0.4 s run.
3. Voltage saturation at +-150 V with a huge static-gain controller.
4. Reproducibility: same seed -> identical histories; different seed ->
   different noise realization.
5. Smoke: moving tool + gain scheduling + delayed state tap.
"""
import numpy as np
import pytest

from avc import milling, sld
from avc.controller import Controller
from avc.modal import reduce_model
from avc.params import DEFAULT
from avc.simulate import CutGeometry, simulate

RPM = 4900.0
XT = 0.03
FS = 50e3
TAU = DEFAULT.tooth_passing_delay(RPM)


@pytest.fixture(scope="module")
def mm3():
    return reduce_model(DEFAULT, 3)


def _lowpass(g, f0=800.0, z0=0.7):
    """u = -g * LP2(y): 2nd-order lowpass, DC gain -g [V/m]."""
    w0 = 2.0 * np.pi * f0
    return Controller(Ak=np.array([[0.0, 1.0], [-w0 ** 2, -2.0 * z0 * w0]]),
                      Bk=np.array([[0.0], [w0 ** 2]]),
                      Ck=np.array([[-g, 0.0]]),
                      Dk=np.array([[0.0]]),
                      label=f"lp2(g={g:g})")


# --------------------------------------------------------------------------
# 1. loss-of-contact equivalence with the linear model
# --------------------------------------------------------------------------
def test_slice_force_matches_alpha4():
    """CutGeometry (no clamping) == a4(t)(w_tau - w) - a4(t) ft exactly."""
    ap = 0.3e-3
    geom = CutGeometry(DEFAULT, ap, RPM, n_axial=40)
    t = np.linspace(0.0, TAU, 200, endpoint=False)
    a4 = milling.alpha4_time(DEFAULT, ap, RPM, t, n_axial=40)
    assert abs(a4.mean() - milling.mean_alpha4(DEFAULT, ap)) \
        <= 0.02 * abs(milling.mean_alpha4(DEFAULT, ap))
    for w, wtau in [(0.0, 0.0), (3e-6, -4e-6), (-2e-6, 5e-6)]:
        F_lin = a4 * (wtau - w) - a4 * DEFAULT.process.ft
        F_geo = np.array([geom.force(tk, w, wtau, loss_of_contact=False)
                          for tk in t])
        assert np.max(np.abs(F_geo - F_lin)) <= 1e-10 * np.max(np.abs(F_lin))


def test_loss_of_contact_equivalence_small_amplitude(mm3):
    """Stable point: loc on/off give near-identical w_mill (chip never
    reverses at amplitudes << ft, so the clamp is never active)."""
    kw = dict(rpm=RPM, ap=0.05e-3, T=0.05, x0=XT, moving=False, fs=FS,
              noise_rms=0.0)
    r_on = simulate(mm3, loss_of_contact=True, **kw)
    r_off = simulate(mm3, loss_of_contact=False, **kw)
    peak = r_on.peak_w()
    assert 1e-8 <= peak <= 5e-5, peak          # sane forced-response scale
    assert np.max(np.abs(r_on.w_mill - r_off.w_mill)) <= 1e-12
    # result-container methods run and are consistent
    assert r_on.rms_w(0.01) <= r_on.peak_w()
    assert r_on.rms_u() == 0.0 and r_on.peak_u() == 0.0   # open loop
    f, P = r_on.psd_w(nperseg=1024)
    assert len(f) == len(P) and np.all(P >= 0.0)


# --------------------------------------------------------------------------
# 2. stability consistency with the semi-discretization ballpark
# --------------------------------------------------------------------------
def test_stable_depth_bounded(mm3):
    # cross-check the operating point against the frozen-parameter SLD
    ap_c = sld.critical_depth(mm3, XT, RPM, m=48)
    assert 0.05e-3 < ap_c < 1.5e-3, ap_c
    r = simulate(mm3, RPM, 0.05e-3, T=0.4, x0=XT, moving=False, fs=FS,
                 noise_rms=0.0)
    assert r.extras["diverged_at"] is None
    assert r.peak_w() < 50e-6, r.peak_w()


def test_unstable_depth_grows(mm3):
    r = simulate(mm3, RPM, 1.5e-3, T=0.4, x0=XT, moving=False, fs=FS,
                 noise_rms=0.0)
    assert r.extras["diverged_at"] is None
    q = len(r.t) // 4
    m_first = np.max(np.abs(r.w_mill[:q]))
    m_last = np.max(np.abs(r.w_mill[-q:]))
    assert m_last > 5.0 * m_first, (m_first, m_last)
    # chatter reaches macroscopic amplitude (>> forced response)
    assert m_last > 20e-6


# --------------------------------------------------------------------------
# 3. voltage saturation
# --------------------------------------------------------------------------
def test_saturation_clamps_at_vmax(mm3):
    huge = Controller(Ak=np.array([[-100.0]]), Bk=np.array([[0.0]]),
                      Ck=np.array([[0.0]]), Dk=np.array([[5e9]]),
                      label="huge-static-gain")
    r = simulate(mm3, RPM, 0.05e-3, controller=huge, T=0.03, x0=XT,
                 moving=False, fs=FS, seed=3, v_max=150.0)
    assert np.all(np.isfinite(r.w_mill))
    assert np.max(np.abs(r.u)) <= 150.0 + 1e-9
    assert np.max(r.u) >= 150.0 - 1e-9        # clamp reached, both signs
    assert np.min(r.u) <= -150.0 + 1e-9
    assert r.extras["sat_frac"] > 0.3


# --------------------------------------------------------------------------
# 4. reproducibility
# --------------------------------------------------------------------------
def test_seed_reproducibility(mm3):
    kw = dict(rpm=RPM, ap=0.05e-3, controller=_lowpass(1e3), T=0.02,
              x0=XT, moving=False, fs=FS)
    r1 = simulate(mm3, seed=0, **kw)
    r2 = simulate(mm3, seed=0, **kw)
    r3 = simulate(mm3, seed=1, **kw)
    for a, b in [(r1.y_s, r2.y_s), (r1.w_mill, r2.w_mill), (r1.u, r2.u)]:
        assert np.array_equal(a, b)
    assert not np.array_equal(r1.y_s, r3.y_s)
    assert not np.array_equal(r1.u, r3.u)


# --------------------------------------------------------------------------
# 5. moving tool + gain scheduling + delayed tap smoke
# --------------------------------------------------------------------------
class _Schedule:
    """Toy schedule: position-dependent lowpass gain + delayed tap."""

    def __init__(self):
        self.calls = []

    def at_theta(self, x_T, removal):
        self.calls.append((x_T, removal))
        base = _lowpass(1e3 * (1.0 + 10.0 * x_T))
        base.c_wmill = np.array([1.0, 0.0])
        return base.with_delayed_feedback(50.0)


def test_moving_scheduled_delayed_tap(mm3):
    sch = _Schedule()
    r = simulate(mm3, RPM, 0.05e-3, T=0.06, x0=0.02, moving=True, fs=FS,
                 noise_rms=0.0, scheduled=sch)
    assert np.all(np.isfinite(r.w_mill)) and np.all(np.isfinite(r.u))
    assert r.x_tool[-1] > r.x_tool[0]          # tool actually moved
    assert np.max(np.abs(r.u)) > 0.0           # controller active
    assert len(sch.calls) == 3                 # t = 0, 20, 40 ms
    assert sch.calls[1][0] > sch.calls[0][0]   # scheduled on new position
    assert "DR(" in r.extras["label"]
