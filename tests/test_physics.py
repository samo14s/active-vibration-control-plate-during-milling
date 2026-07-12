"""Physics sanity tests for the milling simulation."""

import math

import numpy as np
import pytest

from milling_sim.engine import MillingSimulator
from milling_sim.identification import RLSModalIdentifier, remove_harmonics
from milling_sim.parameters import Mode, PaperSetup
from milling_sim.stability import (directional_matrix, fast_lobe_envelope,
                                   lobe_envelope, receptance_matrix,
                                   stability_lobes)


@pytest.fixture()
def setup():
    return PaperSetup()


# ---------------------------------------------------------------------------
# stability module
# ---------------------------------------------------------------------------
class TestStability:
    def test_directional_matrix_slot_milling_classical(self):
        """Full immersion, Kr = 0: axx = ayy = -pi/2 * ... classical values.

        For phi in [0, pi] with Kr = 0:
          axx = 1/2[cos 2phi]_0^pi = 0,  axy = 1/2[-sin2phi - 2phi] = -pi
          ayx = 1/2[-sin2phi + 2phi] = +pi, ayy = 0
        """
        A = directional_matrix(0.0, 0.0, math.pi)
        assert abs(A[0, 0]) < 1e-12
        assert abs(A[0, 1] + math.pi) < 1e-12
        assert abs(A[1, 0] - math.pi) < 1e-12
        assert abs(A[1, 1]) < 1e-12

    def test_receptance_peak_at_resonance(self, setup):
        m = setup.modes[0]
        w = m.wn(0.0)
        H = receptance_matrix(w, [m], 0.0)
        # at resonance, |H| = 1 / (2 zeta k) = 1 / (c * wn)
        expected = 1.0 / (2.0 * m.zeta(0.0) * m.k(0.0))
        assert abs(abs(H[1, 1]) - expected) / expected < 1e-9

    def test_fast_envelope_matches_loop_reference(self, setup):
        grid = np.linspace(8000, 16000, 161)
        Vr = 0.4
        modes_fz = [(m.wn(Vr) / (2 * math.pi), m.zeta(Vr), m.mass_kg)
                    for m in setup.modes]
        dirs = [m.direction for m in setup.modes]
        env_fast = fast_lobe_envelope(setup.tool, setup.engagement, modes_fz,
                                      grid, directions=dirs, n_freq=2400)

        class _M:
            def __init__(self, f, z, mm, d):
                self._f, self._z, self.mass_kg, self.direction = f, z, mm, d

            def wn(self, Vr):
                return 2 * math.pi * self._f

            def zeta(self, Vr):
                return self._z

        ml = [_M(f, z, mm, d) for (f, z, mm), d in zip(modes_fz, dirs)]
        pts = stability_lobes(setup.tool, setup.engagement, ml, 0.0,
                              n_freq=2400)
        env_loop = lobe_envelope(pts, grid)
        sel = np.isfinite(env_loop) & (env_loop < 0.02)
        rel = np.abs(env_fast[sel] - env_loop[sel]) / env_loop[sel]
        assert np.median(rel) < 0.08

    def test_more_damping_raises_boundary(self, setup):
        grid = np.linspace(8000, 16000, 81)
        base = [(m.f0_hz, m.zeta0, m.mass_kg) for m in setup.modes]
        damped = [(f, 2.0 * z, mm) for f, z, mm in base]
        dirs = [m.direction for m in setup.modes]
        e1 = fast_lobe_envelope(setup.tool, setup.engagement, base, grid,
                                directions=dirs)
        e2 = fast_lobe_envelope(setup.tool, setup.engagement, damped, grid,
                                directions=dirs)
        assert np.median(e2 / e1) > 1.5


# ---------------------------------------------------------------------------
# time-domain engine
# ---------------------------------------------------------------------------
class TestEngine:
    def test_no_cut_free_decay_matches_damping(self, setup):
        """With zero feed and an initial modal velocity, the wall mode must
        decay at its structural rate (validates the integrator)."""
        setup.sim.duration_s = 1.0
        sim = MillingSimulator(setup, seed=0, batch_scatter=0.0)
        m = setup.modes[0]
        sim.q[:] = 0.0
        sim.qd[:] = 0.0
        sim.qd[0] = 1e-3
        # ap = 0 -> zero cutting force regardless of engagement
        meas = sim.advance(12000.0, 0.0, 0.0)
        # amplitude after one control period (10 ms)
        wn, zeta = m.wn(sim.Vr) , m.zeta(sim.Vr)
        t = sim.t
        expected_env = 1e-3 / wn * math.exp(-zeta * wn * t)
        # RMS over the interval is of the same order as the envelope
        assert meas.rms_wall_um > 0.0
        q_amp = abs(sim.q[0]) + abs(sim.qd[0]) / wn
        assert q_amp < 1e-3 / wn * 1.05          # never grows
        assert q_amp > expected_env * 0.2        # not over-damped

    def test_mean_forces_scale_with_ap(self, setup):
        setup.sim.duration_s = 1.0
        f1, f2 = [], []
        for ap, sink in ((2.0, f1), (4.0, f2)):
            sim = MillingSimulator(setup, seed=3, batch_scatter=0.0)
            for _ in range(30):
                meas = sim.advance(12000.0, 1800.0, ap)
            sink.append(meas.mean_ft_N)
        assert f2[0] / f1[0] == pytest.approx(2.0, rel=0.15)

    def test_stable_vs_unstable_depth(self, setup):
        """Deep cut far above the boundary must produce far larger
        vibration than a shallow stable cut at the same speed."""
        setup.sim.duration_s = 3.0

        def run(ap):
            sim = MillingSimulator(setup, seed=5, batch_scatter=0.0)
            # jump to mid-pass conditions where 12 krpm is unstable
            sim.Vr = 0.45
            sim._update_modal_arrays()
            last = None
            while not sim.finished:
                last = sim.advance(12000.0, 1800.0, ap)
            return last.rms_wall_um

        stable = run(1.0)
        unstable = run(6.0)
        assert unstable > 5.0 * stable
        assert unstable > 50.0

    def test_removal_volume_bookkeeping(self, setup):
        setup.sim.duration_s = 2.0
        sim = MillingSimulator(setup, seed=1)
        while not sim.finished:
            meas = sim.advance(12000.0, 1800.0, 4.0)
        # Q = 1800 * 4 * 3 / 1000 = 21.6 cm3/min for 2 s = 0.72 cm3
        assert meas.Vr == pytest.approx(0.72 / 15.0, rel=1e-6)


# ---------------------------------------------------------------------------
# identification
# ---------------------------------------------------------------------------
class TestIdentification:
    def test_harmonic_removal_keeps_offgrid_tone(self):
        fs, f_rev = 10000.0, 200.0
        t = np.arange(1000) / fs
        harm = np.sin(2 * np.pi * 800.0 * t) + 0.5 * np.sin(
            2 * np.pi * 1600.0 * t)
        tone = 0.2 * np.sin(2 * np.pi * 1234.0 * t)
        resid = remove_harmonics(harm + tone, fs, f_rev)
        # harmonics removed, non-synchronous tone preserved
        assert np.std(resid) == pytest.approx(np.std(tone), rel=0.15)

    def test_rls_tracks_synthetic_mode(self):
        fs = 10000.0
        rng = np.random.default_rng(0)
        fn, zeta = 1400.0, 0.03
        # exact discrete AR(2) process with the oscillator's poles
        dt = 1.0 / fs
        wn = 2 * np.pi * fn
        wd = wn * math.sqrt(1 - zeta ** 2)
        r = math.exp(-zeta * wn * dt)
        a1 = 2 * r * math.cos(wd * dt)
        a2 = -r * r
        y = np.zeros(8000)
        for k in range(2, len(y)):
            y[k] = a1 * y[k - 1] + a2 * y[k - 2] + rng.standard_normal()
        ident = RLSModalIdentifier(fs=fs)
        for k in range(0, len(y) - 100, 100):
            ident.update(y[k: k + 100], f_rev=190.0)
        assert ident.fn_hz is not None
        assert abs(ident.fn_hz - fn) / fn < 0.03


# ---------------------------------------------------------------------------
# end-to-end sanity
# ---------------------------------------------------------------------------
class TestEndToEnd:
    def test_adaptive_beats_conventional_short(self, setup):
        from milling_sim.runner import make_controller, run_pass
        setup.sim.duration_s = 60.0
        setup.sim.removal_volume_cm3 = 8.0       # shorter pass for CI
        ha, sa = run_pass(setup, make_controller("adaptive", setup), seed=7)
        hc, sc = run_pass(setup, make_controller("conventional", setup),
                          seed=7)
        assert sa["avg_mrr_cm3_min"] > sc["avg_mrr_cm3_min"]
        assert sa["rms_max_um"] < 35.0
        assert sc["rms_max_um"] > 60.0
