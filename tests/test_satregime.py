"""Validation of the saturated-regime extension (psi-injection
structure, circle/small-gain margins, anti-windup plumbing) and of the
SDOF (Ozsoy-class) module."""
import numpy as np
import pytest

from avc.satcert import certify_sampled, saturated_margins
from avc.sdof import (REPRESENTATIVE, lift_sdof, simulate_sdof,
                      tooth_period)


@pytest.fixture(scope="module")
def lift():
    return lift_sdof(REPRESENTATIVE, 0.2e-3, 3000.0)


def test_psi_columns_strictly_causal(lift):
    """Vpsi must be strictly lower-triangular: a deadzone event at
    tick i cannot influence the commanded output before tick i+1."""
    assert np.allclose(np.triu(lift.Vpsi), 0.0)


def test_gpsi_matches_direct_impulse(lift):
    """Column i of Gpsi equals one period of propagation of a unit
    psi injected at tick i (independent reconstruction through Phi and
    a shifted lift is circular, so check the defining property instead:
    injecting psi at the LAST tick must equal the pending-force basis
    vector mapped by nothing further — i.e. -e_upend)."""
    iu = 2                      # [x, xdot | f_pend | hist]
    e_u = np.zeros(lift.N)
    e_u[iu] = -1.0
    assert np.allclose(lift.Gpsi[:, -1], e_u)


def test_margin_ordering(lift):
    """lambda_max(Re L) <= sigma_max(L) pointwise, hence the grid
    maxima obey circle_max <= gamma_2."""
    m = saturated_margins(lift, n_grid=128, n_refine=8)
    assert m["circle_max"] <= m["gamma_2"] + 1e-9
    assert np.isfinite(m["circle_max"])


def test_sdof_forced_peak_matches_simulator():
    """The SDOF lift's forced actuation peak matches the settled
    nonlinear simulator (the same audit as for the plate lift)."""
    cfg = REPRESENTATIVE
    rpm, ap = 3000.0, 0.2e-3
    lift = lift_sdof(cfg, ap, rpm)
    v_model = float(np.max(np.abs(lift.forced_voltages())))
    tau = tooth_period(cfg, rpm)
    r = simulate_sdof(cfg, ap, rpm, T=200 * tau, fs=50e3)
    assert r["diverged"] is None
    w = r["t"] >= 180 * tau
    v_sim = float(np.max(np.abs(r["f_act"][w])))
    assert v_model == pytest.approx(v_sim, rel=0.05)


def test_aw_changes_only_psi_columns():
    """Anti-windup must leave every LINEAR object untouched (Phi,
    forced orbit, region-of-linearity certificate) and act only on
    the deadzone-injection columns."""
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                           / "scripts"))
    from avc import synthesis as syn
    from avc.params import DEFAULT
    from avc.satcert import lift_sampled
    P = DEFAULT
    mm5 = syn.cached_reduce_model(P, P.design.n_modes_design, 0.0)
    mm = syn.cached_reduce_model(P, P.design.n_modes_full, 0.0)
    w = syn.make_weights(P)
    raw = syn.hinf_synthesis(syn.build_design_plant(mm5, 0.05, 0.3e-3, w))
    K = syn.deploy(raw, P, latency=False)
    l0 = lift_sampled(mm, 0.05, 0.3e-3, 4900.0, K)
    l1 = lift_sampled(mm, 0.05, 0.3e-3, 4900.0, K,
                      aw_gain=0.5 * np.ones(K.nk))
    assert np.allclose(l0.Phi, l1.Phi)
    assert np.allclose(l0.acc, l1.acc)
    assert np.allclose(l0.V, l1.V)
    assert not np.allclose(l0.Gpsi, l1.Gpsi)
    c0 = certify_sampled(l0, 150.0)
    c1 = certify_sampled(l1, 150.0)
    if c0.feasible and c1.feasible:
        assert c0.h_max == pytest.approx(c1.h_max, rel=1e-9)
