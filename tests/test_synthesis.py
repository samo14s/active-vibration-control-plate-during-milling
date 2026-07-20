"""Validation of avc/synthesis.py: weights, generalized plant, DGKF
H-infinity synthesis, LQG / delayed-PD baselines, position-scheduled gain
grid, and the robust point design (R-HINF baseline)."""
import numpy as np
import pytest

from avc.params import DEFAULT
from avc import synthesis as syn

X_T = 0.03            # tool position for the nominal design [m]
AP = 0.3e-3           # design axial depth of cut [m]
X_GRID = np.linspace(0.02, 0.08, 5)
FREQS = np.geomspace(10.0, 2.5e4, 400)   # 400-point log grid [Hz]


@pytest.fixture(scope="module")
def mm3():
    return syn.cached_reduce_model(DEFAULT, 3, 0.0)


@pytest.fixture(scope="module")
def weights():
    return syn.make_weights(DEFAULT)


@pytest.fixture(scope="module")
def plant(mm3, weights):
    return syn.build_design_plant(mm3, X_T, AP, weights)


@pytest.fixture(scope="module")
def k_hinf(plant):
    return syn.hinf_synthesis(plant)


# ---------------------------------------------------------------- weights
def test_weights(weights):
    Af, Bf, Cf, Df = weights["Wf"]
    dc = float((Cf @ np.linalg.solve(-Af, Bf))[0, 0])
    assert dc == pytest.approx(DEFAULT.design.wf_dc_gain, rel=1e-9)
    assert np.all(Df == 0.0)                      # strictly proper
    assert syn.spectral_abscissa(Af) < 0.0
    # 2nd-order roll-off: one decade above the corner -> ~1 % of DC
    mag = np.abs(syn._siso_frf(Af, Bf, Cf, np.array([25000.0])))[0]
    assert mag < 0.02 * dc
    # Wu: 1st-order lead, DC value wu_gain, rising to 10x at high frequency
    Au, Bu, Cu, Du = weights["Wu"]
    wu_dc = float((Cu @ np.linalg.solve(-Au, Bu))[0, 0] + Du[0, 0])
    assert wu_dc == pytest.approx(DEFAULT.design.wu_gain, rel=1e-9)
    wu_hf = float(Du[0, 0])
    ratio = (DEFAULT.design.wu_pole_hz / DEFAULT.design.wu_zero_hz) \
        ** DEFAULT.design.wu_order
    assert wu_hf == pytest.approx(DEFAULT.design.wu_gain * ratio, rel=1e-9)
    assert syn.spectral_abscissa(Au) < 0.0
    # Wn: lead cascade, DC value wn_gain
    An, Bn, Cn, Dn = weights["Wn"]
    wn_dc = float((Cn @ np.linalg.solve(-An, Bn))[0, 0] + Dn[0, 0])
    assert wn_dc == pytest.approx(DEFAULT.design.wn_gain, rel=1e-9)
    assert syn.spectral_abscissa(An) < 0.0


def test_additive_uncertainty_weight_overbounds():
    theta = [(0.03, 0.0), (0.07, 0.0)]
    Wa = syn.additive_uncertainty_weight(DEFAULT, 3, 12, theta)
    Aa, Ba, Ca, Da = Wa
    assert syn.spectral_abscissa(Aa) < 0.0
    freqs = np.geomspace(50.0, 2.0e4, 400)
    E = syn._truncation_residual(DEFAULT, 3, 12, theta, freqs)
    mag = np.abs(syn._siso_frf(Aa, Ba, Ca, freqs) + Da[0, 0])
    assert np.all(mag >= E * (1.0 - 1e-9))


# ---------------------------------------------------------- design plant
def test_design_plant_structure(mm3, plant):
    assert np.all(plant.D11 == 0.0)
    assert np.all(plant.D22 == 0.0)
    assert np.linalg.matrix_rank(plant.D12) == 1   # full column rank (Wu)
    assert np.linalg.matrix_rank(plant.D21) == 1   # full row rank (Wn)
    assert plant.n_plant == 6
    # 6 design-model + 5 actuator chain (1 Pade + 4 roll-off) + 2 Wf
    # + 2 Wu + 2 Wn states
    assert plant.nx == 17 and plant.nw == 2 and plant.nz == 2
    # c_wmill = [bf(x_T)^T, 0] on the design-model block
    np.testing.assert_allclose(plant.c_wmill[:3], mm3.bf(X_T))
    assert np.all(plant.c_wmill[3:] == 0.0)
    # frozen design model at ap_design is stable (alpha4 < 0, down-milling)
    assert plant.meta["alpha4"] < 0.0
    assert syn.spectral_abscissa(plant.A) < 0.0


# ------------------------------------------------------------------ hinf
def test_hinf_nominal(plant, k_hinf):
    assert k_hinf.label == "hinf"
    assert np.isfinite(k_hinf.gamma) and 0.0 < k_hinf.gamma < 1e6
    Acl = syn.close_loop(plant, k_hinf)[0]
    assert syn.spectral_abscissa(Acl) < 0.0
    norm = syn.tzw_norm(plant, k_hinf, FREQS)
    assert norm <= 1.05 * k_hinf.gamma
    # observer structure bookkeeping
    assert k_hinf.c_wmill.shape == (plant.nx,)
    assert k_hinf.Ak.shape == (plant.nx, plant.nx)
    assert k_hinf.Bk.shape == (plant.nx, 1)
    assert k_hinf.Ck.shape == (1, plant.nx)


# ------------------------------------------------------------------- lqg
def test_lqg(mm3, plant, k_hinf):
    K = syn.lqg_design(mm3, X_T, AP,
                       q_w=DEFAULT.design.wf_dc_gain ** 2,
                       r_u=DEFAULT.design.wu_gain ** 2,
                       w_noise=1.0,
                       v_noise=DEFAULT.design.wn_gain ** 2)
    assert K.label == "lqg"
    # 6 modal + 5 actuator-chain states in the LQG design model
    assert K.Ak.shape == (11, 11) and K.c_wmill.shape == (11,)
    Acl = syn.close_loop(plant, K)[0]
    assert syn.spectral_abscissa(Acl) < 0.0
    # H-inf optimality: LQG cannot beat the H-inf level (1 % slack)
    n_lqg = syn.tzw_norm(plant, K, FREQS)
    n_hinf = syn.tzw_norm(plant, k_hinf, FREQS)
    assert n_lqg >= 0.99 * n_hinf


# ---------------------------------------------------------- delayed PD
def test_delayed_pd_contract():
    kp, kd = 2.0e5, 15.0
    K = syn.delayed_pd_controller(kp, kd)
    assert K.label == "dpd"
    assert K.Ak.shape == (2, 2) and K.Bk.shape == (2, 1)
    assert K.Ck.shape == (1, 2) and np.all(K.Ck == 0.0)
    assert K.Dk.shape == (1, 1) and np.all(K.Dk == 0.0)
    assert K.gd0.shape == (2,) and np.all(K.gd0 == 0.0)
    np.testing.assert_allclose(K.gd1, [kp, kd])
    # critically damped filter, both poles at -2*pi*f_filt
    wf = 2.0 * np.pi * 8000.0
    poles = np.linalg.eigvals(K.Ak)
    np.testing.assert_allclose(sorted(poles.real), [-wf, -wf], rtol=1e-6)
    np.testing.assert_allclose(poles.imag, [0.0, 0.0], atol=1e-6 * wf)
    # unity DC gain from y_s to state x1 (filtered y_s)
    x_dc = np.linalg.solve(-K.Ak, K.Bk)[:, 0]
    assert x_dc[0] == pytest.approx(1.0, rel=1e-9)
    assert x_dc[1] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------- gain schedule
def test_gain_schedule(mm3, weights):
    sc = syn.gain_schedule(DEFAULT, X_GRID, [0.0], AP, n_modes=3)
    # all frozen grid closed loops stable
    for i, x in enumerate(X_GRID):
        pl = syn.build_design_plant(mm3, float(x), AP, weights)
        Acl = syn.close_loop(pl, sc.controllers[i][0])[0]
        assert syn.spectral_abscissa(Acl) < 0.0
    # interpolated midpoints certified stable against frozen plants
    assert sc.validate_midpoints() < 0.0
    # interpolation reproduces the grid controllers at grid points (raw)
    K1 = sc.at_theta(float(X_GRID[1]), 0.0, deployed=False)
    np.testing.assert_allclose(K1.Ak, sc.controllers[1][0].Ak)
    np.testing.assert_allclose(K1.c_wmill, sc.controllers[1][0].c_wmill)
    # deployed controller appends the 4-state actuator roll-off filter
    # (the Pade latency block models sampling and is NOT deployed)
    K1d = sc.at_theta(float(X_GRID[1]), 0.0)
    assert K1d.nk == K1.nk + 4 and K1d.meta.get("deployed")
    # common gamma across the grid
    g = sc.gammas
    assert np.allclose(g, g[0, 0])


# ----------------------------------------------------- robust point design
def test_robust_point_design(mm3, weights):
    K = syn.robust_point_design(DEFAULT, X_GRID, AP)
    assert K.label == "rhinf"
    assert np.isfinite(K.gamma)
    # defining property: one controller, stable on ALL grid plants
    for x in X_GRID:
        pl = syn.build_design_plant(mm3, float(x), AP, weights)
        Acl = syn.close_loop(pl, K)[0]
        assert syn.spectral_abscissa(Acl) < 0.0
