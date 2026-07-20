"""Validation of the deployment layer: roll-off composition, latency
model, controller order reduction, and the deployment certificates."""
import numpy as np
import pytest

from avc.params import DEFAULT
from avc import synthesis as syn

X_T = 0.03
AP = 0.3e-3


@pytest.fixture(scope="module")
def mm5():
    return syn.cached_reduce_model(DEFAULT, DEFAULT.design.n_modes_design,
                                   0.0)


@pytest.fixture(scope="module")
def mm12():
    return syn.cached_reduce_model(DEFAULT, DEFAULT.design.n_modes_full, 0.0)


@pytest.fixture(scope="module")
def k_raw(mm5):
    w = syn.make_weights(DEFAULT)
    return syn.hinf_synthesis(syn.build_design_plant(mm5, X_T, AP, w))


def test_deploy_composition(k_raw):
    K = syn.deploy(k_raw, DEFAULT)
    # 4 roll-off filter states appended (latency Pade NOT deployed)
    assert K.nk == k_raw.nk + 4
    assert K.meta["deployed"] and not K.meta["latency"]
    Ka = syn.deploy(k_raw, DEFAULT, latency=True)
    assert Ka.nk == k_raw.nk + 5           # + 1 Pade state
    assert Ka.meta["latency"]
    # low-frequency behaviour preserved by the composition (filter ~ 1)
    freqs = np.array([200.0, 500.0])
    Hr = syn._siso_frf(k_raw.Ak, k_raw.Bk, k_raw.Ck, freqs, D=k_raw.Dk)
    Hd = syn._siso_frf(K.Ak, K.Bk, K.Ck, freqs, D=K.Dk)
    np.testing.assert_allclose(np.abs(Hd), np.abs(Hr), rtol=0.05)


def test_deployed_pole_speed(k_raw):
    """gamma back-off keeps every deployed pole below the real-time cap."""
    K = syn.deploy(k_raw, DEFAULT)
    w_cap = 2.0 * np.pi * 0.3 * DEFAULT.design.ctrl_rate_hz
    assert np.abs(np.linalg.eigvals(K.Ak)).max() < w_cap


def test_reduce_controller_preserves_low_freq():
    """Residualizing a genuinely fast, small-residue mode is benign."""
    rng = np.random.default_rng(1)
    A = np.diag([-100.0, -200.0, -1.0e6])
    B = np.array([[1.0], [1.0], [0.05]])
    C = np.array([[2.0, -1.0, 0.05]])
    K = syn.Controller(Ak=A, Bk=B, Ck=C, Dk=np.zeros((1, 1)),
                       c_wmill=rng.normal(size=3))
    Kr = syn.reduce_controller(K, w_cap=1.0e4)
    assert Kr.nk == 2
    freqs = np.geomspace(1.0, 500.0, 20)
    H0 = syn._siso_frf(K.Ak, K.Bk, K.Ck, freqs, D=K.Dk)
    H1 = syn._siso_frf(Kr.Ak, Kr.Bk, Kr.Ck, freqs, D=Kr.Dk)
    np.testing.assert_allclose(H1, H0, rtol=1e-3)


def test_reduce_controller_n_keep():
    A = np.diag([-10.0, -1000.0, -1.0e5])
    K = syn.Controller(Ak=A, Bk=np.ones((3, 1)), Ck=np.ones((1, 3)),
                       Dk=np.zeros((1, 1)))
    assert syn.reduce_controller(K, w_cap=1e9, n_keep=1).nk == 1
    assert syn.reduce_controller(K, w_cap=1e9, n_keep=3).nk == 3


def test_certificates_hold(mm5, mm12, k_raw):
    K = syn.deploy(k_raw, DEFAULT)
    theta = [(x, 0.0) for x in (0.005, 0.05, 0.095)]
    Wa = syn.additive_uncertainty_weight(
        DEFAULT, DEFAULT.design.n_modes_design,
        DEFAULT.design.n_modes_full, theta)
    assert syn.rs_margin(DEFAULT, K, Wa, theta) < 1.0
    assert syn.sampled_rho(DEFAULT, K, mm12) < 1.0


def test_sampled_rho_open_loop(mm12):
    """Inactive controller (fast decaying state, zero output): sampled
    radius equals the plant's own slowest modal decay."""
    K0 = syn.Controller(Ak=np.array([[-1.0e4]]), Bk=np.zeros((1, 1)),
                        Ck=np.zeros((1, 1)), Dk=np.zeros((1, 1)))
    rho = syn.sampled_rho(DEFAULT, K0, mm12)
    dt_c = 1.0 / DEFAULT.design.ctrl_rate_hz
    expected = np.exp(-np.min(mm12.zeta * mm12.omega) * dt_c)
    assert rho == pytest.approx(expected, rel=1e-6)


def test_pade_delay_frf():
    """|Pade| = 1 at all frequencies; phase matches e^{-sT} at low freq."""
    T = 30.0e-6
    A, B, C, D = syn.pade_delay_ss(T)
    freqs = np.array([100.0, 1000.0, 3000.0])
    H = syn._siso_frf(A, B, C, freqs, D=D)
    np.testing.assert_allclose(np.abs(H), 1.0, rtol=1e-9)
    # 1st-order Pade phase accuracy: exact to ~0.3 % up to 1 kHz for
    # T = 30 us; ~2.5 % at 3 kHz (inherent to Pade(1), not a bug)
    np.testing.assert_allclose(np.angle(H[:2]),
                               -2.0 * np.pi * freqs[:2] * T, rtol=0.02)
    assert np.angle(H[2]) == pytest.approx(
        -2.0 * np.pi * freqs[2] * T, rel=0.05)
