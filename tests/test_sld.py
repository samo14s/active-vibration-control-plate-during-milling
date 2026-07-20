"""Validation of avc/sld.py (first-order semi-discretization).

1. Analytic turning benchmark: constant coefficient K < 0, 1-mode plant;
   SD critical K vs. the exact characteristic-equation boundary (< 3%).
2. Convergence of the spectral radius in the discretization number m.
3. Open-loop plate sanity: critical depth in the ballpark of the
   reference paper's experimental ~0.1 mm, and rho crossing 1 at a_lim.
4. Closed-loop smoke: static-negative-feedback lowpass controller changes
   rho; zero controller reproduces open loop; delayed-tap path runs.
"""
import numpy as np
import pytest

from avc import sld
from avc.controller import Controller
from avc.modal import reduce_model
from avc.params import DEFAULT

RPM = 4900.0
XT = 0.03
TAU = DEFAULT.tooth_passing_delay(RPM)


@pytest.fixture(scope="module")
def mm3():
    return reduce_model(DEFAULT, 3)


# --------------------------------------------------------------------------
# 1. analytic benchmark: turning (constant coefficient) stability limit
# --------------------------------------------------------------------------
class TurningStub:
    """1-mode plant exposing the ModalModel interface used by avc.sld.

    Mass-normalized SDOF: eta_dd + 2 z wn eta_d + wn^2 eta = b*Fy,
    w_mill = b*eta. abcd follows the ModalModel.abcd sign convention:
    stiffness block wn^2 - alpha4*b^2 (affine in alpha4).
    """

    n_modes = 1
    params = DEFAULT

    def __init__(self, f_hz=540.0, zeta=0.0031, b=3.0):
        self.omega_n = 2.0 * np.pi * f_hz
        self.zeta_n = zeta
        self.b = b

    def abcd(self, x_T, alpha4=0.0):
        w2 = self.omega_n ** 2
        b = self.b
        A = np.array([[0.0, 1.0],
                      [-(w2 - alpha4 * b * b),
                       -2.0 * self.zeta_n * self.omega_n]])
        Bu = np.array([[0.0], [1.0]])
        Bf = np.array([[0.0], [b]])
        Cs = np.array([[b, 0.0]])
        Cw = np.array([[b, 0.0]])
        return A, Bu, Bf, Cs, Cw


def _turning_K_exact(stub, tau):
    """Exact critical K < 0 from 1 + K b^2 G(jw)(1 - e^{-jw tau}) = 0.

    Boundary requires H(w) = b^2 G(jw)(1 - e^{-jw tau}) real; scan Im H
    for sign changes, refine by bisection, K* = -1/Re H(w*). The
    critical K is the negative boundary closest to zero (first crossing
    reached from the stable K = 0).
    """
    wn, z, b = stub.omega_n, stub.zeta_n, stub.b

    def H(w):
        G = 1.0 / (wn ** 2 - w ** 2 + 2j * z * wn * w)
        return b * b * G * (1.0 - np.exp(-1j * w * tau))

    ws = np.linspace(0.2 * wn, 2.5 * wn, 300001)
    im = H(ws).imag
    idx = np.nonzero(np.sign(im[:-1]) * np.sign(im[1:]) < 0)[0]
    K_neg = []
    for i in idx:
        a, c = ws[i], ws[i + 1]
        for _ in range(60):
            mid = 0.5 * (a + c)
            if np.sign(H(a).imag) * np.sign(H(mid).imag) <= 0:
                c = mid
            else:
                a = mid
        Hs = H(0.5 * (a + c))
        if abs(Hs) < 1e-12:      # zero of (1 - e^{-jw tau}): no finite K
            continue
        K = -1.0 / Hs.real
        if K < 0.0:
            K_neg.append(K)
    assert K_neg, "no negative stability boundary found in the scan"
    return max(K_neg)


def test_turning_benchmark():
    stub = TurningStub()
    K_ex = _turning_K_exact(stub, TAU)
    assert K_ex < 0.0

    def rho(K):
        return sld.spectral_radius(stub, 0.0, 0.0, RPM, m=64,
                                   a4_fn=lambda t: np.full_like(t, K))

    # first crossing of rho = 1 walking from K = 0 toward 1.5*K_ex
    grid = np.linspace(0.0, 1.5 * K_ex, 31)
    rhos = np.array([rho(K) for K in grid])
    assert rhos[0] < 1.0
    assert np.any(rhos >= 1.0)
    j = int(np.argmax(rhos >= 1.0))
    lo, hi = grid[j - 1], grid[j]
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if rho(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    K_sd = 0.5 * (lo + hi)
    assert abs(K_sd - K_ex) <= 0.03 * abs(K_ex), (K_sd, K_ex)


# --------------------------------------------------------------------------
# 2. convergence in the discretization number m
# --------------------------------------------------------------------------
def test_convergence_in_m(mm3):
    r48 = sld.spectral_radius(mm3, XT, 0.2e-3, RPM, m=48)
    r96 = sld.spectral_radius(mm3, XT, 0.2e-3, RPM, m=96)
    assert abs(r96 - r48) <= 0.01 * abs(r48), (r48, r96)


# --------------------------------------------------------------------------
# 3. open-loop plate sanity
# --------------------------------------------------------------------------
def test_open_loop_critical_depth(mm3):
    ap_c = sld.critical_depth(mm3, XT, RPM, m=48)
    # reference paper's experimental open-loop limit is ~0.1 mm
    assert 0.02e-3 <= ap_c <= 1.0e-3, ap_c
    assert sld.spectral_radius(mm3, XT, 0.5 * ap_c, RPM, m=48) < 1.0
    assert sld.spectral_radius(mm3, XT, 2.0 * ap_c, RPM, m=48) > 1.0


# --------------------------------------------------------------------------
# 4. closed-loop smoke tests
# --------------------------------------------------------------------------
def _lowpass(g, f0=800.0, z0=0.7):
    """u = -g * LP2(y): 2nd-order lowpass, DC gain -g [V/m]."""
    w0 = 2.0 * np.pi * f0
    return Controller(Ak=np.array([[0.0, 1.0], [-w0 ** 2, -2.0 * z0 * w0]]),
                      Bk=np.array([[0.0], [w0 ** 2]]),
                      Ck=np.array([[-g, 0.0]]),
                      Dk=np.array([[0.0]]),
                      label=f"lp2(g={g:g})")


def test_closed_loop_smoke(mm3):
    ap = 0.3e-3
    rho_ol = sld.spectral_radius(mm3, XT, ap, RPM, m=48)

    # zero controller (Bk = 0, Ck = 0, stable Ak): open loop reproduced
    zero = Controller(Ak=_lowpass(0.0).Ak, Bk=np.zeros((2, 1)),
                      Ck=np.zeros((1, 2)), Dk=np.zeros((1, 1)))
    rho_z = sld.spectral_radius(mm3, XT, ap, RPM, controller=zero, m=48)
    assert abs(rho_z - rho_ol) <= 1e-6 * max(1.0, rho_ol)

    # static negative output feedback changes rho and stays finite
    rho_g = sld.spectral_radius(mm3, XT, ap, RPM, controller=_lowpass(1e4),
                                m=48)
    assert np.isfinite(rho_g)
    assert abs(rho_g - rho_ol) > 1e-6 * max(1.0, rho_ol)

    # delayed-tap channel (gd0/gd1) runs, stays finite, changes rho
    base = _lowpass(1e4)
    base.c_wmill = np.array([1.0, 0.0])
    tap = base.with_delayed_feedback(1e3)
    rho_t = sld.spectral_radius(mm3, XT, ap, RPM, controller=tap, m=48)
    assert np.isfinite(rho_t)
    assert abs(rho_t - rho_g) > 1e-9
