"""
test_hybrid_validation.py
=========================
Auditable validation of src/hybrid_adrc_fopid.py (v1 and v2 structures).
These are the checks quoted in the module/study docstrings; run this file to
reproduce them:

  (1) FOPID branch zeroed        -> hybrid step() == ADRCController step()
                                    to machine precision (0.0)
  (2) v1 export_lti              == analytic (Gy + F)/(1 - Gu)         ~1e-14
  (3) v2 (leak+filter) export    == analytic (LP*Gy + F)/(1 - LP*Gu)   ~1e-15
  (4) v2 defaults (eps=0,wf=None) reproduce v1 step() bit-exactly
  (5) closed-loop consistency: deployed step() vs exported-LTI controller on a
      mode-1-like plant agree to < 1 % RMS (the residual is the ESO's
      one-sample u-channel delay -- the deployed convention, shared with
      ADRCController; the headline feasibility numbers use step() itself).

Run: python experiments/test_hybrid_validation.py
"""
import os, sys
import numpy as np
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from hybrid_adrc_fopid import HybridADRCFOPID
from adrc_control import ADRCController
from fopid_control import FOPIDController

DT = 2.5e-5
OUST = dict(wb=2 * np.pi * 5.0, wh=2 * np.pi * 8000.0, N=3)


class _P5:
    n_modes = 5
    D_obs = np.array([-0.9, 1.1, -0.5, -0.3, 0.4])
    H_Pe_modal = np.array([0.5, 0.3, -0.2, 0.1, -0.05])


class _P1:
    n_modes = 1
    D_obs = np.array([-0.9])
    H_Pe_modal = np.array([0.5])


def _ys(n=600, seed=1):
    rng = np.random.default_rng(seed)
    return (1e-6 * np.sin(2 * np.pi * 540 * DT * np.arange(n))
            + 2e-7 * rng.standard_normal(n))


def test_eso_branch_equals_adrc():
    wc, wo, b0 = 900.0, 5400.0, -0.6
    h = HybridADRCFOPID(_P5(), DT, wc, wo, b0, Kp=0, Ki=0, Kd=0,
                        lam=0.5, mu=0.5, u_max=1e9, **OUST)
    a = ADRCController(_P5(), dt=DT, wc=wc, wo=wo, b0=b0, u_max=1e9)
    u_prev, err = 0.0, 0.0
    for y in _ys():
        _, uh = h.step(None, u_prev, y)
        _, ua = a.step(None, u_prev, y)
        err = max(err, abs(uh - ua))
        u_prev = ua
    print(f"(1) hybrid(FOPID=0) vs ADRC: max|du| = {err:.3e}")
    assert err == 0.0


def _freq_check(eps, wf, tol):
    kw = dict(Kp=3e4, Ki=2e6, Kd=5e2, lam=0.6, mu=0.7, sign=-1.0, u_max=1e9)
    h = HybridADRCFOPID(_P5(), DT, 900.0, 5400.0, -0.6,
                        eps_leak=eps, wf=wf, **kw, **OUST)
    f = FOPIDController(_P5(), DT, **kw, **OUST)
    w = 2 * np.pi * np.logspace(np.log10(30), np.log10(6000), 25)
    Ch, F = h.freqresp(w), f.freqresp(w)
    relmax = 0.0
    for i, wi in enumerate(w):
        inv = np.linalg.solve(1j * wi * np.eye(3) - h.A_o,
                              np.column_stack([h.B_o, h.L_o]))
        Gu = (h.Ce @ inv[:, 0])[0]
        Gy = (h.Ce @ inv[:, 1])[0]
        LP = 1.0 if wf is None else wf / (1j * wi + wf)
        Cana = (LP * Gy + F[i]) / (1.0 - LP * Gu)
        relmax = max(relmax, abs(Ch[i] - Cana) / abs(Cana))
    assert relmax < tol, relmax
    return relmax


def test_export_v1_analytic():
    r = _freq_check(0.0, None, 1e-12)
    print(f"(2) v1 export vs analytic (Gy+F)/(1-Gu): max rel err = {r:.3e}")


def test_export_v2_analytic():
    r = _freq_check(50.0, 2 * np.pi * 1500, 1e-12)
    print(f"(3) v2 export vs analytic (LP*Gy+F)/(1-LP*Gu): max rel err = {r:.3e}")


def test_v2_defaults_reproduce_v1():
    kw = dict(Kp=3e4, Ki=2e6, Kd=5e2, lam=0.6, mu=0.7, sign=-1.0, u_max=1e9)
    h1 = HybridADRCFOPID(_P5(), DT, 900.0, 5400.0, -0.6, **kw, **OUST)
    h2 = HybridADRCFOPID(_P5(), DT, 900.0, 5400.0, -0.6,
                         eps_leak=0.0, wf=None, **kw, **OUST)
    u_prev, err = 0.0, 0.0
    for y in _ys(seed=2):
        _, u1 = h1.step(None, u_prev, y)
        _, u2 = h2.step(None, u_prev, y)
        err = max(err, abs(u1 - u2))
        u_prev = u1
    print(f"(4) v2 defaults == v1: max|du| = {err:.3e}")
    assert err == 0.0


def test_closed_loop_step_vs_export():
    wn, ze = 2 * np.pi * 540, 0.003
    Ap = np.array([[0, 1], [-wn ** 2, -2 * ze * wn]])
    Bp = np.array([[0.0], [0.5]])
    Cp = np.array([[-0.9, 0.0]])
    M = np.zeros((3, 3)); M[:2, :2] = Ap * DT; M[:2, 2:] = Bp * DT
    E = expm(M); Apd, Bpd = E[:2, :2], E[:2, 2:]

    def closed_loop(mode):
        h = HybridADRCFOPID(_P1(), DT, 700.0, 4200.0, -0.45,
                            Kp=2e4, Ki=1e6, Kd=3e2, lam=0.6, mu=0.7,
                            sign=-1.0, u_max=150.0,
                            eps_leak=50.0, wf=2 * np.pi * 1500, **OUST)
        Ac, Bc, Cc, Dc = h.export_lti()
        nz = Ac.shape[0]
        Mz = np.zeros((nz + 1, nz + 1))
        Mz[:nz, :nz] = Ac * DT; Mz[:nz, nz:] = Bc * DT
        Ez = expm(Mz); Ad, Bd = Ez[:nz, :nz], Ez[:nz, nz:]
        x = np.zeros(2); z = np.zeros(nz); u_prev = 0.0; out = []
        rng = np.random.default_rng(0)
        for k in range(8000):
            d = 1e-4 * np.sin(2 * np.pi * 245 * k * DT) \
                + 2e-5 * rng.standard_normal()
            y = (Cp @ x)[0]
            if mode == "step":
                _, u = h.step(None, u_prev, y)
            else:
                u = float(np.clip((Cc @ z)[0] + Dc[0, 0] * y, -150, 150))
                z = Ad @ z + Bd[:, 0] * y
            x = Apd @ x + Bpd[:, 0] * u + np.array([0.5 * DT * DT * d, DT * d])
            u_prev = u
            out.append(y)
        return np.array(out)

    y1, y2 = closed_loop("step"), closed_loop("lti")
    r1 = np.sqrt(np.mean(y1[2000:] ** 2))
    r2 = np.sqrt(np.mean(y2[2000:] ** 2))
    rel = abs(r1 - r2) / r2 * 100
    print(f"(5) closed-loop step vs export: RMS rel diff = {rel:.3f} %")
    assert rel < 1.0


if __name__ == "__main__":
    test_eso_branch_equals_adrc()
    test_export_v1_analytic()
    test_export_v2_analytic()
    test_v2_defaults_reproduce_v1()
    test_closed_loop_step_vs_export()
    print("ALL VALIDATION CHECKS PASSED")
