"""Frozen-position closed-loop analysis: eigenvalues along the cutter path
and closed-loop FRFs at the cutting point for SLD computation.

The controller is linear time-varying through its schedule; at each frozen
cutter position it is an LTI discrete system, so classical eigenvalue and
Nyquist analysis applies pointwise.  Slow-variation arguments (the
schedule traverses the path in ~10 s versus millisecond loop dynamics)
extend the frozen conclusions to the time-varying loop; the time-domain
simulations provide the final verification.

The frozen discrete plant is the ZOH discretization of the COMPOSITE
continuous system (modal plate + analog anti-aliasing filter), so
above-Nyquist plate modes alias exactly as they do through the real
sampled hardware chain — with the analog attenuation included.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as sig

from .discrete import CompositeCt, CompositeDt
from .modal import ModalModel
from .params import SystemParams


def plant_discrete_ss(sys: SystemParams, model: ModalModel, xc: float):
    """Discrete frozen plant at cutter position xc.

    Inputs: [V_cmd (through the 1-step computation delay), F (ZOH force at
    the cutting point)].  Outputs: [y (sampled filtered accelerometer),
    w_c (deflection at the cutting point)].  States: [eta, etad, x_f, u_prev].
    """
    ct = CompositeCt(model, sys)
    dt = CompositeDt(ct, sys.control.Ts)
    phi_c = model.phi_at(xc)
    nmf = ct.nx
    nx = nmf + 1
    A = np.zeros((nx, nx))
    A[:nmf, :nmf] = dt.Ad
    A[:nmf, nmf] = dt.Bd_V
    B = np.zeros((nx, 2))
    B[nmf, 0] = 1.0
    B[:nmf, 1] = dt.Pd @ phi_c
    C = np.zeros((2, nx))
    C[0, :nmf] = ct.C_y
    C[1, :nmf] = ct.C_w(phi_c)
    D = np.zeros((2, 2))
    return A, B, C, D


def controller_frozen_ss(controller, xc: float):
    """Frozen LTI realization of a controller at position xc.

    Supports PDController and PshlqgController.  Input: y; output: u_cmd.
    """
    from .controllers import PDController, PshlqgController
    if isinstance(controller, PDController):
        kp, kd = controller.gains_at(xc)
        A = np.zeros((1, 1))
        B = np.ones((1, 1))
        C = np.array([[kd]])
        D = np.array([[-(kp + kd)]])
        return A, B, C, D
    if isinstance(controller, PshlqgController):
        p = controller.plant
        A, B, Cm = p.matrices(xc)
        L = controller._interp(controller.L_grid, xc)
        K = controller._interp(controller.K_grid, xc)
        Lam = controller._interp(controller.Lam_grid, xc)
        Lam0 = float(controller._interp(controller.Lam0_grid, xc))
        Kfull = np.zeros(p.nx)
        Kfull[p.idx_lqr] = K
        j = p.zoff
        if p.ndc:
            Kfull[j] += Lam0
            j += 1
        for h in range(p.H):
            Kfull[j] += np.real(Lam[h])
            Kfull[j + 1] += np.imag(Lam[h])
            j += 2
        # u = -Kfull x_hat ; x_hat+ = A x + B u + L (y - C x)
        Ac = A - np.outer(B, Kfull) - np.outer(L, Cm)
        Bc = L.reshape(-1, 1)
        Cc = -Kfull.reshape(1, -1)
        Dc = np.zeros((1, 1))
        return Ac, Bc, Cc, Dc
    raise TypeError(f"unsupported controller {type(controller)}")


def closed_loop_matrix(sys: SystemParams, model: ModalModel, controller,
                      xc: float) -> np.ndarray:
    """Discrete closed-loop A matrix (no cutting-force loop)."""
    Ap, Bp, Cp, Dp = plant_discrete_ss(sys, model, xc)
    Ac, Bc, Cc, Dc = controller_frozen_ss(controller, xc)
    npx, ncx = Ap.shape[0], Ac.shape[0]
    Acl = np.zeros((npx + ncx, npx + ncx))
    y_row = Cp[0]
    Acl[:npx, :npx] = Ap + np.outer(Bp[:, 0], Dc[0, 0] * y_row)
    Acl[:npx, npx:] = np.outer(Bp[:, 0], Cc[0])
    Acl[npx:, :npx] = np.outer(Bc[:, 0], y_row)
    Acl[npx:, npx:] = Ac
    return Acl


def classified_eigs(sys: SystemParams, model: ModalModel, controller,
                    xc: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed-loop eigenvalues and an internal-model membership mask.

    An eigenvalue is classified as internal-model type when more than
    half of its (unit-normalized) eigenvector energy lives in the
    controller's disturbance (phasor + DC) states — a structural
    criterion that, unlike frequency-window classification, cannot
    mistake a plate mode lying near a tooth-passing harmonic (e.g. mode 4
    at 1874.8 Hz vs the 4th harmonic at 1866.8 Hz) for an internal-model
    pole."""
    from .controllers import PshlqgController
    Acl = closed_loop_matrix(sys, model, controller, xc)
    ev, V = np.linalg.eig(Acl)
    if not isinstance(controller, PshlqgController):
        return ev, np.zeros(len(ev), dtype=bool)
    p = controller.plant
    ct = CompositeCt(model, sys)
    npx = ct.nx + 1
    zi = np.arange(npx + p.zoff, npx + p.zoff + p.nz)
    Vn = V / np.linalg.norm(V, axis=0, keepdims=True)
    frac = np.sum(np.abs(Vn[zi, :]) ** 2, axis=0)
    return ev, frac > 0.5


def frozen_stability_map(sys: SystemParams, model: ModalModel, controller,
                         x_grid: np.ndarray) -> np.ndarray:
    """Max |eig| of the frozen closed loop along the path."""
    out = np.zeros(len(x_grid))
    for i, xc in enumerate(x_grid):
        Acl = closed_loop_matrix(sys, model, controller, xc)
        out[i] = np.max(np.abs(np.linalg.eigvals(Acl)))
    return out


def _filter_frf(sys: SystemParams, w: np.ndarray) -> np.ndarray:
    ba_f, aa_f = sig.butter(sys.sensor.lp_order,
                            2.0 * np.pi * sys.sensor.lp_cutoff,
                            btype="low", analog=True)
    return sig.freqs(ba_f, aa_f, worN=w)[1]


def controller_freq_response(controller, xc: float, w: np.ndarray,
                             Ts: float) -> np.ndarray:
    Ac, Bc, Cc, Dc = controller_frozen_ss(controller, xc)
    n = Ac.shape[0]
    K = np.empty(len(w), dtype=complex)
    for i, wi in enumerate(w):
        z = np.exp(1j * wi * Ts)
        if n:
            K[i] = complex((Cc @ np.linalg.solve(z * np.eye(n) - Ac, Bc)
                            + Dc)[0, 0])
        else:
            K[i] = complex(Dc[0, 0])
    return K


def closed_loop_frf_nn(sys: SystemParams, model: ModalModel, controller,
                       xc: float, w: np.ndarray) -> np.ndarray:
    """Closed-loop FRF from cutting force to cutting-point deflection at a
    frozen position, for SLD computation.

    Hybrid continuous/discrete treatment: the plate and analog filter are
    continuous; the discrete controller contributes its z-domain response
    with the ZOH reconstruction  H_zoh = (1 - e^{-j w Ts}) / (j w Ts) and
    the one-sample computation delay e^{-j w Ts}.
    """
    G_wF = model.frf_force_to_disp(xc, xc, w)
    if controller is None:
        return G_wF
    Ts = sys.control.Ts
    Hf = _filter_frf(sys, w)
    G_yF = model.frf_force_to_acc(xc, w) * Hf
    G_yV = model.frf_voltage_to_acc(w) * Hf
    G_wV = model.frf_voltage_to_disp(xc, w)
    Kd = controller_freq_response(controller, xc, w, Ts)
    Hzoh = np.where(w > 0, (1.0 - np.exp(-1j * w * Ts)) / (1j * w * Ts), 1.0)
    Keff = Kd * Hzoh * np.exp(-1j * w * Ts)
    # V = Keff y ; y = G_yV V + G_yF F
    V = Keff * G_yF / (1.0 - Keff * G_yV)
    return G_wF + G_wV * V
