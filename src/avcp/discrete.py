"""Exact discretization of the composite plant (modal + analog filter).

The accelerometer chain applies an ANALOG low-pass before sampling, so the
discrete-time plant seen by the controller must be the ZOH discretization
of the composite continuous system (modal plate + analog filter): only
then do above-Nyquist plate modes alias with the analog attenuation baked
in, exactly as in the physical hardware and in the substep-rate simulator.

For the harmonic disturbance states of the internal model, the coupling
into the plant over one sample is discretized EXACTLY for sinusoidal
inter-sample behaviour (not zero-order-held):

    x_{k+1} = Ad x_k + Re{ M(w) a_k },   a_k = phasor of d at t_k,
    M(w) = (e^{j w Ts} I - Ad) (j w I - A)^{-1} B_F,

which matters at the higher tooth-passing harmonics where a ZOH
approximation rotates the phase by tens of degrees per sample.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.signal as sig

from .modal import ModalModel
from .params import SystemParams


class CompositeCt:
    """Continuous composite system: states [eta, etad, x_filter].

    Inputs: V (actuator voltage), F (point force along the path, entering
    through a mode-shape row phi_c that is supplied at use time).
    Outputs: y = filtered accelerometer voltage-equivalent [m/s^2],
    w(x) = deflection at any path point via phi rows.
    """

    def __init__(self, model: ModalModel, sys: SystemParams):
        self.model = model
        n = model.n
        ba_f, aa_f = sig.butter(sys.sensor.lp_order,
                                2.0 * np.pi * sys.sensor.lp_cutoff,
                                btype="low", analog=True)
        Af, Bf, Cf, Df = sig.tf2ss(ba_f, aa_f)
        assert float(np.atleast_2d(Df)[0, 0]) == 0.0, "filter must be strictly proper"
        nf = Af.shape[0]
        self.n, self.nf = n, nf
        nx = 2 * n + nf
        self.nx = nx

        Om2 = model.omega ** 2
        TzO = 2.0 * model.zeta * model.omega
        A = np.zeros((nx, nx))
        A[:n, n:2 * n] = np.eye(n)
        A[n:2 * n, :n] = -np.diag(Om2)
        A[n:2 * n, n:2 * n] = -np.diag(TzO)
        # filter driven by acc = phi_acc @ etadd
        acc_eta = -model.phi_acc * Om2
        acc_etad = -model.phi_acc * TzO
        Bf1 = Bf.reshape(-1)
        A[2 * n:, :n] = np.outer(Bf1, acc_eta)
        A[2 * n:, n:2 * n] = np.outer(Bf1, acc_etad)
        A[2 * n:, 2 * n:] = Af
        self.A = A

        # input maps: V -> ba, F -> phi_c (phi_c applied at use time)
        self.B_V = np.zeros(nx)
        self.B_V[n:2 * n] = model.ba
        self.B_V[2 * n:] = Bf1 * float(model.phi_acc @ model.ba)
        # P maps a modal force vector g to the state derivative: B_F = P @ g
        P = np.zeros((nx, n))
        P[n:2 * n, :] = np.eye(n)
        P[2 * n:, :] = np.outer(Bf1, model.phi_acc)
        self.P = P

        self.C_y = np.zeros(nx)
        self.C_y[2 * n:] = np.asarray(Cf).reshape(-1)

    def C_w(self, phi_row: np.ndarray) -> np.ndarray:
        row = np.zeros(self.nx)
        row[:self.n] = phi_row
        return row


class CompositeDt:
    """ZOH discretization of CompositeCt at Ts, with exact sinusoid
    coupling matrices for a set of harmonic frequencies."""

    def __init__(self, ct: CompositeCt, Ts: float,
                 w_harm: np.ndarray | None = None):
        self.ct, self.Ts = ct, Ts
        nx = ct.nx
        n = ct.n
        # ZOH of [A, [B_V, P]] jointly via the augmented exponential
        nb = 1 + n
        Maug = np.zeros((nx + nb, nx + nb))
        Maug[:nx, :nx] = ct.A * Ts
        Maug[:nx, nx] = ct.B_V * Ts
        Maug[:nx, nx + 1:] = ct.P * Ts
        E = la.expm(Maug)
        self.Ad = E[:nx, :nx]
        self.Bd_V = E[:nx, nx]
        self.Pd = E[:nx, nx + 1:]          # ZOH map for modal force vectors

        # exact sinusoid coupling per harmonic: M_h = W_h @ g for force
        # vector g;  W_h = (e^{jwTs} I - Ad) (jw I - A)^{-1} P
        self.W = []
        if w_harm is not None:
            for w in w_harm:
                X = np.linalg.solve(1j * w * np.eye(nx) - ct.A, ct.P)
                self.W.append((np.exp(1j * w * Ts) * np.eye(nx) - self.Ad) @ X)

    def force_cols(self, g: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
        """Discrete coupling columns (z1, z2) for harmonic h and modal
        force vector g:  x+ += Re(M) z1 + Im(M) z2, M = W_h @ g."""
        M = self.W[h] @ g
        return np.real(M), np.imag(M)
