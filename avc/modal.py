"""Modal reduction and LPV state-space construction.

Reduced modal model (mass-normalized modes Phi, coordinates eta):

    eta_dd + diag(2 z_i w_i) eta_d + diag(w_i^2) eta
        = bu * u + bf(theta) * Fy(t)

    w_mill = bf(theta)^T eta          (deflection at the milling point)
    y_s    = cs^T eta + noise         (fixed sensor point)

theta = (x_T, height_removed): tool position and material-removal state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

from .fem_plate import PlateModel, build_plate
from .params import Params, milling_point
from .piezo import coupling_vector


@dataclass
class ModalModel:
    """Reduced modal model at a frozen material-removal state."""

    params: Params
    height_removed: float
    n_modes: int
    omega: np.ndarray        # (n,) natural angular frequencies [rad/s]
    zeta: np.ndarray         # (n,) modal damping ratios
    Phi: np.ndarray          # (n_free, n) mass-normalized mode shapes
    bu: np.ndarray           # (n,) modal piezo input  Phi^T Theta   [N/V]
    cs: np.ndarray           # (n,) sensor output row  (w at sensor point)
    model: PlateModel        # underlying FEM model

    @property
    def freqs_hz(self) -> np.ndarray:
        return self.omega / (2.0 * np.pi)

    def bf(self, x_T: float) -> np.ndarray:
        """Modal participation of the milling point at tool position x_T."""
        z_mill = milling_point(self.params, self.height_removed)
        # milling point sits on the receded top edge (slightly inside the
        # element to keep interpolation well-defined)
        z = min(z_mill, self.model.height) - 1e-9
        e = self.model.interp_w(x_T, z)
        return self.Phi.T @ e

    # ---- state-space builders ------------------------------------------
    def abcd(self, x_T: float, alpha4: float = 0.0):
        """(A, Bu, Bf, Cs, Cw) of the frozen-parameter design model.

        SIGN CONVENTION (load-bearing): the `alpha4` argument is the
        coefficient of the CURRENT-time force term  Fy_now = alpha4 * w(t),
        assembled into A as  -(W2 - alpha4 * bf bf^T).  With the signed
        coefficient a4 of avc.milling (a4 < 0 for down-milling here) and
        the physical force  Fy = a4 * (w(t-tau) - w(t)),  the current-time
        part is  -a4 * w(t):  callers realizing the full regenerative loop
        must pass  alpha4 = -a4  and inject the delayed part  +a4 *
        w(t-tau) through Bf themselves (avc/sld.py does exactly this).
        Passing the signed a4 directly is the DESIGN-plant convention used
        by synthesis (static cutting-stiffness shift only, no delayed
        term). alpha4 in [N/m] (already includes depth of cut).
        """
        n = self.n_modes
        bf = self.bf(x_T)
        W2 = np.diag(self.omega ** 2)
        Z = np.diag(2.0 * self.zeta * self.omega)
        A = np.block([
            [np.zeros((n, n)), np.eye(n)],
            [-(W2 - alpha4 * np.outer(bf, bf)), -Z],
        ])
        Bu = np.concatenate([np.zeros(n), self.bu])[:, None]
        Bf = np.concatenate([np.zeros(n), bf])[:, None]
        Cs = np.concatenate([self.cs, np.zeros(n)])[None, :]
        Cw = np.concatenate([bf, np.zeros(n)])[None, :]
        return A, Bu, Bf, Cs, Cw

    def frf_voltage(self, freqs_hz: np.ndarray, x_out: float,
                    z_out: float) -> np.ndarray:
        """Voltage -> displacement FRF at a point [m/V] (all n_modes)."""
        e = self.model.interp_w(x_out, z_out)
        c = self.Phi.T @ e
        w = 2.0 * np.pi * freqs_hz
        H = np.zeros(len(freqs_hz), dtype=complex)
        for i in range(self.n_modes):
            H += c[i] * self.bu[i] / (self.omega[i] ** 2 - w ** 2
                                      + 2j * self.zeta[i] * self.omega[i] * w)
        return H


# Frequency-updated FEM: the first-principles FEM natural frequencies
# (Table 4, 1.5-3.8% of the measured EMA) are corrected to the measured
# values, retaining the FEM mode shapes and couplings -- a standard modal
# frequency-updating step. Material removal is propagated onto the
# measured anchors by the FEM frequency ratio. Set False for the raw,
# uncorrected FEM (used to report the first-principles validation).
ANCHOR_MEASURED_FREQS = True


def modal_damping(params: Params, n: int) -> np.ndarray:
    z = list(params.plate.zeta)
    if n <= len(z):
        return np.array(z[:n])
    return np.array(z + [params.plate.zeta_high] * (n - len(z)))


def _raw_fem_omega(params: Params, n_modes: int,
                   height_removed: float) -> np.ndarray:
    """Lowest n_modes FEM natural angular frequencies [rad/s] (no update)."""
    fem = build_plate(params, height_removed)
    vals = spla.eigsh(fem.K, k=n_modes, M=fem.M, sigma=0.0, which="LM",
                      return_eigenvectors=False)
    return np.sqrt(np.maximum(np.sort(vals), 0.0))


def reduce_model(params: Params, n_modes: int, height_removed: float = 0.0,
                 anchor: bool | None = None) -> ModalModel:
    """FEM -> reduced modal model (lowest n_modes).

    anchor (default: the module flag ANCHOR_MEASURED_FREQS) corrects the
    first len(f_measured) natural frequencies to the measured EMA values
    (scaled by the FEM removal ratio for receded geometry), keeping the
    FEM mode shapes and piezo/sensor couplings.
    """
    if anchor is None:
        anchor = ANCHOR_MEASURED_FREQS
    fem = build_plate(params, height_removed)
    # smallest eigenvalues of (K, M) via shift-invert
    vals, vecs = spla.eigsh(fem.K, k=n_modes, M=fem.M, sigma=0.0,
                            which="LM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    omega = np.sqrt(np.maximum(vals, 0.0))
    # mass-normalize
    for i in range(n_modes):
        mi = vecs[:, i] @ (fem.M @ vecs[:, i])
        vecs[:, i] /= np.sqrt(mi)
        # deterministic sign: positive deflection at free top-right corner
        e = fem.interp_w(params.plate.lp * 0.999, fem.height * 0.999)
        s = e @ vecs[:, i]
        if s < 0:
            vecs[:, i] = -vecs[:, i]

    if anchor:
        f_meas = params.plate.f_measured
        n_anch = min(n_modes, len(f_meas))
        if height_removed == 0.0:
            for i in range(n_anch):
                omega[i] = 2.0 * np.pi * f_meas[i]
        else:                             # scale measured anchor by FEM ratio
            om_nom = _raw_fem_omega(params, n_modes, 0.0)
            for i in range(n_anch):
                omega[i] = 2.0 * np.pi * f_meas[i] * (omega[i] / om_nom[i])

    theta_vec = coupling_vector(fem)
    es = fem.interp_w(params.sensor.xs, min(params.sensor.zs,
                                            fem.height) - 1e-9)
    return ModalModel(
        params=params, height_removed=height_removed, n_modes=n_modes,
        omega=omega, zeta=modal_damping(params, n_modes), Phi=vecs,
        bu=vecs.T @ theta_vec, cs=vecs.T @ es, model=fem,
    )


def bf_sweep(mm: ModalModel, n_pos: int = 41) -> tuple[np.ndarray, np.ndarray]:
    """bf(x_T) over the feed path. Returns (positions, bf array (n_pos, n))."""
    lp = mm.params.plate.lp
    xs = np.linspace(0.01 * lp, 0.99 * lp, n_pos)
    return xs, np.array([mm.bf(x) for x in xs])
