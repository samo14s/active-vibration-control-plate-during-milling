"""Modal reduction of the plate-actuator system and scheduling data.

The controller-design and evaluation ("truth") models are both modal
truncations of the same FEM model, with different orders, so that
observation/control spillover is represented honestly in closed-loop
evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .fem import PlateFem
from .params import SystemParams


@dataclass
class ModalModel:
    """Modal model  eta_k'' + 2 zeta_k w_k eta_k' + w_k^2 eta_k = b_a,k V + phi_c,k(t) f_c.

    Attributes
    ----------
    freqs : natural frequencies [Hz], shape (n,)
    omega : angular frequencies [rad/s]
    zeta  : damping ratios, shape (n,)
    ba    : modal piezo input vector, shape (n,)
    phi_acc : accelerometer mode-shape row, shape (n,)
    path_x  : path grid positions [m], shape (np,)
    phi_path : mode shapes at path grid, shape (np, n) — used both for the
               moving force input and the moving performance output
    """
    freqs: np.ndarray
    zeta: np.ndarray
    ba: np.ndarray
    phi_acc: np.ndarray
    path_x: np.ndarray
    phi_path: np.ndarray

    @property
    def n(self) -> int:
        return len(self.freqs)

    @property
    def omega(self) -> np.ndarray:
        return 2.0 * np.pi * self.freqs

    def truncate(self, n_modes: int) -> "ModalModel":
        s = slice(0, n_modes)
        return ModalModel(self.freqs[s], self.zeta[s], self.ba[s],
                          self.phi_acc[s], self.path_x, self.phi_path[:, s])

    def phi_at(self, xc: float) -> np.ndarray:
        """Mode-shape row at cutter position xc (linear interpolation)."""
        return np.array([np.interp(xc, self.path_x, self.phi_path[:, k])
                         for k in range(self.n)])

    def phi_at_many(self, xc: np.ndarray) -> np.ndarray:
        return np.column_stack([np.interp(xc, self.path_x, self.phi_path[:, k])
                                for k in range(self.n)])

    # ------------------------------------------------------------ FRFs

    def frf_force_to_disp(self, x_in: float, x_out: float,
                          w: np.ndarray) -> np.ndarray:
        """FRF from a point force on the path at x_in to deflection at x_out."""
        gi = self.phi_at(x_in)
        go = self.phi_at(x_out)
        H = np.zeros(len(w), dtype=complex)
        for k in range(self.n):
            H += (go[k] * gi[k]) / (self.omega[k] ** 2 - w ** 2
                                    + 2j * self.zeta[k] * self.omega[k] * w)
        return H

    def frf_voltage_to_disp(self, x_out: float, w: np.ndarray) -> np.ndarray:
        go = self.phi_at(x_out)
        H = np.zeros(len(w), dtype=complex)
        for k in range(self.n):
            H += (go[k] * self.ba[k]) / (self.omega[k] ** 2 - w ** 2
                                         + 2j * self.zeta[k] * self.omega[k] * w)
        return H

    def frf_voltage_to_acc(self, w: np.ndarray) -> np.ndarray:
        H = np.zeros(len(w), dtype=complex)
        for k in range(self.n):
            H += (-w ** 2) * (self.phi_acc[k] * self.ba[k]) / (
                self.omega[k] ** 2 - w ** 2
                + 2j * self.zeta[k] * self.omega[k] * w)
        return H

    def frf_force_to_acc(self, x_in: float, w: np.ndarray) -> np.ndarray:
        gi = self.phi_at(x_in)
        H = np.zeros(len(w), dtype=complex)
        for k in range(self.n):
            H += (-w ** 2) * (self.phi_acc[k] * gi[k]) / (
                self.omega[k] ** 2 - w ** 2
                + 2j * self.zeta[k] * self.omega[k] * w)
        return H


def build_modal_model(sys: SystemParams,
                      thickness_map: np.ndarray | None = None,
                      n_modes: int | None = None,
                      fem: PlateFem | None = None) -> tuple[ModalModel, PlateFem]:
    """Assemble the FEM and reduce to a modal model."""
    cfg = sys.config
    n = n_modes or cfg.n_modes_eval
    if fem is None:
        fem = PlateFem(sys.plate, cfg.nx, cfg.ny, thickness_map)
    freqs, Phi = fem.modes(n)

    q = sys.piezo.moment_constant(sys.plate.h)
    Ba_full = fem.piezo_vector(sys.piezo.u1, sys.piezo.u2,
                               sys.piezo.v1, sys.piezo.v2, q)
    ba = Phi.T @ Ba_full

    phi_acc = fem.point_row_free(sys.sensor.x, sys.sensor.y) @ Phi

    path_x = np.linspace(sys.milling.x_start, sys.milling.x_end,
                         cfg.n_path_grid)
    phi_path = np.zeros((len(path_x), n))
    for ip, xp in enumerate(path_x):
        phi_path[ip] = fem.point_row_free(xp, sys.milling.y_path) @ Phi

    zeta = np.full(n, sys.plate.zeta)
    return ModalModel(freqs, zeta, ba, phi_acc, path_x, phi_path), fem


def removal_thickness_map(sys: SystemParams, x_cut: float) -> np.ndarray:
    """Per-element thickness scaling after the cutter has passed x_cut.

    Side milling removes a layer of radial depth ae from the plate face
    over the band [y_path - ap/2, y_path + ap/2], behind the cutter.
    """
    cfg, mil, plate = sys.config, sys.milling, sys.plate
    tm = np.ones((cfg.nx, cfg.ny))
    scale = (plate.h - mil.ae) / plate.h
    ax, by = plate.Lx / cfg.nx, plate.Ly / cfg.ny
    for i in range(cfg.nx):
        xc = (i + 0.5) * ax
        if xc > x_cut:
            continue
        for j in range(cfg.ny):
            yc = (j + 0.5) * by
            # element counted if its y-extent intersects the machined band
            if abs(yc - mil.y_path) <= mil.ap / 2.0 + by / 2.0:
                tm[i, j] = scale
    return tm
