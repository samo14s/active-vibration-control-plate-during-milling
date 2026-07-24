"""
machining_path.py
=================
A machining programme for a thin cantilever wall, and the sequence of
workpiece models it induces.

WHAT THIS REPRESENTS
--------------------
A thin wall is not machined in one pass. It is brought down from a thicker
blank in successive RADIAL layers, and each layer is swept in AXIAL steps of
height a_p down the wall. The tool therefore visits a two-dimensional grid of
(radial layer, axial band) passes, sweeping the full length in each.

Two things vary along that programme, and both matter for control:

  1. WHERE the tool is. The disturbance enters the modal equations along
     Dp(x_p) = the mode shapes at the contact point, which sweeps as the tool
     feeds. A fixed actuator therefore sees a disturbance direction that
     rotates in modal space -- and can become orthogonal to it, at which
     point that actuator cannot influence the disturbance at all.

  2. WHAT is left of the workpiece. Every completed pass removes material,
     so the mode shapes and frequencies that define both the plant and the
     disturbance direction change underneath the controller. For a
     cantilever the frequency shift is NOT monotone: removing material near
     the free end takes away inertia and raises f1, removing near the clamp
     takes away stiffness and lowers it.

A controller designed once on the blank, and a stability lobe diagram
computed once for the blank, describe only the first instant of this.

STATIONS
--------
`stations()` yields the operating points at which the plant is re-evaluated.
Re-solving the eigenproblem is the expensive step, so it is done at
`mode_updates_per_pass` points within each pass; between those points the
tool position still advances and Dp is re-evaluated from the current modes,
which is cheap. Setting mode_updates_per_pass=1 applies each pass's removal
in one step at its start, which is the usual quasi-static assumption; raising
it refines that and lets the assumption be tested rather than assumed.
"""
from __future__ import annotations

import numpy as np

from evolving_plate import EvolvingPlateModel


class MachiningProgramme:
    """
    Radial layers x axial steps over a cantilever wall.

    Parameters
    ----------
    lp, hp     : wall length and height [m]
    h_blank    : starting thickness [m]
    h_final    : target wall thickness [m]
    n_radial   : number of radial layers to get from h_blank to h_final
    ap         : axial engagement, i.e. height of one pass band [m]
    top_down   : sweep bands from the free edge downwards (the usual practice,
                 it leaves the stiffest material supporting the cut longest)
    """

    def __init__(self, lp, hp, h_blank, h_final, n_radial, ap, top_down=True):
        self.lp, self.hp = lp, hp
        self.h_blank, self.h_final = h_blank, h_final
        self.n_radial = int(n_radial)
        self.ap = ap
        self.top_down = top_down

        self.ae = (h_blank - h_final) / self.n_radial      # radial step
        self.n_axial = int(np.ceil(hp / ap))

        bands = []
        for j in range(self.n_axial):
            z_hi = hp - j * ap
            z_lo = max(0.0, z_hi - ap)
            bands.append((z_lo, z_hi))
        if not top_down:
            bands = bands[::-1]
        self.bands = bands

    @property
    def n_passes(self):
        return self.n_radial * self.n_axial

    def passes(self):
        """Yield (index, radial_layer, z_lo, z_hi) in machining order."""
        i = 0
        for r in range(self.n_radial):
            for (z_lo, z_hi) in self.bands:
                yield i, r, z_lo, z_hi
                i += 1

    def summary(self):
        return (f"{self.n_radial} radial layers of {self.ae*1e3:.2f} mm "
                f"x {self.n_axial} axial bands of {self.ap*1e3:.1f} mm "
                f"= {self.n_passes} passes, "
                f"{self.h_blank*1e3:.1f} -> {self.h_final*1e3:.1f} mm wall")


class PathModelSequence:
    """
    Builds the sequence of workpiece models induced by a MachiningProgramme.

    Each station carries everything the analyses need: the modal model at
    that point of the programme, the tool contact position, and the modal
    disturbance direction there.
    """

    def __init__(self, programme, rho, E, nu, actuator_patches,
                 obs_point, N1=30, N2=40, n_modes=3, zeta_modes=None,
                 d31=175e-12, h_Pa=0.7e-3, E_Pe=63e9, nu_Pe=0.35,
                 mode_updates_per_pass=1, n_x_stations=9, verbose=True):
        self.prog = programme
        self.actuator_patches = list(actuator_patches)
        self.obs_point = obs_point
        self.n_modes = n_modes
        self.mode_updates_per_pass = int(mode_updates_per_pass)
        self.n_x_stations = int(n_x_stations)
        self.verbose = verbose
        self.piezo = dict(d31=d31, h_Pa=h_Pa, E_Pe=E_Pe, nu_Pe=nu_Pe)

        self.plate = EvolvingPlateModel(
            programme.lp, programme.hp, programme.h_blank, rho, E, nu,
            N1=N1, N2=N2, n_modes=n_modes, zeta_modes=zeta_modes,
            verbose=False)

        if verbose:
            print(f"[PathModel] {programme.summary()}")
            print(f"[PathModel] mesh {N1}x{N2}, element row = "
                  f"{self.plate.ley*1e3:.2f} mm")
            self.plate.check_axial_resolution(programme.ap)
            print(f"[PathModel] blank f = {np.round(self.plate.freq_n, 1)} Hz")

    # -----------------------------------------------------------------
    def _actuator_directions(self):
        """
        Modal force per volt for each patch, as columns of H (n_modes x n_act).

        Each patch is applied in turn to the current plate so that its moment
        arm uses the LOCAL remaining thickness, then the resulting modal
        vector is collected.
        """
        cols = []
        for (x1, x2, z1, z2) in self.actuator_patches:
            self.plate.add_piezo_patch(x1, x2, z1, z2, **self.piezo)
            cols.append(self.plate.H_Pe_modal.copy())
        return np.column_stack(cols)

    # -----------------------------------------------------------------
    def stations(self):
        """
        Yield a dict per station, in machining order:

            pass_idx, radial_layer, z_lo, z_hi   the pass being cut
            x_tool                               contact point along the wall
            s_path                               cumulative path length [m]
            omega_n, zeta, freq_n                current modal model
            Dp                                   disturbance direction (n_modes,)
            D_obs                                sensor direction (n_modes,)
            H                                    actuator matrix (n_modes, n_act)
            thickness_min/max                    remaining wall thickness [m]
        """
        prog = self.prog
        s_path = 0.0
        upd = max(1, self.mode_updates_per_pass)

        for (ip, r, z_lo, z_hi) in prog.passes():
            z_mid = 0.5 * (z_lo + z_hi)
            # x windows at which material is committed within this pass
            edges = np.linspace(0.0, prog.lp, upd + 1)

            for k in range(upd):
                # commit the material removed over this segment of the pass
                self.plate.remove_material(edges[k], edges[k + 1],
                                           z_lo, z_hi, prog.ae)
                self.plate.solve_modes()
                self.plate.precompute_Dp(zp_pos=z_mid, n_pos=401)
                self.plate.set_observation(*self.obs_point)
                H = self._actuator_directions()

                xs = np.linspace(edges[k], edges[k + 1],
                                 max(2, self.n_x_stations // upd + 1))
                for xi in xs:
                    idx = int(np.clip(round(xi / prog.lp * 400), 0, 400))
                    yield dict(
                        pass_idx=ip, radial_layer=r, z_lo=z_lo, z_hi=z_hi,
                        x_tool=float(xi), s_path=s_path + float(xi),
                        omega_n=self.plate.omega_n.copy(),
                        freq_n=self.plate.freq_n.copy(),
                        zeta=self.plate.zeta_modes.copy(),
                        Dp=self.plate.Dp_array[:, idx].copy(),
                        D_obs=self.plate.D_obs.copy(),
                        H=H.copy(),
                        thickness_min=float(self.plate.thickness.min()),
                        thickness_max=float(self.plate.thickness.max()),
                    )
            s_path += prog.lp


def matched_fraction(H, Dp):
    """
    Fraction of the disturbance direction that the actuator set can reach.

    With actuator matrix H (n_modes x n_act), the reachable subspace is the
    column span of H. The matched fraction is the norm of the projection of
    the unit disturbance direction onto that span:

        gamma = || Q^T d || ,   Q an orthonormal basis of range(H),
                                d = Dp / ||Dp||

    gamma = 1 means the disturbance is fully input-matched and can in
    principle be cancelled; gamma = 0 means no combination of actuator
    voltages produces any modal force along the disturbance, so at that tool
    position the actuators cannot oppose it at all.

    For a single actuator this reduces to |cos(angle(H, Dp))|.
    """
    H = np.atleast_2d(H)
    if H.shape[0] < H.shape[1]:
        H = H.T
    d = np.asarray(Dp, float).ravel()
    nd = np.linalg.norm(d)
    if nd < 1e-30:
        return 0.0
    Q, _ = np.linalg.qr(H)
    r = np.linalg.matrix_rank(H, tol=1e-12 * max(1.0, np.linalg.norm(H)))
    Q = Q[:, :r]
    return float(np.linalg.norm(Q.T @ (d / nd)))
