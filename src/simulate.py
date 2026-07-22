"""
Fixed-step delay-differential-equation solver and closed-loop simulation harness.

The true plant (plant.MillingPlant) is a delay differential equation.  We
integrate it with a classic 4th-order Runge-Kutta scheme at the simulation rate
FS, holding the control voltage constant across each step (zero-order hold, as a
digital controller would).  Past states needed for the q(t-tau) term are taken
from a history buffer with linear interpolation.
"""

from __future__ import annotations
import numpy as np

from . import config as C


def simulate(plant, controller, t_sim=C.T_SIM, dt=C.DT,
             meas_noise=True, x0=None, control_on_at=0.0):
    """
    Run one closed-loop simulation.

    Parameters
    ----------
    plant       : plant.MillingPlant
    controller  : controllers.base.Controller
    t_sim       : horizon [s]
    control_on_at : time [s] before which the controller output is forced to 0
                    (lets chatter build up first, as in the paper's figures)

    Returns dict with time, y (displacement, noise-free), y_meas, u, and the
    modal state trajectory x.
    """
    n2 = 2 * plant.n
    nsteps = int(round(t_sim / dt))
    tau = plant.tau

    # history buffer of full states, long enough to cover the delay
    ndelay = int(np.ceil(tau / dt)) + 2
    hist = np.zeros((nsteps + 1, n2))
    if x0 is None:
        # small seed displacement so the (self-excited) chatter can grow
        x0 = np.zeros(n2)
        x0[0] = 1.0e-6      # 1 um seed in first modal coordinate
    hist[0] = x0

    tarr = np.zeros(nsteps + 1)
    yarr = np.zeros(nsteps + 1)
    ymeas = np.zeros(nsteps + 1)
    uarr = np.zeros(nsteps + 1)

    controller.reset()

    def x_delayed(t_query, upto_index):
        """Interpolate the delayed state x(t_query) from the history buffer."""
        if t_query <= 0.0:
            return hist[0]
        idx = t_query / dt
        i0 = int(np.floor(idx))
        if i0 >= upto_index:            # not yet computed -> clamp to latest
            return hist[upto_index]
        frac = idx - i0
        return (1.0 - frac) * hist[i0] + frac * hist[i0 + 1]

    x = x0.copy()
    for k in range(nsteps):
        t = k * dt
        tarr[k] = t
        y_true = plant.displacement(x)
        yarr[k] = y_true
        y_meas = plant.measure(x, noise=meas_noise)
        ymeas[k] = y_meas

        # controller (ZOH over the step)
        if t >= control_on_at:
            u = controller.update(t, y_meas)
        else:
            u = 0.0
        uarr[k] = u

        # RK4 step of the DDE, u held constant
        def f(ts, xs):
            xt = x_delayed(ts - tau, k)
            return plant.deriv(ts, xs, xt, u)

        k1 = f(t,            x)
        k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1)
        k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2)
        k4 = f(t + dt,       x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # guard against numerical blow-up (diverging chatter) -> stop cleanly
        if not np.all(np.isfinite(x)) or np.max(np.abs(x[:plant.n])) > 1e3:
            hist[k + 1] = hist[k]
            nsteps = k + 1
            break
        hist[k + 1] = x

    # last sample
    tarr[nsteps] = nsteps * dt
    yarr[nsteps] = plant.displacement(x)
    ymeas[nsteps] = plant.measure(x, noise=meas_noise)
    uarr[nsteps] = uarr[nsteps - 1] if nsteps > 0 else 0.0

    sl = slice(0, nsteps + 1)
    return dict(t=tarr[sl], y=yarr[sl], y_meas=ymeas[sl], u=uarr[sl],
                x=hist[sl], diverged=(nsteps < int(round(t_sim / dt))))
