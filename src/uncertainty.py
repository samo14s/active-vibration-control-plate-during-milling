"""
Paper additive dynamic uncertainty (Du et al. 2024, weight W_Pau, Eq. 19) and a
closed-loop simulator that injects it in the TIME DOMAIN.

The paper models the unmodelled high-mode dynamics of the plate as an ADDITIVE
perturbation on the actuator -> sensor path,

    G_true(s) = G(s) + Delta(s) * W_Pau(s),        ||Delta||_inf <= 1,
    W_Pau(s)  = rPau (s^2 + 2 za1 wa1 s + wa1^2) / (s^2 + 2 za2 wa2 s + wa2^2),

with the published coefficients below.  |W_Pau| is negligible at the two
controlled modes (540, 1068 Hz) but dominates in the inter-modal troughs and the
band above them (>~2 kHz, toward the RHP zero at ~5.2 kHz) -- exactly where an
aggressive, high-frequency-rich controller spills energy into the unmodelled
dynamics.  Robust stability to this set is the paper's mu criterion
||W_Pau * K S||_inf < 1: it is satisfied only by controllers whose control
sensitivity K S rolls off in the W_Pau band.

`simulate_with_uncertainty` augments the plant with the (scaled) perturbation
channel H = kappa * Delta * W_Pau driven by the control voltage; its output
(unmodelled displacement) is added to BOTH the measured and the physical sensor
signal.  The modal chatter dynamics are untouched (the uncertainty is parallel
on the u -> y path).  Delta is realised as a static +/-1 corner or a worst-phase
allpass; the caller sweeps and takes the worst.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import tf2ss, convolve

from . import config as C

# --- W_Pau(s) physical weight (V -> m), Eq. 19, published coefficients ---
_TP = 2.0 * np.pi
R_PAU, ZA1, ZA2 = 4.0e-7, 0.58, 0.22
WA1, WA2 = 1100.0 * _TP, 3500.0 * _TP
NUM_WPAU = R_PAU * np.array([1.0, 2.0 * ZA1 * WA1, WA1 ** 2])
DEN_WPAU = np.array([1.0, 2.0 * ZA2 * WA2, WA2 ** 2])

DELTAS = ("+1", "-1", "allpass")


def perturbation_ss(kind="+1", kappa=1.0, allpass_hz=2000.0):
    """State-space (A, B, C, D) of  H = kappa * Delta * W_Pau  for a Delta
    realisation.  kappa scales the uncertainty magnitude (kappa=1 = full paper
    weight)."""
    num = kappa * NUM_WPAU
    if kind == "0" or kappa == 0.0:
        return (np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), 0.0)
    if kind == "+1":
        den = DEN_WPAU
    elif kind == "-1":
        num, den = -num, DEN_WPAU
    elif kind == "allpass":                       # (-s + wc)/(s + wc): |.|=1, worst phase
        wc = allpass_hz * _TP
        num = convolve(num, [-1.0, wc])
        den = convolve(DEN_WPAU, [1.0, wc])
    else:
        raise ValueError(f"unknown Delta realisation {kind!r}")
    A, B, Cc, D = tf2ss(num, den)
    return A, B.reshape(-1, 1), Cc.reshape(1, -1), float(np.atleast_1d(D.ravel())[0])


def simulate_with_uncertainty(plant, controller, kind="+1", kappa=1.0,
                              t_sim=0.35, dt=C.DT, meas_noise=True):
    """Closed-loop RK4 DDE with the additive-uncertainty channel on the output.

    Returns dict(t, y, u, diverged) with y the physical sensor displacement
    (nominal plant + unmodelled channel), as the metrics see it."""
    Aw, Bw, Cw, Dw = perturbation_ss(kind, kappa)
    Cw = Cw.ravel(); Bw = Bw.ravel()
    nw = Aw.shape[0]
    n2 = 2 * plant.n
    nsteps = int(round(t_sim / dt))
    tau = plant.tau
    hist = np.zeros((nsteps + 1, n2))
    xm = np.zeros(n2); xm[0] = 1.0e-6
    hist[0] = xm.copy()
    xw = np.zeros(nw)
    controller.reset()
    tarr = np.zeros(nsteps + 1); yv = np.zeros(nsteps + 1); uv = np.zeros(nsteps + 1)
    rng = np.random.default_rng(C.SEED)

    def xdel(tq, upto):
        if tq <= 0.0:
            return hist[0]
        idx = tq / dt; i0 = int(np.floor(idx))
        if i0 >= upto:
            return hist[upto]
        fr = idx - i0
        return (1.0 - fr) * hist[i0] + fr * hist[i0 + 1]

    diverged = False
    for k in range(nsteps):
        t = k * dt
        tarr[k] = t
        # digital ZOH: the D feedthrough reflects the currently-held input u[k-1];
        # the strictly-proper state Cw@xw compensates the D-asymptote at low freq.
        u_prev = uv[k - 1] if k > 0 else 0.0
        y_unc = float(Cw @ xw) + Dw * u_prev
        y_phys = plant.displacement(xm) + y_unc
        yv[k] = y_phys
        y_meas = y_phys + (rng.normal(0.0, C.MEAS_NOISE_STD) if meas_noise else 0.0)
        u = controller.update(t, y_meas)
        uv[k] = u

        def f(ts, z):
            xt = xdel(ts - tau, k)
            dm = plant.deriv(ts, z[:n2], xt, u)
            dw = Aw @ z[n2:] + Bw * u
            return np.concatenate([dm, dw])

        z = np.concatenate([xm, xw])
        k1 = f(t, z); k2 = f(t + .5 * dt, z + .5 * dt * k1)
        k3 = f(t + .5 * dt, z + .5 * dt * k2); k4 = f(t + dt, z + dt * k3)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xm, xw = z[:n2], z[n2:]
        if not np.all(np.isfinite(z)) or np.max(np.abs(xm[:plant.n])) > 1e3:
            diverged = True; nsteps = k + 1; break
        hist[k + 1] = xm

    sl = slice(0, nsteps + 1)
    tarr[nsteps] = nsteps * dt
    return dict(t=tarr[sl], y=yv[sl], u=uv[sl], diverged=diverged)


def settled_rms_peakv(plant, controller, kind="+1", kappa=1.0, **kw):
    """(settled RMS in um, peak |u| in V) or None if it diverges / rails off."""
    r = simulate_with_uncertainty(plant, controller, kind=kind, kappa=kappa, **kw)
    y = np.asarray(r["y"]) * 1e6
    u = np.asarray(r["u"])
    if r["diverged"] or not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e4:
        return None
    n = len(y)
    return float(np.sqrt(np.mean(y[2 * n // 3:] ** 2))), float(np.max(np.abs(u)))


def worst_over_delta(plant_factory, controller_factory, kappa=1.0, **kw):
    """Worst (max-RMS) outcome over the Delta realisations; None if any diverges."""
    outs = [settled_rms_peakv(plant_factory(), controller_factory(), kind=kd,
                              kappa=kappa, **kw) for kd in DELTAS]
    if any(o is None for o in outs):
        return None
    return max(outs, key=lambda o: o[0])
