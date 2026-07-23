"""
Particle-Swarm-Optimization (PSO) tuning of the controller gains on the paper's
non-minimum-phase plant.

Motivation
----------
A coarse manual grid left the classical controllers (PID, SMC) tuned for the
NOMINAL cutting force only, so they diverge at the extremes of the milling-force
coefficient alpha4.  PSO is the standard global optimiser for controller-gain
tuning; here it minimises a ROBUST cost — the settled vibration summed over a set
of cutting conditions spanning the whole alpha4 range and the paper's reduced
damping, with a heavy (but finite) penalty for any divergence.  Minimising it
first drives every particle out of the diverging region, then trims the residual
vibration, so the swarm converges to gains that are robust across the whole range
if such gains exist — and demonstrates it rigorously if they do not.

The optimiser is a plain global-best PSO (no external dependency).  Each
controller declares its search space (log-scaled for gains spanning decades) via
a small spec: `bounds`, a `decode(x)->kwargs` map, and a `make(kwargs)->Controller`
factory.  `robust_cost` evaluates a candidate across `ROBUST_CONDITIONS`.
"""

from __future__ import annotations
import numpy as np

from . import config as C
from . import plant as P
from . import metrics as M
from .simulate import simulate


# Cutting conditions the tuned controller must handle simultaneously (robustness).
# Five points spanning the paper's non-smooth milling-force range (alpha4 0.3..2.9x)
# at the paper's reduced damping (-20%).  We do NOT add a mass perturbation here:
# a positive dmass rescales the mode shape phi ~ 1/sqrt(1+dmass) and so REDUCES the
# regenerative coupling, which would make a "hard" corner artificially easy and let
# the swarm hide an unstable tuning behind it.  Intermediate alpha4 values are
# included because a controller can be fine at the extremes yet unstable in between.
ROBUST_CONDITIONS = [
    ("a4_0.3", dict(alpha4_factor=0.3, dzeta=-0.2)),
    ("a4_0.7", dict(alpha4_factor=0.7, dzeta=-0.2)),
    ("a4_1.1", dict(alpha4_factor=1.1, dzeta=-0.2)),
    ("a4_1.5", dict(alpha4_factor=1.5, dzeta=-0.2)),   # ~nominal force
    ("a4_1.9", dict(alpha4_factor=1.9, dzeta=-0.2)),
    ("a4_2.4", dict(alpha4_factor=2.4, dzeta=-0.2)),
    ("a4_2.9", dict(alpha4_factor=2.9, dzeta=-0.2)),   # max chatter force (hardest)
    # modal-frequency rise (material removal, ~+4 %) — breadth against the
    # frequency drift of the pass, not only the force band
    ("f_up",  dict(dmass=-0.08, dzeta=-0.2)),
]


def round_sig(v, sig=4):
    """Round to `sig` significant figures — so the optimiser is only credited for
    IMPLEMENTABLE-precision gains.  A tuning that is only stable at infinite
    precision (a knife-edge island) is thus correctly penalised."""
    if v == 0:
        return 0.0
    from math import log10, floor
    return round(v, -int(floor(log10(abs(v)))) + (sig - 1))

DIVERGE_PENALTY = 3000.0     # per diverging condition (>> any settled RMS in um)


def _rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def robust_cost(make, params, t_sim=0.30, volt_weight=0.02, worst_weight=2.0,
                sat_weight=300.0, growth_weight=150.0):
    """Robust tuning cost for one candidate parameter set.

    The chatter instability grows SLOWLY (the plate is extremely lightly damped),
    so a short-horizon "settled RMS" is misleading — a still-growing response can
    look small.  For each condition we therefore judge on the LAST-third RMS and
    add a growth-trend penalty comparing the last third to the middle third; a
    response that is still rising at the end of the window is treated as unstable
    even before it numerically blows up.  We also penalise:
      * DIVERGED (blow-up): a large finite penalty scaled by the blow-up magnitude
        (gives the swarm a gradient out of the diverging region);
      * RIDING SATURATION: the fraction of time |u| is at the actuator limit — a
        controller pinned to the rail is not really controlling.
    The total is the sum over conditions plus a worst-case emphasis term.
    """
    per = []
    for _, kw in ROBUST_CONDITIONS:
        try:
            ctrl = make(params)
            r = simulate(P.MillingPlant(**kw), ctrl, t_sim=t_sim, meas_noise=True)
            m = M.compute(r)
            y_um = np.asarray(r["y"]) * 1e6
            n = len(y_um)
            if m["diverged"] or not np.all(np.isfinite(y_um)) or n < 30:
                per.append(DIVERGE_PENALTY + min(m["peak_um"], 1e7) / 1e4)
                continue
            mid = _rms(y_um[n // 3:2 * n // 3])
            late = _rms(y_um[2 * n // 3:])
            growth = late / max(mid, 1e-3)            # >1 => still growing (unstable)
            u = np.asarray(r["u"])
            sat_frac = float(np.mean(np.abs(u) >= 0.98 * C.U_MAX))
            per.append(late
                       + volt_weight * m["peak_volt"]
                       + sat_weight * sat_frac
                       + growth_weight * max(0.0, growth - 1.1))
        except Exception:
            per.append(2.0 * DIVERGE_PENALTY)
    per = np.asarray(per)
    return float(per.sum() + worst_weight * per.max())


def pso(cost, bounds, n_particles=16, n_iters=18, seed=0,
        w=0.72, c1=1.5, c2=1.5, verbose=True, tag="", x0=None):
    """Global-best PSO.  `bounds` is a list of (lo, hi) in the SEARCH space.

    `x0` (optional) is a list of seed points injected into the initial swarm
    (e.g. a previously tuned optimum, or the zero-TD point of a combined
    controller so the swarm can never end worse than the base controller).
    Returns (best_x, best_cost, history).
    """
    rng = np.random.default_rng(seed)
    lb = np.array([b[0] for b in bounds], float)
    ub = np.array([b[1] for b in bounds], float)
    dim = len(bounds)

    X = lb + rng.random((n_particles, dim)) * (ub - lb)
    if x0:
        for i, xs in enumerate(x0[:n_particles]):
            X[i] = np.clip(np.asarray(xs, float), lb, ub)
    V = (rng.random((n_particles, dim)) - 0.5) * (ub - lb) * 0.1
    pbest = X.copy()
    pcost = np.array([cost(x) for x in X])
    gi = int(np.argmin(pcost))
    gbest = pbest[gi].copy()
    gcost = float(pcost[gi])
    hist = [gcost]
    if verbose:
        print(f"[{tag}] init  gbest={gcost:8.2f}  x={np.round(gbest, 3)}", flush=True)

    for it in range(n_iters):
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = np.clip(X + V, lb, ub)
        cst = np.array([cost(x) for x in X])
        imp = cst < pcost
        pbest[imp] = X[imp]
        pcost[imp] = cst[imp]
        gi = int(np.argmin(pcost))
        if pcost[gi] < gcost:
            gbest = pbest[gi].copy()
            gcost = float(pcost[gi])
        hist.append(gcost)
        if verbose:
            print(f"[{tag}] iter {it+1:2d}/{n_iters}  gbest={gcost:8.2f}  "
                  f"x={np.round(gbest, 3)}", flush=True)
    return gbest, gcost, hist


# --------------------------------------------------------------------------- #
#  Per-controller search specs (bounds in search space + decode + factory)     #
# --------------------------------------------------------------------------- #
def pid_spec():
    from .controllers.pid import PID
    # search in log10 for Kp, Kd, N (they span decades); Ki fixed at 0
    bounds = [(3.0, 4.7), (1.0, 2.7), (2.7, 3.85)]   # log10(Kp), log10(Kd), log10(N)

    def decode(x):
        return dict(Kp=round_sig(10.0 ** x[0]), Ki=0.0,
                    Kd=round_sig(10.0 ** x[1]), N=round_sig(10.0 ** x[2]))

    def make(x):
        return PID(**decode(x))
    # seed: the previous PSO optimum (log10 Kp, log10 Kd, log10 N)
    seeds = [[4.223, 2.700, 3.096]]
    return dict(name="PID", bounds=bounds, decode=decode, make=make, seeds=seeds)


def smc_spec():
    from .controllers.smc import SMC
    # zeta_s (linear), log10(ks), eta (linear, volts), log10(phi)
    bounds = [(0.05, 0.60), (3.0, 4.7), (10.0, 500.0), (-1.7, -0.3)]

    def decode(x):
        return dict(zeta_s=round_sig(x[0]), ks=round_sig(10.0 ** x[1]),
                    eta=round_sig(x[2]), phi=round_sig(10.0 ** x[3]))

    def make(x):
        return SMC(**decode(x))
    # seed: the previous PSO optimum (zeta_s, log10 ks, eta, log10 phi)
    seeds = [list(_SMC_SEED)]
    return dict(name="SMC", bounds=bounds, decode=decode, make=make, seeds=seeds)


def adrc_spec():
    from .controllers.adrc import ADRC
    # b0 sign fixed negative (physical); search |b0|, log10(wo), log10(wc)
    bounds = [(0.3, 6.0), (3.7, 4.9), (2.7, 4.2)]     # |b0|, log10(wo), log10(wc)

    def decode(x):
        return dict(b0=-round_sig(x[0]), wo=round_sig(10.0 ** x[1]),
                    wc=round_sig(10.0 ** x[2]))

    def make(x):
        return ADRC(**decode(x))
    return dict(name="ADRC", bounds=bounds, decode=decode, make=make)


def ctdc_spec():
    from .controllers.ctdc import CTDC
    # delayed-PD gains of the paper's Eq. (30): KPp [V/m], KPd [V/(m/s)].
    # The cancelling sense of the regenerative force is negative KPp on this
    # geometry (verified by a trend probe), but both signs are searched.
    bounds = [(-2.0e5, 5.0e4), (-60.0, 60.0)]

    def decode(x):
        return dict(KPp=round_sig(x[0]), KPd=round_sig(x[1]))

    def make(x):
        return CTDC(**decode(x))
    # seeds: zero-TD (= plain mu, the base) and the probed cancelling sense
    seeds = [[0.0, 0.0], [-4.0e4, 0.0]]
    return dict(name="CTDC", bounds=bounds, decode=decode, make=make, seeds=seeds)


# previously PSO-tuned SMC optimum, reused as swarm seed (search coordinates:
# zeta_s, log10 ks, eta, log10 phi) — tuned on the paper-exact force model
_SMC_SEED = [0.3644, 3.119, 254.6, -1.056]


def tdsmc_spec():
    from .controllers.tdsmc import TDSMC
    # joint search: sliding-mode gains + the paper's delayed-PD gains
    bounds = [(0.05, 0.60), (3.0, 4.7), (10.0, 500.0), (-1.7, -0.3),
              (-2.0e5, 5.0e4), (-60.0, 60.0)]

    def decode(x):
        return dict(zeta_s=round_sig(x[0]), ks=round_sig(10.0 ** x[1]),
                    eta=round_sig(x[2]), phi=round_sig(10.0 ** x[3]),
                    KPp=round_sig(x[4]), KPd=round_sig(x[5]))

    def make(x):
        return TDSMC(**decode(x))
    # seeds: plain tuned SMC (TD off) and the probed cancelling TD sense
    seeds = [_SMC_SEED + [0.0, 0.0], _SMC_SEED + [-4.0e4, 0.0]]
    return dict(name="TDSMC", bounds=bounds, decode=decode, make=make, seeds=seeds)


SPECS = {"PID": pid_spec, "SMC": smc_spec, "ADRC": adrc_spec,
         "CTDC": ctdc_spec, "TDSMC": tdsmc_spec}


def tune(name, n_particles=16, n_iters=18, seed=0):
    """Run PSO for one controller; return (best_kwargs, best_cost, validation, history)."""
    spec = SPECS[name]()
    best_x, best_cost, hist = pso(
        lambda x: robust_cost(spec["make"], x),
        spec["bounds"], n_particles=n_particles, n_iters=n_iters, seed=seed,
        tag=spec["name"], x0=spec.get("seeds"))
    best_kwargs = spec["decode"](best_x)
    val = validate(spec["make"], best_x)
    return best_kwargs, best_cost, val, hist


def validate(make, x, t_sim=0.30):
    """Evaluate the tuned candidate on the full robustness set at full horizon."""
    out = {}
    for label, kw in ROBUST_CONDITIONS:
        ctrl = make(x)
        r = simulate(P.MillingPlant(**kw), ctrl, t_sim=t_sim, meas_noise=True)
        m = M.compute(r)
        out[label] = dict(rms_um=(None if m["diverged"] else round(m["rms_settled_um"], 2)),
                          peak_V=round(m["peak_volt"], 1),
                          diverged=bool(m["diverged"]))
    return out
