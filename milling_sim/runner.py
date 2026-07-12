"""Experiment orchestration: one machining pass = simulator + controller.

Mirrors the experimental campaign of Sec. 4: each strategy (conventional,
SSV, adaptive) is repeated over several seeds ("n = 5 trials, same
material batch, fresh tools"), and summary statistics are computed per
strategy.
"""

from __future__ import annotations

import copy

import numpy as np

from .control import (AdaptiveController, BaseController,
                      FixedParameterController, SSVController)
from .engine import MillingSimulator
from .metrics import MetricsCollector, RunHistory, summarize
from .parameters import PaperSetup


def run_pass(setup: PaperSetup, controller: BaseController,
             seed: int = 0) -> tuple[RunHistory, dict]:
    sim = MillingSimulator(setup, seed=seed)
    col = MetricsCollector(setup)

    while not sim.finished:
        n_rpm, vf, ap = controller.command(sim.t)
        meas = sim.advance(n_rpm, vf, ap)
        controller.observe(meas)
        fn_true = sim.current_modes()[0][0]        # dominant wall mode
        k_wall = sim.kk[0]
        col.update(meas, fn_true, k_wall)
        if isinstance(controller, AdaptiveController) and controller.log.t:
            controller.log.fn_true[-1] = fn_true

    return col.h, summarize(col.h, setup)


def make_controller(name: str, setup: PaperSetup,
                    boundary_learner=None) -> BaseController:
    if name == "conventional":
        return FixedParameterController(setup)
    if name == "ssv":
        return SSVController(setup)
    if name == "adaptive":
        return AdaptiveController(setup, boundary_learner=boundary_learner)
    raise ValueError(name)


def run_campaign(setup: PaperSetup, strategies: list[str],
                 n_trials: int = 5, base_seed: int = 100,
                 verbose: bool = True):
    """Run every strategy n_trials times; returns
    {strategy: {"histories": [...], "summaries": [...], "controllers": [...]}}
    """
    out: dict[str, dict] = {}
    for name in strategies:
        hs, ss, cs = [], [], []
        for trial in range(n_trials):
            ctl = make_controller(name, copy.deepcopy(setup))
            h, s = run_pass(setup, ctl, seed=base_seed + 17 * trial)
            hs.append(h)
            ss.append(s)
            cs.append(ctl)
            if verbose:
                print(f"  [{name}] trial {trial + 1}/{n_trials}: "
                      f"MRR {s['avg_mrr_cm3_min']:.1f} cm3/min, "
                      f"RMS max {s['rms_max_um']:.0f} um, "
                      f"Ra {s['ra_min_um']:.2f}-{s['ra_max_um']:.2f} um")
        out[name] = {"histories": hs, "summaries": ss, "controllers": cs}
    return out


def aggregate(summaries: list[dict]) -> dict:
    keys = summaries[0].keys()
    agg = {}
    for k in keys:
        vals = np.array([s[k] for s in summaries], dtype=float)
        agg[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return agg
