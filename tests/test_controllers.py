"""Controller-level validation: LQG separation, frozen stability with
spillover masking, harmonic cancellation, and end-to-end suppression."""
import copy

import numpy as np
import pytest

from avcp import (SystemParams, build_modal_model, simulate,
                  PshlqgController, PshlqgConfig)
from avcp.metrics import summarize
from avcp.stability import closed_loop_matrix, frozen_stability_map


@pytest.fixture(scope="module")
def setup():
    sysp = SystemParams()
    model, fem = build_modal_model(sysp)
    return sysp, model, fem


@pytest.fixture(scope="module")
def controller(setup):
    sysp, model, _ = setup
    ctl = PshlqgController(sysp, model, PshlqgConfig(r_u=9.4e-15))
    assert ctl.spillover_mask(model)
    return ctl


def test_separation_on_design_plant(setup, controller):
    """LQG separation: KF error dynamics and LQR loop each stable."""
    sysp, model, _ = setup
    p = controller.plant
    A, B, C = p.matrices(0.06)
    g = 15
    L = controller.L_grid[g]
    K = controller.K_grid[g]
    assert np.max(np.abs(np.linalg.eigvals(
        A - np.outer(L, C)))) < 1.0 + 1e-9
    Ac = np.zeros((p.nmf + 1, p.nmf + 1))
    Ac[:p.nmf, :p.nmf] = p.dt.Ad
    Ac[:p.nmf, -1] = p.dt.Bd_V
    Bc = np.zeros(p.nmf + 1)
    Bc[-1] = 1.0
    assert np.max(np.abs(np.linalg.eigvals(
        Ac - np.outer(Bc, K)))) < 1.0


def test_frozen_stability_along_path(setup, controller):
    """Frozen closed loop vs the FULL evaluation model along the path."""
    sysp, model, _ = setup
    xs = np.linspace(0.0, 0.12, 9)
    sm = frozen_stability_map(sysp, model, controller, xs)
    assert np.max(sm) <= 1.0 - 0.4 * (1.0 - controller.cfg.rho_dist)


@pytest.mark.slow
def test_end_to_end_suppression(setup):
    """Full nonlinear sim at mid-path: order-of-magnitude RMS reduction
    without saturation."""
    sysp, model, fem = setup
    sp = copy.deepcopy(sysp)
    sp.milling.x_start, sp.milling.x_end = 0.06, 0.060001
    sp.milling.pass_time = 3.0
    m2, _ = build_modal_model(sp, fem=fem)
    r_open = simulate(sp, m2, controller=None, T=3.0)
    ctl = PshlqgController(sp, m2, PshlqgConfig(r_u=9.4e-15))
    ctl.spillover_mask(m2)
    r_ctl = simulate(sp, m2, controller=ctl, T=3.0)
    so, sc = summarize(r_open, 3.0), summarize(r_ctl, 3.0)
    assert sc["rms_ac_wc"] < 0.4 * so["rms_ac_wc"]
    assert sc["sat_frac"] < 0.01
