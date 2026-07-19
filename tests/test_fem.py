"""FEM validation: analytical frequencies, reciprocity, piezo statics."""
import numpy as np
import pytest

from avcp import SystemParams, PlateFem, build_modal_model


@pytest.fixture(scope="module")
def sysp():
    return SystemParams()


@pytest.fixture(scope="module")
def fem(sysp):
    return PlateFem(sysp.plate, 40, 40)


def test_cantilever_frequencies_vs_leissa(sysp, fem):
    """Nondimensional lambda^2 vs Leissa's CFFF square-plate values.

    Leissa (NASA SP-160) tabulates nu = 0.3; our nu = 0.33 shifts modes
    2, 4, 5 by up to ~2 %, so tolerances are mode-dependent."""
    freqs, _ = fem.modes(5)
    p = sysp.plate
    lam2 = 2 * np.pi * freqs * p.Lx ** 2 * np.sqrt(p.rho * p.h / p.D)
    ref = np.array([3.492, 8.525, 21.429, 27.331, 31.111])
    assert np.all(np.abs(lam2 - ref) / ref < 0.03)


def test_mesh_convergence(sysp):
    f1, _ = PlateFem(sysp.plate, 20, 20).modes(5)
    f2, _ = PlateFem(sysp.plate, 40, 40).modes(5)
    assert np.all(np.abs(f1 - f2) / f2 < 0.005)


def test_point_load_reciprocity(sysp, fem):
    pa, pb = (0.03, 0.09), (0.10, 0.06)
    ua = fem.static_solve(fem.point_load(*pa))
    wb = float(fem.point_row_free(*pb) @ ua)
    ub = fem.static_solve(fem.point_load(*pb))
    wa = float(fem.point_row_free(*pa) @ ub)
    assert wb == pytest.approx(wa, rel=1e-8)


def test_piezo_static_deflection_matches_calibration(sysp):
    """With the identified authority factor, the 1200 V free-edge static
    deflection reproduces the reference paper's measured ~5.1-5.6e-4 m."""
    fem = PlateFem(sysp.plate, 60, 60)
    q = sysp.piezo.moment_constant(sysp.plate.h)
    Ba = fem.piezo_vector(sysp.piezo.u1, sysp.piezo.u2,
                          sysp.piezo.v1, sysp.piezo.v2, q)
    u = fem.static_solve(Ba * 1200.0)
    xs = np.linspace(0.0, 0.12, 25)
    w = np.array([fem.point_row_free(x, 0.1199) @ u for x in xs])
    assert 4.0e-4 < np.max(np.abs(w)) < 7.0e-4


def test_piezo_symmetry(sysp):
    """Patch is x-symmetric about the plate mid-line, so symmetric modes
    couple and antisymmetric modes (about x = L/2) do not."""
    model, _ = build_modal_model(sysp, n_modes=6)
    # modes 2 and 5 of the CFFF square plate are antisymmetric in x
    ba = np.abs(model.ba)
    assert ba[1] < 1e-6 * ba.max()
    assert ba[4] < 1e-6 * ba.max()


def test_modal_static_consistency(sysp):
    """Modal superposition static deflection ~= direct static solve."""
    model, fem = build_modal_model(sysp, n_modes=20)
    q = sysp.piezo.moment_constant(sysp.plate.h)
    Ba = fem.piezo_vector(sysp.piezo.u1, sysp.piezo.u2,
                          sysp.piezo.v1, sysp.piezo.v2, q)
    u = fem.static_solve(Ba * 1000.0)
    x_t = (0.06, 0.117)
    w_direct = float(fem.point_row_free(*x_t) @ u)
    phi = model.phi_at(0.06)
    w_modal = float(np.sum(model.ba * 1000.0 * phi / model.omega ** 2))
    assert w_modal == pytest.approx(w_direct, rel=0.02)
