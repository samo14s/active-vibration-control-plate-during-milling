"""Validation of the FEM foundation: avc/fem_plate.py, avc/piezo.py,
avc/modal.py.

1. Natural frequencies of the reduced model reproduce the measured
   frequencies of the reference plate (Du et al. 2024, Table 4) within 5%.
2. Mesh convergence: halving the mesh (20x16 vs 40x32) moves the first
   three frequencies by < 1%.
3. Mode-shape sanity: mode 2 is the first torsional mode, with a node
   line at plate mid-length; its top-edge deflection at x = lp/2 is
   negligible compared with the value near the edge.
4. Piezo actuation authority: the DC voltage -> sensor-corner
   displacement gain lies on the scale of the reference measured swept
   response (ref. Fig. 12b): 0.01 .. 0.5 um/V.
5. Material removal: receding the top edge by 1 mm shifts mode 1 by a
   small but nonzero amount.
6. interp_w partition of unity: the bilinear weights of interior points
   sum to exactly 1.
"""
import dataclasses

import numpy as np
import pytest

from avc.fem_plate import build_plate
from avc.modal import reduce_model
from avc.params import DEFAULT, MeshParams


@pytest.fixture(scope="module")
def mm5():
    """Reduced model, 5 modes, fresh plate, reference 40x32 mesh."""
    return reduce_model(DEFAULT, 5)


# --------------------------------------------------------------------------
# 1. natural frequencies vs. measured (ref. Table 4)
# --------------------------------------------------------------------------
def test_natural_frequencies_within_5pct_of_measured(mm5):
    f_model = mm5.freqs_hz
    f_meas = np.array(DEFAULT.plate.f_measured)
    assert f_model.shape == (5,)
    assert np.all(np.diff(f_model) > 0.0), "modes must be sorted ascending"
    rel_err = np.abs(f_model - f_meas) / f_meas
    assert np.all(rel_err < 0.05), (
        f"model freqs {f_model} vs measured {f_meas}, rel err {rel_err}")


# --------------------------------------------------------------------------
# 2. mesh convergence: 20x16 vs 40x32, first three frequencies < 1%
# --------------------------------------------------------------------------
def test_mesh_convergence_first_three_modes():
    p_coarse = dataclasses.replace(DEFAULT, mesh=MeshParams(nx=20, nz=16))
    p_fine = dataclasses.replace(DEFAULT, mesh=MeshParams(nx=40, nz=32))
    f_coarse = reduce_model(p_coarse, 3).freqs_hz
    f_fine = reduce_model(p_fine, 3).freqs_hz
    rel = np.abs(f_coarse - f_fine) / f_fine
    assert np.all(rel < 0.01), (
        f"coarse {f_coarse} vs fine {f_fine}, rel diff {rel}")


# --------------------------------------------------------------------------
# 3. mode 2 torsional: node line at plate mid-length on the top edge
# --------------------------------------------------------------------------
def test_mode2_torsional_node_at_center(mm5):
    lp = DEFAULT.plate.lp
    bf_mid = mm5.bf(0.5 * lp)     # (5,) modal top-edge deflections at lp/2
    bf_edge = mm5.bf(0.05 * lp)
    assert abs(bf_edge[1]) > 0.0
    assert abs(bf_mid[1]) < 0.05 * abs(bf_edge[1]), (
        f"mode-2 top-edge value at lp/2 = {bf_mid[1]:.3e} not << "
        f"value at 0.05 lp = {bf_edge[1]:.3e}")


# --------------------------------------------------------------------------
# 4. piezo authority: DC voltage -> sensor displacement gain scale
# --------------------------------------------------------------------------
def test_piezo_dc_gain_matches_reference_scale(mm5):
    # static modal superposition: w_s/u = sum_i cs_i * bu_i / omega_i^2
    dc_gain = float(np.sum(mm5.cs * mm5.bu / mm5.omega ** 2))  # [m/V]
    assert 0.01e-6 < abs(dc_gain) < 0.5e-6, (
        f"DC gain {dc_gain * 1e6:.4f} um/V outside the "
        "0.01..0.5 um/V scale of ref. Fig. 12b")


# --------------------------------------------------------------------------
# 5. material removal shifts mode 1 by a small but nonzero amount
# --------------------------------------------------------------------------
def test_material_removal_shifts_mode1():
    f_fresh = reduce_model(DEFAULT, 3).freqs_hz[0]
    f_recede = reduce_model(DEFAULT, 3, height_removed=1.0e-3).freqs_hz[0]
    rel_shift = abs(f_recede - f_fresh) / f_fresh
    assert 1.0e-4 < rel_shift < 0.05, (
        f"mode-1 shift {rel_shift * 100:.4f}% for 1 mm removal "
        f"({f_fresh:.2f} -> {f_recede:.2f} Hz)")


# --------------------------------------------------------------------------
# 6. interp_w partition of unity for interior points
# --------------------------------------------------------------------------
def test_interp_w_partition_of_unity():
    fem = build_plate(DEFAULT)
    lp = DEFAULT.plate.lp
    h = fem.height
    rng = np.random.default_rng(42)
    for _ in range(6):
        # interior points away from the clamped bottom row so all four
        # w-DOFs of the containing element are free
        x = rng.uniform(0.05 * lp, 0.95 * lp)
        z = rng.uniform(0.15 * h, 0.95 * h)
        row = fem.interp_w(x, z)
        nz = row[np.abs(row) > 1e-14]
        assert 1 <= len(nz) <= 4, f"expected <=4 bilinear weights, got {len(nz)}"
        assert np.all(nz > 0.0), "bilinear weights of an interior point are positive"
        assert abs(nz.sum() - 1.0) < 1e-12, (
            f"weights at ({x:.4f},{z:.4f}) sum to {nz.sum()!r}")
