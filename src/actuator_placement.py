"""
actuator_placement.py
=====================
Actuator layout for path-wide authority over a thin-walled workpiece.

THE PROBLEM
-----------
Piezo patches for workpiece vibration control are normally placed by a
modal criterion evaluated once -- maximum modal strain energy of the target
mode on the nominal workpiece. That criterion has no notion of where the tool
is, and in thin-wall milling the tool moves the whole length of the wall.

The milling disturbance enters the modal equations along d(s) = Dp(x_tool(s)),
the mode shapes at the contact point, which rotates in modal space as the
tool feeds. The actuator set spans range(H). The reachable fraction of the
disturbance is

    gamma(s) = || Q^T d_hat(s) || ,  Q an orthonormal basis of range(H)

which is 1 when the disturbance is fully input-matched and 0 when no
combination of actuator voltages produces any modal force along it. A layout
chosen on a single-frequency modal criterion can, and here does, have
stations where gamma = 0 -- the actuator is blind to the disturbance at that
tool position, no matter how much voltage is available.

The right objective is therefore a WORST-CASE-OVER-PATH one:

    maximise_over_layouts   min_over_path  gamma(s)

which is what `optimise_layout` solves, by exhaustive search over a candidate
window grid for small layouts and greedy forward selection beyond that.

TWO METRICS, BOTH REPORTED
--------------------------
gamma is dimensionless and measures ALIGNMENT: whether the actuator can push
along the disturbance at all. It says nothing about how hard. `authority`
reports the companion magnitude, min_s ||H^T d_hat(s)|| in N/V, which is what
decides whether the available voltage is enough. A layout must be good on
both: perfect alignment with negligible authority is useless, and so is large
authority pointing the wrong way.

Feedforward cancellation capability is governed by gamma directly, since the
residual after ideal cancellation is sqrt(1 - gamma^2) of the disturbance.
Feedback damping capability is governed instead by modal controllability,
reported separately by `modal_controllability`.
"""
from __future__ import annotations

import itertools

import numpy as np


def matched_fraction_series(H_list, D_series):
    """
    gamma(s) for a layout, over a path.

    H_list   : list of (n_modes,) modal actuator vectors, one per patch
    D_series : (n_modes, n_station) disturbance directions along the path
    """
    H = np.column_stack(H_list)
    Q, R = np.linalg.qr(H)
    tol = 1e-12 * max(1.0, float(np.abs(np.diag(R)).max()))
    r = int(np.sum(np.abs(np.diag(R)) > tol))
    Q = Q[:, :r]
    D = np.asarray(D_series, float)
    nrm = np.linalg.norm(D, axis=0)
    nrm[nrm < 1e-30] = 1.0
    return np.linalg.norm(Q.T @ (D / nrm), axis=0)


def authority_series(H_list, D_series):
    """
    ||H^T d_hat(s)|| along the path, in modal newtons per volt.

    This is the magnitude companion to gamma: the largest modal force per
    volt that the layout can generate along the disturbance direction.
    """
    H = np.column_stack(H_list)
    D = np.asarray(D_series, float)
    nrm = np.linalg.norm(D, axis=0)
    nrm[nrm < 1e-30] = 1.0
    return np.linalg.norm(H.T @ (D / nrm), axis=0)


def modal_controllability(H_list, omega_n, zeta_n):
    """
    Per-mode controllability measure for the layout.

    For a modal second-order system the input matrix row for mode i is
    H[i, :], and the classical modal controllability measure that accounts
    for how lightly damped the mode is (so that a small input still matters)
    is ||H[i, :]|| / (2 zeta_i omega_i). Returns one value per mode; a mode
    with a near-zero value cannot be damped by this layout.
    """
    H = np.column_stack(H_list)
    return np.linalg.norm(H, axis=1) / (2.0 * np.asarray(zeta_n, float)
                                        * np.asarray(omega_n, float))


def build_candidates(lp, hp, n_windows=10, z_lo=0.0, z_hi=None, width=None):
    """Uniform grid of candidate patch windows along the wall length."""
    z_hi = 0.75 * hp if z_hi is None else z_hi
    width = lp / n_windows if width is None else width
    starts = np.linspace(0.0, lp - width, n_windows)
    return [(float(s), float(s + width), float(z_lo), float(z_hi))
            for s in starts]


def optimise_layout(candidates, H_of_window, D_series, n_act=2,
                    exhaustive_limit=4000, verbose=True):
    """
    Maximise the worst-case matched fraction over the path.

        layout* = argmax_L  min_s gamma(s; L)

    Exhaustive over all n_act-subsets when the count is manageable, greedy
    forward selection otherwise. Greedy is not optimal for a max-min
    objective, so which route was taken is reported rather than hidden.

    H_of_window : callable window -> (n_modes,) modal actuator vector
    Returns (best_windows, best_gamma_min, gamma_series, was_exhaustive)
    """
    H_cache = {w: np.asarray(H_of_window(w), float) for w in candidates}
    n_comb = len(list(itertools.combinations(range(len(candidates)), n_act)))

    def score(ws):
        g = matched_fraction_series([H_cache[w] for w in ws], D_series)
        return float(g.min()), g

    if n_comb <= exhaustive_limit:
        if verbose:
            print(f"   exhaustive over {n_comb} layouts of {n_act} patches")
        best = None
        for combo in itertools.combinations(candidates, n_act):
            s, g = score(list(combo))
            if best is None or s > best[1]:
                best = (list(combo), s, g)
        return best[0], best[1], best[2], True

    if verbose:
        print(f"   {n_comb} layouts exceeds the exhaustive limit; "
              f"greedy forward selection (not optimal for max-min)")
    chosen = []
    for _ in range(n_act):
        best = None
        for w in candidates:
            if w in chosen:
                continue
            s, g = score(chosen + [w])
            if best is None or s > best[1]:
                best = (w, s, g)
        chosen.append(best[0])
    s, g = score(chosen)
    return chosen, s, g, False
