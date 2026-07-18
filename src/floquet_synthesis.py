"""
floquet_synthesis.py
====================
Floquet-direct feedback synthesis for chatter suppression.

A standard LQG/LQR design chooses its gain by minimising a quadratic cost, i.e.
by pushing the real part of the *non-delayed* closed-loop eigenvalues to the
left under a gain-norm cap.  That eigenvalue objective is only a **proxy** for
the quantity that actually governs chatter: the dominant Floquet multiplier of
the delayed, time-periodic milling dynamics.

This module builds a family of stabilising gains and scores each one directly by
the true chatter metric -- the critical depth of cut ``a_p,crit`` obtained from
:class:`cl_fdm.ClosedLoopFDM` -- so the gain can be selected on the actual
stability/effort trade-off rather than on the eigenvalue proxy.  Two selection
rules are provided so the two philosophies can be compared on an identical
candidate set:

* :func:`select_eig_proxy`   -- the LQG rule (max left-most eigenvalue real
  part subject to ``||K|| <= gain_cap``);
* :func:`select_floquet`     -- the proposed rule (max ``a_p,crit`` subject to
  ``||K|| <= gain_cap``).

Author: contribution built on the model of
        Du et al., Int. J. Mech. Sci. 274 (2024) 109257.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are


def state_space(plate):
    """Modal state-space (A, B, C) with x = [q; q_dot]."""
    n = plate.n_modes
    nx = 2 * n
    A = np.zeros((nx, nx))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -np.diag(plate.omega_n ** 2)
    A[n:, n:] = -np.diag(2 * plate.zeta_modes * plate.omega_n)
    B = np.zeros((nx, 1))
    B[n:, 0] = plate.H_Pe_modal
    C = np.zeros((1, nx))
    C[0, :n] = plate.D_obs
    return A, B, C


def lqr_gain(plate, w_q, w_qd, w_r=1.0):
    """LQR gain for output-weighted cost.

    Q = blkdiag(w_q * D_obs D_obs^T,  w_qd * I),  R = w_r.
    Returns K (1 x 2n) and the closed-loop eigenvalues.
    """
    A, B, _ = state_space(plate)
    n = plate.n_modes
    M_yp = np.outer(plate.D_obs, plate.D_obs)
    Q = np.block([[w_q * M_yp + 1e-3 * np.eye(n), np.zeros((n, n))],
                  [np.zeros((n, n)), w_qd * np.eye(n) + 1e-3 * np.eye(n)]])
    R = np.array([[w_r]])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    ev = np.linalg.eigvals(A - B @ K)
    return K, ev


def pareto_scan(clf, plate, RPM, w_q_list, w_qd_list, w_r=1.0, verbose=False):
    """Evaluate a family of LQR gains on the true chatter metric.

    Returns a list of records, one per (w_q, w_qd), each holding the gain, its
    norm, the left-most closed-loop eigenvalue real part (the LQG proxy) and the
    Floquet critical depth ``a_p,crit`` (metres) from CL-FDM.
    """
    records = []
    for w_q in w_q_list:
        for w_qd in w_qd_list:
            try:
                K, ev = lqr_gain(plate, w_q, w_qd, w_r)
            except Exception:
                continue
            min_re = float(np.min(np.real(ev)))
            if min_re >= 0:
                continue                       # not stabilising
            ap_c = clf.ap_crit(RPM, K)
            rec = dict(w_q=w_q, w_qd=w_qd, K=K,
                       gain_norm=float(np.linalg.norm(K)),
                       min_re=min_re, ap_crit=ap_c)
            records.append(rec)
            if verbose:
                print(f"  w_q={w_q:.0e} w_qd={w_qd:.0e}  "
                      f"|K|={rec['gain_norm']:.2e}  "
                      f"min_re={min_re:+.1f}  a_p,crit={ap_c*1e3:.3f}mm")
    return records


def select_eig_proxy(records, gain_cap):
    """LQG selection rule: max |min Re(lambda)| subject to ||K|| <= gain_cap."""
    feasible = [r for r in records if r["gain_norm"] <= gain_cap]
    if not feasible:
        return None
    return min(feasible, key=lambda r: r["min_re"])      # most negative real part


def select_floquet(records, gain_cap):
    """Proposed rule: max a_p,crit subject to ||K|| <= gain_cap."""
    feasible = [r for r in records if r["gain_norm"] <= gain_cap]
    if not feasible:
        return None
    return max(feasible, key=lambda r: r["ap_crit"])


def pareto_front(records):
    """Non-dominated set on (gain_norm minimise, a_p,crit maximise)."""
    front = []
    for r in sorted(records, key=lambda x: x["gain_norm"]):
        if not front or r["ap_crit"] > front[-1]["ap_crit"] + 1e-9:
            front.append(r)
    return front
