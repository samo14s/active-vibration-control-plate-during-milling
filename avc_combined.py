"""
Combined control law = PD inner loop + mu-synthesis outer loop.

ARCHITECTURE (cascade in the DESIGN, parallel in the HARDWARE)

    d ---->[ plate + patch ]----> y
              ^          |
              |          +--> K_pd  --+
              |                       +--> u = u_pd + u_mu
              +-----------------------+
                         K_mu designed ON the PD-augmented plant

Both laws drive the same patch, so the realised controller is the parallel
sum of the two:

    Ak = blkdiag(Ap, Am)    Bk = [Bp ; Bm]    Ck = [Cp , Cm]    Dk = Dp + Dm

What makes this more than "run both at once" is that K_mu is synthesised on
the plant that ALREADY has the PD closed around it, and the control-effort
penalty in that generalised plant refers to the TOTAL patch voltage.  The
outer loop therefore spends its authority only on what the inner loop has
not already handled, instead of duplicating it.

Rationale: the PD is broadband, cheap in voltage and passive, so it flattens
the resonant peaks and shrinks the RELATIVE size of the frequency excursion
the robust loop has to cover.  mu-synthesis then buys robustness on a plant
that is far easier to be robust about.
"""
import numpy as np
from avc_plant import Plant, Ctrl, score, VLIM, ZMIN
import avc_mu


def parallel(K1, K2, name='combined'):
    """Parallel sum of two controllers acting on the same actuator."""
    n1, n2 = K1.order, K2.order
    Ak = np.zeros((n1 + n2, n1 + n2))
    if n1:
        Ak[:n1, :n1] = K1.Ak
    if n2:
        Ak[n1:, n1:] = K2.Ak
    Bk = np.zeros(n1 + n2)
    if n1:
        Bk[:n1] = K1.Bk.ravel()
    if n2:
        Bk[n1:] = K2.Bk.ravel()
    Ck = np.zeros(n1 + n2)
    if n1:
        Ck[:n1] = K1.Ck.ravel()
    if n2:
        Ck[n1:] = K2.Ck.ravel()
    return Ctrl(Ak, Bk, Ck, K1.Dk + K2.Dk, name)


def scale_ctrl_gain(K, a, name=None):
    """Scale a controller's gain by a (keeps its dynamics)."""
    return Ctrl(K.Ak, K.Bk.ravel(), a * K.Ck.ravel(), a * K.Dk,
                K.name if name is None else name)


def design(plant, K_pd, c_target, v_target, inner_gain=1.0, n_iter=3, verbose=False):
    """Design the outer mu loop around a (possibly de-rated) PD inner loop."""
    K_in = scale_ctrl_gain(K_pd, inner_gain, 'PD inner')
    avc_mu.C_TARGET, avc_mu.V_TARGET = c_target, v_target
    K_mu, peak, mu, om, hist = avc_mu.mu_ctrl(plant, n_iter=n_iter,
                                              verbose=verbose, inner=K_in)
    K_mu.name = 'mu outer'
    return parallel(K_in, K_mu, 'PD + mu (combined)'), K_in, K_mu, peak, mu, om


def design_all(plant, K_pd, c_target, v_target, inner_gain=1.0, n_iter=3,
               verbose=False):
    """Same, but return the combined law for EVERY D-K iterate (the
    iteration is non-monotone; the tuning driver scores them all).

    rec['K'] is the realised PARALLEL law (inner + outer, what the hardware
    runs and what score() sees); rec['K_ss'] stays the outer controller in
    scaled coordinates, which is what mu_cert closes on the inner-folded
    full-structure plant P_full."""
    K_in = scale_ctrl_gain(K_pd, inner_gain, 'PD inner')
    avc_mu.C_TARGET, avc_mu.V_TARGET = c_target, v_target
    iters, om, P_full, sc = avc_mu.mu_ctrl_all(plant, n_iter=n_iter,
                                               verbose=verbose, inner=K_in)
    out = []
    for rec in iters:
        K = parallel(K_in, rec['K'], 'PD + mu (combined)')
        out.append(dict(rec, K=K))
    return out, om, P_full, sc
