"""
Full-discretisation method (FDM, Ding et al. -- the paper's ref [79]) for
the time-periodic delayed closed loop, plus a time-domain milling
simulator.  This is the verification instrument: the convex synthesis
optimises the zero-order (average-coefficient) limit, the FDM decides.

Closed loop (modal, mass-normalised; controller u = K y, y = cS q):

    x = [q; qd; xk]
    xd(t) = [A0 + a4(t) E] x(t) + a4(t) Ed x(t - tau) + forcing,
    E  injects  -D^T D q(t)   into the modal accelerations,
    Ed injects  +D^T D q(t-tau),
    tau = tooth-passing period = FDM period.

One first-order FDM step (a4 frozen at the step midpoint, delayed state
linearly interpolated; the delay equals the period so the grid aligns):

    x_{i+1} = F0_i x_i + Fa_i x_{i-m} + Fb_i x_{i-m+1}

with F0 = e^{Abar h},  Phi1 = int_0^h e^{Abar (h-s)} ds,
Phi2 = int_0^h e^{Abar (h-s)} s ds  (Van Loan block-exponential),
Fa = a4 (Phi1 - Phi2/h) Ed,  Fb = a4 (Phi2/h) Ed.

The monodromy operator acts on the lifted history z = [x_i, ..., x_{i-m}];
its spectral radius is found with ARPACK on the one-period propagation as
a LinearOperator -- no dense monodromy matrix is ever formed.
"""
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, eigs
import milling_model as mm
from avc_plant import NM


def closed_matrices(plant, D, ctrl=None, couple_u=True):
    """A0, E, Ed of the closed loop at cutting row D.

    couple_u=False leaves the actuator path b*u OUT of A0 (the controller
    states still integrate their own dynamics and the sensor input): the
    time simulator applies u externally so it can saturate it."""
    n = NM
    nk = 0 if ctrl is None else ctrl.order
    Dk = 0.0 if ctrl is None else ctrl.Dk
    N = 2 * n + nk
    A0 = np.zeros((N, N))
    A0[:n, n:2 * n] = np.eye(n)
    A0[n:2 * n, :n] = -np.diag(plant.wn ** 2)
    A0[n:2 * n, n:2 * n] = -np.diag(2.0 * plant.zeta * plant.wn)
    b = plant.bmod
    cS = plant.cS
    if ctrl is not None:
        if couple_u:
            A0[n:2 * n, :n] += np.outer(b, Dk * cS)
        if nk:
            if couple_u:
                A0[n:2 * n, 2 * n:] = np.outer(b, ctrl.Ck.ravel())
            A0[2 * n:, :n] = ctrl.Bk @ cS[None, :]
            A0[2 * n:, 2 * n:] = ctrl.Ak
    DTD = np.outer(D, D)
    E = np.zeros((N, N)); E[n:2 * n, :n] = -DTD
    Ed = np.zeros((N, N)); Ed[n:2 * n, :n] = +DTD
    return A0, E, Ed


def _step_mats(A0, E, Ed, a4mid, h):
    """Per-step F0, Fa, Fb via one Van Loan block exponential each."""
    N = A0.shape[0]
    out = []
    Z = np.zeros((N, N))
    I = np.eye(N)
    for a4 in a4mid:
        Ab = A0 + a4 * E
        M = np.block([[Ab, I, Z],
                      [Z, Z, I],
                      [Z, Z, Z]])
        Ph = expm(M * h)
        F0 = Ph[:N, :N]
        # Van Loan blocks of the [[Ab,I,0],[0,0,I],[0,0,0]] chain:
        #   J1 = Ph[:N, N:2N]  = int_0^h e^{Ab(h-s)} ds
        #   J2 = Ph[:N, 2N:3N] = int_0^h e^{Ab(h-s)} s ds
        J1 = Ph[:N, N:2 * N]
        J2 = Ph[:N, 2 * N:3 * N]
        Fa = a4 * (J1 - J2 / h) @ Ed
        Fb = a4 * (J2 / h) @ Ed
        out.append((F0, Fa, Fb))
    return out


def fdm_radius(plant, D, rpm, ap, ctrl=None, m=40, k=4):
    """Spectral radius of the one-period monodromy operator."""
    A0, E, Ed = closed_matrices(plant, D, ctrl)
    N = A0.shape[0]
    T = mm.tau_of(rpm)
    h = T / m
    a4mid = mm.alpha4_series(rpm, ap, m)
    S = _step_mats(A0, E, Ed, a4mid, h)

    def onep(z):
        # z = [x_i ; x_{i-1} ; ... ; x_{i-m}] most recent first
        X = z.reshape(m + 1, N)
        hist = [X[j] for j in range(m + 1)]
        for i in range(m):
            F0, Fa, Fb = S[i]
            xn = F0 @ hist[0] + Fa @ hist[m] + Fb @ hist[m - 1]
            hist = [xn] + hist[:m]
        return np.concatenate(hist)

    L = LinearOperator(((m + 1) * N, (m + 1) * N), matvec=onep, dtype=float)
    try:
        vals = eigs(L, k=k, which='LM', return_eigenvectors=False,
                    maxiter=3000, tol=1e-7)
        return float(np.abs(vals).max())
    except Exception:
        # power iteration fallback
        rng = np.random.default_rng(0)
        z = rng.standard_normal((m + 1) * N)
        z /= np.linalg.norm(z)
        r = 0.0
        for _ in range(400):
            z2 = onep(z)
            r = np.linalg.norm(z2)
            z = z2 / max(r, 1e-300)
        return float(r)


def fdm_ap_limit(plant, D, rpm, ctrl=None, ap_hi=3e-3, m=40, rel=0.03):
    """Bisection on the axial depth until the monodromy radius crosses 1."""
    lo, hi = 0.0, ap_hi
    if fdm_radius(plant, D, rpm, hi, ctrl, m) < 1.0:
        return hi
    # ensure lo stable
    while hi - lo > rel * max(hi, 1e-6):
        mid = 0.5 * (lo + hi)
        if mid <= 0:
            break
        if fdm_radius(plant, D, rpm, mid, ctrl, m) < 1.0:
            lo = mid
        else:
            hi = mid
    return lo


def fdm_sld(plant, D, rpm_grid, ctrl=None, ap_hi=3e-3, m=40, verbose=False):
    out = np.empty(len(rpm_grid))
    for i, rpm in enumerate(rpm_grid):
        out[i] = fdm_ap_limit(plant, D, rpm, ctrl, ap_hi, m)
        if verbose:
            print('  %5.0f rpm : ap_lim %.3f mm' % (rpm, out[i] * 1e3), flush=True)
    return out


# ---------------------------------------------------------------- time sim
def milling_sim(plant, D, rpm, ap, ctrl=None, T_end=0.5, dt=None,
                vmax=200.0, x0=1e-9):
    """Time-domain milling response at the cutting point with feed forcing
    and voltage saturation.  The smooth free dynamics (plant + controller
    states, WITHOUT the actuator path) step by exact matrix exponential;
    the regenerative force, feed force and the saturated control voltage
    enter as slowly varying forces (first-order hold)."""
    A0, E, Ed = closed_matrices(plant, D, ctrl, couple_u=False)
    N = A0.shape[0]
    tau = mm.tau_of(rpm)
    if dt is None:
        dt = tau / 160.0
    steps = int(T_end / dt)
    F0 = expm(A0 * dt)
    n = NM
    nd = max(int(round(tau / dt)), 1)
    hist = np.zeros((nd + 1, N))            # hist[0] = x(t_i - tau)
    x = np.zeros(N); x[0] = x0
    t_hist = np.empty(steps); y_hist = np.empty(steps); u_hist = np.empty(steps)
    b = plant.bmod; cS = plant.cS
    a_all = mm.alphas(np.arange(steps) * dt, rpm, ap)
    for i in range(steps):
        xd = hist[0]
        w_now = float(D @ x[:n])
        w_del = float(D @ xd[:n])
        f_mod = -a_all[3, i] * (w_now - w_del) + mm.FT * a_all[2, i]
        if ctrl is not None:
            u = ctrl.Dk * float(cS @ x[:n])
            if ctrl.order:
                u += float(np.asarray(ctrl.Ck @ x[2 * n:]).ravel()[0])
            u = float(np.clip(u, -vmax, vmax))
        else:
            u = 0.0
        xf = np.zeros(N)
        xf[n:2 * n] = D * f_mod + b * u
        x = F0 @ x + dt * xf
        hist = np.vstack([hist[1:], x[None, :]])
        t_hist[i] = i * dt
        y_hist[i] = w_now
        u_hist[i] = u
    return t_hist, y_hist, u_hist


if __name__ == '__main__':
    from avc_plant import Plant
    P = Plant(zeta=mm.ZETA_PAPER)
    D = P.edge_D(100.0)
    for rpm in [3600.0, 4900.0, 5400.0]:
        ap = fdm_ap_limit(P, D, rpm, None, ap_hi=3e-3, m=40)
        print('uncontrolled FDM ap_lim @ %4.0f rpm : %.3f mm' % (rpm, ap * 1e3))
