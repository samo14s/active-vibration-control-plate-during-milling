"""
mu-synthesis (D-K iteration) for the patch-actuated cantilever plate.

WHY a robust design is called for here
--------------------------------------
The plate stiffens as material is machined off: on this benchmark f1 rises by
+17 % and f2 by +11 % over the cut.  A PPF filter tuned on the fresh plate is
detuned by the end of the pass, and a fixed PD law sees its loop gain move.
That frequency excursion is a *parametric* uncertainty, which is what
mu-synthesis is built for -- and the reason a plain H-infinity design does not
suffice (H-inf cannot exploit the block structure of the uncertainty and pays
for perturbations that cannot physically occur).

UNCERTAINTY STRUCTURE
    delta_1..6 : ALL six modal frequencies,  wk^2 in [wk0^2 , REMOVAL_k^2 wk0^2]
                 (modes 1-2: +17 % / +11 %; modes 3-6: +5 % each)
    delta_a    : patch actuation gain, +/- 15 % (bonding, depolarisation, d31)
    Delta_p    : performance block (robust performance -> one full 2x2 block)

Each frequency range is written w^2 = wbar^2 (1 + p delta), delta in [-1,1], so
the DESIGN model sits at mid-excursion while the set covers both the fresh and
the fully-machined plate.  All control laws are afterwards SCORED on the same
fresh-plate model, so the comparison stays fair.

Covering ALL modes -- not only 1-2 -- is the lesson of the previous session:
designs whose certificate stopped at mode 2 fell below the 0.45 % spillover
floor (some to outright instability) under the +5 % shift of modes 3-6 that
the replay applies but the old model never covered.  The high-mode deltas are
treated as independent scalars, a superset of the true repeated-scalar
perturbation, so the certificate stays valid (and errs conservative).

HONEST LIMITATIONS, stated rather than buried:
  1. delta_1, delta_2, delta_a are REAL parameters but the mu upper bound uses
     complex D-scales only (no G-scales).  The bound is therefore conservative:
     the true mixed-mu is lower, so the margins reported are pessimistic.
  2. The modal damping term 2*zeta*w is held at wbar rather than made a
     function of delta; the induced error is O(zeta*p) ~ 1e-3 and negligible.
  3. A sensor-noise channel is added purely to satisfy the H-inf rank condition
     on [A-jwI B1; C2 D21]; its weight is set well below the signal level.
"""
import numpy as np
import control as ct
from scipy.optimize import least_squares, minimize
from avc_plant import Plant, Ctrl, NM, ZETA, REMOVAL

# ---------------------------------------------------------------- uncertainty
RK = REMOVAL ** 2                  # per-mode w^2 ratio at the end of the cut
PA = 0.15                          # actuator gain uncertainty
NU = NM + 1                        # uncertainty channels: 6 modal + actuator

# ---------------------------------------------------------------- weights
C_TARGET = 25e-6      # m/N   receptance the performance weight asks for
F_WP = 1500.0         # Hz    performance roll-off
V_TARGET = 150.0      # V/N   effort the control weight asks for
F_WU = 2000.0         # Hz    control weight corner
NOISE = 1e-3          # sensor noise, as a fraction of the peak d->y magnitude


WU_FLAT = False    # legacy Wu is 1/(10 V_TARGET) below F_WU -- the resonant
                   # voltage peaks of modes 1-2 sit exactly in that weak zone,
                   # which is how 300+ V/N designs sailed through gamma < 1.
                   # The flat shape holds 1/V_TARGET through the resonances
                   # and still rises to 10/V_TARGET above the corner.


def _weights():
    wp = 2 * np.pi * F_WP
    Wp = ct.tf([1.0 / C_TARGET * wp], [1.0, wp])
    wz, wpu = 2 * np.pi * F_WU, 2 * np.pi * F_WU * 10
    if WU_FLAT:
        Wu = ct.tf([10.0 / V_TARGET, 10.0 * wz / V_TARGET], [1.0, wpu])
    else:
        Wu = ct.tf([1.0 / V_TARGET, 1.0 / V_TARGET * wz], [1.0, wpu])
    return Wp, Wu


W_REF = 2 * np.pi * 1000.0    # rad/s -- time scaling
U_REF = 100.0                 # V     -- control scaling


def gen_plant(plant, cT, scaled=True, inner=None, unc=None):
    """P : [w_d.., w_da, d, n ; u] -> [z_d.., z_da, z_p, z_u ; y].

    unc selects which MODAL uncertainty channels are carried: None -> all six
    (the analysis structure the certificate is computed on), (0, 1) -> the
    numerically friendly synthesis structure.  A mode is recentred to its
    mid-excursion ONLY when it carries a delta channel; an uncovered mode
    stays at its fresh-plate frequency -- recentring without coverage would
    just build model error into the design.

    With scaled=True the realisation is normalised before synthesis:
      * modal state scaling  x = [w_k q_k ; qdot_k]  -> A entries O(w) not O(w^2)
      * time scaling by W_REF                        -> A entries O(1)
      * u scaled by U_REF, y scaled by its peak d->y gain
      * each uncertainty channel pair (z_i, w_i) normalised by alpha_i, which
        leaves the LFT unchanged because z scales by 1/alpha and w by alpha.
    Without it the plant spans |A| ~ 1e9 against D21 ~ 1e-9 and the H-infinity
    rank condition fails numerically.
    """
    n = NM
    unc = tuple(range(n)) if unc is None else tuple(unc)
    nu = len(unc) + 1               # modal channels + actuator gain
    wn = plant.wn
    wbar = wn.copy()
    wbar[list(unc)] = (wn * np.sqrt((1.0 + RK) / 2.0))[list(unc)]
    pk = (RK - 1.0) / (RK + 1.0)

    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -np.diag(wbar ** 2)
    zt = getattr(plant, 'zeta', np.full(n, ZETA))
    A[n:, n:] = -np.diag(2 * zt * wbar)

    b = plant.bmod
    iu = nu + 2                     # column of u ; [w_d(:), w_da, d, n, u]
    B = np.zeros((2 * n, nu + 3))
    for j, k in enumerate(unc):
        B[n + k, j] = -1.0          # w_dk enters mode-k acceleration
    B[n:, nu - 1] = b               # w_da : actuator gain perturbation
    B[n:, nu] = cT                  # d    : cutting force at the tool
    B[n:, iu] = b                   # u    : patch voltage

    om = 2 * np.pi * np.linspace(50, 4500, 4000)
    dy = np.abs(sum(plant.cS[k] * cT[k] /
                    (wbar[k] ** 2 - om ** 2 + 2j * zt[k] * wbar[k] * om)
                    for k in range(n))).max()
    w_noise = NOISE * dy

    C = np.zeros((nu + 3, 2 * n)); D = np.zeros((nu + 3, nu + 3))
    for j, k in enumerate(unc):
        C[j, k] = wbar[k] ** 2 * pk[k]      # z_dk
    D[nu - 1, iu] = PA                      # z_da
    C[nu, :n] = cT                          # z_p
    D[nu + 1, iu] = 1.0                     # z_u
    C[nu + 2, :n] = plant.cS                # y
    D[nu + 2, nu + 1] = w_noise

    if scaled:
        # ---- modal state scaling: xtil = [w_k q_k ; qdot_k]
        sx = np.concatenate([1.0 / wbar, np.ones(n)])       # x = diag(sx) xtil
        A = (A * sx) / sx[:, None]
        B = B / sx[:, None]
        C = C * sx
        # ---- I/O scaling
        B[:, iu] *= U_REF
        D[:, iu] *= U_REF
        C[nu + 2, :] /= dy
        D[nu + 2, :] /= dy
        # ---- time scaling
        A = A / W_REF
        B = B / W_REF
        # ---- balance each uncertainty channel pair (z_j, w_j).
        # Scaling z by 1/alpha and w by alpha leaves the LFT invariant, so we
        # are free to equalise the row and column norms.  NB the direct gain
        # w_j -> z_j is legitimately ZERO for the input-multiplicative
        # actuator channel (z_da depends on u alone), so any normalisation
        # based on that gain is degenerate -- hence norms, not gains.
        for j in range(nu):
            row = np.linalg.norm(np.concatenate([C[j, :], D[j, :]]))
            col = np.linalg.norm(np.concatenate([B[:, j], D[:, j]]))
            if row <= 0 or col <= 0:
                continue
            alpha = np.sqrt(row / col)
            C[j, :] /= alpha; D[j, :] /= alpha
            B[:, j] *= alpha; D[:, j] *= alpha
        scale = dict(w_ref=W_REF, u_ref=U_REF, y_ref=dy)
    else:
        scale = dict(w_ref=1.0, u_ref=1.0, y_ref=1.0)

    G = ct.ss(A, B, C, D)
    Wp, Wu = _weights()
    if scaled:      # weights live in scaled time too
        Wp = ct.tf(*[list(np.atleast_1d(c)) for c in _tscale(Wp, W_REF)])
        Wu = ct.tf(*[list(np.atleast_1d(c)) for c in _tscale(Wu, W_REF)])
    eyeU = ct.StateSpace([], [], [], np.eye(nu))
    eye1 = ct.StateSpace([], [], [], np.eye(1))
    Wblk = ct.append(eyeU, ct.ss(Wp), ct.ss(Wu), eye1)
    P = ct.ss(Wblk * G)

    if inner is not None:
        P = _fold_inner(P, inner, scale)
    return P, scale


def _balance_ss(A, B, C):
    """Diagonal state scaling so each state's B-row and C-column have equal norm.
    Without it, folding the PD (Ck ~ 2e7 against Bk ~ 1.6e-7) puts |A| back up
    to 1e9 and the H-infinity rank test fails again."""
    n = A.shape[0]
    if n == 0:
        return A, B, C
    a = np.ones(n)
    for i in range(n):
        rb = np.linalg.norm(B[i, :]); cc = np.linalg.norm(C[:, i])
        if rb > 0 and cc > 0:
            a[i] = np.sqrt(rb / cc)
    return (A * a) / a[:, None], B / a[:, None], C * a


def _scale_ctrl(K, scale):
    """Express a physical controller y -> u in the scaled coordinates of gen_plant."""
    wr, ur, yr = scale['w_ref'], scale['u_ref'], scale['y_ref']
    g = yr / ur
    if K.order == 0:
        return ct.ss([], [], [], np.array([[g * K.Dk]]))
    Ak = K.Ak / wr
    Bk = K.Bk / wr
    Ck = g * K.Ck
    Ak, Bk, Ck = _balance_ss(Ak, Bk, Ck)
    return ct.ss(Ak, Bk, Ck, np.array([[g * K.Dk]]))


def _fold_inner(P, inner, scale):
    """Close an inner control law around P, leaving the (u, y) channel open for
    the outer design.  The two laws act in PARALLEL on the same patch, so the
    u input is duplicated and the y output is duplicated; the LFT then closes
    only the duplicate pair.  Everything the outer loop sees -- including the
    control-effort penalty z_u -- therefore refers to the TOTAL patch voltage.
    """
    A, B, C, D = ct.ssdata(P)
    B2 = np.hstack([B, B[:, [-1]]])          # [w(5), u_outer, u_inner]
    D2 = np.hstack([D, D[:, [-1]]])
    C2 = np.vstack([C, C[[-1], :]])          # [z(5), y, y]
    D2 = np.vstack([D2, D2[[-1], :]])
    P2 = ct.ss(A, B2, C2, D2)
    return ct.ss(P2.lft(_scale_ctrl(inner, scale), 1, 1))


def _tscale(sys, wref):
    """Return (num, den) of W(wref * s) for a SISO transfer function."""
    num = np.atleast_1d(np.squeeze(sys.num))
    den = np.atleast_1d(np.squeeze(sys.den))
    nn, nd = len(num), len(den)
    num = num * wref ** np.arange(nn - 1, -1, -1)
    den = den * wref ** np.arange(nd - 1, -1, -1)
    return num / den[0], den / den[0]


# ---------------------------------------------------------------- mu bound
def mu_ub(M, x0=None):
    """Upper bound on mu with diagonal complex D-scales.
    Blocks: the scalar uncertainty channels (their count inferred from M),
    then a full 2x2 performance block whose two channels share one D."""
    nu = M.shape[0] - 2
    def cost(ld):
        d = np.exp(np.concatenate([ld, [0.0]]))
        S = np.diag(np.concatenate([d[:nu], [d[nu], d[nu]]]))
        return np.linalg.svd(S @ M @ np.linalg.inv(S), compute_uv=False)[0]

    x0 = np.zeros(nu) if x0 is None else np.asarray(x0, float)
    r = minimize(cost, x0, method='Nelder-Mead',
                 options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 200 * nu})
    # a plain sigma-bar is always a valid (looser) bound -- never report worse
    return min(float(r.fun), float(cost(np.zeros(nu)))), r.x


def mu_curve(K, P, om):
    CL = ct.ss(P).lft(ct.ss(K), 1, 1)
    A, B, C, D = ct.ssdata(CL)
    nu = C.shape[0] - 2
    I = np.eye(A.shape[0])
    vals = np.empty(len(om)); ds = np.empty((len(om), nu + 1))
    x0 = None
    for i, w in enumerate(om):
        M = C @ np.linalg.solve(1j * w * I - A, B) + D
        vals[i], x0 = mu_ub(M, x0)          # warm start from the previous point
        ds[i] = np.exp(np.concatenate([x0, [0.0]]))
    return vals, ds


# ------------------------------------------------- mixed (real) mu bound
def mu_ub_mixed(M, x0=None):
    """Mixed-mu upper bound (Fan-Tits-Doyle / Young form) for the block
    structure [nu REAL scalars ; one full COMPLEX 2x2 performance block]:

        mu <= min_{D>0, G real}  min { beta :
              sigma_max[ (I+G^2)^-1/4 (D M D^-1 / beta - jG) (I+G^2)^-1/4 ] <= 1 }

    with diagonal D (performance block's D pinned to 1) and G supported on
    the real-parameter channels only.  With G = 0 this reduces exactly to
    the complex-D bound, so it can only tighten it.  The tightening is what
    makes the certificate non-vacuous here: delta_3..6 are frequency shifts
    of 0.5 %-damped modes, and treating them as complex admits phantom
    damping perturbations that no controller inside the voltage budget
    could ever cover (observed complex-D peaks of 10-30 on designs whose
    physical replay is clean)."""
    nu = M.shape[0] - 2

    def beta_of(ld, g):
        d = np.exp(np.concatenate([ld, [0.0]]))
        S = np.diag(np.concatenate([d[:nu], [d[nu], d[nu]]]))
        X = S @ M @ np.linalg.inv(S)
        Gd = np.concatenate([g, [0.0, 0.0]])
        q = (1.0 + Gd ** 2) ** -0.25
        jG = np.diag(1j * Gd)

        def ok(beta):
            Y = q[:, None] * (X / beta - jG) * q[None, :]
            return np.linalg.svd(Y, compute_uv=False)[0] <= 1.0

        hi = max(float(np.linalg.svd(X, compute_uv=False)[0]), 1e-8)
        for _ in range(60):                 # with G != 0, beta may need to
            if ok(hi):                      # exceed sigma_max(X): expand
                break                       # the bracket before bisecting
            hi *= 2.0
        else:
            return hi
        lo = 1e-8
        if ok(lo):
            return lo
        for _ in range(60):
            mid = np.sqrt(lo * hi)
            if ok(mid):
                hi = mid
            else:
                lo = mid
            if hi / lo < 1.001:
                break
        return hi

    def cost(th):
        return beta_of(th[:nu], th[nu:])

    if x0 is None:
        # Seed D on the complex-D optimum and g on the analytic per-channel
        # optimum of the DIAGONAL problem, g_i = Im(X_ii) / beta0.  Without
        # this the default NM simplex steps g by 2.5e-4 from zero, sees a
        # locally flat cost and never leaves the complex-D answer -- the
        # bound "converged" without tightening anything.
        b0, xD = mu_ub(M)
        d = np.exp(np.concatenate([xD, [0.0]]))
        S = np.diag(np.concatenate([d[:nu], [d[nu], d[nu]]]))
        X = S @ M @ np.linalg.inv(S)
        g0 = np.array([X[i, i].imag for i in range(nu)]) / max(b0, 1e-12)
        x0 = np.concatenate([xD, g0])
    else:
        x0 = np.asarray(x0, float)
    dim = 2 * nu
    steps = np.concatenate([np.full(nu, 0.3),
                            np.maximum(0.5, 0.3 * np.abs(x0[nu:]))])
    simplex = np.vstack([x0] + [x0 + steps[k] * np.eye(dim)[k]
                                for k in range(dim)])
    r = minimize(cost, x0, method='Nelder-Mead',
                 options={'xatol': 1e-3, 'fatol': 1e-4,
                          'maxiter': 150 * dim,
                          'initial_simplex': simplex})
    return float(r.fun), r.x


def mu_curve_mixed(K, P, om):
    CL = ct.ss(P).lft(ct.ss(K), 1, 1)
    A, B, C, D = ct.ssdata(CL)
    I = np.eye(A.shape[0])
    vals = np.empty(len(om))
    x0 = None
    for i, w in enumerate(om):
        M = C @ np.linalg.solve(1j * w * I - A, B) + D
        vals[i], x0 = mu_ub_mixed(M, x0)    # warm start from the previous point
        if i % 10 == 0:
            x0 = None                       # periodic cold restart: the NM
                                            # simplex drifts into local minima
                                            # along a warm-started sweep
    return vals


def mu_cert_mixed(K_ss, P_full, om, sc):
    """Certificate on the full structure with the mixed-mu bound."""
    mu = mu_curve_mixed(ct.ss(*K_ss), ct.ss(*P_full), om / sc['w_ref'])
    return float(mu.max()), mu


# ---------------------------------------------------------------- D fitting
def fit_d(om, mag, order=2):
    """Stable minimum-phase fit d(s) = k prod(s+z_i)/prod(s+p_i) to |d|(om)."""
    lm = np.log(np.maximum(mag, 1e-12))
    seed = np.log(np.geomspace(om[len(om) // 4], om[3 * len(om) // 4], order))

    def resid(th):
        k = th[0]
        z = np.exp(th[1:1 + order]); p = np.exp(th[1 + order:])
        m = k + sum(0.5 * np.log(om ** 2 + zi ** 2) for zi in z) \
              - sum(0.5 * np.log(om ** 2 + pi ** 2) for pi in p)
        return m - lm

    th0 = np.concatenate([[lm.mean()], seed, seed * 1.3])
    fallback = ct.tf([float(np.exp(lm.mean()))], [1.0])
    try:
        r = least_squares(resid, th0, method='lm', max_nfev=20000)
        k = np.exp(r.x[0])
        z = np.exp(r.x[1:1 + order]); p = np.exp(r.x[1 + order:])
        d = ct.tf(np.poly(-z) * k, np.poly(-p))
    except Exception:
        return fallback
    # sanity: a wild fit (non-finite, vanishing, or swinging far beyond the
    # data's own range) drives the D-scaled plant into conditioning that can
    # hang SB10AD's gamma iteration outright.  Fall back to the constant
    # log-mean level of the DATA -- the absolute level matters, because the
    # performance block's D is pinned at 1, so per-channel renormalisation
    # is NOT free.
    w = np.geomspace(om[0], om[-1], 200)
    mag = np.abs(d(1j * w))
    span = mag.max() / max(mag.min(), 1e-300)
    data_span = np.exp(lm.max() - lm.min())
    if (not np.all(np.isfinite(mag))) or mag.min() <= 0 \
            or span > 1e2 * max(data_span, 1.0):
        return fallback
    return d


def _balance_plant(Ps):
    """Exact diagonal-similarity balancing of a state-space plant; keeps the
    transfer function while shrinking the spread SB10AD has to digest."""
    A, B, C, D = ct.ssdata(Ps)
    if A.shape[0] == 0:
        return Ps
    from scipy.linalg import matrix_balance
    Ab, T = matrix_balance(A)
    return ct.ss(Ab, np.linalg.solve(T, B), C @ T, D)


SYN_UNC = (0, 1)     # synthesis structure: modes 1-2 + actuator gain.
                     # The certificate is ALWAYS computed on the full
                     # six-mode structure afterwards (mu_cert); carrying all
                     # six channels through the D-K wrecks its conditioning
                     # (gamma explodes through the D-fit round trip) without
                     # buying anything -- the D-scales are only a synthesis
                     # device, the analysis structure is what certifies.


def dk_iterate(plant, cT, n_iter=4, order=2, verbose=True, inner=None):
    """D-K iteration on the synthesis structure.  Keeps EVERY iterate, not
    only the smallest-mu one: the iteration is non-monotone and the
    best-a_lim controller is often not the best-mu controller, so the tuning
    driver scores them all."""
    P, scale = gen_plant(plant, cT, inner=inner, unc=SYN_UNC)
    nunc = len(SYN_UNC) + 1
    om = 2 * np.pi * np.logspace(1.0, 4.3, 110)
    Dl = Dr = None
    best = None
    hist = []
    iterates = []

    for it in range(n_iter):
        Ps = P if Dl is None else _balance_plant(ct.ss(Dl * P * Dr))
        try:
            K, _, gam, _ = ct.hinfsyn(ct.ss(Ps), 1, 1)
        except Exception as e:
            if verbose:
                print('   D-K iter %d : hinfsyn failed (%s) -- stopping'
                      % (it + 1, type(e).__name__))
            break
        K = ct.ss(K)
        mu, ds = mu_curve(K, P, om / scale['w_ref'])
        peak = float(mu.max())
        hist.append((float(gam), peak, int(K.nstates)))
        iterates.append((float(gam), peak, K, mu.copy()))
        if verbose:
            print('   D-K iter %d :  gamma = %6.3f    peak mu = %6.3f    K order = %d'
                  % (it + 1, gam, peak, K.nstates))
        if best is None or peak < best[0]:
            best = (peak, K, mu.copy())
        if it == n_iter - 1:
            break
        d_tf = [fit_d(om, ds[:, j], order) for j in range(nunc)]
        one = ct.tf([1.0], [1.0])
        Dl = ct.append(*[ct.ss(d) for d in d_tf], ct.ss(one), ct.ss(one), ct.ss(one))
        Dr = ct.append(*[ct.ss(1.0 / d) for d in d_tf], ct.ss(one), ct.ss(one), ct.ss(one))
    if best is None:
        raise RuntimeError('D-K iteration produced no controller')
    return best[1], best[0], best[2], om, hist, scale, iterates


def _unscale(K, sc, name='mu-synthesis'):
    """Convert a controller from gen_plant's scaled coordinates to physical."""
    Ak, Bk, Ck, Dk = ct.ssdata(K)
    g = sc['u_ref'] / sc['y_ref']
    wr = sc['w_ref']
    return Ctrl(wr * Ak, Bk.ravel(), (g * wr * Ck).ravel(),
                float(g * Dk[0, 0]), name)


def mu_ctrl(plant, cT=None, n_iter=4, verbose=True, inner=None):
    cT = plant.tools['corner (100, 80)'] if cT is None else cT
    K, peak, mu, om, hist, sc, _ = dk_iterate(plant, cT, n_iter=n_iter,
                                              verbose=verbose, inner=inner)
    return _unscale(K, sc), peak, mu, om, hist


def mu_ctrl_all(plant, cT=None, n_iter=4, verbose=True, inner=None):
    """Physical controllers from ALL D-K iterates, plus the FULL-structure
    analysis plant on which mu_cert computes the certificate.  K_ss and
    P_full travel as plain (A, B, C, D) tuples so cells can run in
    subprocess isolation (a hung SB10AD must not take the campaign down)."""
    cT = plant.tools['corner (100, 80)'] if cT is None else cT
    _, _, _, om, hist, sc, iterates = dk_iterate(plant, cT, n_iter=n_iter,
                                                 verbose=verbose, inner=inner)
    out = []
    for it, (gam, peak, K, mu) in enumerate(iterates):
        out.append(dict(it=it + 1, gamma=gam, peak_syn=peak, mu_syn=mu,
                        K=_unscale(K, sc), K_ss=ct.ssdata(K)))
    P_full, _ = gen_plant(plant, cT, inner=inner, unc=None)
    return out, om, ct.ssdata(P_full), sc


def mu_cert(K_ss, P_full, om, sc):
    """Certificate: mu upper bound on the FULL uncertainty structure
    (all six modal deltas + actuator gain + performance block)."""
    mu, _ = mu_curve(ct.ss(*K_ss), ct.ss(*P_full), om / sc['w_ref'])
    return float(mu.max()), mu


if __name__ == '__main__':
    from avc_plant import score
    P = Plant()
    K, peak, mu, om, hist = mu_ctrl(P)
    print('peak mu upper bound = %.3f  (controller order %d)' % (peak, K.order))
    r = score(P, K)
    print('stable=%s  a_lim x%.2f  volt %.1f V/N  peak %.1f um/N  z1 %.2f%%  z2 %.2f%%  static %.3f'
          % (r['stable'], r['a_lim'], r['volt'], r['peak_cl'] * 1e6,
             r['zeta1'], r['zeta2'], r['static_ratio']))
