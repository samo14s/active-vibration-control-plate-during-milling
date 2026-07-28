"""
Frontier-certified convex synthesis of active chatter control -- the new
strategy, formulated directly on the milling stability physics instead of an
H-infinity surrogate.

THE IDEA IN ONE LINE.  In the single-direction regenerative model the
speed-independent guaranteed depth is

    ap_floor = 2 / ( kappa4 * max_w [ |G(jw)| - Re G(jw) ] ),

where G is the operative (closed-loop) receptance at the cutting point.
|G| - Re G vanishes exactly when G is POSITIVE REAL: chatter-proofing the
plate means passivating the cutting point.  With a Youla/IMC parametrisation
on the (stable) patched plate,

    u = -Q y   =>   G_cl = G_DD - G_Du Q G_sD          (affine in Q!)
                    V(jw) = |Q(jw) G_sD(jw)|           (affine in Q!)

so "maximise the guaranteed depth subject to the voltage budget" is a
convex second-order-cone program in the coefficients of Q on a rational
basis -- solved to GLOBAL optimality, no D-K draws, no weight tuning.

THE CERTIFICATE NOBODY ELSE PRINTS.  Because |.| and Re are 1-Lipschitz,
any stabilising LTI controller obeying the per-frequency voltage spec
|Q G_sD| <= Vbar satisfies

    (|G_cl| - Re G_cl)(w)  >=  (|G_DD| - Re G_DD)(w) - 2 |G_Du(w)| Vbar(w)

pointwise -- a controller-independent lower bound on the achievable
chatter measure, i.e. an UPPER bound on the achievable depth:

    ap*  <=  2 / ( kappa4 * max_w [ (|G_DD| - Re G_DD) - 2 |G_Du| Vbar ]_+ ).

That is the actuator-authority frontier of the single-patch architecture:
it says how much ANY controller could buy, and at which frequencies the
patch runs out of authority.  The gap between our achieved floor and the
frontier quantifies how close to the physical ceiling the design sits.

ROBUSTNESS, convex and honest:
  * the program is solved SIMULTANEOUSLY on the sampled material-removal
    stages (fresh / mid / machined) and all milling positions: one Q, all
    constraints;
  * one fixed realised controller K = Q/(1 - G_su Q) on a perturbed plant
    keeps internal stability iff |Q (G_su^p - G_su)| < 1 (small gain on the
    IMC mismatch); the program constrains |Q| rho_Delta <= 0.5 with
    rho_Delta the sampled worst mismatch, and the full-discretisation
    method verifies every closed loop afterwards;
  * the milling coefficient scale (their 0.3-2.9x band) multiplies every
    law's floor equally and is reported as a band, not hidden.

A speed-SCHEDULED variant solves, per spindle speed, the pure LP
    max m  s.t.  Re[(1 - e^{-jw tau}) G_cl(jw)] >= -m  for all w
which lifts the lobe at the operating speed beyond the broadband floor.
"""
import numpy as np
import control as ct
import cvxpy as cp
import milling_model as mm
from avc_plant import Plant, Ctrl, NM

POSITIONS = [0.0, 25.0, 50.0, 75.0, 100.0]      # mm along the upper edge
VBAR = 150.0                                    # V per N voltage spec
QSG_CAP = 0.5                                   # small-gain cap |Q dG_su|
BSCALE = 1.0e6                                  # basis gain: theta stays O(1)
DRIFT = 0.02                                    # +/- frequency drift covered
                                                # between convex re-solves


# ------------------------------------------------------------------ basis
def basis_bank(f_lo=250.0, f_hi=5200.0, nsec=18, xi=0.7):
    """Stable second-order sections (cos & sin type) + a low-pass tail.
    Returns list of ct.TransferFunction, SISO, unity-ish peak gain."""
    fs = np.geomspace(f_lo, f_hi, nsec)
    bank = []
    for f in fs:
        w = 2 * np.pi * f
        bank.append(ct.tf([w ** 2], [1.0, 2 * xi * w, w ** 2]))
        bank.append(ct.tf([w, 0.0], [1.0, 2 * xi * w, w ** 2]))
    wl = 2 * np.pi * f_lo
    bank.append(ct.tf([wl], [1.0, wl]))
    return bank


def eval_bank(bank, om):
    """Complex responses, shape (len(bank), len(om)); O(1) magnitudes.

    UNIT SYSTEM of the convex programs: receptances are stored in um/N
    (UM = 1e6) and the solver's Q is in V/um, so the physical parameter is
    Q_true = BSCALE * (theta . basis) [V/m] with BSCALE = UM.  Voltage
    products Q_solver * G'_sD are then directly in V/N, the chatter measure
    c is in um/N, and every quantity the solver sees is O(1)-O(1e3)."""
    return np.array([np.squeeze(b(1j * om)) for b in bank])


UM = 1.0e6                                       # um/N unit scale


# ------------------------------------------------------------- plant data
def plant_set():
    P0 = Plant(zeta=mm.ZETA_PAPER)
    Pm = Plant(zeta=mm.ZETA_PAPER, freq_scale=mm.REMOVAL_PAPER)
    Pmid = Plant(zeta=mm.ZETA_PAPER,
                 freq_scale=1.0 + 0.5 * (mm.REMOVAL_PAPER - 1.0))
    return [('fresh', P0), ('mid', Pmid), ('machined', Pm)]


def freq_grid(n=420, f_lo=60.0, f_hi=5200.0):
    return 2 * np.pi * np.geomspace(f_lo, f_hi, n)


def problem_data(plants=None, om=None, positions=POSITIONS):
    """FRFs the program needs, for every (plant, position).  rho is the
    Gsu mismatch envelope ACROSS the given plants: pass one removal stage
    with its +/-DRIFT neighbours for the adaptive design, or all stages
    for a (very conservative) fixed-K-over-the-whole-cut design."""
    plants = plant_set() if plants is None else plants
    om = freq_grid() if om is None else om
    data = []
    Gsu_all = []
    for tag, P in plants:
        b, cS = P.bmod, P.cS
        Gsu = UM * P.frf(cS, b, om)
        Gsu_all.append(Gsu)
        for x in positions:
            D = P.edge_D(x)
            data.append(dict(
                plant=tag, x=x,
                GDD=UM * P.frf(D, D, om),
                GDu=UM * P.frf(D, b, om),
                GsD=UM * P.frf(cS, D, om),
            ))
    Gsu_all = np.array(Gsu_all)
    rho = np.abs(Gsu_all - Gsu_all[0]).max(axis=0)      # mismatch envelope
    return dict(om=om, cases=data, Gsu0=Gsu_all[0], rho=rho)


def stage_data(stage_scale=None, om=None, positions=POSITIONS, drift=DRIFT):
    """Adaptive-design data for ONE removal stage: the stage plant plus its
    +/-drift frequency neighbours (the drift the law must ride out between
    two convex re-solves; a re-solve costs seconds, material removal takes
    minutes, so a few percent is generous)."""
    stage_scale = np.ones(NM) if stage_scale is None else np.asarray(stage_scale)
    plants = [('stage', Plant(zeta=mm.ZETA_PAPER, freq_scale=stage_scale)),
              ('stage-', Plant(zeta=mm.ZETA_PAPER,
                               freq_scale=stage_scale * (1.0 - drift))),
              ('stage+', Plant(zeta=mm.ZETA_PAPER,
                               freq_scale=stage_scale * (1.0 + drift)))]
    return problem_data(plants=plants, om=om, positions=positions)


# --------------------------------------- passivity-damped inner loop layer
def damped_aug(P, K_in):
    """State-space of the plant with the PASSIVE inner law pre-closed
    (u = u_in + u_Q, u_in = K_in y; K_in is collocated PD -- rank-1 PSD
    stiffness + damping updates, stable against ANY frequency drift).
    Returns (Ad, Bu_d, rows) with rows for y, u_in and a Bw/Cz builder."""
    n = NM
    nk = K_in.order
    N = 2 * n + nk
    Ad = np.zeros((N, N))
    Ad[:2 * n, :2 * n] = P.A + P.Bu @ (K_in.Dk * P.Cy)
    if nk:
        Ad[:2 * n, 2 * n:] = P.Bu @ K_in.Ck
        Ad[2 * n:, :2 * n] = K_in.Bk @ P.Cy
        Ad[2 * n:, 2 * n:] = K_in.Ak
    Bu_d = np.zeros((N, 1)); Bu_d[:2 * n] = P.Bu
    y_row = np.zeros(N); y_row[:2 * n] = P.Cy[0]
    uin_row = np.zeros(N)
    uin_row[:2 * n] = K_in.Dk * P.Cy[0]
    if nk:
        uin_row[2 * n:] = K_in.Ck.ravel()
    return Ad, Bu_d, y_row, uin_row


def _frf_rows(Ad, Bcols, Crows, om):
    """FRFs of Crows x (jwI - Ad)^-1 x Bcols via one eigendecomposition."""
    lam, V = np.linalg.eig(Ad)
    Vi = np.linalg.inv(V)
    out = np.empty((len(Crows), len(Bcols), len(om)), complex)
    for jb, Bc in enumerate(Bcols):
        w = Vi @ Bc
        for ic, Cr in enumerate(Crows):
            cv = Cr @ V
            out[ic, jb] = ((cv * w)[None, :] /
                           (1j * om[:, None] - lam[None, :])).sum(1)
    return out


def stage_data_damped(K_in, stage_scale=None, om=None, positions=POSITIONS,
                      drift=DRIFT):
    """Adaptive-stage data on the PD-damped plant.  The passive inner loop
    widens every resonance to a few percent bandwidth, which collapses the
    drift envelope rho and lets the convex layer actually spend the
    voltage budget.  Voltage cones use the TOTAL patch voltage:

        V_tot(w) = V0(w) - W1(w) Q(w)   per unit cutting force, with
        V0 = u_in response, W1 = (1 + L H Bu) G~_sD  -- affine in Q.
    """
    stage_scale = np.ones(NM) if stage_scale is None else np.asarray(stage_scale)
    om = freq_grid() if om is None else om
    plants = [('stage', stage_scale), ('stage-', stage_scale * (1 - drift)),
              ('stage+', stage_scale * (1 + drift))]
    cases = []
    Gsu_all = []
    for tag, sc in plants:
        P = Plant(zeta=mm.ZETA_PAPER, freq_scale=sc)
        Ad, Bu_d, y_row, uin_row = damped_aug(P, K_in)
        n2 = 2 * NM
        for x in positions:
            D = P.edge_D(x)
            Bw = np.zeros((Ad.shape[0], 1)); Bw[NM:n2, 0] = D
            Dz = np.zeros(Ad.shape[0]); Dz[:NM] = D
            F = _frf_rows(Ad, [Bw[:, 0], Bu_d[:, 0]],
                          [Dz, y_row, uin_row], om)
            GDD, GDu = UM * F[0, 0], UM * F[0, 1]
            GsD, Gsu = UM * F[1, 0], UM * F[1, 1]
            uin_d, uin_u = F[2, 0], F[2, 1]
            V0 = uin_d                              # V/N before Q acts
            W1 = (1.0 + uin_u) * GsD / UM           # V/N per unit Q GsD
            cases.append(dict(plant=tag, x=x, GDD=GDD, GDu=GDu, GsD=GsD,
                              V0=V0, W1=W1 * UM))
            if x == positions[0]:
                Gsu_all.append(Gsu)
    Gsu_all = np.array(Gsu_all)
    rho = np.abs(Gsu_all - Gsu_all[0]).max(axis=0)
    return dict(om=om, cases=cases, Gsu0=Gsu_all[0], rho=rho)


# ------------------------------------------------------- convex synthesis
def synth_floor(dat, bank=None, vbar=VBAR, verbose=True):
    """Global maximisation of the guaranteed broadband floor:
        minimise  c
        s.t.  |G_cl| - Re G_cl <= c            all plants, positions, w
              |Q G_sD|         <= vbar         (voltage per unit force)
              |Q| rho_Delta    <= QSG_CAP      (IMC small gain)
    with Q = theta . basis.  Returns theta, c*, and the achieved floors.
    """
    om = dat['om']
    bank = basis_bank() if bank is None else bank
    B = eval_bank(bank, om)                     # (nb, nw)
    nb = B.shape[0]
    th = cp.Variable(nb)
    c = cp.Variable()
    cons = []
    Qre = B.real.T @ th                         # (nw,)
    Qim = B.imag.T @ th
    mul = cp.multiply
    CSCALE = max((np.abs(cs['GDD']) - cs['GDD'].real).max()
                 for cs in dat['cases'])            # normalise the c-cones
    for case in dat['cases']:
        GDD, GDu, GsD = case['GDD'], case['GDu'], case['GsD']
        M = GDu * GsD
        # G_cl = GDD - M*Q ; with Q = Qre + j Qim   (all cones RHS ~ O(1))
        Re_cl = (GDD.real - (mul(M.real, Qre) - mul(M.imag, Qim))) / CSCALE
        Im_cl = (GDD.imag - (mul(M.real, Qim) + mul(M.imag, Qre))) / CSCALE
        cons.append(cp.SOC(Re_cl + c, cp.vstack([Re_cl, Im_cl]).T, axis=1))
        # TOTAL voltage: |V0 - W1 Q| <= vbar, normalised to unit RHS
        # (V0 = inner-loop response, W1 = effective actuation map; for the
        #  plain undamped path V0 = 0 and W1 = GsD)
        V0 = case.get('V0', np.zeros(len(om)))
        W1 = case.get('W1', GsD)
        Vre = V0.real / vbar - (mul(W1.real / vbar, Qre) - mul(W1.imag / vbar, Qim))
        Vim = V0.imag / vbar - (mul(W1.real / vbar, Qim) + mul(W1.imag / vbar, Qre))
        cons.append(cp.SOC(np.ones(len(om)),
                           cp.vstack([Vre, Vim]).T, axis=1))
    # IMC small-gain robustness, unit RHS, coefficient floor for scaling
    w = np.minimum(dat['rho'] / QSG_CAP, 1e4)
    cons.append(cp.SOC(np.ones(len(om)),
                       cp.vstack([mul(w, Qre), mul(w, Qim)]).T, axis=1))
    prob = cp.Problem(cp.Minimize(c), cons)
    stat = None
    for solver, kw in [(cp.CLARABEL, {}),
                       (cp.SCS, dict(max_iters=60000)),
                       (cp.ECOS, {})]:
        try:
            prob.solve(solver=solver, verbose=False, **kw)
            if th.value is not None:
                stat = '%s/%s' % (prob.status, solver)
                break
        except Exception:
            continue
    if verbose:
        print('floor SOCP: %s   c* = %.4e um/N' % (stat, c.value * CSCALE))
    kap = abs(mm.kappa4())
    floor = UM / (kap * c.value * CSCALE)
    return np.asarray(th.value), float(c.value * CSCALE), float(floor), bank


def synth_scheduled(dat, rpm, bank=None, vbar=VBAR, verbose=False):
    """Per-speed LP: maximise the lobe at spindle speed rpm.
        minimise c_tau = max_w  -Re[(1-e^{-jw tau}) G_cl]
    subject to the same voltage and small-gain cones."""
    om = dat['om']
    tau = mm.tau_of(rpm)
    E = 1.0 - np.exp(-1j * om * tau)
    bank = basis_bank() if bank is None else bank
    B = eval_bank(bank, om)
    nb = B.shape[0]
    th = cp.Variable(nb)
    c = cp.Variable()
    cons = []
    Qre = B.real.T @ th
    Qim = B.imag.T @ th
    mul = cp.multiply
    CSCALE = max((np.abs(cs['GDD']) - cs['GDD'].real).max()
                 for cs in dat['cases'])
    for case in dat['cases']:
        GDD, GDu, GsD = case['GDD'], case['GDu'], case['GsD']
        M = GDu * GsD
        Re_cl = (GDD.real - (mul(M.real, Qre) - mul(M.imag, Qim))) / CSCALE
        Im_cl = (GDD.imag - (mul(M.real, Qim) + mul(M.imag, Qre))) / CSCALE
        ReE = mul(E.real, Re_cl) - mul(E.imag, Im_cl)
        cons.append(ReE >= -c)
        V0 = case.get('V0', np.zeros(len(om)))
        W1 = case.get('W1', GsD)
        Vre = V0.real / vbar - (mul(W1.real / vbar, Qre) - mul(W1.imag / vbar, Qim))
        Vim = V0.imag / vbar - (mul(W1.real / vbar, Qim) + mul(W1.imag / vbar, Qre))
        cons.append(cp.SOC(np.ones(len(om)),
                           cp.vstack([Vre, Vim]).T, axis=1))
    w = np.minimum(dat['rho'] / QSG_CAP, 1e4)
    cons.append(cp.SOC(np.ones(len(om)),
                       cp.vstack([mul(w, Qre), mul(w, Qim)]).T, axis=1))
    prob = cp.Problem(cp.Minimize(c), cons)
    for solver, kw in [(cp.CLARABEL, {}), (cp.ECOS, {}),
                       (cp.SCS, dict(max_iters=100000))]:
        try:
            prob.solve(solver=solver, verbose=False, **kw)
            if th.value is not None:
                break
        except Exception:
            continue
    kap = abs(mm.kappa4())
    ap = UM / (kap * max(c.value * CSCALE, 1e-18))
    if verbose:
        print('  %5.0f rpm scheduled LP: %s  ap %.3f mm'
              % (rpm, prob.status, ap * 1e3))
    return np.asarray(th.value), float(ap)


# -------------------------------------------------------------- frontier
def frontier(dat, vbar=VBAR):
    """Controller-independent authority frontier per (plant, position) and
    overall: the depth NO admissible LTI design can exceed."""
    kap = abs(mm.kappa4())
    worst = 0.0
    per = {}
    for case in dat['cases']:
        base = np.abs(case['GDD']) - case['GDD'].real
        relief = 2.0 * np.abs(case['GDu']) * vbar
        rem = np.maximum(base - relief, 0.0)
        per[(case['plant'], case['x'])] = rem
        worst = max(worst, rem.max())
    ap_star = np.inf if worst <= 0 else UM / (kap * worst)
    return ap_star, per


# ------------------------------------------------------------ realisation
def realise(theta, bank, plant):
    """Physical controller u = K_ctrl y from Q = theta . basis:
        K = Q (1 - G_su Q)^{-1},  u = -K y  ->  Ctrl carries -K."""
    Q = None
    for t, b in zip(theta, bank):
        term = ct.ss(b) * float(t * BSCALE)
        Q = term if Q is None else Q + term
    n = NM
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -np.diag(plant.wn ** 2)
    A[n:, n:] = -np.diag(2 * plant.zeta * plant.wn)
    Bu = np.zeros((2 * n, 1)); Bu[n:, 0] = plant.bmod
    Cy = np.zeros((1, 2 * n)); Cy[0, :n] = plant.cS
    Gsu = ct.ss(A, Bu, Cy, 0.0)
    K = ct.feedback(Q, Gsu, sign=+1)            # Q / (1 - Gsu Q)
    K = ct.ss(K)
    Ak, Bk, Ck, Dk = ct.ssdata(K)
    return Ctrl(Ak, Bk.ravel(), (-Ck).ravel(), float(-Dk[0, 0]),
                'convex frontier law'), Q


def q_of(theta, bank, om):
    return eval_bank(bank, om).T @ theta


def realise_damped(theta, bank, plant, K_in):
    """Total law = passive inner PD  +  convex layer realised against the
    DAMPED plant:  K_Q = Q (1 - G~_su Q)^{-1},  u = u_in - K_Q y."""
    from avc_combined import parallel
    Q = None
    for t, b in zip(theta, bank):
        term = ct.ss(b) * float(t * BSCALE)
        Q = term if Q is None else Q + term
    Ad, Bu_d, y_row, uin_row = damped_aug(plant, K_in)
    Gsu = ct.ss(Ad, Bu_d, y_row[None, :], np.zeros((1, 1)))
    KQ = ct.ss(ct.feedback(Q, Gsu, sign=+1))
    Ak, Bk, Ck, Dk = ct.ssdata(KQ)
    K_Q = Ctrl(Ak, Bk.ravel(), (-Ck).ravel(), float(-Dk[0, 0]), 'convex layer')
    return parallel(K_in, K_Q, 'passive + convex (new law)')
