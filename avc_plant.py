"""
Shared plant + scoring for the controller comparison.

Every control law is reduced to one common object -- a state-space
K : y_sensor (m) -> u (V), sign convention u = K y -- so that the three
designs are closed on exactly the same plant and scored by exactly the same
metrics.  Nothing about the comparison depends on which law is being scored.

Plant (modal ROM, NM modes):
    x = [q ; qdot]
    xdot = A x + Bu u + Bw w
    z    = Cz x     tool displacement            [m per N]
    y    = Cy x     patch sensor (collocated)    [m, modal]
"""
import numpy as np
from patch_fem import modal, Lx, Ly, Vmax

# ------------------------------------------------------------------ settings
NX, NY, NM = 40, 32, 6
XP0, YP0 = 0.005, 0.0          # patch lower-left corner (5 mm spanwise, at the clamp)
ZETA = 0.005                   # open-loop modal damping
VLIM = 150.0                   # V per N of cutting force  (patch limit 200 V)
ZMIN = 0.0045                  # no-spillover floor: no mode below the open-loop 0.5 %

# material removal observed on this benchmark: f1 +17 %, f2 +11 %
REMOVAL = np.array([1.17, 1.11, 1.05, 1.05, 1.05, 1.05])


# ------------------------------------------------------------------ plant
class Plant:
    def __init__(self, freq_scale=None, zeta=None):
        f, Phi, bmod, bvec, nodes, K, M, free = modal(NX, NY, XP0, YP0, NM)
        self.f_patched = f.copy()
        if freq_scale is not None:
            f = f * np.asarray(freq_scale)[:NM]
        self.f = f
        self.wn = 2 * np.pi * f
        self.bmod = bmod
        self.nodes = nodes
        self.Phi = Phi
        # per-mode damping; scalar ZETA kept as the legacy default so every
        # earlier result reproduces, the paper-matched configuration passes
        # the measured Table-4 vector instead
        self.zeta = np.full(NM, ZETA) if zeta is None \
            else np.broadcast_to(np.asarray(zeta, float), (NM,)).copy()

        node_at = lambda x, y: int(np.argmin((nodes[:, 0] - x) ** 2 + (nodes[:, 1] - y) ** 2))
        self._node_at = node_at
        self.tools = {
            'corner (100, 80)': Phi[3 * node_at(Lx, Ly), :].copy(),
            'mid free edge (50, 80)': Phi[3 * node_at(Lx / 2, Ly), :].copy(),
        }
        # collocated sensor: second identical patch on the opposite face
        self.cS = bmod.copy()

        n = NM
        self.A = np.zeros((2 * n, 2 * n))
        self.A[:n, n:] = np.eye(n)
        self.A[n:, :n] = -np.diag(self.wn ** 2)
        self.A[n:, n:] = -np.diag(2 * self.zeta * self.wn)
        self.Bu = np.zeros((2 * n, 1)); self.Bu[n:, 0] = bmod
        self.Cy = np.zeros((1, 2 * n)); self.Cy[0, :n] = self.cS

    def edge_D(self, x_mm):
        """w-mode-shape row at the cutting point (x_mm, upper edge)."""
        return self.Phi[3 * self._node_at(x_mm * 1e-3, Ly), :].copy()

    def Bw(self, cT):
        B = np.zeros((2 * NM, 1)); B[NM:, 0] = cT
        return B

    def Cz(self, cT):
        C = np.zeros((1, 2 * NM)); C[0, :NM] = cT
        return C

    def frf(self, cout, cin, om):
        """Generic modal FRF  sum_k cout_k cin_k / (wn_k^2 - w^2 + 2j z w w)."""
        om = np.asarray(om)
        return sum(cout[k] * cin[k] /
                   (self.wn[k] ** 2 - om ** 2 + 2j * self.zeta[k] * self.wn[k] * om)
                   for k in range(NM))

    def open_loop_frf(self, cT, om):
        return self.frf(cT, cT, om)

    def static_compliance(self, cT):
        return float(sum(cT[k] ** 2 / self.wn[k] ** 2 for k in range(NM)))


# ------------------------------------------------------------------ controller
class Ctrl:
    """State-space controller  u = Ck xk + Dk y ,  xkdot = Ak xk + Bk y."""

    def __init__(self, Ak, Bk, Ck, Dk, name=''):
        self.Ak = np.atleast_2d(np.asarray(Ak, float))
        self.Bk = np.asarray(Bk, float).reshape(-1, 1) if np.size(Ak) else np.zeros((0, 1))
        self.Ck = np.asarray(Ck, float).reshape(1, -1) if np.size(Ak) else np.zeros((1, 0))
        self.Dk = float(Dk)
        self.name = name
        if np.size(Ak) == 0:
            self.Ak = np.zeros((0, 0))

    @property
    def order(self):
        return self.Ak.shape[0]

    def tf(self, om):
        """Frequency response K(jw)."""
        if self.order == 0:
            return np.full(np.shape(om), self.Dk, dtype=complex)
        I = np.eye(self.order)
        return np.array([self.Dk + (self.Ck @ np.linalg.solve(1j * w * I - self.Ak,
                                                              self.Bk))[0, 0] for w in np.atleast_1d(om)])


def close(plant, ctrl, cT):
    """Closed-loop (A, Bw, Cz, Cv) with Cv the control-voltage output."""
    n2 = 2 * NM
    nk = ctrl.order
    A = np.zeros((n2 + nk, n2 + nk))
    A[:n2, :n2] = plant.A + plant.Bu @ (ctrl.Dk * plant.Cy)
    if nk:
        A[:n2, n2:] = plant.Bu @ ctrl.Ck
        A[n2:, :n2] = ctrl.Bk @ plant.Cy
        A[n2:, n2:] = ctrl.Ak
    Bw = np.zeros((n2 + nk, 1)); Bw[:n2] = plant.Bw(cT)
    Cz = np.zeros((1, n2 + nk)); Cz[0, :n2] = plant.Cz(cT)
    Cv = np.zeros((1, n2 + nk))
    Cv[0, :n2] = ctrl.Dk * plant.Cy
    if nk:
        Cv[0, n2:] = ctrl.Ck
    return A, Bw, Cz, Cv


# ------------------------------------------------------------------ metrics
FR = np.linspace(5, 4500, 3000)
OM = 2 * np.pi * FR


def _refine(fun, fr, vals, mode='min'):
    """Grid search followed by local Brent refinement -> grid-independent extremum."""
    from scipy.optimize import minimize_scalar
    i = int(np.argmin(vals) if mode == 'min' else np.argmax(vals))
    lo = fr[max(i - 1, 0)]; hi = fr[min(i + 1, len(fr) - 1)]
    sgn = 1.0 if mode == 'min' else -1.0
    if hi <= lo:
        return vals[i]
    r = minimize_scalar(lambda x: sgn * fun(x), bounds=(lo, hi), method='bounded',
                        options={'xatol': 1e-6})
    best = sgn * r.fun
    return min(best, vals[i]) if mode == 'min' else max(best, vals[i])


def score(plant, ctrl, fr=None, full=False):
    """Uniform scoring of any control law on any plant.

    Returns dict with the mechanical / productivity / effort metrics.
    a_lim lift uses the chatter-relevant most-negative real part of the
    receptance, taken as the worst over the tool positions.
    """
    om = OM if fr is None else 2 * np.pi * np.asarray(fr)
    res = {'name': ctrl.name, 'order': ctrl.order, 'stable': True}
    lifts, volts, peaks, peaks_ol, statics = [], [], [], [], []
    per_tool = {}

    for tag, cT in plant.tools.items():
        A, Bw, Cz, Cv = close(plant, ctrl, cT)
        lam = np.linalg.eigvals(A)
        if lam.real.max() > -1e-9:
            res['stable'] = False
        I = np.eye(A.shape[0])
        H = np.array([np.linalg.solve(1j * w * I - A, Bw)[:, 0] for w in om])
        Gcl = H @ Cz[0]
        Vcl = H @ Cv[0]
        Gol = plant.open_loop_frf(cT, om)

        frg = om / (2 * np.pi)
        Ainv = lambda w, C, B=Bw: (C @ np.linalg.solve(1j * 2 * np.pi * w * I - A, B))[0, 0]
        re_cl = _refine(lambda w: Ainv(w, Cz).real, frg, Gcl.real, 'min')
        re_ol = _refine(lambda w: plant.open_loop_frf(cT, 2 * np.pi * w).real,
                        frg, Gol.real, 'min')
        lifts.append(re_ol / re_cl)
        volts.append(_refine(lambda w: abs(Ainv(w, Cv)), frg, np.abs(Vcl), 'max'))
        peaks.append(_refine(lambda w: abs(Ainv(w, Cz)), frg, np.abs(Gcl), 'max'))
        peaks_ol.append(_refine(lambda w: abs(plant.open_loop_frf(cT, 2 * np.pi * w)),
                                frg, np.abs(Gol), 'max'))
        # static compliance: -Cz A^-1 Bw
        statics.append(float(-(Cz @ np.linalg.solve(A, Bw))[0, 0]))
        if full:
            per_tool[tag] = dict(fr=om / 2 / np.pi, Gol=Gol, Gcl=Gcl, Vcl=Vcl)

    res['a_lim'] = min(lifts)
    res['volt'] = max(volts)
    res['peak_cl'] = max(peaks)
    res['peak_ol'] = max(peaks_ol)
    res['peak_ratio'] = max(peaks_ol) / max(peaks)
    res['static_cl'] = statics[0]
    res['static_ol'] = plant.static_compliance(plant.tools['corner (100, 80)'])
    res['static_ratio'] = res['static_ol'] / statics[0]

    A = close(plant, ctrl, plant.tools['corner (100, 80)'])[0]
    lam = np.linalg.eigvals(A)
    osc = lam[lam.imag > 1.0]
    zt = -osc.real / np.abs(osc)
    res['zeta_min'] = zt.min() if len(zt) else 0.0
    order_ = np.argsort(osc.imag)
    res['poles'] = [(osc[i].imag / 2 / np.pi, zt[i] * 100) for i in order_]
    # damping attributed to the two targeted modes: closed-loop poles nearest
    # the open-loop mode-1 / mode-2 frequencies
    res['zeta1'], res['zeta2'] = [], []
    for k in (0, 1):
        near = [(abs(p[0] - plant.f[k]), p[1]) for p in res['poles']]
        near.sort()
        res['zeta1' if k == 0 else 'zeta2'] = near[0][1]
    if full:
        res['per_tool'] = per_tool
    return res


def settling(plant, ctrl, cT, T=0.30, dt=5e-6):
    """5 % settling time of the impulse response at the tool point."""
    from scipy.linalg import expm
    A, Bw, Cz, Cv = close(plant, ctrl, cT)
    t = np.arange(0, T, dt)
    Ad = expm(A * dt)
    x = Bw[:, 0].copy()
    y = np.empty_like(t); v = np.empty_like(t)
    for i in range(len(t)):
        y[i] = Cz[0] @ x; v[i] = Cv[0] @ x; x = Ad @ x
    return t, y, v


def settle_time(t, y, env):
    idx = np.where(np.abs(y) > 0.05 * env)[0]
    return t[idx[-1]] * 1000 if len(idx) else 0.0
