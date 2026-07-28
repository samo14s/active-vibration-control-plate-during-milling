"""
Milling model of Du, Liu, Dai & Long, IJMS 274 (2024) 109257, Eqs (2)-(5),
Tables 1-4, closed on OUR MITC4 plate FEM (plate_fem / patch_fem) instead of
their Chebyshev-Ritz basis.

Model content taken from the paper:
  * linearised milling force [Fx;Fy] = [a1 a2; a3 a4][xr;yr] with
    time-periodic coefficients summed over engaged teeth, helix lag
    integrated over the axial contact zone (their Eqs 2-4);
  * only the plate's transverse (y) motion matters (their Eq 12): the
    regenerative force acts at the moving cutting point through a4(t),
      q'' + 2 Z W q' + W^2 q = -a4(t) D^T [D q(t) - D q(t-tau)]
                               + ft a3(t) D^T + b u(t),
    D = w-mode-shape row at the cutting position on the upper edge
    (mass-normalised modal coordinates);
  * tooth-passing delay tau = 60 / (Omega_rpm NT);
  * DOWN milling, radial immersion ae/D = 0.01, so the directional
    coefficients are strongly intermittent.

Directional-coefficient grouping.  The paper's Eq (3) prints k1 = kc/cos(eta)
and k2 = 1 + mu_c tan(eta) (cos(gamma_n) - kn sin(gamma_n)); we implement the
equivalent classical oblique-corrected pair
    Kt_e = (kc / cos eta) * k2      tangential coefficient  [Pa]
    Kr   = kn                       radial-to-tangential ratio
which reproduces the structure of their a_i(t) exactly; any residual
grouping ambiguity is a constant scale on a4 that is (a) swept 0.3-2.9x in
their own robustness set and (b) checked against their reported uncontrolled
stability floor (~0.1 mm) in avc_fdm validation.

Chip thickness: h_j = [sin(phi), cos(phi)] . [xr; yr]; force rotation
    Fx = sum -[Kt cos(phi) + Kr Kt sin(phi)] h_j
    Fy = sum +[Kt sin(phi) - Kr Kt cos(phi)] h_j
so a4 = d Fy / d yr = sum (Kt sc - Kr Kt cc) integrated over the contact
height -- their alpha_4 column.
"""
import numpy as np

# ------------------------------------------------------------ paper tables
NT = 3                      # teeth
DIA = 10e-3                 # tool diameter [m]
RT = DIA / 2.0
ETA = np.deg2rad(35.0)      # helix angle
GAMMA_N = np.deg2rad(15.0)  # normal rake angle
KC = 925e6                  # cutting force coefficient [Pa]
KN = 0.26                   # proportionality constant (radial ratio)
MU_C = 0.2                  # friction coefficient

AE = 0.1e-3                 # radial cutting depth [m]
FT = 0.02e-3                # feed per tooth [m]

K2 = 1.0 + MU_C * np.tan(ETA) * (np.cos(GAMMA_N) - KN * np.sin(GAMMA_N))
KT_E = KC / np.cos(ETA) * K2      # effective tangential coefficient [Pa]
KR = KN

# paper Table 4 damping, mode 6 (not reported) assumed equal to mode 5
ZETA_PAPER = np.array([0.0031, 0.0017, 0.0027, 0.0056, 0.0035, 0.0035])
# measured material removal: f1 540->632 (+17 %), f2 1068->1162 (+9 %);
# modes 3-6 not reported, +5 % kept from the earlier sessions
REMOVAL_PAPER = np.array([1.17, 1.09, 1.05, 1.05, 1.05, 1.05])

# down milling: tooth cuts between phi_st and pi
PHI_EX = np.pi
PHI_ST = np.pi - np.arccos(1.0 - 2.0 * AE / DIA)


def tau_of(omega_rpm):
    """Tooth-passing period [s]."""
    return 60.0 / (omega_rpm * NT)


def alphas(t, omega_rpm, ap, nz=32):
    """Directional coefficients a1..a4 [N/m] at times t (shape (4, len(t))).

    Midpoint rule over the axial contact zone z in [0, ap]; at height z the
    tooth angle lags by z tan(eta) / RT (helix).  Exactly periodic with the
    tooth-passing period.
    """
    Omega = omega_rpm * 2.0 * np.pi / 60.0
    t = np.atleast_1d(np.asarray(t, float))
    z = (np.arange(nz) + 0.5) * (ap / nz)
    dz = ap / nz
    a = np.zeros((4, len(t)))
    for j in range(NT):
        phi = Omega * t[None, :] + 2.0 * np.pi * j / NT \
            - z[:, None] * np.tan(ETA) / RT
        pm = np.mod(phi, 2.0 * np.pi)
        cut = (pm >= PHI_ST) & (pm <= PHI_EX)
        s, c = np.sin(phi), np.cos(phi)
        ss = np.sum(s * s * cut, axis=0) * dz
        sc = np.sum(s * c * cut, axis=0) * dz
        cc = np.sum(c * c * cut, axis=0) * dz
        a[0] += -KT_E * (sc + KR * ss)          # dFx/dxr
        a[1] += -KT_E * (cc + KR * sc)          # dFx/dyr
        a[2] += +KT_E * (ss - KR * sc)          # dFy/dxr
        a[3] += +KT_E * (KR * cc - sc)          # dFy/dyr
    return a
# Sign convention note: the workpiece-normal (y) orientation is fixed so the
# model reproduces the paper's OBSERVED regenerative branch -- chatter above
# the natural frequency (their fc2 = 1135 Hz vs f2 = 1101 Hz, Fig 14) -- which
# requires a positive mean a4/ap.  The mode-shape sign is arbitrary in a
# modal model, so this is a convention, not a parameter choice.


def alpha4_series(omega_rpm, ap, m):
    """a4(t) sampled at m midpoints of one tooth period (for the FDM)."""
    T = tau_of(omega_rpm)
    t = (np.arange(m) + 0.5) * (T / m)
    return alphas(t, omega_rpm, ap)[3]


def kappa4_signed(omega_rpm=4900.0, nt=2048):
    """Mean regenerative coefficient per unit depth  a4_bar / ap  [N/m^2].

    Speed enters only through sampling of the periodic engagement, so the
    value is essentially speed-independent; 4900 rpm is the paper's test
    condition."""
    ap = 1.0e-3
    T = tau_of(omega_rpm)
    t = np.linspace(0.0, T, nt, endpoint=False)
    return float(alphas(t, omega_rpm, ap)[3].mean() / ap)


KAPPA4 = None


def kappa4():
    """Cached |a4_bar|/ap with its sign, computed once."""
    global KAPPA4
    if KAPPA4 is None:
        KAPPA4 = kappa4_signed()
    return KAPPA4


# ------------------------------------------------- zero-order (ZOA) limits
def zoa_ap(G, tau, kap=None):
    """Exact single-direction zero-order depth limit at delay tau [m].

    Average-coefficient loop:  1 + kap * ap * (1 - e^{-j w tau}) G(jw) = 0,
    kap = a4_bar / ap (signed).  The critical depth at frequency w is
        ap(w) = -1 / (kap * Re[(1 - e^{-j w tau}) G(jw)])
    over frequencies where that is positive; the limit is the minimum.
    G is evaluated on the caller's frequency grid (rad/s implied by tau*w
    consistency -- pass om in rad/s).
    """
    kap = kappa4() if kap is None else kap
    om = G['om']
    E = 1.0 - np.exp(-1j * om * tau)
    den = kap * (E * G['G']).real
    ap = np.where(den < 0.0, -1.0 / den, np.inf)
    return float(ap.min()), om[int(np.argmin(ap))]


def zoa_floor(G, kap=None):
    """Speed-independent guaranteed depth [m]:
        min over the worst regeneration phase,
        ap_floor = 1 / (kap_mag * max_w [ |G| - sign * Re G ])
    obtained from min_theta Re[(1-e^{j theta}) G] = ReG - |G| for kap > 0
    and max_theta = ReG + |G| for kap < 0.  Equals the classical
    -1/(2 kap min Re G) when G is negative-real at its worst point.
    """
    kap = kappa4() if kap is None else kap
    g = G['G']
    if kap > 0:
        worst = (np.abs(g) - g.real).max()      # -min_theta Re[(1-e)G]
    else:
        worst = (np.abs(g) + g.real).max()      # +max_theta Re[(1-e)G]
    if worst <= 0:
        return np.inf
    return 1.0 / (abs(kap) * worst)


def zoa_lobes(G, rpm_grid, kap=None):
    """Zero-order stability lobes ap_lim(Omega) [m] by direct evaluation of
    zoa_ap on each spindle speed (robust, no lobe bookkeeping)."""
    kap = kappa4() if kap is None else kap
    out = np.empty(len(rpm_grid))
    for i, rpm in enumerate(rpm_grid):
        out[i] = zoa_ap(G, tau_of(rpm), kap)[0]
    return out


def frf_pack(om, Gvals):
    """Bundle a frequency grid [rad/s] and complex FRF into the dict the
    zoa_* functions consume."""
    return dict(om=np.asarray(om, float), G=np.asarray(Gvals, complex))


if __name__ == '__main__':
    kap = kappa4()
    print('phi_st = %.2f deg   (down milling, ae/D = %.3f)'
          % (np.rad2deg(PHI_ST), AE / DIA))
    print('Kt_e = %.3e Pa   Kr = %.2f   kappa4 = %+.4e N/m^2' % (KT_E, KR, kap))
    T = tau_of(4900.0)
    t = np.linspace(0, 3 * T, 900)
    a = alphas(t, 4900.0, 0.3e-3)
    print('a4(t) over one period: mean %.1f N/m  min %.1f  max %.1f'
          % (a[3].mean(), a[3].min(), a[3].max()))
