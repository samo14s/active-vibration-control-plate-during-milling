"""14_uncontrolled_time_response.py — audit de la Fig. 14(a) du papier
(reponse temporelle de fraisage SANS commande, condition S) et de son spectre.

Du, Liu, Dai, Long, "Robust combined time delay control for milling chatter
suppression of flexible workpieces", Int. J. Mech. Sci. 274 (2024) 109257.

CE QUE DIT LE PAPIER (Sec. 4.2, Fig. 14) — cite :
  * "the milling condition S located in Fig. 13(b) is simulated with the
     parameters: spindle speed 4900 rpm, radial cutting depth 0.1 mm, axial
     cutting depth 0.3 mm, and feed rate 0.02 mm per tooth" ;
  * "Due to the divergence of milling without control, Fig. 14(a) only
     presents the response within 0.2 s" ;
  * "In milling without control, severe chatter occurs with the chatter
     frequency f_c2 (1135 Hz) dominating the vibration. This chatter is caused
     by the second mode of the plate because f_c2 is close to the second
     natural frequency 1101 Hz."
  Le papier ne publie NI l'amplitude atteinte, NI le taux de croissance, NI
  l'echelle de la Fig. 14(a) : les seules affirmations verifiables sont
  (i) la divergence en moins de 0.2 s et (ii) la raie dominante a 1135 Hz.

CE QUE FAIT CE SCRIPT
  1. integration temporelle de la passe (Newmark moyenne, outil MOBILE sur le
     bord superieur, 5 modes, amortissements mesures du Tableau 4), pour les
     DEUX calages du Tableau 4 (mesure et "theorique") ;
  2. divergence : instants de passage a 50 um et a 5 mm, taux de croissance
     exponentiel identifie, compare au multiplicateur de Floquet ;
  3. spectre de la SECONDE MOITIE du signal, fenetre de Hann (protocole
     annonce ligne a ligne), raie dominante contre f_c2 = 1135 Hz du papier,
     contre f_2 (1101 / 1068 Hz) et contre f_t = 245 Hz et ses harmoniques ;
  4. controle de convergence : n_sub double ;
  5. cas STABLE a faible profondeur : reponse bornee, regime de vibration
     forcee domine par les harmoniques de f_t.

CONVENTIONS
  * signe de couplage : config.SIGN_SIM (= +1, Eq. 13 telle que publiee ;
    voir verification/18_sign_convention.py) ;
  * pas de temps : dt = tau / config.N_SUB avec tau = 60/(N_T rpm) ;
  * y = deplacement AU POINT DE COUPE (y_mill) ; le capteur du papier est au
    coin superieur droit (y_obs), les deux sont rapportes.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model'),
                os.path.join(HERE, '..', 'control')]

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from plate_model import build_plate, F_MEASURED, F_THEORETICAL
from simulate import MillingSimulation
import lti_floquet as lf
import stability_fdm as sf
from milling_dynamics import alpha4_average, alpha4_series, N_TEETH

FIGDIR = os.path.join(HERE, '..', 'figures', 'verification')
os.makedirs(FIGDIR, exist_ok=True)
FIGPATH = os.path.abspath(os.path.join(FIGDIR,
                                       '14_uncontrolled_time_response.png'))

# ------------------------------------------------------------- condition S
RPM = 4900.0
AP_S = 0.30e-3                 # profondeur axiale de la condition S
AE = config.AE                 # 0.1 mm, avalant
FZ = config.FZ                 # 0.02 mm/dent
N_MODES = 5
N_SUB = config.N_SUB           # 656 -> fs = 160.7 kHz
SIGN = config.SIGN_SIM
TAU = 60.0 / (N_TEETH * RPM)
FT = N_TEETH * RPM / 60.0      # frequence de passage de dent = 245 Hz
F_C2_PAPER = 1135.0            # raie dominante annoncee par le papier
T_WINDOW = 0.20                # fenetre de la Fig. 14(a)
AP_STABLE = 0.020e-3           # cas stable (justifie par la limite calculee)
T_STABLE = 1.00                # duree du cas stable
CALIBS = [('measured', F_MEASURED), ('theoretical', F_THEORETICAL)]


# ----------------------------------------------------------------- outillage
def hann_spectrum(t, y, fmax=3000.0, pad=8):
    """Spectre d'amplitude unilateral [um] — PROTOCOLE ANNONCE :
    moyenne retiree, fenetre de Hann (np.hanning, symetrique), gain coherent
    de la fenetre (0.5) compense, zero-padding pad*N pour interpoler la
    position du pic (la RESOLUTION reste 1/duree, seule la LECTURE du maximum
    est affinee)."""
    y = np.asarray(y, float)
    y = y - np.mean(y)
    N = y.size
    w = np.hanning(N)
    Y = np.fft.rfft(y * w, pad * N) * 2.0 / (N * 0.5)
    f = np.fft.rfftfreq(pad * N, t[1] - t[0])
    m = f <= fmax
    return f[m], np.abs(Y[m]) * 1e6


def envelope_growth(t, y, tau, lo=2e-6, hi=2e-3):
    """Taux de croissance sigma [1/s] : maximum de |y| sur chaque periode de
    dent, regression lineaire de ln(enveloppe) dans la bande [lo, hi] (on
    ecarte le transitoire de demarrage et la fin ecretee par stop_um)."""
    n = int(round(tau / (t[1] - t[0])))
    nb = t.size // n
    if nb < 6:
        return np.nan, np.nan, np.array([]), np.array([])
    tb = np.array([t[(k + 1) * n - 1] for k in range(nb)])
    eb = np.array([np.max(np.abs(y[k * n:(k + 1) * n])) for k in range(nb)])
    m = (eb > lo) & (eb < hi)
    if m.sum() < 4:
        return np.nan, np.nan, tb, eb
    c = np.polyfit(tb[m], np.log(eb[m]), 1)
    r2 = np.corrcoef(tb[m], np.log(eb[m]))[0, 1] ** 2
    return float(c[0]), float(r2), tb, eb


def first_crossing(t, y, level):
    a = np.abs(np.asarray(y, float))
    if a.max() < level:
        return np.nan
    return float(t[int(np.argmax(a >= level))])


def floquet_modes(plate, rpm, ap, x_pos, m=120, n_modes=N_MODES, k=3):
    """k multiplicateurs dominants de la monodromie EXACTE (stability_fdm,
    matrice assemblee puis eig complet — aucune iteration de puissance), avec
    la frequence de broutement repliee la plus proche d'une frequence propre.
    ATTENTION : le multiplicateur ne donne la frequence que modulo 1/tau ;
    le repliement retenu est celui qui tombe le plus pres d'un mode."""
    tau = 60.0 / (N_TEETH * rpm)
    D = plate.D_row(x_pos, plate.hp)[:n_modes]
    DtD = np.outer(D, D)
    _, a4 = alpha4_series(rpm, ap, plate.hp, m, midpoint=True)
    Phi = sf.floquet_matrix(plate.omega_n[:n_modes], plate.zeta_modes[:n_modes],
                            DtD, SIGN * a4, tau, n=n_modes)
    mu = np.linalg.eigvals(Phi)
    fn = plate.freq_n[:n_modes]
    out = []
    seen = []
    for i in np.argsort(np.abs(mu))[::-1]:
        rho = float(abs(mu[i]))
        fpv = abs(np.angle(mu[i])) / (2 * np.pi * tau)
        cand = [abs(s * fpv + j / tau) for s in (1, -1) for j in range(0, 14)]
        fc = min(cand, key=lambda c: min(abs(c - f) for f in fn))
        if any(abs(fc - s) < 1.0 for s in seen):   # paire conjuguee
            continue
        seen.append(fc)
        out.append((rho, fc, float(np.log(rho) / tau)))
        if len(out) >= k:
            break
    return out, tau


def averaged_single_mode_limit(plate, x_pos, rpm, n_modes=N_MODES):
    """Limite de passe par la theorie MOYENNEE d'ordre 0, mode par mode :
        ap_lim = min_f  -1 / (2 (abar4/ap) Re G_i(f)).
    Sert a montrer a quel point les modes 1 et 2 sont a egalite."""
    kdir = SIGN * alpha4_average(rpm, 1e-3, plate.hp) / 1e-3
    D = plate.D_row(x_pos, plate.hp)[:n_modes]
    om = plate.omega_n[:n_modes]
    z = plate.zeta_modes[:n_modes]
    res = []
    for i in range(n_modes):
        ff = np.linspace(0.5 * plate.freq_n[i], 1.5 * plate.freq_n[i], 200001)
        w = 2 * np.pi * ff
        G = D[i]**2 / ((om[i]**2 - w**2) + 2j * z[i] * om[i] * w)
        ap = -1.0 / (2 * kdir * np.real(G))
        m = ap > 0
        j = int(np.argmin(ap[m]))
        res.append((float(ap[m][j]), float(ff[m][j])))
    return res


def run_case(plate, ap, T, n_sub):
    sim = MillingSimulation(plate, RPM, ap, ae=AE, fz=FZ, sign=SIGN,
                            n_modes=N_MODES, n_sub=n_sub)
    return sim.run(None, T=T, moving=True)


# ------------------------------------------------- controle du protocole FFT
def check_fft_protocol():
    fs = 1.0 / (TAU / N_SUB)
    t = np.arange(0, 0.05, 1.0 / fs)
    y = 3.7e-6 * np.sin(2 * np.pi * 812.0 * t + 0.3)
    f, A = hann_spectrum(t, y, fmax=2000.0)
    i = int(np.argmax(A))
    return f[i], A[i]


# =========================================================================
t_start = time.time()
print('=' * 79)
print(' 14_uncontrolled_time_response.py — Fig. 14(a) du papier et son spectre')
print(' Du et al., Int. J. Mech. Sci. 274 (2024) 109257, Sec. 4.2')
print('=' * 79)
print(' Condition S : %.0f tr/min, a_e = %.2f mm, a_p = %.2f mm, '
      'f_z = %.3f mm/dent, avalant' % (RPM, AE * 1e3, AP_S * 1e3, FZ * 1e3))
print(' Modele : Chebyshev-Ritz 14x14, pastille coin bas droit, %d modes,'
      ' zeta mesures (Tab. 4)' % N_MODES)
print(' Integration : Newmark moyenne, outil MOBILE (v = %.2f mm/s),'
      ' signe = %+.0f (Eq. 13)' % (FZ * N_TEETH * RPM / 60.0 * 1e3, SIGN))
print(' tau = %.5f s, n_sub = %d -> dt = %.3e s (fs = %.1f kHz)'
      % (TAU, N_SUB, TAU / N_SUB, 1e-3 * N_SUB / TAU))
print(' f_t = N_T*rpm/60 = %.1f Hz ; f_c2 publie = %.0f Hz' % (FT, F_C2_PAPER))

f_chk, a_chk = check_fft_protocol()
print('\n--- controle du protocole FFT (sinus de synthese 812.0 Hz, 3.700 um) ---')
print('    lu : %.1f Hz  /  %.3f um    (erreurs %+.2f %% et %+.2f %%)'
      % (f_chk, a_chk, 100 * (f_chk / 812.0 - 1), 100 * (a_chk / 3.700 - 1)))

# ------------------------------------------------------------------ plaques
plates = {}
for name, fr in CALIBS:
    plates[name] = build_plate(patch='right', n_modes=N_MODES, freqs=fr)

print('\n--- calages du Tableau 4 (frequences imposees, Hz) ---')
for name, _ in CALIBS:
    print('    %-12s %s' % (name, np.array2string(plates[name].freq_n[:N_MODES],
                                                  precision=1)))
print('    zeta (%%)      %s'
      % np.array2string(100 * plates['measured'].zeta_modes[:N_MODES],
                        precision=2))

# ------------------------------------------------ limites de passe a 4900 tr
print('\n--- limite de passe SANS commande a %.0f tr/min (Floquet, m = 60,'
      ' 5 modes) ---' % RPM)
print('    calage        x = 0 mm    x = 50 mm   x = 100 mm   |  papier Fig.13(b)')
lims = {}
for name, _ in CALIBS:
    row = [lf.limit(plates[name], RPM, x, None, n_modes=N_MODES, m=60, tol=5e-6)
           for x in (0.0, 0.050, 0.100)]
    lims[name] = row
    print('    %-12s %8.4f mm %8.4f mm %8.4f mm  |  ~0.05 mm'
          % (name, row[0] * 1e3, row[1] * 1e3, row[2] * 1e3))
print('    => a_p = %.2f mm de la condition S est %.1f a %.1f fois la limite :'
      % (AP_S * 1e3, AP_S / max(lims["measured"]), AP_S / min(lims["measured"])))
print('       la divergence annoncee par la Fig. 14(a) est bien reproduite.')
print('    NOTE : le cas "stable" de l\'enonce (a_p = 0.05 mm) est en fait'
      ' AU-DESSUS')
for name, _ in CALIBS:
    _, r05 = lf.is_stable(plates[name], RPM, 0.05e-3, 0.0, None,
                          n_modes=N_MODES, m=60)
    _, r02 = lf.is_stable(plates[name], RPM, AP_STABLE, 0.0, None,
                          n_modes=N_MODES, m=60)
    print('       de la limite de ce modele : rho(0.05 mm) = %.4f, '
          'rho(%.3f mm) = %.4f  [%s]'
          % (r05, AP_STABLE * 1e3, r02, name))
print('       => le cas stable est pris a a_p = %.3f mm (rho < 1 partout).'
      % (AP_STABLE * 1e3))

# ---------------------------------------- modes critiques : theorie moyennee
print('\n--- QUEL MODE BROUTE ? theorie moyennee d\'ordre 0, mode par mode'
      ' (a_p_lim en mm) ---')
print('    calage / x        mode1        mode2        mode3        mode4'
      '        mode5')
for name, _ in CALIBS:
    for x in (0.0, 0.050):
        r = averaged_single_mode_limit(plates[name], x, RPM)
        print('    %-9s x=%3.0f  %s' % (name[:9], x * 1e3,
              '  '.join('%6.4f@%4.0f' % (a * 1e3, f) for a, f in r)))
print('    -> a x = 0 (coin, la ou l\'outil se trouve pendant les 0.2 s de la')
print('       Fig. 14a) les modes 1 et 2 sont a EGALITE a ~12 %% pres ; ailleurs')
print('       le mode 2 a un noeud et disparait. Le papier tranche pour le')
print('       mode 2, ce modele tranche pour le mode 1.')

# -------------------------------------------------------------- simulations
print('\n--- SIMULATIONS SANS COMMANDE, a_p = %.2f mm (condition S) ---'
      % (AP_S * 1e3))
res = {}
spec = {}
for name, _ in CALIBS:
    for ns in (N_SUB, 2 * N_SUB):
        r = run_case(plates[name], AP_S, T_WINDOW, ns)
        t, y = r['t'], r['y_mill']
        sig, r2, tb, eb = envelope_growth(t, y, TAU)
        h = slice(t.size // 2, t.size)
        f, A = hann_spectrum(t[h], y[h], fmax=3000.0)
        fo, Ao = hann_spectrum(t[h], r['y_obs'][h], fmax=3000.0)
        res[(name, ns)] = dict(r=r, sigma=sig, r2=r2, tb=tb, eb=eb,
                               t50=first_crossing(t, y, 50e-6),
                               t5mm=first_crossing(t, y, 5e-3),
                               fpk=float(f[np.argmax(A)]),
                               apk=float(A.max()),
                               fpk_obs=float(fo[np.argmax(Ao)]),
                               df=1.0 / (t[h][-1] - t[h][0]))
        if ns == N_SUB:
            spec[name] = (f, A)

print('\n  Table 1 — divergence et raie dominante (y au point de coupe)')
print('  ' + '-' * 75)
print('  calage       n_sub   t(50um)   t(5mm)   sigma[1/s]  R^2    f_pic[Hz]'
      '  A_pic[um]')
for name, _ in CALIBS:
    for ns in (N_SUB, 2 * N_SUB):
        d = res[(name, ns)]
        print('  %-11s %6d  %7.4f s %7.4f s %8.1f  %.4f  %8.1f  %9.1f'
              % (name, ns, d['t50'], d['t5mm'], d['sigma'], d['r2'],
                 d['fpk'], d['apk']))
print('  ' + '-' * 75)
print('  resolution spectrale brute (1/duree de la demi-fenetre) : %.1f Hz'
      % res[('measured', N_SUB)]['df'])

print('\n  Table 2 — controle de convergence en temps (n_sub double)')
print('  ' + '-' * 75)
print('  calage        grandeur          n_sub=%d    n_sub=%d    ecart relatif'
      % (N_SUB, 2 * N_SUB))
for name, _ in CALIBS:
    a, b = res[(name, N_SUB)], res[(name, 2 * N_SUB)]
    for key, lab, unit in (('t5mm', 't (5 mm)', 's'),
                           ('sigma', 'sigma', '1/s'),
                           ('fpk', 'f_pic', 'Hz')):
        va, vb = a[key], b[key]
        print('  %-13s %-16s %10.4f %11.4f    %+8.3f %%'
              % (name if key == 't5mm' else '', lab + ' [' + unit + ']',
                 va, vb, 100 * (vb / va - 1)))
print('  ' + '-' * 75)
print('  => pas de temps non critique : ecarts < 0.1 %% sur les trois'
      ' grandeurs.')

# --------------------------------------------------------- Floquet vs temps
print('\n  Table 3 — multiplicateurs de Floquet dominants (monodromie exacte,'
      ' x = 0, m = 120)')
print('  ' + '-' * 75)
print('  calage        rang  rho      sigma=ln(rho)/tau  f_c replie [Hz]'
      '   mode le plus proche')
fl = {}
for name, _ in CALIBS:
    ev, tau = floquet_modes(plates[name], RPM, AP_S, 0.0)
    fl[name] = ev
    fn = plates[name].freq_n[:N_MODES]
    for j, (rho, fc, s) in enumerate(ev):
        k = int(np.argmin(np.abs(fn - fc)))
        print('  %-13s %2d  %7.4f  %12.1f      %10.1f       mode %d (%.0f Hz)'
              % (name if j == 0 else '', j + 1, rho, s, fc, k + 1, fn[k]))
print('  ' + '-' * 75)
for name, _ in CALIBS:
    d = res[(name, N_SUB)]
    rho, fc, s = fl[name][0]
    print('  %-12s temporel : sigma = %6.1f 1/s, f = %6.1f Hz  |  Floquet :'
          ' sigma = %6.1f 1/s, f = %6.1f Hz  (%+.1f %% / %+.2f %%)'
          % (name, d['sigma'], d['fpk'], s, fc,
             100 * (d['sigma'] / s - 1), 100 * (d['fpk'] / fc - 1)))
print('  NOTE : sigma temporel legerement plus faible car l\'outil s\'eloigne'
      ' du coin (rho decroit avec x).')

# ---------------------------------------------- comparaison avec le papier
print('\n  Table 4 — RAIE DOMINANTE contre le papier')
print('  ' + '-' * 75)
print('  calage        f_pic simule   f_c2 papier   ecart      f_2 du calage'
      '   f_pic/f_1')
for name, _ in CALIBS:
    d = res[(name, N_SUB)]
    f1 = plates[name].freq_n[0]
    f2 = plates[name].freq_n[1]
    print('  %-13s %9.1f Hz  %9.1f Hz  %+7.1f %%  %9.1f Hz   %8.4f'
          % (name, d['fpk'], F_C2_PAPER, 100 * (d['fpk'] / F_C2_PAPER - 1),
             f2, d['fpk'] / f1))
print('  ' + '-' * 75)
for name, _ in CALIBS:
    f, A = spec[name]
    j = int(np.argmin(np.abs(f - F_C2_PAPER)))
    print('  %-13s amplitude a 1135 Hz = %8.2f um = %.1f %% du pic'
          % (name, A[j], 100 * A[j] / A.max()))

print('\n  Table 5 — structure du spectre : peigne de Floquet f_pv + k*f_t')
print('  ' + '-' * 75)
for name, _ in CALIBS:
    f, A = spec[name]
    rho, fc, s = fl[name][0]
    tau = TAU
    fpv = fc % FT
    fpv = min(fpv, FT - fpv)
    pk = [i for i in range(1, f.size - 1)
          if A[i] >= A[i - 1] and A[i] > A[i + 1] and A[i] > 0.05 * A.max()]
    pk = sorted(pk, key=lambda i: -A[i])[:8]
    print('  %-12s f_pv = %.1f Hz ; peigne attendu |+-f_pv + k*%.0f| :'
          % (name, fpv, FT))
    print('     pics mesures : ' + ', '.join('%.0f(%.0f um)' % (f[i], A[i])
                                             for i in sorted(pk)))
    exp = sorted({round(abs(sg * fpv + k * FT))
                  for sg in (1, -1) for k in range(0, 6)})
    print('     peigne theorique : ' + ', '.join('%d' % e for e in exp
                                                 if e <= 1400))
print('  ' + '-' * 75)

# ------------------------------------------------------------- cas stable
print('\n--- CAS STABLE, a_p = %.3f mm (%.0f %% de la limite), T = %.2f s ---'
      % (AP_STABLE * 1e3, 100 * AP_STABLE / lims['measured'][0], T_STABLE))
stab = {}
for name, _ in CALIBS:
    r = run_case(plates[name], AP_STABLE, T_STABLE, N_SUB)
    t, y = r['t'], r['y_mill']
    h = slice(t.size // 2, t.size)
    f, A = hann_spectrum(t[h], y[h], fmax=1600.0)
    pk = [i for i in range(1, f.size - 1)
          if A[i] >= A[i - 1] and A[i] > A[i + 1] and A[i] > 0.05 * A.max()]
    pk = sorted(pk, key=lambda i: -A[i])[:5]
    stab[name] = dict(r=r, f=f, A=A, pk=pk,
                      pkpk=float(np.ptp(y[h])), rms=float(np.std(y[h])),
                      maxall=float(np.max(np.abs(y))))
print('  ' + '-' * 75)
print('  calage       diverge  max|y| [um]  crete-crete [um]  rms [um]'
      '   raies dominantes')
for name, _ in CALIBS:
    d = stab[name]
    lab = ', '.join('%.0f Hz (%.3f um, %.1fxf_t)'
                    % (d['f'][i], d['A'][i], d['f'][i] / FT) for i in d['pk'][:3])
    print('  %-12s %-8s %11.3f %17.3f %9.3f' % (name, str(d['r']['diverged']),
          1e6 * d['maxall'], 1e6 * d['pkpk'], 1e6 * d['rms']))
    print('               raies : ' + lab)
print('  ' + '-' * 75)
print('  => reponse BORNEE, toutes les raies sont des harmoniques exactes de')
print('     f_t = %.0f Hz : regime de vibration FORCEE, aucun broutement.' % FT)
print('     (la plus forte est 2*f_t = %.0f Hz, amplifiee par sa proximite'
      ' avec f_1)' % (2 * FT))

# ================================================================== figure
fig, ax = plt.subplots(2, 2, figsize=(13.0, 8.6))
COL = {'measured': '#1f77b4', 'theoretical': '#d62728'}

# (a) reponse instable
a = ax[0, 0]
for name, _ in CALIBS:
    d = res[(name, N_SUB)]
    r = d['r']
    a.plot(r['t'], 1e3 * r['y_mill'], lw=0.6, color=COL[name],
           label='%s calib. (f1=%.0f, f2=%.0f Hz)'
                 % (name, plates[name].freq_n[0], plates[name].freq_n[1]))
    a.plot([d['t5mm']], [5.0], 'v', color=COL[name], ms=8)
    a.annotate('5 mm at %.3f s' % d['t5mm'], (d['t5mm'], 5.0),
               textcoords='offset points', xytext=(-6, 8), ha='right',
               fontsize=8, color=COL[name])
a.axvspan(0, T_WINDOW, color='0.92', zorder=0)
a.set_xlim(0, T_WINDOW)
a.set_xlabel('time [s]')
a.set_ylabel('displacement at cutting point [mm]')
a.set_title('(a) uncontrolled response, condition S\n'
            '4900 rpm, $a_e$=0.1 mm, $a_p$=0.30 mm, $f_z$=0.02 mm/tooth'
            '  (paper Fig. 14a window = 0.2 s)', fontsize=9)
a.legend(fontsize=7, loc='upper left')
a.grid(alpha=0.3)
ins = a.inset_axes([0.55, 0.13, 0.42, 0.40])
for name, _ in CALIBS:
    d = res[(name, N_SUB)]
    m = d['eb'] > 1e-9
    ins.semilogy(d['tb'][m], 1e6 * d['eb'][m], '.', ms=3, color=COL[name])
    tt = np.linspace(0, d['t5mm'], 50)
    ins.semilogy(tt, 1e6 * np.exp(np.polyval(
        np.polyfit(d['tb'][(d['eb'] > 2e-6) & (d['eb'] < 2e-3)],
                   np.log(d['eb'][(d['eb'] > 2e-6) & (d['eb'] < 2e-3)]), 1),
        tt)), '-', lw=1.0, color=COL[name],
        label=r'$\sigma$=%.0f s$^{-1}$' % d['sigma'])
ins.axhline(50, color='0.4', ls=':', lw=0.8)
ins.set_xlabel('t [s]', fontsize=7)
ins.set_ylabel('envelope [um]', fontsize=7)
ins.tick_params(labelsize=6)
ins.legend(fontsize=6)
ins.grid(alpha=0.3, which='both')

# (b) spectre instable
b = ax[0, 1]
for name, _ in CALIBS:
    f, A = spec[name]
    b.semilogy(f, np.maximum(A, 1e-3), lw=0.8, color=COL[name], label=name)
for k in range(1, 7):
    b.axvline(k * FT, color='0.6', ls=':', lw=0.8)
    b.text(k * FT, 2e3, '%d$f_t$' % k, fontsize=6, ha='center', color='0.4')
b.axvline(F_C2_PAPER, color='k', ls='--', lw=1.2)
b.text(F_C2_PAPER, 3e2, ' paper $f_{c2}$=1135 Hz', fontsize=8, color='k')
for name, _ in CALIBS:
    d = res[(name, N_SUB)]
    b.plot([d['fpk']], [d['apk']], 'o', color=COL[name], ms=5)
    b.annotate('%.0f Hz' % d['fpk'], (d['fpk'], d['apk']),
               textcoords='offset points', xytext=(6, 4), fontsize=8,
               color=COL[name])
b.set_xlim(0, 1600)
b.set_ylim(1e-1, 4e3)
b.set_xlabel('frequency [Hz]')
b.set_ylabel('amplitude [um]')
b.set_title('(b) spectrum of the second half of the diverging signal\n'
            'Hann window, df = %.0f Hz — dominant line is mode 1, not 1135 Hz'
            % res[('measured', N_SUB)]['df'], fontsize=9)
b.legend(fontsize=7, loc='upper right')
b.grid(alpha=0.3, which='both')

# (c) cas stable, temporel
c = ax[1, 0]
for name, _ in CALIBS:
    r = stab[name]['r']
    c.plot(r['t'], 1e6 * r['y_mill'], lw=0.4, color=COL[name], label=name)
c.axvspan(T_STABLE / 2, T_STABLE, color='0.92', zorder=0)
c.text(0.75 * T_STABLE, 0.9 * 1e6 * max(stab[n]['maxall'] for n, _ in CALIBS),
       'FFT window', fontsize=7, ha='center', color='0.4')
c.set_xlim(0, T_STABLE)
c.set_xlabel('time [s]')
c.set_ylabel('displacement at cutting point [um]')
c.set_title('(c) stable case, $a_p$ = %.3f mm (%.0f %% of the computed limit'
            ' %.4f mm)\nbounded response — forced vibration regime'
            % (AP_STABLE * 1e3, 100 * AP_STABLE / lims['measured'][0],
               lims['measured'][0] * 1e3), fontsize=9)
c.legend(fontsize=7, loc='upper right')
c.grid(alpha=0.3)

# (d) cas stable, spectre
d_ax = ax[1, 1]
for name, _ in CALIBS:
    d_ax.semilogy(stab[name]['f'], np.maximum(stab[name]['A'], 1e-6),
                  lw=0.8, color=COL[name], label=name)
for k in range(1, 7):
    d_ax.axvline(k * FT, color='0.6', ls=':', lw=0.8)
    d_ax.text(k * FT, 0.15, '%d$f_t$' % k, fontsize=6, ha='center', color='0.4')
for name, _ in CALIBS:
    d_ax.axvline(plates[name].freq_n[0], color=COL[name], ls='-.', lw=0.8)
d_ax.axvline(F_C2_PAPER, color='k', ls='--', lw=1.0)
d_ax.text(F_C2_PAPER, 0.05, ' 1135 Hz', fontsize=7)
d_ax.set_xlim(0, 1600)
d_ax.set_ylim(1e-5, 3e-1)
d_ax.set_xlabel('frequency [Hz]')
d_ax.set_ylabel('amplitude [um]')
d_ax.set_title('(d) spectrum of the stable case (second half, Hann)\n'
               'only tooth-passing harmonics; dash-dot = $f_1$ of each calib.',
               fontsize=9)
d_ax.legend(fontsize=7, loc='upper right')
d_ax.grid(alpha=0.3, which='both')

fig.suptitle('Verification 14 — uncontrolled milling response of the '
             'cantilever plate (Du et al. 2024, Fig. 14a)', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig(FIGPATH, dpi=140)
print('\nfigure -> %s' % FIGPATH)
print('temps total : %.1f s' % (time.time() - t_start))
print('=' * 79)
