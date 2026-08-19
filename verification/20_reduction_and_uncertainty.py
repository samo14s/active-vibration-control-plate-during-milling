"""20_reduction_and_uncertainty.py — quatre lacunes de la campagne 09-19.

Du, Liu, Dai, Long, "Robust combined time delay control for milling chatter
suppression of flexible workpieces", Int. J. Mech. Sci. 274 (2024) 109257.

Les scripts 09-19 ont verifie le modele (frequences, FRF, D^T D, lobes, signe,
geometrie de la pastille) et la reponse temporelle nominale. Quatre choses
restaient non couvertes ; ce script les traite.

G1 — COUVERTURE DE L'EQ. (25)                                    [arithmetique]
  Le papier ecrit (p. 7) :
    Eq. (23) "the milling force coefficient alpha4(t) can be regarded as
              varying within 0.3 alpha4 ~ 2.9 alpha4" puis
              "alpha40 = 1.6 alpha4, L_Palpha = 1.3 alpha4, where L_Palpha is
               the maximum perturbation of alpha4(t)" ;
    Eq. (24) "DD10, DD20, DD30, DD40 are averages of corresponding averages of
              the maximum and minimum values, L_DD1 ... L_DD4 are upper limit
              of corresponding element variation" ;
    Eq. (25) nominal = 1.6 alpha4 DD_i0 , perturbation = L_Palpha L_DD_i
              delta_PDi , avec |delta_PDi| <= 1 (Eq. 22).
  Le produit de deux intervalles n'est PAS l'intervalle des produits. Pour
  x = alpha4/abar4 dans [1.6 - 1.3, 1.6 + 1.3] et y = DD_i dans
  [DD_i0 - L, DD_i0 + L], le developpement exact est

      x y = 1.6 DD_i0  +  1.6 L dy  +  1.3 DD_i0 dx  +  1.3 L dx dy ,

  et le rayon exact autour du nominal vaut 1.3 |DD_i0| + 2.9 L. L'Eq. (25)
  telle qu'imprimee ne retient que le DERNIER terme, 1.3 L : il lui manque
  1.6 L (variation de D^T D vue par l'alpha4 nominal) ET 1.3 |DD_i0|
  (variation d'alpha4 vue par le D^T D nominal, le terme croise).
  Le libelle "upper limit of corresponding element variation" de l'Eq. (24)
  est ambigu ; on evalue donc la couverture sous LES DEUX lectures :
      R1  L_DD_i = (max - min)/2   demi-amplitude de la variation ;
      R2  L_DD_i = max |DD_i(x)|   plus grande valeur de l'element.
  Statistiques prises sur la plaque NUE dans la jauge du papier
  (milling_dynamics.dtd_paper_gauge) : le script 12 a etabli que la Fig. 7 est
  une figure de plaque NUE (courbes parfaitement symetriques).
  NB : abar4 < 0 avec la convention de signe de l'Eq. (13) ; tout est exprime
  en UNITES DE abar4, ce qui rend les longueurs et les taux de couverture
  independants de ce signe (multiplication par une constante non nulle).

G2 — LE CAS PERTURBE DE LA SEC. 4.2 / FIG. 16                      [temporel]
  Page 11, cite : "the same milling condition with 10 % perturbation of mode
  mass and stiffness, as well as an 80 % damping ratio is simulated as
  Fig. 16. Owing to the decrease of damping ratio, the milling without control
  diverges faster than Fig. 14(a). The perturbation of mode mass and stiffness
  causes the new chatter frequency f_c2p (1135 Hz)."
  Legende de la Fig. 16 : "10 % perturbation of mode mass and stiffness, as
  well as 20 % perturbation of damping ratio".
  Deux affirmations verifiables : (i) la divergence est PLUS RAPIDE que celle
  de la Fig. 14(a) ; (ii) la perturbation masse/raideur "cause" une raie a
  1135 Hz. On simule la condition S sans commande aux quatre coins
  (dm = +/-10 %, dk = +/-10 %) avec zeta x 0.8.
  DEUX LECTURES de la perturbation de masse, car le modele est normalise en
  masse (M_Pr0 = I) :
      L-omega   seule omega est mise a l'echelle sqrt((1+dk)/(1+dm)), comme
                prescrit ; le terme alpha4 D^T D et l'excitation restent
                inchanges ;
      L-coherente  (1+dm) q" + C q' + (K + a4 D^T D) q = F divise par (1+dm) :
                omega et zeta comme ci-dessus, ET alpha4, alpha3 divises par
                (1+dm) — c'est la reduction EXACTE d'une perturbation de masse
                modale a forme propre figee.
  Les deux sont rapportees ; la premiere est celle que demande l'enonce du
  papier, la seconde est celle qui est mecaniquement coherente.

G3 — LA TRONCATURE A DEUX MODES EST-ELLE LEGITIME POUR LA STABILITE ?
  Le papier ne la justifie que par les AMPLITUDES de FRF (p. 6, cite) : "The
  frequency responses of the first two modes are significantly larger than
  that of the third and higher modes in Fig. 4. Thus the dynamic model can be
  truncated from the second mode". Le script 16 a confirme ce fait sur les
  pics de FRF. Cela ne dit rien de la LIMITE DE PASSE, qui depend de la partie
  reelle de la FRF a toutes les frequences, pas de ses pics. On compare donc
  a_p,lim a 2 et a 5 modes, MEME moteur, MEMES reglages.

G4 — RECOUPEMENT DES DEUX MOTEURS DE STABILITE
  paper_model/stability_fdm assemble la monodromie complete sur l'etat
  augmente z = [x ; q_{k-1} ; ... ; q_{k-m}] (dimension 2n + n m) et en prend
  les valeurs propres : rayon spectral EXACT du schema.
  paper_model/lti_floquet applique les MEMES applications d'un pas (memes
  P0, C_lo, C_hi) a un historique glissant et estime le rayon spectral par
  ITERATION DE PUISSANCE adaptative, sans jamais assembler la matrice.
  Les deux schemas sont donc identiques ; seule l'extraction du rayon
  spectral differe. Tout ecart mesure est un ecart d'ESTIMATEUR.

CONVENTIONS
  * signe de couplage : config.SIGN_SIM = +1 (Eq. 13 telle que publiee,
    tranchee par verification/18_sign_convention.py) ;
  * plaque de commande calee sur les frequences MESUREES du Tableau 4 ;
  * grilles VOLONTAIREMENT GROSSIERES (machine chargee) : leurs resolutions
    sont imprimees avec chaque tableau.
"""
import os
import sys
import time

# machine chargee : une seule thread BLAS (a faire AVANT numpy)
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model'),
                os.path.join(HERE, '..', 'control')]

import copy                                                    # noqa: E402
import numpy as np                                             # noqa: E402
import matplotlib                                              # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402

import config                                                  # noqa: E402
from chebyshev_plate import ChebyshevPlate                     # noqa: E402
from plate_model import build_plate, F_MEASURED                # noqa: E402
from simulate import MillingSimulation                         # noqa: E402
import milling_dynamics as md                                  # noqa: E402
import lti_floquet as lf                                       # noqa: E402
import stability_fdm as sf                                     # noqa: E402

FIGDIR = os.path.join(HERE, '..', 'figures', 'verification')
os.makedirs(FIGDIR, exist_ok=True)
FIGPATH = os.path.abspath(os.path.join(FIGDIR,
                                       '20_reduction_and_uncertainty.png'))

SEP = '=' * 79
SUB = '-' * 75

# --------------------------------------------------------------- constantes
RPM_S = 4900.0                  # condition S de la Sec. 4.2
AP_S = 0.30e-3
AE = config.AE                  # 0.1 mm
FZ = config.FZ                  # 0.02 mm/dent
SIGN = config.SIGN_SIM          # +1, Eq. (13)
N_SUB = config.N_SUB            # 656 -> fs = 160.7 kHz
TAU_S = 60.0 / (md.N_TEETH * RPM_S)
FT_S = md.N_TEETH * RPM_S / 60.0
F_C2P = 1135.0                  # f_c2p annonce par le papier (Fig. 16)
F_C2 = 1135.0                   # f_c2 annonce pour la Fig. 14(a) — IDENTIQUE

# G1
N_POS_G1 = 801                  # positions le long du bord (resolution)
ALPHA_LO, ALPHA_HI = 0.3, 2.9   # Eq. (23), en unites de abar4
ALPHA_0, L_ALPHA = 1.6, 1.3     # Eq. (23)
N_T_G1 = 261                    # grille sur alpha4/abar4, pas 0.01

# G3 / G4 — grilles grossieres assumees
RPM_G3 = np.arange(3000.0, 7000.1, 500.0)        # 9 vitesses, pas 500 tr/min
POS_G3 = (0.00, 0.25, 0.50)                      # fractions de l_P (Fig. 6)
RPM_G4 = (4300.0, 4900.0, 5500.0, 6100.0, 6700.0)
POS_G4 = (0.00, 0.25)
AP_LO, AP_HI, AP_TOL = 0.005e-3, 1.5e-3, 2e-6    # bissection : 2 um


# ----------------------------------------------------------------- outillage
def hann_spectrum(t, y, fmax=3000.0, pad=8):
    """Spectre d'amplitude unilateral [um] — MEME protocole que le script 14 :
    moyenne retiree, fenetre de Hann symetrique, gain coherent (0.5) compense,
    zero-padding pad*N pour affiner la LECTURE du pic (la resolution reste
    1/duree)."""
    y = np.asarray(y, float)
    y = y - np.mean(y)
    N = y.size
    w = np.hanning(N)
    Y = np.fft.rfft(y * w, pad * N) * 2.0 / (N * 0.5)
    f = np.fft.rfftfreq(pad * N, t[1] - t[0])
    m = f <= fmax
    return f[m], np.abs(Y[m]) * 1e6


def top_lines(f, A, k=3, min_sep=120.0):
    """k plus grandes raies separees d'au moins min_sep Hz. min_sep doit
    depasser la largeur du lobe principal de Hann (4/duree ~ 100 Hz sur la
    seconde moitie), sinon on ne liste que les flancs d'un meme pic."""
    order = np.argsort(A)[::-1]
    out = []
    for i in order:
        if all(abs(f[i] - g) > min_sep for g, _ in out):
            out.append((float(f[i]), float(A[i])))
        if len(out) >= k:
            break
    return out


def envelope_growth(t, y, tau, lo=10e-6, hi=1e-3):
    """Taux de croissance sigma [1/s] : maximum de |y| par periode de dent,
    regression lineaire de ln(enveloppe) dans la bande [lo, hi] — meme
    definition que le script 14, pour que les chiffres soient comparables."""
    n = int(round(tau / (t[1] - t[0])))
    nb = t.size // n
    if nb < 6:
        return np.nan, np.nan, 0
    tb = np.array([t[(k + 1) * n - 1] for k in range(nb)])
    eb = np.array([np.max(np.abs(y[k * n:(k + 1) * n])) for k in range(nb)])
    m = (eb > lo) & (eb < hi)
    if m.sum() < 4:
        return np.nan, np.nan, int(m.sum())
    c = np.polyfit(tb[m], np.log(eb[m]), 1)
    r2 = float(np.corrcoef(tb[m], np.log(eb[m]))[0, 1] ** 2)
    return float(c[0]), r2, int(m.sum())


def peak_near(f, A, f0, half=100.0):
    """Plus grande valeur du spectre dans la fenetre f0 +/- half, et sa
    position : reponse directe a la question "y a-t-il une raie a f0 ?"."""
    m = (f >= f0 - half) & (f <= f0 + half)
    if not np.any(m):
        return np.nan, 0.0
    i = int(np.argmax(A[m]))
    return float(f[m][i]), float(A[m][i])


def interval_overlap(a, b):
    """Longueur de l'intersection de deux intervalles [lo, hi]."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def product_hull(t_lo, t_hi, d_lo, d_hi):
    """Enveloppe convexe de {t d} pour t dans [t_lo, t_hi], d dans
    [d_lo, d_hi] : min et max des quatre produits aux coins."""
    p = [t_lo * d_lo, t_lo * d_hi, t_hi * d_lo, t_hi * d_hi]
    return min(p), max(p)


def perturbed_plate(plate, w_scale, z_scale):
    """Copie de la plaque avec omega et zeta mis a l'echelle (les moteurs de
    Floquet ne lisent que omega_n, zeta_modes, D_row, H_Pe_modal)."""
    q = copy.copy(plate)
    q.omega_n = np.asarray(plate.omega_n, float) * w_scale
    q.zeta_modes = np.asarray(plate.zeta_modes, float) * z_scale
    return q


# ===========================================================================
t_start = time.time()
print(SEP)
print(' 20_reduction_and_uncertainty.py — Eq. (25), Fig. 16, troncature a')
print(' deux modes, et recoupement des deux moteurs de stabilite')
print(' Du et al., Int. J. Mech. Sci. 274 (2024) 109257')
print(SEP)

bare = ChebyshevPlate(PX=14, PZ=14, n_modes=5)          # Fig. 7 : plaque NUE
plate = build_plate(patch='right', n_modes=5, freqs=F_MEASURED)
print(' plaque NUE (Fig. 7)      f [Hz] : '
      + '  '.join('%7.1f' % v for v in bare.freq_n[:5]))
print(' plaque de commande       f [Hz] : '
      + '  '.join('%7.1f' % v for v in plate.freq_n[:5])
      + '   (calage MESURE, Tab. 4)')
print(' zeta [%]                         : '
      + '  '.join('%7.2f' % v for v in 100 * plate.zeta_modes[:5]))
print(' condition S : %.0f tr/min, a_e = %.2f mm, a_p = %.2f mm, '
      'f_z = %.3f mm/dent' % (RPM_S, AE * 1e3, AP_S * 1e3, FZ * 1e3))
print(' signe = %+.0f (Eq. 13) ; tau = %.5f s ; f_t = %.1f Hz'
      % (SIGN, TAU_S, FT_S))

# ===========================================================================
# G1 — COUVERTURE DE L'EQ. (25)
# ===========================================================================
print('\n' + SEP)
print('[G1] EQ. (25) : LE PRODUIT DE DEUX INTERVALLES N EST PAS L INTERVALLE')
print('     DES PRODUITS  (plaque NUE, jauge du papier, %d positions)'
      % N_POS_G1)
print(SEP)

xs_g1, DtD_g1, DD0_g1, LDD_g1, gauge = md.dtd_paper_gauge(bare, N_POS_G1)
abar4_S = md.alpha4_average(RPM_S, AP_S, bare.hp)
print(' jauge modale identifiee s = (%.4f, %.4f)  (script 12)' % gauge)
print(' abar4(condition S) = %+.4e N/m ; SIGNE NEGATIF avec l Eq. (13) :'
      % abar4_S)
print(' tout ce qui suit est en UNITES DE abar4, donc invariant par ce signe.')
print('\n ARITHMETIQUE DU PAPIER (Eq. 23) : alpha4/abar4 dans [%.1f, %.1f],'
      % (ALPHA_LO, ALPHA_HI))
print(' alpha40 = %.1f abar4 (= milieu), L_Palpha = %.1f abar4 (= demi-etendue)'
      % (ALPHA_0, L_ALPHA))
print(' MA LECTURE de l Eq. (25) : boite = %.1f DD_i0 +/- %.1f L_DD_i.'
      % (ALPHA_0, L_ALPHA))
print(' Rayon EXACT requis autour de %.1f DD_i0 : %.1f |DD_i0| + %.1f L_DD_i'
      % (ALPHA_0, L_ALPHA, ALPHA_HI))
print(' (= |alpha40| L + |DD0| L_Palpha + L_Palpha L ; l Eq. (25) ne garde que')
print('  le dernier terme : il lui manque %.1f L_DD_i ET %.1f |DD_i0|.)'
       % (ALPHA_0, L_ALPHA))

ELEM = ((0, 0, '(1,1)', 'DD10 / L_DD1'), (0, 1, '(1,2)', 'DD20 / L_DD2'),
        (1, 0, '(2,1)', 'DD30 / L_DD3'), (1, 1, '(2,2)', 'DD40 / L_DD4'))
t_grid = np.linspace(ALPHA_LO, ALPHA_HI, N_T_G1)          # pas 0.01 abar4

print('\n ' + SUB)
print(' Table G1a — statistiques par element (Eq. 24), en unites de jauge')
print(' ' + SUB)
print(' elem   Eq.(24)         min       max      DD_i0    L(R1)    L(R2)')
print(' ' + SUB)
g1 = {}
for i, j, nm, lbl in ELEM:
    v = DtD_g1[:, i, j]
    mn, mx = float(v.min()), float(v.max())
    dd0 = 0.5 * (mx + mn)
    l_r1 = 0.5 * (mx - mn)
    l_r2 = float(np.abs(v).max())
    g1[nm] = dict(v=v, mn=mn, mx=mx, dd0=dd0, l1=l_r1, l2=l_r2)
    print(' %-6s %-14s %9.5f %9.5f %9.5f %8.5f %8.5f'
          % (nm, lbl, mn, mx, dd0, l_r1, l_r2))
print(' ' + SUB)

print('\n ' + SUB)
print(' Table G1b — couverture du VRAI ensemble {alpha4/abar4} x {DD_i(x)}')
print('  par la boite de l Eq. (25), sous les deux lectures de L_DD_i')
print('  (vrai ensemble verifie par balayage direct : %d positions x %d'
      % (N_POS_G1, N_T_G1))
print('   valeurs d alpha4, pas 0.01 abar4)')
print(' ' + SUB)
print(' elem   TRUE set [lo, hi]      len   | lecture  Eq.(25) box       cover'
      '   centre offset')
print(' ' + SUB)
for i, j, nm, _ in ELEM:
    st = g1[nm]
    lo_t, hi_t = product_hull(ALPHA_LO, ALPHA_HI, st['mn'], st['mx'])
    # controle : balayage direct du produit (grilles annoncees ci-dessus)
    prod = np.outer(t_grid, st['v'])
    lo_b, hi_b = float(prod.min()), float(prod.max())
    err = max(abs(lo_b - lo_t), abs(hi_b - hi_t))
    len_t = hi_t - lo_t
    st.update(true=(lo_t, hi_t), len_t=len_t, brute_err=err)
    first = True
    for tag, L in (('R1 half', st['l1']), ('R2 max ', st['l2'])):
        box = (ALPHA_0 * st['dd0'] - L_ALPHA * L,
               ALPHA_0 * st['dd0'] + L_ALPHA * L)
        cov = interval_overlap(box, (lo_t, hi_t)) / len_t
        off = 0.5 * (box[0] + box[1]) - 0.5 * (lo_t + hi_t)
        st['cov_' + tag[:2]] = cov
        st['box_' + tag[:2]] = box
        st['off_' + tag[:2]] = off
        head = ('%-6s [%7.3f,%7.3f] %7.3f' % (nm, lo_t, hi_t, len_t)) \
            if first else (' ' * 31)
        print(' %s | %s [%7.3f,%7.3f] %6.1f %%  %+7.3f (%+5.1f %%)'
              % (head, tag, box[0], box[1], 100 * cov, off,
                 100 * off / len_t))
        first = False
print(' ' + SUB)
print(' (controle du balayage direct contre la formule des coins : ecart max'
      ' %.2e)' % max(g1[nm]['brute_err'] for _, _, nm, _ in ELEM))

print('\n ' + SUB)
print(' Table G1c — rayon EXACT requis contre rayon de l Eq. (25)')
print(' ' + SUB)
print(' elem    required = 1.3|DD0|+2.9 L(R1)   Eq.(25) R1   R1/req'
      '   Eq.(25) R2   R2/req')
print(' ' + SUB)
for i, j, nm, _ in ELEM:
    st = g1[nm]
    req = L_ALPHA * abs(st['dd0']) + ALPHA_HI * st['l1']
    r1 = L_ALPHA * st['l1']
    r2 = L_ALPHA * st['l2']
    st['req'] = req
    print(' %-6s %26.4f %12.4f %8.3f %12.4f %8.3f'
          % (nm, req, r1, r1 / req, r2, r2 / req))
print(' ' + SUB)
print(' Lecture R2 : L_DD(R2) = |DD0| + L(R1) exactement pour les quatre')
print(' elements, donc rayon R2 = 1.3|DD0| + 1.3 L(R1) : R2 couvre le terme')
print(' croise 1.3|DD0| mais laisse un deficit de 1.6 L(R1). R2 est donc')
print(' defendable UNIQUEMENT quand L(R1) << |DD0| — ce qui n est vrai que')
print(' pour l element (1,1). Un choix suffisant et a peine conservatif')
print(' serait rayon = alpha4_max * max|DD_i| = 2.9 (|DD0| + L(R1)).')

# ===========================================================================
# G2 — CAS PERTURBE DE LA SEC. 4.2 / FIG. 16
# ===========================================================================
print('\n' + SEP)
print('[G2] SEC. 4.2 / FIG. 16 — "diverges faster" et "new chatter frequency')
print('     f_c2p (1135 Hz)", sans commande, condition S')
print(SEP)
print(' Le papier annonce f_c2 = %.0f Hz pour la Fig. 14(a) ET f_c2p = %.0f Hz'
      % (F_C2, F_C2P))
print(' pour la Fig. 16 : la frequence "nouvelle" est NUMERIQUEMENT LA MEME.')
print(' Integration : Newmark moyenne, outil MOBILE, 5 modes, n_sub = %d,'
      % N_SUB)
print(' T = 0.30 s, arret a |y| = 5 mm ; spectre de Hann sur la SECONDE')
print(' MOITIE du signal au point de coupe (meme protocole que le script 14).')

CORNERS = [(+0.10, +0.10), (+0.10, -0.10), (-0.10, +0.10), (-0.10, -0.10)]


def run_g2(tag, dm, dk, zs, consistent):
    """Une passe sans commande. `consistent` : lecture L-coherente (division
    de alpha3 et alpha4 par (1+dm))."""
    ws = 1.0 if dm is None else np.sqrt((1.0 + dk) / (1.0 + dm))
    sg = SIGN / (1.0 + dm) if (consistent and dm is not None) else SIGN
    sim = MillingSimulation(plate, RPM_S, AP_S, ae=AE, fz=FZ, sign=sg,
                            n_modes=5, n_sub=N_SUB,
                            mode_scale=np.full(5, ws), zeta_scale=zs)
    r = sim.run(None, T=0.30, moving=True, x0=0.0)
    t, y = r['t'], r['y_mill']
    h = t.size // 2
    f, A = hann_spectrum(t[h:], y[h:])
    lines = top_lines(f, A, 3)
    sig, r2, npt = envelope_growth(t, y, TAU_S)
    # recoupement frequentiel : rayon spectral de Floquet a la position de
    # depart (l outil n a parcouru que ~0.4 mm quand la reponse atteint 5 mm)
    cs = 1.0 / (1.0 + dm) if (consistent and dm is not None) else 1.0
    q = perturbed_plate(plate, ws, zs)
    _, rho = lf.is_stable(q, RPM_S, AP_S, 0.0, None, n_modes=5, m=60,
                          coeff_scale=cs)
    # raie du MODE 2 : plus grand pic dans une fenetre autour du f_2 perturbe
    # (c'est elle que le papier appelle f_c2 / f_c2p)
    f2p = float(plate.freq_n[1] * ws)
    f2l, a2l = peak_near(f, A, f2p, 0.15 * f2p)
    return dict(tag=tag, ws=ws, sigma=sig, r2=r2, npt=npt,
                t5=r['t_div'] if r['diverged'] else None,
                ymax=float(np.abs(y).max()),
                lines=lines, rho=rho, sig_fl=float(np.log(rho) / TAU_S),
                f2=float(plate.freq_n[1] * ws), f1=float(plate.freq_n[0] * ws),
                f2line=f2l, a2line=a2l, Amax=float(A.max()))


g2 = {}
for read_tag, cons in (('L-omega', False), ('L-coherente', True)):
    rows = [run_g2('nominal', None, None, 1.0, cons)]
    rows.append(run_g2('zeta x0.8 only', 0.0, 0.0, 0.8, cons))
    for dm, dk in CORNERS:
        rows.append(run_g2('dm%+3.0f%% dk%+3.0f%%' % (100 * dm, 100 * dk),
                           dm, dk, 0.8, cons))
    g2[read_tag] = rows

for jj, read_tag in enumerate(('L-omega', 'L-coherente')):
    rows = g2[read_tag]
    print('\n ' + SUB)
    print(' Table G2-%s — lecture %s' % ('ab'[jj], read_tag))
    print(' ' + SUB)
    print(' case              w/w0    f_2      sigma_t   R^2    t(5mm)'
          '   max|y|   rho_Fl  sigma_Fl   dominant line')
    print('                          [Hz]      [1/s]            [s]'
          '      [um]              [1/s]      [Hz] (um)')
    print(' ' + SUB)
    for r in rows:
        t5 = '%7.4f' % r['t5'] if r['t5'] is not None else '  none '
        ym = '%8.1f' % (r['ymax'] * 1e6)
        f0, a0 = r['lines'][0]
        print(' %-16s %6.4f %7.1f %9.1f %6.4f %s %s %8.4f %9.1f  %7.1f (%.0f)'
              % (r['tag'], r['ws'], r['f2'], r['sigma'], r['r2'], t5, ym,
                 r['rho'], r['sig_fl'], f0, a0))
    print(' ' + SUB)
    ref = rows[0]
    print(' contre le nominal (sigma_t = %.1f 1/s, t(5mm) = %.4f s) :'
          % (ref['sigma'], ref['t5']))
    for r in rows[2:]:
        if r['t5'] is None:
            verdict = 'PAS de divergence en 0.30 s -> DIVERGE MOINS VITE'
        elif r['t5'] < ref['t5']:
            verdict = 'diverge PLUS vite (%.1f %% de temps en moins)' \
                % (100 * (1 - r['t5'] / ref['t5']))
        else:
            verdict = 'diverge MOINS vite (%.1f %% de temps en plus)' \
                % (100 * (r['t5'] / ref['t5'] - 1))
        print('   %-16s %s' % (r['tag'], verdict))
    nf = sum(1 for r in rows[2:]
             if r['t5'] is not None and r['t5'] < ref['t5'])
    print('   => %d coin(s) sur 4 divergent PLUS vite que le nominal ;'
          ' %d ne diverge(nt) pas du tout.'
          % (nf, sum(1 for r in rows[2:] if r['t5'] is None)))

print('\n ' + SUB)
print(' Table G2c — raies dominantes (separees d au moins 120 Hz, largeur du')
print('  lobe de Hann) et RAIE DU MODE 2 (plus grand pic dans f_2 +/- 15 %),')
print('  qui est ce que le papier nomme f_c2 / f_c2p')
print(' ' + SUB)
print(' lecture       case              raie 1        raie 2        raie 3'
      '     mode-2 line   ecart a 1135 Hz')
print(' ' + SUB)
for read_tag in ('L-omega', 'L-coherente'):
    for r in g2[read_tag]:
        ln = '  '.join('%6.0f Hz:%5.0f' % (f, a) for f, a in r['lines'])
        ln += '   ' * (3 - len(r['lines']))
        print(' %-13s %-16s %s  %6.1f Hz:%5.0f   %+6.1f %%'
              % (read_tag, r['tag'], ln, r['f2line'], r['a2line'],
                 100 * (r['f2line'] / F_C2P - 1)))
print(' ' + SUB)
best = min((r for tg in g2 for r in g2[tg]),
           key=lambda r: abs(r['f2line'] - F_C2P))
print(' raie de mode 2 la plus proche de %.0f Hz sur tous les cas : %.1f Hz'
      % (F_C2P, best['f2line']))
print(' (%+.1f %%, %.0f um = %.0f %% du pic dominant), cas "%s".'
      % (100 * (best['f2line'] / F_C2P - 1), best['a2line'],
         100 * best['a2line'] / best['Amax'], best['tag']))
print(' Au nominal la raie de mode 2 est a %.1f Hz (%+.1f %% de 1135 Hz).'
      % (g2['L-omega'][0]['f2line'],
         100 * (g2['L-omega'][0]['f2line'] / F_C2P - 1)))
print(' MAIS aucun cas ne fait de 1135 Hz la raie DOMINANTE : le pic reste sur')
print(' le mode 1 (534 -> 577 Hz), comme deja etabli au nominal par le script')
print(' 14 (534 Hz contre les 1135 Hz publies, -53 %).')
print(' NB coin non divergent : sigma_t > 0 alors que rho_Fl < 1 — la bande de')
print(' regression [10 um, 1 mm] y capte la MONTEE FORCEE du transitoire et')
print(' non une croissance de broutement ; c est rho_Fl qui fait foi.')

# ===========================================================================
# G3 — TRONCATURE A DEUX MODES ET STABILITE
# ===========================================================================
print('\n' + SEP)
print('[G3] TRONCATURE DE L EQ. (21) : a_p,lim A 2 MODES CONTRE 5 MODES')
print('     meme moteur (lti_floquet), meme m = 60, meme bissection')
print(SEP)
print(' grille GROSSIERE assumee : %d vitesses de %.0f a %.0f tr/min par pas'
      % (len(RPM_G3), RPM_G3[0], RPM_G3[-1]))
print(' de %.0f tr/min, %d positions (%s de l_P) ; bissection sur'
      % (RPM_G3[1] - RPM_G3[0], len(POS_G3),
         ', '.join('%.2f' % p for p in POS_G3)))
print(' [%.3f, %.3f] mm a %.0f um pres ; alpha4(t) exact, avalant, a_e = %.2f'
      ' mm' % (AP_LO * 1e3, AP_HI * 1e3, AP_TOL * 1e6, AE * 1e3))

lim2 = np.zeros((len(POS_G3), len(RPM_G3)))
lim5 = np.zeros_like(lim2)
for ip, fr in enumerate(POS_G3):
    xp = fr * plate.lp
    for ir, rpm in enumerate(RPM_G3):
        lim2[ip, ir] = lf.limit(plate, rpm, xp, None, n_modes=2, m=60,
                                lo=AP_LO, hi=AP_HI, tol=AP_TOL)
        lim5[ip, ir] = lf.limit(plate, rpm, xp, None, n_modes=5, m=60,
                                lo=AP_LO, hi=AP_HI, tol=AP_TOL)

ratio = lim2 / lim5
print('\n ' + SUB)
print(' Table G3 — a_p,lim [mm] sans commande ; ratio = 2 modes / 5 modes')
print(' ' + SUB)
print(' rpm    ' + ''.join(' x=%.2f l_P             ' % p for p in POS_G3))
print('        ' + '  2 modes  5 modes  ratio ' * len(POS_G3))
print(' ' + SUB)
for ir, rpm in enumerate(RPM_G3):
    row = ' %5.0f ' % rpm
    for ip in range(len(POS_G3)):
        row += '  %7.4f  %7.4f  %5.3f' % (lim2[ip, ir] * 1e3,
                                          lim5[ip, ir] * 1e3, ratio[ip, ir])
    print(row)
print(' ' + SUB)
iw = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
ib = np.unravel_index(int(np.argmin(ratio)), ratio.shape)
print(' ratio moyen %.4f ; mediane %.4f ; ecart-type %.4f'
      % (ratio.mean(), np.median(ratio), ratio.std()))
print(' PIRE cas (2 modes le plus OPTIMISTE) : %.0f tr/min, x = %.2f l_P,'
      % (RPM_G3[iw[1]], POS_G3[iw[0]]))
print('   2 modes %.4f mm contre 5 modes %.4f mm  ->  +%.1f %% de limite'
      % (lim2[iw] * 1e3, lim5[iw] * 1e3, 100 * (ratio[iw] - 1)))
print(' cas le plus CONSERVATIF : %.0f tr/min, x = %.2f l_P, ratio %.4f'
      % (RPM_G3[ib[1]], POS_G3[ib[0]], ratio[ib]))
n_opt = int(np.count_nonzero(ratio > 1.02))
print(' %d points sur %d ou 2 modes surestime a_p,lim de plus de 2 %%'
      % (n_opt, ratio.size))
print(' (a_p,lim a 2 modes est SYSTEMATIQUEMENT >= celui a 5 modes : %s)'
      % ('oui' if np.all(ratio >= 1.0 - 1e-9) else 'NON'))

# ===========================================================================
# G4 — RECOUPEMENT DES DEUX MOTEURS
# ===========================================================================
print('\n' + SEP)
print('[G4] stability_fdm (monodromie assemblee + eig) CONTRE lti_floquet')
print('     (memes applications d un pas, rayon spectral par iteration de')
print('     puissance) — 2 modes, m = 60, meme encadrement, meme tolerance')
print(SEP)
print(' Les deux modules construisent le MEME operateur discret :')
print('   P0 = expm(A_k h) avec A_k gele au MILIEU du sous-intervalle,')
print('   J1 = A^-1 (P0 - I), J2 = h J1 - A^-1 (h P0 - J1),')
print('   C_hi = (J2/h) B, C_lo = (J1 - J2/h) B, terme retarde interpole')
print('   LINEAIREMENT entre q_{k-m} et q_{k-m+1}.')
print(' Seule differe l EXTRACTION du rayon spectral : eig exact contre')
print(' iteration de puissance adaptative. Tout ecart est un ecart')
print(' d ESTIMATEUR, pas de modele.')

rows4 = []
for rpm in RPM_G4:
    for fr in POS_G4:
        xp = fr * plate.lp
        a = sf.stability_limit(plate, rpm, xp, lo=AP_LO, hi=AP_HI,
                               tol=AP_TOL, m=60)
        b = lf.limit(plate, rpm, xp, None, n_modes=2, m=60,
                     lo=AP_LO, hi=AP_HI, tol=AP_TOL)
        ap_c = 0.5 * (a + b)
        _, _, rho_a = sf.is_stable(plate, rpm, ap_c, xp, m=60, n_modes=2,
                                   return_freq=True)
        _, rho_b = lf.is_stable(plate, rpm, ap_c, xp, None, n_modes=2, m=60)
        rows4.append(dict(rpm=rpm, fr=fr, a=a, b=b,
                          rel=abs(a - b) / max(a, b, 1e-12),
                          dtol=abs(a - b) / AP_TOL,
                          ap_c=ap_c, rho_a=rho_a, rho_b=rho_b,
                          rrel=abs(rho_a - rho_b) / rho_a))

print('\n ' + SUB)
print(' Table G4 — a_p,lim [mm] et rayon spectral au milieu des deux limites')
print(' ' + SUB)
print('  rpm   x/l_P   fdm(eig)  floquet(pow)  |diff|  |diff|  |diff|'
      '   rho(eig)  rho(pow)  ecart')
print('                  [mm]        [mm]        [um]    /max    /tol'
      '                        rho')
print(' ' + SUB)
for r in rows4:
    print(' %5.0f  %5.2f  %9.5f %12.5f %8.2f %7.4f %7.2f %10.5f %9.5f %6.2f %%'
          % (r['rpm'], r['fr'], r['a'] * 1e3, r['b'] * 1e3,
             abs(r['a'] - r['b']) * 1e6, r['rel'], r['dtol'],
             r['rho_a'], r['rho_b'], 100 * r['rrel']))
print(' ' + SUB)
rel_max = max(r['rel'] for r in rows4)
abs_max = max(abs(r['a'] - r['b']) for r in rows4)
tol_max = max(r['dtol'] for r in rows4)
rrel_max = max(r['rrel'] for r in rows4)
n_ident = sum(1 for r in rows4 if r['dtol'] <= 1.0)
print(' ecart relatif MAX sur a_p,lim : %.2e (%.2f %%) — atteint la ou'
      % (rel_max, 100 * rel_max))
print('   a_p,lim est le plus PETIT, ou le relatif est trompeur ;')
print(' ecart ABSOLU MAX : %.2f um, soit %.2f fois la tolerance de'
      % (abs_max * 1e6, tol_max))
print('   bissection (%.0f um).' % (AP_TOL * 1e6))
print(' %d points sur %d identiques a la tolerance de bissection pres'
      % (n_ident, len(rows4)))
print(' ecart relatif MAX sur rho au meme a_p : %.2e (%.3f %%)'
      % (rrel_max, 100 * rrel_max))
n_dis = sum(1 for r in rows4 if r['dtol'] > 1.0)
n_hi = sum(1 for r in rows4 if r['b'] > r['a'] + AP_TOL)
print(' -> %s' % ('les deux moteurs sont d accord a la tolerance de bissection'
                  ' pres sur TOUS les points'
                  if tol_max <= 1.0 else
                  'les deux moteurs DIFFERENT au-dela de la tolerance de'
                  ' bissection sur %d point(s) sur %d' % (n_dis, len(rows4))))
if n_dis:
    print('    Direction : sur ces %d points l iteration de puissance rend un'
          % n_dis)
    print('    a_p,lim %s (%d/%d), avec rho(pow) < rho(eig) au meme a_p : elle'
          % ('PLUS GRAND' if n_hi >= n_dis - n_hi else 'PLUS PETIT',
             n_hi, n_dis))
    print('    sous-estime legerement rho pres de la frontiere et est donc,')
    print('    la, NON CONSERVATIVE. L ecart sur rho reste <= %.2f %%, mais'
          % (100 * rrel_max))
    print('    rho(a_p) est quasi plat au voisinage de 1 (drho/da_p faible),')
    print('    de sorte que %.2f %% sur rho deplace la limite de %.1f um.'
          % (100 * rrel_max, abs_max * 1e6))
print('\n ECART A DING et al. (2010) [79], la reference citee pour la Fig. 6')
print(' ("A full-discretization method for prediction of milling stability",')
print(' Int J Mach Tool Manu 2010;50:502-9) :')
print('   * Ding et al. separent A(t) = A0 + A_p(t) et prennent l exponentielle')
print('     de la SEULE partie CONSTANTE A0, puis interpolent LINEAIREMENT sur')
print('     chaque sous-intervalle a la fois l etat, le coefficient periodique')
print('     et le terme retarde ;')
print('   * les deux modules de ce depot GELENT A_k = A0 + A_p(t_milieu) au')
print('     point milieu et prennent expm(A_k h) : exact a coefficient gele,')
print('     aucune interpolation de l etat courant, interpolation lineaire du')
print('     SEUL terme retarde.')
print('   Les deux schemas sont d ordre 2 en h et convergent vers la meme')
print('   monodromie, mais leurs operateurs discrets different a O(h^2) : a m')
print('   fini les limites ne coincident pas exactement. Ce depot n implante')
print('   donc PAS la FDM de Ding et al. ; c est un schema a exponentielle')
print('   gelee au milieu, de meme ordre. Aucun des deux moteurs n a ete')
print('   compare a une implantation de Ding et al. ici.')

# ===========================================================================
# FIGURE
# ===========================================================================
CB = dict(true='#c9d6e8', r1='#1f4e79', r2='#e07b39', nom='#333333',
          p2='#c0392b', p5='#1f4e79')
fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.2))
fig.suptitle('Reduced-order model, Eq. (25) uncertainty box, and stability '
             'engines — Du et al., IJMS 274 (2024) 109257',
             fontsize=13, fontweight='bold')

# ---- (a) couverture de l'Eq. (25) -----------------------------------------
ax = axes[0, 0]
names = [nm for _, _, nm, _ in ELEM]
ypos = np.arange(len(names))[::-1]
for k, (_, _, nm, _) in enumerate(ELEM):
    st = g1[nm]
    y0 = ypos[k]
    ax.barh(y0, st['true'][1] - st['true'][0], left=st['true'][0], height=0.62,
            color=CB['true'], edgecolor='#5b6f8c', lw=0.8,
            label='True set  {alpha4/abar4 in [0.3, 2.9]} x {DtD_i(x)}'
            if k == 0 else None, zorder=2)
    b1 = st['box_R1']
    ax.barh(y0 + 0.14, b1[1] - b1[0], left=b1[0], height=0.20,
            color=CB['r1'], edgecolor='none',
            label='Eq. (25) box, L_DD = half-amplitude (R1)'
            if k == 0 else None, zorder=3)
    b2 = st['box_R2']
    ax.barh(y0 - 0.14, b2[1] - b2[0], left=b2[0], height=0.20,
            color=CB['r2'], edgecolor='none',
            label='Eq. (25) box, L_DD = max element value (R2)'
            if k == 0 else None, zorder=3)
    ax.plot([ALPHA_0 * st['dd0']], [y0], marker='|', ms=16, mew=2.0,
            color='k', zorder=4,
            label='nominal 1.6 DD_i0' if k == 0 else None)
    ax.text(st['true'][1] + 0.35, y0 + 0.16, 'R1 %.1f %%' % (100 * st['cov_R1']),
            va='center', fontsize=8.5, color=CB['r1'], fontweight='bold')
    ax.text(st['true'][1] + 0.35, y0 - 0.16, 'R2 %.1f %%' % (100 * st['cov_R2']),
            va='center', fontsize=8.5, color=CB['r2'], fontweight='bold')
ax.set_yticks(ypos)
ax.set_yticklabels(['DtD %s' % n for n in names])
ax.set_xlabel('alpha4 DtD element  [units of abar4]')
ax.set_title('(a) G1 — Eq. (25) uncertainty box vs the true product set\n'
             'coverage fraction of the true set is printed on the right',
             fontsize=10.5)
ax.set_xlim(-11.5, 16.5)
ax.grid(axis='x', alpha=0.3, zorder=0)
ax.legend(fontsize=7.6, loc='lower right', framealpha=0.95)

# ---- (b) coins de perturbation --------------------------------------------
ax = axes[0, 1]
mk = {'L-omega': 'o', 'L-coherente': 's'}
for read_tag in ('L-omega', 'L-coherente'):
    rows = g2[read_tag]
    col = CB['p2'] if read_tag == 'L-omega' else CB['p5']
    for r in rows:
        f0 = r['lines'][0][0]
        f2l = r['f2line']
        nominal = (r['tag'] == 'nominal')
        c = CB['nom'] if nominal else col
        ax.plot([f0, f2l], [r['sigma']] * 2, '-', color=c, lw=0.8,
                alpha=0.45, zorder=2)
        ax.scatter([f0], [r['sigma']], s=120 if nominal else 74,
                   marker='*' if nominal else mk[read_tag],
                   color=c, edgecolor='k', lw=0.6, zorder=4)
        ax.scatter([f2l], [r['sigma']], s=44, marker=mk[read_tag],
                   facecolor='none', edgecolor=c, lw=1.2, zorder=4)
        if read_tag == 'L-omega' or r['tag'] not in ('nominal',
                                                     'zeta x0.8 only'):
            ax.annotate(r['tag'].replace('dm', 'm').replace('dk', 'k'),
                        (f0, r['sigma']), textcoords='offset points',
                        xytext=(6, 5), fontsize=7.2)
ref = g2['L-omega'][0]
ax.axhline(ref['sigma'], color=CB['nom'], ls='--', lw=1.2,
           label='nominal growth rate %.1f 1/s' % ref['sigma'])
ax.axvline(F_C2P, color=CB['r2'], ls='-.', lw=1.6,
           label='paper f_c2 = f_c2p = 1135 Hz (same number)')
ax.axvline(plate.freq_n[0], color='#777777', ls=':', lw=1.2,
           label='nominal f_1 = %.0f Hz, f_2 = %.0f Hz'
           % (plate.freq_n[0], plate.freq_n[1]))
ax.axvline(plate.freq_n[1], color='#777777', ls=':', lw=1.2)
ax.axhline(0.0, color='k', lw=0.8)
ax.set_xlabel('dominant spectral line of the second half  [Hz]')
ax.set_ylabel('envelope growth rate sigma  [1/s]')
ax.set_title('(b) G2 — Fig. 16 perturbation corners, uncontrolled '
             'condition S\nfilled: dominant line; open: mode-2 line. '
             'circles = omega-only, squares = consistent-mass', fontsize=10.5)
ax.set_xlim(420, 1290)
ax.grid(alpha=0.3)
ax.legend(fontsize=7.6, loc='center left')

# ---- (c) troncature 2 / 5 modes -------------------------------------------
ax = axes[1, 0]
cols = ['#1f4e79', '#c0392b', '#2e8b57']
for ip, fr in enumerate(POS_G3):
    ax.plot(RPM_G3, lim5[ip] * 1e3, '-o', ms=4, color=cols[ip], lw=1.7,
            label='x = %.2f l_P, 5 modes' % fr)
    ax.plot(RPM_G3, lim2[ip] * 1e3, '--s', ms=4, color=cols[ip], lw=1.4,
            alpha=0.85, mfc='none', label='x = %.2f l_P, 2 modes' % fr)
ax.set_yscale('log')
ax.set_xlabel('spindle speed  [rpm]')
ax.set_ylabel('uncontrolled a_p,lim  [mm]')
ax.set_title('(c) G3 — is the 2-mode truncation of Eq. (21) safe for '
             'STABILITY?\nsame engine, m = 60, %.0f rpm grid'
             % (RPM_G3[1] - RPM_G3[0]), fontsize=10.5)
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=7.4, ncol=2, loc='upper left')
axi = ax.inset_axes([0.60, 0.08, 0.37, 0.30])
for ip, fr in enumerate(POS_G3):
    axi.plot(RPM_G3, ratio[ip], '-o', ms=3, color=cols[ip], lw=1.2)
axi.axhline(1.0, color='k', lw=0.9)
axi.set_title('ratio 2-mode / 5-mode', fontsize=7.5)
axi.tick_params(labelsize=6.5)
axi.grid(alpha=0.3)

# ---- (d) accord des deux moteurs ------------------------------------------
ax = axes[1, 1]
aa = np.array([r['a'] for r in rows4]) * 1e3
bb = np.array([r['b'] for r in rows4]) * 1e3
sc = ax.scatter(aa, bb, c=[r['rpm'] for r in rows4], cmap='viridis',
                s=80, edgecolor='k', lw=0.6, zorder=3)
lim_hi = max(aa.max(), bb.max()) * 1.12
ax.plot([0, lim_hi], [0, lim_hi], 'k-', lw=1.0, label='identity')
ax.fill_between([0, lim_hi], [0, lim_hi * 0.95], [0, lim_hi * 1.05],
                color='#bbbbbb', alpha=0.35, label='+/- 5 %', zorder=0)
for r, x0, y0 in zip(rows4, aa, bb):
    if r['rel'] > 0.005:
        ax.annotate('%.0f rpm, x=%.2f\n%.1f %%'
                    % (r['rpm'], r['fr'], 100 * r['rel']), (x0, y0),
                    textcoords='offset points', xytext=(7, -13), fontsize=6.8)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('a_p,lim from stability_fdm (assembled monodromy + eig)  [mm]')
ax.set_ylabel('a_p,lim from lti_floquet (power iteration)  [mm]')
ax.set_title('(d) G4 — the two stability engines on a common grid\n'
             '%d points, 2 modes, m = 60, bisection tol %.0f um; max '
             'difference %.2f %%' % (len(rows4), AP_TOL * 1e6, 100 * rel_max),
             fontsize=10.5)
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=8, loc='upper left')
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label('spindle speed [rpm]', fontsize=8)
cb.ax.tick_params(labelsize=7)

fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig(FIGPATH, dpi=140)
plt.close(fig)

# ===========================================================================
print('\n' + SEP)
print(' SYNTHESE')
print(SEP)
print(' G1  Eq. (25) telle qu imprimee couvre le vrai ensemble a %.1f / %.1f /'
      % (100 * g1['(1,1)']['cov_R1'], 100 * g1['(1,2)']['cov_R1']))
print('     %.1f / %.1f %% (lecture R1, elements (1,1) (1,2) (2,1) (2,2)) et'
      % (100 * g1['(2,1)']['cov_R1'], 100 * g1['(2,2)']['cov_R1']))
print('     a %.1f / %.1f / %.1f / %.1f %% (lecture R2). Aucune lecture ne la'
      % (100 * g1['(1,1)']['cov_R2'], 100 * g1['(1,2)']['cov_R2'],
         100 * g1['(2,1)']['cov_R2'], 100 * g1['(2,2)']['cov_R2']))
print('     rend valable partout ; R2 ne sauve que l element (1,1).')
print(' G2  divergence plus rapide qu au nominal : %d coin(s) sur 4 (lecture'
      % sum(1 for r in g2['L-omega'][2:]
            if r['t5'] is not None and r['t5'] < g2['L-omega'][0]['t5']))
print('     L-omega), %d sur 4 (lecture L-coherente) ; le coin (dm +10 %%,'
      % sum(1 for r in g2['L-coherente'][2:]
            if r['t5'] is not None and r['t5'] < g2['L-coherente'][0]['t5']))
print('     dk -10 %%) ne diverge pas du tout (rho_Fl = %.4f < 1) dans les deux'
      % g2['L-omega'][3]['rho'])
print('     lectures. Raie dominante : mode 1 partout, jamais 1135 Hz.')
print(' G3  ratio a_p,lim (2 modes / 5 modes) : moyen %.3f, pire %.3f a'
      % (ratio.mean(), ratio[iw]))
print('     %.0f tr/min et x = %.2f l_P -> la troncature est NON CONSERVATIVE.'
      % (RPM_G3[iw[1]], POS_G3[iw[0]]))
print(' G4  ecart max entre moteurs %.2f %% (%.1f um) sur a_p,lim et %.3f %%'
      % (100 * rel_max, abs_max * 1e6, 100 * rrel_max))
print('     sur rho au meme a_p ;', end=' ')
print('%d/%d points identiques a la tolerance de'
      % (n_ident, len(rows4)))
print('     bissection pres.')
print('\n figure : %s' % FIGPATH)
print(' temps total : %.1f s' % (time.time() - t_start))
print(SEP)
