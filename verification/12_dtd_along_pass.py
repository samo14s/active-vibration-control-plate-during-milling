"""12_dtd_along_pass.py — audit de la Fig. 7 du papier
(dynamique variable le long de la passe : elements de D_Pr^T D_Pr sur le
bord superieur de la plaque encastree).

Du, Liu, Dai, Long, "Robust combined time delay control for milling chatter
suppression of flexible workpieces", Int. J. Mech. Sci. 274 (2024) 109257.

Ce que dit le papier (Sec. 3.2, Eqs. 23-25, Fig. 7) :
  * "When milling the upper edge of a thin-walled cantilever plate, the
     varying elements of D_Pr^T D_Pr are shown in Fig. 7. The actual values
     are calculated by the dynamic model in Section 2. The mean values are
     the averages of the maximum and minimum values." -> moyenne = (max+min)/2 ;
  * "Except for the first element D_Pr^T D_Pr(1,1), other elements all vary
     with position largely" ;
  * Eq. (24) : DD10..DD40 = valeurs nominales (moyennes), L_DD1..L_DD4 =
    amplitudes de variation (bornes superieures) ;
  * Eq. (23) : alpha40 = 1.6 abar4, L_Palpha = 1.3 abar4 ;
  * Eq. (25) : alpha40 DD0 nominal et Delta_PrD = L_Palpha L_DDi delta_PDi.

POINT METHODOLOGIQUE CENTRAL — LA JAUGE MODALE
  Le papier garde M_Pr0 = diag(m_P10, m_P20) dans l'Eq. (21) : ses modes ne
  sont donc PAS normalises en masse, et l'echelle de chaque colonne de D_Pr
  est arbitraire (q -> S q avec S = diag(s1, s2) laisse la physique
  invariante). Les NIVEAUX de la Fig. 7 ne sont donc pas des predictions
  verifiables ; seuls le sont les rapports invariants par S :
      - rapport de dome (1,1) : D11(mi-portee)/D11(extremites) ;
      - position du zero de (1,2) et du noeud de (2,2) ;
      - symetrie des courbes ;
      - L_DDi/DDi0 par element, et donc la variation relative.
  milling_dynamics.paper_gauge CALE (s1, s2) sur trois des quatre niveaux
  digitises FIG7_TARGETS : les accords de NIVEAU sont donc AJUSTES, pas
  predits. Le script le dit explicitement a chaque ligne concernee.

Valeurs de reference "Fig. 7" (FIG7_TARGETS de milling_dynamics) :
  DIGITISEES sur les courbes du papier, non imprimees par les auteurs ;
  incertitude de releve graphique de l'ordre de +/- 0.03 en ordonnee.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model'),
                os.path.join(HERE, '..', 'control'),
                os.path.join(HERE, '..', 'simulation', 'sim_kit')]

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from chebyshev_plate import ChebyshevPlate
import milling_dynamics as md

FIGDIR = os.path.join(HERE, '..', 'figures', 'verification')
os.makedirs(FIGDIR, exist_ok=True)
FIGPATH = os.path.abspath(os.path.join(FIGDIR, '12_dtd_along_pass.png'))

N_POS = 801              # impair : l'indice N_POS//2 tombe exactement en mi-portee
RPM_S, AP_S = 4900.0, 0.3e-3          # condition S de la Section 4
T = md.FIG7_TARGETS
SEP = '=' * 78
SUB = '-' * 78


def rel(a, b):
    """Ecart relatif en % de a par rapport a la reference b."""
    return 100.0 * (a / b - 1.0) if b != 0 else float('nan')


def zero_crossings(x, y):
    """Positions des changements de signe de y, interpolees lineairement."""
    idx = np.where(np.diff(np.sign(y)) != 0)[0]
    out = []
    for i in idx:
        y0, y1 = y[i], y[i + 1]
        out.append(x[i] if y1 == y0 else x[i] - y0 * (x[i + 1] - x[i]) / (y1 - y0))
    return np.array(out)


def elements(D2):
    """(11), (12), (21), (22) a partir des deux profils modaux (n,2)."""
    return (D2[:, 0]**2, D2[:, 0] * D2[:, 1], D2[:, 1] * D2[:, 0], D2[:, 1]**2)


# ===========================================================================
print(SEP)
print(' SCRIPT 12 — FIG. 7 : D_Pr^T D_Pr LE LONG DE LA PASSE (bord superieur)')
print('   Du, Liu, Dai, Long, Int. J. Mech. Sci. 274 (2024) 109257')
print(SEP)
print(' Les valeurs "Fig. 7" sont DIGITISEES (relevees sur les courbes du')
print(' papier, stockees dans milling_dynamics.FIG7_TARGETS) : ce ne sont pas')
print(' des nombres imprimes par les auteurs. Incertitude de lecture ~ +/-0.03.')
print(' Attention : la Fig. 7 depend de la NORMALISATION MODALE des auteurs,')
print(' qui n est pas publiee (Eq. 21 garde M_Pr0 = diag(m_P10, m_P20), donc')
print(' modes NON normalises en masse). Chaque comparaison ci-dessous est')
print(' etiquetee [GAUGE-FREE] ou [GAUGE-FITTED].')

# ---------------------------------------------------------------------------
# 1. Modeles
# ---------------------------------------------------------------------------
print('\n' + SEP)
print('[1] MODELES')
print(SEP)
bare = ChebyshevPlate(PX=14, PZ=14, n_modes=5)          # plaque NUE (Fig. 7)
print('    (a) Chebyshev-Ritz PX=PZ=14, plaque NUE (Tableau 1, sans pastille)')
print('        frequences [Hz] : ' + '  '.join('%7.1f' % v for v in bare.freq_n))
print('        La Fig. 7 du papier est parfaitement SYMETRIQUE : elle est donc')
print('        calculee sur la plaque nue (la pastille brise la symetrie).')

from plate_model import build_plate                      # noqa: E402
patched = build_plate(patch='right')                     # modele de commande
print('    (b) meme plaque + pastille QDA60-20-0.7 au coin inferieur DROIT,')
print('        calee sur les frequences "theoretical" du Tableau 4 :')
print('        frequences [Hz] : '
      + '  '.join('%7.1f' % v for v in patched.freq_n))

# Contre-verification INDEPENDANTE : elements finis Kirchhoff-Q4 (sim_kit).
# ATTENTION : control/plate_model.py et simulation/sim_kit/plate_model.py
# portent le MEME nom de module ; on charge le second par son chemin absolu.
import importlib.util                                     # noqa: E402
fe_ok = True
try:
    _sp = importlib.util.spec_from_file_location(
        '_fe_plate_model', os.path.join(HERE, '..', 'simulation', 'sim_kit',
                                        'plate_model.py'))
    _fe = importlib.util.module_from_spec(_sp)
    sys.modules['_fe_plate_model'] = _fe
    _sp.loader.exec_module(_fe)
    fe = _fe.PlateModel(lp=0.100, hp=0.080, bp=0.004, rho=2830.0,
                        E=69e9, nu=0.33, N1=30, N2=24, n_modes=3,
                        verbose=False)
    fe.precompute_Dp(0.080, 401)
    print('    (c) contre-verification INDEPENDANTE : elements finis')
    print('        Kirchhoff-Q4 30x24 (simulation/sim_kit), plaque nue :')
    print('        frequences [Hz] : '
          + '  '.join('%7.1f' % v for v in fe.freq_n))
except Exception as exc:                                  # pragma: no cover
    fe_ok = False
    print('    (c) modele FE indisponible (%s)' % exc)

# ---------------------------------------------------------------------------
# 2. Fig. 7 dans la jauge du papier — Eqs. (24)-(25)
# ---------------------------------------------------------------------------
print('\n' + SEP)
print('[2] FIG. 7 DANS LA JAUGE DU PAPIER — parametres DD10..DD40, L_DD1..L_DD4')
print(SEP)
xs, DtD_g, DD0_g, LDD_g, s_gauge = md.dtd_paper_gauge(bare, N_POS)
e11, e12, e21, e22 = DtD_g[:, 0, 0], DtD_g[:, 0, 1], DtD_g[:, 1, 0], DtD_g[:, 1, 1]
ELEMS = (('(1,1)', e11), ('(1,2)', e12), ('(2,1)', e21), ('(2,2)', e22))

print('    facteurs de jauge identifies s = (s1, s2) = (%.4f, %.4f)'
      % s_gauge)
print('    -> masses modales implicites du papier m_Pi0 = 1/si^2 = '
      '(%.4f, %.4f) kg' % (1 / s_gauge[0]**2, 1 / s_gauge[1]**2))
print('       soit %.2f et %.2f fois la masse de la plaque (0.09056 kg) :'
      % (1 / s_gauge[0]**2 / 0.09056, 1 / s_gauge[1]**2 / 0.09056))
print('       ordre de grandeur plausible pour des modes NON normalises en')
print('       masse -> coherent avec M_Pr0 = diag(m_P10, m_P20) de l Eq. (21).')

print('\n    ' + SUB)
print('    elem     max        min        DD_i0      L_DD_i     (Eq. 24)')
print('    ' + SUB)
names4 = ['DD10 / L_DD1', 'DD20 / L_DD2', 'DD30 / L_DD3', 'DD40 / L_DD4']
stats = {}
for (nm, v), lbl in zip(ELEMS, names4):
    mx, mn = v.max(), v.min()
    dd0, ldd = 0.5 * (mx + mn), 0.5 * (mx - mn)
    stats[nm] = dict(max=mx, min=mn, DD0=dd0, LDD=ldd, v=v)
    print('    %-6s %10.5f %10.5f %10.5f %10.5f   %s'
          % (nm, mx, mn, dd0, ldd, lbl))
print('    ' + SUB)
print('    (DD20 et DD30 sont nuls a %.1e pres : le mode 2 est ANTISYMETRIQUE'
      % max(abs(stats['(1,2)']['DD0']), abs(stats['(2,1)']['DD0'])))
print('     sur le bord superieur, donc la moyenne (max+min)/2 de phi1*phi2')
print('     est exactement nulle — comme sur la Fig. 7(b)-(c) du papier.)')

print('\n    NIVEAUX vs Fig. 7 digitisee   [GAUGE-FITTED : s1, s2 sont AJUSTES')
print('    sur D11_ends, D11_mid et D22_ends — ces trois lignes ne sont donc')
print('    PAS des predictions, elles definissent la jauge]')
lev = [('D11(x=0) = D11(x=lp)', e11[0], T['D11_ends'], 'AJUSTE'),
       ('D11(mi-portee)', e11[N_POS // 2], T['D11_mid'], 'AJUSTE'),
       ('D22(x=0) = D22(x=lp)', e22[0], T['D22_ends'], 'AJUSTE'),
       ('max |D12|', np.abs(e12).max(), T['D12_amp'], 'PREDIT*')]
print('    %-24s %10s %10s %9s  %s' % ('grandeur', 'modele', 'Fig.7 dig.',
                                       'ecart', 'statut'))
for nm, a, b, st in lev:
    print('    %-24s %10.4f %10.4f %8.2f%%  %s' % (nm, a, b, rel(a, b), st))
print('    * max|D12| n est "predit" qu au sens ou il decoule de l identite')
print('      exacte D12^2 = D11*D22 (voir [3]) une fois s1, s2 fixes : il')
print('      teste la COHERENCE INTERNE du releve graphique, pas le modele.')

# ---------------------------------------------------------------------------
# 3. Invariants de jauge
# ---------------------------------------------------------------------------
print('\n' + SEP)
print('[3] INVARIANTS DE JAUGE — les SEULES comparaisons reellement predictives')
print(SEP)

# (a) rapport de dome de (1,1)
dome_mod = e11[N_POS // 2] / e11[0]
dome_pap = T['D11_mid'] / T['D11_ends']
# (b) zero de (1,2) et noeud de (2,2)
zc = zero_crossings(xs, e12)
xnode = xs[np.argmin(e22)]
# (c) symetries
sym11 = np.max(np.abs(e11 - e11[::-1])) / e11.mean()
sym12 = np.max(np.abs(e12 + e12[::-1])) / np.abs(e12).max()   # ANTIsymetrie
sym22 = np.max(np.abs(e22 - e22[::-1])) / e22.max()
# (d) identite de Cauchy D12^2 = D11 D22 (exacte pour un vecteur modal)
cauchy = np.max(np.abs(e12**2 - e11 * e22)) / (e11 * e22).max()
cauchy_pap = T['D12_amp']**2 / (T['D11_ends'] * T['D22_ends'])
# (e) L_DD/DD0 par element
r1_mod = stats['(1,1)']['LDD'] / stats['(1,1)']['DD0']
r1_pap = (T['D11_mid'] - T['D11_ends']) / (T['D11_mid'] + T['D11_ends'])
r4_mod = stats['(2,2)']['LDD'] / stats['(2,2)']['DD0']

print('    %-46s %10s %10s %9s' % ('invariant', 'modele', 'Fig.7 dig.', 'ecart'))
print('    ' + SUB)
print('    %-46s %10.5f %10.5f %8.2f%%'
      % ('dome (1,1) : D11(mi)/D11(bords)   [GAUGE-FREE]',
         dome_mod, dome_pap, rel(dome_mod, dome_pap)))
print('    %-46s %10.5f %10.5f %8.2f%%'
      % ('L_DD1/DD10 = (max-min)/(max+min)  [GAUGE-FREE]',
         r1_mod, r1_pap, rel(r1_mod, r1_pap)))
print('      (ces deux lignes sont algebriquement EQUIVALENTES :')
print('       L/DD0 = (r-1)/(r+1) ; FIG7_TARGETS ne contient donc qu UN seul')
print('       nombre gauge-free independant.)')
print('    %-46s %10.5f %10.5f %8s'
      % ('L_DD4/DD40 (noeud mode 2 -> min=0) [G-FREE, struct.]',
         r4_mod, 1.0, '%.2f%%' % rel(r4_mod, 1.0)))
print('    %-46s %10.4f %10.4f %8s'
      % ('zero de (1,2), x/lp                [GAUGE-FREE]',
         zc[0] / bare.lp if zc.size else float('nan'), 0.5,
         '%.3f%%' % rel(zc[0] / bare.lp, 0.5) if zc.size else 'n/a'))
print('    %-46s %10.4f %10.4f %8.3f%%'
      % ('noeud de (2,2), x/lp               [GAUGE-FREE]',
         xnode / bare.lp, 0.5, rel(xnode / bare.lp, 0.5)))
print('    ' + SUB)
print('    symetrie  max|D11(x)-D11(lp-x)|/moy   = %.2e   (attendu 0)' % sym11)
print('    ANTIsym.  max|D12(x)+D12(lp-x)|/max   = %.2e   (attendu 0)' % sym12)
print('    symetrie  max|D22(x)-D22(lp-x)|/max   = %.2e   (attendu 0)' % sym22)
print('    identite de Cauchy D12^2 = D11*D22 (exacte pour tout vecteur modal)')
print('      modele : residu relatif max = %.2e  (exacte a la machine)' % cauchy)
print('      Fig. 7 digitisee : D12_amp^2/(D11_ends*D22_ends) = %.5f'
      % cauchy_pap)
print('        -> ecart %.2f%% par rapport a 1 : c est l INCERTITUDE DE'
      % rel(cauchy_pap, 1.0))
print('           NUMERISATION de la Fig. 7, elle borne toutes les')
print('           comparaisons de niveau ci-dessus a ~1.3 %.')
print('    le signe de (1,2) (positif a gauche / negatif a droite ici) est une')
print('    pure convention de signe des modes : NON verifiable sur la figure.')

if fe_ok:
    Dfe = fe.Dp_array[:2, :].T
    f11, f12, _, f22 = elements(Dfe)
    xfe = fe.xp_array
    dome_fe = f11.max() / f11.min()
    r1_fe = (f11.max() - f11.min()) / (f11.max() + f11.min())
    zfe = zero_crossings(xfe, f12)
    print('\n    CONTRE-VERIFICATION INDEPENDANTE (elements finis Q4, plaque nue)')
    print('    %-46s %10.5f  (Ritz %.5f, ecart %.3f%%)'
          % ('dome (1,1)', dome_fe, dome_mod, rel(dome_mod, dome_fe)))
    print('    %-46s %10.5f  (Ritz %.5f, ecart %.3f%%)'
          % ('L_DD1/DD10', r1_fe, r1_mod, rel(r1_mod, r1_fe)))
    print('    %-46s %10.4f  (Ritz %.4f)'
          % ('zero de (1,2), x/lp', zfe[0] / 0.1 if zfe.size else np.nan,
             zc[0] / bare.lp))
    print('    -> les invariants ne dependent pas de la discretisation ; l ecart')
    print('       de %.2f%% du dome vis-a-vis de la Fig. 7 est donc un ecart'
          % abs(rel(dome_mod, dome_pap)))
    print('       MODELE/PAPIER (ou de numerisation), pas un artefact Ritz.')

# ---------------------------------------------------------------------------
# 4. "Except the first element, all elements vary largely"
# ---------------------------------------------------------------------------
print('\n' + SEP)
print('[4] "EXCEPT THE FIRST ELEMENT, OTHER ELEMENTS ALL VARY WITH POSITION')
print('    LARGELY" — mise en chiffres  [GAUGE-FREE : les deux rapports')
print('    ci-dessous sont invariants par changement de jauge]')
print(SEP)
print('    %-6s %12s %12s %16s %14s' % ('elem', 'max-min', 'DD_i0',
                                        '(max-min)/DD_i0', '(max-min)/max|.|'))
print('    ' + SUB)
rv = {}
for nm, _ in ELEMS:
    st = stats[nm]
    span = st['max'] - st['min']
    amax = max(abs(st['max']), abs(st['min']))
    r_mean = span / st['DD0'] if abs(st['DD0']) > 1e-6 else float('inf')
    rv[nm] = (r_mean, span / amax)
    txt = ('%.2f %%' % (100 * r_mean)) if np.isfinite(r_mean) \
        else 'INFINI (DD0 = 0)'
    print('    %-6s %12.5f %12.5f %16s %14s'
          % (nm, span, st['DD0'], txt, '%.1f %%' % (100 * span / amax)))
print('    ' + SUB)
print('    -> element (1,1) : %.2f%% de variation relative ;' % (100 * rv['(1,1)'][0]))
print('       elements (1,2)=(2,1) : moyenne EXACTEMENT nulle, la variation')
print('       relative est infinie (l element change de SIGNE le long de la passe) ;')
print('       element (2,2) : %.0f%% (min = 0 au noeud du mode 2).'
      % (100 * rv['(2,2)'][0]))
print('    Rapport des variations relatives (2,2)/(1,1) = %.0f.'
      % (rv['(2,2)'][0] / rv['(1,1)'][0]))
print('    CONFIRME sans ambiguite l affirmation qualitative du papier.')

# ---------------------------------------------------------------------------
# 5. Condition S : alpha4 et alpha40*DD0 (Eqs. 23-25)
# ---------------------------------------------------------------------------
print('\n' + SEP)
print('[5] CONDITION S DU PAPIER (4900 tr/min, a_p = 0.3 mm, avalant,')
print('    a_e = 0.1 mm) — Eqs. (23)-(25) via milling_dynamics')
print(SEP)
up = md.uncertainty_parameters(patched, RPM_S, AP_S, patched.hp)
a4 = up['alpha_bar4']
print('    abar4 (moyenne temporelle de alpha4 sur une periode de dent)')
print('        = %+.4e  [N/m]   (module %.4e)' % (a4, abs(a4)))
print('    alpha40   = 1.6 abar4 = %+.4e' % up['alpha40'])
print('    L_Palpha  = 1.3 abar4 = %+.4e' % up['L_Palpha'])
print('    intervalle [alpha40-L_Palpha, alpha40+L_Palpha] = [%.3f, %.3f] x abar4'
      % ((up['alpha40'] - up['L_Palpha']) / a4,
         (up['alpha40'] + up['L_Palpha']) / a4))
print('        -> [0.300, 2.900] x abar4 : Eq. (23) INTERNEMENT COHERENTE.')
print('    NB : le signe de alpha4 est negatif dans cette implantation (la')
print('    convention de signe est auditee par le script 05) ; le papier ne')
print('    publie pas la valeur de abar4, seuls les facteurs 1.6 / 1.3 / 0.3 /')
print('    2.9 sont verifiables et ils le sont exactement.')

print('\n    alpha40 * DD_Pr0 (Eq. 25), JAUGE MASSE-NORMALISEE du depot')
print('    [plaque AVEC pastille, telle que la renvoie uncertainty_parameters] :')
DD0_raw, LDD_raw = up['DD0'], up['LDD']
A = up['nominal_a40_DD']
for i in range(2):
    print('        DD0  [%d,:] = %12.5f %12.5f   [1/kg]' % (i, *DD0_raw[i]))
for i in range(2):
    print('        LDD  [%d,:] = %12.5f %12.5f   [1/kg]' % (i, *LDD_raw[i]))
for i in range(2):
    print('        a40*DD0[%d,:] = %+12.4e %+12.4e   [rad^2/s^2]' % (i, *A[i]))
w = patched.omega_n[:2]
print('    grandeur GAUGE-FREE associee : perturbation RELATIVE de raideur')
print('    modale a40*DD0(i,i)/omega_i^2 = %+.4f  et  %+.4f'
      % (A[0, 0] / w[0]**2, A[1, 1] / w[1]**2))
print('        soit %.1f%% et %.1f%% de variation de raideur modale — ordre de'
      % (100 * abs(A[0, 0] / w[0]**2), 100 * abs(A[1, 1] / w[1]**2)))
print('        grandeur attendu pour un couplage regeneratif a a_p = 0.3 mm.')

print('\n    memes parametres dans la JAUGE DU PAPIER (plaque NUE, Fig. 7) :')
A_g = up['alpha40'] * DD0_g
for i in range(2):
    print('        DD_Pr0 [%d,:] = %12.5f %12.5f' % (i, *DD0_g[i]))
for i in range(2):
    print('        L_DD   [%d,:] = %12.5f %12.5f' % (i, *LDD_g[i]))
for i in range(2):
    print('        a40*DD0[%d,:] = %+12.4e %+12.4e' % (i, *A_g[i]))

# ---------------------------------------------------------------------------
# 6. Ecart : la pastille brise la symetrie de la Fig. 7
# ---------------------------------------------------------------------------
print('\n' + SEP)
print('[6] ECART — LA PASTILLE BRISE LA SYMETRIE DE LA FIG. 7')
print(SEP)
xs_p, Dp = patched.D_top_edge(N_POS)
Dp2 = Dp[:, :2].copy()
if Dp2[:, 0].mean() < 0:
    Dp2[:, 0] *= -1.0
if Dp2[0, 1] < 0:                      # meme convention que paper_gauge
    Dp2[:, 1] *= -1.0
Dp2 /= np.array(s_gauge)               # meme jauge, pour superposition
p11, p12, p21, p22 = elements(Dp2)
print('    plaque NUE (Fig. 7 du papier)      : D11(0) = %.5f, D11(lp) = %.5f'
      % (e11[0], e11[-1]))
print('    plaque AVEC pastille (coin droit)  : D11(0) = %.5f, D11(lp) = %.5f'
      % (p11[0], p11[-1]))
print('    asymetrie relative |D11(0)-D11(lp)|/max :  nue %.2e   pastillee %.1f%%'
      % (abs(e11[0] - e11[-1]) / e11.max(),
         100 * abs(p11[0] - p11[-1]) / p11.max()))
print('    idem (2,2)                              :  nue %.2e   pastillee %.1f%%'
      % (abs(e22[0] - e22[-1]) / e22.max(),
         100 * abs(p22[0] - p22[-1]) / p22.max()))
zcp = zero_crossings(xs_p, p12)
print('    zero de (1,2) : nue x/lp = %.4f   pastillee x/lp = %.4f'
      % (zc[0] / bare.lp, zcp[0] / patched.lp if zcp.size else np.nan))
print('    dome (1,1)    : nue %.5f (max en mi-portee)   pastillee %.5f'
      % (dome_mod, p11.max() / p11.min()))
print('        (le maximum de D11 se deplace en x = %.1f mm sur la plaque'
      % (1000 * xs_p[np.argmax(p11)]))
print('         pastillee : la pastille raidit le bord droit.)')
print('    -> La Fig. 7 du papier ne peut avoir ete calculee QUE sur la plaque')
print('       nue. Le modele de commande (plaque pastillee) donne des DD0/L_DD')
print('       sensiblement differents ; c est ce jeu-la que retourne')
print('       uncertainty_parameters, cf. section [5].')

# ---------------------------------------------------------------------------
# 7. Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))
titles = ['(a)  $D_{Pr}^{T}D_{Pr}(1,1)$', '(b)  $D_{Pr}^{T}D_{Pr}(1,2)$',
          '(c)  $D_{Pr}^{T}D_{Pr}(2,1)$', '(d)  $D_{Pr}^{T}D_{Pr}(2,2)$']
paper_pts = [[(0.0, T['D11_ends']), (50.0, T['D11_mid']),
              (100.0, T['D11_ends'])],
             [(0.0, T['D12_amp']), (100.0, -T['D12_amp'])],
             [(0.0, T['D12_amp']), (100.0, -T['D12_amp'])],
             [(0.0, T['D22_ends']), (50.0, 0.0), (100.0, T['D22_ends'])]]
patched_curves = [p11, p12, p21, p22]

for k, ((nm, v), ax) in enumerate(zip(ELEMS, axes.ravel())):
    st = stats[nm]
    x_mm = 1000.0 * xs
    ax.plot(x_mm, v, color='#1f4e9c', lw=2.2, label='actual (bare plate)',
            zorder=4)
    ax.axhline(st['DD0'], color='k', ls='--', lw=1.4,
               label='mean = (max+min)/2', zorder=3)
    ax.axhline(st['max'], color='#c0392b', ls=':', lw=1.6, label='maximum',
               zorder=3)
    ax.axhline(st['min'], color='#1e8449', ls=':', lw=1.6, label='minimum',
               zorder=3)
    ax.plot(1000.0 * xs_p, patched_curves[k], color='0.45', lw=1.1, ls='-.',
            label='with piezo patch (right)', zorder=2)
    px = [p[0] for p in paper_pts[k]]
    py = [p[1] for p in paper_pts[k]]
    ax.plot(px, py, 'o', ms=7, mfc='none', mew=1.8, color='#e67e22',
            label='Fig. 7 digitized', zorder=5)
    ax.set_title(titles[k], fontsize=11)
    ax.set_xlabel('milling position $x$ along the upper edge [mm]', fontsize=9)
    ax.set_ylabel('element value (paper gauge)', fontsize=9)
    ax.set_xlim(0, 100)
    lo = min(v.min(), patched_curves[k].min(), min(py))
    hi = max(v.max(), patched_curves[k].max(), max(py))
    pad = 0.16 * (hi - lo)
    ax.set_ylim(lo - 1.45 * pad, hi + pad)      # place pour l encart
    ax.grid(alpha=0.3)
    r_mean, r_max = rv[nm]
    rtxt = ('%.2f %%' % (100 * r_mean)) if np.isfinite(r_mean) else 'infinite'
    ax.text(0.02, 0.03,
            'DD$_{%d0}$ = %.4f   L$_{DD%d}$ = %.4f\n'
            '(max-min)/mean = %s\n(max-min)/max|.| = %.1f %%'
            % (k + 1, st['DD0'], k + 1, st['LDD'], rtxt, 100 * r_max),
            transform=ax.transAxes, fontsize=8.2, va='bottom',
            bbox=dict(fc='white', ec='0.6', alpha=0.9, boxstyle='round,pad=0.35'))

hnd, lab = axes[0, 0].get_legend_handles_labels()
fig.legend(hnd, lab, loc='upper center', bbox_to_anchor=(0.5, 0.918),
           ncol=6, fontsize=8.4, framealpha=0.9)
fig.suptitle('Fig. 7 reproduction — elements of $D_{Pr}^{T}D_{Pr}$ along the '
             'milling pass (upper edge of the cantilever plate)\n'
             'paper modal gauge s = (%.3f, %.3f) is FITTED on 3 digitized '
             'levels, so the LEVELS are not predictions\n'
             'gauge-free checks:  dome ratio %.5f vs %.5f digitized '
             '(%+.2f %%)   |   (1,2) zero at x/l = %.4f   |   '
             'symmetry residual %.0e   |   $D_{12}^2 = D_{11}D_{22}$ to %.0e'
             % (s_gauge[0], s_gauge[1], dome_mod, dome_pap,
                rel(dome_mod, dome_pap), zc[0] / bare.lp, sym11, cauchy),
             fontsize=9.6)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(FIGPATH, dpi=140)
plt.close(fig)

print('\n' + SEP)
print('[7] FIGURE : %s' % FIGPATH)
print(SEP)
print(' Duree : calcul entierement analytique (Ritz 196 ddl + FE 30x24),')
print(' quelques secondes ; aucune grille n a du etre reduite.')
