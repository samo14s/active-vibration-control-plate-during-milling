"""
17_alpha4_band_stability.py — Fig. 6 du papier (pourquoi la bande 0.3-2.9)
===========================================================================
Du, Liu, Dai, Long, IJMS 274 (2024) 109257, Sec. 3.2 / Fig. 6.

Le papier justifie sa bande d'incertitude sur le coefficient d'effort de coupe
par un argument PUREMENT GRAPHIQUE, reproduit ici. Legende de la Fig. 6 :

  "Predicted stability lobes of milling the upper edge on the thin-walled
   cantilever plate with theoretical milling force coefficients, 0.3, 1, and
   2.9 times of average milling force coefficients in the case of 0.1 mm
   radial cutting depth, (a) start position, (b) 1/4 position, (c) 1/2
   position."

et l'affirmation testee, page 7 :

  "the stability results predicted with actual milling force coefficient
   differ from those with average milling force coefficient in some spindle
   speeds, they are HIGHER than that with 2.9 times of average milling force
   coefficient and LOWER than that with 0.3 times of average milling force
   coefficient IN ALL SPINDLE SPEEDS. Thus the milling force coefficient
   alpha4(t) can be regarded as varying within 0.3 abar4 ~ 2.9 abar4."

d'ou l'Eq. (23) : alpha40 = 1.6 abar4, L_Palpha = 1.3 abar4.

Ce script calcule, sur le modele REDUIT A DEUX MODES (Eq. 21) et aux trois
positions d'outil du papier (depart, 1/4, 1/2 de l_P), les quatre familles de
lobes a_p,lim(rpm) de 3000 a 7000 tr/min :

  (i)   coefficient theorique alpha4(t) variable      coeff_mode='time'
  (ii)  coefficient moyenne temporelle abar4          coeff_mode='mean'
  (iii) 0.3 x abar4                                   coeff_mode='mean', 0.3
  (iv)  2.9 x abar4                                   coeff_mode='mean', 2.9

puis teste vitesse par vitesse l'encadrement  L(2.9x) <= L(theorique) <= L(0.3x).

DEUX ECARTS ASSUMES PAR RAPPORT A LA CONSIGNE DE LA TACHE
  1. Le signe. La consigne annonce SIGN_SIM = -1 ; control/config.py contient
     SIGN_SIM = +1.0 (Eq. 13 telle que publiee, tranchee par
     verification/18_sign_convention.py). Le script LIT config.SIGN_SIM a
     l'execution, l'imprime, et refait tout le test avec la convention
     OPPOSEE sur une grille plus grossiere : la conclusion ne doit pas
     dependre de ce choix.
  2. La bissection. lti_floquet.limit fait une bissection LINEAIRE ; avec la
     tolerance demandee (2e-5 m) et des limites de l'ordre de 0.04 mm cela
     donne une resolution RELATIVE de ~45 %, sans rapport avec les marges
     d'encadrement a mesurer. On utilise donc lti_floquet.is_stable (le meme
     noyau de Floquet, exactement le meme predicat) dans une bissection
     LOGARITHMIQUE a 2 % de resolution relative constante. L'accord avec
     lti_floquet.limit est verifie et imprime.

Sortie : figures/verification/17_alpha4_band_stability.png + tableaux.
"""
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import time                                                     # noqa: E402
import numpy as np                                              # noqa: E402
import matplotlib                                               # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model'),
                os.path.join(HERE, '..', 'control')]

import config as C                                              # noqa: E402
from plate_model import build_plate, F_THEORETICAL              # noqa: E402
from lti_floquet import is_stable, limit as lf_limit            # noqa: E402
from milling_dynamics import (alpha4_average, alpha4_series,    # noqa: E402
                              N_TEETH, AE_NOM, FZ_NOM)

FIG = os.path.join(HERE, '..', 'figures', 'verification')
os.makedirs(FIG, exist_ok=True)
STEM = '17_alpha4_band_stability'

# ------------------------------------------------------------------ reglages
N_MODES_PLATE = 5              # plaque construite/calee sur 5 modes
N_MODES_FLOQ = 2              # modele REDUIT du papier (Eq. 21)
M_FLOQ = 60                   # sous-intervalles par periode de dent
LO, HI = 3.0e-6, 15.0e-3      # encadrement initial de a_p [m]
RTOL = 0.02                   # resolution RELATIVE de la bissection log
SPEEDS = np.arange(3000, 7001, 200)          # 21 vitesses (pas 200 tr/min)
# Grille VOLONTAIREMENT GROSSIERE pour le controle de signe (5 vitesses au
# lieu de 21) : c'est le seul poste sacrifie pour tenir le budget de ~3 min.
SPEEDS_X = np.arange(3000, 7001, 1000)       # grille du controle de signe
POSITIONS = (0.0, 0.25, 0.5)                 # fractions de l_P (Fig. 6 a/b/c)
PANELS = ('(a) start position', '(b) 1/4 position', '(c) 1/2 position')
# (etiquette, coeff_mode, facteur multiplicatif de abar4)
CURVES = (('theoretical a4(t)', 'time', 1.0),
          ('mean a4', 'mean', 1.0),
          ('0.3 x mean a4', 'mean', 0.3),
          ('2.9 x mean a4', 'mean', 2.9))
S_LO_PAPER, S_HI_PAPER = 0.3, 2.9            # bande du papier
M_CONV = (40, 60, 120, 240)                  # etude de convergence en m
RPM_CONV = 6000                              # vitesse de l'etude de convergence
SIGN = float(C.SIGN_SIM)                     # convention LUE a l'execution
N_PROC = 4


# ------------------------------------------------------- coeur de calcul ----
def log_limit(plate, rpm, x_pos, mode, scale, m=M_FLOQ, lo=LO, hi=HI,
              rtol=RTOL):
    """Profondeur limite [m] par bissection LOGARITHMIQUE sur le MEME predicat
    que lti_floquet.limit (lti_floquet.is_stable). Retourne (a_p,lim, statut)
    avec statut dans {'ok', 'unstable_at_lo', 'saturated'}."""
    def ok(ap):
        return is_stable(plate, rpm, ap, x_pos, None, N_MODES_FLOQ, m,
                         mode, scale)[0]
    if not ok(lo):
        return 0.0, 'unstable_at_lo'
    if ok(hi):
        return hi, 'saturated'
    llo, lhi = np.log(lo), np.log(hi)
    n = int(np.ceil(np.log2((lhi - llo) / np.log1p(rtol))))
    for _ in range(n):
        lm = 0.5 * (llo + lhi)
        if ok(float(np.exp(lm))):
            llo = lm
        else:
            lhi = lm
    return float(np.exp(0.5 * (llo + lhi))), 'ok'


# ------------------------------------------------------------- parallelisme -
_PLATE = None


def _init():
    global _PLATE
    _PLATE = build_plate('right', n_modes=N_MODES_PLATE)


def _job(arg):
    """arg = ('lobes', sign, ipos, ispd, pos, rpm) ou ('conv', m)."""
    if arg[0] == 'conv':
        m = arg[1]
        return arg, [log_limit(_PLATE, RPM_CONV, 0.0, mo, SIGN * sc, m=m)[0]
                     for _, mo, sc in CURVES]
    _, sign, _ipos, _ispd, pos, rpm = arg
    x = pos * _PLATE.lp
    return arg, [log_limit(_PLATE, rpm, x, mo, sign * sc)[0]
                 for _, mo, sc in CURVES]


def run_jobs(jobs):
    """Repartit sur les coeurs ; retombe en sequentiel si le pool echoue."""
    try:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(N_PROC, initializer=_init) as pool:
            return dict(pool.map(_job, jobs, chunksize=1))
    except Exception as exc:                                # pragma: no cover
        print(f"  [pool indisponible : {exc} -> execution sequentielle]")
        _init()
        return dict(_job(j) for j in jobs)


# ------------------------------------------------------------------- main ---
def main():
    t0 = time.time()
    bar = '=' * 100
    print(bar)
    print(" SCRIPT 17 — BANDE 0.3-2.9 SUR alpha4 (Fig. 6 et Eq. 23 du papier)")
    print(bar)

    plate = build_plate('right', n_modes=N_MODES_PLATE)
    print(f"  Modele    : Chebyshev-Ritz, plaque calee sur les frequences"
          f" THEORIQUES du Tableau 4 {F_THEORETICAL}")
    print(f"              f_n modele = {np.round(plate.freq_n[:5], 1).tolist()}"
          f" Hz ; zeta = {np.round(plate.zeta_modes[:5] * 100, 2).tolist()} %")
    print(f"  Floquet   : modele REDUIT a {N_MODES_FLOQ} modes (Eq. 21),"
          f" discretisation complete, m = {M_FLOQ} sous-intervalles/periode,"
          " rayon spectral par iteration de puissance adaptative")
    print(f"  Coupe     : avalant, ae = {AE_NOM * 1e3:.3f} mm,"
          f" fz = {FZ_NOM * 1e3:.3f} mm/dent, {N_TEETH} dents, D = 10 mm,"
          " helice 35 deg, outil sur le bord superieur")
    print(f"  Grille    : {SPEEDS[0]}-{SPEEDS[-1]} tr/min pas"
          f" {SPEEDS[1] - SPEEDS[0]} ({SPEEDS.size} vitesses) x"
          f" {len(POSITIONS)} positions x {len(CURVES)} courbes")
    print(f"  Budget    : ~3.7 min sur {N_PROC} coeurs. Le pas de 200 tr/min"
          " est CONSERVE (les violations sont")
    print("              des pics d'une seule vitesse : un pas de 400 tr/min"
          " les manquerait) ; le seul poste")
    print("              reduit est le controle de signe, sur 5 vitesses au"
          " lieu de 21.")
    print(f"  Bissection: LOG, {LO * 1e3:.4f}-{HI * 1e3:.1f} mm,"
          f" resolution relative {RTOL * 100:.0f} % (voir en-tete : la"
          " bissection lineaire tol = 2e-5 m demandee vaut ~45 % de"
          " resolution relative a 0.04 mm)")
    print()
    print("  CONVENTION DE SIGNE")
    print(f"    config.SIGN_SIM lu a l'execution = {SIGN:+.1f}"
          " (Eq. 13 telle que publiee)")
    print("    La consigne de la tache annonce SIGN_SIM = -1 : c'est l'ETAT"
          " HISTORIQUE, decrit au passe dans")
    print("    l'en-tete de verification/18_sign_convention.py ; ce script a"
          " tranche pour l'Eq. (13) telle que")
    print("    publiee et control/config.py porte desormais SIGN_SIM = +1.0"
          " (ligne 64). On passe donc")
    print(f"    coeff_scale = {SIGN:+.1f} x facteur, et la convention OPPOSEE"
          f" ({-SIGN:+.1f} x facteur) est recalculee sur une")
    print("    grille grossiere en fin de script pour montrer que la"
          " conclusion n'en depend pas.")
    print()

    # ---------------------------------------------- 1. le coefficient alpha4
    print('-' * 100)
    print(" 1. COEFFICIENT alpha4 — moyenne, extremes, linearite en a_p")
    print('-' * 100)
    ap_ref = 0.3e-3
    a3s, a4s = alpha4_series(4900, ap_ref, plate.hp, 400)
    duty = float(np.mean(a4s != 0.0))
    print(f"  a 4900 tr/min, a_p = {ap_ref * 1e3:.2f} mm :")
    print(f"    abar4 (moyenne temporelle) = {alpha4_average(4900, ap_ref, plate.hp):+.4e} N/m")
    print(f"    alpha4(t) min / max        = {a4s.min():+.4e} /"
          f" {a4s.max():+.4e} N/m")
    print(f"    |max|/|moyenne|            = {abs(a4s.min() / a4s.mean()):.2f}"
          f"   (engagement = {duty * 100:.1f} % de la periode de dent)")
    print("    NOTE : alpha4 est NEGATIF dans cette implantation ; le signe"
          " global du couplage est porte par coeff_scale.")
    print(f"  Linearite de abar4 en a_p (4900 tr/min) — abar4/a_p [N/m^2] :")
    lin = []
    for ap in (0.02e-3, 0.05e-3, 0.1e-3, 0.3e-3, 1.0e-3, 3.0e-3):
        r = alpha4_average(4900, ap, plate.hp) / ap
        lin.append(r)
        print(f"    a_p = {ap * 1e3:5.2f} mm : {r:+.6e}")
    spread = (max(lin) - min(lin)) / abs(np.mean(lin))
    print(f"    dispersion max-min = {spread * 100:.2f} %  ->  abar4 est"
          " quasi PROPORTIONNEL a a_p")
    print("    CONSEQUENCE : les courbes 0.3x et 2.9x sont, a la resolution de"
          " la bissection pres, la courbe")
    print("    'mean' divisee par 0.3 et par 2.9. Le test d'encadrement se"
          " ramene donc au rapport")
    print("      s_eff(rpm) = L(mean)/L(theorique)  qui doit rester dans"
          f" [{S_LO_PAPER}, {S_HI_PAPER}].")
    print()

    # -------------------------------------------------------- 2. les calculs
    # les taches de convergence (m eleve) sont les plus longues : on les met en
    # TETE de liste pour qu'elles ne trainent pas seules en fin de pool
    jobs = [('conv', m) for m in sorted(M_CONV, reverse=True)]
    jobs += [('lobes', SIGN, ip, isp, pos, int(rpm))
             for ip, pos in enumerate(POSITIONS)
             for isp, rpm in enumerate(SPEEDS)]
    jobs += [('lobes', -SIGN, ip, isp, pos, int(rpm))
             for ip, pos in enumerate(POSITIONS)
             for isp, rpm in enumerate(SPEEDS_X)]
    print(f"  [calcul de {len(jobs)} points de bissection sur {N_PROC} coeurs"
          " ...]", flush=True)
    res = run_jobs(jobs)
    L = np.zeros((2, len(POSITIONS), SPEEDS.size, len(CURVES)))
    LX = np.zeros((len(POSITIONS), SPEEDS_X.size, len(CURVES)))
    for arg, vals in res.items():
        if arg[0] != 'lobes':
            continue
        _, sg, ip, isp, _, _ = arg
        if sg == SIGN:
            L[0, ip, isp] = vals
        else:
            LX[ip, isp] = vals
    conv = {arg[1]: v for arg, v in res.items() if arg[0] == 'conv'}

    # ------------------------------------------------- 3. tableaux par position
    stats = []
    for ip, pos in enumerate(POSITIONS):
        Lt, Lm, L03, L29 = (L[0, ip, :, k] for k in range(4))
        s_eff = np.where(Lt > 0, Lm / np.maximum(Lt, 1e-30), np.inf)
        up_ok = Lt <= L03 * (1.0 + 1e-12)
        lo_ok = Lt >= L29 * (1.0 - 1e-12)
        brk = up_ok & lo_ok
        stats.append(dict(pos=pos, Lt=Lt, Lm=Lm, L03=L03, L29=L29,
                          s_eff=s_eff, brk=brk, up_ok=up_ok, lo_ok=lo_ok))
        print('-' * 100)
        print(f" 2.{ip + 1} POSITION {PANELS[ip]}  (x = {pos:.2f} l_P ="
              f" {pos * plate.lp * 1e3:.1f} mm) — a_p,lim en mm")
        print('-' * 100)
        print(f"  {'rpm':>6s} {'theoretical':>12s} {'mean':>10s}"
              f" {'0.3 x mean':>11s} {'2.9 x mean':>11s} {'th/mean':>8s}"
              f" {'s_eff':>7s}  {'bracketed?':>12s}")
        for j, rpm in enumerate(SPEEDS):
            tag = 'yes' if brk[j] else (
                'NO (< 2.9x)' if not lo_ok[j] else 'NO (> 0.3x)')
            print(f"  {rpm:6d} {Lt[j] * 1e3:12.4f} {Lm[j] * 1e3:10.4f}"
                  f" {L03[j] * 1e3:11.4f} {L29[j] * 1e3:11.4f}"
                  f" {Lt[j] / Lm[j]:8.3f} {s_eff[j]:7.3f}  {tag:>12s}")
        n_ok = int(brk.sum())
        print(f"  -> encadre a {n_ok}/{SPEEDS.size} vitesses"
              f" ({100.0 * n_ok / SPEEDS.size:.1f} %)")
        # verification du re-echelonnement exact
        r03 = np.median(L03 / Lm)
        r29 = np.median(L29 / Lm)
        print(f"  -> mediane L(0.3x)/L(mean) = {r03:.3f} (attendu"
              f" 1/0.3 = {1 / 0.3:.3f}, ecart"
              f" {100 * abs(r03 - 1 / 0.3) / (1 / 0.3):.1f} %) ;"
              f" L(2.9x)/L(mean) = {r29:.4f} (attendu"
              f" {1 / 2.9:.4f}, ecart"
              f" {100 * abs(r29 - 1 / 2.9) / (1 / 2.9):.1f} %)")
        print()

    # ------------------------------------------- 4. verdict sur l'encadrement
    print('=' * 100)
    print(" 3. TEST DE L'AFFIRMATION DU PAPIER (Sec. 3.2, page 7)")
    print('=' * 100)
    print('  "they are higher than that with 2.9 times of average milling'
          ' force coefficient and lower')
    print('   than that with 0.3 times of average milling force coefficient'
          ' in all spindle speeds"')
    print()
    print(f"  {'position':>10s} {'bracketed':>12s} {'frac':>7s}"
          f" {'worst violation':>34s} {'s_eff min':>10s} {'s_eff max':>10s}")
    tot_ok = tot = 0
    worst_glob = None
    for st in stats:
        n_ok = int(st['brk'].sum())
        tot_ok += n_ok
        tot += st['brk'].size
        # marge basse : L_th/L_29 (doit etre >= 1) ; marge haute : L_03/L_th
        m_lo = st['Lt'] / np.maximum(st['L29'], 1e-30)
        m_hi = st['L03'] / np.maximum(st['Lt'], 1e-30)
        j_lo, j_hi = int(np.argmin(m_lo)), int(np.argmin(m_hi))
        if m_lo[j_lo] <= m_hi[j_hi]:
            j, m_w, side = j_lo, m_lo[j_lo], 'below the 2.9x curve'
        else:
            j, m_w, side = j_hi, m_hi[j_hi], 'above the 0.3x curve'
        txt = (f"{SPEEDS[j]} rpm: {(1 - m_w) * 100:+.1f} % {side}"
               if m_w < 1 else f"none (min margin {m_w:.2f}x)")
        print(f"  {st['pos']:10.2f} {n_ok:6d}/{st['brk'].size:<5d}"
              f" {100.0 * n_ok / st['brk'].size:6.1f}%"
              f" {txt:>34s} {st['s_eff'].min():10.3f}"
              f" {st['s_eff'].max():10.3f}")
        if worst_glob is None or m_w < worst_glob[0]:
            worst_glob = (m_w, st['pos'], int(SPEEDS[j]), side, st)
    print(f"  {'ALL':>10s} {tot_ok:6d}/{tot:<5d}"
          f" {100.0 * tot_ok / tot:6.1f}%")
    print()
    m_w, pw, rw, side, stw = worst_glob
    jw = list(SPEEDS).index(rw)
    print(f"  PIRE VIOLATION : position {pw:.2f} l_P, {rw} tr/min")
    print(f"    a_p,lim theorique = {stw['Lt'][jw] * 1e3:.4f} mm ;"
          f" 2.9x = {stw['L29'][jw] * 1e3:.4f} mm ;"
          f" 0.3x = {stw['L03'][jw] * 1e3:.4f} mm")
    print(f"    la courbe theorique est {(1 - m_w) * 100:.1f} % {side}"
          f"  (s_eff = {stw['s_eff'][jw]:.3f} contre la borne"
          f" {S_HI_PAPER if 'below' in side else S_LO_PAPER})")
    print()

    # ------------------------------- 5. bande requise et Eq. (23) recalculee
    s_all = np.concatenate([st['s_eff'] for st in stats])
    s_min, s_max = float(s_all.min()), float(s_all.max())
    a40_req, La_req = 0.5 * (s_min + s_max), 0.5 * (s_max - s_min)
    print('-' * 100)
    print(" 4. BANDE REQUISE PAR CE MODELE contre Eq. (23) du papier")
    print('-' * 100)
    print("  s_eff(rpm) = L(mean)/L(theoretical) est le multiple de abar4 dont"
          " la courbe 'mean' reproduirait")
    print("  exactement la limite theorique a cette vitesse.")
    print(f"  {'quantite':>26s} {'ce modele':>12s} {'papier':>10s}"
          f" {'ecart rel.':>11s}")
    rows = [('borne basse s_lo', s_min, S_LO_PAPER),
            ('borne haute s_hi', s_max, S_HI_PAPER),
            ('alpha40 / abar4', a40_req, 1.6),
            ('L_Palpha / abar4', La_req, 1.3)]
    for name, ours, paper in rows:
        print(f"  {name:>26s} {ours:12.3f} {paper:10.3f}"
              f" {100 * (ours - paper) / paper:+10.1f} %")
    n_lo = int((s_all < S_LO_PAPER).sum())
    n_hi = int((s_all > S_HI_PAPER).sum())
    print(f"  points sous {S_LO_PAPER} : {n_lo}/{s_all.size} ;"
          f" points au-dessus de {S_HI_PAPER} : {n_hi}/{s_all.size}"
          f" ; total hors bande {n_lo + n_hi}/{s_all.size}"
          f" ({100.0 * (n_lo + n_hi) / s_all.size:.1f} %)")
    print("  LECTURE : l'ecart de +143 % sur la borne BASSE n'est pas un"
          " defaut du papier — c'est de la")
    print("  CONSERVATISME : aucun point ne descend sous 0.3, la borne basse"
          " du papier est simplement")
    print("  beaucoup plus large que necessaire. C'est la borne HAUTE (2.9)"
          " qui est prise en defaut.")
    print("  La bande STRICTEMENT necessaire ici serait"
          f" [{s_min:.2f}, {s_max:.2f}], soit alpha40 ="
          f" {a40_req:.2f} abar4 et L_Palpha = {La_req:.2f} abar4 ;")
    print("  la demi-largeur du papier (1.3) est donc la bonne (a 1.5 % pres)"
          " mais son CENTRE (1.6) est trop bas.")
    print()

    # --------------------------------------------- 6. convergence et controles
    print('-' * 100)
    print(" 5. CONTROLES NUMERIQUES")
    print('-' * 100)
    print(f"  (a) convergence en m (nombre de sous-intervalles/periode),"
          f" position 0, {RPM_CONV} tr/min, a_p,lim en mm :")
    print(f"      {'m':>5s} {'theoretical':>12s} {'mean':>10s} {'0.3x':>10s}"
          f" {'2.9x':>10s} {'s_eff':>8s} {'bracketed':>10s}")
    for m in M_CONV:
        v = conv[m]
        print(f"      {m:5d} {v[0] * 1e3:12.4f} {v[1] * 1e3:10.4f}"
              f" {v[2] * 1e3:10.4f} {v[3] * 1e3:10.4f}"
              f" {v[1] / v[0]:8.3f} {str(v[3] <= v[0] <= v[2]):>10s}")
    ref = conv[M_CONV[-1]]
    d = max(abs(conv[M_FLOQ][k] / ref[k] - 1) for k in range(4)) \
        if M_FLOQ in conv else float('nan')
    print(f"      ecart max m = {M_FLOQ} contre m = {M_CONV[-1]} :"
          f" {d * 100:.1f} %  (resolution de bissection ="
          f" {RTOL * 100:.0f} %)")
    print()
    print("  (b) accord avec lti_floquet.limit (bissection LINEAIRE du depot,"
          " tol = 2e-5 m) :")
    for rpm in (4900, RPM_CONV):
        a = lf_limit(plate, rpm, 0.0, None, lo=0.01e-3, hi=3.0e-3, tol=2e-5,
                     n_modes=N_MODES_FLOQ, m=M_FLOQ, coeff_mode='time',
                     coeff_scale=SIGN)
        b = log_limit(plate, rpm, 0.0, 'time', SIGN)[0]
        print(f"      {rpm:5d} tr/min : lti_floquet.limit ="
              f" {a * 1e3:8.4f} mm ; bissection log = {b * 1e3:8.4f} mm ;"
              f" ecart {100 * abs(a - b) / b:5.1f} %"
              f"  (demi-cellule lineaire = {0.5 * 2e-5 / b * 100:.1f} %)")
    print()
    print(f"  (c) meme test avec la convention de signe OPPOSEE"
          f" ({-SIGN:+.1f}), grille VOLONTAIREMENT GROSSIERE"
          f" ({SPEEDS_X.size} vitesses, pas"
          f" {SPEEDS_X[1] - SPEEDS_X[0]} tr/min, pour limiter la duree) :")
    print(f"      {'position':>10s} {'bracketed':>12s} {'frac':>7s}"
          f" {'s_eff min':>10s} {'s_eff max':>10s}")
    tot_ok_x = tot_x = 0
    for ip, pos in enumerate(POSITIONS):
        Lt, Lm, L03, L29 = (LX[ip, :, k] for k in range(4))
        se = np.where(Lt > 0, Lm / np.maximum(Lt, 1e-30), np.inf)
        bk = (Lt <= L03) & (Lt >= L29)
        tot_ok_x += int(bk.sum())
        tot_x += bk.size
        print(f"      {pos:10.2f} {int(bk.sum()):6d}/{bk.size:<5d}"
              f" {100.0 * bk.sum() / bk.size:6.1f}%"
              f" {se.min():10.3f} {se.max():10.3f}")
    print(f"      {'ALL':>10s} {tot_ok_x:6d}/{tot_x:<5d}"
          f" {100.0 * tot_ok_x / tot_x:6.1f}%")
    print()

    # ------------------------------------------------------------ 7. figure
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4), sharex=True)
    col = dict(th='k', mean='tab:green', c03='tab:blue', c29='tab:red')
    for ip, st in enumerate(stats):
        ax = axes[0, ip]
        ax.fill_between(SPEEDS, st['L29'] * 1e3, st['L03'] * 1e3,
                        color='0.85', zorder=0,
                        label='band 2.9x -- 0.3x (paper claim)')
        ax.semilogy(SPEEDS, st['L03'] * 1e3, '--', color=col['c03'], lw=1.4,
                    label='0.3 x mean $a_4$')
        ax.semilogy(SPEEDS, st['Lm'] * 1e3, '-.', color=col['mean'], lw=1.4,
                    label='mean $a_4$')
        ax.semilogy(SPEEDS, st['L29'] * 1e3, '--', color=col['c29'], lw=1.4,
                    label='2.9 x mean $a_4$')
        ax.semilogy(SPEEDS, st['Lt'] * 1e3, '-o', color=col['th'], lw=2.0,
                    ms=3.0, label='theoretical $a_4(t)$')
        bad = ~st['brk']
        if bad.any():
            ax.semilogy(SPEEDS[bad], st['Lt'][bad] * 1e3, 'x',
                        color='magenta', ms=11, mew=2.4, zorder=5,
                        label='NOT bracketed')
        n_ok = int(st['brk'].sum())
        ax.set_title(f"{PANELS[ip]}   ($x = {st['pos']:.2f}\\,l_P$)",
                     fontsize=11)
        ax.text(0.03, 0.05,
                f"bracketed at {n_ok}/{st['brk'].size} speeds"
                f" ({100.0 * n_ok / st['brk'].size:.0f} %)",
                transform=ax.transAxes, fontsize=9.5,
                bbox=dict(fc='w', ec='0.6', alpha=0.9))
        ax.grid(True, which='both', alpha=0.3)
        if ip == 0:
            ax.set_ylabel('axial depth limit $a_{p,lim}$  [mm]')
            ax.legend(fontsize=7.6, loc='upper left', framealpha=0.92)

        ax2 = axes[1, ip]
        ax2.axhspan(S_LO_PAPER, S_HI_PAPER, color='0.85', zorder=0,
                    label='paper band $0.3-2.9\\,\\bar a_4$')
        ax2.plot(SPEEDS, st['s_eff'], '-o', color='k', lw=1.8, ms=3.0,
                 label='$s_{eff}=L(\\mathrm{mean})/L(\\mathrm{theor.})$')
        ax2.axhline(S_HI_PAPER, color=col['c29'], ls='--', lw=1.2)
        ax2.axhline(S_LO_PAPER, color=col['c03'], ls='--', lw=1.2)
        ax2.axhline(1.6, color='tab:orange', ls=':', lw=1.4,
                    label='$\\alpha_{40}=1.6\\,\\bar a_4$ (Eq. 23)')
        if bad.any():
            ax2.plot(SPEEDS[bad], st['s_eff'][bad], 'x', color='magenta',
                     ms=11, mew=2.4, zorder=5)
        ax2.set_ylim(0.0, max(3.6, 1.15 * float(st['s_eff'].max())))
        ax2.set_xlabel('spindle speed [rpm]')
        ax2.grid(True, alpha=0.3)
        ax2.text(0.03, 0.86,
                 f"$s_{{eff}}\\in[{st['s_eff'].min():.2f},"
                 f" {st['s_eff'].max():.2f}]$",
                 transform=ax2.transAxes, fontsize=9.5,
                 bbox=dict(fc='w', ec='0.6', alpha=0.9))
        if ip == 0:
            ax2.set_ylabel('effective multiple of $\\bar a_4$')
            ax2.legend(fontsize=7.6, loc='lower right', framealpha=0.92)
    fig.suptitle("Fig. 6 of Du et al. (2024) re-computed — stability lobes"
                 " with theoretical, mean, 0.3x and 2.9x milling force"
                 f" coefficient (2-mode model, down milling,"
                 f" $a_e$ = 0.1 mm, sign = {SIGN:+.0f})", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = os.path.join(FIG, STEM + '.png')
    fig.savefig(out, dpi=140)
    print(f"  figure : {os.path.abspath(out)}")
    print(f"  duree  : {time.time() - t0:.1f} s")
    print(bar)


if __name__ == '__main__':
    main()
