"""
13_open_loop_stability_lobes.py — Fig. 13 du papier (lobes SANS commande)
==========================================================================
Le papier (Sec. 4.2, Fig. 13) predit la profondeur axiale limite a_p,lim du
bord superieur de la plaque encastree, en avalant, avec ae = 0.1 mm, pour
toutes les positions de l'outil le long de l'arete et toutes les vitesses de
broche. Il en tire trois affirmations verifiables :

  A1  "The stability limit of the plate is lower than 0.1 mm in most spindle
       speeds" ;
  A2  "only relatively larger in a few spindle speeds, such as around
       3600 rpm and 5400 rpm" ;
  A3  Fig. 14(a) : la condition S (4900 tr/min, a_p = 0.30 mm) DIVERGE sans
       commande — donc 0.30 mm doit se situer AU-DESSUS de a_p,lim(4900).

Ce script recalcule a_p,lim avec la machinerie de Floquet du depot
(discretisation complete de Ding et al., control/closed_loop.py) sur le
modele Chebyshev-Ritz a 5 modes, dans les DEUX calages du Tableau 4 :
mesure (540/1068/2787/3351/4122 Hz) et theorique (537/1101/2805/3423/4254 Hz).
Le papier a calcule sa Fig. 13 avec SON modele theorique : c'est la colonne
"theoretical" qui lui est directement comparable ; la colonne "measured" est
la plaque reelle, qui sert a la couche de commande.

CONVENTION DE SIGNE — le signe de reference est LU dans config.SIGN_SIM (il
est imprime tel quel) : c'est la convention de la couche control/ du depot.
Le signe oppose est calcule sur toute la grille lui aussi, parce que ce choix
inverse creux et bosses des lobes et change d'un facteur ~10 la limite a une
vitesse donnee (cf. verification/18_sign_convention.py). Ainsi le tableau
reste lisible quel que soit l'etat de config.py.

ACCELERATION — la carte de periode est identique a celle de
control/closed_loop.period_maps (verifiee bit a bit, cf. sortie), mais les
exponentielles de matrice sont MEMOISEES sur la valeur de alpha4 : en avalant
a ae = 0.1 mm l'outil n'est engage que ~12 % de la periode de dent, donc
alpha4 = 0 EXACTEMENT sur la grande majorite des sous-intervalles et une
seule expm suffit pour eux. Gain ~7x, resultat inchange.

Sortie : figures/verification/13_open_loop_stability_lobes.png + tableaux.
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
from matplotlib.colors import LogNorm                           # noqa: E402
from matplotlib.lines import Line2D                             # noqa: E402
from scipy.linalg import expm                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model'),
                os.path.join(HERE, '..', 'control')]

import config as C                                              # noqa: E402
from plate_model import build_plate, F_MEASURED, F_THEORETICAL  # noqa: E402
from closed_loop import build_matrices, spectral_radius         # noqa: E402
from closed_loop import period_maps, limit as repo_limit        # noqa: E402
from milling_dynamics import alpha4_series, N_TEETH             # noqa: E402

FIG = os.path.join(HERE, '..', 'figures', 'verification')
os.makedirs(FIG, exist_ok=True)
STEM = '13_open_loop_stability_lobes'

# ------------------------------------------------------------------ reglages
N_MODES = C.N_MODES                          # 5
AE = C.AE                                    # 0.1 mm
SIGN = float(C.SIGN_SIM)                     # signe de reference du depot
M_FLOQ = 100                                 # sous-intervalles par periode
N_PERIOD = 30                                # periodes d'iteration de puissance
AP_LO, AP_HI, N_BIS = 5.0e-6, 2.5e-3, 8      # bissection LOG sur a_p
SPEEDS = np.arange(3000, 7001, 200)          # pas 200 tr/min -> 21 vitesses
POSITIONS = tuple(np.round(np.arange(0.0, 1.001, 0.125), 3))    # 9 positions
POS_PAPER = (0.0, 0.25, 0.5, 0.75, 1.0)      # les 5 demandees explicitement
RPM_S, AP_S = 4900, 0.30e-3                  # condition S du papier
CALIBS = (('measured', F_MEASURED), ('theoretical', F_THEORETICAL))
PAPER_PEAKS = (3600, 5400)
M_STUDY = (40, 60, 100, 200, 300)            # etude de convergence a 4900

# ------------------------------------------------------- coeur de calcul ----
_A4 = {}


def a4_series(rpm, ap, hp, m, ae):
    """alpha4(t) sur une periode de dent, memoise (independant de x)."""
    key = (rpm, ap, m)
    if key not in _A4:
        _A4[key] = alpha4_series(rpm, ap, hp, m, ae=ae, midpoint=True)[1]
    return _A4[key]


def fast_period_maps(plate, rpm, ap, x_pos, sign, n_modes, m, ae):
    """Identique a closed_loop.period_maps(ctrl=None, pd=None), avec
    memoisation de expm sur la valeur de alpha4 (nulle hors engagement)."""
    tau = 60.0 / (N_TEETH * rpm)
    h = tau / m
    D = plate.D_row(x_pos, plate.hp)[:n_modes]
    DtD = np.outer(D, D)
    D_obs = plate.D_row(plate.lp, plate.hp)[:n_modes]
    H = np.asarray(plate.H_Pe_modal, float)[:n_modes]
    a4 = sign * a4_series(rpm, ap, plate.hp, m, ae)
    cache, out = {}, []
    for v in a4:
        k = float(v)
        blk = cache.get(k)
        if blk is None:
            A, At = build_matrices(plate, DtD, D_obs, H, k, None, None,
                                   n_modes)
            nx = A.shape[0]
            P0 = expm(A * h)
            J1 = np.linalg.solve(A, P0 - np.eye(nx))
            J2 = h * J1 - np.linalg.solve(A, h * P0 - J1)
            blk = cache[k] = (P0, (J1 - J2 / h) @ At, (J2 / h) @ At)
        out.append(blk)
    return out, tau


def stable(plate, rpm, ap, x_pos, sign, m=M_FLOQ, n_period=N_PERIOD):
    maps, _ = fast_period_maps(plate, rpm, ap, x_pos, sign, N_MODES, m, AE)
    return spectral_radius(maps, m, maps[0][0].shape[0], n_period) <= 1.0


def ap_limit(plate, rpm, x_pos, sign, m=M_FLOQ, lo=AP_LO, hi=AP_HI,
             n_bis=N_BIS):
    """Profondeur limite [m] par bissection LOGARITHMIQUE : resolution
    RELATIVE constante (hi/lo)^(2^-n_bis) - 1, mieux adaptee qu'une
    bissection lineaire a des limites qui vont de 0.04 a 2.5 mm."""
    if not stable(plate, rpm, lo, x_pos, sign, m):
        return 0.0
    if stable(plate, rpm, hi, x_pos, sign, m):
        return hi                            # sature : limite >= hi
    llo, lhi = np.log(lo), np.log(hi)
    for _ in range(n_bis):
        lm = 0.5 * (llo + lhi)
        if stable(plate, rpm, float(np.exp(lm)), x_pos, sign, m):
            llo = lm
        else:
            lhi = lm
    return float(np.exp(0.5 * (llo + lhi)))


# ------------------------------------------------------------- parallelisme -
_PLATES = {}


def _init():
    global _PLATES
    _PLATES = {name: build_plate('right', n_modes=N_MODES, freqs=f)
               for name, f in CALIBS}


def _job(arg):
    calib, rpm, sign, m, pos = arg
    p = _PLATES[calib]
    return arg, [ap_limit(p, rpm, fr * p.lp, sign, m) for fr in pos]


def run_jobs(jobs, n_proc=4):
    """Repartit les taches sur les coeurs ; retombe en sequentiel au besoin."""
    try:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(n_proc, initializer=_init) as pool:
            return dict(pool.map(_job, jobs, chunksize=1))
    except Exception as exc:                                # pragma: no cover
        print(f"  [pool indisponible : {exc} -> execution sequentielle]")
        _init()
        return dict(_job(j) for j in jobs)


def local_maxima(x, y):
    return [(float(x[i]), float(y[i])) for i in range(1, len(y) - 1)
            if y[i] > y[i - 1] and y[i] > y[i + 1]]


def sign_tag(s):
    return 'Eq. (13) as printed' if s > 0 else 'derived (Eqs. 1-2-5-10)'


# ------------------------------------------------------------------- main ---
def main():
    t0 = time.time()
    line = '=' * 78
    other = -SIGN
    print(line)
    print(" SCRIPT 13 — LOBES DE STABILITE SANS COMMANDE (Fig. 13 du papier)")
    print(line)
    print(f"  Modele   : Chebyshev-Ritz, {N_MODES} modes, pastille coin droit,"
          " outil sur le bord superieur")
    print(f"  Coupe    : avalant, ae = {AE * 1e3:.3f} mm, fz ="
          f" {C.FZ * 1e3:.3f} mm/dent, 3 dents, D = 10 mm, helice 35 deg")
    print(f"  Floquet  : discretisation complete, m = {M_FLOQ}"
          f" sous-intervalles/periode de dent"
          f" (C.M_FLOQUET = {C.M_FLOQUET} en production), n_period ="
          f" {N_PERIOD}")
    print(f"  Recherche: bissection LOG de a_p dans"
          f" [{AP_LO * 1e3:.3f}, {AP_HI * 1e3:.3f}] mm, {N_BIS} niveaux"
          f" -> resolution relative"
          f" {((AP_HI / AP_LO) ** (2.0 ** -N_BIS) - 1) * 100:.1f} %")
    print(f"  Grille   : {SPEEDS[0]}-{SPEEDS[-1]} tr/min, PAS"
          f" {SPEEDS[1] - SPEEDS[0]} tr/min ({len(SPEEDS)} vitesses)"
          f" x {len(POSITIONS)} positions x/l_P"
          f" ({POSITIONS[0]:.3f}..{POSITIONS[-1]:.3f} pas 0.125)"
          " x 2 calages x 2 signes")
    print(f"  SIGNE    : coeff_scale = config.SIGN_SIM = {SIGN:+.1f}"
          f"  ->  {sign_tag(SIGN)}")
    print(f"             signe oppose {other:+.1f} = {sign_tag(other)}"
          " (calcule aussi, pour la ligne de comparaison)")
    print("  NOTE     : config.SIGN_SIM a ete lu A L'EXECUTION ; l'historique"
          " du depot montre")
    print("             qu'il a bascule de -1 a +1 (commit \"Make both model"
          " layers follow Eq. (13)\").")

    # --- controles de la voie rapide ---------------------------------------
    plate_m = build_plate('right', n_modes=N_MODES, freqs=F_MEASURED)
    A, _ = fast_period_maps(plate_m, RPM_S, 0.3e-3, 0.5 * plate_m.lp, SIGN,
                            N_MODES, M_FLOQ, AE)
    B, _ = period_maps(plate_m, RPM_S, 0.3e-3, 0.5 * plate_m.lp, ctrl=None,
                       pd=None, n_modes=N_MODES, m=M_FLOQ,
                       coeff_mode='time', coeff_scale=SIGN, ae=AE)
    d = max(float(np.abs(a[i] - b[i]).max())
            for a, b in zip(A, B) for i in range(3))
    lref = repo_limit(plate_m, RPM_S, 0.5 * plate_m.lp, ctrl=None, pd=None,
                      lo=AP_LO, hi=AP_HI, tol=2e-5, n_modes=N_MODES,
                      m=M_FLOQ, coeff_mode='time', coeff_scale=SIGN,
                      n_period=N_PERIOD, ae=AE)
    lfast = ap_limit(plate_m, RPM_S, 0.5 * plate_m.lp, SIGN)
    print(f"\n  [controle 1] cartes de periode rapides vs"
          f" closed_loop.period_maps : ecart max = {d:.3e}")
    print(f"  [controle 2] a_p,lim(4900, x/l=0.5), calage mesure :"
          f" closed_loop.limit = {lref * 1e3:.4f} mm,"
          f" voie rapide = {lfast * 1e3:.4f} mm"
          f"  (ecart {abs(lfast - lref) / max(lref, 1e-12) * 100:.1f} %,"
          " du seul maillage de bissection)")

    # --- grille principale --------------------------------------------------
    jobs = [(cal, int(r), sg, M_FLOQ, POSITIONS)
            for cal, _ in CALIBS for sg in (SIGN, other) for r in SPEEDS]
    jobs += [(cal, RPM_S, SIGN, mm, POS_PAPER)
             for cal, _ in CALIBS for mm in M_STUDY]
    res = run_jobs(jobs)
    print(f"\n  [{len(jobs)} taches (vitesse x calage x signe) en"
          f" {time.time() - t0:.0f} s sur 4 coeurs]")

    grid, low, arg = {}, {}, {}
    for cal, _ in CALIBS:
        for sg in (SIGN, other):
            g = np.array([res[(cal, int(r), sg, M_FLOQ, POSITIONS)]
                          for r in SPEEDS])
            grid[(cal, sg)] = g
            low[(cal, sg)] = g.min(axis=1)
            arg[(cal, sg)] = g.argmin(axis=1)

    # --- (4) tableau principal ----------------------------------------------
    print("\n" + line)
    print(f" (4) a_p,lim MINIMALE sur le bord superieur [mm], signe"
          f" {SIGN:+.0f}   ('*' = bornee a {AP_HI * 1e3:.2f} mm)")
    print(line)
    print("   rpm |  MESURE  min   x/l_P |  THEORIQUE  min   x/l_P |"
          " rapport mes./theo.")
    print("  " + "-" * 74)
    for i, r in enumerate(SPEEDS):
        lm, lt = low[('measured', SIGN)][i], low[('theoretical', SIGN)][i]
        sm = '*' if lm >= AP_HI - 1e-12 else ' '
        st = '*' if lt >= AP_HI - 1e-12 else ' '
        print(f"  {r:5d} |    {lm * 1e3:7.4f}{sm}   {POSITIONS[arg[('measured', SIGN)][i]]:5.3f} "
              f"|     {lt * 1e3:7.4f}{st}   {POSITIONS[arg[('theoretical', SIGN)][i]]:5.3f} "
              f"|      {lm / max(lt, 1e-12):6.2f}")

    # --- (2) affirmations du papier -----------------------------------------
    print("\n" + line)
    print(" (2) AFFIRMATIONS DU PAPIER, testees sur la courbe la plus basse")
    print(line)
    print("  A1  papier : \"stability limit lower than 0.1 mm in most spindle"
          " speeds\"")
    for cal, _ in CALIBS:
        v = low[(cal, SIGN)]
        n = int((v < 0.1e-3).sum())
        print(f"      {cal:11s} : {n}/{len(SPEEDS)} vitesses sous 0.100 mm"
              f" ({100 * n / len(SPEEDS):3.0f} %) ;"
              f" mediane {np.median(v) * 1e3:.4f}, min {v.min() * 1e3:.4f},"
              f" max {v.max() * 1e3:.4f} mm")

    print("\n  A2  papier : limite \"relatively larger\" vers 3600 et"
          " 5400 tr/min")
    peaks = {}
    for cal, _ in CALIBS:
        v = low[(cal, SIGN)]
        mx = sorted(local_maxima(SPEEDS, v), key=lambda p: -p[1])
        peaks[cal] = mx
        print(f"      {cal:11s} : maxima locaux (decroissants) = "
              + ", ".join(f"{int(s)} tr/min ({a * 1e3:.3f} mm)"
                          for s, a in mx[:4]))
        for p in PAPER_PEAKS:
            near = [s for s, _ in mx if abs(s - p) <= 200]
            j = int(np.argmin(np.abs(SPEEDS - p)))
            rank = int((v > v[j]).sum()) + 1
            print(f"      {'':11s}   {p} tr/min -> a_p,lim ="
                  f" {v[j] * 1e3:.4f} mm, rang {rank}/{len(SPEEDS)} ;"
                  f" maximum local a +-200 tr/min :"
                  f" {int(near[0]) if near else 'AUCUN'}")

    # --- (3) condition S ----------------------------------------------------
    print("\n  A3  papier : condition S (4900 tr/min, a_p = 0.300 mm) DIVERGE"
          " sans commande (Fig. 14a)")
    j = int(np.argmin(np.abs(SPEEDS - RPM_S)))
    for cal, _ in CALIBS:
        ls = low[(cal, SIGN)][j]
        f = AP_S / ls if ls > 0 else np.inf
        print(f"      {cal:11s} : a_p,lim(4900) = {ls * 1e3:.4f} mm"
              f"  ->  a_p,S / a_p,lim = {f:5.2f} x"
              f"  ->  {'INSTABLE, accord avec Fig. 14a' if f > 1.0 else 'STABLE : DESACCORD avec Fig. 14a'}")
    print("      [convergence en m a 4900 tr/min, 5 positions du papier,"
          " min sur positions, mm]")
    for cal, _ in CALIBS:
        vals = [float(np.min(res[(cal, RPM_S, SIGN, mm, POS_PAPER)]))
                for mm in M_STUDY]
        print(f"      {cal:11s} : "
              + "  ".join(f"m={mm}:{v * 1e3:.4f}"
                          for mm, v in zip(M_STUDY, vals))
              + f"   (dispersion +-{50 * (max(vals) - min(vals)) / np.mean(vals):.0f} %)")

    # --- (1) signe oppose ---------------------------------------------------
    lo_o = {cal: low[(cal, other)][j] for cal, _ in CALIBS}
    fo = {cal: AP_S / max(lo_o[cal], 1e-12) for cal, _ in CALIBS}
    print(f"\n  SIGNE OPPOSE ({other:+.0f}, {sign_tag(other)}) a 4900 tr/min :"
          f" mesure {lo_o['measured'] * 1e3:.4f} mm (S = {fo['measured']:.2f} x,"
          f" {'instable' if fo['measured'] > 1 else 'STABLE -> contredit Fig. 14a'}),"
          f" theorique {lo_o['theoretical'] * 1e3:.4f} mm"
          f" (S = {fo['theoretical']:.2f} x,"
          f" {'instable' if fo['theoretical'] > 1 else 'STABLE -> contredit Fig. 14a'}).")
    for cal, _ in CALIBS:
        v = low[(cal, other)]
        mx = sorted(local_maxima(SPEEDS, v), key=lambda p: -p[1])[:3]
        print(f"      {cal:11s} signe {other:+.0f} :"
              f" {int((v < 0.1e-3).sum())}/{len(SPEEDS)} vitesses < 0.1 mm,"
              " maxima locaux = "
              + ", ".join(f"{int(s)} ({a * 1e3:.3f} mm)" for s, a in mx))

    # --- (1) surface --------------------------------------------------------
    print("\n" + line)
    print(f" (1) SURFACE a_p,lim(vitesse, position) [mm], signe {SIGN:+.0f},"
          " 5 positions du papier, 1 vitesse sur 2")
    print(line)
    idx = [POSITIONS.index(f) for f in POS_PAPER]
    for cal, _ in CALIBS:
        g = grid[(cal, SIGN)]
        print(f"  -- calage {cal} --")
        print("   rpm |" + "".join(f"  x/l={f:4.2f}" for f in POS_PAPER)
              + " |  max/min")
        for i in range(0, len(SPEEDS), 2):
            row = g[i][idx]
            print(f"  {SPEEDS[i]:5d} |"
                  + "".join(f"  {v * 1e3:7.4f}" for v in row)
                  + f" |  {row.max() / max(row.min(), 1e-12):7.2f}")
        r9 = g.max(1) / np.maximum(g.min(1), 1e-12)
        print(f"   (sur les {len(POSITIONS)} positions : ratio median"
              f" max/min = {np.median(r9):.2f}, position du minimum"
              f" = bord x/l=0 ou 1 dans"
              f" {100 * np.mean([a in (0, len(POSITIONS) - 1) for a in arg[(cal, SIGN)]]):.0f} % des cas)")

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(16.2, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.06, 1.0, 1.16], wspace=.28,
                          left=.035, right=.985, bottom=.13, top=.845)
    P = np.array(POSITIONS)
    X, Y = np.meshgrid(SPEEDS, P)

    ax = fig.add_subplot(gs[0, 0], projection='3d')
    Zm = grid[('measured', SIGN)].T * 1e3
    ax.plot_surface(X, Y, Zm, cmap='viridis', rstride=1, cstride=1,
                    edgecolor='k', linewidth=.15, antialiased=True, alpha=.96)
    ax.set_xlabel('Spindle speed (rpm)', fontsize=8, labelpad=2)
    ax.set_ylabel('Tool position $x/l_P$', fontsize=8, labelpad=0)
    ax.set_zlabel('$a_{p,lim}$ (mm)', fontsize=8, labelpad=-1)
    ax.tick_params(labelsize=7, pad=0)
    ax.view_init(elev=28, azim=-127)
    ax.set_box_aspect((1.35, 1.0, 0.75), zoom=1.06)
    ax.set_title('(a) 3D stability surface over all positions\n'
                 'measured calibration  —  paper Fig. 13(a)', fontsize=9.5)

    ax2 = fig.add_subplot(gs[0, 1])
    Zt = grid[('theoretical', SIGN)].T * 1e3
    pc = ax2.pcolormesh(SPEEDS, P, Zt, shading='nearest', cmap='viridis',
                        norm=LogNorm(vmin=max(Zt.min(), 1e-3), vmax=Zt.max()))
    ax2.contour(SPEEDS, P, Zt, levels=[0.1], colors='w', linewidths=1.3)
    ax2.contour(SPEEDS, P, Zt, levels=[0.3], colors='r', linewidths=1.3)
    ax2.axvline(RPM_S, color='r', ls='--', lw=1.3)
    ax2.annotate('condition S\n4900 rpm', (RPM_S, 0.5), xytext=(6, 0),
                 textcoords='offset points', rotation=90, ha='left',
                 va='center', color='r', fontsize=7.6, weight='bold',
                 bbox=dict(fc='w', ec='none', alpha=.8, pad=1.2))
    cb = fig.colorbar(pc, ax=ax2, pad=.02)
    cb.set_label('$a_{p,lim}$ (mm)', fontsize=9)
    cb.ax.tick_params(labelsize=7.5)
    ax2.legend(handles=[
        Line2D([], [], color='w', lw=1.6, label='$a_{p,lim}$ = 0.1 mm'),
        Line2D([], [], color='r', lw=1.6,
               label='$a_{p,lim}$ = 0.3 mm ($=a_p$ of S)')],
        fontsize=7.2, loc='lower left', framealpha=.85)
    ax2.set_xlabel('Spindle speed (rpm)')
    ax2.set_ylabel('Tool position $x/l_P$')
    ax2.set_title('(b) same map, theoretical calibration\n'
                  '(the paper computed Fig. 13 with its theoretical model)',
                  fontsize=9.5)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.semilogy(SPEEDS, low[('measured', other)] * 1e3, ':', lw=1.3,
                 color='0.5', label=f'measured, opposite sign ({other:+.0f})')
    for cal, col in (('measured', '#1a3f8f'), ('theoretical', '#c0392b')):
        ax3.semilogy(SPEEDS, low[(cal, SIGN)] * 1e3, '-o', ms=3.6, lw=1.7,
                     color=col, label=f'{cal} calibration')
    for cal, col, off in (('measured', '#1a3f8f', (0, 9)),
                          ('theoretical', '#c0392b', (13, -11))):
        for sp_, a in peaks[cal][:2]:
            ax3.annotate(f'{int(sp_)}', (sp_, a * 1e3), ha='center',
                         textcoords='offset points', xytext=off,
                         fontsize=8, color=col, weight='bold')
    ax3.axhline(0.1, color='k', ls='--', lw=1.2)
    ax3.annotate('0.1 mm — paper: limit is below this at most speeds',
                 (3050, 0.104), fontsize=7.2, color='k', va='bottom')
    for p in PAPER_PEAKS:
        ax3.axvline(p, color='0.6', ls=':', lw=1.1)
    ax3.plot([RPM_S], [AP_S * 1e3], 'r*', ms=16, mec='k', mew=.6, zorder=6,
             label='condition S (4900 rpm, $a_p$ = 0.30 mm)')
    ax3.annotate('paper: limit "relatively larger"\naround 3600 and 5400 rpm',
                 (0.50, 0.035), xycoords='axes fraction', ha='center',
                 fontsize=7.4, color='0.3')
    ax3.set_ylim(0.030, 3.0)
    ax3.set_xlabel('Spindle speed (rpm)')
    ax3.set_ylabel('lowest $a_{p,lim}$ over all positions (mm)')
    ax3.set_title('(c) lowest limit of all positions  —  paper Fig. 13(b)',
                  fontsize=9.5)
    ax3.grid(alpha=.3, which='both')
    ax3.legend(fontsize=6.9, loc='upper center', ncol=2, framealpha=.92,
               handlelength=1.8, columnspacing=1.0, borderpad=.4)

    fig.suptitle('Uncontrolled milling stability of the cantilever plate — '
                 'reproduction of Fig. 13 (down milling, $a_e$ = 0.1 mm, '
                 '5-mode Chebyshev-Ritz, full-discretisation Floquet, '
                 f'sign = {SIGN:+.0f})', fontsize=11)
    out = os.path.join(FIG, STEM + '.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\n  -> {os.path.abspath(out)}   (total {time.time() - t0:.0f} s)")


if __name__ == '__main__':
    main()
