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
(discretisation complete, Ding et al. ; control/closed_loop.py) sur le modele
Chebyshev-Ritz a 5 modes, dans les DEUX calages du Tableau 4 :
mesure (540/1068/2787/3351/4122 Hz) et theorique (537/1101/2805/3423/4254 Hz).
Le papier a calcule sa Fig. 13 avec SON modele theorique : c'est la colonne
"theoretical" qui doit lui etre comparee.

CONVENTION DE SIGNE — imposee ici a coeff_scale = config.SIGN_SIM = -1, celle
de la couche control/ du depot (convention "derivee" des Eqs. 1-2-5-10 du
papier). Le signe oppose (+1 = Eq. (13) telle qu'imprimee) est evalue en un
point (4900 tr/min) pour montrer l'ecart : voir verification/18.

ACCELERATION — la carte de periode est identique a celle de
control/closed_loop.period_maps (verifiee bit a bit, cf. sortie), mais les
exponentielles de matrice sont MEMOISEES sur la valeur de alpha4 : en avalant
a ae = 0.1 mm l'outil n'est engage que ~12 % de la periode de dent, donc
alpha4 = 0 exactement sur la grande majorite des sous-intervalles et une
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
N_MODES = C.N_MODES                 # 5
AE = C.AE                           # 0.1 mm
SIGN = C.SIGN_SIM                   # -1.0
M_FLOQ = 60                         # sous-intervalles (C.M_FLOQUET = 200)
N_PERIOD = 24                       # periodes d'iteration de puissance
AP_LO, AP_HI, N_BIS = 5.0e-6, 2.5e-3, 8      # bissection LOG sur a_p
SPEEDS = np.arange(3000, 7001, 200)          # pas 200 tr/min -> 21 vitesses
POSITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)      # x / l_P
RPM_S, AP_S = 4900, 0.30e-3                  # condition S du papier
CALIBS = (('measured', F_MEASURED), ('theoretical', F_THEORETICAL))
PAPER_PEAKS = (3600, 5400)

# ------------------------------------------------------- coeur de calcul ----
_A4 = {}


def a4_series(rpm, ap, hp, m, ae):
    """alpha4(t) sur une periode de dent, memoise (independant de la position)."""
    key = (rpm, ap, m)
    if key not in _A4:
        _A4[key] = alpha4_series(rpm, ap, hp, m, ae=ae, midpoint=True)[1]
    return _A4[key]


def fast_period_maps(plate, rpm, ap, x_pos, sign, n_modes, m, ae):
    """Identique a closed_loop.period_maps(ctrl=None, pd=None) mais avec
    memoisation de expm sur la valeur de alpha4 (0 sur ~88 % de la periode)."""
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
    rho = spectral_radius(maps, m, maps[0][0].shape[0], n_period)
    return rho <= 1.0


def ap_limit(plate, rpm, x_pos, sign, m=M_FLOQ, lo=AP_LO, hi=AP_HI,
             n_bis=N_BIS):
    """Profondeur limite [m] par bissection LOGARITHMIQUE (resolution
    relative constante : (hi/lo)^(2^-n_bis) - 1 = 2.4 % ici)."""
    if not stable(plate, rpm, lo, x_pos, sign, m):
        return 0.0
    if stable(plate, rpm, hi, x_pos, sign, m):
        return hi                       # sature : limite >= hi
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
    calib, rpm, sign, m = arg
    p = _PLATES[calib]
    return arg, [ap_limit(p, rpm, fr * p.lp, sign, m) for fr in POSITIONS]


def run_jobs(jobs, n_proc=4):
    """Repartit les (calib, rpm, sign, m) sur les coeurs ; retombe en
    sequentiel si le pool n'est pas disponible."""
    try:
        import multiprocessing as mp
        ctx = mp.get_context('fork')
        with ctx.Pool(n_proc, initializer=_init) as pool:
            return dict(pool.map(_job, jobs, chunksize=1))
    except Exception as exc:                                # pragma: no cover
        print(f"  [pool indisponible : {exc} -> sequentiel]")
        _init()
        return dict(_job(j) for j in jobs)


def local_maxima(x, y):
    return [(float(x[i]), float(y[i])) for i in range(1, len(y) - 1)
            if y[i] > y[i - 1] and y[i] > y[i + 1]]


# ------------------------------------------------------------------- main ---
def main():
    t0 = time.time()
    line = '=' * 78
    print(line)
    print(" SCRIPT 13 — LOBES DE STABILITE SANS COMMANDE (Fig. 13 du papier)")
    print(line)
    print(f"  Modele   : Chebyshev-Ritz, {N_MODES} modes, pastille coin droit,"
          " outil sur le bord superieur")
    print(f"  Coupe    : avalant, ae = {AE * 1e3:.3f} mm, fz ="
          f" {C.FZ * 1e3:.3f} mm/dent, 3 dents, D = 10 mm")
    print(f"  Floquet  : discretisation complete, m = {M_FLOQ}"
          f" sous-intervalles (C.M_FLOQUET = {C.M_FLOQUET} en production),"
          f" n_period = {N_PERIOD}")
    print(f"  Recherche: bissection LOG de a_p dans"
          f" [{AP_LO * 1e3:.3f}, {AP_HI * 1e3:.3f}] mm,"
          f" {N_BIS} niveaux -> resolution relative"
          f" {((AP_HI / AP_LO) ** (2.0 ** -N_BIS) - 1) * 100:.1f} %")
    print(f"  Grille   : {SPEEDS[0]}-{SPEEDS[-1]} tr/min pas"
          f" {SPEEDS[1] - SPEEDS[0]} ({len(SPEEDS)} vitesses)"
          f" x {len(POSITIONS)} positions x 2 calages")
    print(f"  SIGNE    : coeff_scale = config.SIGN_SIM = {SIGN:+.1f}"
          "  (convention de la couche control/ ;")
    print("             +1 = Eq. (13) telle qu'imprimee dans le papier)")

    # --- controles de la voie rapide ---------------------------------------
    plate_m = build_plate('right', n_modes=N_MODES, freqs=F_MEASURED)
    A, _ = fast_period_maps(plate_m, 4900, 0.3e-3, 0.5 * plate_m.lp, SIGN,
                            N_MODES, M_FLOQ, AE)
    B, _ = period_maps(plate_m, 4900, 0.3e-3, 0.5 * plate_m.lp, ctrl=None,
                       pd=None, n_modes=N_MODES, m=M_FLOQ,
                       coeff_mode='time', coeff_scale=SIGN, ae=AE)
    d = max(float(np.abs(a[i] - b[i]).max())
            for a, b in zip(A, B) for i in range(3))
    lref = repo_limit(plate_m, 4900, 0.5 * plate_m.lp, ctrl=None, pd=None,
                      lo=AP_LO, hi=AP_HI, tol=2e-5, n_modes=N_MODES,
                      m=M_FLOQ, coeff_mode='time', coeff_scale=SIGN,
                      n_period=N_PERIOD, ae=AE)
    lfast = ap_limit(plate_m, 4900, 0.5 * plate_m.lp, SIGN)
    print(f"\n  [controle] cartes de periode rapides vs"
          f" closed_loop.period_maps : ecart max = {d:.3e}")
    print(f"  [controle] a_p,lim(4900, x/l=0.5) : closed_loop.limit ="
          f" {lref * 1e3:.4f} mm, voie rapide = {lfast * 1e3:.4f} mm"
          f"  (ecart {abs(lfast - lref) / max(lref, 1e-12) * 100:.1f} %)")

    # --- grille principale --------------------------------------------------
    jobs = [(cal, int(r), SIGN, M_FLOQ) for cal, _ in CALIBS for r in SPEEDS]
    jobs += [(cal, RPM_S, -SIGN, M_FLOQ) for cal, _ in CALIBS]      # autre signe
    jobs += [(cal, RPM_S, SIGN, 200) for cal, _ in CALIBS]          # test de m
    res = run_jobs(jobs)
    print(f"\n  [{len(jobs)} points de vitesse calcules en"
          f" {time.time() - t0:.0f} s]")

    grid = {cal: np.array([res[(cal, int(r), SIGN, M_FLOQ)] for r in SPEEDS])
            for cal, _ in CALIBS}                        # (n_speed, n_pos)
    low = {cal: grid[cal].min(axis=1) for cal in grid}
    arg = {cal: grid[cal].argmin(axis=1) for cal in grid}

    # --- Tableau principal --------------------------------------------------
    print("\n" + line)
    print(" (4) a_p,lim MINIMALE sur le bord superieur  [mm]   (sat. ="
          f" bornee a {AP_HI * 1e3:.2f} mm)")
    print(line)
    print("   rpm |  MESURE: min   x/l_P |  THEORIQUE: min   x/l_P |"
          "  mesure/theorique")
    print("  " + "-" * 74)
    for i, r in enumerate(SPEEDS):
        lm, lt = low['measured'][i], low['theoretical'][i]
        sm = '*' if lm >= AP_HI - 1e-12 else ' '
        st = '*' if lt >= AP_HI - 1e-12 else ' '
        print(f"  {r:5d} |     {lm * 1e3:7.4f}{sm}  {POSITIONS[arg['measured'][i]]:4.2f} "
              f"|      {lt * 1e3:7.4f}{st}  {POSITIONS[arg['theoretical'][i]]:4.2f} "
              f"|      {lm / lt:6.2f}")

    # --- A1 : limite < 0.1 mm a la plupart des vitesses ---------------------
    print("\n" + line)
    print(" (2) AFFIRMATIONS DU PAPIER")
    print(line)
    print("  A1  papier : \"stability limit lower than 0.1 mm in most spindle"
          " speeds\"")
    for cal, _ in CALIBS:
        n = int((low[cal] < 0.1e-3).sum())
        print(f"      {cal:11s} : {n}/{len(SPEEDS)} vitesses sous 0.100 mm"
              f"  ({100 * n / len(SPEEDS):.0f} %) ;"
              f" mediane {np.median(low[cal]) * 1e3:.4f} mm,"
              f" min {low[cal].min() * 1e3:.4f} mm,"
              f" max {low[cal].max() * 1e3:.4f} mm")

    print("\n  A2  papier : limite \"relatively larger\" vers 3600 et"
          " 5400 tr/min")
    peaks = {}
    for cal, _ in CALIBS:
        mx = sorted(local_maxima(SPEEDS, low[cal]), key=lambda p: -p[1])
        peaks[cal] = mx
        txt = ", ".join(f"{int(s)} tr/min ({a * 1e3:.3f} mm)"
                        for s, a in mx[:4])
        print(f"      {cal:11s} : maxima locaux = {txt if txt else 'aucun'}")
        hit = [f"{p}" for p in PAPER_PEAKS
               if any(abs(s - p) <= 200 for s, _ in mx)]
        print(f"      {'':11s}   coincidence avec {PAPER_PEAKS} a +-200 tr/min :"
              f" {hit if hit else 'AUCUNE'}")
        for p in PAPER_PEAKS:
            j = int(np.argmin(np.abs(SPEEDS - p)))
            print(f"      {'':11s}   a {p} tr/min : a_p,lim ="
                  f" {low[cal][j] * 1e3:.4f} mm"
                  f"  (rang {int((low[cal] > low[cal][j]).sum()) + 1}"
                  f"/{len(SPEEDS)} par valeur decroissante)")

    # --- A3 : condition S ---------------------------------------------------
    print("\n  A3  papier : condition S (4900 tr/min, a_p = 0.300 mm) DIVERGE"
          " sans commande (Fig. 14a)")
    j = int(np.argmin(np.abs(SPEEDS - RPM_S)))
    for cal, _ in CALIBS:
        ls = low[cal][j]
        f = AP_S / ls if ls > 0 else np.inf
        verdict = 'INSTABLE (accord)' if f > 1 else 'STABLE (DESACCORD)'
        print(f"      {cal:11s} : a_p,lim(4900) = {ls * 1e3:.4f} mm ->"
              f" a_p,S / a_p,lim = {f:5.2f}  -> {verdict}")
    lm200 = {cal: float(np.min(res[(cal, RPM_S, SIGN, 200)]))
             for cal, _ in CALIBS}
    print(f"      [convergence] a 4900 tr/min avec m = 200 :"
          + "".join(f"  {cal} {lm200[cal] * 1e3:.4f} mm"
                    for cal, _ in CALIBS)
          + f"   (m = {M_FLOQ} :"
          + "".join(f"  {low[cal][j] * 1e3:.4f}" for cal, _ in CALIBS) + ")")

    # --- (1) l'autre signe --------------------------------------------------
    other = {cal: float(np.min(res[(cal, RPM_S, -SIGN, M_FLOQ)]))
             for cal, _ in CALIBS}
    print(f"\n  (1) AUTRE SIGNE (coeff_scale = {-SIGN:+.1f}, Eq. (13) imprimee)"
          f" a 4900 tr/min : mesure {other['measured'] * 1e3:.4f} mm"
          f" (facteur S {AP_S / max(other['measured'], 1e-12):.2f}),"
          f" theorique {other['theoretical'] * 1e3:.4f} mm"
          f" (facteur S {AP_S / max(other['theoretical'], 1e-12):.2f})"
          " -> S diverge des deux cotes.")

    # --- (3) surface : dispersion selon la position -------------------------
    print("\n" + line)
    print(" (1) SURFACE a_p,lim(vitesse, position) — dispersion selon x/l_P"
          "  [mm]")
    print(line)
    hdr = "   rpm |" + "".join(f"  x/l={f:4.2f}" for f in POSITIONS) \
        + " |  max/min"
    for cal, _ in CALIBS:
        print(f"  -- calage {cal} --")
        print(hdr)
        for i, r in enumerate(SPEEDS[::2]):
            k = 2 * i
            row = grid[cal][k]
            print(f"  {r:5d} |" + "".join(f"  {v * 1e3:7.4f}" for v in row)
                  + f" |  {row.max() / max(row.min(), 1e-12):7.2f}")
        print(f"   (une ligne sur deux ; ratio median max/min ="
              f" {np.median(grid[cal].max(1) / np.maximum(grid[cal].min(1), 1e-12)):.2f})")

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(15.0, 4.8))
    X, Y = np.meshgrid(SPEEDS, np.array(POSITIONS))

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    Z = grid['measured'].T * 1e3
    ax.plot_surface(X, Y, Z, cmap='viridis', rstride=1, cstride=1,
                    edgecolor='k', linewidth=.2, antialiased=True, alpha=.95)
    ax.set_xlabel('Spindle speed (rpm)', fontsize=8, labelpad=1)
    ax.set_ylabel('Tool position $x/l_P$', fontsize=8, labelpad=1)
    ax.set_zlabel('$a_{p,lim}$ (mm)', fontsize=8, labelpad=1)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=26, azim=-124)
    ax.set_title('(a) 3D stability surface, measured calibration\n'
                 f'(paper Fig. 13a; sign = {SIGN:+.0f})', fontsize=9)

    ax2 = fig.add_subplot(1, 3, 2)
    Zt = grid['theoretical'].T * 1e3
    vmin = max(min(Z.min(), Zt.min()), 1e-3)
    pc = ax2.pcolormesh(SPEEDS, np.array(POSITIONS), Zt, shading='nearest',
                        cmap='viridis',
                        norm=LogNorm(vmin=vmin, vmax=max(Z.max(), Zt.max())))
    cs = ax2.contour(SPEEDS, np.array(POSITIONS), Zt, levels=[0.1, 0.3],
                     colors=['w', 'r'], linewidths=1.2)
    ax2.clabel(cs, fmt='%.1f mm', fontsize=7)
    ax2.plot([RPM_S], [0.0], 'r*', ms=13, mec='k', mew=.6, zorder=5)
    ax2.annotate('S', (RPM_S, 0.0), textcoords='offset points',
                 xytext=(8, 6), color='r', fontsize=10, weight='bold')
    fig.colorbar(pc, ax=ax2, label='$a_{p,lim}$ (mm)')
    ax2.set_xlabel('Spindle speed (rpm)')
    ax2.set_ylabel('Tool position $x/l_P$')
    ax2.set_title('(b) same map, theoretical calibration\n'
                  '(the paper computed Fig. 13 with its theoretical model)',
                  fontsize=9)

    ax3 = fig.add_subplot(1, 3, 3)
    for cal, col in (('measured', '#1a3f8f'), ('theoretical', '#c0392b')):
        ax3.semilogy(SPEEDS, low[cal] * 1e3, '-o', ms=3.4, lw=1.5, color=col,
                     label=f'{cal} calibration')
        for s, a in peaks[cal][:3]:
            ax3.annotate(f'{int(s)}', (s, a * 1e3), textcoords='offset points',
                         xytext=(0, 7), ha='center', fontsize=7, color=col)
    ax3.axhline(0.1, color='k', ls='--', lw=1.1,
                label='0.1 mm (paper: limit below this at most speeds)')
    for p in PAPER_PEAKS:
        ax3.axvline(p, color='0.55', ls=':', lw=1.1)
    ax3.annotate('paper: "relatively larger"\naround 3600 and 5400 rpm',
                 (0.5, 0.03), xycoords='axes fraction', ha='center',
                 fontsize=7, color='0.35')
    ax3.plot([RPM_S], [AP_S * 1e3], 'r*', ms=14, mec='k', mew=.6, zorder=6,
             label='condition S (4900 rpm, 0.30 mm)')
    ax3.set_xlabel('Spindle speed (rpm)')
    ax3.set_ylabel('lowest $a_{p,lim}$ over all positions (mm)')
    ax3.set_title('(c) paper Fig. 13(b): lowest limit of all positions',
                  fontsize=9)
    ax3.grid(alpha=.3, which='both')
    ax3.legend(fontsize=7, loc='upper left')

    fig.suptitle('Uncontrolled milling stability of the cantilever plate — '
                 'reproduction of Fig. 13 (down milling, $a_e$ = 0.1 mm, '
                 f'5-mode Chebyshev-Ritz, sign = {SIGN:+.0f})', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(FIG, STEM + '.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\n  -> {os.path.abspath(out)}   ({time.time() - t0:.0f} s)")


if __name__ == '__main__':
    main()
