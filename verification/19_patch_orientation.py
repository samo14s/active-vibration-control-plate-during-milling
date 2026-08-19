"""
19_patch_orientation.py — Ou est la pastille, et dans quel sens ?
==================================================================
Le papier se contredit sur le COTE (§3/§4.1 : "left lower corner" ; §5 :
"pasted in the right lower corner of the plate back") et ne dit rien du SENS :
la QDA60-20-0.7 mesure 60 x 20 mm, elle peut donc etre collee 60 mm le long de
l'arete libre x (horizontale) ou 60 mm le long de la hauteur z (verticale).
Les deux couches de ce depot avaient tranche differemment :

  * simulation/model_v2.py (elements finis) : coin bas DROIT, 60 x 20 (horizontale) ;
  * control/plate_model.py (Chebyshev)      : coin bas DROIT, 20 x 60 (verticale).

Ce script tranche par la SEULE preuve qui distingue : la signature des zeros de
la fonction de transfert tension -> deplacement, Fig. 12(b) du papier. Cette
signature ne se regle pas : le nombre de zeros dans chaque intervalle entre
poles consecutifs est fixe par les SIGNES des residus D_obs(i) * H_Pe(i), donc
par la geometrie seule.

  creux profonds digitalises, Fig. 12(b) : 788 / 1493 / 3609 Hz
  frequences propres mesurees (Tableau 4) : 540 / 1068 / 2787 / 3351 / 4122 Hz
  => occupation des quatre intervalles    : (1, 1, 0, 1)

Sortie : figures/verification/19_patch_orientation.png + tableau chiffre.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model')]

from chebyshev_plate import ChebyshevPlate                # noqa: E402

FIG = os.path.join(HERE, '..', 'figures', 'verification')
os.makedirs(FIG, exist_ok=True)

F_MEASURED = np.array([540.0, 1068.0, 2787.0, 3351.0, 4122.0])
NOTCHES = np.array([788.0, 1493.0, 3609.0])          # Fig. 12(b), digitalise
TARGET = (1, 1, 0, 1)
LP, HP = 0.100, 0.080

# quatre geometries compatibles avec le texte du papier
CASES = {
    'right, 60x20 (horizontal)': dict(x1=LP - 0.060, x2=LP, z1=0.0, z2=0.020),
    'right, 20x60 (vertical)': dict(x1=LP - 0.020, x2=LP, z1=0.0, z2=0.060),
    'left, 60x20 (horizontal)': dict(x1=0.0, x2=0.060, z1=0.0, z2=0.020),
    'left, 20x60 (vertical)': dict(x1=0.0, x2=0.020, z1=0.0, z2=0.060),
}


def zeros_of(res, f_hz):
    """Zeros exacts de G(w) = sum_i res_i / (w_i^2 - w^2), en Hz.

    Racines du numerateur de la somme mise au meme denominateur : polynome de
    degre n-1 en s = w^2. On ne cherche PAS des minima locaux de |G| (avec
    amortissement, un intervalle sans zero en montre un quand meme)."""
    s_poles = (2 * np.pi * np.asarray(f_hz, float)) ** 2
    n = len(s_poles)
    num = np.zeros(n)                                  # coefficients en s
    for i in range(n):
        p = np.array([1.0])
        for j in range(n):
            if j != i:
                p = np.convolve(p, np.array([-1.0, s_poles[j]]))
        num += res[i] * p
    r = np.roots(num)
    r = r[np.abs(r.imag) < 1e-6 * np.maximum(np.abs(r.real), 1.0)].real
    r = r[r > 0]
    return np.sort(np.sqrt(r) / (2 * np.pi))


def occupancy(zs, f_hz):
    return tuple(int(np.sum((zs > f_hz[k]) & (zs < f_hz[k + 1])))
                 for k in range(len(f_hz) - 1))


def main():
    print("=" * 84)
    print(" ORIENTATION ET COTE DE LA PASTILLE — signature des zeros,"
          " Fig. 12(b)")
    print("=" * 84)
    print(f"  cible mesuree : creux a {NOTCHES} Hz -> occupation {TARGET}")
    print(f"  poles (Tableau 4, mesure) : {F_MEASURED} Hz\n")

    out = {}
    for name, geo in CASES.items():
        p = ChebyshevPlate(PX=14, PZ=14, n_modes=5)
        p.add_piezo_patch(**geo)
        p.calibrate_frequencies(list(F_MEASURED))
        D = p.D_row(p.lp, p.hp)
        H = np.asarray(p.H_Pe_modal, float)
        res = D * H
        zs = zeros_of(res, p.freq_n)
        occ = occupancy(zs, p.freq_n)
        out[name] = dict(freq=p.freq_n.copy(), res=res, zeros=zs, occ=occ,
                         raw_f1=None, authority=float(np.sum(np.abs(res))),
                         b0=float(D @ H), signs=np.sign(res).astype(int))
        # frequence brute (avant recalage) pour information
        q = ChebyshevPlate(PX=14, PZ=14, n_modes=5)
        q.add_piezo_patch(**geo)
        out[name]['raw_f1'] = float(q.freq_n[0])
        print(f"  {name:26s} : f1 brut = {out[name]['raw_f1']:7.2f} Hz"
              f"   occupation {occ}"
              f"   {'<== CORRESPOND' if occ == TARGET else ''}")
        print(f"  {'':26s}   signes D*H = {out[name]['signs']}"
              f"   b0 = D.H = {out[name]['b0']:+.4f}"
              f"   autorite sum|D*H| = {out[name]['authority']:.4f}")
        print(f"  {'':26s}   zeros = {np.round(zs, 1)} Hz\n")

    match = [k for k, v in out.items() if v['occ'] == TARGET]
    print("  ---------------------------------------------------------------")
    print(f"  configurations reproduisant l'occupation {TARGET} :"
          f" {match if match else 'AUCUNE'}")
    if len(match) == 1:
        k = match[0]
        err = 100 * (np.array(
            [z for z in out[k]['zeros']
             if z < F_MEASURED[-1]][:3]) / NOTCHES - 1)
        print(f"  ecart des trois creux pour '{k}' : {np.round(err, 1)} %")
    print("  NB : l'occupation est une signature de SIGNES, insensible au"
          " niveau et")
    print("       a l'amortissement ; les FREQUENCES des zeros, elles,"
          " dependent")
    print("       du modele de couplage et restent approchees.")

    # ------------------------------------------------------------------ figure
    names = list(CASES)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    f = np.linspace(50, 4600, 6000)
    zeta = np.array([0.0031, 0.0017, 0.0027, 0.0056, 0.0035])
    for a, name in zip(ax.ravel(), names):
        v = out[name]
        w = 2 * np.pi * v['freq']
        om = 2 * np.pi * f
        den = (w[:, None] ** 2 - om[None, :] ** 2
               + 2j * zeta[:, None] * w[:, None] * om[None, :])
        G = np.sum(v['res'][:, None] / den, axis=0)
        a.plot(f, 20 * np.log10(np.abs(G) / 1e-8 + 1e-30), lw=1.2,
               color='#1a3f8f')
        for fz in v['freq']:
            a.axvline(fz, color='0.75', lw=.8)
        for z in v['zeros']:
            if z < f[-1]:
                a.axvline(z, color='k', ls='-.', lw=1)
        for nt in NOTCHES:
            a.axvline(nt, color='#16a085', ls=':', lw=1.6)
        ok = v['occ'] == TARGET
        a.set_title(f"{name}\nmodel occupancy {v['occ']} vs measured {TARGET}"
                    f"  {'MATCH' if ok else 'no'}", fontsize=9.5,
                    color=('#0b6b52' if ok else '#8b2f2f'))
        a.set_xlabel('frequency [Hz]')
        a.set_ylabel('|G| [dB re 0.01 $\\mu$m/V]')
        a.grid(alpha=.3)
    ax[0, 0].plot([], [], color='0.75', lw=.8, label='natural frequency')
    ax[0, 0].plot([], [], color='k', ls='-.', lw=1, label='exact model zero')
    ax[0, 0].plot([], [], color='#16a085', ls=':', lw=1.6,
                  label='digitized notch, Fig. 12(b)')
    ax[0, 0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Which patch geometry does the paper describe? '
                 'The zero fingerprint of Fig. 12(b) is not tunable',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, '19_patch_orientation.png'), dpi=140)
    plt.close(fig)
    print(f"\n  -> {FIG}/19_patch_orientation.png")


if __name__ == '__main__':
    main()
