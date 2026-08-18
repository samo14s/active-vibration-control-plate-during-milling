"""schedule_im.py — Article 2, section 6 : ordonnancement du modele interne
resonnant sur le point de fonctionnement.

Le modele interne du R-ESO doit etre place aux frequences de broutement que le
procede produirait reellement. Celles-ci ne sont pas les frequences propres :
elles resultent de la condition de phase du systeme retarde et se deduisent du
MULTIPLICATEUR DE FLOQUET dominant, calcule en boucle ouverte juste au-dessus
de la limite de coupe :

    f = | +/- f_pv + j / tau | ,   tau = 60 / (N_t * rpm)

Parmi cette famille on retient le representant le plus proche de chaque mode
retenu. La chaine analyse de stabilite -> frequences -> modele interne ferme la
boucle entre l'outil de prediction et le correcteur."""
import json
import numpy as np

from chebyshev_plate import ChebyshevPlate
from stability_fdm import is_stable as ol_is_stable, stability_limit
from milling_dynamics import N_TEETH

F_MEAS = [540, 1068, 2787, 3351, 4122]


def chatter_frequencies(plate, rpm, x_pos, ap=None, n_modes=2, sign=1.0,
                        kmax=8):
    """(f_c pres du mode 1, f_c pres du mode 2) en Hz, et la limite utilisee."""
    if ap is None:
        ap = 1.15 * stability_limit(plate, rpm, x_pos, hi=3e-3, tol=2e-5,
                                    coeff_mode='time', coeff_scale=sign)
        ap = max(ap, 2e-5)
    _, f_pv, _ = ol_is_stable(plate, rpm, ap, x_pos, coeff_mode='time',
                              coeff_scale=sign, n_modes=n_modes,
                              return_freq=True)
    inv_tau = N_TEETH * rpm / 60.0
    cand = sorted({round(abs(s * f_pv + j * inv_tau), 2)
                   for s in (1, -1) for j in range(0, kmax + 1)})
    out = []
    for k in range(n_modes):
        fk = plate.omega_n[k] / (2 * np.pi)
        out.append(min(cand, key=lambda c: abs(c - fk)))
    return out, ap


if __name__ == '__main__':
    p = ChebyshevPlate(PX=14, PZ=14)
    p.add_piezo_patch()
    p.calibrate_frequencies(F_MEAS)
    EXP = [4300, 4900, 5500, 6100, 6700]
    POS = [0.0, 0.025, 0.050]
    table = {}
    print("frequences de broutement predites (Hz) — modes propres "
          f"{F_MEAS[0]} et {F_MEAS[1]} Hz")
    print("  rpm   position   f_c1     f_c2     (limite BO utilisee, mm)")
    for r in EXP:
        rows = []
        for x in POS:
            fcs, ap = chatter_frequencies(p, r, x)
            rows.append(fcs)
            print(f"  {r}   x={x*1e3:4.0f}mm  {fcs[0]:7.1f}  {fcs[1]:7.1f}"
                  f"   ({ap*1e3:.3f})")
        arr = np.array(rows)
        table[str(r)] = dict(par_position=arr.tolist(),
                             moyenne=[round(float(v), 1) for v in arr.mean(0)],
                             etendue=[round(float(v), 1)
                                      for v in (arr.max(0) - arr.min(0))])
    json.dump(table, open('results/art2_chatter_freqs.json', 'w'), indent=1)
    print("\nmoyenne sur les positions :")
    for r, d in table.items():
        print(f"  {r} rpm : f_c = {d['moyenne']} Hz   (etendue {d['etendue']})")
