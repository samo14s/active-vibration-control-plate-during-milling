"""
test_invariants.py — les identites sur lesquelles tout le reste repose
=======================================================================
Rejouer les scripts de verification prouve qu'ils s'executent ; cela ne prouve
pas que les proprietes MATHEMATIQUES etablies au fil de ce travail tiennent
encore. Ce fichier teste ces proprietes-la, celles dont la fausseté
invaliderait des conclusions entieres, et il le fait en quelques secondes.

Chaque test correspond a un resultat annonce dans les rapports, avec la
tolerance qui y est citee. Si l'un d'eux casse, ce n'est pas un detail
d'implementation : c'est une conclusion du rapport qui ne tient plus.

    python -m pytest tests/ -q          (ou : python tests/test_invariants.py)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path[:0] = [os.path.join(ROOT, 'paper_model'), os.path.join(ROOT, 'control')]

import config as C                                              # noqa: E402
from plate_model import build_plate, plant_vectors, plant_frf    # noqa: E402
from fopid import ss_frf, fopid_ss                               # noqa: E402
from adrc import adrc_fopid_ss                                   # noqa: E402
from fdob import fdob_fopid_ss, target_modes                     # noqa: E402
from diagnose_adrc import plant_zeros                            # noqa: E402
from chatter_estimator import Biquad, notch, bandpass            # noqa: E402

OM = 2 * np.pi * np.logspace(0, 4.4, 2000)
PAR = dict(Kp=2.516e4, Ki=1.045e7, Kd=6.065, lam=0.2677, mu=0.1322)


def _plate():
    return build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)


# --------------------------------------------------------------- procede
def test_rhp_zero_exists():
    """Le procede complet a UN zero dans le demi-plan droit, vers 2459 Hz.

    C'est le fait dont decoulent le plafond de Poisson (DIAGNOSTIC §6),
    l'obligation d'inverser mode par mode (OBSERVATEUR §2.2) et la defaite
    de la variante a cinq modes (OBSERVATEUR §6.7).
    """
    plate = _plate()
    w, z, H, D_obs, _ = plant_vectors(plate, C.N_MODES)
    zr = plant_zeros(w, z, D_obs * H)
    rhp = zr[zr.real > 1e-9]
    assert len(rhp) == 1, f'attendu 1 zero instable, obtenu {len(rhp)}'
    f = abs(rhp[0]) / (2 * np.pi)
    assert 2400 < f < 2520, f'zero instable a {f:.0f} Hz, attendu ~2459'


def test_reduced_model_is_minimum_phase():
    """Les modeles reduits a 2 et 3 modes n'ont AUCUN zero instable.

    Sans cela l'inversion modale de l'observateur serait impossible : c'est
    la justification de OBSERVATEUR §2.2.
    """
    plate = _plate()
    w, z, H, D_obs, _ = plant_vectors(plate, C.N_MODES)
    res = D_obs * H
    for n in (2, 3):
        zr = plant_zeros(w[:n], z[:n], res[:n])
        assert not (zr.real > 1e-9).any(), \
            f'modele a {n} modes : zero instable trouve'


def test_residues_change_sign_at_mode_four():
    """r = D_obs.H_Pe change de signe entre les modes 3 et 4.

    C'est l'origine geometrique du zero instable, et la raison pour laquelle
    b0 — un scalaire unique — ne peut pas suivre le signe de la voie.
    """
    plate = _plate()
    _, _, H, D_obs, _ = plant_vectors(plate, C.N_MODES)
    r = D_obs * H
    assert (r[:3] < 0).all() and (r[3:] > 0).all(), f'residus {np.round(r, 4)}'


# ------------------------------------------------------------------ ADRC
def test_leso_identity():
    """z3(s) = Q(s) [s^2 y - b0 u] EXACTEMENT (DIAGNOSTIC §2, 7.9e-15).

    Le maillon 2 de la chaine de perte supposee est INFIRME par cette
    identite : l'observateur etendu n'est pas en faute.
    """
    plate = _plate()
    _, _, _, _, sl = plant_vectors(plate, C.N_MODES)
    b0, wo = -40.72, 1.863e4
    core = adrc_fopid_ss(PAR['Kp'], PAR['Ki'], PAR['Kd'], PAR['lam'],
                         PAR['mu'], wo, b0, C.OUST_WB, C.OUST_WH, C.OUST_N,
                         1.0)
    A, B = np.atleast_2d(core[0]), np.atleast_2d(core[1]).reshape(-1, 1)
    n = A.shape[0]
    e3 = np.zeros((1, n))
    e3[0, 2] = 1.0
    Z3 = np.array([(e3 @ np.linalg.solve(1j * w * np.eye(n) - A, B))[0, 0]
                   for w in OM])
    Q = (wo / (1j * OM + wo)) ** 3
    pred = Q * ((1j * OM) ** 2 - b0 * ss_frf(core, OM))
    err = float(np.max(np.abs(Z3 - pred) / np.abs(pred)))
    assert err < 1e-11, f'identite du LESO violee : {err:.2e}'


def test_b0_is_a_global_gain():
    """K(s) = G(s)/b0 : doubler b0 divise |K| par 2 PARTOUT (DIAGNOSTIC §3).

    Maillon 3 INFIRME : b0 n'est pas un parametre de modele.
    """
    plate = _plate()
    b0, wo = -40.72, 1.863e4
    k = [ss_frf(adrc_fopid_ss(PAR['Kp'], PAR['Ki'], PAR['Kd'], PAR['lam'],
                              PAR['mu'], wo, s * b0, C.OUST_WB, C.OUST_WH,
                              C.OUST_N, 1.0), OM) for s in (1.0, 2.0)]
    ratio = np.abs(k[1]) / np.abs(k[0])
    assert np.allclose(ratio, 0.5, rtol=1e-6), \
        f'rapport hors tolerance : {ratio.min():.6f} a {ratio.max():.6f}'


# ------------------------------------------------------------------ FDOB
def test_fdob_is_a_superset_of_fopid():
    """A alpha = 0 la structure redonne le FOPID EXACTEMENT.

    C'est la propriete que l'ADRC-FOPID n'a pas (son ensemble est DECALE, pas
    plus grand). Si elle casse, OBSERVATEUR §2.4 tombe.
    """
    plate = _plate()
    _, _, _, _, sl = plant_vectors(plate, C.N_MODES)
    w, z, r = target_modes(plate, (0, 1))
    K0 = ss_frf(fdob_fopid_ss(**PAR, zeta_q=0.2, alpha=0.0, w=w, zeta=z,
                              res=r, wc=C.FDOB_WC, wb=C.OUST_WB,
                              wh=C.OUST_WH, N=C.OUST_N, sign_loop=sl), OM)
    Kf = ss_frf(fopid_ss(PAR['Kp'], PAR['Ki'], PAR['Kd'], PAR['lam'],
                         PAR['mu'], C.OUST_WB, C.OUST_WH, C.OUST_N, sl), OM)
    # l'epine dorsale agit sur e = -y, d'ou le signe
    err = float(np.max(np.abs(K0 + Kf) / np.abs(Kf)))
    assert err < 1e-12, f'alpha = 0 ne redonne pas le FOPID : {err:.2e}'


def test_fdob_closed_form():
    """K = -(C + alpha W)/(1 - alpha V) contre la realisation (3e-14)."""
    plate = _plate()
    _, _, _, _, sl = plant_vectors(plate, C.N_MODES)
    w, z, r = target_modes(plate, (0, 1))
    zq, al = 0.05, 0.5
    K = ss_frf(fdob_fopid_ss(**PAR, zeta_q=zq, alpha=al, w=w, zeta=z, res=r,
                             wc=C.FDOB_WC, wb=C.OUST_WB, wh=C.OUST_WH,
                             N=C.OUST_N, sign_loop=sl), OM)
    Cs = ss_frf(fopid_ss(PAR['Kp'], PAR['Ki'], PAR['Kd'], PAR['lam'],
                         PAR['mu'], C.OUST_WB, C.OUST_WH, C.OUST_N, sl), OM)
    s = 1j * OM
    V = np.zeros_like(s)
    W = np.zeros_like(s)
    for wk, zk, rk in zip(w, z, r):
        q = (2 * zq * wk * s) / (s ** 2 + 2 * zq * wk * s + wk ** 2) \
            * (C.FDOB_WC / (s + C.FDOB_WC)) ** 2
        V = V + q
        W = W + q * (s ** 2 + 2 * zk * wk * s + wk ** 2) / rk
    err = float(np.max(np.abs(-(Cs + al * W) / (1 - al * V) - K)
                       / np.abs(K)))
    assert err < 1e-11, f'forme fermee du FDOB : {err:.2e}'


def test_modal_nominal_model_defect_is_small():
    """Le defaut |P/Pn - 1| est ~1e-3 aux modes vises, contre 2.5 et 7.1 pour
    le double integrateur de l'ADRC (OBSERVATEUR §2.1)."""
    plate = _plate()
    f = np.logspace(2, 3.5, 20000)
    P, _ = plant_frf(plate, f, C.N_MODES)
    w, z, r = target_modes(plate, (0, 1))
    om = 2 * np.pi * f
    Pn = np.zeros_like(P)
    for wk, zk, rk in zip(w, z, r):
        Pn = Pn + rk / (wk ** 2 - om ** 2 + 2j * zk * wk * om)
    for wk in w:
        i = int(np.argmin(np.abs(om - wk)))
        d = abs(P[i] / Pn[i] - 1.0)
        assert d < 0.02, f'defaut modal {d:.4f} a {wk / 2 / np.pi:.0f} Hz'


# ------------------------------------------------------- estimateur
def test_comb_rejects_tooth_harmonics():
    """Le peigne pre-distordu attenue les harmoniques de dent sous 1 % tout en
    laissant passer les modes au-dessus de 80 % (ESTIMATEUR, en-tete).

    Sans la pre-distorsion bilineaire l'attenuation a 490 Hz remonte a 0.30 et
    le module ne sert plus a rien : c'est ce que ce test protege.
    """
    fs = 160.7e3 / 8
    combs = [Biquad(*notch(k * 245.0, 0.005, fs)) for k in (1, 2, 3, 4, 5)]
    bp = Biquad(*bandpass(400.0, 1300.0, fs))
    t = np.arange(0, 0.5, 1 / fs)
    for f0, kind in ((490, 'harm'), (735, 'harm'), (980, 'harm'),
                     (540, 'mode'), (1068, 'mode')):
        for b in combs:
            b.reset()
        bp.reset()
        x = np.sin(2 * np.pi * f0 * t)
        out = []
        for v in x:
            for b in combs:
                v = b(v)
            out.append(bp(v))
        g = np.asarray(out)[len(t) // 2:].std() / x.std()
        if kind == 'harm':
            assert g < 0.01, f'harmonique {f0} Hz mal rejetee : gain {g:.4f}'
        else:
            assert g > 0.80, f'mode {f0} Hz trop attenue : gain {g:.4f}'


def _main():
    fs = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    bad = 0
    for f in fs:
        try:
            f()
            print(f'  OK    {f.__name__}')
        except AssertionError as e:
            bad += 1
            print(f'  ECHEC {f.__name__} : {e}')
    print(f'\n  {len(fs) - bad}/{len(fs)} invariants verifies')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(_main())
