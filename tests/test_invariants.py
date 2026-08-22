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

    python tests/test_invariants.py     (ou : python -m pytest tests/ -q)
"""
import os
import sys

import numpy as np
from scipy.linalg import expm as _expm

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
from lti_floquet import dominant_eigs                            # noqa: E402
from step_integrals import step_integrals                        # noqa: E402

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
    # Sans cette garde, (r[3:] > 0).all() serait vrai par vacuite si le
    # modele tombait a trois modes, et l'invariant cesserait silencieusement
    # d'etre verifie.
    assert len(r) >= 5, f'invariant non testable : {len(r)} modes seulement'
    assert (r[:3] < 0).all() and (r[3:] > 0).all(), f'residus {np.round(r, 4)}'


# ------------------------------------------------------------------ ADRC
def test_leso_identity():  # noqa: D401
    """z3(s) = Q(s) [s^2 y - b0 u] EXACTEMENT (DIAGNOSTIC §2, 7.9e-15).

    Le maillon 2 de la chaine de perte supposee est INFIRME par cette
    identite : l'observateur etendu n'est pas en faute.
    """
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


# -------------------------------------------------------------- Floquet
def _random_maps(seed, nx=6, n_sub=3):
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal((nx, nx)) * 0.35,
             rng.standard_normal((nx, nx)) * 0.05,
             rng.standard_normal((nx, nx)) * 0.05) for _ in range(n_sub)]


def _exact_monodromy(maps, m, nx):
    """La monodromie ASSEMBLEE, colonne par colonne, de la meme recurrence."""
    dim = (m + 1) * nx
    Z = np.zeros((m + 1, nx, dim))
    for j in range(m + 1):
        Z[j, :, j * nx:(j + 1) * nx] = np.eye(nx)
    for P0, C_lo, C_hi in maps:
        new = P0 @ Z[0] + C_lo @ Z[m] + C_hi @ Z[m - 1]
        Z = np.roll(Z, 1, axis=0)
        Z[0] = new
    return Z.reshape(dim, dim)


def test_dominant_eigs_matches_the_assembled_monodromy():
    """L'iteration de sous-espace rend le multiplicateur dominant EXACT.

    C'est le multiplicateur dont le module decide de la stabilite et dont la
    PHASE donne la frequence de broutement en boucle fermee. La version
    d'origine ne reorthonormalisait pas le bloc et gardait le facteur R de la
    QR dans la projection : sur la graine 3 elle rendait -0.2633 la ou la
    monodromie assemblee donne +1.0765 — un cas instable declare stable.
    """
    m, nx = 4, 6
    for seed in (3, 11, 42):
        maps = _random_maps(seed, nx)
        mu = np.linalg.eigvals(_exact_monodromy(maps, m, nx))
        ref = mu[int(np.argmax(np.abs(mu)))]
        ev = dominant_eigs(maps, m, nx, q=4)
        got = ev[int(np.argmax(np.abs(ev)))]
        assert abs(got - ref) < 1e-9 * max(1.0, abs(ref)), \
            f'graine {seed} : obtenu {got:+.6f}, attendu {ref:+.6f}'


def test_floquet_spectrum_matches_the_assembled_monodromy_on_the_real_plate():
    """Le rayon spectral rendu par le depot est CELUI de la monodromie.

    Le test precedent verifie cette identite sur des applications aleatoires,
    donc sur une monodromie bien conditionnee. Ce n'est pas le regime dans
    lequel le depot travaille : sur la vraie plaque en boucle fermee la
    monodromie est tres NON NORMALE (conditionnement de la base propre
    1.35e29 sur le FDOB), et c'est exactement la que l'iteration de puissance
    qui a longtemps vecu ici echouait — en rendant, selon la graine,
    0.79107 / 0.91421 / 0.90243 pour des valeurs exactes de
    0.967392 / 0.959809 / 0.939341.

    L'erreur allait TOUJOURS dans le meme sens : rho sous-estime, donc coupe
    declaree plus stable qu'elle ne l'est, donc a_p,lim surestimee. Un test
    sur des matrices aleatoires ne pouvait pas la voir ; celui-ci la voit.

    Trois structures reelles, deux profondeurs, et la monodromie assemblee
    comme unique reference.
    """
    from closed_loop import period_maps, spectral_radius, dominant_eig
    from fopid import series, rolloff_ss
    plate = _plate()
    n = 5
    D_obs = plate.D_row(plate.lp, plate.hp)[:n]
    res = D_obs * np.asarray(plate.H_Pe_modal, float)[:n]
    ro = rolloff_ss(C.ROLLOFF_HZ, C.ROLLOFF_ORDER)
    ctrls = {
        'fopid': series(fopid_ss(2.5e4, 1.0e7, 6.0, 0.27, 0.13,
                                 C.OUST_WB, C.OUST_WH, C.OUST_N, -1.0), ro),
        'adrc': series(adrc_fopid_ss(6.2e4, 5.5e7, 3.4e3, 0.58, 0.085,
                                     1.86e4, 40.72, C.OUST_WB, C.OUST_WH,
                                     C.OUST_N, -1.0), ro),
        'fdob': series(fdob_fopid_ss(133.0, 6.6e6, 389.0, 0.25, 0.33, 7.0e-3,
                                     0.33, plate.omega_n[:2],
                                     plate.zeta_modes[:2], res[:2],
                                     C.FDOB_WC, C.OUST_WB, C.OUST_WH,
                                     C.OUST_N, -1.0), ro),
    }
    m = 12                                   # petit : la reference est en O(dim^3)
    for kind, ss in ctrls.items():
        for ap in (0.10e-3, 0.20e-3):
            maps, _ = period_maps(plate, C.RPM_DESIGN, ap, 0.5 * plate.lp,
                                  ctrl=ss, pd=None, n_modes=n, m=m)
            nx = maps[0][0].shape[0]
            mu = np.linalg.eigvals(_exact_monodromy(maps, m, nx))
            lam = mu[int(np.argmax(np.abs(mu)))]
            ref = float(abs(lam))
            got = spectral_radius(maps, m, nx)
            assert abs(got - ref) < 1e-8 * max(1.0, ref), \
                f'{kind} a a_p = {ap * 1e3:.2f} mm : {got:.9f} vs {ref:.9f}'
            # ... et la PHASE, qui donne la frequence de broutement en boucle
            # fermee. En valeur absolue : d'une paire conjuguee, argmax(|.|)
            # peut rendre l'un ou l'autre membre, et `chatter_freq` prend
            # justement |arg|.
            ev = dominant_eig(maps, m, nx)
            got_l = ev[int(np.argmax(np.abs(ev)))]
            assert abs(abs(np.angle(got_l)) - abs(np.angle(lam))) < 1e-7, \
                (f'{kind} a a_p = {ap * 1e3:.2f} mm : phase '
                 f'{np.angle(got_l):.9f} contre {np.angle(lam):.9f}')


def test_nmp_dob_reduces_to_the_fopid_at_alpha_zero():
    """A alpha = 0 le NMP-DOB EST le FOPID — meme note, 30 etats contre 16.

    C'est le controle le plus severe disponible sur cette structure, et il
    porte sur DEUX choses a la fois : la realisation (une cascade de sections
    d'ordre <= 2 au lieu d'un polynome de degre 11) et l'ESTIMATEUR de
    Floquet. Un objectif qui depend du nombre d'etats ne peut pas le passer.

    Il ne le passait pas : avec l'iteration de puissance, ajouter au FOPID
    quatorze etats decouples et non observes — donc de fonction de transfert
    rigoureusement identique — deplacait deja J de 2.3e-4, et ce controle-ci
    rendait 5.8e-4 d'ecart. Les deux causes possibles etaient indiscernables ;
    l'ecart mesure aujourd'hui est de 5e-15.
    """
    import objective as OB
    import nmp_dob
    from fopid import series, rolloff_ss
    plate = _plate()
    n = 5
    D_obs = plate.D_row(plate.lp, plate.hp)[:n]
    res = D_obs * np.asarray(plate.H_Pe_modal, float)[:n]
    ro = rolloff_ss(C.ROLLOFF_HZ, C.ROLLOFF_ORDER)
    a = series(fopid_ss(PAR['Kp'], PAR['Ki'], PAR['Kd'], PAR['lam'],
                        PAR['mu'], C.OUST_WB, C.OUST_WH, C.OUST_N, -1.0), ro)
    b = series(nmp_dob.nmp_dob_fopid_ss(
        PAR['Kp'], PAR['Ki'], PAR['Kd'], PAR['lam'], PAR['mu'],
        2 * np.pi * 3000.0, 0.0, plate.omega_n[:n], plate.zeta_modes[:n],
        res, C.OUST_WB, C.OUST_WH, C.OUST_N, -1.0), ro)
    # la structure rend MOINS le FOPID pour un meme argument de signe
    b = (b[0], b[1], -b[2], -b[3])
    err = (np.max(np.abs(ss_frf(b, OM) - ss_frf(a, OM)))
           / np.max(np.abs(ss_frf(a, OM))))
    assert err < 1e-12, f'les deux transferts different de {err:.2e}'
    assert b[0].shape[0] > a[0].shape[0] + 8, \
        'le controle perd son objet si les deux ont le meme nombre d\'etats'
    ja = OB.evaluate(plate, a, C.RPM_DESIGN)
    jb = OB.evaluate(plate, b, C.RPM_DESIGN)
    assert abs(jb - ja) < 1e-9 * max(1.0, abs(ja)), \
        f'J = {ja:.9f} (16 etats) contre {jb:.9f} ({b[0].shape[0]} etats)'


def test_controller_realizations_are_conditioned():
    """Les correcteurs assembles ne portent pas de forme compagne brute.

    LE POINT AVEUGLE QUE CE TEST COMBLE. Tous les invariants de reponse
    frequentielle de ce fichier (zeros a droite, phase minimale du modele
    reduit, identite du LESO, superset FOPID) passent a l'identique sur une
    realisation d'etat INUTILISABLE : ss_frf resout (jw I - A) x = B, et un
    solve est insensible a une mise a l'echelle des etats. Le defaut de
    nmp_dob est passe entre les mailles pour cette raison exacte — la
    fonction de transfert etait bonne a 1e-15 et la matrice d'etat avait un
    conditionnement de 2.5e70, ce que Floquet lisait comme J = 0.0000.

    Floquet, lui, prend A telle quelle et l'exponentie. Il faut donc un
    invariant sur la MATRICE, pas sur la reponse.
    """
    from scipy.signal import tf2ss
    import nmp_dob
    plate = _plate()
    n = 5
    D_obs = plate.D_row(plate.lp, plate.hp)[:n]
    res = D_obs * np.asarray(plate.H_Pe_modal, float)[:n]
    num, den = nmp_dob.plant_tf(plate.omega_n[:n], plate.zeta_modes[:n], res)
    (nm, dm), _, _ = nmp_dob.inner_outer(num, den)
    wq = 2 * np.pi * 3000.0
    A = nmp_dob._w_ss(nm, dm, wq, nmp_dob.Q_ORDER)[0]
    c = np.linalg.cond(A)
    assert c < 1e6, f'realisation de W : conditionnement {c:.3e}'
    # ... et la forme compagne que cette cascade remplace, pour que le test
    # echoue si quelqu'un revient en arriere.
    bad = np.linalg.cond(tf2ss(np.convolve(np.r_[wq ** nmp_dob.Q_ORDER], dm),
                               np.convolve(
                                   np.poly([-wq] * nmp_dob.Q_ORDER), nm))[0])
    assert bad > 1e30, ('la forme compagne n\'est plus pathologique : '
                        f'{bad:.3e} — l\'invariant a perdu son objet')


def test_the_three_floquet_engines_agree():
    """Les TROIS moteurs de Floquet du depot rendent le meme rayon spectral.

    Ils existent pour de bonnes raisons — `stability_fdm` assemble la
    monodromie et n'accepte pas de correcteur, `lti_floquet` ne retarde que
    les positions, `closed_loop` retarde l'etat augmente complet — mais rien
    ne verifiait qu'ils decrivent le MEME systeme quand leurs hypotheses
    coincident, c'est-a-dire en boucle ouverte.

    Ce n'est pas une precaution abstraite : le remplacement de l'iteration de
    puissance par Arnoldi a touche deux des trois (le troisieme diagonalise
    la monodromie assemblee, donc sert ici de reference exacte). Un moteur
    qui derive des deux autres ne se verrait autrement qu'a travers un
    resultat publie.
    """
    import closed_loop as CL
    import lti_floquet as LF
    from stability_fdm import floquet_matrix, spectral_radius_and_freq
    from milling_dynamics import alpha4_series, N_TEETH
    plate = _plate()
    n = 2
    for rpm, ap in ((4900, 0.05e-3), (5500, 0.30e-3), (4300, 0.15e-3)):
        m = 30
        tau = 60.0 / (N_TEETH * rpm)
        x = 0.5 * plate.lp
        D = plate.D_row(x, plate.hp)[:n]
        _, a4 = alpha4_series(rpm, ap, plate.hp, m, midpoint=True)
        Phi = floquet_matrix(plate.omega_n[:n], plate.zeta_modes[:n],
                             np.outer(D, D), C.SIGN_SIM * a4, tau, n)
        ref = spectral_radius_and_freq(Phi, tau)[0]
        got = []
        for eng, kw in ((CL, dict(pd=None)), (LF, {})):
            maps, _ = eng.period_maps(plate, rpm, ap, x, ctrl=None,
                                      n_modes=n, m=m, coeff_mode='time',
                                      coeff_scale=C.SIGN_SIM, **kw)
            got.append(eng.spectral_radius(maps, m, maps[0][0].shape[0]))
        for name, r in zip(('closed_loop', 'lti_floquet'), got):
            assert abs(r - ref) < 1e-8 * max(1.0, ref), \
                (f'{name} a {rpm} tr/min, a_p = {ap * 1e3:.2f} mm : '
                 f'{r:.9f} contre {ref:.9f} (monodromie assemblee)')


def test_w_realization_survives_leading_zeros_and_rejects_bad_roots():
    """Deux pieges du realisateur en cascade, tous deux mesures.

    1. UN ZERO DE TETE dans le denominateur eteignait tout. `np.roots` ignore
       les zeros de tete ; `dm[0]` non. Le gain valait alors zero, puis
       np.sign(0) = 0 eteignait chaque section : la reponse tombait a
       IDENTIQUEMENT ZERO (max|H| = 0 au lieu de 6.87e8) sans exception ni
       avertissement. Inatteignable depuis nmp_dob_fopid_ss, ou dm sort de
       `inner_outer` donc monique — mais `stable_inverse_ss` est publique et
       ne promet rien de tel a son appelant.

    2. LE CONTROLE DE CONJUGAISON n'existait pas. `_real_factors` ne
       verifiait que le DEGRE total, si bien qu'un jeu comme {a+bi, c-di}
       avec a != c passait et etait silencieusement remplace par la
       factorisation conjuguee — donc par d'autres racines que celles
       demandees. Le message d'erreur annoncait pourtant ce controle.
    """
    import nmp_dob
    plate = _plate()
    n = 5
    D_obs = plate.D_row(plate.lp, plate.hp)[:n]
    res = D_obs * np.asarray(plate.H_Pe_modal, float)[:n]
    num, den = nmp_dob.plant_tf(plate.omega_n[:n], plate.zeta_modes[:n], res)
    (nm, dm), _, _ = nmp_dob.inner_outer(num, den)
    wq = 2 * np.pi * 3000.0
    ref = ss_frf(nmp_dob._w_ss(nm, dm, wq), OM)
    for pad in (1, 2):
        got = ss_frf(nmp_dob._w_ss(nm, np.r_[np.zeros(pad), dm], wq), OM)
        e = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
        assert e < 1e-12, f'{pad} zero(s) de tete : ecart {e:.3e}'
    import pytest
    with pytest.raises(ValueError):
        nmp_dob._real_factors(np.array([1.0 + 2.0j, 3.0 - 4.0j]))
    # ... et le controle ne doit pas refuser un jeu legitime
    f = nmp_dob._real_factors(np.array([1.0 + 2.0j, 1.0 - 2.0j, -5.0]))
    assert sum(len(x) - 1 for x in f) == 3


def test_step_integrals_agree_on_both_paths():
    """Les deux chemins de J1 et J2 donnent la meme chose quand A est
    inversible, et le chemin augmente REPOND encore quand elle ne l'est pas.

    Les trois moteurs de Floquet en dependent ; deux d'entre eux levaient une
    exception au lieu de rendre une integrale parfaitement definie.
    """
    rng = np.random.default_rng(7)
    h = 1e-4
    A = rng.standard_normal((8, 8)) * 50.0
    _, J1, J2 = step_integrals(A, h)
    ref1 = np.linalg.solve(A, np.eye(8) * 0 + _expm(A * h) - np.eye(8))
    ref2 = h * ref1 - np.linalg.solve(A, h * _expm(A * h) - ref1)
    for got, ref, nm in ((J1, ref1, 'J1'), (J2, ref2, 'J2')):
        err = float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
        assert err < 1e-8, f'{nm} : les deux chemins divergent de {err:.2e}'
    # A SINGULIERE : c'est le cas qui faisait planter stability_fdm.
    S = A.copy()
    S[:, 3] = S[:, 0] * 2.0                     # colonne dependante -> det = 0
    _, K1, K2 = step_integrals(S, h)
    assert np.all(np.isfinite(K1)) and np.all(np.isfinite(K2)), \
        'A singuliere : integrales non finies'
    # controle independant : J1 = int_0^h e^{As} ds, par quadrature fine
    ts = (np.arange(4000) + 0.5) * (h / 4000)
    quad = sum(_expm(S * t) for t in ts) * (h / 4000)
    err = float(np.max(np.abs(K1 - quad)) / np.max(np.abs(quad)))
    assert err < 1e-6, f'A singuliere : J1 faux de {err:.2e}'


# -------------------------------------------------------------- H-infini
def _hinf_problem(k=50.0, f0=543.0, zw=0.3, w2=1.0):
    from hinf import plant_ss, bandpass_weight, augment
    w = np.array([2 * np.pi * f0])
    return augment(plant_ss(w, np.array([0.02]), np.array([3.0])),
                   bandpass_weight(k, f0, zw), w2, 1e-3)


def test_care_residual_is_zero():
    """La solution du hamiltonien verifie VRAIMENT A'X + XA + Q - XSX = 0.

    `care` n'utilise pas scipy.solve_continuous_are : S est INDEFINIE en
    H-infini (B2B2' - B1B1'/g^2), ce qu'aucune ecriture (A,B,Q,R) ne
    represente. Ce test verifie l'equation elle-meme, pas la routine.
    """
    from hinf import care
    rng = np.random.default_rng(5)
    for _ in range(4):
        n = 6
        A = rng.standard_normal((n, n)) - 2.0 * np.eye(n)
        B2 = rng.standard_normal((n, 1))
        B1 = rng.standard_normal((n, 2))
        C1 = rng.standard_normal((2, n))
        S = B2 @ B2.T - (B1 @ B1.T) / 9.0        # indefinie a dessein
        Q = C1.T @ C1
        X = care(A, S, Q)
        R = A.T @ X + X @ A + Q - X @ S @ X
        err = float(np.max(np.abs(R))) / max(1.0, float(np.max(np.abs(Q))))
        assert err < 1e-8, f'residu de Riccati {err:.2e}'
        assert np.min(np.linalg.eigvalsh(S)) < 0, 'S devait etre indefinie'


def test_hinf_assumptions_hold_by_construction():
    """`augment` produit D11 = 0, D12'C1 = 0 et B1 D21' = 0 SANS loop-shift.

    C'est ce qui autorise les formules simplifiees de Glover-Doyle. Si cette
    propriete tombe (par exemple si quelqu'un rend W2 dynamique ou W1
    seulement propre), la synthese devient fausse SANS rien signaler — d'ou
    ce test.
    """
    from hinf import assumptions
    for kw in (dict(), dict(w2=1e-3), dict(zw=0.05), dict(f0=1068.0)):
        ok, rep = assumptions(*_hinf_problem(**kw))
        assert ok, f'hypotheses violees pour {kw} : {rep}'


def test_hinf_reaches_the_gamma_it_claims():
    """La norme H-infini ANNONCEE est retrouvee par un balayage frequentiel.

    Le controle est independant du solveur : il ne partage aucune ligne de
    code avec les equations de Riccati. Un solveur faux rend une matrice
    plausible qui ne fait pas ce qu'elle promet ; c'est ce test, et lui seul,
    qui separe "les equations ont converge" de "le correcteur atteint gamma".
    """
    from hinf import synthesize, lower_lft, hinf_norm
    P = _hinf_problem()
    K, g = synthesize(P, check=False)
    got = hinf_norm(lower_lft(P, K))
    assert np.isfinite(got), 'boucle fermee instable'
    assert abs(got - g) <= 0.05 * g, \
        f'gamma annonce {g:.6g}, mesure {got:.6g}'
    ev = np.linalg.eigvals(lower_lft(P, K)[0])
    assert ev.real.max() < 0, f'boucle fermee instable : {ev.real.max():.3g}'


# ----------------------------------------------------------------- mu-synthese
def test_d_scaling_is_exact_for_a_dynamic_scale():
    """Inserer D sur z_Delta et D^-1 sur w_Delta multiplie EXACTEMENT la ligne
    par D(jw) et divise la colonne par D(jw).

    C'est le coeur de l'iteration D-K, et le piege est precis : `w_Delta`
    n'entre pas dans l'etat, il s'AJOUTE A LA SORTIE. Toute sa voie passe donc
    par le terme direct, et l'etat de D^-1 doit etre reinjecte dans C2. Sans
    cela la mise a l'echelle reste exacte pour un D CONSTANT — ou D^-1 n'a pas
    d'etat — et fausse de 85 % des que D devient dynamique, c'est-a-dire dans
    tous les cas qui servent a quelque chose. Un D-K silencieusement faux
    rendrait des bornes mu sans aucun rapport avec la realite.
    """
    from hinf import plant_ss, bandpass_weight
    from musyn import augment_mu, _scale_channel, _frf
    P = augment_mu(plant_ss(np.array([2 * np.pi * 543.0]), np.array([0.02]),
                            np.array([3.0])),
                   bandpass_weight(50.0, 543.0, 0.3), 1.0, 1e-3)

    def openmap(Q):
        A, B1, B2, C1, C2, D12, D21 = Q
        B = np.hstack([B1, B2])
        Cc = np.vstack([C1, C2])
        D = np.block([[np.zeros((C1.shape[0], B1.shape[1])), D12],
                      [D21, np.zeros((1, 1))]])
        return (A, B, Cc, D)

    om = 2 * np.pi * np.logspace(1, 4, 60)
    G0 = _frf(openmap(P), om)
    for k, a, b in ((3.0, 1.0, 1.0),
                    (2.0, 2 * np.pi * 300.0, 2 * np.pi * 2000.0),
                    (0.4, 2 * np.pi * 5000.0, 2 * np.pi * 200.0)):
        Dss = (np.array([[-b]]), np.array([[1.0]]),
               np.array([[k * (a - b)]]), np.array([[k]]))
        Dj = k * (1j * om + a) / (1j * om + b)
        G1 = _frf(openmap(_scale_channel(P, Dss)), om)
        E = np.array(G0, copy=True)
        E[:, 0, :] *= Dj[:, None]
        E[:, :, 0] /= Dj[:, None]
        err = float(np.max(np.abs(G1 - E)) / np.max(np.abs(E)))
        assert err < 1e-10, f'mise a l echelle D = {k}(s+{a:.3g})/(s+{b:.3g})' \
                            f' fausse de {err:.2e}'


def test_mu_bound_is_between_spectral_radius_and_sigma_max():
    """rho(M) <= mu(M) <= sigma_max(M) — l'encadrement qui definit mu.

    Deux reperes independants du code qui calcule la borne. Si la section
    doree se trompait de sens, ou si le bloc mis a l'echelle n'etait pas le
    bon, l'un des deux cotes tomberait.
    """
    from musyn import mu_upper
    rng = np.random.default_rng(11)
    for _ in range(20):
        M = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        mu, d = mu_upper(M, n_a=1)
        rho = float(np.max(np.abs(np.linalg.eigvals(M))))
        sig = float(np.linalg.svd(M, compute_uv=False)[0])
        assert rho <= mu + 1e-9, f'mu = {mu:.6g} sous rho = {rho:.6g}'
        assert mu <= sig + 1e-9, f'mu = {mu:.6g} au-dessus de sigma = {sig:.6g}'
        assert d > 0.0


def test_gamma_search_is_not_fooled_by_infeasibility_holes():
    """La recherche de gamma trouve un gamma aussi bon qu'un BALAYAGE direct.

    L'invariant qui manquait. La faisabilite en gamma est monotone EN THEORIE
    — si un gamma passe, tous les plus grands passent — mais elle ne l'est pas
    numeriquement : les deux Riccati echouent par endroits AU MILIEU de la
    region faisable, en trous larges d'un facteur cinq. Une bissection, qui
    n'est valide que sous la monotonie, sonde un trou, conclut "infaisable" et
    perd definitivement toute la region des gamma plus petits. Elle rend alors
    un correcteur bien plus faible que celui qui existe, SANS AUCUN SIGNE
    EXTERIEUR — c'est ce qui a invalide une campagne d'optimisation entiere.

    Ce test compare donc le gamma rendu a celui d'un balayage exhaustif
    grossier, qui ne suppose aucune monotonie. Si quelqu'un revient un jour a
    une bissection, ce test tombe.
    """
    from hinf import (HinfFailure, augment, bandpass_weight, central,
                      lower_lft, plant_ss, scale_problem, synthesize)
    plate = _plate()
    w, z, H, D_obs, _ = plant_vectors(plate, C.N_MODES_DESIGN)
    P0 = plant_ss(w, z, D_obs * H)
    for kw, f0 in ((5166.0, 1151.0), (5200.0, 1151.0), (1e4, 543.0)):
        P = augment(P0, bandpass_weight(kw, f0, 0.03969), 2.202, 1.152e-6)
        Ps, alpha, beta = scale_problem(P)
        # Balayage direct : aucune hypothese de monotonie.
        best_scan = np.inf
        for lg in np.linspace(-8, 2, 60):
            g = 10.0 ** lg
            try:
                K = central(*Ps, g)
                ev = np.linalg.eigvals(lower_lft(P, K)[0])
                if np.all(np.isfinite(ev)) and ev.real.max() < 0:
                    best_scan = min(best_scan, g)
            except HinfFailure:
                pass
        if not np.isfinite(best_scan):
            continue                      # rien de faisable : rien a comparer
        got = synthesize(P)[1] / (alpha * beta)
        # Le balayage a 60 points sur 10 decades a un pas de 1.47 ; on tolere
        # le double, ce qui laisse passer la discretisation mais pas un
        # optimiseur qui aurait rate toute une region.
        assert got <= 3.0 * best_scan, (
            f'kw = {kw} : gamma rendu {got:.4g} contre {best_scan:.4g} '
            f'trouve par balayage — facteur {got / best_scan:.1f}')


def test_nmp_dob_is_a_superset_of_fopid_and_matches_its_closed_form():
    """A alpha = 0 la structure redonne le FOPID EXACTEMENT, et sa realisation
    verifie K = -(C + alpha W)/(1 - alpha V) avec W = Q P_min^-1, V = Q.

    Meme paire de proprietes que pour l'observateur modal, et pour la meme
    raison : sans la premiere, la structure ne CONTIENT pas le FOPID et son
    ensemble est decale plutot que plus grand — c'est precisement le defaut
    que le diagnostic avait impute a l'ADRC-FOPID. Sans la seconde, le montage
    d'etat ne calcule pas ce que l'algebre annonce.
    """
    from nmp_dob import (Q_ORDER, _q_tf, frf_tf, inner_outer,
                         nmp_dob_fopid_ss, plant_tf)
    plate = _plate()
    w, z, H, D_obs, sl = plant_vectors(plate, C.N_MODES)
    res = D_obs * H
    kw = dict(w=w, zeta=z, res=res, wb=C.OUST_WB, wh=C.OUST_WH, N=C.OUST_N,
              sign_loop=sl)
    Kf = ss_frf(fopid_ss(PAR['Kp'], PAR['Ki'], PAR['Kd'], PAR['lam'],
                         PAR['mu'], C.OUST_WB, C.OUST_WH, C.OUST_N, sl), OM)
    K0 = ss_frf(nmp_dob_fopid_ss(**PAR, wq=2 * np.pi * 3000, alpha=0.0, **kw),
                OM)
    err = float(np.max(np.abs(K0 + Kf) / np.abs(Kf)))
    assert err < 1e-12, f'alpha = 0 ne redonne pas le FOPID : {err:.2e}'

    wq, al = 2 * np.pi * 3000, 0.5
    K = ss_frf(nmp_dob_fopid_ss(**PAR, wq=wq, alpha=al, **kw), OM)
    num, den = plant_tf(w, z, res)
    (nm, dm), _, _ = inner_outer(num, den)
    qn, qd = _q_tf(wq, Q_ORDER)
    V = frf_tf(qn, qd, OM)
    W = frf_tf(np.convolve(qn, dm),
               np.convolve(qd, np.trim_zeros(np.asarray(nm, float), 'f')), OM)
    pred = -(Kf + al * W) / (1 - al * V)
    err = float(np.max(np.abs(pred - K) / np.abs(K)))
    assert err < 1e-9, f'forme fermee du NMP-DOB : {err:.2e}'


def test_inner_outer_factorization_reconstructs_the_plant():
    """P = B . P_min avec |B| = 1 et P_min sans zero instable.

    Les trois proprietes ensemble, car SEPAREMENT elles ne prouvent rien. Une
    orientation fausse du facteur de Blaschke — (z-s)/(z+s) au lieu de
    (s-z)/(s+z), la convention la plus repandue — laisse |P_min| = |P| exact
    et |B| = 1 exact, et donne pourtant B.P_min = -P. Un observateur bati
    la-dessus inverserait le signe de la boucle sans qu'aucun des deux
    controles evidents ne bronche. C'est le controle de RECONSTRUCTION qui
    tranche, et c'est pour cela qu'il est ici.
    """
    from nmp_dob import check_factorization
    plate = _plate()
    w, z, H, D_obs, _ = plant_vectors(plate, C.N_MODES)
    r = check_factorization(w, z, D_obs * H)
    assert r['n_rhp'] == 1, f"attendu 1 zero instable, obtenu {r['n_rhp']}"
    assert 2400 < r['f_rhp'][0] < 2520, f"zero a {r['f_rhp'][0]:.0f} Hz"
    assert r['mag_err'] < 1e-9, f"|P_min| != |P| : {r['mag_err']:.2e}"
    assert r['allpass_err'] < 1e-12, f"|B| != 1 : {r['allpass_err']:.2e}"
    assert r['recon_err'] < 1e-9, f"B.P_min != P : {r['recon_err']:.2e}"
    assert r['n_rhp_after'] == 0, 'P_min garde un zero instable'


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
        except Exception as e:            # noqa: BLE001
            # Ne rattraper qu'AssertionError coupait la boucle des qu'un test
            # levait autre chose : les invariants suivants n'etaient alors
            # jamais evalues et le total jamais imprime.
            bad += 1
            print(f'  ERREUR {f.__name__} : {type(e).__name__}: {e}')
    print(f'\n  {len(fs) - bad}/{len(fs)} invariants verifies')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(_main())


def test_les_exceptions_de_synthese_sont_toutes_rattrapees_par_build():
    """Toute exception qu'une voie de synthese peut lever DOIT etre rattrapee
    par `Design.build`, qui rend alors None — une conception qu'on ne sait pas
    noter est INFAISABLE, pas une panne du programme.

    Ce test existe a cause d'une panne reelle. J'ai ajoute `musyn.MuBracket`
    pour qu'une section doree saturee cesse de rendre une borne mu gonflee en
    silence, en la faisant heriter de RuntimeError — sans verifier que les
    appelants la rattrapaient. `Design.build` n'attrape que
    (HinfFailure, LinAlgError, ValueError, FloatingPointError) : la campagne
    musyn est morte au bout de 43 minutes en emportant tout le tirage. Le
    garde etait devenu un mode de panne pire que le defaut qu'il signalait.

    On verifie donc la PROPRIETE, pas le cas particulier : chaque exception
    definie dans les modules de synthese doit etre une sous-classe de l'un
    des types que `build` rattrape.
    """
    import ast
    import inspect
    import numpy as np
    import classical
    import hinf
    import musyn
    import nonlinear
    import pso

    # L'UNION des types rattrapes par les `except` de Design.build — et non un
    # tuple ecrit a la main. Les branches n'attrapent pas toutes la meme
    # chose : la voie lqg/mpc rattrape LqgFailure, la voie hinf/musyn rattrape
    # HinfFailure. Coder un seul tuple ici ferait echouer le test sur une
    # exception parfaitement geree, et c'est ce qu'a fait sa premiere version.
    src = inspect.getsource(pso.Design.build)
    arbre = ast.parse(src.lstrip())
    noms = set()
    for n in ast.walk(arbre):
        if isinstance(n, ast.ExceptHandler) and n.type is not None:
            cibles = (n.type.elts if isinstance(n.type, ast.Tuple)
                      else [n.type])
            for c in cibles:
                noms.add(ast.unparse(c).split('.')[-1])
    rattrapees = tuple(
        t for t in (hinf.HinfFailure, classical.LqgFailure, ValueError,
                    FloatingPointError, np.linalg.LinAlgError, RuntimeError)
        if t.__name__ in noms)
    assert rattrapees, f'aucun type reconnu parmi {sorted(noms)}'

    manquantes = []
    for mod in (hinf, musyn, classical, nonlinear):
        for nom, obj in vars(mod).items():
            if (inspect.isclass(obj) and issubclass(obj, BaseException)
                    and obj.__module__ == mod.__name__
                    and not issubclass(obj, rattrapees)):
                manquantes.append(f'{mod.__name__}.{nom}')
    assert not manquantes, (
        'exceptions qu aucun except de Design.build ne rattrape : '
        + ', '.join(manquantes)
        + ' — les faire heriter d un type deja rattrape, ou les ajouter '
          'a la branche correspondante dans control/pso.py')


# ------------------------------------------------- controle a retard actif
def test_le_spectre_par_monodromie_redonne_les_poles_sans_retard():
    """`nominal_max_re` doit coincider avec les valeurs propres quand pd = 0.

    A profondeur nulle et SANS terme retarde, la boucle fermee est un simple
    probleme aux valeurs propres. Avec le terme de l'Eq. (30) elle ne l'est
    plus : les gains vivent sur l'etat retarde, le spectre devient infini, et
    le maximum se lit sur la monodromie — log(rho)/tau, puisque le
    multiplicateur dominant vaut e^{lambda tau}.

    Le risque est que ce SECOND chemin, celui qui sert a `musyn_td`, soit
    faux d'un facteur ou d'un signe sans que rien ne le montre : il ne
    s'applique qu'a une structure, et il n'y a aucun autre chiffre a
    comparer. On l'ancre donc sur le cas ou les deux chemins doivent donner
    LE MEME nombre — pd = (0, 0), qui est un terme retarde d'amplitude nulle
    et laisse pourtant le calcul passer par la monodromie.
    """
    import numpy as np
    import config as C
    from plate_model import build_plate
    from objective import nominal_poles, nominal_max_re
    from pso import Design

    plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
    D = Design('fopid', plate, -1.0, sign_variant=1.0)
    ss = D.build(np.full(D.n, 0.5))
    assert ss is not None

    par_valeurs_propres = float(nominal_poles(plate, ss,
                                              n_modes=C.N_MODES).real.max())
    par_monodromie = nominal_max_re(plate, ss, (0.0, 0.0), n_modes=C.N_MODES)
    ech = max(1.0, abs(par_valeurs_propres))
    assert abs(par_monodromie - par_valeurs_propres) < 1e-3 * ech, (
        f'monodromie {par_monodromie:.6g} contre valeurs propres '
        f'{par_valeurs_propres:.6g}')


def test_le_correcteur_a_retard_simule_a_la_frf_qui_le_contraint():
    """Le simulateur temporel et la contrainte frequentielle, meme loi.

    `frequency_metrics` note le terme de l'Eq. (30) par sa FRF analytique,
    (K_Pp + K_Pd jw) e^{-jw tau} ; `DelayedPDController` l'execute pas a pas
    sur un historique. Ce sont deux ECRITURES de la meme loi, et rien ne les
    relie dans le code : un retard decale d'un pas, un signe de K_Pd, une
    convention d'historique a l'envers passeraient les deux fois — la
    contrainte porterait sur une loi, la simulation sur une autre, et les
    deux tableaux se contrediraient sans qu'on sache lequel croire.

    On excite donc le correcteur seul par une sinusoide et on compare le gain
    ET la phase mesures a la formule. La phase est le point sensible : c'est
    elle qui porte tout l'effet du retard.
    """
    import numpy as np
    from sim_controller import DelayedPDController

    tau = 4.0816e-3                 # une periode de dent a 4900 tr/min
    n_sub = 164
    dt = tau / n_sub
    Kp, Kd = 3.7e4, -21.0
    c = DelayedPDController(None, (Kp, Kd), n_sub, dt)

    for f0 in (37.0, 245.0, 613.0):
        c.reset()
        w = 2 * np.pi * f0
        n = int(round(24.0 / (f0 * dt)))          # 24 periodes
        t = np.arange(n) * dt
        y = np.sin(w * t)
        yd = w * np.cos(w * t)
        u = np.array([c(y=y[k], yd=yd[k]) for k in range(n)])

        # on ne garde que la fin, une fois l'historique rempli
        k0 = 4 * n_sub
        s, cc = np.sin(w * t[k0:]), np.cos(w * t[k0:])
        m = 2.0 / (n - k0)
        a, b = m * float(u[k0:] @ s), m * float(u[k0:] @ cc)
        mesure = a + 1j * b
        exact = (Kp + Kd * 1j * w) * np.exp(-1j * w * tau)
        rel = abs(mesure - exact) / abs(exact)
        assert rel < 2e-2, (
            f'a {f0} Hz : mesure {abs(mesure):.4g} @ '
            f'{np.angle(mesure, deg=True):.2f} deg contre formule '
            f'{abs(exact):.4g} @ {np.angle(exact, deg=True):.2f} deg '
            f'(ecart relatif {rel:.3%})')


def test_tout_l_aval_recharge_les_gains_de_retard():
    """Aucun script d'aval ne doit reassembler un correcteur a la main.

    Les gains de l'Eq. (30) ne tiennent pas dans un (A, B, C, D) : ils vivent
    sur l'etat retarde. Un script qui recharge un correcteur en lisant les
    quatre champs directement recharge donc mu TOUT SEUL — sans lever, sans
    rien afficher d'anormal, et en publiant sous le nom de la reference du
    papier des chiffres qui ne sont pas les siens. C'est exactement ce que
    faisaient les quatre boucles recopiees avant `stored_ctrl`.

    Le test porte sur la PROPRIETE structurelle, pas sur une liste de
    fichiers : tout module de `control/` qui reassemble les quatre champs
    pour une structure VARIABLE doit passer par `stored_ctrl`. Un cinquieme
    script d'aval ecrit plus tard tombera dessus tout seul.

    Deux choses ne sont volontairement PAS visees. Lire `{k}__A` seulement
    pour ENUMERER les structures presentes ne recharge rien
    (`audit_fairness`, `analyse_fdob` font cela). Et nommer la structure en
    toutes lettres — `d['adrc__A']` dans `diagnose_adrc` — est un choix
    delibere et verifiable : ni `adrc` ni `fopid` n'ont de terme retarde, et
    le jour ou l'un en aurait un, ce serait a la relecture de ce nom-la de
    s'en apercevoir, pas a une regle generale.
    """
    import glob
    import os
    import re

    ici = os.path.dirname(os.path.abspath(__file__))
    # les quatre champs, avec un nom de structure VARIABLE (f-string)
    motifs = [re.compile(r'\{[A-Za-z_][A-Za-z_0-9]*\}__' + ch)
              for ch in 'ABCD']
    fautifs = []
    for chemin in sorted(glob.glob(os.path.join(ici, '..', 'control',
                                                '*.py'))):
        nom = os.path.basename(chemin)
        if nom in ('stored_ctrl.py', 'run_pso.py', 'merge_pso.py'):
            continue                       # ceux-la ECRIVENT le fichier
        src = open(chemin).read()
        if not all(m.search(src) for m in motifs):
            continue
        if 'stored_ctrl' not in src:
            fautifs.append(nom)
    assert not fautifs, (
        'rechargent un correcteur sans ses gains de retard : '
        + ', '.join(fautifs)
        + ' — passer par control/stored_ctrl.discover()')


def test_la_fusion_de_robustesse_ne_perd_pas_de_colonne():
    """Fusionner ne doit jamais RETRANCHER une structure du tableau.

    La fusion reconstruisait le tableau a partir des seuls morceaux presents
    sur le disque, sans jamais relire sa propre destination. Une structure
    dont le morceau avait ete efface — parce qu'une campagne precedente
    l'avait deja repliee dans le fichier fusionne — disparaissait donc a la
    fusion suivante, sans un mot : la sortie annonce le nombre de colonnes
    ECRITES, pas celles qu'elle a perdues. En ajoutant la douzieme structure,
    la fusion a rendu onze colonnes ou `hinf` et `musyn` n'etaient plus.

    Le test construit la situation exacte : une destination qui contient une
    structure, un morceau frais qui en contient une autre, et la verification
    que la fusion rend LES DEUX.
    """
    import os
    import tempfile
    import numpy as np
    from robustness_new import merge

    labels = np.array(['cas A', 'cas B'])
    with tempfile.TemporaryDirectory() as tmp:
        dest = os.path.join(tmp, 'fusionne.npz')
        np.savez_compressed(dest, labels=labels,
                            kinds=np.array(['ancienne']),
                            limits=np.array([[1.0], [2.0]]))
        part = os.path.join(tmp, 'fusionne_neuve.npz')
        np.savez_compressed(part, labels=labels,
                            kinds=np.array(['neuve']),
                            limits=np.array([[3.0], [4.0]]))

        merge(dest, [part])

        d = np.load(dest, allow_pickle=True)
        got = [str(k) for k in d['kinds']]
        assert 'ancienne' in got, (
            f'colonne perdue a la fusion : {got} — la destination doit etre '
            f'relue comme source')
        assert 'neuve' in got, got
        M = np.asarray(d['limits'], float)
        assert list(M[:, got.index('ancienne')]) == [1.0, 2.0]
        assert list(M[:, got.index('neuve')]) == [3.0, 4.0]


def test_nominal_max_re_redonne_exactement_l_ancienne_voie():
    """Le detour par `nominal_max_re` ne doit deplacer aucun chiffre.

    Le tableau des poles a coupe nulle assemblait sa matrice avec
    `closed_loop.build_matrices` ; il passe desormais par
    `objective.nominal_max_re`, pour que la meme colonne affichee par
    `run_compare` vienne du meme code et pour que le terme retarde soit traite
    par la monodromie plutot que par des valeurs propres qui ne le decrivent
    pas. Une refonte pareille doit etre un NO-OP sur les structures sans
    retard — sinon elle change en silence une colonne deja publiee.

    Le test compare les deux assemblages sur toutes les structures stockees,
    boucle ouverte comprise. La boucle ouverte est le cas qui a effectivement
    casse : `nominal_poles` n'acceptait pas ss = None, et le tableau levait
    TypeError sur sa premiere colonne.
    """
    import numpy as np
    import config as C
    from plate_model import build_plate
    from closed_loop import build_matrices
    from objective import nominal_max_re
    from stored_ctrl import discover

    plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
    n = C.N_MODES

    def par_build_matrices(ss):
        D = plate.D_row(0.5 * plate.lp, plate.hp)[:n]
        D_obs = plate.D_row(plate.lp, plate.hp)[:n]
        H = np.asarray(plate.H_Pe_modal, float)[:n]
        A, _ = build_matrices(plate, np.outer(D, D), D_obs, H, 0.0, ss,
                              None, n)
        return float(np.max(np.real(np.linalg.eigvals(A))))

    cas = [('boucle ouverte', None)]
    cas += [(k, ss) for k, (ss, pd) in discover().items() if pd is None]
    assert len(cas) > 1, 'aucun correcteur stocke : lancer run_pso.py'

    for k, ss in cas:
        a = par_build_matrices(ss)
        b = nominal_max_re(plate, ss, None, n_modes=n)
        assert abs(a - b) <= 1e-9 * max(1.0, abs(a)), (
            f'{k} : build_matrices {a:.9g} contre nominal_max_re {b:.9g}')
