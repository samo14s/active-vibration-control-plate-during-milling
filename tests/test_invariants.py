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
        ev = dominant_eigs(maps, m, nx, q=4, n_period=60)
        got = ev[int(np.argmax(np.abs(ev)))]
        assert abs(got - ref) < 1e-9 * max(1.0, abs(ref)), \
            f'graine {seed} : obtenu {got:+.6f}, attendu {ref:+.6f}'


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
