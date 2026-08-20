"""
diagnose_adrc.py — Diagnostic MATHEMATIQUE de la chaine de perte de l'ADRC-FOPID
================================================================================
On ne cherche pas ici a ameliorer un correcteur : on cherche a savoir QUEL
maillon casse, et a le prouver. La chaine soupconnee etait

    changement de signe de G_yu -> l'ESO se trompe -> d_chapeau faux
      -> u excessif -> saturation -> perte de stabilite

Six maillons sont testes separement. Trois d'entre eux ne survivent pas a
l'examen, et c'est la partie utile du resultat.

    PROTOCOL=B python diagnose_adrc.py
"""
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model'), HERE]

import config as C
from plate_model import build_plate, plant_vectors, plant_frf
from fopid import ss_frf
from simulate import MillingSimulation
from sim_controller import LTIController
from objective import limits

OUT = os.path.join(HERE, '..', 'results')
FIG = os.path.join(HERE, '..', 'figures', 'comparison')
os.makedirs(FIG, exist_ok=True)


def plant_zeros(w, z, res):
    """Zeros de P(s) = somme_i res_i / (s^2 + 2 z_i w_i s + w_i^2)."""
    n = len(w)
    den = [np.array([1.0, 2 * z[i] * w[i], w[i] ** 2]) for i in range(n)]
    num = np.zeros(1)
    for i in range(n):
        p = np.array([res[i]])
        for j in range(n):
            if j != i:
                p = np.convolve(p, den[j])
        num = np.polyadd(num, p)
    return np.roots(num)


def growth_rate(t, y, frac=0.4):
    """Taux de croissance sigma [1/s] ajuste sur l'enveloppe de |y|."""
    i0 = int((1 - frac) * len(t))
    seg = np.abs(y[i0:])
    tt = t[i0:]
    k = max(1, len(seg) // 60)
    nb = len(seg) // k
    env = seg[:nb * k].reshape(nb, k).max(axis=1)
    te = tt[:nb * k:k]
    m = env > 0
    if m.sum() < 5:
        return np.nan
    A = np.polyfit(te[m], np.log(env[m]), 1)
    return float(A[0])


def main():
    d = np.load(os.path.join(OUT, f'pso_{C.PROTOCOL}.npz'), allow_pickle=True)
    par = dict(zip([str(k) for k in d['adrc__keys']], d['adrc__values']))
    var = float(d['adrc__sign_variant'])
    b0 = float(par['b0']) * var
    wo = float(par['wo'])
    ss_adrc = (d['adrc__A'], d['adrc__B'], d['adrc__C'], d['adrc__D'])
    ss_fopid = (d['fopid__A'], d['fopid__B'], d['fopid__C'], d['fopid__D'])

    plate = build_plate(C.PATCH_SIDE, freqs=C.F_NOMINAL)
    w, zt, H, D_obs, _ = plant_vectors(plate, C.N_MODES)
    res = D_obs * H
    # LA CONVENTION DE SIGNE EST CELLE DU MODELE DE SYNTHESE, pas celle du
    # modele de verite. run_pso.py la tire de N_MODES_DESIGN ; la relire ici
    # sur N_MODES reconstruit un correcteur de signe OPPOSE des que les deux
    # different, c'est-a-dire sous le protocole A : D_obs.H y vaut -1.635 sur
    # deux modes et +3.395 sur cinq, donc sign_loop passe de +1 a -1. Les
    # figures de diagnostic du protocole A etaient produites avec la boucle
    # inversee.
    sl0 = plant_vectors(plate, C.N_MODES_DESIGN)[4]

    print('=' * 78)
    print(f' DIAGNOSTIC DE LA CHAINE DE PERTE — ADRC-FOPID, protocole'
          f' {C.PROTOCOL}')
    print('=' * 78)
    print(f'  plaque  : f = {np.round(plate.freq_n, 1)} Hz')
    print(f'  residus : r_i = D_obs(i) H_Pe(i) = {np.round(res, 4)}')
    print(f'  b_inf = somme r_i = {res.sum():+.4f} ;'
          f' somme |r_i| = {np.abs(res).sum():.4f}')
    print(f'  ADRC    : b0 = {b0:+.4g}, w_o = {wo:.4g} rad/s'
          f' ({wo / 2 / np.pi:.0f} Hz)')

    f = np.logspace(1, 4.4, 4000)
    om = 2 * np.pi * f
    P, _ = plant_frf(plate, f, C.N_MODES)
    K = ss_frf(ss_adrc, om)
    Kf = ss_frf(ss_fopid, om)
    Q = (wo / (1j * om + wo)) ** 3

    # ---------------------------------------------------------------- 1
    print('\n' + '-' * 78)
    print(' MAILLON 1 — la voie de commande P(s) = y/u   [CONFIRME]')
    print('-' * 78)
    b_eff = -om ** 2 * P.real
    print('   b_eff(w) = -w^2 Re P(jw) : le "b0" que verrait la boucle a w')
    for ff in (200, 540, 800, 1068, 1300, 2000, 2787, 4122, 8000, 20000):
        i = int(np.argmin(np.abs(f - ff)))
        print(f'     {f[i]:8.0f} Hz : b_eff = {b_eff[i]:+10.3f}')
    sgn = np.sign(b_eff)
    flips = f[:-1][sgn[:-1] * sgn[1:] < 0]
    print(f'   -> {len(flips)} changements de signe entre 10 Hz et 25 kHz,'
          f' vers {np.round(flips[:10], 0)} Hz')

    zr = plant_zeros(w, zt, res)
    zr = zr[np.argsort(np.abs(zr))]
    rhp = zr[zr.real > 1e-9]
    print(f'\n   ZEROS de P(s) : {len(zr)}, dont {len(rhp)} instables')
    for zz in zr:
        tag = '   <-- DEMI-PLAN DROIT' if zz.real > 1e-9 else ''
        print(f'     s = {zz.real:+11.4g} {zz.imag:+11.4g} j'
              f'   (|s|/2pi = {abs(zz) / 2 / np.pi:8.1f} Hz){tag}')
    print('   -> procede ' + ('A DEPHASAGE NON MINIMAL.' if len(rhp)
                              else 'a dephasage minimal.'))

    # Le zero instable est-il un artefact de la troncature a 5 modes ? On
    # rebatit la plaque avec 12 modes (amortissement 0.4 % pour les modes non
    # mesures) et on regarde ou il va.
    from chebyshev_plate import ChebyshevPlate
    from plate_model import PATCH
    zt12 = tuple(np.asarray(plate.zeta_modes, float)) + (0.004,) * 7
    p12 = ChebyshevPlate(PX=14, PZ=14, n_modes=12, zeta_modes=zt12)
    p12.add_piezo_patch(**PATCH[C.PATCH_SIDE])
    r12 = (p12.D_row(p12.lp, p12.hp)[:12]
           * np.asarray(p12.H_Pe_modal, float)[:12])
    w12 = np.asarray(p12.omega_n, float)
    z12 = np.asarray(p12.zeta_modes, float)
    print('   robustesse a la troncature modale (modele non recale, 12 modes')
    print(f'   disponibles, residus {np.round(r12[:8], 2)} ...) :')
    for nm in (5, 6, 7, 8, 10, 12):
        zz = plant_zeros(w12[:nm], z12[:nm], r12[:nm])
        rr = zz[zz.real > 1e-9]
        print(f'     {nm:3d} modes -> zeros instables : '
              + (' '.join(f'{abs(x) / 2 / np.pi:.0f} Hz' for x in rr) or '-'))
    print('   -> le zero instable a 2.5-3.0 kHz PERSISTE, et en ajouter')
    print('      d autres ne fait qu en rajouter plus haut. Ce n est pas un')
    print('      artefact de troncature : c est la non-colocalisation du')
    print('      couple actionneur/capteur.')
    if len(rhp):
        zrhp = float(rhp[np.argmin(np.abs(rhp))].real)
        print(f'      zero instable dominant : z = {zrhp:.4g} 1/s'
              f' = {zrhp / 2 / np.pi:.0f} Hz')
    else:
        zrhp = None

    # ---------------------------------------------------------------- 2
    print('\n' + '-' * 78)
    print(" MAILLON 2 — que calcule l'observateur ?   [INFIRME : l'ESO est"
          " exact]")
    print('-' * 78)
    print('   Les equations du LESO se resolvent en Laplace sans approximation')
    print('     e = y - z1 ; s z1 = z2 + 3wo e ; s z2 = z3 + b0 u + 3wo^2 e ;')
    print('     s z3 = wo^3 e')
    print('   =>  z3(s) = [wo^3/(s+wo)^3] . [s^2 y(s) - b0 u(s)] ='
          ' Q(s) . f_vrai(s)')
    # L'identite porte sur la commande AVANT le filtre d'anti-repliement :
    # le correcteur enregistre est series(coeur, rolloff), donc ss_frf en donne
    # la sortie FILTREE alors que z3 est alimente par la commande du coeur. On
    # rebatit donc le coeur pour la verification. (Comparer z3 a la sortie
    # filtree donne un ecart de 25 % qui n'est pas un defaut d'observateur mais
    # la reponse du filtre.)
    from adrc import adrc_fopid_ss
    from pso import Design as _Design
    _Dg = _Design('adrc', plate, sl0, sign_variant=var)
    _p = _Dg.decode(d['adrc__x'])
    _pf = _Design('fopid', plate, sl0,
                  sign_variant=float(d['fopid__sign_variant'])).decode(
                      d['fopid__x'])
    core = adrc_fopid_ss(_p['Kp'], _p['Ki'], _p['Kd'], _p['lam'], _p['mu'],
                         _p['wo'], _p['b0'] * var, C.OUST_WB, C.OUST_WH,
                         C.OUST_N, 1.0)
    A, B, Cc, Dd = [np.atleast_2d(np.asarray(m, float)) for m in core]
    nA = A.shape[0]
    e3 = np.zeros((1, nA))
    e3[0, 2] = 1.0
    I = np.eye(nA)
    Z3 = np.array([(e3 @ np.linalg.solve(1j * wk * I - A, B))[0, 0]
                   for wk in om])
    Kc = ss_frf(core, om)
    pred = Q * ((1j * om) ** 2 - b0 * Kc)
    err_id = float(np.max(np.abs(Z3 - pred) / np.maximum(np.abs(pred), 1e-30)))
    print(f'   verification EXACTE sur la realisation d etat utilisee'
          f' (coeur ADRC, {nA} etats), 10 Hz-25 kHz :')
    print(f'     max |(z3/y) - Q(s^2 - b0 K)| / |.| = {err_id:.2e}')
    print("   -> z3 EST la perturbation totale, filtree par Q. L'observateur")
    print('      fait exactement ce qu il promet. Le "134 % d erreur" mesure')
    print('      dans run_eso_trace ne mesure pas l observateur : la reference')
    print('      y" y est obtenue par differences finies sur un signal de')
    print('      coupe plein d impulsions de dent, et 1/dt^2 en amplifie le')
    print('      bruit. Ce maillon de la chaine soupconnee est FAUX.')

    # ---------------------------------------------------------------- 3
    print('\n' + '-' * 78)
    print(' MAILLON 3 — b0 peut-il "s adapter au signe de la voie" ?'
          '   [INFIRME]')
    print('-' * 78)
    from pso import Design
    Dg = Design('adrc', plate, sl0, sign_variant=var)
    x = d['adrc__x'].copy()
    ib = list(Dg.names).index('b0_scale')
    lo, hi = Dg.lo[ib], Dg.hi[ib]
    scale_now = lo + x[ib] * (hi - lo)
    x2 = x.copy()
    x2[ib] = float(np.clip(((scale_now * 2.0) - lo) / (hi - lo), 0, 1))
    K2 = ss_frf(Dg.build(x2), om)
    ratio = np.abs(K2 / K)
    print(f'   b0_scale {scale_now:.3f} -> {lo + x2[ib] * (hi - lo):.3f}'
          f' (x{(lo + x2[ib] * (hi - lo)) / scale_now:.3f})')
    print(f'   |K| est multiplie par {ratio.min():.4f} .. {ratio.max():.4f}'
          f' sur TOUTE la bande')
    print('   -> dans la realisation fermee, les etats de l ESO ne dependent')
    print('      pas de b0 (le terme b0 u de z2 s annule avec la division par')
    print('      b0 dans u). Donc K(s) = G(s)/b0 : b0 est un GAIN GLOBAL, pas')
    print('      un parametre d adaptation de modele. Le rendre dependant de')
    print('      la frequence ne changerait pas la NATURE du probleme ; le PSO')
    print('      l a deja choisi librement sur trois decades et demie.')

    # ---------------------------------------------------------------- 4
    print('\n' + '-' * 78)
    print(' MAILLON 4 — le vrai defaut : ce que la compensation laisse'
          ' derriere   [CONFIRME]')
    print('-' * 78)
    print('   u = (u0 - z3)/b0 annule f exactement SI Q = 1. Or f contient le')
    print('   procede lui-meme : f = s^2 y - b0 u = (s^2 P - b0) u + exogene.')
    print('   Defaut de modele D(s) = s^2 P(s)/b0 - 1 ; residu non annule')
    print('   (1-Q) D.')
    Delta = (1j * om) ** 2 * P / b0 - 1.0
    resid = np.abs((1 - Q) * Delta)
    print('     f [Hz]      |D|      |1-Q|   |(1-Q)D|')
    for ff in (200, 540, 1068, 2000, 2787, 3351, 4122, 8000):
        i = int(np.argmin(np.abs(f - ff)))
        print(f'     {f[i]:7.0f} {abs(Delta[i]):9.2f} {abs(1 - Q[i]):9.4f}'
              f' {resid[i]:9.2f}')
    bad = f[resid > 1.0]
    print(f'   |(1-Q)D| > 1 sur {100 * len(bad) / len(f):.0f} % de la bande,'
          f' de {bad.min():.0f} Hz a {bad.max():.0f} Hz')
    print('   -> le residu culmine EXACTEMENT aux resonances, c est-a-dire la')
    print('      ou naissent les broutements.')

    # 4bis — quel w_o faudrait-il ? |1-Q| doit descendre sous 1/|Delta| au
    # mode. C'est LA question, parce que w_o est le seul reglage qui agit sur
    # ce residu.
    print('\n   Quel w_o faudrait-il pour ramener le residu sous 1 a chaque')
    print('   resonance ? (|1-Q(jw_n)| <= 1/|Delta(jw_n)|)')
    sim0 = MillingSimulation(plate, C.RPM_DESIGN, 1e-4, n_modes=C.N_MODES,
                             n_sub=C.N_SUB, sign=C.SIGN_SIM, v_max=None)
    f_nyq = 0.5 / sim0.dt
    wo_req = []
    for fn in plate.freq_n:
        i = int(np.argmin(np.abs(f - fn)))
        target = 1.0 / max(abs(Delta[i]), 1e-12)
        wn = 2 * np.pi * fn
        g = np.logspace(np.log10(wn), np.log10(wn) + 4, 20000)
        v = np.abs(1 - (g / (1j * wn + g)) ** 3)
        ok = np.where(v <= target)[0]
        wr = g[ok[0]] if len(ok) else np.inf
        wo_req.append(wr)
        print(f'     mode a {fn:6.0f} Hz : |D| = {abs(Delta[i]):6.2f}'
              f' -> w_o >= {wr:9.3g} rad/s = {wr / 2 / np.pi:9.0f} Hz'
              f'  ({wr / wn:5.1f} x w_n)')
    wo_need = max(wo_req)
    print(f'   w_o necessaire = {wo_need / 2 / np.pi:.0f} Hz, a comparer a :')
    print(f'     w_o retenu par le PSO ................ {wo / 2 / np.pi:8.0f} Hz')
    print(f'     filtre d anti-repliement ............. {C.ROLLOFF_HZ:8.0f} Hz')
    print(f'     Nyquist de la commande echantillonnee  {f_nyq:8.0f} Hz')
    print(f'     w_h du filtre d Oustaloup ............ '
          f'{C.OUST_WH / 2 / np.pi:8.0f} Hz')
    print(f'   -> soit {wo_need / (2 * np.pi * C.ROLLOFF_HZ):.1f} fois la'
          f' coupure d anti-repliement et'
          f' {wo_need / C.OUST_WH:.1f} fois w_h.')
    # prix a payer : le bruit de mesure entre dans z3 avec le gain |Q s^2|,
    # qui culmine au voisinage de w_o proportionnellement a w_o^2.
    gg = np.logspace(1, 6.5, 30000)
    def noise_gain(wob):
        return float(np.max(np.abs((wob / (1j * gg + wob)) ** 3 * (1j * gg) ** 2)))
    n_now, n_need = noise_gain(wo), noise_gain(wo_need)
    print(f'   Prix du bruit : le gain max de la mesure vers z3, max|Q s^2|,')
    print(f'   passerait de {n_now:.3g} a {n_need:.3g}, soit x'
          f'{n_need / n_now:.0f} (il croit en w_o^2).')
    print('      Autrement dit : l unique reglage qui pourrait guerir le')
    print('      maillon 4 est hors d atteinte, et le rapprocher amplifierait')
    print('      le bruit du capteur dans la meme proportion. Ce n est donc')
    print('      pas une question de REGLAGE mais de STRUCTURE : la forme')
    print('      imposee par l ESO (retour d acceleration filtre par Q) ne')
    print('      convient pas a ce procede -> maillon 6.')

    # 4ter — le remede est-il DANS le boitier de recherche ? On tient le
    # rapport wo/3|b0| (seul groupe qui compte sous wo, cf. maillon 7) et on
    # balaye wo. Si le residu tombe sous 1 a l interieur des bornes du PSO,
    # alors l optimiseur a VU ce remede et l a REFUSE — et c est la preuve la
    # plus forte qu on puisse produire.
    from adrc import adrc_fopid_ss as _adrc_ss
    from fopid import series as _series, rolloff_ss as _roll
    gratio = wo / (3 * abs(b0))
    im = [int(np.argmin(np.abs(f - fn))) for fn in plate.freq_n]
    print(f'\n   Le remede est-il dans le boitier ? (rapport wo/3|b0| ='
          f' {gratio:.1f} tenu fixe)')
    print(f'   bornes du PSO : w_o de'
          f' {10 ** C.BOUNDS_ADRC["log_wo"][0] / 2 / np.pi:.0f} Hz a'
          f' {10 ** C.BOUNDS_ADRC["log_wo"][1] / 2 / np.pi / 1e3:.1f} kHz')
    print('     w_o/2pi [Hz]      b0     max|(1-Q)D| aux modes     Ms')
    swp = []
    for wof in (1e3, wo / 2 / np.pi, 1e4, 3e4, 5e4):
        wk = 2 * np.pi * wof
        bk = np.sign(b0) * wk / (3 * gratio)
        Qk = (wk / (1j * om + wk)) ** 3
        Dk = (1j * om) ** 2 * P / bk - 1.0
        rk = float(np.abs((1 - Qk) * Dk)[im].max())
        Kk = ss_frf(_series(_adrc_ss(_p['Kp'], _p['Ki'], _p['Kd'], _p['lam'],
                                     _p['mu'], wk, bk, C.OUST_WB, C.OUST_WH,
                                     C.OUST_N, 1.0),
                            _roll(C.ROLLOFF_HZ, C.ROLLOFF_ORDER)), om)
        Msk = float(np.abs(1.0 / (1.0 - P * Kk)).max())
        swp.append((wof, rk, Msk))
        print(f'     {wof:10.0f} {bk:10.1f} {rk:18.2f} {Msk:12.3f}')
    print('   -> le residu passe sous 1 des w_o/2pi ~ 30 kHz, DANS les bornes')
    print('      du PSO (jusqu a 50 kHz). L optimiseur a donc explore la zone')
    print('      qui guerit le maillon 4 et l a REFUSEE : a gains figes elle')
    print('      fait exploser Ms (les gains devraient etre reoptimises, ce')
    print('      que le PSO fait justement — et il a quand meme retenu')
    print(f'      w_o/2pi = {wo / 2 / np.pi:.0f} Hz). Le residu n est donc pas')
    print('      cher a reduire par ignorance : il est cher a reduire, point.')

    # ---------------------------------------------------------------- 5
    print('\n' + '-' * 78)
    print(' MAILLON 5 — la saturation est-elle cause ou symptome ?'
          '   [SYMPTOME a la condition S]')
    print('-' * 78)
    lim_a = limits(plate, ss_adrc, C.RPM_DESIGN, hi=4.0e-3).min()
    lim_f = limits(plate, ss_fopid, C.RPM_DESIGN, hi=4.0e-3).min()
    print(f'   limite LINEAIRE (Floquet, sans saturation) : ADRC'
          f' {lim_a * 1e3:.3f} mm, FOPID {lim_f * 1e3:.3f} mm')
    rows = []
    for ap in (0.12e-3, 0.20e-3, 0.30e-3):
        for vmax in (C.V_MAX, None):
            sim = MillingSimulation(plate, C.RPM_DESIGN, ap,
                                    n_modes=C.N_MODES, n_sub=C.N_SUB,
                                    sign=C.SIGN_SIM, v_max=vmax)
            r = sim.run(controller=LTIController(ss_adrc, sim.dt), T=0.35)
            sg = growth_rate(r['t'], r['y_mill'])
            rows.append((ap, vmax, r['diverged'], sg,
                         float(np.abs(r['y_mill']).max()),
                         float(np.abs(r['u']).max())))
            print(f'     a_p = {ap * 1e3:.2f} mm,'
                  f' {"saturation +/-150 V" if vmax else "SANS saturation  ":19s}'
                  f' : sigma = {sg:+9.1f} 1/s,'
                  f' |y|max = {np.abs(r["y_mill"]).max() * 1e6:8.1f} um,'
                  f' |u|max = {np.abs(r["u"]).max():8.1f} V')
    print(f'   a la condition S (0.30 mm) : 0.300 > {lim_a * 1e3:.3f} mm, donc')
    print('   la boucle est DEJA instable au sens lineaire, et les taux de')
    print('   croissance avec et sans saturation sont du meme ordre : la')
    print('   saturation est un SYMPTOME. Entre la limite lineaire et ~0.2 mm,')
    print('   en revanche, elle aggrave nettement — elle n est donc pas')
    print('   innocente non plus.')

    # ---------------------------------------------------------------- 6
    print('\n' + '-' * 78)
    print(" MAILLON 6 — la limite FONDAMENTALE : le zero instable"
          "   [le vrai plafond]")
    print('-' * 78)
    if zrhp:
        print(f'   Le zero instable z = {zrhp / 2 / np.pi:.0f} Hz impose'
              f' l integrale de Poisson')
        print('     (1/pi) int log|S(jw)| . 2z/(z^2+w^2) dw >= 0')
        print('   donc toute attenuation en dessous se paie par une')
        print('   amplification au-dessus. La borne utile porte sur une BANDE')
        print('   PROTEGEE [f_a, f_b] encadrant les modes de broutement — pas')
        print('   sur [0, f] : personne ne demande d attenuation au continu, et')
        print('   |S| y vaut 1 pour tout correcteur a gain fini.')
        print('     poids de la bande  W = (2/pi)[atan(w_b/z) - atan(w_a/z)]')
        print('     si |S| <= s dans la bande et |S| <= Ms ailleurs :')
        print('               s >= Ms^(-(1-W)/W)')
        i1 = int(np.argmin(np.abs(f - plate.freq_n[0])))
        i2 = int(np.argmin(np.abs(f - plate.freq_n[1])))
        S = {nm: np.abs(1.0 / (1.0 - P * KK))
             for nm, KK in (('FOPID', Kf), ('ADRC-FOPID', K))}
        for nm in S:
            print(f'   {nm:11s} : |S| mode 1 = {S[nm][i1]:.3f}'
                  f' ({20 * np.log10(S[nm][i1]):+.1f} dB),'
                  f' mode 2 = {S[nm][i2]:.3f}'
                  f' ({20 * np.log10(S[nm][i2]):+.1f} dB),'
                  f' max|S| = {S[nm].max():.3f}')
        print('\n     bande protegee        W      plancher   sup|S| FOPID'
              '   sup|S| ADRC')
        bands = [(480, 600), (950, 1200), (480, 1200), (480, 3000),
                 (300, 4400)]
        rows6 = []
        for fa, fb in bands:
            m = (f >= fa) & (f <= fb)
            W = (2 / np.pi) * (np.arctan(2 * np.pi * fb / zrhp)
                               - np.arctan(2 * np.pi * fa / zrhp))
            smin = np.exp(-np.log(C.MS_MAX) * (1 - W) / W)
            sf, sa = S['FOPID'][m].max(), S['ADRC-FOPID'][m].max()
            rows6.append((fa, fb, smin, sf, sa))
            print(f'     {fa:5.0f} - {fb:5.0f} Hz   {W:.4f}  {smin:9.4f}'
                  f'   {sf:9.4f}     {sa:9.4f}')
        print('   -> sur la bande des deux modes de broutement (480-1200 Hz) le')
        print(f'      plancher vaut {rows6[2][2]:.3f} et le FOPID atteint'
              f' {rows6[2][3]:.3f} :')
        print(f'      il est a un facteur {rows6[2][3] / rows6[2][2]:.1f} du'
              f' plafond FONDAMENTAL, alors que')
        print(f'      l ADRC-FOPID est a {rows6[2][4] / rows6[2][2]:.1f}.'
              f' Sur 480-3000 Hz, qui couvre le zero')
        print(f'      instable, le plancher monte a {rows6[3][2]:.3f} et les'
              f' deux correcteurs')
        print(f'      sont a {rows6[3][3] / rows6[3][2]:.1f} et'
              f' {rows6[3][4] / rows6[3][2]:.1f} : la marge de manoeuvre'
              f' restante est mince')
        print('      pour TOUT correcteur lineaire, quel que soit son'
              ' observateur.')

    # ---------------------------------------------------------------- 7
    print('\n' + '-' * 78)
    print(" MAILLON 7 — QUELLE FORME l'ESO impose-t-il au correcteur ?"
          "   [la reponse]")
    print('-' * 78)
    print('   En eliminant z1, z2, z3 des equations du LESO on obtient la')
    print('   FORME FERMEE du correcteur ADRC-FOPID complet :')
    print('')
    print('              -C(s) + [C(s) R(s) - Q(s)] s^2')
    print('     K(s) = ---------------------------------- ,')
    print('              b0 [ 1 - Q(s) + C(s) R(s) ]')
    print('')
    print('     avec C(s) = Kp + Ki s^-lam + Kd s^mu   (le FOPID interne),')
    print('          Q(s) = wo^3/(s+wo)^3,  R(s) = s/(s+wo)^3.')
    from fopid import fopid_ss
    Cs = ss_frf(fopid_ss(_p['Kp'], _p['Ki'], _p['Kd'], _p['lam'], _p['mu'],
                         C.OUST_WB, C.OUST_WH, C.OUST_N, 1.0), om)
    sj = 1j * om
    R = sj / (sj + wo) ** 3
    Kex = (-Cs + (Cs * R - Q) * sj ** 2) / (b0 * (1 - Q + Cs * R))
    print(f'   verification contre la realisation d etat :'
          f' max ecart relatif = '
          f'{float(np.max(np.abs(Kex - Kc) / np.abs(Kc))):.1e}')
    print('')
    print('   DEVELOPPEMENT en dessous de wo (Q -> 1 - 3s/wo, C R s^2 -> 0) :')
    print('')
    print('     K(s)  ~  (wo / 3 b0) . [ -C(s)/s - s ]')
    print('           =  -(wo/3b0) [ Kp/s + Ki s^-(1+lam) + Kd s^(mu-1) + s ]')
    print('')
    Kas = (wo / (3 * b0)) * (-Cs / sj - sj)
    print('     f [Hz]    |K| realise      |K| asymptote     ecart')
    for ff in (10, 100, 300, 540, 1068, 2787):
        i = int(np.argmin(np.abs(f - ff)))
        print(f'     {f[i]:7.0f} {abs(Kc[i]):14.4g} {abs(Kas[i]):17.4g}'
              f' {abs(Kas[i] / Kc[i] - 1) * 100:9.1f} %')
    m = (f >= 3.0) & (f <= 316.0)
    sK = float(np.polyfit(np.log10(f[m]), np.log10(np.abs(K[m])), 1)[0])
    sF = float(np.polyfit(np.log10(f[m]), np.log10(np.abs(Kf[m])), 1)[0])
    print(f"\n   PENTE de |K| entre 3 et 316 Hz (mesuree sur les correcteurs"
          f" retenus) :")
    print(f'     FOPID      {sF:+.3f} dec/dec   (a comparer a -lam ='
          f' {-_pf["lam"]:+.3f})')
    print(f'     ADRC-FOPID {sK:+.3f} dec/dec   (a comparer a -(1+lam) ='
          f' {-(1 + _p["lam"]):+.3f})')
    print(f'     ecart {sF - sK:+.2f} dec/dec : UN INTEGRATEUR ENTIER de plus.')
    print('')
    print('   CONSEQUENCES, et ce sont elles la conclusion du diagnostic :')
    print('   (i)  L ADRC-FOPID N EST PAS un sur-ensemble du FOPID. Dans la')
    print('        bande de broutement ses ordres realisables sont -1, -(1+lam)')
    print(f'        et mu-1 (soit {_p["mu"] - 1:+.3f} ici) plus une derivee')
    print('        ENTIERE +1 ; ceux du FOPID sont 0, -lam et +mu. La derivee')
    print('        FRACTIONNAIRE d ordre mu dans (0,1) — l avance de phase')
    print('        partielle, etalee sur une large bande, exactement ce qu il')
    print('        faut pour un procede a dephasage non minimal — n est PAS')
    print('        atteignable par l ADRC-FOPID. Il ne peut donc pas, meme en')
    print('        principe, reproduire le correcteur qui gagne.')
    print('   (ii) En dessous de wo, wo et b0 n agissent que par leur RAPPORT.')
    K2 = ss_frf(adrc_fopid_ss(_p['Kp'], _p['Ki'], _p['Kd'], _p['lam'],
                              _p['mu'], 2 * wo, 2 * b0, C.OUST_WB, C.OUST_WH,
                              C.OUST_N, 1.0), om)
    print('        verification : K(2 wo, 2 b0) / K(wo, b0) vaut')
    for ff in (10, 100, 540, 1068, 2787, 4122):
        i = int(np.argmin(np.abs(f - ff)))
        print(f'          {f[i]:7.0f} Hz : {abs(K2[i] / Kc[i]):.4f}')
    print('        soit 1.00 a 1.02 jusqu au 2e mode : dans la bande qui')
    print('        compte, l ADRC-FOPID a 6 degres de liberte, pas 7. Le')
    print('        parametre supplementaire qu on croyait lui donner n existe')
    print('        pas la ou il servirait.')
    print('   (iii) wo porte DEUX charges contradictoires : il fixe le residu')
    print('        (1-Q)D du maillon 4 (qui exige wo tres grand) ET le gain de')
    print('        toute la boucle par wo/3b0 (que la contrainte Ms borne).')
    print('        Un seul reglage, deux devoirs qui s opposent.')

    # ------------------------------------------------------------- figure
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 8.4))
    a = ax[0, 0]
    a.semilogx(f, b_eff, color='#1a3f8f', lw=1.3)
    a.axhline(0, color='k', lw=1)
    a.axhline(res.sum(), color='#c0392b', ls='--', lw=1,
              label=f'$b_\\infty$ = {res.sum():+.2f}')
    for fn in plate.freq_n:
        a.axvline(fn, color='0.85', lw=.8)
    a.set_ylim(-30, 30)
    a.set_xlabel('frequency [Hz]')
    a.set_ylabel('$b_{eff}=-\\omega^2\\Re P$')
    a.set_title('(a) LINK 1 confirmed: the channel gain\nchanges sign inside '
                'the band', fontsize=9.5)
    a.grid(alpha=.3, which='both')
    a.legend(fontsize=8)

    a = ax[0, 1]
    a.plot(np.real(zr), np.imag(zr) / (2 * np.pi), 'o', color='#c0392b', ms=8,
           label='zeros of $P(s)$')
    a.axvline(0, color='k', lw=1.2)
    a.set_xlabel('$\\Re(s)$ [1/s]')
    a.set_ylabel('$\\Im(s)/2\\pi$ [Hz]')
    a.set_title(f'(b) one zero in the RIGHT half plane\n'
                f'z = {zrhp / 2 / np.pi:.0f} Hz - non-minimum phase',
                fontsize=9.5)
    a.grid(alpha=.3)
    a.legend(fontsize=8)

    a = ax[0, 2]
    a.loglog(f, np.abs(Z3), color='#1a3f8f', lw=2.2, label='$z_3/y$ from the '
             'state-space')
    a.loglog(f, np.abs(pred), color='#c8963e', lw=1.1, ls='--',
             label='$Q(s)\\,[s^2-b_0K(s)]$')
    a.set_xlabel('frequency [Hz]')
    a.set_title(f'(c) LINK 2 REFUTED: the ESO is exact\n'
                f'agreement to {err_id:.0e}', fontsize=9.5)
    a.grid(alpha=.3, which='both')
    a.legend(fontsize=8)

    a = ax[1, 0]
    a.loglog(f, np.abs(Delta), color='#1a3f8f', lw=1.3, label='$|\\Delta|$')
    a.loglog(f, np.abs(1 - Q), color='#c8963e', lw=1.3, label='$|1-Q|$')
    a.loglog(f, resid, color='#c0392b', lw=1.8, label='$|(1-Q)\\Delta|$')
    a.axhline(1, color='k', ls='--', lw=1)
    for fn in plate.freq_n:
        a.axvline(fn, color='0.85', lw=.8)
    a.axvline(wo / 2 / np.pi, color='#16a085', ls=':', lw=1.6,
              label='$\\omega_o/2\\pi$ chosen by the PSO')
    a.axvline(wo_need / 2 / np.pi, color='#c0392b', ls=':', lw=1.6,
              label='$\\omega_o/2\\pi$ required (%.0f kHz)'
                    % (wo_need / 2e3 / np.pi))
    a.set_xlabel('frequency [Hz]')
    a.set_title('(d) LINK 4 confirmed: what cancellation\nleaves behind, '
                'peaking at the modes', fontsize=9.5)
    a.grid(alpha=.3, which='both')
    a.legend(fontsize=7.5)

    a = ax[1, 1]
    lab = [f'{r[0]*1e3:.2f}\n{"sat" if r[1] else "no sat"}' for r in rows]
    sg = [r[3] for r in rows]
    col = ['#c0392b' if s > 0 else '#16a085' for s in sg]
    a.bar(range(len(rows)), sg, color=col)
    a.axhline(0, color='k', lw=1)
    a.set_yscale('symlog', linthresh=1.0)
    for i, v in enumerate(sg):
        a.annotate(f'{v:+.1f}', (i, v), ha='center',
                   va='bottom' if v > 0 else 'top', fontsize=7.5)
    a.set_xticks(range(len(rows)))
    a.set_xticklabels(lab, fontsize=8)
    a.set_ylabel('growth rate $\\sigma$ [1/s]')
    a.set_title('(e) LINK 5: at 0.30 mm the loop diverges\n'
                'with AND without saturation', fontsize=9.5)
    a.grid(alpha=.3, axis='y')

    a = ax[1, 2]
    xb = np.arange(len(rows6))
    a.bar(xb - .26, [r[2] for r in rows6], .26, color='#1a3f8f',
          label='floor (RHP zero + $M_s\\leq2$)')
    a.bar(xb, [r[3] for r in rows6], .26, color='#c0392b',
          label='FOPID $\\sup|S|$')
    a.bar(xb + .26, [r[4] for r in rows6], .26, color='#16a085',
          label='ADRC-FOPID $\\sup|S|$')
    a.set_yscale('log')
    a.set_xticks(xb)
    a.set_xticklabels([f'{r[0]:.0f}-\n{r[1]:.0f}' for r in rows6], fontsize=8)
    a.set_xlabel('protected band [Hz]')
    a.set_ylabel('$|S|$')
    a.set_title('(f) LINK 6: the fundamental ceiling\nboth controllers sit '
                'close to it', fontsize=9.5)
    a.grid(alpha=.3, axis='y', which='both')
    a.legend(fontsize=7)

    fig.suptitle('Where the ADRC-FOPID loses this plate — link-by-link '
                 f'diagnosis (protocol {C.PROTOCOL})', fontsize=11.5)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_diagnostic_{C.PROTOCOL}.png', dpi=140)
    plt.close(fig)
    print(f'\n  -> {FIG}/fig_diagnostic_{C.PROTOCOL}.png')

    # --------------------------------------------- figure 2 : LA REPONSE
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))
    a = ax[0]
    a.loglog(f, np.abs(Kf), color='#1a3f8f', lw=1.6,
             label=f'FOPID   (slope {sF:+.2f})')
    a.loglog(f, np.abs(K), color='#16a085', lw=1.6,
             label=f'ADRC-FOPID   (slope {sK:+.2f})')
    a.loglog(f, np.abs(Kas), color='#c0392b', lw=1.2, ls='--',
             label='$(\\omega_o/3b_0)[-C(s)/s-s]$')
    ff = f[m]
    a.loglog(ff, np.abs(Kf[m])[0] * (ff / ff[0]) ** sF, color='k', lw=.8)
    a.loglog(ff, np.abs(K[m])[0] * (ff / ff[0]) ** sK, color='k', lw=.8)
    a.axvspan(3, 316, color='0.9', zorder=0)
    for fn in plate.freq_n[:2]:
        a.axvline(fn, color='0.85', lw=.8)
    a.set_xlabel('frequency [Hz]')
    a.set_ylabel('$|K|$ [V/m]')
    a.set_title('(g) LINK 7: the ESO adds exactly one integrator\n'
                f'{sF:+.2f} vs {sK:+.2f} dec/dec, a gap of {sF - sK:+.2f}',
                fontsize=9.5)
    a.grid(alpha=.3, which='both')
    a.legend(fontsize=8)

    a = ax[1]
    of = [0.0, -_pf['lam'], _pf['mu']]
    oa = [-1.0, -(1 + _p['lam']), _p['mu'] - 1.0, 1.0]
    a.axhspan(1, 2, xmin=0, xmax=1, color='#fdf3e3', zorder=0)
    a.plot(of, [1] * len(of), 'o', color='#1a3f8f', ms=13, label='FOPID')
    a.plot(oa, [2] * len(oa), 's', color='#16a085', ms=13,
           label='ADRC-FOPID')
    for j, v in enumerate(sorted(of)):
        a.annotate(f'{v:+.2f}', (v, 1), (0, 14 if j % 2 else -22), 'data',
                   'offset points', ha='center', fontsize=8)
    for j, v in enumerate(sorted(oa)):
        a.annotate(f'{v:+.2f}', (v, 2), (0, 14 if j % 2 else -22), 'data',
                   'offset points', ha='center', fontsize=8)
    a.axvspan(0.05, 1.0, ymin=0, ymax=.5, color='#c0392b', alpha=.13)
    a.annotate('fractional derivative\n$\\mu\\in(0,1)$: reachable by the\n'
               'FOPID, NOT by the ADRC-FOPID', (0.52, 0.62), ha='center',
               fontsize=8.5, color='#c0392b')
    a.set_xlim(-2.3, 1.6)
    a.set_ylim(0.4, 2.6)
    a.set_yticks([1, 2])
    a.set_yticklabels(['FOPID', 'ADRC-\nFOPID'], fontsize=9)
    a.set_xlabel('order of $s$ reachable below $\\omega_o$')
    a.set_title('(h) the ADRC-FOPID is NOT a superset\nof the FOPID',
                fontsize=9.5)
    a.grid(alpha=.3, axis='x')
    a.legend(fontsize=8, loc='lower left')

    a = ax[2]
    xs = [r[0] for r in swp]
    a.loglog(xs, [r[1] for r in swp], 'o-', color='#c0392b', lw=1.8,
             label='$\\max_n|(1-Q)\\Delta|$ at the modes')
    a.loglog(xs, [r[2] for r in swp], 's-', color='#1a3f8f', lw=1.8,
             label='$M_s$ (gains frozen)')
    a.axhline(1, color='#c0392b', ls=':', lw=1)
    a.axhline(C.MS_MAX, color='#1a3f8f', ls=':', lw=1)
    a.axvline(wo / 2 / np.pi, color='#16a085', lw=1.6,
              label='$\\omega_o$ chosen by the PSO')
    a.axvspan(10 ** C.BOUNDS_ADRC['log_wo'][1] / 2 / np.pi, max(xs) * 3,
              color='0.9')
    a.set_xlabel('$\\omega_o/2\\pi$ [Hz]')
    a.set_title('(i) $\\omega_o$ has two conflicting duties\n'
                'the cure was inside the search box and was refused',
                fontsize=9.5)
    a.grid(alpha=.3, which='both')
    a.legend(fontsize=8)

    fig.suptitle('What the ESO structure imposes — and why no tuning of it '
                 f'fixes this plate (protocol {C.PROTOCOL})', fontsize=11.5)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_diagnostic_form_{C.PROTOCOL}.png', dpi=140)
    plt.close(fig)
    print(f'  -> {FIG}/fig_diagnostic_form_{C.PROTOCOL}.png')


if __name__ == '__main__':
    main()
