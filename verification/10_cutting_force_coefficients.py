"""10_cutting_force_coefficients.py — audit du modele d'effort de coupe,
Eqs. (2)-(5) de Du, Liu, Dai, Long, Int. J. Mech. Sci. 274 (2024) 109257.

Contenu :
  1. constantes k1 = kn/cos(eta), k2 = 1 + mu_c tan(eta)(cos g_n - kn sin g_n)
     confrontees aux nombres du Tableau 3 ;
  2. integrales helicoidales analytiques ss, sc, cc de l'Eq. (4) utilisees par
     milling_dynamics.alpha34, confrontees (a) a une reecriture algebrique
     independante des memes primitives et (b) a une QUADRATURE NUMERIQUE
     directe en tranches axiales ;
  3. angles d'entree/sortie en avalant ;
  4. alpha3(t), alpha4(t) sur une periode de dent a la condition S du papier
     (4900 tr/min, ap = 0.3 mm) : forme, rapport cyclique, crete, moyenne, et
     confrontation a la bande [0.3, 2.9] abar4 de la Section 3.2 (Eq. 23) ;
  5. idem a 4300 et 6700 tr/min.

Toutes les valeurs de reference sont celles du papier (Tableaux 1 et 3,
Section 3.2, page 7). Les ecarts sont donnes en relatif.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model')]

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import milling_dynamics as MD

FIGDIR = os.path.join(HERE, '..', 'figures', 'verification')
os.makedirs(FIGDIR, exist_ok=True)
FIGPATH = os.path.abspath(os.path.join(FIGDIR, '10_cutting_force_coefficients.png'))

# ---------------------------------------------------------------------------
# Parametres du papier
# ---------------------------------------------------------------------------
HP = 0.080          # Tableau 1 : hauteur de plaque (usinage du bord superieur)
AP = 0.30e-3        # condition S : profondeur axiale
AE = MD.AE_NOM      # 0.1 mm, avalant
RPM_S = 4900.0      # condition S
NT = MD.N_TEETH
R = MD.R_TOOL
ETA = MD.HELIX
TE = np.tan(ETA)
KT = MD.KT

PHI_ST, PHI_EX = MD.down_milling_angles(AE, R)


# ---------------------------------------------------------------------------
# 2a. Reecriture INDEPENDANTE des integrales de l'Eq. (4)
#     On repart des primitives exactes de sin^2, sin cos, cos^2 le long de z,
#     avec theta(z) = theta0 - z tan(eta)/R :
#        Int sin^2 dz = (z2-z1)/2 + R/(4 te) (sin 2t2 - sin 2t1)
#        Int sin cos dz = R/(4 te) (cos 2t2 - cos 2t1)
#        Int cos^2 dz = (z2-z1)/2 - R/(4 te) (sin 2t2 - sin 2t1)
#     La detection des tranches engagees est faite en z (et non en theta comme
#     dans le module), ce qui donne un second chemin de calcul.
# ---------------------------------------------------------------------------
def helical_analytic(t, Omega, zl, zh, ae=AE, form='sum'):
    """(ss, sc, cc) analytiques. form='sum' : primitives sin2/cos2 ;
    form='paper' : forme produit litterale de l'Eq. (4)."""
    phi_st, phi_ex = MD.down_milling_angles(ae, R)
    a = TE / R
    ss = sc = cc = 0.0
    for j in range(NT):
        th0 = Omega * t + 2 * np.pi * j / NT
        # theta(z) dans [phi_st + 2 pi k, phi_ex + 2 pi k]  <=>  z dans
        #   [ (th0 - phi_ex - 2 pi k)/a , (th0 - phi_st - 2 pi k)/a ]
        kk_lo = int(np.floor((th0 - phi_st - a * zl) / (2 * np.pi))) - 1
        kk_hi = int(np.ceil((th0 - phi_ex - a * zh) / (2 * np.pi))) + 1
        for kk in range(min(kk_lo, kk_hi), max(kk_lo, kk_hi) + 1):
            z1 = max(zl, (th0 - phi_ex - 2 * np.pi * kk) / a)
            z2 = min(zh, (th0 - phi_st - 2 * np.pi * kk) / a)
            if z2 <= z1:
                continue
            t1 = th0 - a * z1
            t2 = th0 - a * z2
            if form == 'sum':
                d = R / (4.0 * TE) * (np.sin(2 * t2) - np.sin(2 * t1))
                ss += 0.5 * (z2 - z1) + d
                sc += R / (4.0 * TE) * (np.cos(2 * t2) - np.cos(2 * t1))
                cc += 0.5 * (z2 - z1) - d
            else:                                    # forme produit Eq. (4)
                d = R / (2.0 * TE) * np.sin(t2 - t1) * np.cos(t2 + t1)
                ss += 0.5 * (z2 - z1) + d
                sc += R / (2.0 * TE) * np.sin(t1 - t2) * np.sin(t1 + t2)
                cc += 0.5 * (z2 - z1) - d
    return ss, sc, cc


def alpha34_from_integrals(ss, sc, cc):
    """Eq. (3) : alpha3 = k2 kt ss - k1 kt sc ; alpha4 = k2 kt sc - k1 kt cc."""
    return MD.K2 * KT * ss - MD.K1 * KT * sc, MD.K2 * KT * sc - MD.K1 * KT * cc


# ---------------------------------------------------------------------------
# 2b. Quadrature numerique directe : tranches axiales, regle du point milieu
# ---------------------------------------------------------------------------
def helical_quadrature(t, Omega, zl, zh, nslice, ae=AE):
    """(ss, sc, cc) par sommation sur nslice tranches dz de la dent."""
    phi_st, phi_ex = MD.down_milling_angles(ae, R)
    dz = (zh - zl) / nslice
    z = zl + (np.arange(nslice) + 0.5) * dz
    ss = sc = cc = 0.0
    for j in range(NT):
        th = Omega * t + 2 * np.pi * j / NT - z * TE / R
        m = np.mod(th, 2 * np.pi)
        g = (m >= phi_st) & (m <= phi_ex)
        if not g.any():
            continue
        s = np.sin(th[g])
        c = np.cos(th[g])
        ss += float(np.dot(s, s)) * dz
        sc += float(np.dot(s, c)) * dz
        cc += float(np.dot(c, c)) * dz
    return ss, sc, cc


# ===========================================================================
sep = '=' * 78
print(sep)
print(' SCRIPT 10 — MODELE D EFFORT DE COUPE / CUTTING FORCE MODEL, Eqs. (2)-(5)')
print('   Du, Liu, Dai, Long, Int. J. Mech. Sci. 274 (2024) 109257')
print(sep)

# ---------------------------------------------------------------------------
# 1. Constantes k1, k2
# ---------------------------------------------------------------------------
print('\n[1] Eq. (3) : constantes k1 et k2 (Tableau 3)')
kn, muc = MD.KN, MD.MU_C
eta_d, gam_d = np.rad2deg(MD.HELIX), np.rad2deg(MD.RAKE)
k1_ref = kn / np.cos(np.deg2rad(eta_d))
k2_ref = 1.0 + muc * np.tan(np.deg2rad(eta_d)) * (np.cos(np.deg2rad(gam_d))
                                                  - kn * np.sin(np.deg2rad(gam_d)))
print('    entrees Tableau 3 : kn = %.4f   mu_c = %.4f   eta = %.2f deg   '
      'gamma_n = %.2f deg   kt = %.1f MPa' % (kn, muc, eta_d, gam_d, KT / 1e6))
print('    %-34s %-14s %-14s %s' % ('grandeur', 'module', 'formule Tab.3', 'ecart rel.'))
for name, mod, ref in (('k1 = kn/cos(eta)', MD.K1, k1_ref),
                       ('k2 = 1+mu tan(eta)(cos g -kn sin g)', MD.K2, k2_ref)):
    print('    %-34s %-14.6f %-14.6f %.2e' % (name, mod, ref, abs(mod - ref) / abs(ref)))
print('    -> k1 = %.6f, k2 = %.6f  (valeurs a 6 decimales demandees)' % (MD.K1, MD.K2))

# ---------------------------------------------------------------------------
# 3. Angles d'entree/sortie (place ici car utilises par la suite)
# ---------------------------------------------------------------------------
print('\n[2] Angles d entree / sortie en avalant (down milling), ae = %.3f mm, R = %.3f mm'
      % (AE * 1e3, R * 1e3))
dlt = np.arccos(1.0 - AE / R)
print('    phi_st = pi - acos(1 - ae/R) = %.6f rad = %.4f deg' % (PHI_ST, np.rad2deg(PHI_ST)))
print('    phi_ex = pi                  = %.6f rad = %.4f deg' % (PHI_EX, np.rad2deg(PHI_EX)))
print('    largeur angulaire d immersion  Delta = acos(1-ae/R) = %.6f rad = %.4f deg'
      % (dlt, np.rad2deg(dlt)))
print('    immersion radiale ae/R = %.4f  -> Delta/(2 pi/N_T) = %.2f %% du pas de dent'
      % (AE / R, 100 * dlt / (2 * np.pi / NT)))
print('    retard helicoidal sur ap = %.2f mm : ap tan(eta)/R = %.6f rad = %.4f deg'
      % (AP * 1e3, AP * TE / R, np.rad2deg(AP * TE / R)))

# ---------------------------------------------------------------------------
# 2. Integrales helicoidales : analytique vs quadrature
# ---------------------------------------------------------------------------
print('\n[3] Eq. (4) : integrales helicoidales ss, sc, cc')
Om_S = 2 * np.pi * RPM_S / 60.0
tau_S = 60.0 / (NT * RPM_S)
zl, zh = HP - AP, HP

# 3.1 coherence module <-> Eq. (3)/(4) reecrite
nt_chk = 601
tchk = np.linspace(0.0, tau_S, nt_chk, endpoint=False) + 0.5 * tau_S / nt_chk
e_paper = e_sum = 0.0
ref_scale = 0.0
for tt in tchk:
    a3m, a4m = MD.alpha34(tt, Om_S, zl, zh, AE)
    a3p, a4p = alpha34_from_integrals(*helical_analytic(tt, Om_S, zl, zh, AE, 'paper'))
    a3s, a4s = alpha34_from_integrals(*helical_analytic(tt, Om_S, zl, zh, AE, 'sum'))
    e_paper = max(e_paper, abs(a3m - a3p), abs(a4m - a4p))
    e_sum = max(e_sum, abs(a3m - a3s), abs(a4m - a4s))
    ref_scale = max(ref_scale, abs(a3m), abs(a4m))
print('    (a) module milling_dynamics.alpha34 vs reecriture litterale Eq. (4)')
print('        max|delta alpha| / max|alpha| = %.2e  (forme produit Eq. 4)'
      % (e_paper / ref_scale))
print('        max|delta alpha| / max|alpha| = %.2e  (primitives sin2/cos2, '
      'detection des tranches en z)' % (e_sum / ref_scale))
print('        -> le module implante exactement les Eqs. (3)-(4) ; les deux formes')
print('           algebriques des primitives coincident a la precision machine.')
# test de robustesse du reperage des tranches engagees (enroulement helicoidal
# sur plusieurs tours quand ap est grand : ap tan(eta)/R = 5.6 rad a ap = 40 mm)
print('        test d enroulement (detection des intervalles engages), 313 instants :')
print('        %-10s' % 'ap \\ ae' + ''.join('%-11s' % ('%.1f mm' % (a * 1e3))
                                            for a in (0.1e-3, 1e-3, 5e-3, 9e-3)))
worst_wrap = 0.0
for ap_t in (0.3e-3, 2e-3, 10e-3, 40e-3):
    line = '        %-10s' % ('%.1f mm' % (ap_t * 1e3))
    for ae_t in (0.1e-3, 1e-3, 5e-3, 9e-3):
        em = sc_ = 0.0
        for tt2 in np.linspace(0.0, tau_S, 313):
            m = MD.alpha34(tt2, Om_S, HP - ap_t, HP, ae_t)
            rr = alpha34_from_integrals(*helical_analytic(tt2, Om_S, HP - ap_t, HP,
                                                          ae_t, 'sum'))
            em = max(em, abs(m[0] - rr[0]), abs(m[1] - rr[1]))
            sc_ = max(sc_, abs(rr[0]), abs(rr[1]))
        line += '%-11.1e' % (em / sc_)
        worst_wrap = max(worst_wrap, em / sc_)
    print(line)
print('        -> ecart relatif maximal %.1e sur toute la grille (a ap = 40 mm la dent'
      % worst_wrap)
print('           s enroule de %.2f rad = %.2f pas de dent, donc jusqu a %d fenetres'
      % (40e-3 * TE / R, 40e-3 * TE / R / (2 * np.pi / NT),
         int(np.ceil(40e-3 * TE / R / (2 * np.pi / NT))) + 1))
print('           d engagement simultanees sur une meme dent) :')
print('           la logique de reperage modulo 2 pi du module est correcte.')

# 3.2 quadrature numerique directe
print('\n    (b) quadrature directe en tranches axiales dz sur [hp-ap, hp],')
print('        theta(z) = Omega t + 2 pi j / N_T - z tan(eta)/R, fenetre [phi_st, phi_ex] mod 2 pi')
print('        condition : %.0f tr/min, ap = %.2f mm, ae = %.2f mm, avalant ; '
      '%d instants sur une periode de dent' % (RPM_S, AP * 1e3, AE * 1e3, 96))

nt_cv = 96
tcv = np.linspace(0.0, tau_S, nt_cv, endpoint=False) + 0.5 * tau_S / nt_cv
ana = np.array([helical_analytic(tt, Om_S, zl, zh, AE, 'paper') for tt in tcv])
peak = np.abs(ana).max(axis=0)
nslices = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
errs = np.zeros((len(nslices), 3))       # max_t normalise par la crete
errs_rms = np.zeros((len(nslices), 3))   # rms_t normalise par la crete
errs_pw = np.zeros((len(nslices), 3))    # max relatif ponctuel (|ana| > 1 % crete)
for i, ns in enumerate(nslices):
    q = np.array([helical_quadrature(tt, Om_S, zl, zh, ns, AE) for tt in tcv])
    d = np.abs(q - ana)
    errs[i] = d.max(axis=0) / peak
    errs_rms[i] = np.sqrt((d**2).mean(axis=0)) / peak
    big = np.abs(ana) > 0.01 * peak
    with np.errstate(divide='ignore', invalid='ignore'):
        rp = np.where(big, d / np.abs(ana), 0.0)
    errs_pw[i] = rp.max(axis=0)
print('        %-10s | %-9s %-9s %-9s | %-9s %-9s %-9s | %-9s'
      % ('n tranches', 'ss max', 'sc max', 'cc max', 'ss rms', 'sc rms', 'cc rms',
         'max ponct.'))
for i, ns in enumerate(nslices):
    print('        %-10d | %-9.2e %-9.2e %-9.2e | %-9.2e %-9.2e %-9.2e | %-9.2e'
          % (ns, errs[i, 0], errs[i, 1], errs[i, 2],
             errs_rms[i, 0], errs_rms[i, 1], errs_rms[i, 2], errs_pw[i].max()))
print('        "max"/"rms" = max_t (resp. rms_t) |quad - analytique| / max_t |analytique|')
print('        "max ponct." = max_t |quad - analytique| / |analytique| sur les instants ou')
print('                       |analytique| > 1 % de sa crete.')
print('        L integrande est DISCONTINU en z (fenetre d engagement) : la seule source')
print('        d erreur est la cellule partielle a chaque bord, donc O(1/n) et non O(1/n^2),')
print('        avec une constante qui depend de la position du bord dans la cellule. Le max')
print('        sur un echantillon fini d instants est donc bruite (non monotone) ; la rms')
print('        suit proprement la pente -1.')
for j, nm in enumerate(('ss', 'sc', 'cc')):
    print('        pente log-log rms (%s) = %+.2f  (attendu -1.00)'
          % (nm, np.polyfit(np.log(nslices), np.log(errs_rms[:, j]), 1)[0]))
print('        -> a %d tranches : erreur normalisee max <= %.1e, rms <= %.1e ;'
      % (nslices[-1], errs[-1].max(), errs_rms[-1].max()))
print('           l ecart analytique/quadrature est donc purement numerique : l Eq. (4)')
print('           telle qu implantee EST bien l integrale de sin^2, sin cos, cos^2 sur dz.')
i5 = nslices.index(10000)
print('        (repere demande "~1e-5 ou mieux" : atteint des n = %d, max %.1e)'
      % (nslices[i5], errs[i5].max()))

# ---------------------------------------------------------------------------
# 4-5. alpha3(t), alpha4(t) : forme, rapport cyclique, moyennes, bande Eq. (23)
# ---------------------------------------------------------------------------
print('\n[4] alpha3(t) et alpha4(t) sur une periode de dent — usinage du bord superieur')
print('    (hp = %.0f mm, ap = %.2f mm, ae = %.2f mm, avalant, kt = %.0f MPa)'
      % (HP * 1e3, AP * 1e3, AE * 1e3, KT / 1e6))

NS = 20000                      # echantillons par periode de dent
duty_ana = (dlt + AP * TE / R) / (2 * np.pi / NT)

rows = []
for rpm in (4300.0, RPM_S, 6700.0):
    a3, a4 = MD.alpha4_series(rpm, AP, HP, NS, AE)
    tau = 60.0 / (NT * rpm)
    duty = float(np.mean(a4 != 0.0))
    ab3, ab4 = float(a3.mean()), float(a4.mean())
    pk3 = float(a3[np.argmax(np.abs(a3))])
    pk4 = float(a4[np.argmax(np.abs(a4))])
    rows.append(dict(rpm=rpm, tau=tau, duty=duty, ab3=ab3, ab4=ab4,
                     pk3=pk3, pk4=pk4, a3=a3, a4=a4))

print('\n    %-7s %-9s %-9s %-11s %-11s %-9s %-12s %-12s %-9s'
      % ('rpm', 'tau (ms)', 'duty (%)', 'abar3 (N/m)', 'alpha3 pic', 'pic/abar3',
         'abar4 (N/m)', 'alpha4 pic', 'pic/abar4'))
for r in rows:
    print('    %-7.0f %-9.4f %-9.4f %-11.2f %-11.1f %-9.3f %-12.2f %-12.1f %-9.3f'
          % (r['rpm'], r['tau'] * 1e3, 100 * r['duty'], r['ab3'], r['pk3'],
             r['pk3'] / r['ab3'], r['ab4'], r['pk4'], r['pk4'] / r['ab4']))
print('    (alpha3 > 0, alpha4 < 0 avec la convention d ecriture de l Eq. (12)-(13) ;')
print('     "pic" = valeur de plus grand module sur la periode.)')
print('    rapport cyclique analytique = (Delta + ap tan(eta)/R)/(2 pi / N_T) = %.4f %%'
      % (100 * duty_ana))
print('    ecart rel. sur le rapport cyclique (numerique vs analytique) = %.2e'
      % (abs(rows[1]['duty'] - duty_ana) / duty_ana))

sp = np.array([r['ab4'] for r in rows])
print('\n    abar4 est IDENTIQUE aux trois vitesses (ecart max %.2e en relatif) :'
      % (np.ptp(sp) / abs(sp.mean())))
print('    alpha4 ne depend du temps que par Omega t, donc sa moyenne sur une periode')
print('    de dent est une moyenne ANGULAIRE, independante de la vitesse de broche.')
print('    Seule l echelle de temps (tau) change. Consequence pratique : les courbes')
print('    0.3x / 1x / 2.9x abar4 de la Fig. 6 utilisent un meme abar4 pour tout le')
print('    domaine de vitesses (a ap et ae fixes).')

# convergence de la moyenne (defaut n = 246 dans alpha4_average)
print('\n    convergence de la moyenne temporelle (echantillonnage point milieu) :')
ab_ref = MD.alpha4_average(RPM_S, AP, HP, AE, n=200000)
print('        %-9s %-14s %-10s' % ('n echant.', 'abar4 (N/m)', 'ecart rel.'))
for n in (82, 246, 1000, 5000, 20000):
    v = MD.alpha4_average(RPM_S, AP, HP, AE, n=n)
    print('        %-9d %-14.4f %.2e' % (n, v, abs(v - ab_ref) / abs(ab_ref)))
print('        (n = 246 est le defaut de milling_dynamics.alpha4_average ; ecart %.1e)'
      % (abs(MD.alpha4_average(RPM_S, AP, HP, AE, n=246) - ab_ref) / abs(ab_ref)))

# ---------------------------------------------------------------------------
# Bande d'incertitude Eq. (23)
# ---------------------------------------------------------------------------
r = rows[1]
ab4 = r['ab4']
print('\n[5] Bande d incertitude de la Section 3.2 (Eq. 23), condition S : %.0f tr/min, ap = %.2f mm'
      % (RPM_S, AP * 1e3))
ab4_mod = MD.alpha4_average(RPM_S, AP, HP, AE)     # defaut n = 246 du module
print('    abar4  (n = %d echant.)      = %+12.3f N/m' % (NS, ab4))
print('    abar4  (defaut module n=246) = %+12.3f N/m   ecart %.2e'
      % (ab4_mod, abs(ab4_mod - ab4) / abs(ab4)))
print('    alpha40   = 1.6 abar4        = %+12.3f N/m' % (1.6 * ab4))
print('    L_Palpha  = 1.3 abar4        = %+12.3f N/m' % (1.3 * ab4))
print('    bande alpha40 +/- L_Palpha = [0.3, 2.9] abar4')
print('      -> abar4 < 0, donc en valeurs signees : [%+.3f, %+.3f] N/m'
      % (2.9 * ab4, 0.3 * ab4))
print('      -> en module                          : [%.3f, %.3f] N/m'
      % (abs(0.3 * ab4), abs(2.9 * ab4)))
rat = r['a4'] / ab4
rat[rat == 0.0] = 0.0                              # supprime les -0.0
print('\n    rapport instantane alpha4(t)/abar4 sur une periode de dent :')
print('        max_t alpha4(t)/abar4 = %8.4f' % rat.max())
print('        min_t alpha4(t)/abar4 = %8.4f  (alpha4 = 0 hors engagement)' % rat.min())
print('        fraction du temps avec alpha4 != 0        : %.2f %%' % (100 * r['duty']))
print('        fraction du temps dans [0.3, 2.9] abar4   : %.2f %%'
      % (100 * np.mean((rat >= 0.3) & (rat <= 2.9))))
print('        fraction du temps > 2.9 abar4             : %.2f %%' % (100 * np.mean(rat > 2.9)))
print('        fraction du temps < 0.3 abar4             : %.2f %%' % (100 * np.mean(rat < 0.3)))
print('\n    LECTURE HONNETE DU TEXTE (page 7). Le papier ecrit : "the stability results')
print('    predicted with actual milling force coefficient ... are higher than that with')
print('    2.9 times of average milling force coefficient and lower than that with 0.3')
print('    times ... in all spindle speeds. Thus the milling force coefficient alpha4(t)')
print('    can be regarded as varying within 0.3 abar4 ~ 2.9 abar4."')
print('    La justification porte donc sur les COURBES DE STABILITE (Fig. 6) : les lobes')
print('    calcules avec le coefficient reel periodique sont encadres par ceux calcules')
print('    avec 0.3 abar4 et 2.9 abar4. Ce n est PAS une borne sur le coefficient')
print('    instantane. Notre calcul le confirme sans ambiguite : a %.0f tr/min / ap = %.2f mm,'
      % (RPM_S, AP * 1e3))
print('    alpha4(t) vaut exactement 0 pendant %.1f %% de la periode et atteint %.2f x abar4'
      % (100 * (1 - r['duty']), rat.max()))
print('    en crete, soit %.2f fois la borne superieure 2.9 de l Eq. (23). L intervalle'
      % (rat.max() / 2.9))
print('    [0.3, 2.9] abar4 est donc un intervalle EQUIVALENT AU SENS DE LA STABILITE, et')
print('    non une enveloppe de alpha4(t) — enonce a conserver tel quel dans la')
print('    reimplantation. VERDICT : la bande de l Eq. (23) est CONFIRMEE comme lue par')
print('    le papier (encadrement des lobes), et DEMENTIE comme borne instantanee.')
print('    (a immersion radiale faible, ae/R = %.3f, ce facteur de crete est inevitable :'
      % (AE / R))
print('     pic/moyenne ~ 1/duty = %.2f, a comparer a %.2f mesure.)'
      % (1.0 / r['duty'], rat.max()))

# ---------------------------------------------------------------------------
# FIGURE
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(13.0, 9.2))

# (a) alpha3(t) et alpha4(t) sur une periode de dent
tt = np.linspace(0.0, 1.0, NS, endpoint=False) + 0.5 / NS
eng = r['a4'] != 0.0
i0, i1 = int(np.argmax(eng)), int(NS - np.argmax(eng[::-1]))
w0, w1 = tt[i0] - 0.03, tt[i1 - 1] + 0.03

a0 = ax[0, 0]
a0.fill_between(tt, -152, 34, where=eng, color='0.85', step='mid',
                label='tooth engaged (%.2f %% duty cycle)' % (100 * r['duty']))
a0.plot(tt, r['a3'] / 1e3, lw=1.7, color='tab:blue', label=r'$\alpha_3(t)$')
a0.plot(tt, r['a4'] / 1e3, lw=1.7, color='tab:red', label=r'$\alpha_4(t)$')
a0.axhline(r['ab3'] / 1e3, ls='--', lw=1.1, color='tab:blue',
           label=r'$\bar\alpha_3$ = %+.2f kN/m' % (r['ab3'] / 1e3))
a0.axhline(r['ab4'] / 1e3, ls='--', lw=1.1, color='tab:red',
           label=r'$\bar\alpha_4$ = %+.2f kN/m' % (r['ab4'] / 1e3))
a0.set_xlim(0, 1)
a0.set_ylim(-152, 34)
a0.set_xlabel('time over one tooth period  $t/\\tau$   ($\\tau$ = %.4f ms)' % (r['tau'] * 1e3))
a0.set_ylabel('milling force coefficient  [kN/m]')
a0.set_title('(a) Eq. (3) coefficients, %.0f rpm, $a_p$ = %.2f mm, $a_e$ = %.2f mm (down milling)'
             % (RPM_S, AP * 1e3, AE * 1e3), fontsize=10)
a0.legend(fontsize=8, loc='upper left')
a0.grid(alpha=0.3)
azi = a0.inset_axes([0.055, 0.07, 0.40, 0.44])
azi.fill_between(tt, -152, 34, where=eng, color='0.85', step='mid')
azi.plot(tt, r['a3'] / 1e3, lw=1.5, color='tab:blue')
azi.plot(tt, r['a4'] / 1e3, lw=1.5, color='tab:red')
azi.set_xlim(w0, w1)
azi.set_ylim(-152, 34)
azi.tick_params(labelsize=6.5)
azi.set_title('zoom on the engagement pulse', fontsize=7.5)
azi.grid(alpha=0.3)

# (b) alpha4 / abar4 face a la bande de l'Eq. (23)
a1 = ax[0, 1]
a1.fill_between([0, 1], 0.3, 2.9, color='tab:purple', alpha=0.10,
                label=r'Eq. (23) band  [0.3, 2.9] $\bar\alpha_4$')
a1.plot(tt, rat, lw=1.7, color='tab:red', label=r'$\alpha_4(t)/\bar\alpha_4$')
for lev, col, lab in ((0.3, 'tab:green', r'0.3 $\bar\alpha_4$  (lower)'),
                      (1.0, 'k', r'1.0 $\bar\alpha_4$  (average)'),
                      (1.6, 'tab:orange', r'1.6 $\bar\alpha_4$ = $\alpha_{40}$'),
                      (2.9, 'tab:purple', r'2.9 $\bar\alpha_4$  (upper)')):
    a1.axhline(lev, ls='--', lw=1.2, color=col, label=lab)
a1.annotate('peak = %.2f $\\bar\\alpha_4$\n= %.2f x the 2.9 bound' % (rat.max(), rat.max() / 2.9),
            xy=(tt[int(np.argmax(rat))], rat.max()),
            xytext=(0.22, 0.90 * rat.max()), fontsize=9,
            arrowprops=dict(arrowstyle='->', lw=1.0))
a1.annotate('$\\alpha_4$ = 0 over %.1f %% of the period' % (100 * (1 - r['duty'])),
            xy=(0.30, 0.0), xytext=(0.10, 1.9), fontsize=8.5,
            arrowprops=dict(arrowstyle='->', lw=0.9))
a1.set_xlim(0, 1)
a1.set_ylim(-0.6, 1.38 * rat.max())
a1.set_xlabel('time over one tooth period  $t/\\tau$')
a1.set_ylabel(r'$\alpha_4(t)\,/\,\bar\alpha_4$')
a1.set_title('(b) instantaneous coefficient vs the Sec. 3.2 band  [0.3, 2.9] $\\bar\\alpha_4$',
             fontsize=10)
a1.legend(fontsize=8, loc='upper right')
a1.grid(alpha=0.3)

# (c) convergence de la quadrature
a2 = ax[1, 0]
for i, (lab, col) in enumerate((('ss', 'tab:blue'), ('sc', 'tab:orange'), ('cc', 'tab:green'))):
    a2.loglog(nslices, errs_rms[:, i], 'o-', color=col, lw=1.5, ms=4, label='%s  (rms)' % lab)
    a2.loglog(nslices, errs[:, i], '^:', color=col, lw=0.9, ms=4, alpha=0.6,
              label='%s  (max)' % lab)
ref = errs_rms[0, 0] * nslices[0] / np.array(nslices, float)
a2.loglog(nslices, ref, 'k--', lw=1.0, label=r'slope $-1$ reference')
a2.axhline(1e-5, color='0.4', lw=0.8, ls='-.')
a2.text(nslices[0] * 1.1, 1.2e-5, '$10^{-5}$ target', fontsize=7.5, color='0.4')
a2.set_xlabel('number of axial slices $n$ over $[h_p-a_p,\\ h_p]$')
a2.set_ylabel(r'$\max_t |{\rm quad}-{\rm analytic}| \,/\, \max_t|{\rm analytic}|$')
a2.set_title('(c) analytic Eq. (4) vs direct axial quadrature (%.0f rpm, %d instants)'
             % (RPM_S, nt_cv),
             fontsize=10)
a2.legend(fontsize=7, ncol=2)
a2.grid(alpha=0.3, which='both')

# (d) geometrie d'engagement
a3ax = ax[1, 1]
RMM = R * 1e3
XR, YB = 1.62 * RMM, -1.58 * RMM


def draw_geometry(axg, lw_arc=3.0):
    th = np.linspace(0, 2 * np.pi, 721)
    axg.plot(RMM * np.sin(th), RMM * np.cos(th), color='0.55', lw=1.0, ls='--')
    ph = np.linspace(PHI_ST, PHI_EX, 300)
    xi, yi = RMM * np.sin(PHI_ST), RMM * np.cos(PHI_ST)
    xo, yo = RMM * np.sin(PHI_EX), RMM * np.cos(PHI_EX)
    px = np.concatenate(([-XR, 0.0], RMM * np.sin(ph[::-1]), [XR]))
    py = np.concatenate(([yo, yo], RMM * np.cos(ph[::-1]), [yi]))
    axg.fill_between(px, YB, py, color='tab:blue', alpha=0.20, zorder=0)
    axg.plot(px, py, color='tab:blue', lw=1.5, zorder=4)
    axg.plot(RMM * np.sin(ph), RMM * np.cos(ph), color='tab:red', lw=lw_arc,
             solid_capstyle='butt', zorder=5)
    axg.plot([-XR, XR], [yi, yi], color='0.6', lw=0.8, ls=':', zorder=3)
    axg.plot(xi, yi, 'o', color='tab:red', ms=6, zorder=6)
    axg.plot(xo, yo, 's', color='tab:red', ms=6, zorder=6)
    axg.set_xlim(-XR, XR)
    axg.set_ylim(YB, 1.32 * RMM)
    axg.set_aspect('equal')
    return xi, yi, xo, yo


xin, yin, xout, yout = draw_geometry(a3ax)
a3ax.plot([0], [0], 'k+', ms=9)
a3ax.annotate('', xy=(0.0, 0.0), xytext=(RMM * np.sin(0.62), RMM * np.cos(0.62)),
              arrowprops=dict(arrowstyle='<-', lw=1.0, color='0.35'))
a3ax.text(1.30, 2.35, '$R_T$ = %.1f mm' % RMM, fontsize=9, color='0.35')
arc = np.linspace(0.75, 2.35, 80)
a3ax.plot(0.42 * RMM * np.sin(arc), 0.42 * RMM * np.cos(arc), color='0.35', lw=1.0)
a3ax.annotate('', xy=(0.42 * RMM * np.sin(2.35), 0.42 * RMM * np.cos(2.35)),
              xytext=(0.42 * RMM * np.sin(2.22), 0.42 * RMM * np.cos(2.22)),
              arrowprops=dict(arrowstyle='->', lw=1.1, color='0.35'))
a3ax.text(3.05, -1.15, r'$\theta$ increases', fontsize=8, color='0.35')
a3ax.text(0.30 * RMM, -0.80 * RMM,
          'radial immersion $a_e$ = %.2f mm\n(= %.1f %% of $R_T$ — see zoom)' % (AE * 1e3, 100 * AE / R),
          fontsize=8, color='tab:green', ha='center')
a3ax.annotate('feed  $X_T$', xy=(0.98 * XR, 1.12 * RMM),
              xytext=(0.28 * XR, 1.12 * RMM), fontsize=9,
              arrowprops=dict(arrowstyle='->', lw=1.3, color='k'))
a3ax.text(-XR + 0.25, YB + 0.20,
          'workpiece (plate) — machined wall at $R_T$, uncut wall at $R_T-a_e$',
          fontsize=7.5, color='tab:blue', va='bottom')
a3ax.set_xlabel('$X_T$  [mm]')
a3ax.set_ylabel('$Y_T$  [mm]')
a3ax.set_title('(d) down-milling engagement geometry, $a_e/R_T$ = %.3f' % (AE / R), fontsize=10)

zb = (-0.55, -5.07, 2.10, 0.24)      # (x0, y0, dx, dy) region zoomee
a3ax.add_patch(plt.Rectangle(zb[:2], zb[2], zb[3], fill=False, ec='0.35', lw=1.0, zorder=7))
axins = a3ax.inset_axes([0.03, 0.635, 0.45, 0.30])
draw_geometry(axins, lw_arc=2.4)
axins.set_xlim(zb[0], zb[0] + zb[2])
axins.set_ylim(zb[1], zb[1] + zb[3])
axins.set_aspect('auto')
axins.set_xticks([])
axins.set_yticks([])
axins.annotate('entry  $\\varphi_{st}$ = %.2f$^\\circ$\n($h$ = $f_z\\sin\\varphi$ max)'
               % np.rad2deg(PHI_ST), xy=(xin, yin), xytext=(0.50, -5.045), fontsize=6.5,
               arrowprops=dict(arrowstyle='->', lw=0.8))
axins.annotate('exit  $\\varphi_{ex}$ = 180$^\\circ$\n($h$ = 0)',
               xy=(xout, yout), xytext=(-0.50, -4.925), fontsize=6.5,
               arrowprops=dict(arrowstyle='->', lw=0.8))
axins.annotate('', xy=(1.28, yout), xytext=(1.28, yin),
               arrowprops=dict(arrowstyle='<->', lw=1.2, color='tab:green'))
axins.text(1.31, 0.5 * (yin + yout), '$a_e$', fontsize=8, color='tab:green', va='center')
axins.set_title('zoom (vertical scale exaggerated)', fontsize=7.5)
for sp_ in axins.spines.values():
    sp_.set_color('0.35')

fig.tight_layout()
fig.savefig(FIGPATH, dpi=140)
print('\nfigure : %s' % FIGPATH)
print(sep)
