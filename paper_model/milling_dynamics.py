"""
milling_dynamics.py — Efforts de coupe (Eqs. 2-5), modele reduit a deux modes
(Eq. 21) et parametres d'incertitude (Eqs. 22-25), dans la remodelisation
Chebyshev du papier IJMS 274 (2024) 109257.

Les formules d'engagement (ss, sc, cc, Eq. 4) sont celles de Balachandran &
Zhao [74], integrees analytiquement dent par dent — memes expressions que dans
milling_plant.py fourni, reecrites ici pour que le paquet soit autonome et que
les deux implantations puissent etre confrontees.

Parametres outil (Tableau 3) : 3 dents, D 10 mm, helice 35 deg, coupe normale
15 deg, kt = 925 MPa, kn = 0.26, mu_c = 0.2. Conditions nominales : avalant,
ae = 0.1 mm, fz = 0.02 mm/dent.
"""
import numpy as np

# ---- Tableau 3 -------------------------------------------------------------
N_TEETH = 3
D_TOOL = 0.010
R_TOOL = D_TOOL / 2
HELIX = np.deg2rad(35.0)
RAKE = np.deg2rad(15.0)
KT, KN, MU_C = 925e6, 0.26, 0.20
AE_NOM, FZ_NOM = 0.1e-3, 0.02e-3

# ---- Eq. (3) : k1 = kn/cos(eta), k2 = 1 + mu tan(eta)(cos g - kn sin g) ----
K1 = KN / np.cos(HELIX)
K2 = 1.0 + MU_C * np.tan(HELIX) * (np.cos(RAKE) - KN * np.sin(RAKE))


def down_milling_angles(ae=AE_NOM, R=R_TOOL):
    """Angles d'entree/sortie en avalant : phi_st = pi - acos(1 - ae/R)."""
    return np.pi - np.arccos(1.0 - ae / R), np.pi


def alpha34(t, Omega, za_low, za_high, ae=AE_NOM,
            NT=N_TEETH, R=R_TOOL, eta=HELIX, k1=K1, k2=K2, kt=KT):
    """(alpha3, alpha4) a l'instant t — Eqs. (3)-(4) integrees analytiquement
    sur la portion de dent engagee (angles) x (profondeur axiale)."""
    phi_st, phi_ex = down_milling_angles(ae, R)
    te = np.tan(eta)
    ss = sc = cc = 0.0
    for j in range(NT):
        th0 = Omega * t + 2 * np.pi * j / NT
        th_max = th0 - za_low * te / R
        th_min = th0 - za_high * te / R
        k_min = int(np.floor((th_min - phi_ex) / (2 * np.pi)))
        k_max = int(np.ceil((th_max - phi_st) / (2 * np.pi)))
        for kk in range(k_min, k_max + 1):
            lo = max(th_min, phi_st + 2 * np.pi * kk)
            hi = min(th_max, phi_ex + 2 * np.pi * kk)
            if hi <= lo:
                continue
            z2 = min((th0 - lo) * R / te, za_high)
            z1 = max((th0 - hi) * R / te, za_low)
            if z2 <= z1:
                continue
            t1 = th0 - z1 * te / R
            t2 = th0 - z2 * te / R
            ss += (z2 - z1) / 2 + R / (2 * te) * np.sin(t2 - t1) * np.cos(t2 + t1)
            sc += R / (2 * te) * np.sin(t1 - t2) * np.sin(t1 + t2)
            cc += (z2 - z1) / 2 - R / (2 * te) * np.sin(t2 - t1) * np.cos(t2 + t1)
    return k2 * kt * ss - k1 * kt * sc, k2 * kt * sc - k1 * kt * cc


def alpha4_series(rpm, ap, hp, n_samples, ae=AE_NOM, midpoint=True):
    """alpha4 echantillonne sur UNE periode de dent tau = 60/(NT rpm),
    profondeur axiale [hp - ap, hp]. Retour (a3, a4) de taille n_samples."""
    Omega = 2 * np.pi * rpm / 60.0
    tau = 60.0 / (N_TEETH * rpm)
    dt = tau / n_samples
    off = 0.5 * dt if midpoint else 0.0
    a3 = np.empty(n_samples)
    a4 = np.empty(n_samples)
    for k in range(n_samples):
        a3[k], a4[k] = alpha34(k * dt + off, Omega, hp - ap, hp, ae)
    return a3, a4


def alpha4_average(rpm, ap, hp, ae=AE_NOM, n=246):
    """Moyenne temporelle de alpha4 sur une periode (le "average milling force
    coefficient" du papier, base des courbes 0.3x / 2.9x de la Fig. 6)."""
    return float(np.mean(alpha4_series(rpm, ap, hp, n, ae)[1]))


# ---------------------------------------------------------------------------
# Modele reduit a deux modes + incertitudes (Section 3.2 du papier)
# ---------------------------------------------------------------------------
def reduced_two_mode(plate):
    """Extrait (omega, zeta, M0=I, C0, K0) des deux premiers modes."""
    w = plate.omega_n[:2]
    z = plate.zeta_modes[:2]
    return dict(omega=w, zeta=z,
                M0=np.eye(2), C0=np.diag(2 * z * w), K0=np.diag(w**2))


def dtd_statistics(plate, n_pos=201):
    """Elements de D_Pr^T D_Pr le long du bord superieur (Fig. 7) :
    valeurs, moyennes (max+min)/2 = nominal, amplitudes = perturbation
    (Eq. 24 : DD10..DD40 et L_DD1..L_DD4)."""
    xs, D = plate.D_top_edge(n_pos)
    D2 = D[:, :2]
    DtD = np.einsum('pi,pj->pij', D2, D2)          # (n_pos, 2, 2)
    dmax = DtD.max(axis=0)
    dmin = DtD.min(axis=0)
    DD0 = 0.5 * (dmax + dmin)                      # nominal (moyenne)
    LDD = 0.5 * (dmax - dmin)                      # amplitude de variation
    return dict(x=xs, DtD=DtD, DD0=DD0, LDD=LDD)


def uncertainty_parameters(plate, rpm, ap, hp, ae=AE_NOM):
    """Eqs. (23)-(25) : alpha40 = 1.6 abar4, L_Palpha = 1.3 abar4 (soit
    alpha4 dans [0.3, 2.9] abar4), et alpha40 * DD0 nominal + Delta_PrD.
    Perturbations de modes : 10 % masse/raideur, 20 % amortissement."""
    abar4 = alpha4_average(rpm, ap, hp, ae)
    st = dtd_statistics(plate)
    return dict(
        alpha_bar4=abar4,
        alpha40=1.6 * abar4,
        L_Palpha=1.3 * abar4,
        alpha4_range=(0.3 * abar4, 2.9 * abar4),
        DD0=st['DD0'], LDD=st['LDD'],
        nominal_a40_DD=1.6 * abar4 * st['DD0'],
        mode_mass_stiffness_perturbation=0.10,
        damping_perturbation=0.20,
    )


# ---------------------------------------------------------------------------
# Fonctions de poids d'incertitude additive, Eqs. (18)-(19)
# ---------------------------------------------------------------------------
W_PAF = dict(r=14e-6, zeta1=0.56, zeta2=0.12,
             w1=2 * np.pi * 1400.0, w2=2 * np.pi * 2800.0)
W_PAU = dict(r=4e-7, zeta1=0.58, zeta2=0.22,
             w1=2 * np.pi * 1100.0, w2=2 * np.pi * 3500.0)


def weight_tf(p, w):
    """|W(jw)| pour W(s) = r (s^2 + 2 z1 w1 s + w1^2)/(s^2 + 2 z2 w2 s + w2^2)."""
    s = 1j * np.asarray(w, float)
    num = s**2 + 2 * p['zeta1'] * p['w1'] * s + p['w1']**2
    den = s**2 + 2 * p['zeta2'] * p['w2'] * s + p['w2']**2
    return np.abs(p['r'] * num / den)


# ---------------------------------------------------------------------------
# Jauge du papier (Fig. 7) — recalage demande apres comparaison fine
# ---------------------------------------------------------------------------
FIG7_TARGETS = dict(D11_ends=3.60, D11_mid=3.81, D12_amp=3.45, D22_ends=3.35)


def paper_gauge(plate_bare, n_pos=201):
    """Cale la jauge modale du papier sur la Fig. 7.

    La Fig. 7 du papier est calculee sur les modes SYMETRIQUES de la plaque
    (l'effet du patch sur les formes n'y apparait pas : courbes parfaitement
    symetriques, moyenne de (1,2) exactement nulle) et dans une normalisation
    modale propre aux auteurs. Les rapports independants de la normalisation
    (dome de (1,1) : 3.81/3.60 -> phi1(mi)/phi1(0)=1.029 ; zero de (1,2) a
    50 mm ; linearite) coincident avec notre modele a 0.1 % pres ; seule
    l'echelle PAR MODE differe. On identifie donc s = diag(s1, s2) par
    moindres carres sur les niveaux de la Fig. 7 ; c'est un pur changement de
    coordonnees modales q -> s q (transformation de similitude), sans effet
    sur les predictions physiques (frequences, lobes, reponses) — verifie
    numeriquement dans refit_fig7.py.

    Retourne (s, xs, Phi_tilde) : facteurs (s1, s2), positions (m), et les
    deux profils de bord superieur dans la jauge du papier.
    """
    import numpy as np
    xs = np.linspace(0.0, plate_bare.lp, n_pos)
    W = np.array([plate_bare.Y_row(x, plate_bare.hp) for x in xs])
    Ph = W @ plate_bare.V[:, :2]
    if Ph[:, 0].mean() < 0:
        Ph[:, 0] *= -1.0
    if Ph[0, 1] < 0:
        Ph[:, 1] *= -1.0
    t = FIG7_TARGETS
    s1 = np.sqrt(0.5 * (Ph[0, 0]**2 / t['D11_ends']
                        + Ph[n_pos // 2, 0]**2 / t['D11_mid']))
    s2 = np.sqrt(Ph[0, 1]**2 / t['D22_ends'])
    return (s1, s2), xs, Ph / np.array([s1, s2])


def dtd_paper_gauge(plate_bare, n_pos=201):
    """Elements de DPr^T DPr dans la jauge du papier + parametres (24)-(25).

    Retourne (xs, DtD(n,2,2), DD0(2,2), LDD(2,2), s)."""
    import numpy as np
    s, xs, Pt = paper_gauge(plate_bare, n_pos)
    DtD = np.einsum('ni,nj->nij', Pt, Pt)
    mx, mn = DtD.max(axis=0), DtD.min(axis=0)
    return xs, DtD, 0.5 * (mx + mn), 0.5 * (mx - mn), s
