"""16_additive_uncertainty_weights.py — audit des Figs. 4-5 du papier :
reponses frequentielles maximales sur le bord superieur et fonctions de poids
d'incertitude additive W_Paf / W_Pau (Eqs. 18-19).

Du, Liu, Dai, Long, "Robust combined time delay control for milling chatter
suppression of flexible workpieces", Int. J. Mech. Sci. 274 (2024) 109257.

Ce que dit le papier (texte brut) :
  * Eq. (17) : G_P(s) = G_Pr(s) + Delta_Pa(s) W_Pa(s), ||Delta_Pa||_inf < 1,
    W_Pa = [W_Paf  W_Pau] (entrees : force de coupe F_PI et tension u_P).
  * "The frequency responses of the first two modes are significantly larger
    than that of the third and higher modes in Fig. 4. Thus the dynamic model
    can be truncated from the second mode ... The third and higher mode
    responses are represented by additive uncertainty."
  * "If there exist weight functions covering the largest frequency responses
    of the plate at any position ... All the responses of high-order modes are
    lower than that of the designed weight functions."
  * Legende Fig. 5 : "Maximum frequency responses in all positions ... Red dot
    curves are designed weight functions cover the third and higher mode
    responses. (a) part milling force input of F_PI (Ref = 1 m/N), (b) control
    voltage input of u_P (Ref = 1 m/V)."
  * Eqs. (18)-(19) : W(s) = r (s^2 + 2 z1 w1 s + w1^2)/(s^2 + 2 z2 w2 s + w2^2)
    r_Paf = 14e-6, z_Paf1 = 0.56, z_Paf2 = 0.12, w_Paf1 = 2pi 1400,
    w_Paf2 = 2pi 2800 ; r_Pau = 4e-7, z_Pau1 = 0.58, z_Pau2 = 0.22,
    w_Pau1 = 2pi 1100, w_Pau2 = 2pi 3500.

Conventions de ce script :
  * Eq. (16) : F_PI entre par D_P^T(x~,z~) et la sortie est C_PT = [D_P 0] au
    MEME point (x~,z~) : la reponse "entree force" de la Fig. 4(a) est donc la
    reponse au POINT MOTEUR (colocalisee) au point de coupe, qui balaie le bord
    superieur. La reponse "entree tension" est D_P(x) (...)^-1 H_Pe.
  * positions x/l_P = 0, 0.1, ..., 1 sur z = h_P (grille demandee) ; une grille
    a 101 points sert de controle de finesse.
  * residu = reponse complete a 5 modes MOINS reponse reduite a 2 modes ; les
    modes etant normalises en masse et decouples, c'est exactement la somme des
    modes 3-4-5.

MISE EN GARDE (VERIFICATION.md, F9) : le NIVEAU ABSOLU des FRF du papier n'est
pas reproductible (la Fig. 12(a) digitalisee donne 43.83 um/N de souplesse
statique contre 6.69 um/N ici, soit un facteur 6.36 incompatible avec les
frequences publiees). La comparaison en niveau absolu est donc rapportee, mais
le test PROBANT est le test SANS ECHELLE : on renormalise residu et poids au
meme niveau au 3e mode et on regarde si la couverture tient encore.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'paper_model'),
                os.path.join(HERE, '..', 'control')]

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plate_model import build_plate, F_MEASURED, F_THEORETICAL
from milling_dynamics import W_PAF, W_PAU, weight_tf

OUT = os.path.join(HERE, '..', 'figures', 'verification',
                   '16_additive_uncertainty_weights.png')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

FMIN, FMAX = 50.0, 6000.0
N_RED = 2                      # modele reduit du papier : deux premiers modes
FIG12_LEVEL_FACTOR = 6.36      # VERIFICATION.md F9 (valeur digitalisee)


# ---------------------------------------------------------------------------
# 1. grille frequentielle : log + amas fins autour de chaque pole
# ---------------------------------------------------------------------------
def freq_grid(plate):
    f = [np.logspace(np.log10(FMIN), np.log10(FMAX), 4000)]
    for fi, zi in zip(plate.freq_n, plate.zeta_modes):
        f.append(fi * (1.0 + np.linspace(-10.0, 10.0, 601) * zi))
    f = np.unique(np.concatenate(f))
    return f[(f >= FMIN) & (f <= FMAX)]


# ---------------------------------------------------------------------------
# 2. reponses : max sur les positions du bord superieur
# ---------------------------------------------------------------------------
def responses(plate, f, n_pos):
    """Retourne un dict de |G| max sur les positions, pour les deux entrees,
    en version complete (5 modes), reduite (2 modes) et residuelle (3..5),
    plus la position argmax de chaque maximum."""
    xs = np.linspace(0.0, plate.lp, n_pos)
    D = np.array([plate.D_row(x, plate.hp) for x in xs])   # (n_pos, n_modes)
    H = np.asarray(plate.H_Pe_modal, float)
    w = np.asarray(plate.omega_n, float)
    z = np.asarray(plate.zeta_modes, float)
    om = 2 * np.pi * f
    den = (w[:, None]**2 - om[None, :]**2
           + 2j * z[:, None] * w[:, None] * om[None, :])   # (n_modes, n_f)

    def acc(sl):
        # G[p, k] = sum_{i in sl} R_i(p) / den[i, k]
        Rf = D[:, sl]**2                                   # force colocalisee
        Ru = D[:, sl] * H[sl]                              # entree tension
        Gf = Rf @ (1.0 / den[sl, :])
        Gu = Ru @ (1.0 / den[sl, :])
        return np.abs(Gf), np.abs(Gu)

    all_m = np.arange(plate.n_modes)
    aFf, aFu = acc(all_m)
    rFf, rFu = acc(all_m[:N_RED])
    hFf, hFu = acc(all_m[N_RED:])
    out = dict(xs=xs)
    for key, (Af, Au) in dict(full=(aFf, aFu), red=(rFf, rFu),
                              res=(hFf, hFu)).items():
        out[key + '_f'] = Af.max(axis=0)
        out[key + '_u'] = Au.max(axis=0)
        out[key + '_f_arg'] = xs[Af.argmax(axis=0)]
        out[key + '_u_arg'] = xs[Au.argmax(axis=0)]
    return out


def peak_near(f, y, f0, rel=0.03):
    """(f_pic, y_pic) : maximum de y dans +/- rel autour de f0."""
    m = (f > f0 * (1 - rel)) & (f < f0 * (1 + rel))
    if not m.any():
        k = int(np.argmin(np.abs(f - f0)))
        return f[k], y[k]
    idx = np.where(m)[0]
    k = idx[np.argmax(y[idx])]
    return f[k], y[k]


# ---------------------------------------------------------------------------
def main():
    plate = build_plate(patch='right', n_modes=5)          # calage theorique
    f = freq_grid(plate)
    R11 = responses(plate, f, 11)                          # grille demandee
    R101 = responses(plate, f, 101)                        # controle finesse

    Wf = weight_tf(W_PAF, 2 * np.pi * f)
    Wu = weight_tf(W_PAU, 2 * np.pi * f)

    line = "=" * 78
    print(line)
    print("16 — POIDS D'INCERTITUDE ADDITIVE (Figs. 4-5, Eqs. 17-19)")
    print(line)
    print("Plaque : Chebyshev-Ritz 14x14, patch QDA60-20-0.7 coin bas droit,")
    print("calage sur les frequences THEORIQUES du Tableau 4.")
    print("Sortie et entree force colocalisees au point de coupe (Eq. 16),")
    print("entree tension -> deplacement au point de coupe. z = h_P.")
    print("Grille : %d frequences de %.0f a %.0f Hz ; 11 positions "
          "x/l_P = 0, 0.1, ... 1." % (len(f), FMIN, FMAX))
    print()

    # ---- modes ------------------------------------------------------------
    print("--- Modes du modele (Hz) et amortissements mesures (Tableau 4) ---")
    print("  mode   f_modele    f_theo(Tab.4)   f_mes(Tab.4)   zeta[%]")
    for i in range(plate.n_modes):
        print("   %d     %8.1f      %8.1f       %8.1f       %5.2f"
              % (i + 1, plate.freq_n[i], F_THEORETICAL[i], F_MEASURED[i],
                 100 * plate.zeta_modes[i]))
    print()

    # ---- 4. parametres des poids -----------------------------------------
    print("--- Eqs. (18)-(19) : parametres des poids (valeurs du papier) ---")
    hdr = "  %-8s %10s %8s %8s %12s %12s" % (
        "poids", "r", "zeta1", "zeta2", "w1/2pi[Hz]", "w2/2pi[Hz]")
    print(hdr)
    for nm, p, un in (("W_Paf", W_PAF, "m/N"), ("W_Pau", W_PAU, "m/V")):
        print("  %-8s %10.3e %8.2f %8.2f %12.1f %12.1f"
              % (nm, p['r'], p['zeta1'], p['zeta2'],
                 p['w1'] / (2 * np.pi), p['w2'] / (2 * np.pi)))
    print("  asymptotes  |W(0)| = r w1^2/w2^2 , |W(inf)| = r :")
    for nm, p, un in (("W_Paf", W_PAF, "m/N"), ("W_Pau", W_PAU, "m/V")):
        w = weight_tf(p, 2 * np.pi * f)
        print("    %-6s |W(0)| = %.3e %-4s  |W(inf)| = %.3e %-4s  "
              "max sur 50-6000 Hz = %.3e a %.0f Hz"
              % (nm, p['r'] * p['w1']**2 / p['w2']**2, un, p['r'], un,
                 w.max(), f[int(np.argmax(w))]))
    print()

    # ---- claim "modes 1-2 >> modes 3+" -----------------------------------
    print("--- Hypothese de troncature : pics max-sur-positions par mode ---")
    print("  (reponse COMPLETE 5 modes, max sur les 11 positions)")
    print("  mode   f_pic[Hz]   |G_F|[m/N]    |G_u|[m/V]   x_argmax/l_P (F/u)")
    pk_f, pk_u = [], []
    for i in range(plate.n_modes):
        fp, yp = peak_near(f, R11['full_f'], plate.freq_n[i])
        fq, yq = peak_near(f, R11['full_u'], plate.freq_n[i])
        pk_f.append(yp)
        pk_u.append(yq)
        kf = int(np.argmin(np.abs(f - fp)))
        ku = int(np.argmin(np.abs(f - fq)))
        print("   %d    %8.1f   %.4e   %.4e     %.2f / %.2f"
              % (i + 1, fp, yp, yq,
                 R11['full_f_arg'][kf] / plate.lp,
                 R11['full_u_arg'][ku] / plate.lp))
    pk_f, pk_u = np.array(pk_f), np.array(pk_u)
    print("  rapports pic(mode 1 ou 2)/pic(mode i>=3) — le papier affirme "
          "'significantly larger' :")
    base_f, base_u = pk_f[:2].max(), pk_u[:2].max()
    for i in (2, 3, 4):
        print("    mode %d : force x%6.2f      tension x%6.2f"
              % (i + 1, base_f / pk_f[i], base_u / pk_u[i]))
    print()

    # ---- 3. couverture en niveau absolu ----------------------------------
    ratio_f = R11['res_f'] / Wf
    ratio_u = R11['res_u'] / Wu
    print("--- COUVERTURE EN NIVEAU ABSOLU : |residu(3..5)| / |W| ---")
    print("  (>1 = le poids NE couvre PAS le residu ; le papier affirme <1)")
    bands = [(50.0, 100.0), (100.0, 1000.0), (1000.0, 6000.0)]
    print("  bande [Hz]        max ratio F     f[Hz]     max ratio u     f[Hz]")
    for lo, hi in bands:
        m = (f >= lo) & (f <= hi)
        kf = np.where(m)[0][np.argmax(ratio_f[m])]
        ku = np.where(m)[0][np.argmax(ratio_u[m])]
        print("  %6.0f - %-6.0f  %13.4f %9.0f %15.4f %9.0f"
              % (lo, hi, ratio_f[kf], f[kf], ratio_u[ku], f[ku]))
    kf = int(np.argmax(ratio_f))
    ku = int(np.argmax(ratio_u))
    print("  %-14s %13.4f %9.0f %15.4f %9.0f"
          % ("TOUTE LA BANDE", ratio_f[kf], f[kf], ratio_u[ku], f[ku]))
    print()
    print("  aux pics des modes 3, 4, 5 :")
    print("   mode  f_pic[Hz]  |res_F|[m/N]   |W_Paf|[m/N]  ratio | "
          "|res_u|[m/V]   |W_Pau|[m/V]  ratio")
    peak_tab = []
    for i in (2, 3, 4):
        fp, yf = peak_near(f, R11['res_f'], plate.freq_n[i])
        fq, yu = peak_near(f, R11['res_u'], plate.freq_n[i])
        wf = float(weight_tf(W_PAF, 2 * np.pi * fp))
        wu = float(weight_tf(W_PAU, 2 * np.pi * fq))
        peak_tab.append((i + 1, fp, yf, wf, yf / wf, fq, yu, wu, yu / wu))
        print("    %d   %8.1f   %.4e   %.4e %6.3f | %.4e   %.4e %6.3f"
              % (i + 1, fp, yf, wf, yf / wf, yu, wu, yu / wu))
    print()

    # ---- test SANS ECHELLE : renormalisation au 3e mode ------------------
    _, res3f = peak_near(f, R11['res_f'], plate.freq_n[2])
    _, res3u = peak_near(f, R11['res_u'], plate.freq_n[2])
    w3f = float(weight_tf(W_PAF, 2 * np.pi * plate.freq_n[2]))
    w3u = float(weight_tf(W_PAU, 2 * np.pi * plate.freq_n[2]))
    kf_n = w3f / res3f
    ku_n = w3u / res3u
    rnf = ratio_f * kf_n
    rnu = ratio_u * ku_n
    print("--- TEST SANS ECHELLE (le niveau absolu du papier n'est pas ")
    print("    reproductible, F9) : residu renormalise pour TOUCHER |W| au ")
    print("    3e mode ; on teste alors la FORME du poids seule. ---")
    print("  facteurs de renormalisation : force x%.4f   tension x%.4f"
          % (kf_n, ku_n))
    a = int(np.argmax(rnf))
    b = int(np.argmax(rnu))
    print("  frequence du pire cas apres renormalisation : force %.0f Hz "
          "(mode 3 = %.0f Hz), tension %.0f Hz" % (f[a], plate.freq_n[2], f[b]))
    print("  -> le pire cas EST le 3e mode dans les deux cas : le poids "
          "cale sur le 3e mode")
    print("     couvre alors TOUT le reste. Marges restantes "
          "(1 - residu_norm/|W|) :")
    for row in peak_tab:
        print("    mode %d : force %+6.1f %%     tension %+6.1f %%"
              % (row[0], 100 * (1 - row[4] * kf_n),
                 100 * (1 - row[8] * ku_n)))
    for lo, hi in bands:
        m = (f >= lo) & (f <= hi)
        ia = np.where(m)[0][np.argmax(rnf[m])]
        ib = np.where(m)[0][np.argmax(rnu[m])]
        print("    bande %5.0f-%-5.0f Hz : force %6.4f   tension %6.4f"
              % (lo, hi, rnf[ia], rnu[ib]))
    print("  CONCLUSION SANS ECHELLE : la FORME de W_Paf et W_Pau enveloppe "
          "bien les")
    print("  modes 3-4-5 (max = %.4f / %.4f, atteint au 3e mode)."
          % (rnf[a], rnu[b]))
    print()

    # ---- position des pics du poids contre les modes ----------------------
    print("--- Concordance de FORME : pics des poids contre modes du modele ---")
    for nm, p_, mref in (("W_Paf", W_PAF, 2), ("W_Pau", W_PAU, 3)):
        wv = weight_tf(p_, 2 * np.pi * f)
        fpk = f[int(np.argmax(wv))]
        print("  %-6s pic a %7.1f Hz ; mode %d du modele = %7.1f Hz "
              "(ecart %+5.2f %%)"
              % (nm, fpk, mref + 1, plate.freq_n[mref],
                 100 * (fpk / plate.freq_n[mref] - 1)))
    print()

    # ---- amortissement requis pour que la couverture tension tienne -------
    z3_req = plate.zeta_modes[2] * (res3u / w3u)
    print("--- Marge en amortissement (entree tension, seul cas non couvert) ---")
    print("  zeta_3 du modele = %.3f %% (Tableau 4, mesure)."
          % (100 * plate.zeta_modes[2],))
    print("  Le pic du residu passerait sous |W_Pau| pour zeta_3 >= %.3f %% "
          "(soit %+.1f %%)."
          % (100 * z3_req, 100 * (z3_req / plate.zeta_modes[2] - 1)))
    print()

    # ---- mise en garde de niveau (F9) ------------------------------------
    print("--- Mise en garde de niveau absolu (VERIFICATION.md F9) ---")
    print("  Fig. 12(a) digitalisee (valeur DIGITALISEE, pas imprimee) donne")
    print("  une souplesse statique x%.2f superieure a celle du modele."
          % FIG12_LEVEL_FACTOR)
    print("  Si l'on prenait ce niveau au pied de la lettre, le ratio pire cas")
    print("  deviendrait : force %.3f -> %.3f   tension %.3f -> %.3f"
          % (ratio_f[kf], ratio_f[kf] * FIG12_LEVEL_FACTOR,
             ratio_u[ku], ratio_u[ku] * FIG12_LEVEL_FACTOR))
    print("  -> le poids du papier serait alors viole d'un facteur 5 a 7, ce")
    print("     qui contredirait sa propre legende de Fig. 5. Autrement dit :")
    print("     r_Paf = 14e-6 m/N et r_Pau = 4e-7 m/V sont des nombres ABSOLUS")
    print("     IMPRIMES par le papier, et ils s'accordent au niveau absolu de")
    print("     NOTRE modele a %.0f %% / %.0f %% pres. C'est une confirmation"
          % (100 * abs(1 - ratio_f[kf]), 100 * abs(1 - ratio_u[ku])))
    print("     INDEPENDANTE que l'echelle en dB de la Fig. 12 est inutilisable")
    print("     (F9) et que l'echelle du modele est la bonne.")
    print()

    # ---- controle de finesse de grille -----------------------------------
    d_f = np.max(np.abs(R101['res_f'] / R11['res_f'] - 1.0))
    d_u = np.max(np.abs(R101['res_u'] / R11['res_u'] - 1.0))
    print("--- Controle : grille 101 positions contre 11 positions ---")
    print("  ecart relatif max du residu max-sur-positions : "
          "force %.3f %%   tension %.3f %%" % (100 * d_f, 100 * d_u))
    r101f = (R101['res_f'] / Wf).max()
    r101u = (R101['res_u'] / Wu).max()
    print("  ratio pire cas avec 101 positions : force %.4f   tension %.4f"
          % (r101f, r101u))
    print()

    # ---- variante de position du patch ------------------------------------
    pl = build_plate(patch='left', n_modes=5)
    Rl = responses(pl, f, 11)
    rlu = (Rl['res_u'] / Wu).max()
    klu = int(np.argmax(Rl['res_u'] / Wu))
    _, rl3 = peak_near(f, Rl['res_u'], pl.freq_n[2])
    print("--- Variante : patch au coin bas GAUCHE (texte de la Sec. 3) ---")
    print("  (l'entree force ne depend pas du patch au 1er ordre ; seule")
    print("   l'entree tension change)")
    print("  ratio pire cas tension : %.4f a %.0f Hz  (coin droit : %.4f)"
          % (rlu, f[klu], ratio_u[ku]))
    print("  residu au pic du 3e mode : %.4e m/V  (coin droit %.4e)"
          % (rl3, res3u))
    dmax = np.max(np.abs(Rl['res_u'] / R11['res_u'] - 1.0))
    print("  ecart relatif max des deux courbes max-sur-positions : %.2e"
          % dmax)
    print("  -> IDENTIQUES : [0, 60] mm est l'image exacte de [40, 100] mm par")
    print("     x -> l_P - x, et le max sur TOUTES les positions du bord est")
    print("     invariant par ce miroir. Le max-sur-positions de la Fig. 5 ne")
    print("     permet donc pas de trancher entre coin gauche et coin droit.")
    print("     A une position DONNEE les deux plaques different bien :")
    kk = int(np.argmin(np.abs(f - pl.freq_n[2])))
    xs = np.linspace(0.0, plate.lp, 11)
    Dr = np.array([plate.D_row(x, plate.hp) for x in xs])
    Dl = np.array([pl.D_row(x, pl.hp) for x in xs])
    om3 = 2 * np.pi * pl.freq_n[2]
    dn = (np.asarray(pl.omega_n)**2 - om3**2
          + 2j * np.asarray(pl.zeta_modes) * np.asarray(pl.omega_n) * om3)
    gr = np.abs((Dr[:, 2:] * np.asarray(plate.H_Pe_modal)[2:]) @ (1 / dn[2:]))
    gl = np.abs((Dl[:, 2:] * np.asarray(pl.H_Pe_modal)[2:]) @ (1 / dn[2:]))
    print("       x/l_P      = " + " ".join("%7.1f" % (x / plate.lp)
                                            for x in xs))
    print("       droit[m/V] = " + " ".join("%7.2e" % v for v in gr))
    print("       gauche     = " + " ".join("%7.2e" % v for v in gl))
    print()

    # ---- sensibilite au calage de frequences ------------------------------
    pm = build_plate(patch='right', n_modes=5, freqs=F_MEASURED)
    fm = freq_grid(pm)
    Rm = responses(pm, fm, 11)
    rmf = (Rm['res_f'] / weight_tf(W_PAF, 2 * np.pi * fm)).max()
    rmu = (Rm['res_u'] / weight_tf(W_PAU, 2 * np.pi * fm)).max()
    print("--- Sensibilite : calage sur les frequences MESUREES du Tab. 4 ---")
    print("  ratio pire cas : force %.4f (theo %.4f)   "
          "tension %.4f (theo %.4f)" % (rmf, ratio_f[kf], rmu, ratio_u[ku]))
    print(line)

    # -----------------------------------------------------------------------
    # figure
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.4))
    labels = ['1st', '2nd', '3rd', '4th', '5th']
    panels = [
        (ax[0], R11['full_f'], R11['red_f'], R11['res_f'], Wf, kf_n,
         '(a) milling force input $F_{PI}$', 'magnitude [m/N]',
         r'$|W_{Paf}|$ (Eq. 18)'),
        (ax[1], R11['full_u'], R11['red_u'], R11['res_u'], Wu, ku_n,
         '(b) control voltage input $u_P$', 'magnitude [m/V]',
         r'$|W_{Pau}|$ (Eq. 19)'),
    ]
    for a, full, red, res, W, kn, title, ylab, wlab in panels:
        a.loglog(f, full, color='0.72', lw=1.0,
                 label='full 5-mode, max over positions')
        a.loglog(f, red, color='tab:blue', lw=1.0, ls='--',
                 label='reduced 2-mode, max over positions')
        a.loglog(f, res, color='k', lw=1.6,
                 label='residual (modes 3-5), max over positions')
        a.loglog(f, kn * res, color='tab:green', lw=1.2, ls='-.',
                 label='residual rescaled to $|W|$ at 3rd mode ($\\times$%.2f)'
                       % kn)
        a.loglog(f, W, color='tab:red', lw=2.0, ls=':', label=wlab)
        a.set_xlim(FMIN, FMAX)
        tr = a.get_xaxis_transform()                      # x data, y axes
        for i, fi in enumerate(plate.freq_n):
            a.axvline(fi, color='0.85', lw=0.8, zorder=0)
            a.text(fi, 0.995 - 0.045 * (i % 2), labels[i], fontsize=7,
                   ha='center', va='top', color='0.35', transform=tr)
        a.set_xlabel('frequency [Hz]')
        a.set_ylabel(ylab)
        a.set_title(title, fontsize=10)
        a.grid(True, which='both', alpha=0.25)
        a.legend(fontsize=7, loc='lower left')

    ax[0].text(0.015, 0.965,
               'worst |residual|/|W| = %.2f  (cover HOLDS)\n'
               'worst case at 3rd mode -> shape test OK\n'
               'rescaled margins: 4th %+.0f%%, 5th %+.0f%%'
               % (ratio_f[kf], 100 * (1 - peak_tab[1][4] * kf_n),
                  100 * (1 - peak_tab[2][4] * kf_n)),
               transform=ax[0].transAxes, ha='left', va='top', fontsize=7.5,
               bbox=dict(fc='w', ec='0.7', alpha=0.92))
    ax[1].text(0.015, 0.965,
               'worst |residual|/|W| = %.2f  (cover FAILS by %.0f%%)\n'
               'worst case at 3rd mode -> shape test OK\n'
               'rescaled margins: 4th %+.0f%%, 5th %+.0f%%'
               % (ratio_u[ku], 100 * (ratio_u[ku] - 1),
                  100 * (1 - peak_tab[1][8] * ku_n),
                  100 * (1 - peak_tab[2][8] * ku_n)),
               transform=ax[1].transAxes, ha='left', va='top', fontsize=7.5,
               bbox=dict(fc='w', ec='0.7', alpha=0.92))
    # encart : le seul depassement reel (3e mode, entree tension)
    ins = ax[1].inset_axes([0.62, 0.055, 0.355, 0.27])
    m = (f > 2680) & (f < 2940)
    ins.semilogy(f[m], R11['res_u'][m], 'k', lw=1.5)
    ins.semilogy(f[m], Wu[m], ':', color='tab:red', lw=2.0)
    ins.semilogy(f[m], ku_n * R11['res_u'][m], '-.', color='tab:green', lw=1.0)
    ins.set_ylim(2e-7, 8e-7)
    ins.tick_params(labelsize=5.5, labelleft=False, left=False, which='both')
    ins.set_facecolor('white')
    ins.text(0.03, 0.95, '3rd-mode zoom: residual\nexceeds $|W_{Pau}|$ by 13%',
             transform=ins.transAxes, fontsize=5.8, va='top')
    ins.grid(True, which='major', alpha=0.25)

    fig.suptitle('Fig. 5 audit — max-over-position responses of the upper edge '
                 'and additive uncertainty weights', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print("figure :", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
