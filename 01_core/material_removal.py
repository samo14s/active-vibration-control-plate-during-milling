"""
material_removal.py
==================
NMODÉLISATION PHYSIQUE de la dérive de fréquence en cours d'usinage d'une
paroi mince — l'origine mécanique du détuning que l'AIMC identifie et suit.

ORIGINE PHYSIQUE (enlèvement de matière) :
Lorsqu'on usine une paroi mince, l'épaisseur locale B diminue (fraisage
périphérique : chaque passe enlève l'engagement radial a_e ; en descendant la
paroi, la portion déjà usinée est plus mince).  Les fréquences propres de la
paroi dérivent en conséquence.

LOI D'ÉCHELLE EXACTE (plaque) :  ω_i ∝ B
Pour une plaque (flexion ET torsion), la rigidité par unité varie en B³
(D = E B³/[12(1−ν²)], rigidité de torsion GJ ∝ B³) et l'inertie en B, d'où
    ω_i = √(rigidité / inertie) ∝ √(B³/B) = B .
=> TOUTES les fréquences (flexion, torsion, flexion-largeur) se mettent à
   l'échelle du MÊME facteur B(t)/B₀.  La dérive uniforme entre modes
   (k_scale identique pour tous) est donc EXACTE pour la flexion de plaque,
   pas une approximation.  Elle JUSTIFIE physiquement le paramètre unique ρ
   de la grille MMAE de l'AIMC :
       ρ(t) = B(t)/B₀ − 1 ,  ω_i(t) = (1+ρ(t))·ω_{i,0} ,
       k_scale(t) = (ω(t)/ω₀)² = (B(t)/B₀)²   (facteur sur Kp ; √ sur Cp).

Un détuning de −15 % correspond exactement à un amincissement de 15 %
(0.60 mm retirés d'une paroi de 4 mm).

CE QUI N'EST PAS MODÉLISÉ (honnêteté) :
- Amincissement SPATIALEMENT non uniforme : une paroi réellement usinée est
  plus mince dans la zone déjà passée que devant l'outil ; cela ferait
  aussi évoluer les FORMES propres et Dp, non seulement les fréquences.  Ici
  on ne fait varier que les fréquences (effet dominant), les formes restent
  celles de la plaque nominale — approximation « paroi d'épaisseur effective
  uniforme instantanée ».
- Couplage avec l'amortissement : ζ modaux supposés constants (Cp mis à
  l'échelle en √k_scale pour garder ζ = C/(2√(KM)) constant).
- Rétroaction géométrique de la matière enlevée sur le champ de contrainte.

Ces limites sont explicites ; l'objet est de fournir une TRAJECTOIRE de
fréquence physiquement fondée pour éprouver les commandes adaptatives.
"""
import numpy as np


def kscale_from_thickness(B_current, B_nominal):
    """
    Facteur multiplicatif de raideur modale pour une épaisseur B_current
    (loi de plaque ω ∝ B) :  k_scale = (B_current/B_nominal)².
    ω(t) = √k_scale · ω₀ = (B_current/B_nominal)·ω₀.
    """
    r = float(B_current) / float(B_nominal)
    return r * r


def rho_from_thickness(B_current, B_nominal):
    """Détuning fréquentiel ρ = ω/ω₀ − 1 = B_current/B_nominal − 1."""
    return float(B_current) / float(B_nominal) - 1.0


def thinning_schedule(t_vec, B_nominal, removed_total,
                      t_start=None, t_end=None, profile='linear',
                      passes=None, ae=None):
    """
    Construit la trajectoire k_scale(t) d'un amincissement PHYSIQUE de paroi.

    Deux façons de spécifier la matière enlevée :
      - directement par ``removed_total`` [m] (épaisseur totale retirée), ou
      - par un nombre de passes ``passes`` × engagement radial ``ae`` [m]
        (fraisage périphérique : removed_total = passes·ae).

    Parameters
    ----------
    t_vec       : (nstep,) axe temps de la simulation [s].
    B_nominal   : épaisseur initiale de la paroi [m].
    removed_total : épaisseur totale retirée à la fin [m] (ex. 0.6e-3).
    t_start, t_end : début/fin de l'enlèvement [s] (défaut : 20 %..80 % de
                  la durée).  Avant t_start : paroi pleine ; après t_end :
                  paroi amincie finale.
    profile     : 'linear' (enlèvement à débit constant) ou 'smooth'
                  (S-curve : accélération/décélération douce).
    passes, ae  : alternative à removed_total (removed_total = passes·ae).

    Returns
    -------
    k_scale_t : (nstep,) facteur de raideur modale, à passer à
                NewmarkSimulator.simulate(k_scale_t=...).
    B_t       : (nstep,) épaisseur de paroi [m] (diagnostic).
    rho_t     : (nstep,) détuning fréquentiel ρ(t) (diagnostic).
    """
    t = np.asarray(t_vec, dtype=float)
    T = t[-1]
    if passes is not None and ae is not None:
        removed_total = passes * ae
    if t_start is None:
        t_start = 0.2 * T
    if t_end is None:
        t_end = 0.8 * T

    # fraction d'avancement de l'enlèvement ∈ [0,1]
    frac = np.clip((t - t_start) / max(t_end - t_start, 1e-12), 0.0, 1.0)
    if profile == 'smooth':
        frac = 3*frac**2 - 2*frac**3          # S-curve (C¹, dérivée nulle aux bouts)

    B_t = B_nominal - removed_total * frac
    if np.any(B_t <= 0):
        raise ValueError("Amincissement total >= épaisseur : paroi percée.")
    k_scale_t = (B_t / B_nominal)**2
    rho_t = B_t / B_nominal - 1.0
    return k_scale_t, B_t, rho_t


def machining_modal_schedule(t_vec, mplate, B_nominal, removed_total,
                             z_frac_start=1.0, z_frac_end=0.5,
                             t_start=None, t_end=None,
                             zeta=(0.0031, 0.0017, 0.0027),
                             n_knots=25, profile='linear', verbose=True):
    """
    Construit une PLANIFICATION MODALE PHYSIQUE d'un usinage de paroi mince :
    ré-dérive (Mp(t), Kp(t), Cp(t)) dans la base fixe des modes nominaux à
    partir de la GÉOMÉTRIE réellement usinée (amincissement LOCAL).

    Modèle d'usinage : une bande usinée de hauteur croissante près du bord
    libre.  À l'instant t, la région Z ∈ [z_mach(t)·H_P, H_P] est amincie de
    `removed_total`, où z_mach passe de ``z_frac_start`` à ``z_frac_end``
    (fraction de H_P) sur [t_start, t_end] (l'outil descend la paroi ;
    fraction usinée = 1 − z_mach).  Épaisseur uniforme en X (passe pleine
    largeur), non uniforme en Z -> dérive PAR MODE différente (loi ω∝B seule
    ne suffit pas).

    Parameters
    ----------
    mplate : instance de MachinedPlate (fournit l'assemblage Ritz + base Φ₀).
    B_nominal, removed_total : épaisseur pleine et profondeur retirée [m].
    z_frac_start/end : bord de la région usinée (fraction de H_P) au début/fin.
    n_knots : nombre d'instantanés modaux (interpolés en escalier par le solveur).

    Returns
    -------
    dict schedule : {'t_knots','Mp','Kp','Cp','omega','B_field_diag'} — listes
        de n_knots matrices 3×3 (+ pulsations instantanées), + méta.  À passer
        à NewmarkSimulator.simulate(modal_schedule=...).
    """
    t = np.asarray(t_vec, dtype=float)
    T = t[-1]
    if t_start is None:
        t_start = 0.2 * T
    if t_end is None:
        t_end = 0.8 * T
    coeff0 = mplate.nominal_basis()
    ng = len(mplate._zx)
    Zg = np.meshgrid(mplate._zx, mplate._zz, indexing='ij')[1]

    tk = np.linspace(t[0], T, n_knots)
    Mp_l, Kp_l, Cp_l, om_l, zmach_l = [], [], [], [], []
    for tt in tk:
        frac = np.clip((tt - t_start)/max(t_end - t_start, 1e-12), 0.0, 1.0)
        if profile == 'smooth':
            frac = 3*frac**2 - 2*frac**3
        z_mach = z_frac_start + (z_frac_end - z_frac_start)*frac
        B = np.full((ng, ng), B_nominal)
        B[Zg >= z_mach*mplate.hp] = B_nominal - removed_total
        Mp, Kp, om = mplate.modal_matrices(B, coeff0)
        Cp = mplate.damping_matrix(Mp, Kp, zeta)
        Mp_l.append(Mp); Kp_l.append(Kp); Cp_l.append(Cp)
        om_l.append(om); zmach_l.append(z_mach)
    if verbose:
        d0 = (om_l[0]/om_l[0] - 1)*100
        dend = (om_l[-1]/om_l[0] - 1)*100
        print(f"[machining] {n_knots} instantanés, région usinée "
              f"{(1-z_frac_start)*100:.0f}%→{(1-z_frac_end)*100:.0f}% de H_P, "
              f"−{removed_total*1e3:.2f}mm")
        print(f"[machining] dérive finale par mode : "
              f"{np.array2string(dend, precision=1, floatmode='fixed')} %")
    return dict(t_knots=tk, Mp=Mp_l, Kp=Kp_l, Cp=Cp_l,
                omega=om_l, z_mach=zmach_l)
