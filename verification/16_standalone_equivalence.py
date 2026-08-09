"""milling_plant.py est-il NUMERIQUEMENT le meme procede que la base a cinq
fichiers ? Ce script le prouve, ou echoue.

Un fichier autonome n'a d'interet que s'il ne diverge pas de la base validee.
Le risque n'est pas theorique : regrouper cinq modules en un seul, c'est cinq
occasions de recopier une constante de travers. On compare donc, terme a terme :

  1. les grandeurs modales  — omega, zeta, H_Pe, D_obs, D_tool
  2. une trajectoire complete — y(t) et u(t), pas seulement leur RMS
  3. les verdicts de stabilite et les limites de passe, aux cinq vitesses
  4. la meme chose EN BOUCLE FERMEE, avec un correcteur non trivial branche
     par chacune des quatre formes d'interface annoncees

Le point 4 est le plus important : c'est la seule partie ou milling_plant.py a
ete REECRIT et non recopie. Un retour a zero silencieux — correcteur non
reconnu, donc u = 0 — se lirait comme une boucle fermee parfaitement stable.

Duree : ~4 min.
"""
import copy
import os
import sys

import numpy as np

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(_R, 'simulation', 'sim_kit'),
                os.path.join(_R, 'simulation')]
os.chdir(os.path.join(_R, 'simulation'))

import simulation_base as SB                                     # noqa: E402
import milling_plant as MP                                       # noqa: E402
from model_v2 import make_sim                                    # noqa: E402

print(__doc__.split('Duree')[0])
print('=' * 74)

# ---------------------------------------------------------------- 0. constantes
print('0. CONSTANTES FIGEES')
SHARED = ['PLATE_L', 'PLATE_H', 'PLATE_T', 'RHO', 'YOUNG', 'POISSON',
          'MESH_N1', 'MESH_N2', 'N_MODES', 'F_MEASURED', 'ZETA_MODES',
          'SENSOR_XZ', 'V_MAX', 'H_IDENT', 'COUPLING', 'N_TEETH', 'D_TOOL',
          'HELIX', 'RAKE', 'KT', 'KN', 'MU_C', 'AE_NOM', 'FZ_NOM',
          'STEPS_PER_TOOTH', 'SPEEDS_DEFAULT', 'AP_TEST', 'SIGMA_MAX',
          'T_RUN', 'T_LIMIT', 'FORCE_SIGN', 'PATCH']
bad = []
for name in SHARED:
    a, b = getattr(SB, name), getattr(MP, name)
    if isinstance(a, (dict, str)):
        same = a == b
    else:
        same = np.allclose(np.asarray(a, float), np.asarray(b, float))
    if not same:
        bad.append(f'{name}: base={a!r} autonome={b!r}')
print(f'   {len(SHARED)} constantes comparees, {len(bad)} ecart(s)')
assert not bad, 'constantes divergentes :\n  ' + '\n  '.join(bad)

# ---------------------------------------------------------------- 1. modal
print('\n1. GRANDEURS MODALES')
print('   Comparaison sur les grandeurs INVARIANTES PAR SIGNE DE MODE, et non')
print('   sur H et D_obs pris separement. Raison : ARPACK part d\'un vecteur')
print('   aleatoire, si bien que le signe de chaque mode change d\'une')
print('   construction a l\'autre — y compris entre deux SimBase() du meme')
print('   processus (verifie ci-dessous). Tout observable est un produit')
print('   D_i H_i ou D_i^2, donc invariant ; ce sont ces produits qui font foi.')
print('   milling_plant.py, lui, FIXE le signe (composante dominante positive).')
sim = make_sim()
plant = MP.MillingPlant()
INVARIANT = {
    'omega': lambda m: m['omega'],
    'zeta': lambda m: m['zeta'],
    '|H|': lambda m: np.abs(m['H']),
    '|D_obs|': lambda m: np.abs(m['D_obs']),
    '|D_tool|': lambda m: np.abs(m['D_tool']),
    'H*D_obs': lambda m: m['H']*m['D_obs'],
    'D_obs*D_tool': lambda m: m['D_obs']*m['D_tool'],
    'H*D_tool': lambda m: m['H']*m['D_tool'],
}
def relerr(a, b):
    """Ecart relatif NORMALISE PAR LE PLUS GRAND TERME du vecteur, et non
    terme a terme. Une comparaison relative terme a terme echouerait sur les
    composantes petites — H_Pe elements finis a un residu 40 fois plus petit
    que les autres — alors que l'erreur d'arrondi, elle, est fixee par le plus
    grand."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return np.abs(b - a).max()/max(np.abs(a).max(), 1e-300)


# Le seuil n'est pas choisi a la main : c'est la DISPERSION PROPRE DE LA BASE.
# ARPACK ne converge pas au meme bit d'une construction a l'autre, donc deux
# SimBase() du meme processus different deja entre eux. Exiger mieux que cela
# du fichier autonome n'aurait aucun sens ; exiger moins laisserait passer une
# vraie divergence. On mesure donc la dispersion interne, et on demande que
# l'ecart base <-> autonome reste dedans.
ref2 = [make_sim().modal for _ in range(2)]
disp = max(relerr(fn(ref2[0]), fn(ref2[1])) for fn in INVARIANT.values())
TOL_MODAL = max(10*disp, 1e-12)
print(f'\n   dispersion propre de la base (deux constructions) : {disp:.3e}')
print(f'   seuil retenu = 10 x cette dispersion             : {TOL_MODAL:.3e}\n')
for key, fn in INVARIANT.items():
    a = np.asarray(fn(sim.modal), float)
    b = np.asarray(fn(plant.modal), float)
    rel = relerr(a, b)
    verdict = 'OK' if rel < TOL_MODAL else 'ECART'
    print(f'   {key:13s} ecart relatif max = {rel:.3e}   {verdict}')
    assert rel < TOL_MODAL, f'{key} differe au-dela de la dispersion de la base'

# La base a cinq fichiers n'est PAS reproductible sur les signes ; le fichier
# autonome doit l'etre. On le montre plutot que de l'affirmer.
sgn_base = [tuple(np.sign(SB.SimBase().modal['D_obs']).astype(int))
            for _ in range(3)]
sgn_std = [tuple(np.sign(MP.MillingPlant().modal['D_obs']).astype(int))
           for _ in range(3)]
print(f'\n   signes de D_obs, base     : {sgn_base}')
print(f'   signes de D_obs, autonome : {sgn_std}')
assert len(set(sgn_std)) == 1, \
    "milling_plant.py doit fixer le signe des modes"
if len(set(sgn_base)) > 1:
    print('   -> la base varie, l\'autonome ne varie pas : convention fixee.')
else:
    print('   -> la base n\'a pas varie sur ces 3 tirages (elle le peut).')

# ---------------------------------------------------------------- 2. libre
print('\n2. TRAJECTOIRE COMPLETE, BOUCLE OUVERTE (4900 tr/min, 0.10 mm)')
ra = sim.run(None, rpm=4900, ap=0.10e-3, T=SB.T_RUN, keep_signals=True)
rb = plant.run(None, rpm=4900, ap=0.10e-3, T=MP.T_RUN, keep_signals=True)
assert len(ra['y']) == len(rb['y']), 'longueurs de releve differentes'
dy = np.abs(rb['y'] - ra['y']).max()/max(np.abs(ra['y']).max(), 1e-300)
print(f"   y(t) : {len(ra['y'])} points, ecart relatif max = {dy:.3e}")
print(f"   sigma : base {ra['sigma']:+.9f}   autonome {rb['sigma']:+.9f}")
print(f"   RMS   : base {ra['rms_um']:.9f}   autonome {rb['rms_um']:.9f} um")
# TOL n'est pas 0 a cause des signes de modes : les deux fichiers somment les
# memes termes dans un ordre de signes different, donc l'arrondi flottant
# differe au dernier bit. 1e-9 en relatif est ~100 fois au-dessus du bruit
# mesure et reste ~7 decades sous toute difference de modele.
TOL = 1e-9
assert dy < TOL, 'la trajectoire libre differe'
assert ra['stable'] == rb['stable']
assert abs(ra['sigma'] - rb['sigma']) < TOL*max(abs(ra['sigma']), 1.0)

# ---------------------------------------------------------------- 3. limites
print('\n3. LIMITES DE PASSE EN BOUCLE OUVERTE, AUX CINQ VITESSES')
print('   tr/min        base    autonome')
for rpm in SB.SPEEDS_DEFAULT:
    la = sim.stability_limit(None, rpm=rpm, lo=0.02e-3, hi=4.0e-3, tol=2e-6)
    lb = plant.stability_limit(None, rpm=rpm, lo=0.02e-3, hi=4.0e-3, tol=2e-6)
    print(f'   {rpm:5d}     {la*1e3:8.4f}   {lb*1e3:8.4f} mm', flush=True)
    assert abs(la - lb) < 1e-9, f'limite differente a {rpm} tr/min'

# ---------------------------------------------------------------- 4. commande
print('\n4. BOUCLE FERMEE — LES QUATRE FORMES D\'INTERFACE')
print('   meme loi de commande a chaque fois : un retour de vitesse estimee')
print('   par difference finie, avec memoire — donc un correcteur A ETAT, qui')
print('   rend zero si reset() n\'est pas appele.\n')

GAIN, GAIN_D = -2.0e4, -3.0e1


class Legacy:
    """Forme historique : step(x_hat_prev, u_prev, y_meas) -> (x_hat, u)."""

    def __init__(self, dt):
        self.dt = dt
        self.reset()

    def reset(self):
        self.y_prev = 0.0

    def step(self, x_hat_prev, u_prev, y_meas):
        dy = (y_meas - self.y_prev)/self.dt
        self.y_prev = y_meas
        return x_hat_prev, GAIN*y_meas + GAIN_D*dy


class Modern:
    """Forme objet moderne : step(y, dt) -> u."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.y_prev = 0.0

    def step(self, y, dt):
        dy = (y - self.y_prev)/dt
        self.y_prev = y
        return GAIN*y + GAIN_D*dy


def _closure_named():
    """Forme fonction qui NOMME ses arguments."""
    st = {'y': 0.0}

    def f(y, dt):
        dy = (y - st['y'])/dt
        st['y'] = y
        return GAIN*y + GAIN_D*dy
    return f


def _closure_positional(dt):
    """Forme fonction d'une seule variable, dont le nom n'est PAS un des noms
    reconnus : elle doit quand meme recevoir la mesure, positionnellement."""
    st = {'y': 0.0}

    def f(mesure):
        dy = (mesure - st['y'])/dt
        st['y'] = mesure
        return GAIN*mesure + GAIN_D*dy
    return f


RPM, AP = 4900, 0.25e-3
dt_ref, tau_ref = plant.sampling(RPM)

# reference : la base a cinq fichiers, forme historique
ref = sim.run(Legacy(dt_ref), rpm=RPM, ap=AP, T=SB.T_RUN, keep_signals=True)
print(f"   reference (base)          stable={ref['stable']}  "
      f"RMS={ref['rms_um']:.4f} um  crete u={ref['peak_u']:.2f} V")
assert ref['peak_u'] > 1.0, \
    "le correcteur de test ne commande rien : essai sans valeur"

CASES = [('4 historique  step(x_hat, u_prev, y)', Legacy(dt_ref)),
         ('3 objet       step(y, dt)', Modern()),
         ('2 fonction    f(y, dt)', _closure_named()),
         ('1 fonction    f(mesure)', _closure_positional(dt_ref))]

for label, ctrl in CASES:
    r = plant.run(ctrl, rpm=RPM, ap=AP, T=MP.T_RUN, keep_signals=True)
    dy = np.abs(r['y'] - ref['y']).max()/max(np.abs(ref['y']).max(), 1e-300)
    du = np.abs(r['u'] - ref['u']).max()/max(np.abs(ref['u']).max(), 1e-300)
    print(f"   forme {label:36s} y {dy:.2e}  u {du:.2e}")
    assert dy < TOL and du < TOL, f'forme {label} : trajectoire differente'

# reset() doit reellement etre appele entre deux essais
c = Modern()
r1 = plant.run(c, rpm=RPM, ap=AP, T=MP.T_RUN)
r2 = plant.run(c, rpm=RPM, ap=AP, T=MP.T_RUN)
print(f"\n   deux essais successifs avec LE MEME objet : "
      f"RMS {r1['rms_um']:.9f} et {r2['rms_um']:.9f} um")
assert abs(r1['rms_um'] - r2['rms_um']) == 0.0, \
    "reset() n'est pas appele : l'etat fuit d'un essai au suivant"

# un correcteur inconnu doit LEVER, pas rendre zero en silence
try:
    plant.run(object(), rpm=RPM, ap=AP, T=MP.T_RUN)
except TypeError as e:
    print(f"   correcteur invalide -> TypeError, comme voulu")
else:
    raise AssertionError("un correcteur invalide a ete accepte en silence")

# ---------------------------------------------------------------- 5. gain_H
print('\n5. PLAQUE PERTURBEE (gain_H = 2.0), BOUCLE OUVERTE')
sa = SB.SimBase(gain_H=2.0)
pb = MP.MillingPlant(gain_H=2.0)
assert relerr(np.abs(sa.modal['H']), np.abs(pb.modal['H'])) < TOL_MODAL
ra = sa.run(None, rpm=6000, ap=0.10e-3, T=SB.T_RUN)
rb = pb.run(None, rpm=6000, ap=0.10e-3, T=MP.T_RUN)
print(f"   sigma : base {ra['sigma']:+.9f}   autonome {rb['sigma']:+.9f}")
assert abs(ra['sigma'] - rb['sigma']) < TOL*max(abs(ra['sigma']), 1.0)

# ---------------------------------------------------------------- 6. fem
print('\n6. REPARTITION EF (coupling = \'fem\')')
sf = SB.SimBase(coupling='fem')
pf = MP.MillingPlant(coupling='fem')
e_fem = relerr(sf.modal['H']*sf.modal['D_obs'],
               pf.modal['H']*pf.modal['D_obs'])
print(f'   residus H*D_obs, ecart relatif : {e_fem:.3e}')
assert e_fem < TOL_MODAL
print(f"   creux calcules : {np.round(pf.antiresonances(), 0)} Hz")
print(f"   occupation     : {pf.notch_occupancy()}  (identifie : "
      f"{plant.notch_occupancy()}, mesure (1, 1, 1, 1))")
assert pf.notch_occupancy() != (1, 1, 1, 1), \
    "la version EF devrait justement NE PAS reproduire la mesure"

print('\n' + '=' * 74)
print('milling_plant.py est numeriquement identique a la base a cinq fichiers.')
print('TOUTES LES VERIFICATIONS PASSENT.')
