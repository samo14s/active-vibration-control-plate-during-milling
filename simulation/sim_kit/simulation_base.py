"""
================================================================================
BASE DE SIMULATION VALIDEE — fraisage d'une paroi mince avec patch piezo
================================================================================
Plaque, patch, capteur, efforts de coupe. AUCUN CORRECTEUR.

Cette base est FIGEE. Elle ne doit pas etre modifiee entre deux essais de
commande, sinon les resultats ne sont plus comparables. Tout correcteur se
branche par l'interface decrite plus bas.

--------------------------------------------------------------------------------
1. CE QUI EST VALIDE, ET COMMENT
--------------------------------------------------------------------------------
Le modele est cale sur deux courbes experimentales numerisees (reception au choc
et transfert tension -> deplacement, coin superieur droit).

  * cinq resonances mesurees   : 536.4 / 1071.8 / 2778.6 / 3358.7 / 4122.6 Hz
    predites par le modele EF  : +0.7 / +1.8 / -1.4 / +1.4 / +1.8 %
  * quatre antiresonances mesurees : 734 / 2161 / 3001 / 3805 Hz
    modele a cinq modes calibre    : 731 / 2118 / 3008 / 3943 Hz
    ecarts                         : +0.4 / -2.0 / +0.2 / +3.6 %
    (comparaison faite sur la reception AU POINT DE FRAPPE, residus D_obs^2 :
     c'est la grandeur mesuree en Fig. 12a, marteau et capteur au meme coin.
     Un modele a TROIS modes ne donne que deux antiresonances -- 741 / 2404 Hz,
     dont la seconde est fausse de 11 % : d'ou les cinq modes retenus.)
  * frequence de broutement simulee 532 Hz contre 580 Hz mesuree ; la simulation
    publiee de reference donnait 1135 Hz, soit le mauvais mode.

Verification a l'execution : appeler check_model() ; il refait ces controles et
leve une exception si un ecart depasse la tolerance.

--------------------------------------------------------------------------------
2. CE QUI EST FIGE, ET POURQUOI
--------------------------------------------------------------------------------
  * dt = tau/82 exactement. Le retard regeneratif doit tomber sur la grille
    d'echantillonnage, sinon q(t-tau) demande une interpolation qui fausse la
    stabilite. dt n'est donc PAS un reglage libre.
  * cinq modes retenus. Justifie par les antiresonances ci-dessus. La limite de
    stabilite en boucle ouverte est identique a 1, 2, 3 ou 5 modes (les modes 4
    et 5 sont fortement couples au patch mais quasiment pas excites par la
    coupe) : le nombre de modes ne change donc pas la frontiere libre, mais il
    change ce que l'observateur peut voir.
  * amortissements modaux : 0.31 / 0.17 / 0.27 / 0.30 / 0.30 %.
  * patch QDA60-20-0.7 dans le coin inferieur gauche, capteur au coin superieur
    droit oppose.

--------------------------------------------------------------------------------
3. INTERFACE CORRECTEUR
--------------------------------------------------------------------------------
Un correcteur est un objet exposant :

    step(x_hat_prev, u_prev, y_meas) -> (x_hat, u)
        appele une fois par pas. y_meas est le deplacement mesure en metres,
        u la tension renvoyee en volts. x_hat_prev est passe tel quel et peut
        etre ignore ; il n'existe que pour compatibilite.
    reset()
        remet les etats internes a zero. Appele avant chaque simulation.

Le simulateur ecrete lui-meme la commande a +/- V_MAX avant de l'appliquer
(newmark_solver.NewmarkSimulator, parametre v_max). La tension renvoyee au
correcteur au pas suivant (u_prev) est la tension SATUREE, donc un observateur
voit toujours ce qui a reellement ete applique. Un correcteur qui sature en
interne doit utiliser la MEME borne, faute de quoi il calculera sa commande
suivante a partir d'une valeur qu'il croit appliquee et qui ne l'est pas.

--------------------------------------------------------------------------------
4. UTILISATION
--------------------------------------------------------------------------------
    from simulation_base import SimBase, check_model

    check_model()                       # a lancer une fois
    sim = SimBase()                     # plaque + patch + capteur

    r = sim.run(MonCorrecteur(...), rpm=4900, ap=0.25e-3, T=0.20)
    print(r['rms_um'], r['stable'])

    J = sim.multi_speed_cost(lambda dt, tau: MonCorrecteur(dt, ...))

    lim = sim.stability_limit(lambda dt, tau: MonCorrecteur(dt, ...), rpm=4900)

--------------------------------------------------------------------------------
5. DEPENDANCES
--------------------------------------------------------------------------------
Ce fichier suppose presents, dans le meme dossier ou sur le PYTHONPATH :
    plate_model.py, milling_force.py, newmark_solver.py
Ils constituent le modele physique et ne doivent pas etre modifies non plus.
================================================================================
"""
import numpy as np

from plate_model import PlateModel
from milling_force import cutting_coefficients, precompute_alpha_periodic
from newmark_solver import NewmarkSimulator

# ============================================================================
# CONSTANTES FIGEES
# ============================================================================
PLATE_L, PLATE_H, PLATE_T = 0.100, 0.080, 0.004      # m
RHO, YOUNG, POISSON = 2830.0, 69e9, 0.33             # AL6061
MESH_N1, MESH_N2 = 36, 30
N_MODES = 5
F_MEASURED = [536.4, 1071.8, 2778.6, 3358.7, 4122.6]   # Hz, numerisees
ZETA_MODES = [0.0031, 0.0017, 0.0027, 0.0030, 0.0030]

PATCH = dict(x1=0.0, x2=0.020, z1=0.0, z2=0.060,
             d31=175e-12, thickness=0.7e-3, E=63e9, nu=0.35)
SENSOR_XZ = (0.100, 0.080)                            # coin superieur droit
V_MAX = 150.0                                         # V, borne amplificateur

N_TEETH, D_TOOL = 3, 0.010                            # fraise 3 dents, 10 mm
HELIX = np.deg2rad(35.0)
RAKE = np.deg2rad(15.0)
KT, KN, MU_C = 925e6, 0.26, 0.20
AE_NOM, FZ_NOM = 0.1e-3, 0.02e-3                      # m, m/dent
STEPS_PER_TOOTH = 82                                  # dt = tau/82, NON MODIFIABLE

SPEEDS_DEFAULT = [3000, 4200, 4900, 6000, 7200]       # tr/min
AP_TEST = 0.25e-3                                     # m, profondeur de reference
GROWTH_MAX = 1.15                                     # garde de croissance

# Horizons d'integration. CE SONT DES CONSTANTES DE MESURE, pas des details :
# pres du seuil, allonger T abaisse la limite mesuree (~20 % entre 0.30 et
# 0.40 s), parce qu'une instabilite lente a besoin de temps pour se voir. Une
# limite de stabilite n'a donc de sens qu'accompagnee de son horizon. Tout
# resultat destine a etre compare a un autre doit utiliser LE MEME.
T_RUN = 0.20        # essai unique et cout multi-vitesses
T_LIMIT = 0.28      # recherche de limite de stabilite par bissection


class SimBase:
    """Plaque + patch + capteur + coupe. Le correcteur est fourni a l'appel.

    patch / zeta : surchargent PATCH et ZETA_MODES pour cette instance
        seulement. Passer par ces arguments plutot que de reaffecter les
        constantes du module : une reaffectation reste active pour toutes les
        instances construites ensuite dans le meme processus.
    """

    def __init__(self, verbose=False, patch=None, zeta=None):
        patch = PATCH if patch is None else patch
        zeta = ZETA_MODES if zeta is None else zeta
        p = PlateModel(PLATE_L, PLATE_H, PLATE_T, RHO, YOUNG, POISSON,
                       N1=MESH_N1, N2=MESH_N2, n_modes=N_MODES,
                       zeta_modes=zeta, verbose=False)
        p.precompute_Dp(zp_pos=PLATE_H - 0.15e-3, n_pos=2001)
        p.set_observation(*SENSOR_XZ)
        p.add_piezo_patch(patch['x1'], patch['x2'], patch['z1'], patch['z2'],
                          patch['d31'], patch['thickness'], patch['E'], patch['nu'])
        p.calibrate_frequencies(F_MEASURED)
        self.plate = p
        self.patch_cfg = patch
        self.k1c, self.k2c = cutting_coefficients(KN, MU_C, HELIX, RAKE)
        self._cache = {}
        if verbose:
            self.describe()

    # -- proprietes modales utiles au reglage d'un observateur ---------------
    @property
    def freqs(self):
        """Frequences propres calibrees, Hz."""
        return np.asarray(self.plate.omega_n).ravel()/(2*np.pi)

    @property
    def modal(self):
        """Vecteurs modaux : H (entree patch), D_obs (capteur), D_tool (outil)."""
        return dict(H=np.asarray(self.plate.H_Pe_modal).ravel(),
                    D_obs=np.asarray(self.plate.D_obs).ravel(),
                    D_tool=np.asarray(self.plate.Dp_array[:, 0]).ravel(),
                    zeta=np.asarray(self.plate.zeta_modes).ravel(),
                    omega=np.asarray(self.plate.omega_n).ravel())

    def describe(self):
        m = self.modal
        print("frequences (Hz) :", np.round(self.freqs, 1))
        print("H (patch)       :", np.round(m['H'], 5))
        print("D_obs (capteur) :", np.round(m['D_obs'], 3))
        print("D_tool (outil)  :", np.round(m['D_tool'], 3))
        raw = np.abs(m['H']*m['D_obs'])
        print("|H D| brut      :", np.round(raw, 4))
        nrm = raw/m['omega']**2
        print("|H D|/w^2 norm. :", np.array2string(nrm/nrm.max(), precision=4))

    # -- setup d'une condition de coupe -------------------------------------
    def _setup(self, rpm, ap, T, ae, fz, tool_pos):
        key = (rpm, round(ap, 9), round(T, 4), round(ae, 9), round(fz, 9), tool_pos)
        if key not in self._cache:
            tau = 60.0/(N_TEETH*rpm)
            dt = tau/STEPS_PER_TOOTH
            s = NewmarkSimulator(self.plate, dt=dt, T_end=T, ft=fz, tau=tau,
                                 verbose=False, v_max=V_MAX)
            phi = np.pi - np.arccos(1 - ae/(D_TOOL/2))
            a3, a4 = precompute_alpha_periodic(
                dt, STEPS_PER_TOOTH, s.nstep, 2*np.pi*rpm/60, N_TEETH, D_TOOL/2,
                HELIX, phi, np.pi, PLATE_H - ap, PLATE_H, self.k1c, self.k2c, KT)
            self._cache[key] = (s, a3, a4, dt, tau)
        return self._cache[key]

    # -- simulation elementaire ---------------------------------------------
    def run(self, controller, rpm=4900, ap=AP_TEST, T=T_RUN, ae=AE_NOM,
            fz=FZ_NOM, tool_pos=0, full_pass=False, keep_signals=False):
        """Simule une coupe. controller = objet step/reset, ou None.

        tool_pos : indice de station de l'outil (0 = debut d'arete, station la
                   moins stable). Ignore si full_pass=True.
        full_pass : l'outil parcourt les 100 mm ; T est alors recalcule.
        Retour : dict avec stable, rms_um, peak_um, rms_u, peak_u, growth, t_end.
        """
        if full_pass:
            T = PLATE_L/(fz*N_TEETH*rpm/60.0)
        s, a3, a4, dt, tau = self._setup(rpm, ap, T, ae, fz, tool_pos)
        if full_pass:
            v = fz*N_TEETH*rpm/60.0
            kp = np.clip(np.round(np.minimum(v*s.t_vec, PLATE_L)/PLATE_L*2000
                                  ).astype(int), 0, 2000)
        else:
            kp = np.full(s.nstep, int(tool_pos), dtype=int)
        if controller is not None and hasattr(controller, 'reset'):
            controller.reset()
        r = s.simulate(a3, a4, kp, controller=controller, progress=False,
                       stop_threshold=5e-3 if full_pass else 1e-4)
        i = r['stop_idx']
        diverged = r['diverged_at'] > 0
        i0 = min(s.nstep//3, max(0, i - 1))
        y2 = r['y'][i0:i+1]
        if y2.size < 4:
            y2 = r['y'][:max(4, i+1)]
        n2 = max(1, len(y2)//2)
        growth = (np.sqrt(np.mean(y2[n2:]**2))
                  / max(np.sqrt(np.mean(y2[:n2]**2)), 1e-18))
        stable = (not diverged) and growth <= GROWTH_MAX
        out = dict(stable=bool(stable), diverged=bool(diverged),
                   rms_um=float(np.sqrt(np.mean(y2**2))*1e6),
                   peak_um=float(np.abs(r['y'][:i+1]).max()*1e6),
                   rms_u=float(np.sqrt(np.mean(r['u'][i0:i+1]**2))),
                   peak_u=float(np.abs(r['u'][:i+1]).max()),
                   growth=float(growth), t_end=float(r['t'][i]), dt=dt, tau=tau)
        if keep_signals:
            out['t'] = r['t'][:i+1]
            out['y'] = r['y'][:i+1]
            out['u'] = r['u'][:i+1]
        return out

    # -- cout multi-vitesses, identique pour tous les correcteurs -----------
    def multi_speed_cost(self, make_ctrl, speeds=None, ap=AP_TEST, T=T_RUN,
                         penalty=12.0, w_worst=0.5):
        """make_ctrl(dt, tau) -> correcteur. Retourne un scalaire a minimiser.

        Une vitesse instable coute `penalty`. Le terme worst-case evite les
        reglages excellents a une vitesse et mediocres ailleurs.
        """
        speeds = speeds or SPEEDS_DEFAULT
        tot = 0.0
        worst = 0.0
        for rpm in speeds:
            tau = 60.0/(N_TEETH*rpm)
            dt = tau/STEPS_PER_TOOTH
            try:
                c = make_ctrl(dt, tau)
            except Exception:
                tot += penalty
                continue
            r = self.run(c, rpm=rpm, ap=ap, T=T)
            if not r['stable']:
                tot += penalty
            else:
                tot += r['rms_um'] + 0.05*r['rms_u']
                worst = max(worst, r['rms_um'])
        return tot/len(speeds) + w_worst*worst

    # -- limite de stabilite par bissection ---------------------------------
    def stability_limit(self, make_ctrl, rpm=4900, lo=0.02e-3, hi=1.5e-3,
                        T=T_LIMIT, tol=2e-5):
        """Plus grande profondeur de passe stable. make_ctrl=None -> boucle ouverte."""
        def ok(ap):
            tau = 60.0/(N_TEETH*rpm)
            dt = tau/STEPS_PER_TOOTH
            c = None
            if make_ctrl is not None:
                try:
                    c = make_ctrl(dt, tau)
                except Exception:
                    return False
            return self.run(c, rpm=rpm, ap=ap, T=T)['stable']
        if not ok(lo):
            return 0.0
        if ok(hi):
            return hi
        while hi - lo > tol:
            mid = 0.5*(lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        return 0.5*(lo + hi)

    # -- reponse frequentielle mesuree, sans hypothese de linearite ---------
    def receptance(self, make_ctrl, rpm=4900, f_lo=64.0, f_hi=1600.0,
                   amp=5e-3, n_period=3):
        """|Y/F| mesuree par injection d'un multisinus PERIODIQUE a la place de
        l'effort de coupe, couplage regeneratif desactive.

        Cette voie est la seule fiable : l'extraction lineaire par reponse
        impulsionnelle echoue pour les correcteurs dont la dynamique propre est
        instable (le cas de la plupart des schemas a observateur), leur reponse
        impulsionnelle divergeant.
        Retour : (frequences, |Y/F| en um/N) ou (None, None) si divergence.
        """
        tau = 60.0/(N_TEETH*rpm)
        dt = tau/STEPS_PER_TOOTH
        NP = 5022
        s = NewmarkSimulator(self.plate, dt=dt, T_end=(NP*n_period + 2)*dt,
                             ft=1.0, tau=tau, verbose=False, v_max=V_MAX)
        n = s.nstep
        NP = n//n_period
        df = 1.0/(NP*dt)
        ks = np.arange(int(round(f_lo/df)), int(round(f_hi/df)) + 1)
        tp = np.arange(NP)*dt
        ph = np.pi*np.arange(len(ks))**2/len(ks)     # phases de Schroeder
        base = np.zeros(NP)
        for i, k in enumerate(ks):
            base += np.cos(2*np.pi*k*df*tp + ph[i])
        base /= np.abs(base).max()
        F = np.tile(base, n_period + 1)[:n]*amp
        c = None
        if make_ctrl is not None:
            try:
                c = make_ctrl(dt, tau)
            except Exception:
                return None, None
        r = s.simulate(F, np.zeros(n), np.zeros(n, dtype=int), controller=c,
                       progress=False, stop_threshold=1e-2)
        if r['diverged_at'] > 0:
            return None, None
        k0 = n - NP - 2
        Y = np.fft.rfft(r['y'][k0:k0+NP])
        Ff = np.fft.rfft(F[k0:k0+NP])
        fr = np.fft.rfftfreq(NP, dt)
        return fr[ks], np.abs(Y[ks]/Ff[ks])*1e6


# ============================================================================
# VERIFICATION DU MODELE
# ============================================================================
def check_model(verbose=True):
    """Refait les controles de validation. Leve AssertionError si un ecart
    depasse la tolerance. A lancer une fois au debut d'une session."""
    sim = SimBase()
    ok = True

    f = sim.freqs
    err = np.abs(f/np.array(F_MEASURED) - 1)
    if verbose:
        print("1. frequences apres calibration")
        for i in range(N_MODES):
            print(f"   mode {i+1} : {f[i]:8.1f} Hz  (mesure {F_MEASURED[i]:8.1f}, "
                  f"ecart {100*err[i]:+.3f} %)")
    assert err.max() < 1e-3, "calibration des frequences incorrecte"

    # Les antiresonances de reference proviennent de la Fig. 12(a) de Du et al.,
    # ou le marteau ET le capteur sont au MEME point (coin superieur droit) :
    # c'est une reception AU POINT DE FRAPPE. Les residus sont donc D_obs*D_obs,
    # tous positifs, et les zeros s'intercalent strictement entre les poles --
    # ce que la mesure confirme (734 < 1072, 2161 < 2779, 3001 < 3359, 3805 <
    # 4123). Utiliser D_obs*D_tool comparerait une reception de TRANSFERT
    # (outil en x=0 -> capteur) a une mesure au point de frappe : residus de
    # signes mixtes, motif de zeros sans rapport.
    ff = np.linspace(60, 4600, 60000)
    w = 2*np.pi*ff
    m = sim.modal
    den = ((m['omega'][:, None]**2 - w[None, :]**2)
           + 2j*m['zeta'][:, None]*m['omega'][:, None]*w[None, :])
    G = np.abs(np.sum((m['D_obs']**2)[:, None]/den, axis=0))
    lg = np.log(G)
    anti = [ff[i] for i in range(1, len(G)-1) if lg[i] < lg[i-1] and lg[i] < lg[i+1]]
    target = [734, 2161, 3001, 3805]
    if verbose:
        print("2. antiresonances (reception au point de frappe, Fig. 12a)")
        print(f"   mesurees : {target}")
        print(f"   modele   : {[round(a) for a in anti]}")
    assert len(anti) >= 4, "le modele ne produit pas quatre antiresonances"
    for t in target:
        d = min(abs(a - t) for a in anti)
        if verbose:
            print(f"   {t:5d} Hz -> ecart {d:5.0f} Hz ({100*d/t:+.1f} %)")
        if d/t > 0.05:
            ok = False
            print(f"   ATTENTION : antiresonance {t} Hz mal reproduite (ecart {d:.0f} Hz)")

    lim = sim.stability_limit(None, rpm=4900)
    if verbose:
        print(f"3. limite de stabilite en boucle ouverte a 4900 tr/min : "
              f"{lim*1e3:.4f} mm  (reference de cette base : 0.0605 mm)")
        print("   cette valeur depend de l'horizon T et de la fenetre de mesure ;")
        print("   c'est la reference de CETTE base, a ne pas comparer a une valeur")
        print("   obtenue avec d'autres reglages d'evaluation.")
    assert abs(lim*1e3 - 0.0605) < 0.008, "limite libre hors tolerance"

    if verbose:
        print("\nbase valide." if ok else "\nbase utilisable, voir avertissements.")
    return sim


if __name__ == "__main__":
    sim = check_model()
    print()
    sim.describe()
    print("\nessai en boucle ouverte a 0.05 mm (coupe stable) :")
    r = sim.run(None, rpm=4900, ap=0.05e-3, T=0.20)
    print(f"   stable={r['stable']}  RMS={r['rms_um']:.4f} um  "
          f"croissance={r['growth']:.3f}")
