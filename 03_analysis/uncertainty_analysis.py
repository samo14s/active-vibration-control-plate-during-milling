"""
uncertainty_analysis.py
========================
Analyse de robustesse par Monte Carlo avec incertitudes paramétriques.

Inspirée de l'article : ±5% sur les paramètres de coupe (kt, kn, mu_c)
et ±2% sur les fréquences propres et amortissements modaux.

Le contrôleur LQG est conçu sur le modèle nominal puis testé sur
N_MC simulations avec paramètres aléatoires.
"""
import numpy as np
import time
import copy

from milling_force import precompute_alpha_periodic, cutting_constants
from newmark_solver import NewmarkSimulator


# ====================================================================
#  Parametres d'incertitude (inspirés de l'article)
# ====================================================================
DEFAULT_UNCERTAINTY = dict(
    # Coefficients de coupe (Tableau 3) ±5%
    kt_pct      = 0.05,
    kn_pct      = 0.05,
    mu_c_pct    = 0.05,

    # Modes (Tableaux 1, 4) ±2%
    omega_pct   = 0.02,        # variations de fréquence propre
    zeta_pct    = 0.20,        # variations d'amortissement (large car difficile à mesurer)

    # Amortissement matériau (E_AL ±3%)
    E_pct       = 0.03,
)


def sample_uncertain_params(rng, nominal, unc=None):
    """
    Échantillonne des paramètres incertains autour des valeurs nominales.
    Distribution uniforme sur [-pct, +pct].
    """
    if unc is None:
        unc = DEFAULT_UNCERTAINTY

    p = copy.copy(nominal)

    p['kt']    *= 1 + rng.uniform(-unc['kt_pct'],   unc['kt_pct'])
    p['kn']    *= 1 + rng.uniform(-unc['kn_pct'],   unc['kn_pct'])
    p['mu_c']  *= 1 + rng.uniform(-unc['mu_c_pct'], unc['mu_c_pct'])

    # Variations sur les modes (omega_n et zeta)
    n_modes = len(p['omega_n'])
    omega_perturb = rng.uniform(-unc['omega_pct'], unc['omega_pct'], n_modes)
    zeta_perturb  = rng.uniform(-unc['zeta_pct'],  unc['zeta_pct'],  n_modes)

    p['omega_n'] = nominal['omega_n'] * (1 + omega_perturb)
    p['zeta']    = np.maximum(1e-4, nominal['zeta'] * (1 + zeta_perturb))

    return p


def run_monte_carlo(plate_nominal, ctrl, sim_template,
                     nominal_params, kp_idx, dt,
                     n_samples=50, T_stop_no_ctrl=0.2,
                     stop_threshold=5e-3, seed=42, verbose=True):
    """
    Lance N simulations Monte Carlo avec paramètres incertains.

    Pour chaque échantillon :
      - on construit alpha3, alpha4 avec les paramètres perturbés
      - on perturbe les modes en remplaçant Kp = diag(omega²) et Cp = diag(2*zeta*omega)
      - on lance Sim 1 (sans contrôle) et Sim 2 (avec LQG nominal)
      - on stocke y(t) et u(t)

    Le contrôleur LQG est conçu sur le modèle nominal — c'est le test
    de robustesse : peut-il gérer des plantes différentes du modèle
    sur lequel il a été conçu ?
    """
    rng = np.random.default_rng(seed)

    nstep = sim_template.nstep
    n_modes = plate_nominal.n_modes

    # Stockage : y[i, k] = trajectoire i au pas k
    y_no_ctrl_all   = np.full((n_samples, nstep), np.nan)
    y_with_ctrl_all = np.full((n_samples, nstep), np.nan)
    u_with_ctrl_all = np.zeros((n_samples, nstep))
    stop_idx_no = np.zeros(n_samples, dtype=int)
    stop_idx_yes = np.zeros(n_samples, dtype=int)

    if verbose:
        print(f"\n=== Monte Carlo : {n_samples} échantillons ===")
        t0 = time.time()

    for i in range(n_samples):
        # 1. Échantillonner paramètres incertains
        p = sample_uncertain_params(rng, nominal_params)

        # 2. Mettre à jour les coefficients de force (k1, k2) — Eq. (3) corrigée
        k1, k2 = cutting_constants(p['kn'], p['mu_c'],
                                   nominal_params['eta_h'],
                                   nominal_params['gamma_n'])

        # 3. Pré-calculer alpha3, alpha4 perturbés
        n_per = int(np.round(nominal_params['tau']/dt))
        alpha3_t, alpha4_t = precompute_alpha_periodic(
            dt, n_per, nstep,
            nominal_params['Omega'],
            nominal_params['NT'],
            nominal_params['RT'],
            nominal_params['eta_h'],
            nominal_params['phi_st'],
            nominal_params['phi_ex'],
            nominal_params['za_low'],
            nominal_params['za_high'],
            k1, k2, p['kt'])

        # 4. Modifier les matrices modales de la plaque
        # On crée une copie superficielle et on remplace Kp, Cp
        plate_perturbed = copy.copy(plate_nominal)
        plate_perturbed.Kp = np.diag(p['omega_n']**2)
        plate_perturbed.Cp = np.diag(2 * p['zeta'] * p['omega_n'])
        plate_perturbed.omega_n = p['omega_n']
        plate_perturbed.zeta_modes = p['zeta']

        # 5. Sim 1 : sans contrôle
        sim_no = NewmarkSimulator(plate_perturbed,
                                   dt=dt, T_end=sim_template.T_end,
                                   ft=sim_template.ft, tau=sim_template.tau,
                                   verbose=False)
        res_no = sim_no.simulate(alpha3_t, alpha4_t, kp_idx,
                                  controller=None,
                                  stop_at_time=T_stop_no_ctrl,
                                  stop_threshold=stop_threshold,
                                  progress=False)
        i1 = res_no['stop_idx']
        y_no_ctrl_all[i, :i1+1] = res_no['y'][:i1+1]
        stop_idx_no[i] = i1

        # 6. Sim 2 : avec LQG nominal sur plante perturbée
        # Le LQG (ctrl) reste celui conçu sur le modèle nominal
        sim_yes = NewmarkSimulator(plate_perturbed,
                                    dt=dt, T_end=sim_template.T_end,
                                    ft=sim_template.ft, tau=sim_template.tau,
                                    verbose=False)
        res_yes = sim_yes.simulate(alpha3_t, alpha4_t, kp_idx,
                                    controller=ctrl,
                                    stop_threshold=stop_threshold,
                                    progress=False)
        i2 = res_yes['stop_idx']
        y_with_ctrl_all[i, :i2+1] = res_yes['y'][:i2+1]
        u_with_ctrl_all[i, :i2+1] = res_yes['u'][:i2+1]
        stop_idx_yes[i] = i2

        if verbose and ((i+1) % max(1, n_samples//5) == 0):
            elapsed = time.time() - t0
            eta_s = elapsed * (n_samples - i - 1) / (i + 1)
            print(f"   {i+1:3d}/{n_samples}  "
                  f"y_open_max={np.nanmax(np.abs(y_no_ctrl_all[i]))*1e6:7.1f}um  "
                  f"y_ctrl_max={np.nanmax(np.abs(y_with_ctrl_all[i]))*1e6:7.2f}um  "
                  f"u_max={np.max(np.abs(u_with_ctrl_all[i])):6.1f}V  "
                  f"({elapsed:5.1f}s écoulées, ETA {eta_s:4.0f}s)")

    if verbose:
        print(f"   Monte Carlo terminé en {time.time()-t0:.1f}s")

    return dict(
        y_no_ctrl=y_no_ctrl_all,
        y_with_ctrl=y_with_ctrl_all,
        u_with_ctrl=u_with_ctrl_all,
        stop_idx_no=stop_idx_no,
        stop_idx_yes=stop_idx_yes,
        n_samples=n_samples,
    )


def run_mc_controllers(plant_nominal, controllers, nominal_params, kp_idx,
                       dt, T_end, ft, tau, n_per,
                       unc=None, n_samples=60, meas_noise_std=1e-8,
                       stop_threshold=5e-3, seed=42, verbose=True):
    """
    Monte-Carlo robustness comparison of an ARBITRARY set of FROZEN controllers
    (all designed once on the nominal model — held-out) over parametric uncertainty.

    Parameters
    ----------
    controllers : dict {name: controller}. The FIRST entry is the baseline for
        the pairwise gain statistics. Controllers exposing `reset_adaptation()`
        (e.g. A-ADRC) have it called before every sample, so each run starts from
        the same nominal design — the ADAPTATION runs, but never carries state
        across samples.

    For each sample the plant is perturbed (per-mode omega, zeta, and the cutting
    parameters kt/kn/mu_c), the controllers are UNCHANGED, and all run with the same
    measurement-noise realisation. Divergence is recorded EXPLICITLY (no survivorship
    bias): a sample counts as "converged" only if it did not hit the stop threshold
    and its RMS stayed bounded. Pairwise gain statistics vs the baseline are reported
    over samples where BOTH converged; diverged counts are returned separately.
    """
    from milling_force import precompute_alpha_periodic, cutting_constants
    from newmark_solver import NewmarkSimulator
    rng = np.random.default_rng(seed)

    names = list(controllers.keys())
    base = names[0]
    rms = {nm: np.full(n_samples, np.nan) for nm in names}
    conv = {nm: np.zeros(n_samples, dtype=bool) for nm in names}

    if verbose:
        print(f"\n=== Monte-Carlo {' vs '.join(names)} : {n_samples} samples ===")
        t0 = time.time()

    def _rms_conv(res, nstep):
        i = res['stop_idx']
        y = res['y'][:i+1]
        r = np.sqrt(np.mean(y**2)) * 1e6 if len(y) else np.inf
        converged = (i >= nstep - 2) and (r < 50.0)   # 50 um sanity ceiling
        return r, converged

    for s in range(n_samples):
        p = sample_uncertain_params(rng, nominal_params, unc)
        k1, k2 = cutting_constants(p['kn'], p['mu_c'],
                                   nominal_params['eta_h'], nominal_params['gamma_n'])

        pp = copy.copy(plant_nominal)
        pp.omega_n = p['omega_n']
        pp.freq_n = p['omega_n'] / (2 * np.pi)
        pp.Kp = np.diag(p['omega_n']**2)
        pp.Cp = np.diag(2 * p['zeta'] * p['omega_n'])
        pp.zeta_modes = p['zeta']

        sim = NewmarkSimulator(pp, dt=dt, T_end=T_end, ft=ft, tau=tau, verbose=False)
        a3, a4 = precompute_alpha_periodic(
            dt, n_per, sim.nstep, nominal_params['Omega'], nominal_params['NT'],
            nominal_params['RT'], nominal_params['eta_h'], nominal_params['phi_st'],
            nominal_params['phi_ex'], nominal_params['za_low'],
            nominal_params['za_high'], k1, k2, p['kt'])

        for nm in names:
            ctrl = controllers[nm]
            if hasattr(ctrl, 'reset_adaptation'):
                ctrl.reset_adaptation()
            r = sim.simulate(a3, a4, kp_idx, controller=ctrl,
                             meas_noise_std=meas_noise_std,
                             rng=np.random.default_rng(1234),
                             stop_threshold=stop_threshold, progress=False)
            rms[nm][s], conv[nm][s] = _rms_conv(r, sim.nstep)

        if verbose and ((s + 1) % max(1, n_samples // 6) == 0):
            msg = "  ".join(f"{nm} {rms[nm][s]:6.3f}um" for nm in names)
            print(f"   {s+1:3d}/{n_samples}  {msg}  ({time.time()-t0:5.1f}s)")

    # Pairwise gains vs the baseline (first controller)
    gains = {}
    gain_stats = {}
    for nm in names[1:]:
        both = conv[base] & conv[nm]
        g = np.full(n_samples, np.nan)
        g[both] = (1 - rms[nm][both] / rms[base][both]) * 100
        gains[nm] = g
        gain_stats[nm] = dict(
            n_both=int(both.sum()),
            mean=float(np.nanmean(g)) if both.any() else float('nan'),
            median=float(np.nanmedian(g)) if both.any() else float('nan'),
            p05=float(np.nanpercentile(g, 5)) if both.any() else float('nan'),
            p95=float(np.nanpercentile(g, 95)) if both.any() else float('nan'),
            pct_better=float(np.mean(g[both] > 0) * 100) if both.any() else float('nan'),
        )

    stats = dict(
        names=names, baseline=base, rms=rms, conv=conv,
        gains=gains, gain_stats=gain_stats, n_samples=n_samples,
        n_conv={nm: int(conv[nm].sum()) for nm in names},
    )
    if verbose:
        conv_msg = ", ".join(f"{nm} {stats['n_conv'][nm]}/{n_samples}" for nm in names)
        print(f"   done in {time.time()-t0:.1f}s | converged: {conv_msg}")
    return stats


def envelope_stats(y_array):
    """
    Calcule mean, min, max, percentile 5 et 95 ligne par ligne.
    Ignore les NaN (simulation arrêtée tôt).
    """
    return dict(
        mean = np.nanmean(y_array, axis=0),
        std  = np.nanstd (y_array, axis=0),
        min  = np.nanmin (y_array, axis=0),
        max  = np.nanmax (y_array, axis=0),
        p05  = np.nanpercentile(y_array,  5, axis=0),
        p95  = np.nanpercentile(y_array, 95, axis=0),
    )


def fft_envelope(y_array, dt, n_freq_max=2000):
    """
    Calcule la FFT pour chaque échantillon et retourne f, |Y(f)| en µm
    pour calculer l'enveloppe dans le domaine fréquentiel.
    """
    n_samples = y_array.shape[0]

    # Trouver longueur commune (nombre de points non-NaN minimum)
    valid_lengths = []
    for i in range(n_samples):
        idx = np.where(~np.isnan(y_array[i]))[0]
        if len(idx) > 100:
            valid_lengths.append(idx[-1] - idx[0] + 1)
    if not valid_lengths:
        return None, None

    N_common = min(valid_lengths)
    NFFT = 2**int(np.ceil(np.log2(max(N_common, 2))))
    f = (1/dt)/2 * np.linspace(0, 1, NFFT//2 + 1)

    Y_all = np.zeros((n_samples, NFFT//2 + 1))
    for i in range(n_samples):
        idx = np.where(~np.isnan(y_array[i]))[0]
        if len(idx) < N_common:
            continue
        y_seg = y_array[i, idx[0]:idx[0]+N_common]
        win = np.hanning(N_common)
        Y = np.fft.fft(y_seg*win, NFFT)/N_common
        Y_all[i, :] = 2*np.abs(Y[:NFFT//2 + 1]) * 1e6

    # Tronquer à f_max
    mask = f <= n_freq_max
    return f[mask], Y_all[:, mask]
