"""main_adaptive_drift.py
=======================
Traitement de la LACUNE de l'article ASME sur la plaque de CET article.

Scénario : pendant l'usinage, l'enlèvement de matière fait dériver les
fréquences propres de la plaque de 0 à -15 % en T = 5 s (l'équivalent de
l'Eq. (6) de l'article ASME ; -15 % est aussi l'amplitude du scénario S3
du package).  Les contrôleurs STATIQUES (LQG, DARC-feedforward) sont
conçus une fois sur le modèle nominal et deviennent progressivement
faux ; les contrôleurs ADAPTATIFS identifient la dérive en ligne (RLS) et
re-conçoivent Kalman + LQR (+ le modèle inverse du feedforward) en cours
d'usinage.

4 cas, ablation propre (mêmes structures, seule l'adaptation change) :
   1. LQG statique            2. DARC-FF statique
   3. LQG adaptatif           4. DARC-FF adaptatif

Sortie : figs_adaptive_drift/ + résumé console + JSON.
"""

import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plate_model import PlateModel
from milling_force import precompute_nonlinear_periodic
from newmark_solver import NewmarkSimulator
from adaptive_lqg_controller import (AdaptiveDARCFFController,
                                     AdaptiveLQGController)

# ----------------------------------------------------------------------
# Paramètres physiques (identiques au package)
# ----------------------------------------------------------------------
LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU_AL = 2830, 69e9, 0.33
NT, DT_TOOL = 3, 0.010
ETA_H = np.deg2rad(35.0)
GAMMA_N = np.deg2rad(15.0)
KT, KN, MU_C = 925e6, 0.26, 0.20
RPM = 4900
FT = 0.02e-3
AE = 0.1e-3
AP = 0.3e-3
N1, N2, N_MODES = 30, 24, 3
ZETA = [0.0031, 0.0017, 0.0027]
D31, H_PA, E_PE, NU_PE = 175e-12, 0.7e-3, 63e9, 0.35

DT = 5e-5
T_END = 6.0
DRIFT_END = 0.78            # s(T) : -22 % de fréquences en fin de passe
                            # (traverse la falaise d'instabilité du
                            #  contrôleur statique, située vers s ≈ 0.84)
SENSOR_NOISE = 1.0e-7       # capteur fin 0.1 um RMS (comme main_simulation)

OUT = "figs_adaptive_drift"
os.makedirs(OUT, exist_ok=True)

COLORS = {"LQG statique": "#D55E00", "DARC-FF statique": "#CC79A7",
          "LQG adaptatif": "#0072B2", "DARC-FF adaptatif": "#009E73"}

# ----------------------------------------------------------------------
# Plaque + coupe
# ----------------------------------------------------------------------
plate = PlateModel(LP, HP, BP, RHO, E_AL, NU_AL, N1=N1, N2=N2,
                   n_modes=N_MODES, zeta_modes=ZETA, verbose=False)
plate.precompute_Dp(zp_pos=HP - AP / 2, n_pos=2001)
plate.set_observation(x_obs=LP, z_obs=HP)
plate.add_piezo_patch(0, 0.020, 0, 0.060, D31, H_PA, E_PE, NU_PE)
f1_nom = float(plate.freq_n[0])

Omega = 2 * np.pi * RPM / 60
RT = DT_TOOL / 2
tau = 60 / (NT * RPM)
phi_st = np.pi - np.arccos(1 - AE / RT)
phi_ex = np.pi
k1 = KN * np.cos(ETA_H)
k2 = 1 + MU_C * np.tan(ETA_H) * np.cos(GAMMA_N) - KN * np.sin(ETA_H)
v_feed = FT * NT * RPM / 60

sim = NewmarkSimulator(plate, dt=DT, T_end=T_END, ft=FT, tau=tau,
                       verbose=False)
nstep = sim.nstep
n_per = int(np.round(tau / DT))

# trajet outil : aller-retour sur la largeur pour couvrir T_END
xp = np.abs(((v_feed * sim.t_vec / LP) % 2.0) - 1.0) * LP
kp_idx = np.clip(np.round(xp / LP * 2000).astype(int), 0, 2000)

a3_t, a4_t, a42_t, a43_t = precompute_nonlinear_periodic(
    DT, n_per, nstep, Omega, NT, RT, ETA_H, phi_st, phi_ex,
    HP - AP, HP, k1, k2, KT)

# dérive structurelle : s(t) linéaire 1 -> DRIFT_END  (matière enlevée)
s_t = 1.0 + (DRIFT_END - 1.0) * sim.t_vec / T_END
Dp_ff = plate.get_Dp_at(int(kp_idx[nstep // 2]))[0]


def make_controller(kind: str):
    common = dict(dt=DT, rpm=RPM, ident_ft=FT,
                  ident_a3_period=a3_t[:n_per], ident_Dp=Dp_ff)
    if kind == "LQG statique":
        c = AdaptiveLQGController(plate, **common)
        c.s_tol = np.inf                     # ne se re-conçoit jamais
    elif kind == "LQG adaptatif":
        c = AdaptiveLQGController(plate, **common, verbose=True)
    elif kind == "DARC-FF statique":
        c = AdaptiveDARCFFController(plate, ft=FT, a3_period=a3_t[:n_per],
                                     Dp_ff=Dp_ff, **common)
        c.s_tol = np.inf
    elif kind == "DARC-FF adaptatif":
        c = AdaptiveDARCFFController(plate, ft=FT, a3_period=a3_t[:n_per],
                                     Dp_ff=Dp_ff, **common, verbose=True)
    return c


def run_case(kind: str):
    ctl = make_controller(kind)
    sim_c = NewmarkSimulator(plate, dt=DT, T_end=T_END, ft=FT, tau=tau,
                             verbose=False)
    sim_c._stiff_scale_t = s_t              # plante à dynamique dérivante
    t0 = time.time()
    res = sim_c.simulate(a3_t, a4_t, kp_idx, controller=ctl,
                         alpha4_2_t=a42_t, alpha4_3_t=a43_t,
                         sensor_noise=SENSOR_NOISE,
                         sensor_rng=np.random.default_rng(7),
                         progress=False)
    wall = time.time() - t0
    y = res["y"]
    u = res.get("u", res.get("u_cmd", np.zeros_like(y)))
    seg = int(1.0 / DT)
    out = {
        "kind": kind,
        "y_rms_um": float(np.std(y) * 1e6),
        "y_rms_first_um": float(np.std(y[:seg]) * 1e6),
        "y_rms_last_um": float(np.std(y[-seg:]) * 1e6),
        "y_max_um": float(np.max(np.abs(y)) * 1e6),
        "u_rms_V": float(np.std(u)),
        "u_max_V": float(np.max(np.abs(u))),
        "wall_s": wall,
    }
    print(f"  {kind:20s}: RMS {out['y_rms_um']:.3f} um "
          f"(début {out['y_rms_first_um']:.3f} → fin "
          f"{out['y_rms_last_um']:.3f}), max {out['y_max_um']:.2f} um, "
          f"u_max {out['u_max_V']:.1f} V   [{wall:.0f}s]")
    return out, res, ctl


print("=" * 70)
print(" Traitement de la lacune ASME sur la plaque de l'article")
print(f"  dérive: f -> {DRIFT_END:.2f} f  en {T_END:.0f} s   "
      f"(f1 {f1_nom:.0f} -> {DRIFT_END * f1_nom:.0f} Hz)")
print("=" * 70)

cases, results, ctls = {}, {}, {}
for kind in ["LQG statique", "DARC-FF statique",
             "LQG adaptatif", "DARC-FF adaptatif"]:
    cases[kind], results[kind], ctls[kind] = run_case(kind)

# ----------------------------------------------------------------------
# Résumé + amélioration
# ----------------------------------------------------------------------
imp_lqg = 100 * (1 - cases["LQG adaptatif"]["y_rms_last_um"]
                 / cases["LQG statique"]["y_rms_last_um"])
imp_darc = 100 * (1 - cases["DARC-FF adaptatif"]["y_rms_last_um"]
                  / cases["DARC-FF statique"]["y_rms_last_um"])
print("\n  Amélioration ADAPTATIF vs STATIQUE (RMS dernière seconde) :")
print(f"    LQG      : {imp_lqg:+.1f}%")
print(f"    DARC-FF  : {imp_darc:+.1f}%")

# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 130, "font.size": 9.5,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False,
                     "figure.constrained_layout.use": True})

fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))

ax = axes[0, 0]
win = int(0.1 / DT)
for kind, res in results.items():
    y = res["y"]
    nwin = len(y) // win
    rms = np.std(y[: nwin * win].reshape(nwin, win), axis=1) * 1e6
    tt = (np.arange(nwin) + 0.5) * 0.1
    ax.plot(tt, rms, lw=1.6, color=COLORS[kind], label=kind)
ax.set_xlabel("temps (s)")
ax.set_ylabel("RMS glissant 0.1 s (μm)")
ax.set_title("(a) Vibration pendant la dérive structurelle "
             f"(f → {DRIFT_END:.2f} f)", loc="left", fontsize=10)
ax.legend(fontsize=8)

ax = axes[0, 1]
f1_true = s_t * f1_nom
ax.plot(sim.t_vec[::200], f1_true[::200], color="#333333", lw=1.8,
        label="f₁ réel (dérive imposée)")
ctl_a = ctls["LQG adaptatif"]
ax.plot(ctl_a.log_t, np.asarray(ctl_a.log_s) * f1_nom,
        color=COLORS["LQG adaptatif"], lw=1.4, ls="--",
        label="f̂₁ identifié (banc d'observateurs multi-modèles)")
ax.set_xlabel("temps (s)")
ax.set_ylabel("fréquence mode 1 (Hz)")
ax.set_title("(b) Identification en ligne de la dérive", loc="left",
             fontsize=10)
ax.legend(fontsize=8)

ax = axes[1, 0]
kinds = list(cases)
x = np.arange(len(kinds))
first = [cases[k]["y_rms_first_um"] for k in kinds]
last = [cases[k]["y_rms_last_um"] for k in kinds]
ax.bar(x - 0.18, first, 0.36, label="1ʳᵉ seconde (modèle encore juste)",
       color="#88BBDD")
ax.bar(x + 0.18, last, 0.36, label="dernière seconde (modèle dérivé)",
       color="#334455")
ax.set_xticks(x, [k.replace(" ", "\n") for k in kinds], fontsize=8)
ax.set_ylabel("y_RMS (μm)")
ax.set_title("(c) Dégradation début → fin de passe", loc="left",
             fontsize=10)
ax.legend(fontsize=8)

ax = axes[1, 1]
t_show = 0.88 * T_END
for kind in ["LQG statique", "LQG adaptatif"]:
    y = results[kind]["y"]
    i0 = int(t_show / DT)
    ax.plot(sim.t_vec[i0:i0 + 6000], y[i0:i0 + 6000] * 1e6, lw=0.7,
            color=COLORS[kind], label=kind, alpha=0.9)
ax.set_xlabel("temps (s)")
ax.set_ylabel("y (μm)")
ax.set_title(f"(d) Détail t = {t_show:.1f}–{t_show + 0.3:.1f} s : "
             "le statique diverge, l'adaptatif tient", loc="left",
             fontsize=10)
ax.legend(fontsize=8)

fig.savefig(os.path.join(OUT, "fig_adaptive_drift.png"),
            bbox_inches="tight")
plt.close(fig)

with open(os.path.join(OUT, "adaptive_drift_summary.json"), "w") as f:
    json.dump({"drift_end": DRIFT_END, "T_end": T_END,
               "cases": cases,
               "improvement_last_second_pct": {
                   "LQG": imp_lqg, "DARC_FF": imp_darc}}, f, indent=2)

print(f"\n  figure -> {OUT}/fig_adaptive_drift.png")
