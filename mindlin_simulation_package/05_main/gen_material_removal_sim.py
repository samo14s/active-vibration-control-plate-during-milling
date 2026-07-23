"""
gen_material_removal_sim.py
===========================
In-process (time-varying) workpiece dynamics: the cutter removes material as it
feeds along its path, so the plate's natural frequencies / mode shapes evolve
DURING the cut. Uses `InProcessPlate` (per-element thickness + material removal)
and a fixed-basis reduced-order model for the time-varying response.

Produces figs_removal/inprocess_dynamics.png with:
  (a) natural frequencies vs tool feed position X_tool — finishing vs roughing
  (b) remaining-thickness field h(X_P, Z_P) after the roughing pass
  (c) time response y(t): frozen dynamics vs in-process (time-varying) dynamics
  (d) FFT[y]: the resonance shift caused by material removal

Run:  python gen_material_removal_sim.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("01_core", "02_controllers"):
    sys.path.insert(0, os.path.join(_HERE, "..", sub))

from inprocess_plate import InProcessPlate
from mindlin_q8 import shape_at_point_M

# ------------------------------------------------------------------ params
LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU_AL = 2830, 69e9, 0.33
NT = 3; DT_TOOL = 0.010
ETA_H = np.deg2rad(35); GAMMA_N = np.deg2rad(15)
KT = 925e6; KN = 0.26; MU_C = 0.20
RPM = 4900
FT = 0.02e-3; AE = 0.1e-3
D31 = 175e-12; H_PA = 0.7e-3
E_PE = 63e9; NU_PE = 0.35
N1, N2 = 30, 24
N_MODES = 3
ZETA = np.array([0.0031, 0.0017, 0.0027])
DT = 5e-5
F_TOOTH = NT * RPM / 60.0

# removal scenarios (band height a_p_rem at the top, radial depth a_e_rem)
FINISH = dict(a_p=0.3e-3, a_e=0.1e-3, label="finishing (a_p=0.3, a_e=0.1 mm)")
ROUGH  = dict(a_p=0.030,  a_e=0.4e-3, label="roughing (top 30 mm band, a_e=0.4 mm)")

OUT = os.path.join(_HERE, "figs_removal")
os.makedirs(OUT, exist_ok=True)


def fresh_plate(n_modes=N_MODES):
    p = InProcessPlate(LP, HP, BP, RHO, E_AL, NU_AL, N1=N1, N2=N2,
                       n_modes=n_modes, zeta_modes=list(ZETA), verbose=False)
    return p


# ============================================================ (1) freq sweep
def freq_vs_position(scn, n_steps=41):
    """Sweep the tool 0 -> L_P, machining one pass; return X positions and the
    instantaneous natural frequencies (n_steps x N_MODES)."""
    p = fresh_plate()
    p.new_pass()
    xs = np.linspace(0, LP, n_steps)
    F = np.zeros((n_steps, N_MODES))
    for i, x in enumerate(xs):
        p.machine_to(x, a_p=scn["a_p"], a_e=scn["a_e"])
        p.rebuild()
        f, _ = p.instantaneous_modes(N_MODES)
        F[i] = f
    return xs, F, p


print("Computing in-process frequency evolution ...")
x_fin, F_fin, _ = freq_vs_position(FINISH)
x_rou, F_rou, plate_rou = freq_vs_position(ROUGH)
f0 = F_rou[0]
print(f"  intact frequencies (Hz): {np.round(f0,1)}")
print(f"  after finishing pass   : {np.round(F_fin[-1],1)}  "
      f"(d%={np.round((F_fin[-1]-f0)/f0*100,3)})")
print(f"  after roughing pass    : {np.round(F_rou[-1],1)}  "
      f"(d%={np.round((F_rou[-1]-f0)/f0*100,2)})")


# ============================================================ (2) FRF at stages
#  Receptance FRF (force at cutter -> displacement at sensor) at three stages of
#  the roughing pass. Uses the exact instantaneous modes at each stage, so the
#  resonance shift caused by material removal is shown directly.
NFRF = 6
ZETA_FRF = np.concatenate([ZETA, np.full(NFRF - N_MODES, 0.003)])
X_FORCE, Z_FORCE = LP * 0.5, HP - 0.05e-3      # cutter (excitation) point
X_OBS,  Z_OBS = LP,        HP                   # displacement sensor point


def frf_stages(scn, stages, freqs_hz):
    p = fresh_plate(); p.new_pass()
    DOFf = p.DOFf
    Nf = shape_at_point_M(X_FORCE, Z_FORCE, p.N1, p.N2, p.lex, p.ley,
                          p.node_id, p.ndof)[DOFf]
    No = shape_at_point_M(X_OBS, Z_OBS, p.N1, p.N2, p.lex, p.ley,
                          p.node_id, p.ndof)[DOFf]
    w = 2*np.pi*freqs_hz
    out = []
    for xt in stages:
        p.machine_to(xt, a_p=scn["a_p"], a_e=scn["a_e"]); p.rebuild()
        fhz, V = p.instantaneous_modes(NFRF)
        om = 2*np.pi*fhz
        Dp = Nf @ V; Do = No @ V
        H = np.zeros_like(w, dtype=complex)
        for m in range(NFRF):
            H += (Do[m]*Dp[m]) / (om[m]**2 - w**2 + 2j*ZETA_FRF[m]*om[m]*w)
        out.append((xt, fhz, np.abs(H)))
    return out


print("Computing receptance FRF at 3 stages of the roughing pass ...")
freqs = np.linspace(50, 3000, 6000)
stages = [0.0, LP/2, LP]
FRFs = frf_stages(ROUGH, stages, freqs)
for xt, fhz, _ in FRFs:
    print(f"  X_tool={xt*1e3:5.1f} mm  f(modes)= {np.round(fhz[:3],1)} Hz")


# ============================================================ plot
plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
cols = ['#1f4e79', '#c0392b', '#2e8b57']

# (a) frequencies vs tool position
a = ax[0, 0]
for m in range(N_MODES):
    a.plot(x_rou*1e3, F_rou[:, m], color=cols[m], lw=2,
           label=f"mode {m+1} (roughing)")
    a.plot(x_fin*1e3, F_fin[:, m], color=cols[m], lw=1.2, ls=':',
           label=f"mode {m+1} (finishing)" if m == 0 else None)
a.set_title("(a) In-process natural frequencies vs tool feed position",
            fontweight='bold')
a.set_xlabel("tool position $X_{tool}$ (mm)"); a.set_ylabel("frequency (Hz)")
a.grid(alpha=0.4); a.legend(fontsize=8, ncol=2)

# (b) thickness field after roughing
a = ax[0, 1]
H = plate_rou.h_elem.T * 1e3        # (N2 x N1) so Z on y-axis
im = a.imshow(H, origin='lower', aspect='auto', cmap='viridis',
              extent=[0, LP*1e3, 0, HP*1e3], vmin=H.min(), vmax=BP*1e3)
a.set_title("(b) Remaining thickness $h(X_P,Z_P)$ after roughing pass",
            fontweight='bold')
a.set_xlabel("$X_P$ (mm)"); a.set_ylabel("$Z_P$ (mm)")
cb = fig.colorbar(im, ax=a); cb.set_label("thickness (mm)")

# (c) receptance FRF (full range) at 3 stages
a = ax[1, 0]
stage_col = ['#1f4e79', '#e08a1e', '#c0392b']
stage_lbl = ['start (intact)', 'mid pass (50%)', 'end (100%)']
for (xt, fhz, H), c, lb in zip(FRFs, stage_col, stage_lbl):
    a.semilogy(freqs, H*1e6, color=c, lw=1.4, label=lb)
a.set_xlim(50, 3000)
a.set_title("(c) Receptance FRF $|y/F|$ — 3 stages of the roughing pass",
            fontweight='bold')
a.set_xlabel("frequency (Hz)"); a.set_ylabel("|FRF| (µm/N, log)")
a.grid(alpha=0.4, which='both'); a.legend(fontsize=9, loc='upper right')

# (d) FRF zoom on mode 1 — resonance shift
a = ax[1, 1]
for (xt, fhz, H), c, lb in zip(FRFs, stage_col, stage_lbl):
    a.plot(freqs, H*1e6, color=c, lw=1.6,
           label=f"{lb}: $f_1$={fhz[0]:.0f} Hz")
a.axvline(f0[0], color='#1f4e79', ls=':', lw=1, alpha=0.6)
a.axvline(F_rou[-1, 0], color='#c0392b', ls=':', lw=1, alpha=0.6)
a.set_xlim(460, 620)
a.set_title("(d) Mode-1 resonance shift as material is removed",
            fontweight='bold')
a.set_xlabel("frequency (Hz)"); a.set_ylabel("|FRF| (µm/N)")
a.grid(alpha=0.4); a.legend(fontsize=9, loc='upper right')

df1 = (F_rou[-1, 0]-f0[0])/f0[0]*100
fig.suptitle("Mindlin plate — in-process material removal: workpiece dynamics "
             "evolve as the cutter feeds along its path\n"
             f"roughing pass: mode 1 {f0[0]:.0f} → {F_rou[-1,0]:.0f} Hz "
             f"({df1:+.1f}%), mean thickness {plate_rou.h_elem.mean()*1e3:.2f} mm",
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
png = os.path.join(OUT, "inprocess_dynamics.png")
fig.savefig(png, dpi=150, bbox_inches='tight')
fig.savefig(png.replace(".png", ".pdf"), bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {png}")
print(f"Mode-1 drift over roughing pass: {df1:+.2f}%")
