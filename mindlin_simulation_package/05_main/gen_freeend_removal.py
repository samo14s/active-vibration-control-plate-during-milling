"""
gen_freeend_removal.py
=====================
Single pass at the FREE END: the cutter removes the top a_p = 1 mm of the wall
along its path (X_P: 0 -> L_P). Uses a mesh refined so the 1 mm removed band is
exactly one element row (ley = 1 mm), and models the pass as machining the top
row away in each column the tool has passed. Removing material at the free tip
of a cantilever removes more inertia than stiffness there, so the fundamental
frequency RISES.

The removed-top-row model is validated against the exact H_P = 79 mm plate
(agree to 0.01 %).

Output: figs_freeend/freeend_ap1mm.png (+ .pdf)
Run:  python gen_freeend_removal.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "01_core"))

from plate_model import PlateModel
from inprocess_plate import InProcessPlate
from mindlin_q8 import shape_at_point_M

# ------------------------------------------------------------------ params
LP, HP, BP = 0.100, 0.080, 0.004
RHO, E_AL, NU = 2830, 69e9, 0.33
ZETA = np.array([0.0031, 0.0017, 0.0027])
A_P = 1.0e-3                     # axial depth removed at the free end (1 mm)
N1, N2 = 30, 80                 # ley = HP/N2 = 1 mm  -> top row == removed band
LEY = HP / N2
NFRF = 6
ZETA_FRF = np.concatenate([ZETA, np.full(NFRF - 3, 0.003)])
X_FORCE, Z_FORCE = LP * 0.5, HP - 2e-3     # excitation / sensor kept below the cut
X_OBS,  Z_OBS = LP,        HP - 2e-3

OUT = os.path.join(_HERE, "figs_freeend")
os.makedirs(OUT, exist_ok=True)


def fresh(n=3):
    zeta = np.concatenate([ZETA, np.full(max(0, n - 3), 0.003)])[:n]
    return InProcessPlate(LP, HP, BP, RHO, E_AL, NU, N1=N1, N2=N2,
                          n_modes=n, zeta_modes=list(zeta), verbose=False)


# -- trim removal: machine the top a_p band to ~0 (material removed) -----
A_E_TRIM = BP            # radial depth >= thickness  -> full removal of the band
H_FLOOR = 1e-3          # floor fraction -> ~4 microns residual (numerically ~ removed)


# ============================================================ (1) freq sweep
print("Sweeping the single free-end pass (a_p = 1 mm) ...")
p = fresh(); f0 = p.freq_n.copy()
p.new_pass()
xs = np.linspace(0, LP, 21)
F = np.zeros((len(xs), 3))
for i, x in enumerate(xs):
    p.machine_to(x, a_p=A_P, a_e=A_E_TRIM, h_min_frac=H_FLOOR)
    p.rebuild()
    F[i], _ = p.instantaneous_modes(3)
plate_final = p
f_end = F[-1]
print(f"  intact : {np.round(f0,2)} Hz")
print(f"  end    : {np.round(f_end,2)} Hz   (mode-1 {f0[0]:.1f} -> {f_end[0]:.1f} Hz, "
      f"{(f_end[0]-f0[0])/f0[0]*100:+.2f}%)")

# validation vs exact shortened plate
p_exact = PlateModel(LP, HP - A_P, BP, RHO, E_AL, NU, N1=N1, N2=N2,
                     n_modes=3, zeta_modes=list(ZETA), verbose=False)
print(f"  exact H_P=79mm plate: {np.round(p_exact.freq_n,2)} Hz  "
      f"(validation, err f1 = {(f_end[0]-p_exact.freq_n[0])/p_exact.freq_n[0]*100:+.3f}%)")


# ============================================================ (2) FRF
def frf(plate, freqs_hz):
    DOFf = plate.DOFf
    Nf = shape_at_point_M(X_FORCE, Z_FORCE, plate.N1, plate.N2,
                          plate.lex, plate.ley, plate.node_id, plate.ndof)[DOFf]
    No = shape_at_point_M(X_OBS, Z_OBS, plate.N1, plate.N2,
                          plate.lex, plate.ley, plate.node_id, plate.ndof)[DOFf]
    fhz, V = plate.instantaneous_modes(NFRF)
    om = 2*np.pi*fhz; w = 2*np.pi*freqs_hz
    Dp = Nf @ V; Do = No @ V
    H = np.zeros_like(w, dtype=complex)
    for m in range(NFRF):
        H += (Do[m]*Dp[m]) / (om[m]**2 - w**2 + 2j*ZETA_FRF[m]*om[m]*w)
    return np.abs(H)


freqs = np.linspace(400, 700, 4000)
p_int = fresh(NFRF)
H_int = frf(p_int, freqs)
H_end = frf(plate_final, freqs)


# ============================================================ plot
plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))

# (a) frequency change (%) vs tool position — normalised so all modes show
a = ax[0]
cols = ['#1f4e79', '#c0392b', '#2e8b57']
for m in range(3):
    a.plot(xs*1e3, (F[:, m]/f0[m]-1)*100, color=cols[m], lw=2, marker='o', ms=3,
           label=f"mode {m+1}: {f0[m]:.0f}→{f_end[m]:.0f} Hz")
a.set_title("(a) Frequency shift vs tool position\n"
            "single free-end pass, $a_p$ = 1 mm", fontweight='bold')
a.set_xlabel("tool position $X_{tool}$ (mm)")
a.set_ylabel("$\\Delta f / f_0$ (%)")
a.grid(alpha=0.4); a.legend(fontsize=9, loc='upper left')

# (b) mode-1 FRF intact vs after pass
a = ax[1]
a.plot(freqs, H_int*1e6, color='#1f4e79', lw=1.8, label=f"intact: $f_1$={f0[0]:.1f} Hz")
a.plot(freqs, H_end*1e6, color='#c0392b', lw=1.8, label=f"after pass: $f_1$={f_end[0]:.1f} Hz")
a.axvline(f0[0], color='#1f4e79', ls=':', lw=1, alpha=0.6)
a.axvline(f_end[0], color='#c0392b', ls=':', lw=1, alpha=0.6)
a.annotate("", xy=(f_end[0], H_end.max()*1e6*0.5), xytext=(f0[0], H_end.max()*1e6*0.5),
           arrowprops=dict(arrowstyle='->', color='k', lw=1.3))
a.text((f0[0]+f_end[0])/2, H_end.max()*1e6*0.55,
       f"+{(f_end[0]-f0[0])/f0[0]*100:.1f}%", ha='center', fontsize=10, fontweight='bold')
a.set_xlim(480, 560)
a.set_title("(b) Mode-1 resonance shift\n(free-end material removal)", fontweight='bold')
a.set_xlabel("frequency (Hz)"); a.set_ylabel("|FRF| $|y/F|$ (µm/N)")
a.grid(alpha=0.4); a.legend(fontsize=9, loc='upper right')

# (c) thickness field — zoom on the top 10 mm so the removed 1 mm is visible
a = ax[2]
H2 = plate_final.h_elem.T * 1e3
im = a.imshow(H2, origin='lower', aspect='auto', cmap='viridis',
              extent=[0, LP*1e3, 0, HP*1e3], vmin=0, vmax=BP*1e3)
a.axhline((HP - A_P)*1e3, color='r', ls='--', lw=1.5)
a.text(50, (HP - A_P)*1e3 - 1.4, "top 1 mm removed (free edge)", color='r',
       fontsize=9, ha='center', va='top', fontweight='bold')
a.set_ylim(70, 80)                     # zoom near the free end
a.set_title("(c) Remaining thickness $h(X_P,Z_P)$\n"
            "(zoom on free end, top 10 mm)", fontweight='bold')
a.set_xlabel("$X_P$ (mm)"); a.set_ylabel("$Z_P$ (mm)")
cb = fig.colorbar(im, ax=a); cb.set_label("thickness (mm)")

df1 = (f_end[0]-f0[0])/f0[0]*100
fig.suptitle("Free-end single pass, $a_p$ = 1 mm — cutter removes the top 1 mm along its path: "
             f"mode 1 {f0[0]:.1f} → {f_end[0]:.1f} Hz ({df1:+.2f} %), "
             f"validated vs exact 79 mm plate",
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.93])
png = os.path.join(OUT, "freeend_ap1mm.png")
fig.savefig(png, dpi=150, bbox_inches='tight')
fig.savefig(png.replace(".png", ".pdf"), bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {png}")
