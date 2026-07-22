#!/usr/bin/env python3
"""
Visualisation of the ACTUAL model implemented in the code: the cantilever-plate
mode shapes computed by config.plate_mode_shape(), with the exact sensor,
actuator and milling-point locations the simulation samples them at.

Everything here is read/derived straight from src/config.py, so the picture is
the code's real model (not an artist's schematic).

Writes results/fig_model.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src import config as C

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

lp = C.PLATE["lp"] * 1e3
hp = C.PLATE["hp"] * 1e3

# grid over the plate, evaluate the code's mode shapes
nx, nz = 160, 130
xg = np.linspace(0, 1, nx)
zg = np.linspace(0, 1, nz)
X, Z = np.meshgrid(xg, zg)
# shape[k] over the grid
PHI = np.zeros((C.N_MODES, nz, nx))
for j in range(nz):
    for i in range(nx):
        PHI[:, j, i] = C.plate_mode_shape(xg[i], zg[j])

# exact locations (normalised) used by the code
sx, sz = C.SENSOR_POS
axp, azp = C.ACT_POS
freqs = C.MODE_FREQ_HZ

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

def draw_locations(ax):
    # sensor (right-upper)
    ax.plot(sx * lp, sz * hp, "*", ms=18, color="k", mec="w", mew=1.2, zorder=6)
    # actuator patch (right-lower)
    pw, ph = 0.20 * lp, 0.26 * hp
    ax.add_patch(Rectangle((axp * lp - pw / 2, azp * hp - ph / 2), pw, ph,
                           fill=False, edgecolor="k", lw=2.0, zorder=6))
    # milling path (free upper edge)
    ax.plot([0, lp], [hp, hp], "w--", lw=2.0, zorder=5)

for k in range(3):
    ax = axes.ravel()[k]
    m = PHI[k]
    vmax = np.max(np.abs(m))
    im = ax.pcolormesh(X * lp, Z * hp, m, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       shading="auto")
    draw_locations(ax)
    # clamped base hatch
    ax.add_patch(Rectangle((0, -6), lp, 6, facecolor="none", edgecolor="k",
                           hatch="////", lw=1.2))
    ax.set_title(f"Mode {k+1}  ({freqs[k]:.0f} Hz)  —  "
                 f"{'1st bending' if k==0 else '1st torsion' if k==1 else '2nd bending'}",
                 fontsize=10)
    ax.set_xlabel("x  (mm, along free edge)")
    ax.set_ylabel("z  (mm, clamp→free)")
    ax.set_xlim(-4, lp + 4); ax.set_ylim(-8, hp + 4)
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="φ (norm.)")

# 4th panel: how the milling-point mode shape varies ALONG the path (the code's
# phi_mill(xi)), plus the fixed sensor / actuator vectors
ax = axes.ravel()[3]
xi = np.linspace(0, 1, 200)
pm = np.array([C.phi_mill(x) for x in xi])          # scaled, as used in the plant
colors = ["#1a73e8", "#e8710a", "#188038"]
for k in range(3):
    ax.plot(xi * lp, pm[:, k], color=colors[k], lw=1.8, label=f"mode {k+1}")
ax.axhline(0, color="gray", lw=0.7)
ax.axvline(C.MILL_POS_STATIC * lp, color="k", ls=":", lw=1.2)
ax.text(C.MILL_POS_STATIC * lp, ax.get_ylim()[1] * 0.9,
        " snapshot\n position", fontsize=8, va="top")
ax.set_title("Milling-point mode shape Φ_mill(x) sampled ALONG the path\n"
             "(this is the code's varying dynamics)", fontsize=10)
ax.set_xlabel("tool position x  (mm)")
ax.set_ylabel("Φ_mill  (1/√kg)")
ax.legend(fontsize=8, ncol=3)
ax.grid(alpha=0.3)

txt = (f"locations sampled by the code (config.py):\n"
       f"  sensor  (x,z)=({sx:.2f},{sz:.2f})  →  Φ_sensor = {np.round(C.PHI_SENSOR,2)}\n"
       f"  piezo   (x,z)=({axp:.2f},{azp:.2f})  →  Φ_act    = {np.round(C.PHI_ACT,2)}\n"
       f"  milling  z=1, x moves 0→{lp:.0f} mm    →  Φ_mill(x) (curve at left)")
fig.text(0.5, 0.005, txt, ha="center", va="bottom", fontsize=9, family="monospace",
         bbox=dict(boxstyle="round", facecolor="#f2f2f2", edgecolor="#999"))

fig.suptitle("The model implemented in the code — cantilever-plate mode shapes "
             "and the non-collocated sampling points\n"
             "(★ sensor · ▭ piezo patch · ┄ milling path)", fontsize=12)
fig.tight_layout(rect=[0, 0.06, 1, 0.96])
out = os.path.join(RESULTS, "fig_model.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print("wrote", out)
