#!/usr/bin/env python3
"""
Schematic of the active-control milling setup used in this simulation
(matches paper Fig. 2, with THIS project's non-collocated geometry):
cantilever plate, piezo patch actuator (right-lower), displacement sensor
(right-upper), milling tool and its path along the free upper edge.

Writes results/fig_setup.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow, Circle, FancyBboxPatch, FancyArrowPatch

from src import config as C

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

lp = C.PLATE["lp"] * 1e3      # 100 mm
hp = C.PLATE["hp"] * 1e3      # 80 mm
Rtool = C.TOOL["radius"] * 1e3  # 5 mm

# our simulation's normalised locations -> mm
sx, sz = C.SENSOR_POS[0] * lp, C.SENSOR_POS[1] * hp     # sensor (right-upper)
ax_, az = C.ACT_POS[0] * lp, C.ACT_POS[1] * hp          # actuator centroid (right-lower)
mill_x = 0.55 * lp                                       # tool drawn mid-path (clarity)

fig, ax = plt.subplots(figsize=(12.5, 7.2))

# ---------------- the plate (front view) ----------------
ax.add_patch(Rectangle((0, 0), lp, hp, facecolor="#cfe8f3",
                        edgecolor="#2b6b86", lw=2, zorder=1))

# clamped base (hatched foundation) at the bottom edge
base_h = 9
ax.add_patch(Rectangle((-8, -base_h), lp + 16, base_h, facecolor="#e6e6e6",
                        edgecolor="k", lw=1.5, hatch="////", zorder=1))
ax.text(lp / 2, -base_h - 4, "clamped base (cantilever)", ha="center",
        va="top", fontsize=9, style="italic")

# coordinate axes at O_P (lower-left)
ax.add_patch(FancyArrow(0, 0, 22, 0, width=0.3, head_width=2.5, head_length=3,
                        color="k", zorder=6))
ax.add_patch(FancyArrow(0, 0, 0, 20, width=0.3, head_width=2.5, head_length=3,
                        color="k", zorder=6))
ax.text(25, 2.5, r"$x_P$", fontsize=12)
ax.text(-7, 21, r"$z_P$", fontsize=12)
ax.text(-9, -7, r"$O_P$", fontsize=11)

# ---------------- piezo actuator patch (right-lower) ----------------
pw, ph = 22.0, 26.0
ax.add_patch(Rectangle((ax_ - pw / 2, az - ph / 2), pw, ph, facecolor="#f5a15a",
                        edgecolor="#b5651d", lw=1.8, zorder=3))
ax.text(ax_, az, "piezo\npatch", ha="center", va="center", fontsize=8.5,
        weight="bold", color="#5a3410", zorder=4)
_side = "left" if C.ACT_POS[0] < 0.5 else "right"
ax.annotate(f"Piezoelectric actuator\n({_side}-lower corner, paper Fig. 2)",
            xy=(ax_ + pw / 2, az), xytext=(lp + 22, az - 6),
            fontsize=9.5, va="center",
            arrowprops=dict(arrowstyle="->", color="#b5651d", lw=1.4))

# ---------------- displacement sensor (right-upper) ----------------
# small probe body offset from the plate, pointing at the upper-right corner
ax.add_patch(Rectangle((sx + 10, sz - 4), 16, 8, facecolor="#f0873a",
                        edgecolor="#a04a12", lw=1.5, zorder=5))
ax.add_patch(FancyArrow(sx + 10, sz, -9, 0, width=0.2, head_width=2.2,
                        head_length=2.5, color="#a04a12", zorder=5))
ax.plot([sx], [sz], "o", color="#a04a12", ms=6, zorder=6)
ax.annotate("Displacement sensor\n(right-upper corner)",
            xy=(sx + 26, sz), xytext=(lp + 30, hp + 20),
            fontsize=9.5, va="center", ha="left",
            arrowprops=dict(arrowstyle="->", color="#a04a12", lw=1.4))

# ---------------- milling tool + its path along the free upper edge -------
# path (dashed) along the top edge
ax.plot([0, lp], [hp, hp], color="#1a7a1a", lw=1.2, ls="--", zorder=2)
for xg, alpha, lbl in [(2, 0.30, "start"), (lp - 2, 0.30, "end")]:
    ax.add_patch(Circle((xg, hp), Rtool, facecolor="#9ccc9c", edgecolor="#1a7a1a",
                        lw=1.2, alpha=alpha, zorder=2))
# current tool
ax.add_patch(Rectangle((mill_x - 3.2, hp + 1), 6.4, 26, facecolor="#8fbf6a",
                        edgecolor="#3c6b1f", lw=1.4, zorder=7))   # shank
ax.add_patch(Circle((mill_x, hp), Rtool, facecolor="#b7e0a0",
                    edgecolor="#3c6b1f", lw=1.6, zorder=7))       # cutter (Ø10 mm)
ax.text(mill_x, hp + 30, "milling tool\n(Ø10 mm, 3 teeth)", ha="center",
        va="bottom", fontsize=9)
# feed-direction arrow along the path
ax.add_patch(FancyArrow(lp * 0.12, hp + 7, 22, 0, width=0.4, head_width=3.2,
                        head_length=4, color="#1a7a1a", zorder=6))
ax.text(lp * 0.30, hp + 11,
        f"tool path along free edge: 0 → {lp:.0f} mm\n"
        f"(feed {C.FEED_VELOCITY*1e3:.1f} mm/s, pass {C.T_PASS:.1f} s)",
        ha="center", va="bottom", fontsize=9, color="#1a5a1a")

# regenerative-force + control annotations
ax.annotate("regenerative cutting force\n(chatter source, delay τ)",
            xy=(mill_x, hp - 3), xytext=(lp * 0.30, hp * 0.62),
            fontsize=8.5, color="#8a1f1f", ha="center",
            arrowprops=dict(arrowstyle="->", color="#8a1f1f", lw=1.2))

# ---------------- control-loop block diagram (right) ----------------
def block(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.5",
                                facecolor=fc, edgecolor="k", lw=1.3, zorder=8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=9, zorder=9)

bx = lp + 60
block(bx, hp - 6, 46, 16, "Displacement\nsensor", "#ffd9b3")
block(bx, hp / 2 - 8, 46, 16, "Control\nsystem", "#f3d1f0")
block(bx, 6, 46, 16, "Piezoelectric\nactuator", "#ffd9b3")
arr = dict(arrowstyle="->", lw=1.6, color="k")
ax.add_patch(FancyArrowPatch((bx + 23, hp - 6), (bx + 23, hp / 2 + 8), **arr))
ax.add_patch(FancyArrowPatch((bx + 23, hp / 2 - 8), (bx + 23, 22), **arr))
# feedback from plate to sensor block, and actuator block back to plate
ax.add_patch(FancyArrowPatch((sx + 26, sz), (bx, hp + 2), connectionstyle="arc3,rad=0.15", **arr))
ax.add_patch(FancyArrowPatch((bx, 14), (ax_ + pw / 2, az - 4), connectionstyle="arc3,rad=0.15", **arr))

ax.set_xlim(-16, lp + 116)
ax.set_ylim(-base_h - 16, hp + 42)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Active-vibration-control milling setup — thin-walled cantilever plate "
             f"({lp:.0f}×{hp:.0f}×{C.PLATE['bp']*1e3:.0f} mm), paper geometry "
             "(sensor right-upper, piezo left-lower)", fontsize=11)
fig.tight_layout()
out = os.path.join(RESULTS, "fig_setup.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print("wrote", out)
