"""
gen_control_strategy_diagram.py
=================================
Block diagram PALF-LQG — VERSION AMÉLIORÉE avec flèches BIEN VISIBLES.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (FancyBboxPatch, Rectangle, FancyArrowPatch,
                                  Circle, Polygon, Ellipse)
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.linewidth'] = 1.2

# Colors
COLOR_PLANT = '#A8D8E8'
COLOR_PLANT_EDGE = '#1F5582'
COLOR_LQG = '#90EE90'
COLOR_LQG_EDGE = '#2E8B57'
COLOR_NN = '#FFB6C1'
COLOR_NN_EDGE = '#C71585'
COLOR_OBSERVER = '#FFD700'
COLOR_OBSERVER_EDGE = '#B8860B'
COLOR_SAFETY = '#FFA07A'
COLOR_SAFETY_EDGE = '#CD5C5C'
COLOR_SUM = '#E6E6FA'
COLOR_SUM_EDGE = '#4B0082'
COLOR_FF_PATH = '#DC143C'
COLOR_FB_PATH = '#2E8B57'
COLOR_LEARN = '#9370DB'
COLOR_OBSERVER_PATH = '#B8860B'

ARROW_LW_MAIN = 3.0
ARROW_LW_SECONDARY = 2.2
ARROW_MUT_SCALE = 28

OUT_DIR = "figs_article_publication"
os.makedirs(OUT_DIR, exist_ok=True)


def save_fig(fig, name):
    fig.savefig(f"{OUT_DIR}/{name}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{OUT_DIR}/{name}.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {name}.png + .pdf")


def make_box(ax, x, y, w, h, label, color_face, color_edge,
              fontsize=11, fontweight='bold', linewidth=2.0,
              boxstyle='round,pad=0.1', sublabel=None):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=boxstyle,
                          facecolor=color_face, edgecolor=color_edge,
                          linewidth=linewidth, zorder=5)
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + 0.18*h, label, ha='center', va='center',
                 fontsize=fontsize, fontweight=fontweight, zorder=6)
        ax.text(x, y - 0.22*h, sublabel, ha='center', va='center',
                 fontsize=fontsize-2, style='italic', zorder=6)
    else:
        ax.text(x, y, label, ha='center', va='center',
                 fontsize=fontsize, fontweight=fontweight, zorder=6)


def big_arrow(ax, x1, y1, x2, y2, color='black', lw=ARROW_LW_MAIN,
               connectionstyle='arc3,rad=0', zorder=10):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle='-|>', color=color,
                              lw=lw, mutation_scale=ARROW_MUT_SCALE,
                              connectionstyle=connectionstyle, zorder=zorder)
    ax.add_patch(arrow)


def signal_label(ax, x, y, text, color='black', fontsize=12, bg='white'):
    ax.text(x, y, text, ha='center', va='center',
             fontsize=fontsize, fontweight='bold', color=color,
             bbox=dict(boxstyle='round,pad=0.35', facecolor=bg,
                        edgecolor=color, linewidth=1.5, alpha=0.95),
             zorder=15)


def make_summing_junction(ax, x, y, r=0.3, signs=None):
    circle = Circle((x, y), r, facecolor=COLOR_SUM,
                     edgecolor=COLOR_SUM_EDGE, linewidth=2.5, zorder=8)
    ax.add_patch(circle)
    ax.plot([x-r*0.5, x+r*0.5], [y, y], color=COLOR_SUM_EDGE,
             linewidth=2.0, zorder=9)
    ax.plot([x, x], [y-r*0.5, y+r*0.5], color=COLOR_SUM_EDGE,
             linewidth=2.0, zorder=9)
    if signs:
        for pos, sign in signs.items():
            if pos == 'top':
                ax.text(x, y+r*1.6, sign, ha='center', va='center',
                         fontsize=18, fontweight='bold', color='red',
                         zorder=10)
            elif pos == 'left':
                ax.text(x-r*1.6, y, sign, ha='center', va='center',
                         fontsize=18, fontweight='bold', color='red',
                         zorder=10)


# ============================================================
# FIGURE A : Main architecture
# ============================================================
print("[Figure A] Control architecture (clear arrows) ...")

fig, ax = plt.subplots(figsize=(20, 11))

y_top = 8.0
y_mid = 5.0
y_obs = 2.5
y_ilc = 0.3

x_phase = 1.0
x_nn = 4.5
x_alpha = 7.0
x_sum = 9.0
x_lqg = 4.5
x_safety = 11.5
x_plant = 14.5
x_output = 17.5
x_obs = 14.5

# Phase
make_box(ax, x_phase, y_top, 1.6, 1.1, r'$\phi(t)$',
          COLOR_NN, COLOR_NN_EDGE, sublabel='Tool phase', fontsize=14)

# NN
make_box(ax, x_nn, y_top, 2.8, 1.7, 'Phase-aware NN',
          COLOR_NN, COLOR_NN_EDGE,
          sublabel=r'$NN_{FF}(\phi, \hat{x})$', fontsize=13)

# NN icon
n_layer = 4
n_nodes = [3, 5, 5, 1]
nn_x_start = x_nn - 0.9; nn_x_end = x_nn + 0.9
nn_y_start = y_top - 0.45; nn_y_end = y_top - 0.15
for i, n in enumerate(n_nodes):
    xn = nn_x_start + i * (nn_x_end - nn_x_start) / (n_layer - 1)
    for j in range(n):
        yn = (nn_y_start + j * (nn_y_end - nn_y_start) / (n - 1)
              if n > 1 else (nn_y_start + nn_y_end) / 2)
        ax.plot(xn, yn, 'o', color='purple', markersize=4, zorder=7)
        if i < n_layer - 1:
            xn_next = nn_x_start + (i+1) * (nn_x_end - nn_x_start) / (n_layer - 1)
            n_next = n_nodes[i+1]
            for k in range(n_next):
                yn_next = (nn_y_start + k * (nn_y_end - nn_y_start) / (n_next - 1)
                           if n_next > 1 else (nn_y_start + nn_y_end) / 2)
                ax.plot([xn, xn_next], [yn, yn_next],
                         color='purple', linewidth=0.4, alpha=0.4, zorder=6)

# Alpha
make_box(ax, x_alpha, y_top, 1.0, 1.0, r'$\alpha$',
          COLOR_NN, COLOR_NN_EDGE, sublabel='Gain', fontsize=18)

# Sum
make_summing_junction(ax, x_sum, y_mid, r=0.35, signs={'top': '+', 'left': '+'})

# Safety
make_box(ax, x_safety, y_mid, 2.4, 1.5, 'CLF Voltage\nGovernor',
          COLOR_SAFETY, COLOR_SAFETY_EDGE,
          sublabel=r'$V(x) \leq V_{max}$', fontsize=12)

# Plant
make_box(ax, x_plant, y_mid, 2.6, 1.7, 'Plant',
          COLOR_PLANT, COLOR_PLANT_EDGE,
          sublabel='Plate + Piezo + Tool', fontsize=13)

# Plant icon
plate_w = 0.7; plate_h = 0.35
ax.add_patch(Rectangle((x_plant - plate_w/2, y_mid - 0.55),
                         plate_w, plate_h,
                         facecolor='#7FB8D4', edgecolor=COLOR_PLANT_EDGE,
                         linewidth=1.2, zorder=7))
ax.add_patch(Rectangle((x_plant - 0.06, y_mid - 0.25),
                         0.12, 0.22,
                         facecolor='#444', edgecolor='black',
                         linewidth=1.0, zorder=8))
for xh in np.linspace(x_plant - plate_w/2, x_plant + plate_w/2, 6):
    ax.plot([xh, xh - 0.04], [y_mid - 0.55, y_mid - 0.65],
             color='black', linewidth=0.8, zorder=7)

# Output
ax.plot(x_output, y_mid + 0.3, 'o', color=COLOR_PLANT_EDGE,
         markersize=10, zorder=10)

# Observer
make_box(ax, x_obs, y_obs, 2.6, 1.1, 'Kalman Observer',
          COLOR_OBSERVER, COLOR_OBSERVER_EDGE,
          sublabel=r'State estimate $\hat{x}(t)$', fontsize=12)

# LQG
make_box(ax, x_lqg, y_mid, 2.4, 1.5, 'LQG Controller',
          COLOR_LQG, COLOR_LQG_EDGE,
          sublabel=r'$u_{LQG} = -K\hat{x}$', fontsize=12)


# === ARROWS ===

# 1. φ → NN
big_arrow(ax, x_phase + 0.8, y_top, x_nn - 1.4, y_top,
            color=COLOR_NN_EDGE)

# 2. NN → α
big_arrow(ax, x_nn + 1.4, y_top, x_alpha - 0.5, y_top,
            color=COLOR_NN_EDGE)
signal_label(ax, (x_nn + x_alpha)/2 + 0.1, y_top + 0.5,
              r'$u_{FF}^{raw}$', color=COLOR_NN_EDGE, fontsize=12)

# 3. α → sum (orthogonal: down then horizontal then down into sum)
ax.plot([x_alpha, x_alpha], [y_top - 0.5, y_mid + 0.6],
         color=COLOR_FF_PATH, linewidth=ARROW_LW_MAIN,
         solid_capstyle='round', zorder=10)
ax.plot([x_alpha, x_sum], [y_mid + 0.6, y_mid + 0.6],
         color=COLOR_FF_PATH, linewidth=ARROW_LW_MAIN,
         solid_capstyle='round', zorder=10)
ax.annotate('', xy=(x_sum, y_mid + 0.37), xytext=(x_sum, y_mid + 0.6),
             arrowprops=dict(arrowstyle='-|>', color=COLOR_FF_PATH,
                              lw=ARROW_LW_MAIN, mutation_scale=ARROW_MUT_SCALE),
             zorder=11)
signal_label(ax, (x_alpha + x_sum)/2, y_mid + 0.95,
              r'$\alpha \cdot NN_{FF}$', color=COLOR_FF_PATH, fontsize=12)

# 4. LQG → sum
big_arrow(ax, x_lqg + 1.2, y_mid, x_sum - 0.37, y_mid,
            color=COLOR_FB_PATH)
signal_label(ax, (x_lqg + x_sum)/2 + 0.5, y_mid + 0.45,
              r'$u_{LQG}$', color=COLOR_FB_PATH, fontsize=12)

# 5. sum → safety
big_arrow(ax, x_sum + 0.37, y_mid, x_safety - 1.2, y_mid, color='black')
signal_label(ax, (x_sum + x_safety)/2 + 0.1, y_mid + 0.45,
              r'$u^{*}$', color='black', fontsize=13)

# 6. safety → plant
big_arrow(ax, x_safety + 1.2, y_mid, x_plant - 1.3, y_mid,
            color='black', lw=ARROW_LW_MAIN + 0.5)
signal_label(ax, (x_safety + x_plant)/2, y_mid + 0.5,
              r'$u(t)$', color='black', fontsize=13)

# 7. plant → output
big_arrow(ax, x_plant + 1.3, y_mid + 0.3, x_output - 0.15, y_mid + 0.3,
            color=COLOR_PLANT_EDGE, lw=ARROW_LW_MAIN + 0.5)
signal_label(ax, x_output + 0.4, y_mid + 0.3, r'$y(t)$',
              color=COLOR_PLANT_EDGE, fontsize=14)
ax.text(x_output + 0.4, y_mid - 0.05, '(displacement)',
         fontsize=10, style='italic', ha='center', color=COLOR_PLANT_EDGE)

# 8. plant → observer (down)
ax.plot([x_plant + 1.3, x_plant + 1.3], [y_mid - 0.1, y_obs + 0.55],
         color=COLOR_OBSERVER_PATH, linewidth=ARROW_LW_SECONDARY,
         solid_capstyle='round', zorder=9)
ax.annotate('', xy=(x_plant + 1.3, y_obs + 0.55),
             xytext=(x_plant + 1.3, y_obs + 0.75),
             arrowprops=dict(arrowstyle='-|>', color=COLOR_OBSERVER_PATH,
                              lw=ARROW_LW_SECONDARY, mutation_scale=ARROW_MUT_SCALE),
             zorder=10)
signal_label(ax, x_plant + 2.4, (y_mid + y_obs)/2 + 0.3,
              r'$y_{meas}$', color=COLOR_OBSERVER_PATH, fontsize=11)

# 9. Observer → LQG (orthogonal: long path)
ax.plot([x_obs - 1.3, x_lqg], [y_obs, y_obs],
         color=COLOR_OBSERVER_PATH, linewidth=ARROW_LW_SECONDARY,
         solid_capstyle='round', zorder=9)
ax.plot([x_lqg, x_lqg], [y_obs, y_mid - 0.65],
         color=COLOR_OBSERVER_PATH, linewidth=ARROW_LW_SECONDARY,
         solid_capstyle='round', zorder=9)
ax.annotate('', xy=(x_lqg, y_mid - 0.65), xytext=(x_lqg, y_mid - 0.85),
             arrowprops=dict(arrowstyle='-|>', color=COLOR_OBSERVER_PATH,
                              lw=ARROW_LW_SECONDARY, mutation_scale=ARROW_MUT_SCALE),
             zorder=10)
signal_label(ax, (x_obs + x_lqg)/2, y_obs + 0.4,
              r'$\hat{x}(t)$', color=COLOR_OBSERVER_PATH, fontsize=12)

# 10. Branch from x_hat path → NN (dotted)
branch_y = y_obs - 0.4
ax.plot(x_lqg, branch_y, 'o', color=COLOR_OBSERVER_PATH, markersize=8, zorder=11)
# Connect from main x_hat path to branch point first
ax.plot([x_lqg, x_lqg], [y_obs, branch_y], color=COLOR_OBSERVER_PATH,
         linewidth=1.5, linestyle=':', alpha=0.8, zorder=8)
# Then horizontal to NN x position
ax.plot([x_lqg, x_nn], [branch_y, branch_y], color=COLOR_OBSERVER_PATH,
         linewidth=1.5, linestyle=':', alpha=0.8, zorder=8)
# Up to NN box bottom
ax.plot([x_nn, x_nn], [branch_y, y_top - 0.9],
         color=COLOR_OBSERVER_PATH, linewidth=1.5, linestyle=':',
         alpha=0.8, zorder=8)
ax.annotate('', xy=(x_nn, y_top - 0.9), xytext=(x_nn, y_top - 1.05),
             arrowprops=dict(arrowstyle='-|>', color=COLOR_OBSERVER_PATH,
                              lw=1.5, mutation_scale=ARROW_MUT_SCALE-4),
             zorder=10)


# === ITERATIVE LEARNING LOOP ===
ilc_box = FancyBboxPatch((1.5, y_ilc - 0.5), 12.0, 1.0,
                           boxstyle='round,pad=0.15',
                           facecolor='#F0E6FA', edgecolor=COLOR_LEARN,
                           linewidth=2.5, linestyle='--', zorder=3)
ax.add_patch(ilc_box)
ax.text(7.5, y_ilc + 0.2, 'Iterative Learning Control (offline pre-training)',
         fontsize=13, fontweight='bold', ha='center',
         color=COLOR_LEARN, style='italic')
ax.text(7.5, y_ilc - 0.2,
         r'Update $NN_{FF}$ weights to minimize residual vibration',
         fontsize=10, ha='center', color=COLOR_LEARN, style='italic')

# ILC arrows
arrow_back = FancyArrowPatch(
    (x_output + 0.2, y_mid - 0.2), (12.5, y_ilc + 0.4),
    arrowstyle='-|>', color=COLOR_LEARN,
    lw=ARROW_LW_SECONDARY, mutation_scale=ARROW_MUT_SCALE,
    connectionstyle='arc3,rad=0.4', zorder=11)
ax.add_patch(arrow_back)
signal_label(ax, 16.5, 1.8, r'$y(t)$ residual',
              color=COLOR_LEARN, fontsize=11)

arrow_update = FancyArrowPatch(
    (3.0, y_ilc + 0.4), (x_nn - 1.4, y_top - 0.85),
    arrowstyle='-|>', color=COLOR_LEARN,
    lw=ARROW_LW_SECONDARY, mutation_scale=ARROW_MUT_SCALE,
    connectionstyle='arc3,rad=-0.35', zorder=11)
ax.add_patch(arrow_update)
signal_label(ax, 1.6, 4.5, 'Weight\nupdate',
              color=COLOR_LEARN, fontsize=11)


# === EQUATION + TITLE ===
eq_box = FancyBboxPatch((2.0, 9.5), 15.0, 1.0,
                          boxstyle='round,pad=0.15',
                          facecolor='#FFFACD', edgecolor='black',
                          linewidth=2.0, zorder=2)
ax.add_patch(eq_box)
ax.text(9.5, 9.95,
         r'$u(t) \;=\; u_{LQG}(\hat{x}) \;+\; \alpha \cdot NN_{FF}(\phi, \hat{x})$',
         fontsize=18, ha='center', va='center', fontweight='bold')
ax.text(9.5, 9.55,
         '(reactive baseline)            (anticipative feedforward)',
         fontsize=11, ha='center', va='center', style='italic', color='#555')

ax.text(9.5, 11.0, 'PALF-LQG Control Architecture',
         fontsize=20, ha='center', va='center', fontweight='bold')

# Legend
legend_y = -0.7
legend_items = [
    (1.5, COLOR_LQG, 'Feedback (LQG)'),
    (5.0, COLOR_NN, 'Feedforward (NN)'),
    (8.5, COLOR_OBSERVER, 'State estimation'),
    (12.0, COLOR_SAFETY, 'Safety filter'),
    (15.5, COLOR_PLANT, 'Plant'),
]
for x_l, c, lbl in legend_items:
    rect = Rectangle((x_l - 0.35, legend_y - 0.18), 0.7, 0.36,
                       facecolor=c, edgecolor='black', linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x_l + 0.5, legend_y, lbl, fontsize=10, va='center',
             fontweight='bold')

ax.set_xlim([-0.5, 19.0])
ax.set_ylim([-1.5, 11.7])
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
save_fig(fig, "fig15_control_architecture")


# ============================================================
# FIGURE B : Algorithm flowchart
# ============================================================
print("[Figure B] Algorithm flowchart (clear arrows) ...")

fig, ax = plt.subplots(figsize=(14, 12))

ax.text(7.0, 15.5, 'PALF-LQG Algorithm Flow',
         fontsize=18, ha='center', fontweight='bold')

# Initialization
make_box(ax, 7.0, 14.3, 9.0, 1.0,
          'INITIALIZATION', '#FFE4B5', '#D2691E',
          sublabel=r'Build LQG (optimal $w_q, w_qd, w_r$),  Init $NN_{FF}$ weights randomly',
          fontsize=13)

ax.annotate('', xy=(7.0, 13.4), xytext=(7.0, 13.8),
             arrowprops=dict(arrowstyle='-|>', color='black',
                              lw=ARROW_LW_MAIN, mutation_scale=ARROW_MUT_SCALE),
             zorder=10)

# ILC box
ilc_box = FancyBboxPatch((2.0, 11.0), 10.0, 2.0,
                           boxstyle='round,pad=0.15',
                           facecolor='#E6E6FA', edgecolor=COLOR_LEARN,
                           linewidth=2.5, zorder=3)
ax.add_patch(ilc_box)
ax.text(7.0, 12.8, 'OFFLINE — Iterative Learning Control',
         fontsize=14, ha='center', fontweight='bold', color=COLOR_LEARN)
ax.text(7.0, 12.4, 'For iter = 1 to N (N = 30):',
         fontsize=12, ha='center', fontweight='bold', color=COLOR_LEARN)
ax.text(2.5, 11.9, r'1. Simulate plant with current $NN_{FF}$',
         fontsize=11, ha='left', va='center')
ax.text(2.5, 11.55, r'2. Compute residual: $y_{res}(t) = y_{LQG}(t) - y_{PALF}(t)$',
         fontsize=11, ha='left', va='center')
ax.text(2.5, 11.2, r'3. Targets: $u_{target}(t) = -K_{corr} \cdot y_{res}(t)$',
         fontsize=11, ha='left', va='center')

ax.annotate('', xy=(7.0, 10.3), xytext=(7.0, 10.95),
             arrowprops=dict(arrowstyle='-|>', color='black',
                              lw=ARROW_LW_MAIN, mutation_scale=ARROW_MUT_SCALE),
             zorder=10)

# NN training
nn_train_y = 9.7
make_box(ax, 7.0, nn_train_y, 8.5, 1.0,
          'Update NN weights', COLOR_NN, COLOR_NN_EDGE,
          sublabel=r'Backprop: minimize $|NN_{FF}(\phi,\hat{x}) - u_{target}|^2$',
          fontsize=12)

# Loop arrow
arrow_loop = FancyArrowPatch(
    (11.5, nn_train_y), (12.0, 11.0),
    arrowstyle='-|>', color=COLOR_LEARN,
    lw=ARROW_LW_MAIN, mutation_scale=ARROW_MUT_SCALE+2,
    connectionstyle='arc3,rad=0.5', zorder=10)
ax.add_patch(arrow_loop)
signal_label(ax, 13.0, 10.5, 'iterate\n(N = 30)',
              color=COLOR_LEARN, fontsize=11)

# Down to online
ax.annotate('', xy=(7.0, 8.5), xytext=(7.0, 9.2),
             arrowprops=dict(arrowstyle='-|>', color='black',
                              lw=ARROW_LW_MAIN+0.5,
                              mutation_scale=ARROW_MUT_SCALE+2),
             zorder=10)
signal_label(ax, 9.5, 8.85, r'Trained $NN_{FF}$',
              color='black', fontsize=11)

ax.text(7.0, 8.15, '── ONLINE OPERATION ──',
         fontsize=13, ha='center', fontweight='bold',
         color='darkgreen', style='italic')

# Online loop box
loop_box = FancyBboxPatch((1.0, 0.8), 12.0, 7.0,
                            boxstyle='round,pad=0.15',
                            facecolor='#F0FFF0', edgecolor='darkgreen',
                            linewidth=2.5, linestyle='--', zorder=2)
ax.add_patch(loop_box)
ax.text(7.0, 7.5, 'For each time step k (real-time, dt = 50 µs):',
         fontsize=12, ha='center', fontweight='bold', color='darkgreen')

# Steps
make_box(ax, 4.0, 6.7, 4.0, 0.7, r'1. Measure $y(t_k)$',
          COLOR_PLANT, COLOR_PLANT_EDGE, fontsize=11)
make_box(ax, 10.0, 6.7, 4.0, 0.7, r'2. Update $\hat{x}(t_k)$ (Kalman)',
          COLOR_OBSERVER, COLOR_OBSERVER_EDGE, fontsize=11)
make_box(ax, 4.0, 5.3, 4.0, 0.7, r'3a. Compute $u_{LQG} = -K\hat{x}$',
          COLOR_LQG, COLOR_LQG_EDGE, fontsize=11)
make_box(ax, 10.0, 5.3, 4.0, 0.7, r'3b. Compute $u_{FF} = NN(\phi,\hat{x})$',
          COLOR_NN, COLOR_NN_EDGE, fontsize=11)
make_box(ax, 7.0, 3.9, 6.5, 0.7,
          r'4. Combine: $u^{*} = u_{LQG} + \alpha \cdot u_{FF}$',
          COLOR_SUM, COLOR_SUM_EDGE, fontsize=12)
make_box(ax, 7.0, 2.5, 7.0, 0.7,
          r'5. Safety: $u = $ CLF_governor$(u^{*}, \hat{x})$ (heuristic)',
          COLOR_SAFETY, COLOR_SAFETY_EDGE, fontsize=12)
make_box(ax, 7.0, 1.2, 7.5, 0.7,
          '6. Apply $u(t_k)$ to piezoelectric actuator',
          '#D3D3D3', '#696969', fontsize=12)

# Arrows between steps
# 1 → 2
ax.annotate('', xy=(8.0, 6.7), xytext=(6.0, 6.7),
             arrowprops=dict(arrowstyle='-|>', color='black',
                              lw=ARROW_LW_MAIN, mutation_scale=ARROW_MUT_SCALE),
             zorder=10)
# 2 → 3b (down)
ax.annotate('', xy=(10.0, 5.65), xytext=(10.0, 6.35),
             arrowprops=dict(arrowstyle='-|>', color='black',
                              lw=ARROW_LW_MAIN, mutation_scale=ARROW_MUT_SCALE),
             zorder=10)
# 2 → 3a (orthogonal)
ax.plot([10.0, 4.0], [6.0, 6.0], color='black',
         linewidth=ARROW_LW_SECONDARY, solid_capstyle='round', zorder=9)
ax.plot([4.0, 4.0], [6.0, 5.7], color='black',
         linewidth=ARROW_LW_SECONDARY, solid_capstyle='round', zorder=9)
ax.annotate('', xy=(4.0, 5.65), xytext=(4.0, 5.85),
             arrowprops=dict(arrowstyle='-|>', color='black',
                              lw=ARROW_LW_SECONDARY,
                              mutation_scale=ARROW_MUT_SCALE),
             zorder=10)
ax.plot(10.0, 6.0, 'o', color='black', markersize=7, zorder=11)

# 3a → 4
ax.plot([4.0, 4.0], [4.95, 4.4], color='black',
         linewidth=ARROW_LW_SECONDARY, solid_capstyle='round', zorder=9)
ax.plot([4.0, 4.5], [4.4, 4.4], color='black',
         linewidth=ARROW_LW_SECONDARY, solid_capstyle='round', zorder=9)
ax.annotate('', xy=(4.5, 4.25), xytext=(4.5, 4.4),
             arrowprops=dict(arrowstyle='-|>', color='black',
                              lw=ARROW_LW_SECONDARY,
                              mutation_scale=ARROW_MUT_SCALE),
             zorder=10)
# 3b → 4
ax.plot([10.0, 10.0], [4.95, 4.4], color='black',
         linewidth=ARROW_LW_SECONDARY, solid_capstyle='round', zorder=9)
ax.plot([10.0, 9.5], [4.4, 4.4], color='black',
         linewidth=ARROW_LW_SECONDARY, solid_capstyle='round', zorder=9)
ax.annotate('', xy=(9.5, 4.25), xytext=(9.5, 4.4),
             arrowprops=dict(arrowstyle='-|>', color='black',
                              lw=ARROW_LW_SECONDARY,
                              mutation_scale=ARROW_MUT_SCALE),
             zorder=10)

# 4 → 5
ax.annotate('', xy=(7.0, 2.85), xytext=(7.0, 3.55),
             arrowprops=dict(arrowstyle='-|>', color='black',
                              lw=ARROW_LW_MAIN, mutation_scale=ARROW_MUT_SCALE),
             zorder=10)

# 5 → 6
ax.annotate('', xy=(7.0, 1.55), xytext=(7.0, 2.15),
             arrowprops=dict(arrowstyle='-|>', color='black',
                              lw=ARROW_LW_MAIN, mutation_scale=ARROW_MUT_SCALE),
             zorder=10)

# Loop back: 6 → 1
arrow_loop2 = FancyArrowPatch(
    (3.5, 1.2), (2.0, 6.7),
    arrowstyle='-|>', color='darkgreen',
    lw=ARROW_LW_MAIN, mutation_scale=ARROW_MUT_SCALE+2,
    connectionstyle='arc3,rad=-0.45', zorder=10)
ax.add_patch(arrow_loop2)
signal_label(ax, 1.5, 4.0, 'next step\nk+1',
              color='darkgreen', fontsize=11)

ax.set_xlim([0, 14])
ax.set_ylim([0, 16])
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
save_fig(fig, "fig16_algorithm_flow")


# ============================================================
# FIGURE C : Compact summary
# ============================================================
print("[Figure C] Compact summary ...")

fig, axes = plt.subplots(2, 1, figsize=(18, 13),
                           gridspec_kw={'height_ratios': [1.2, 1]})

ax = axes[0]
ax.text(9.0, 11.0, 'PALF-LQG: Phase-Aware Learned Feedforward + LQG',
         fontsize=20, ha='center', fontweight='bold')

eq_box = FancyBboxPatch((2.5, 10.0), 13.0, 0.7,
                          boxstyle='round,pad=0.15',
                          facecolor='#FFFACD', edgecolor='black',
                          linewidth=2.0, zorder=2)
ax.add_patch(eq_box)
ax.text(9.0, 10.35,
         r'$u(t) = u_{LQG}(\hat{x}) + \alpha \cdot NN_{FF}(\phi, \hat{x})$',
         fontsize=16, ha='center', va='center', fontweight='bold')

y_top = 7.5; y_mid = 5.0; y_obs = 2.5

# Phase
make_box(ax, 1.0, y_top, 1.2, 0.9, r'$\phi(t)$',
          COLOR_NN, COLOR_NN_EDGE, fontsize=14)

# NN
make_box(ax, 4.0, y_top, 2.8, 1.4, 'Phase-aware\n NN feedforward',
          COLOR_NN, COLOR_NN_EDGE, fontsize=12)

# alpha
make_box(ax, 6.7, y_top, 0.9, 0.9, r'$\alpha$',
          COLOR_NN, COLOR_NN_EDGE, fontsize=18)

# Sum
make_summing_junction(ax, 9.0, y_mid, r=0.35, signs={'top': '+', 'left': '+'})

# Safety
make_box(ax, 11.5, y_mid, 2.2, 1.3, 'CLF Voltage\nGovernor',
          COLOR_SAFETY, COLOR_SAFETY_EDGE, fontsize=12)

# Plant
make_box(ax, 14.5, y_mid, 2.6, 1.6, 'Plant\n(plate + piezo)',
          COLOR_PLANT, COLOR_PLANT_EDGE, fontsize=13)

# Observer
make_box(ax, 14.5, y_obs, 2.6, 1.0, 'Kalman Observer',
          COLOR_OBSERVER, COLOR_OBSERVER_EDGE, fontsize=12)

# LQG
make_box(ax, 6.0, y_mid, 2.2, 1.3, 'LQG\nController',
          COLOR_LQG, COLOR_LQG_EDGE, fontsize=13)

# Output
ax.plot(17.5, y_mid + 0.3, 'o', color=COLOR_PLANT_EDGE,
         markersize=12, zorder=10)

# Arrows
big_arrow(ax, 1.6, y_top, 2.55, y_top, color=COLOR_NN_EDGE)
big_arrow(ax, 5.4, y_top, 6.2, y_top, color=COLOR_NN_EDGE)
signal_label(ax, 5.8, y_top + 0.55, r'$u_{FF}^{raw}$',
              color=COLOR_NN_EDGE, fontsize=11)

# α → sum
ax.plot([6.7, 6.7], [y_top - 0.45, y_mid + 0.7], color=COLOR_FF_PATH,
         linewidth=ARROW_LW_MAIN, solid_capstyle='round', zorder=9)
ax.plot([6.7, 9.0], [y_mid + 0.7, y_mid + 0.7], color=COLOR_FF_PATH,
         linewidth=ARROW_LW_MAIN, solid_capstyle='round', zorder=9)
ax.annotate('', xy=(9.0, y_mid + 0.37), xytext=(9.0, y_mid + 0.7),
             arrowprops=dict(arrowstyle='-|>', color=COLOR_FF_PATH,
                              lw=ARROW_LW_MAIN, mutation_scale=ARROW_MUT_SCALE),
             zorder=10)
signal_label(ax, 7.85, y_mid + 1.0, r'$\alpha \cdot u_{FF}$',
              color=COLOR_FF_PATH, fontsize=11)

# LQG → sum
big_arrow(ax, 7.1, y_mid, 8.65, y_mid, color=COLOR_FB_PATH)
signal_label(ax, 7.9, y_mid + 0.45, r'$u_{LQG}$',
              color=COLOR_FB_PATH, fontsize=12)

# sum → safety
big_arrow(ax, 9.35, y_mid, 10.4, y_mid, color='black')
signal_label(ax, 9.85, y_mid + 0.45, r'$u^{*}$',
              color='black', fontsize=12)

# safety → plant
big_arrow(ax, 12.6, y_mid, 13.2, y_mid, color='black',
            lw=ARROW_LW_MAIN+0.5)
signal_label(ax, 12.9, y_mid + 0.5, r'$u(t)$',
              color='black', fontsize=13)

# plant → output
big_arrow(ax, 15.8, y_mid + 0.3, 17.35, y_mid + 0.3,
            color=COLOR_PLANT_EDGE, lw=ARROW_LW_MAIN+0.5)
signal_label(ax, 17.55, y_mid + 0.3, r'$y(t)$',
              color=COLOR_PLANT_EDGE, fontsize=14)

# plant → observer
ax.plot([15.8, 15.8], [y_mid - 0.8, y_obs + 0.55], color=COLOR_OBSERVER_PATH,
         linewidth=ARROW_LW_SECONDARY, solid_capstyle='round', zorder=9)
ax.annotate('', xy=(15.8, y_obs + 0.55), xytext=(15.8, y_obs + 0.75),
             arrowprops=dict(arrowstyle='-|>', color=COLOR_OBSERVER_PATH,
                              lw=ARROW_LW_SECONDARY,
                              mutation_scale=ARROW_MUT_SCALE),
             zorder=10)
signal_label(ax, 16.3, 3.7, r'$y_{meas}$',
              color=COLOR_OBSERVER_PATH, fontsize=11)

# observer → LQG
ax.plot([13.2, 6.0], [y_obs, y_obs], color=COLOR_OBSERVER_PATH,
         linewidth=ARROW_LW_SECONDARY, solid_capstyle='round', zorder=9)
ax.plot([6.0, 6.0], [y_obs, y_mid - 0.65], color=COLOR_OBSERVER_PATH,
         linewidth=ARROW_LW_SECONDARY, solid_capstyle='round', zorder=9)
ax.annotate('', xy=(6.0, y_mid - 0.65), xytext=(6.0, y_mid - 0.85),
             arrowprops=dict(arrowstyle='-|>', color=COLOR_OBSERVER_PATH,
                              lw=ARROW_LW_SECONDARY,
                              mutation_scale=ARROW_MUT_SCALE),
             zorder=10)
signal_label(ax, 10.5, y_obs + 0.4, r'$\hat{x}(t)$',
              color=COLOR_OBSERVER_PATH, fontsize=12)

# observer → NN (branch)
ax.plot(6.0, y_obs - 0.4, 'o', color=COLOR_OBSERVER_PATH,
         markersize=8, zorder=11)
ax.plot([6.0, 6.0], [y_obs, y_obs - 0.4], color=COLOR_OBSERVER_PATH,
         linewidth=1.5, linestyle=':', alpha=0.8, zorder=8)
ax.plot([6.0, 4.0], [y_obs - 0.4, y_obs - 0.4], color=COLOR_OBSERVER_PATH,
         linewidth=1.5, linestyle=':', alpha=0.8, zorder=8)
ax.plot([4.0, 4.0], [y_obs - 0.4, y_top - 0.75], color=COLOR_OBSERVER_PATH,
         linewidth=1.5, linestyle=':', alpha=0.8, zorder=8)
ax.annotate('', xy=(4.0, y_top - 0.78), xytext=(4.0, y_top - 0.95),
             arrowprops=dict(arrowstyle='-|>', color=COLOR_OBSERVER_PATH,
                              lw=1.5, mutation_scale=ARROW_MUT_SCALE-4),
             zorder=10)

ax.set_xlim([0, 19])
ax.set_ylim([1.0, 11.7])
ax.set_aspect('equal')
ax.axis('off')


# Bottom : key features
ax = axes[1]
features = [
    {
        'x': 3.0, 'color': COLOR_LQG, 'edge': COLOR_LQG_EDGE,
        'title': '1. Reactive Baseline (LQG)',
        'features': [
            '• Optimal LQR + Kalman observer',
            '• Stabilizes 3 dominant modes',
            '• Damping: 0.31% → 20.3%',
            '• Places closed-loop poles (LQR)',
            '• Reactive, model-based baseline',
        ]
    },
    {
        'x': 9.0, 'color': COLOR_NN, 'edge': COLOR_NN_EDGE,
        'title': '2. Anticipative Feedforward (NN)',
        'features': [
            '• Phase-aware periodic NN',
            r'• Inputs: $(\phi, \hat{x})$, output: $u_{FF}$',
            '• Iterative Learning Control',
            '• Cancels periodic forcing',
            '• Trained offline, deployed online',
        ]
    },
    {
        'x': 15.0, 'color': COLOR_SAFETY, 'edge': COLOR_SAFETY_EDGE,
        'title': '3. CLF Voltage Governor (heuristic)',
        'features': [
            r'• Quadratic $V(x) = x^T P x$ on nominal loop',
            r'• Blends $u^{*}$ toward LQG when $\dot V$ grows',
            '• Falls back to LQG if needed',
            r'• Voltage saturation: $|u| \leq 150 V$',
            '• Soft guard, NOT a stability certificate',
        ]
    },
]
for feat in features:
    x = feat['x']
    make_box(ax, x, 5.5, 4.5, 1.0, feat['title'],
              feat['color'], feat['edge'], fontsize=12)
    for i, line in enumerate(feat['features']):
        ax.text(x - 2.1, 4.5 - i*0.55, line,
                 fontsize=11, ha='left', va='center')

ax.text(9.0, 0.5,
         '→ Held-out effect:  +4.8% RMS (nominal) · +9.8% RMS under -8% model mismatch · '
         'SLD unchanged (feedforward does not move poles)',
         fontsize=12, ha='center', fontweight='bold',
         color='darkgreen', style='italic',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E8',
                    edgecolor='darkgreen', linewidth=2.5))

ax.text(9.0, 7.0, 'Key Innovations of PALF-LQG',
         fontsize=16, ha='center', fontweight='bold')

ax.set_xlim([0, 18])
ax.set_ylim([0, 7.5])
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
save_fig(fig, "fig17_DARC_summary")


print(f"\n{'='*70}")
print(" Diagrams generated with CLEAR ARROWS")
print(f"{'='*70}")
