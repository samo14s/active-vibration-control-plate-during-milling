"""
gen_geometry_figure.py
=======================
PERIPHERAL MILLING — geometry correcte:
   - Plate verticale (80mm haut, encastrée en bas)
   - Tool axis VERTICAL parallel to Z
   - ae = 0.1mm radial (Y direction)
   - ap = 0.3mm axial (Z direction, top of plate)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Circle, Polygon, Ellipse
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.linewidth'] = 1.2

LP, HP, BP = 100.0, 80.0, 4.0
TOOL_D = 10.0
TOOL_RADIUS = TOOL_D / 2
TOOL_HEIGHT = 30
PATCH_X = (0, 20)
PATCH_Z = (0, 60)
AP = 0.3
AE = 0.1
H_PA = 0.7

COLOR_PLATE = '#A8D8E8'
COLOR_PLATE_EDGE = '#1F5582'
COLOR_PLATE_DARK = '#5A95B5'
COLOR_TOOL = '#5A5A5A'
COLOR_TOOL_DARK = '#2A2A2A'
COLOR_TOOL_TIP = '#444444'
COLOR_PATCH = '#FF8C42'
COLOR_PATCH_EDGE = '#A04000'
COLOR_PATH = '#DC143C'
COLOR_GROUND = '#C0C0C0'

OUT_DIR = "figs_article_publication"
os.makedirs(OUT_DIR, exist_ok=True)


def save_fig(fig, name):
    fig.savefig(f"{OUT_DIR}/{name}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{OUT_DIR}/{name}.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {name}.png + .pdf")


def iso(x, y, z):
    angle1 = np.deg2rad(30)
    angle2 = np.deg2rad(15)
    px = x * np.cos(angle1) - y * np.cos(angle2)
    py = z + x * np.sin(angle1) * 0.5 + y * np.sin(angle2) * 0.7
    return px, py


# ============================================================
# FIGURE A : 3D Isometric — VERTICAL tool axis
# ============================================================
print("[Figure A] 3D Isometric — peripheral milling (vertical tool axis) ...")

fig, ax = plt.subplots(figsize=(14, 11))

# Ground (encastrement)
ground_pts = [iso(-15, -15, 0), iso(LP+15, -15, 0),
                iso(LP+15, 30, 0), iso(-15, 30, 0)]
ax.add_patch(Polygon(ground_pts, facecolor=COLOR_GROUND, edgecolor='black',
                       linewidth=1.5, hatch='///', alpha=0.6, zorder=1))
for x_h in np.linspace(-10, LP+10, 25):
    p1 = iso(x_h, 0, 0); p2 = iso(x_h - 3, 0, -8)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
             color='black', linewidth=1.0, zorder=2)

# Plate 3D
ax.add_patch(Polygon([iso(0, 0, 0), iso(LP, 0, 0),
                        iso(LP, 0, HP), iso(0, 0, HP)],
                       facecolor=COLOR_PLATE, edgecolor=COLOR_PLATE_EDGE,
                       linewidth=2.0, zorder=5))
ax.add_patch(Polygon([iso(0, 0, HP), iso(LP, 0, HP),
                        iso(LP, BP, HP), iso(0, BP, HP)],
                       facecolor=COLOR_PLATE_DARK, edgecolor=COLOR_PLATE_EDGE,
                       linewidth=2.0, zorder=5))
ax.add_patch(Polygon([iso(LP, 0, 0), iso(LP, BP, 0),
                        iso(LP, BP, HP), iso(LP, 0, HP)],
                       facecolor='#5A95B5', edgecolor=COLOR_PLATE_EDGE,
                       linewidth=2.0, zorder=4))

# Patch
ax.add_patch(Polygon([iso(PATCH_X[0], -0.5, PATCH_Z[0]),
                        iso(PATCH_X[1], -0.5, PATCH_Z[0]),
                        iso(PATCH_X[1], -0.5, PATCH_Z[1]),
                        iso(PATCH_X[0], -0.5, PATCH_Z[1])],
                       facecolor=COLOR_PATCH, edgecolor=COLOR_PATCH_EDGE,
                       linewidth=2.5, zorder=6, alpha=0.95))

patch_center = iso((PATCH_X[0]+PATCH_X[1])/2, -0.5, (PATCH_Z[0]+PATCH_Z[1])/2)
ax.annotate('Piezo Patch\n(QDA60-200.7)\n20 × 60 × 0.7 mm',
             xy=patch_center, xytext=(patch_center[0]-38, patch_center[1]-8),
             fontsize=10, ha='center', fontweight='bold', color=COLOR_PATCH_EDGE,
             arrowprops=dict(arrowstyle='->', color=COLOR_PATCH_EDGE, lw=1.5),
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor=COLOR_PATCH_EDGE, linewidth=1.5))

# === TOOL : VERTICAL CYLINDER axis parallel to Z ===
TOOL_X = LP / 2
TOOL_Y = BP + TOOL_RADIUS - AE   # tool center y, cutter touching plate at y=BP-ae
TOOL_Z_BOT = HP - AP              # cutter bottom at engagement
TOOL_Z_TOP = TOOL_Z_BOT + TOOL_HEIGHT

theta = np.linspace(0, 2*np.pi, 40)
bot_circle = [iso(TOOL_X + TOOL_RADIUS*np.cos(t),
                   TOOL_Y + TOOL_RADIUS*np.sin(t), TOOL_Z_BOT) for t in theta]
top_circle = [iso(TOOL_X + TOOL_RADIUS*np.cos(t),
                   TOOL_Y + TOOL_RADIUS*np.sin(t), TOOL_Z_TOP) for t in theta]

# Side polygon
bot_xs = [p[0] for p in bot_circle]
li_b = np.argmin(bot_xs); ri_b = np.argmax(bot_xs)
side_corners = [bot_circle[li_b], bot_circle[ri_b],
                  top_circle[ri_b], top_circle[li_b]]
ax.add_patch(Polygon(side_corners, facecolor=COLOR_TOOL,
                       edgecolor=COLOR_TOOL_DARK, linewidth=1.8, zorder=18))

# Top and bottom ellipses
top_xs_arr = np.array([p[0] for p in top_circle])
top_ys_arr = np.array([p[1] for p in top_circle])
ellipse_w = top_xs_arr.max() - top_xs_arr.min()
ellipse_h = top_ys_arr.max() - top_ys_arr.min()
top_center_2d = iso(TOOL_X, TOOL_Y, TOOL_Z_TOP)
bot_center_2d = iso(TOOL_X, TOOL_Y, TOOL_Z_BOT)
ax.add_patch(Ellipse(top_center_2d, ellipse_w, ellipse_h,
                       facecolor=COLOR_TOOL_DARK, edgecolor='black',
                       linewidth=1.5, zorder=19))
ax.add_patch(Ellipse(bot_center_2d, ellipse_w, ellipse_h,
                       facecolor=COLOR_TOOL_TIP, edgecolor='black',
                       linewidth=1.5, zorder=17))

# Helical flutes (3 teeth)
for n_tooth in range(3):
    flute_x = []; flute_y = []
    for z_frac in np.linspace(0, 1, 25):
        t_now = n_tooth * 2*np.pi/3 + z_frac * np.pi/4
        z_now = TOOL_Z_BOT + z_frac * TOOL_HEIGHT
        x_now = TOOL_X + TOOL_RADIUS * np.cos(t_now)
        y_now = TOOL_Y + TOOL_RADIUS * np.sin(t_now)
        if np.sin(t_now) < 0.3:
            p = iso(x_now, y_now, z_now)
            flute_x.append(p[0]); flute_y.append(p[1])
    if len(flute_x) > 2:
        ax.plot(flute_x, flute_y, color='red', linewidth=1.3,
                 alpha=0.7, zorder=20)

# Cutting marks at tip
for n_tooth in range(3):
    t_now = n_tooth * 2*np.pi/3
    if np.sin(t_now) < 0.5:
        x_now = TOOL_X + TOOL_RADIUS*0.95*np.cos(t_now)
        y_now = TOOL_Y + TOOL_RADIUS*0.95*np.sin(t_now)
        p = iso(x_now, y_now, TOOL_Z_BOT - 0.5)
        ax.plot(p[0], p[1], 'r^', markersize=8,
                 markeredgecolor='darkred', zorder=21)

# Rotation arrow Ω at top of tool
omega_z = TOOL_Z_TOP + 6
omega_pts = []
for t in np.linspace(np.pi*0.2, np.pi*1.8, 30):
    x_now = TOOL_X + TOOL_RADIUS*1.4 * np.cos(t)
    y_now = TOOL_Y + TOOL_RADIUS*1.4 * np.sin(t)
    omega_pts.append(iso(x_now, y_now, omega_z))
ax.plot([p[0] for p in omega_pts], [p[1] for p in omega_pts],
         color='red', linewidth=2.2, zorder=22)
ax.annotate('', xy=omega_pts[-1], xytext=omega_pts[-3],
             arrowprops=dict(arrowstyle='-|>', color='red',
                              lw=2.0, mutation_scale=18), zorder=22)
omega_lbl = iso(TOOL_X + TOOL_RADIUS*2, TOOL_Y, omega_z + 4)
ax.text(omega_lbl[0], omega_lbl[1], r'$\Omega$ = 4900 RPM',
         fontsize=11, color='red', fontweight='bold')

# Tool path
path_x_arr = np.linspace(0, LP, 50)
path_pts = [iso(px, TOOL_Y, HP - AP/2 + 4) for px in path_x_arr]
ax.plot([p[0] for p in path_pts], [p[1] for p in path_pts],
         color=COLOR_PATH, linewidth=2.8, linestyle='--',
         alpha=0.9, zorder=10)
for i in [10, 25, 40]:
    p_start = iso(path_x_arr[i] - 1, TOOL_Y, HP - AP/2 + 4)
    p_end = iso(path_x_arr[i] + 1, TOOL_Y, HP - AP/2 + 4)
    ax.annotate('', xy=p_end, xytext=p_start,
                 arrowprops=dict(arrowstyle='-|>', color=COLOR_PATH,
                                  lw=2.0, mutation_scale=18), zorder=11)

# Feed arrow
feed_pt = iso(TOOL_X + 6, TOOL_Y, TOOL_Z_TOP + 2)
feed_start = iso(TOOL_X + 2, TOOL_Y, TOOL_Z_TOP + 2)
ax.annotate('', xy=feed_pt, xytext=feed_start,
             arrowprops=dict(arrowstyle='-|>', color='#0066CC',
                              lw=2.5, mutation_scale=22), zorder=22)
ax.text(feed_pt[0] + 3, feed_pt[1], r'$v_{feed}$',
         fontsize=11, color='#0066CC', fontweight='bold')

# Coordinate axes
origin_iso = iso(0, 0, 0)
for end_3d, lbl in [(iso(20, 0, 0), '$X_P$'),
                       (iso(0, 15, 0), '$Y_P$'),
                       (iso(0, 0, 25), '$Z_P$')]:
    ax.annotate('', xy=end_3d, xytext=origin_iso,
                 arrowprops=dict(arrowstyle='->', color='black',
                                  lw=2.5, mutation_scale=20), zorder=20)
ax.text(iso(22, 0, 0)[0]+1, iso(22, 0, 0)[1]-1, '$X_P$',
         fontsize=14, fontweight='bold')
ax.text(iso(0, 17, 0)[0]-3, iso(0, 17, 0)[1], '$Y_P$',
         fontsize=14, fontweight='bold')
ax.text(iso(0, 0, 27)[0]-4, iso(0, 0, 27)[1], '$Z_P$',
         fontsize=14, fontweight='bold')
ax.plot(origin_iso[0], origin_iso[1], 'ko', markersize=8, zorder=21)
ax.text(origin_iso[0]-3, origin_iso[1]-4, '$O_P$',
         fontsize=13, fontweight='bold')

# Dimensions
lp_y = -8
ax.annotate('', xy=(iso(LP, 0, 0)[0], lp_y - 4),
             xytext=(iso(0, 0, 0)[0], lp_y - 4),
             arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
ax.text((iso(0, 0, 0)[0]+iso(LP, 0, 0)[0])/2, lp_y - 8,
         f'$L_P$ = {LP:.0f} mm', fontsize=11, ha='center',
         fontweight='bold', color='blue')

hp_top = iso(LP+5, 0, HP); hp_bot = iso(LP+5, 0, 0)
ax.annotate('', xy=(hp_top[0]+8, hp_top[1]),
             xytext=(hp_bot[0]+8, hp_bot[1]),
             arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
ax.text(hp_top[0]+12, (hp_bot[1]+hp_top[1])/2,
         f'$H_P$ = {HP:.0f} mm', fontsize=11,
         fontweight='bold', color='blue', rotation=90)

# Tool label
tool_lbl = iso(TOOL_X, TOOL_Y, TOOL_Z_TOP + 25)
ax.annotate(f'Milling Tool\n$N_T$ = 3 teeth\n$D$ = {TOOL_D} mm\nVertical axis',
             xy=iso(TOOL_X, TOOL_Y, TOOL_Z_TOP),
             xytext=(tool_lbl[0]+30, tool_lbl[1]),
             fontsize=10, ha='center', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                        edgecolor='black', linewidth=1.5))

# Path label
path_lbl = iso(LP*0.2, TOOL_Y, HP+18)
ax.annotate(f'Tool path  ($X_P$: 0 → {LP:.0f} mm,  $T_{{path}}$ = 20.4 s)',
             xy=iso(LP*0.4, TOOL_Y, HP - AP/2 + 4),
             xytext=(path_lbl[0], path_lbl[1]),
             fontsize=10, color=COLOR_PATH, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=COLOR_PATH, lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=COLOR_PATH, linewidth=1.2))

# Encastrement
enc_pt = iso(-10, 0, -3)
ax.text(enc_pt[0]-5, enc_pt[1]-8, 'Clamped base\n(encastrement)',
         fontsize=10, fontweight='bold', color='black',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='black', linewidth=1.0))

# Material
ax.text(0.02, 0.97, 'AL6061\n$\\rho$=2830 kg/m³\n$E$=69 GPa\n$\\nu$=0.33',
         transform=ax.transAxes, fontsize=10, va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFACD',
                    edgecolor='black', linewidth=1.5),
         fontweight='bold')

# Cutting parameters
ax.text(0.98, 0.50,
         f'Peripheral milling:\n'
         f'$f_t$ = 0.02 mm/tooth\n'
         f'$a_p$ = {AP} mm (axial, $Z$)\n'
         f'$a_e$ = {AE} mm (radial, $Y$)\n'
         f'$v_{{feed}}$ = 4.9 mm/s ($X$)',
         transform=ax.transAxes, fontsize=10, va='top', ha='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E8',
                    edgecolor='black', linewidth=1.5),
         fontweight='bold')

ax.set_title("Peripheral milling of thin-walled cantilever plate (AL6061)",
              fontsize=14, fontweight='bold', pad=15)
ax.set_xlim([-30, 130])
ax.set_ylim([-25, 140])
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
save_fig(fig, "fig00a_geometry_3D")


# ============================================================
# FIGURE B : Front view 2D
# ============================================================
print("[Figure B] Front view 2D ...")

fig, ax = plt.subplots(figsize=(13, 10))

ax.add_patch(Rectangle((0, 0), LP, HP, facecolor=COLOR_PLATE,
                         edgecolor=COLOR_PLATE_EDGE, linewidth=2.5, zorder=2))
for x_h in np.linspace(-5, LP+5, 30):
    ax.plot([x_h, x_h - 2], [0, -4], color='black', linewidth=1.2, zorder=3)
ax.plot([-5, LP+5], [0, 0], color='black', linewidth=2.0, zorder=3)
ax.text(LP/2, -7, 'Clamped edge', fontsize=11, fontweight='bold',
         ha='center', style='italic')

ax.add_patch(Rectangle((PATCH_X[0], PATCH_Z[0]),
                         PATCH_X[1]-PATCH_X[0], PATCH_Z[1]-PATCH_Z[0],
                         facecolor=COLOR_PATCH, edgecolor=COLOR_PATCH_EDGE,
                         linewidth=2.5, alpha=0.9, zorder=4))
ax.text(10, 30, 'Piezo\nPatch', fontsize=12, fontweight='bold',
         ha='center', color='white',
         bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_PATCH_EDGE,
                    edgecolor='white', linewidth=1.5))

ax.annotate('', xy=(PATCH_X[1]+8, PATCH_Z[0]),
             xytext=(PATCH_X[1]+8, PATCH_Z[1]),
             arrowprops=dict(arrowstyle='<->', color=COLOR_PATCH_EDGE, lw=1.5))
ax.text(PATCH_X[1]+11, (PATCH_Z[0]+PATCH_Z[1])/2, '60 mm',
         fontsize=10, color=COLOR_PATCH_EDGE, fontweight='bold',
         rotation=90, va='center')
ax.annotate('', xy=(PATCH_X[0], -3), xytext=(PATCH_X[1], -3),
             arrowprops=dict(arrowstyle='<->', color=COLOR_PATCH_EDGE, lw=1.5))
ax.text((PATCH_X[0]+PATCH_X[1])/2, -5, '20 mm',
         fontsize=10, color=COLOR_PATCH_EDGE, fontweight='bold', ha='center')

# Tool: rectangle (front view of vertical cylinder)
TOOL_X = LP / 2
TOOL_TOP = HP + TOOL_HEIGHT
TOOL_BOT = HP - AP

ax.add_patch(Rectangle((TOOL_X - TOOL_RADIUS, TOOL_BOT),
                         TOOL_D, TOOL_TOP - TOOL_BOT,
                         facecolor=COLOR_TOOL, edgecolor=COLOR_TOOL_DARK,
                         linewidth=2.0, zorder=10, alpha=0.92))

# Helical flutes
for n_tooth in range(3):
    fx = []; fz = []
    for z_frac in np.linspace(0, 1, 30):
        z_pos = TOOL_BOT + z_frac * (TOOL_TOP - TOOL_BOT)
        angle = n_tooth * 2*np.pi/3 + z_frac * np.pi/2
        x_offset = TOOL_RADIUS * 0.92 * np.cos(angle)
        if np.sin(angle) < 0.4:
            fx.append(TOOL_X + x_offset); fz.append(z_pos)
    if len(fx) > 2:
        ax.plot(fx, fz, color='red', linewidth=1.3, alpha=0.65, zorder=11)

# Tool tip teeth
for n_tooth in range(3):
    angle = n_tooth * 2*np.pi/3
    if np.sin(angle) < 0.5:
        x_pos = TOOL_X + TOOL_RADIUS*0.85*np.cos(angle)
        ax.plot(x_pos, TOOL_BOT, 'rv', markersize=10,
                 markeredgecolor='darkred', linewidth=1.5, zorder=12)

# Rotation Ω at top
arc = mpatches.FancyArrowPatch(
    (TOOL_X - TOOL_RADIUS*1.5, TOOL_TOP + 4),
    (TOOL_X + TOOL_RADIUS*1.5, TOOL_TOP + 4),
    connectionstyle="arc3,rad=-0.45",
    arrowstyle='->', color='red', lw=2.5, mutation_scale=18, zorder=15)
ax.add_patch(arc)
ax.text(TOOL_X, TOOL_TOP + 13, r'$\Omega$ = 4900 RPM',
         fontsize=11, color='red', fontweight='bold', ha='center')

# Feed arrow
ax.annotate('', xy=(TOOL_X + 14, TOOL_TOP + 1),
             xytext=(TOOL_X + 4, TOOL_TOP + 1),
             arrowprops=dict(arrowstyle='-|>', color='#0066CC',
                              lw=2.5, mutation_scale=20))
ax.text(TOOL_X + 9, TOOL_TOP + 4, r'$v_{feed}$',
         fontsize=11, color='#0066CC', fontweight='bold', ha='center')

# Tool path
ax.plot([0, LP], [HP-0.3, HP-0.3], color=COLOR_PATH, linewidth=2.8,
         linestyle='--', alpha=0.85, zorder=5)
for x_arr in [25, 50, 75]:
    ax.annotate('', xy=(x_arr+3, HP-0.3), xytext=(x_arr-3, HP-0.3),
                 arrowprops=dict(arrowstyle='-|>', color=COLOR_PATH,
                                  lw=2.0, mutation_scale=20))

# Sensor
ax.plot(LP, HP, 'ks', markersize=12, markerfacecolor='yellow',
         markeredgecolor='black', markeredgewidth=1.5, zorder=12)
ax.annotate(f'Sensor point\n($x_p$={LP:.0f}, $z_p$={HP:.0f})',
             xy=(LP, HP), xytext=(LP+8, HP-15),
             fontsize=10, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='black', lw=1.3))

# ap annotation
ax.annotate('', xy=(TOOL_X + TOOL_RADIUS + 3, HP),
             xytext=(TOOL_X + TOOL_RADIUS + 3, TOOL_BOT),
             arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
ax.text(TOOL_X + TOOL_RADIUS + 6, (HP + TOOL_BOT)/2,
         f'$a_p$={AP}mm', fontsize=10, color='red',
         fontweight='bold', va='center')

# Axes
ax.annotate('', xy=(15, 0), xytext=(0, 0),
             arrowprops=dict(arrowstyle='-|>', color='black',
                              lw=2.5, mutation_scale=22), zorder=20)
ax.text(16, -1, '$X_P$', fontsize=14, fontweight='bold')
ax.annotate('', xy=(0, 15), xytext=(0, 0),
             arrowprops=dict(arrowstyle='-|>', color='black',
                              lw=2.5, mutation_scale=22), zorder=20)
ax.text(-3, 16, '$Z_P$', fontsize=14, fontweight='bold')
ax.plot(0, 0, 'ko', markersize=8, zorder=21)
ax.text(-4, -3, '$O_P$', fontsize=13, fontweight='bold')

# Dimensions
ax.annotate('', xy=(LP, -10), xytext=(0, -10),
             arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
ax.text(LP/2, -13, f'$L_P$ = {LP:.0f} mm', fontsize=12,
         color='blue', fontweight='bold', ha='center')
ax.annotate('', xy=(-10, HP), xytext=(-10, 0),
             arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
ax.text(-14, HP/2, f'$H_P$ = {HP:.0f} mm', fontsize=12,
         color='blue', fontweight='bold', ha='center', rotation=90)

ax.text(0.02, 0.98, 'AL6061\n$\\rho$ = 2830 kg/m³\n$E$ = 69 GPa\n$B_P$ = 4 mm',
         transform=ax.transAxes, fontsize=10, va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFACD',
                    edgecolor='black', linewidth=1.5),
         fontweight='bold')

ax.text(0.98, 0.40,
         f'Peripheral milling:\n'
         f'$f_t$ = 0.02 mm/t\n'
         f'$a_p$ = {AP} mm (Z)\n'
         f'$a_e$ = {AE} mm (Y)\n'
         f'$N_T$ = 3 teeth\n'
         f'$D$ = {TOOL_D} mm',
         transform=ax.transAxes, fontsize=10, va='top', ha='right',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E8',
                    edgecolor='black', linewidth=1.5),
         fontweight='bold')

ax.annotate(f'Milling Tool\nVertical axis ($\\parallel Z_P$)\n$D$ = {TOOL_D} mm',
             xy=(TOOL_X, (TOOL_TOP + TOOL_BOT)/2),
             xytext=(TOOL_X - 28, (TOOL_TOP + TOOL_BOT)/2 + 5),
             fontsize=10, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='black', lw=1.3),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                        edgecolor='black', linewidth=1.2))

ax.annotate(f'Tool path: $x_p(t) = v_{{feed}} \\cdot t$\n($T_{{path}}$ = 20.4 s)',
             xy=(LP*0.4, HP), xytext=(LP*0.05, HP+22),
             fontsize=10, color=COLOR_PATH, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=COLOR_PATH, lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=COLOR_PATH, linewidth=1.2))

ax.set_xlim([-25, LP+30])
ax.set_ylim([-20, TOOL_TOP + 22])
ax.set_aspect('equal')
ax.set_xlabel("$X_P$ (mm)", fontsize=12)
ax.set_ylabel("$Z_P$ (mm)", fontsize=12)
ax.set_title("Peripheral milling — front view ($Y_P$ = 0 plane)",
              fontsize=13, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
save_fig(fig, "fig00b_geometry_front")


# ============================================================
# FIGURE C : Top view - KEY for peripheral milling
# ============================================================
print("[Figure C] Top view (KEY for peripheral milling) ...")

fig, ax = plt.subplots(figsize=(14, 8))

ax.add_patch(Rectangle((0, 0), LP, BP, facecolor=COLOR_PLATE,
                         edgecolor=COLOR_PLATE_EDGE, linewidth=2.5, zorder=3))

# Tool circle (top view)
TOOL_Y_top = BP + TOOL_RADIUS - AE
ax.add_patch(Circle((TOOL_X, TOOL_Y_top), TOOL_RADIUS,
                      facecolor=COLOR_TOOL, edgecolor=COLOR_TOOL_DARK,
                      linewidth=2.5, zorder=10, alpha=0.92))

# Tool teeth
for n_tooth in range(3):
    angle = n_tooth * 2*np.pi/3 + np.pi/6
    tx = TOOL_X + TOOL_RADIUS * 0.88 * np.cos(angle)
    ty = TOOL_Y_top + TOOL_RADIUS * 0.88 * np.sin(angle)
    ax.plot(tx, ty, marker='^', color='red', markersize=11,
             markeredgecolor='darkred', linewidth=1.5, zorder=11)

# Rotation Ω (curved arrow in plane)
arc = mpatches.FancyArrowPatch(
    (TOOL_X - TOOL_RADIUS*0.6, TOOL_Y_top + TOOL_RADIUS*1.2),
    (TOOL_X + TOOL_RADIUS*0.6, TOOL_Y_top + TOOL_RADIUS*1.2),
    connectionstyle="arc3,rad=0.45",
    arrowstyle='->', color='red', lw=2.5, mutation_scale=18, zorder=15)
ax.add_patch(arc)
ax.text(TOOL_X, TOOL_Y_top + TOOL_RADIUS*2.0,
         r'$\Omega$ = 4900 RPM', fontsize=11,
         color='red', fontweight='bold', ha='center')

# Feed arrow (along X)
ax.annotate('', xy=(TOOL_X + 12, TOOL_Y_top), xytext=(TOOL_X + 4, TOOL_Y_top),
             arrowprops=dict(arrowstyle='-|>', color='#0066CC',
                              lw=2.5, mutation_scale=22))
ax.text(TOOL_X + 8, TOOL_Y_top + 1.5, '$v_{feed}$',
         fontsize=11, color='#0066CC', fontweight='bold', ha='center')

# Tool path
ax.plot([0, LP], [TOOL_Y_top, TOOL_Y_top], color=COLOR_PATH, linewidth=2.5,
         linestyle='--', alpha=0.7, zorder=5)
for x_arr in [25, 50, 75]:
    ax.annotate('', xy=(x_arr+3, TOOL_Y_top), xytext=(x_arr-3, TOOL_Y_top),
                 arrowprops=dict(arrowstyle='-|>', color=COLOR_PATH,
                                  lw=2.0, mutation_scale=18))

# Cut depth line
ax.plot([0, LP], [BP - AE, BP - AE], color='red', linewidth=1.5,
         linestyle=':', alpha=0.8, zorder=6)
ax.text(85, BP - AE - 0.6, 'Cut depth line',
         fontsize=9, color='red', fontweight='bold', va='top')

# ae annotation
ax.annotate('', xy=(60, BP), xytext=(60, BP - AE),
             arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
ax.text(62, BP - AE/2, f'$a_e$ = {AE} mm\n(radial)',
         fontsize=10, color='red', fontweight='bold', va='center')

# Plate dimensions
ax.annotate('', xy=(LP, -1.5), xytext=(0, -1.5),
             arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
ax.text(LP/2, -2.5, f'$L_P$ = {LP:.0f} mm',
         fontsize=12, color='blue', fontweight='bold', ha='center')
ax.annotate('', xy=(-3, BP), xytext=(-3, 0),
             arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
ax.text(-5, BP/2, f'$B_P$ = {BP:.0f} mm', fontsize=11,
         color='blue', fontweight='bold', ha='center', rotation=90)

# Axes
ax.annotate('', xy=(15, 0), xytext=(0, 0),
             arrowprops=dict(arrowstyle='-|>', color='black',
                              lw=2.5, mutation_scale=22), zorder=20)
ax.text(16, -0.5, '$X_P$', fontsize=14, fontweight='bold')
ax.annotate('', xy=(0, 4), xytext=(0, 0),
             arrowprops=dict(arrowstyle='-|>', color='black',
                              lw=2.5, mutation_scale=22), zorder=20)
ax.text(-2.5, 4, '$Y_P$', fontsize=14, fontweight='bold')
ax.plot(0, 0, 'ko', markersize=8, zorder=21)
ax.text(-3, -1, '$O_P$', fontsize=13, fontweight='bold')

ax.annotate(f'Tool (vertical axis ⊥ page)\n$D$ = {TOOL_D} mm,  $N_T$ = 3 teeth',
             xy=(TOOL_X, TOOL_Y_top - TOOL_RADIUS),
             xytext=(LP*0.15, TOOL_Y_top + TOOL_RADIUS*0.5),
             fontsize=10, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='black', lw=1.3),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                        edgecolor='black', linewidth=1.2))

ax.text(LP*0.3, BP/2, 'Plate (top view)',
         fontsize=11, fontweight='bold', ha='center', va='center',
         color=COLOR_PLATE_EDGE, alpha=0.7)

ax.set_xlim([-10, LP+15])
ax.set_ylim([-5, TOOL_Y_top + TOOL_RADIUS*2.5 + 3])
ax.set_aspect('equal')
ax.set_xlabel("$X_P$ (mm)", fontsize=12)
ax.set_ylabel("$Y_P$ (mm)", fontsize=12)
ax.set_title("Peripheral milling — top view ($Z_P = H_P$ plane)\n"
              "Tool circle visible because vertical axis ⊥ to view",
              fontsize=12, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
save_fig(fig, "fig00c_geometry_top")


print(f"\n{'='*70}")
print(" Geometry figures generated (peripheral milling, vertical tool)")
print(f"{'='*70}")
