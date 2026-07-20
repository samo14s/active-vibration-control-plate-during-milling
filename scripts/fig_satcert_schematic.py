"""SatCERT proposal schematic: (a) architecture block diagram,
(b) concept sketch of the certified permissible-depth envelope
(illustrative, not computed). Output: docs/figures/satcert_schematic.*"""
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 7.5, "font.family": "serif"})


def box(ax, xy, w, h, text, fc, fs=7.0, ec="k"):
    b = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012",
                       fc=fc, ec=ec, lw=0.9)
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fs)
    return b


def arrow(ax, p, q, **kw):
    a = FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9,
                        lw=1.0, color="k", **kw)
    ax.add_patch(a)


def main():
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(10.4, 4.4), width_ratios=[1.45, 1.0])

    # ============ (a) architecture ====================================
    ax = axa
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("(a) SatCERT architecture", fontsize=9)

    C_OFF = "#dbe9f6"
    C_ON = "#e8f5e9"
    C_VAL = "#fdf2d9"

    # offline layer container
    box(ax, (0.15, 5.4), 9.7, 4.4, "", "#f4f8fc", ec="#8aa8c8")
    ax.text(0.35, 9.55, "OFFLINE — certification pipeline", fontsize=7.5,
            style="italic", color="#38618c")
    box(ax, (0.4, 7.6), 2.6, 1.5,
        "validated plant chain\n(FEM + piezo + regenerative\ndelay; paper-1)",
        C_OFF)
    box(ax, (3.4, 7.6), 3.0, 1.5,
        "semi-discretization LIFT\nperiodic DDE + deadzone\n"
        r"$z_{k+1}=A_k z_k + B_k\,\varphi(C_K z_k)$", C_OFF)
    box(ax, (6.8, 7.6), 2.9, 1.5,
        "REGIONAL CERTIFICATE\nperiodic Lyapunov LMI +\ngeneralized sector "
        r"$\Rightarrow\ \mathcal{E}(P_k)$", C_OFF)
    box(ax, (0.4, 5.7), 2.9, 1.4,
        "anti-windup synthesis\niterative LMI maximizing\ncertified basin",
        C_OFF)
    box(ax, (3.7, 5.7), 6.0, 1.4,
        "ENVELOPE SWEEP over $(a_p, \\Omega, x_T)$\n"
        "certified permissible-depth envelope:\n"
        "guaranteed | linear-only (islands possible) | unstable", C_OFF)
    arrow(ax, (3.0, 8.35), (3.4, 8.35))
    arrow(ax, (6.4, 8.35), (6.8, 8.35))
    arrow(ax, (8.25, 7.6), (1.85, 7.1))
    arrow(ax, (3.3, 6.4), (3.7, 6.4))

    # online layer
    box(ax, (0.15, 2.6), 9.7, 2.3, "", "#f2faf3", ec="#7cb08a")
    ax.text(0.35, 4.62, "ONLINE — production", fontsize=7.5,
            style="italic", color="#3d7a4e")
    box(ax, (0.4, 2.9), 2.4, 1.3,
        "$H_\\infty$ controller\n(frozen or PS-LPV,\npaper-1) + AW", C_ON)
    box(ax, (3.2, 2.9), 1.7, 1.3, "sat($\\pm$150 V)\namplifier", C_ON)
    box(ax, (5.3, 2.9), 2.1, 1.3, "piezo patch +\nthin-walled plate", C_ON)
    box(ax, (7.8, 2.9), 1.9, 1.3,
        "milling force\nregenerative delay\n$\\tau=60/(N\\Omega)$", C_ON)
    arrow(ax, (2.8, 3.55), (3.2, 3.55))
    arrow(ax, (4.9, 3.55), (5.3, 3.55))
    arrow(ax, (7.4, 3.55), (7.8, 3.55))
    arrow(ax, (8.75, 2.9), (8.75, 2.45))
    ax.plot([8.75, 1.6], [2.45, 2.45], "k", lw=1.0)
    arrow(ax, (1.6, 2.45), (1.6, 2.9))
    ax.text(5.2, 2.18, "sensor $y_s$ (eddy-current)", fontsize=6.5,
            ha="center")
    arrow(ax, (5.6, 5.7), (5.6, 5.15), linestyle=(0, (3, 2)))
    ax.text(5.75, 5.32, "gains + envelope\nto process planning",
            fontsize=6.0, va="center")

    # validation layer
    box(ax, (0.15, 0.15), 9.7, 1.7, "", "#fdf8ec", ec="#c9a94e")
    ax.text(0.35, 1.62, "VALIDATION — no rig required", fontsize=7.5,
            style="italic", color="#8a6d1f")
    box(ax, (0.4, 0.35), 2.9, 1.0,
        "numerical basins\n(IC sweeps, nonlinear simulator)", C_VAL)
    box(ax, (3.6, 0.35), 3.1, 1.0,
        "reproduce measured saturation\nislands (Ozsoy et al., MSSP 2025)",
        C_VAL)
    box(ax, (7.0, 0.35), 2.7, 1.0,
        "certificate $\\subseteq$ basin $\\subseteq$\nisland-free region",
        C_VAL)
    arrow(ax, (5.0, 2.6), (5.0, 1.85), linestyle=(0, (3, 2)))

    # ============ (b) envelope concept sketch =========================
    ax = axb
    ax.set_title("(b) certified envelope — concept sketch\n"
                 "(illustrative, not computed)", fontsize=8.5)
    om = np.linspace(2, 10, 400)
    lin = 1.1 + 0.55 * np.abs(np.sin(0.85 * om + 0.4)) \
        + 0.35 * np.abs(np.sin(2.1 * om))
    cert = 0.55 * lin - 0.12 * np.abs(np.sin(1.3 * om))
    cert = np.clip(cert, 0.12, None)
    ax.fill_between(om, 0, cert, color="#2ca02c", alpha=0.35,
                    label="certified (basin margin $\\geq h_{\\max}$)")
    ax.fill_between(om, cert, lin, color="#ff7f0e", alpha=0.30,
                    label="linear-only (saturation islands possible)")
    ax.fill_between(om, lin, 2.2, color="#d62728", alpha=0.25,
                    label="linearly unstable")
    ax.plot(om, lin, "k", lw=1.2)
    ax.plot(om, cert, "k--", lw=1.1)
    # a few illustrative "islands" in the linear-only zone
    for xo, yo in ((3.4, 1.05), (6.1, 1.25), (8.3, 0.95)):
        ax.scatter([xo], [yo], marker="x", s=35, color="#7f1d1d", zorder=5)
    ax.scatter([], [], marker="x", s=35, color="#7f1d1d",
               label="measured-island analogue")
    ax.set_xlabel("spindle speed [krpm]")
    ax.set_ylabel(r"axial depth of cut $a_p$ [mm]")
    ax.set_ylim(0, 2.2)
    ax.legend(fontsize=6.2, loc="upper left")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"satcert_schematic.{ext}", dpi=250,
                    bbox_inches="tight")
    print("saved", OUT / "satcert_schematic.png")


if __name__ == "__main__":
    main()
