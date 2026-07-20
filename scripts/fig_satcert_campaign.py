"""SatCERT campaign figure: (a) certified-envelope census maps,
(b) h_req sensitivity of the certified worst-position depth,
(c) certificate-vs-simulator validation parity, (d) saturation-island
causal demonstration. Inputs: results/satcert_campaign.json,
results/satcert_islands.json. Output: docs/figures/satcert_campaign.*"""
import json
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 7.5, "font.family": "serif",
                     "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.4})

# zone colors: lightness-separated (grayscale-safe) + hatch on S
C_CERT, C_LIN, C_SAT, C_UNS = "#3f8f4f", "#f2c14e", "#6b1f2a", "#d1948f"
ZID = {"C": 0, "L": 1, "S": 2, "U": 3}
CMAP = ListedColormap([C_CERT, C_LIN, C_SAT, C_UNS])
X_MM = (5, 25, 50, 75, 95)


def main():
    camp = json.loads((ROOT / "results" / "satcert_campaign.json").read_text())
    try:
        isl = json.loads((ROOT / "results" / "satcert_islands.json").read_text())
    except FileNotFoundError:
        isl = None

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.9))
    axa, axb, axc, axd = axes.flat

    # ---------------- (a) census maps --------------------------------
    aps = np.array(camp["census"]["ap_grid_mm"])
    strategies = ("FROZEN", "PS-LPV")
    grid = np.zeros((2 * len(X_MM), len(aps)))
    for si, s in enumerate(strategies):
        for xi, x in enumerate((0.005, 0.025, 0.05, 0.075, 0.095)):
            row = camp["census"]["zones"][f"{s}|{x}"]
            grid[si * 5 + xi] = [ZID[c] for c in row]
    da = aps[1] - aps[0]
    axa.imshow(grid, aspect="auto", cmap=CMAP, vmin=-0.5, vmax=3.5,
               extent=[aps[0] - da / 2, aps[-1] + da / 2, 2 * 5 - 0.5, -0.5],
               interpolation="nearest")
    for i in range(2 * 5):          # hatch the forced-saturated cells
        for j, v in enumerate(grid[i]):
            if v == 2:
                axa.add_patch(plt.Rectangle((aps[j] - da / 2, i - 0.5),
                                            da, 1.0, fill=False,
                                            hatch="////", lw=0,
                                            edgecolor="w"))
    for si, s in enumerate(strategies):     # cert + linear markers
        for xi, x in enumerate((0.005, 0.025, 0.05, 0.075, 0.095)):
            t = camp["table"][f"{s}|{x}"]
            y = si * 5 + xi
            axa.plot(t["ap_cert_mm"], y, "k<", ms=4.5, mew=0.6, mfc="w",
                     zorder=5)
            if t["ap_lin_mm"] <= aps[-1]:
                axa.plot(t["ap_lin_mm"], y, "ko", ms=3.6, mfc="w", zorder=5)
    axa.set_yticks(range(10))
    axa.set_yticklabels([f"{s[:1]} x={x}" for s in ("F", "P")
                         for x in X_MM], fontsize=6)
    axa.axhline(4.5, color="k", lw=0.8)
    axa.text(0.02, 0.96, "best frozen H$_\\infty$", transform=axa.transAxes,
             fontsize=6.5, va="top")
    axa.text(0.02, 0.46, "PS-LPV", transform=axa.transAxes, fontsize=6.5,
             va="top")
    axa.set_xlabel("axial depth of cut $a_p$ [mm]")
    axa.set_ylabel("tool position [mm]")
    axa.set_title("(a) certified envelope census, 4.9 krpm "
                  "($h_{req}$ = 20 µm)", fontsize=8.5)
    axa.grid(False)
    axa.legend(handles=[
        Patch(fc=C_CERT, label="certified"),
        Patch(fc=C_LIN, label="linear-only (islands possible)"),
        Patch(fc=C_SAT, hatch="////", ec="w", label="forced-orbit saturated"),
        Patch(fc=C_UNS, label="linearly unstable"),
        plt.Line2D([], [], marker="<", ls="", mfc="w", c="k",
                   label="certified depth"),
        plt.Line2D([], [], marker="o", ls="", mfc="w", c="k",
                   label="linear boundary")],
        fontsize=5.6, loc="lower right", ncol=2, framealpha=0.95)

    # ---------------- (b) h_req sensitivity --------------------------
    hs = (1, 2, 5, 10, 20, 50)
    for s, c, mk in (("FROZEN", "#4c72b0", "o"), ("PS-LPV", "#c44e52", "s")):
        w = [camp["hreq_sensitivity"][f"{s}|{h}um"]["worst_mm"] for h in hs]
        axb.plot(hs, w, marker=mk, ms=4, lw=1.4, color=c,
                 label=("best frozen H$_\\infty$" if s == "FROZEN"
                        else "PS-LPV"))
    axb.set_xscale("log")
    axb.set_xticks(hs)
    axb.set_xticklabels([str(h) for h in hs])
    axb.set_xlabel("declared surface-step tolerance $h_{req}$ [µm]")
    axb.set_ylabel("worst-position certified depth [mm]")
    axb.set_title("(b) certified depth vs declared tolerance", fontsize=8.5)
    axb.annotate("2.7×", xy=(1, 2.656), xytext=(1.5, 2.3), fontsize=7,
                 arrowprops=dict(arrowstyle="-", lw=0.6))
    axb.annotate("0.79×\n(ranking inverted)", xy=(20, 0.508),
                 xytext=(9, 1.45), fontsize=7,
                 arrowprops=dict(arrowstyle="-", lw=0.6))
    axb.legend(fontsize=6.5)

    # ---------------- (c) validation parity --------------------------
    pts = []
    for ap, v in camp["validate"].items():
        pts.append((v["cert_h_plus_um"], v["sim_onset_plus_um"],
                    f"+h, {ap}"))
        pts.append((v["cert_h_minus_um"], v["sim_onset_minus_um"],
                    f"$-$h, {ap}"))
    lim = (3, 200)
    axc.plot(lim, lim, "k--", lw=0.8, label="perfect agreement")
    for (m, s, lab), mk in zip(pts, ("o", "s", "^", "D")):
        axc.plot(m, s, mk, ms=5, mew=0.7, mfc="#4c72b0", mec="k", alpha=0.9)
        axc.annotate(lab, (m, s), textcoords="offset points",
                     xytext=(5, -7), fontsize=6)
    axc.set_xscale("log")
    axc.set_yscale("log")
    axc.set_xlim(lim)
    axc.set_ylim(lim)
    axc.set_xlabel("certificate: model clip-onset step [µm]")
    axc.set_ylabel("simulator: measured clip-onset step [µm]")
    axc.set_title("(c) certificate vs nonlinear simulator (0.02–1.9 %)",
                  fontsize=8.5)
    axc.legend(fontsize=6.5, loc="upper left")

    # ---------------- (d) island demonstration -----------------------
    axd.set_title("(d) saturation island: causal demonstration",
                  fontsize=8.5)
    shown = False
    if isl:
        for key, rec in isl["points"].items():
            if "ISLAND CONFIRMED" in str(rec.get("verdict", "")) and \
                    "island_trace_on" in rec:
                tr_on, tr_off = rec["island_trace_on"], rec["island_trace_off"]
                t0 = isl["settle_periods"] * 60.0 / (3 * isl["rpm"])
                t_on = np.array(tr_on["t"]) - t0
                t_off = np.array(tr_off["t"]) - t0
                axd.plot(t_off * 1e3, tr_off["w_um"], lw=0.7,
                         color="#4c72b0",
                         label="bound lifted → decays")
                axd.plot(t_on * 1e3, tr_on["w_um"], lw=0.7,
                         color="#c44e52",
                         label="±150 V active → sustained chatter")
                axd.axvline(0.0, color="k", lw=0.7, ls=":")
                axd.text(0.02, 0.03,
                         f"{key}\nlinearly stable ($\\rho$="
                         f"{rec['rho']:.3f}); step h="
                         f"{rec['island_h_um']:.0f} µm",
                         transform=axd.transAxes, fontsize=6.2, va="bottom")
                axd.set_xlabel("time after surface step [ms]")
                axd.set_ylabel("milling-point deflection [µm]")
                axd.legend(fontsize=6.5, loc="upper left")
                shown = True
                break
    if not shown:
        axd.text(0.5, 0.5, "no confirmed island trace available",
                 ha="center", va="center", transform=axd.transAxes)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"satcert_campaign.{ext}", dpi=250,
                    bbox_inches="tight")
    print("saved", OUT / "satcert_campaign.png")


if __name__ == "__main__":
    main()
