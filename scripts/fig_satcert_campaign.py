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
        fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, -0.13),
        ncol=3, framealpha=0.95)

    # ---------------- (b) h_req sensitivity --------------------------
    hs = (1, 2, 5, 10, 20, 50)
    worst = {s: [camp["hreq_sensitivity"][f"{s}|{h}um"]["worst_mm"]
                 for h in hs] for s in ("FROZEN", "PS-LPV")}
    for s, c, mk in (("FROZEN", "#4c72b0", "o"), ("PS-LPV", "#c44e52", "s")):
        axb.plot(hs, worst[s], marker=mk, ms=4, lw=1.4, color=c,
                 label=("best frozen H$_\\infty$" if s == "FROZEN"
                        else "PS-LPV"))
    axb.set_xscale("log")
    axb.set_xticks(hs)
    axb.set_xticklabels([str(h) for h in hs])
    axb.set_xlabel("declared surface-step tolerance $h_{req}$ [µm]")
    axb.set_ylabel("worst-position certified depth [mm]")
    axb.set_title("(b) certified depth vs declared tolerance", fontsize=8.5)
    y1 = worst["PS-LPV"][0] / worst["FROZEN"][0]
    y20 = worst["PS-LPV"][hs.index(20)] / worst["FROZEN"][hs.index(20)]
    axb.annotate(f"{y1:.1f}×", xy=(1, worst["PS-LPV"][0]),
                 xytext=(1.5, worst["PS-LPV"][0] * 0.87), fontsize=7,
                 arrowprops=dict(arrowstyle="-", lw=0.6))
    axb.annotate(f"{y20:.2f}×\n(ranking inverted)",
                 xy=(20, worst["PS-LPV"][hs.index(20)]),
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
    devs = [abs(s / m - 1.0) * 100 for m, s, _ in pts if s]
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
    axc.set_title(f"(c) certificate vs nonlinear simulator "
                  f"({min(devs):.2g}–{max(devs):.2g} %)", fontsize=8.5)
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
                tau = 60.0 / (3 * isl["rpm"])
                t0 = isl["settle_periods"] * tau

                def envelope(tr):
                    t = np.array(tr["t"]) - t0
                    w = np.abs(np.array(tr["w_um"]))
                    nb = max(1, int((t[-1] - t[0]) / (2 * tau)))
                    edges = np.linspace(t[0], t[-1], nb + 1)
                    idx = np.clip(np.searchsorted(edges, t) - 1, 0, nb - 1)
                    env = np.zeros(nb)
                    np.maximum.at(env, idx, w)
                    mid = 0.5 * (edges[:-1] + edges[1:])
                    keep = mid >= -0.15
                    return mid[keep] * 1e3, np.maximum(env[keep], 1e-4)

                for tr, c, lab in ((tr_off, "#4c72b0",
                                    "bound lifted → decays"),
                                   (tr_on, "#c44e52",
                                    "±150 V active → chatter growth "
                                    "(clip duty 99 %)")):
                    tm, env = envelope(tr)
                    axd.semilogy(tm, env, lw=1.2, color=c, label=lab)
                axd.axvline(0.0, color="k", lw=0.7, ls=":")
                axd.axhspan(1e3, 1e7, color="0.85", alpha=0.5, zorder=0)
                axd.text(0.35, 0.93, "beyond model amplitude validity",
                         transform=axd.transAxes, fontsize=6.2,
                         color="0.35", va="top")
                axd.text(0.98, 0.32,
                         f"{key}\nlinearly stable ($\\rho$="
                         f"{rec['rho']:.3f}); step h="
                         f"{rec['island_h_um']:.0f} µm",
                         transform=axd.transAxes, fontsize=6.2,
                         va="bottom", ha="right")
                axd.set_xlabel("time after surface step [ms]")
                axd.set_ylabel("deflection envelope |w| [µm]")
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
