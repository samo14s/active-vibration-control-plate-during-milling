"""Publication-style figures reproducing the paper's Figs. 3-4 plus the
stability-lobe, identification and spectrum diagnostics.

Colour assignments are fixed per strategy (colour follows the entity) and
use a colourblind-safe palette (Okabe-Ito subset, validated for CVD
separation): adaptive #0072B2, conventional #D55E00, SSV #009E73.
"""

from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from .parameters import PaperSetup
from .stability import fast_lobe_envelope

COLORS = {"adaptive": "#0072B2", "conventional": "#D55E00",
          "ssv": "#009E73"}
LABELS = {"adaptive": "Adaptive control", "conventional": "Conventional",
          "ssv": "SSV baseline"}
GRAY = "#666666"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9.5,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})


def _band(ax, histories, key, color, label, n_pts: int = 400,
          scale: float = 1.0, floor: float | None = 0.0,
          smooth_s: float = 0.3):
    """Mean +/- 1 SD band over trials on a common time grid.

    A short rolling mean (default 0.3 s) removes within-trial modulation
    (e.g. the 5 Hz SSV wiggle) for readability; the summary statistics in
    the report are computed on the raw series.
    """
    t_end = min(h.t[-1] for h in histories)
    tg = np.linspace(0.0, t_end, n_pts)
    Y = np.vstack([np.interp(tg, h.t, np.asarray(getattr(h, key)) * scale)
                   for h in histories])
    if smooth_s > 0.0:
        k = max(1, int(round(smooth_s / (tg[1] - tg[0]))))
        kern = np.ones(k) / k
        Y = np.vstack([np.convolve(y, kern, mode="same") for y in Y])
    mu, sd = Y.mean(axis=0), Y.std(axis=0)
    lo = mu - sd if floor is None else np.maximum(mu - sd, floor)
    ax.plot(tg, mu, color=color, lw=1.8, label=label)
    ax.fill_between(tg, lo, mu + sd, color=color, alpha=0.18, lw=0)
    return tg, mu


def fig3_performance(campaign: dict, path: str) -> None:
    """Paper Fig. 3: (a) vibration RMS, (b) MRR, (c) spindle speed."""
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=False)

    ax = axes[0]
    for name in ("conventional", "ssv", "adaptive"):
        _band(ax, campaign[name]["histories"], "rms_um",
              COLORS[name], LABELS[name])
    ax.axhline(25.0, color=GRAY, ls="--", lw=1.0)
    ax.annotate("25 μm stability threshold", xy=(0.99, 27),
                xycoords=("axes fraction", "data"), ha="right",
                fontsize=8, color=GRAY)
    ax.set_ylabel("RMS vibration (μm)")
    ax.set_title("(a) Vibration amplitude, mean ± 1 SD (n = 5 trials)",
                 loc="left", fontsize=10)
    ax.legend(loc="upper left", ncols=3)

    ax = axes[1]
    for name in ("conventional", "ssv", "adaptive"):
        _band(ax, campaign[name]["histories"], "mrr",
              COLORS[name], LABELS[name])
    ax.set_ylabel("MRR (cm³/min)")
    ax.set_title("(b) Material removal rate", loc="left", fontsize=10)

    ax = axes[2]
    for name in ("conventional", "ssv", "adaptive"):
        _band(ax, campaign[name]["histories"], "n_rpm",
              COLORS[name], LABELS[name], scale=1e-3)
    ax.set_ylabel("Spindle speed (krpm)")
    ax.set_xlabel("Machining time (s), time-compressed pass")
    ax.set_title("(c) Spindle speed commanded by each strategy",
                 loc="left", fontsize=10)

    fig.savefig(path)
    plt.close(fig)


def fig4_quality(campaign: dict, setup: PaperSetup, path: str) -> None:
    """Paper Fig. 4: surface roughness and wall-thickness comparison,
    mapped onto position along the 200 mm workpiece (progress ~ Vr)."""
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0))

    def _pos_band(ax, histories, key, color, label, scale=1.0):
        xg = np.linspace(0.0, 200.0, 300)
        Y = []
        for h in histories:
            pos = np.asarray(h.Vr) * 200.0
            Y.append(np.interp(xg, pos, np.asarray(getattr(h, key)) * scale))
        Y = np.vstack(Y)
        ax.plot(xg, Y.mean(0), color=color, lw=1.8, label=label)
        ax.fill_between(xg, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0),
                        color=color, alpha=0.18, lw=0)

    ax = axes[0]
    for name in ("conventional", "adaptive"):
        _pos_band(ax, campaign[name]["histories"], "ra_um", COLORS[name],
                  LABELS[name])
    ax.axhspan(0.9, 1.3, color="#0072B2", alpha=0.06, lw=0)
    ax.annotate("paper band 0.9–1.3 μm", xy=(2, 1.32),
                fontsize=8, color=GRAY)
    ax.set_ylabel("Ra (μm)")
    ax.set_title("(a) Surface roughness along the wall, mean ± 1 SD",
                 loc="left", fontsize=10)
    ax.legend(loc="upper left", ncols=2)

    ax = axes[1]
    bins = np.linspace(0.5, 5.0, 46)
    for name in ("conventional", "adaptive"):
        ra_all = np.concatenate(
            [h.ra_um for h in campaign[name]["histories"]])
        ax.hist(ra_all, bins=bins, density=True, alpha=0.55,
                color=COLORS[name], label=LABELS[name], edgecolor="none")
    ax.set_xlabel("Ra (μm)")
    ax.set_ylabel("Density")
    ax.set_title("(b) Roughness distribution (71 % variation reduction "
                 "reported in the paper)", loc="left", fontsize=10)
    ax.legend()

    ax = axes[2]
    for name in ("conventional", "adaptive"):
        _pos_band(ax, campaign[name]["histories"], "thick_dev_mm",
                  COLORS[name], LABELS[name])
    ax.axhline(0.02, color=COLORS["adaptive"], ls=":", lw=1.0)
    ax.axhline(0.08, color=COLORS["conventional"], ls=":", lw=1.0)
    ax.annotate("±0.02 mm (paper, adaptive)", xy=(2, 0.023),
                fontsize=8, color=COLORS["adaptive"])
    ax.annotate("±0.08 mm (paper, conventional)", xy=(2, 0.083),
                fontsize=8, color=COLORS["conventional"])
    ax.set_ylabel("Thickness deviation (mm)")
    ax.set_xlabel("Position along workpiece (mm)")
    ax.set_title("(c) Wall-thickness deviation", loc="left", fontsize=10)

    fig.savefig(path)
    plt.close(fig)


def fig_lobes(campaign: dict, setup: PaperSetup, path: str) -> None:
    """Stability-lobe evolution with the operating trajectories."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    grid = np.linspace(6000, 20000, 561)
    shades = ["#bbbbbb", "#999999", "#777777", "#444444"]
    for shade, Vr in zip(shades, (0.0, 0.3, 0.6, 0.9)):
        modes = [(m.wn(Vr) / (2 * math.pi), m.zeta(Vr), m.mass_kg)
                 for m in setup.modes]
        dirs = [m.direction for m in setup.modes]
        env = fast_lobe_envelope(setup.tool, setup.engagement, modes, grid,
                                 directions=dirs) * 1000.0
        ax.plot(grid / 1000.0, np.minimum(env, 12.0), color=shade, lw=1.4,
                label=f"boundary at Vr = {Vr:.1f}")

    h = campaign["adaptive"]["histories"][0]
    n = np.asarray(h.n_rpm) / 1000.0
    ap = np.asarray(h.ap)
    sc = ax.scatter(n[::20], ap[::20], c=np.asarray(h.t)[::20],
                    cmap="viridis", s=9, zorder=5,
                    label="adaptive trajectory (colour = time)")
    fig.colorbar(sc, ax=ax, label="time (s)", pad=0.01)
    b = setup.baseline
    ax.scatter([b.n_rpm / 1000.0], [b.ap_mm], marker="X", s=90,
               color=COLORS["conventional"], zorder=6,
               label="fixed parameters (12 krpm, 4 mm)")
    ax.set_xlabel("Spindle speed (krpm)")
    ax.set_ylabel("Axial depth of cut (mm)")
    ax.set_ylim(0, 12)
    ax.set_title("Stability boundary shrinks and shifts as material is "
                 "removed (Eqs. (6)-(12))", loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def fig_identification(campaign: dict, path: str) -> None:
    """Real-time identification tracking (Sec. 4.2: within 3.2 %)."""
    ctl = campaign["adaptive"]["controllers"][0]
    lg = ctl.log
    t = np.asarray(lg.t)
    f_id = np.asarray(lg.fn_id, float)
    f_tr = np.asarray(lg.fn_true, float)
    ok = np.isfinite(f_id) & np.isfinite(f_tr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True,
                                   height_ratios=[2.2, 1.0])
    ax1.plot(t[ok], f_tr[ok], color="#333333", lw=1.8, label="true wall mode")
    ax1.plot(t[ok], f_id[ok], color=COLORS["adaptive"], lw=1.4, ls="--",
             label="identified (RLS + Eq. (6) fusion)")
    ax1.set_ylabel("Natural frequency (Hz)")
    ax1.set_title("Real-time tracking of the dominant wall mode",
                  loc="left", fontsize=10)
    ax1.legend()

    err = np.abs(f_id[ok] - f_tr[ok]) / f_tr[ok] * 100.0
    ax2.plot(t[ok], err, color=COLORS["adaptive"], lw=1.2)
    ax2.axhline(3.2, color=GRAY, ls="--", lw=1.0)
    ax2.annotate("3.2 % (paper)", xy=(0.99, 3.5),
                 xycoords=("axes fraction", "data"), ha="right",
                 fontsize=8, color=GRAY)
    ax2.set_ylabel("|error| (%)")
    ax2.set_xlabel("Machining time (s)")
    ax2.set_ylim(0, max(6.0, float(np.percentile(err, 99)) * 1.2))
    fig.savefig(path)
    plt.close(fig)


def fig_spectrum(campaign: dict, setup: PaperSetup, path: str) -> None:
    """Force spectra: stable adaptive cut vs conventional chatter window."""
    fs = setup.sim.force_sample_hz
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    for name in ("conventional", "adaptive"):
        h = campaign[name]["histories"][0]
        rms = np.asarray(h.rms_um)
        idx = int(np.argmax(rms)) if name == "conventional" \
            else int(np.argmin(np.abs(np.asarray(h.t) - 0.6 * h.t[-1])))
        lo = max(0, idx - 25)
        sig = np.concatenate(h.force_chunks[lo: lo + 50])
        f, P = welch(sig - sig.mean(), fs=fs, nperseg=1024)
        ax.semilogy(f, P, color=COLORS[name], lw=1.3,
                    label=f"{LABELS[name]} "
                          f"(n = {h.n_rpm[idx]:.0f} rpm)")
        if name == "conventional":
            f_tooth = h.n_rpm[idx] / 60.0 * setup.tool.n_teeth
            for k in range(1, 4):
                ax.axvline(k * f_tooth, color=GRAY, ls=":", lw=0.8)
            ax.annotate("tooth-passing\nharmonics", xy=(f_tooth * 1.02, 2e-4),
                        fontsize=8, color=GRAY, va="bottom")

    ax.set_xlim(0, 2400)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Force PSD (N²/Hz)")
    ax.set_title("Chatter appears as nonharmonic content near the wall "
                 "mode; adaptive cutting stays harmonic", loc="left",
                 fontsize=10)
    ax.legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def fig_learning(part_summaries: list[dict], path: str) -> None:
    """Cross-part learning demo: MRR across successive identical parts."""
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    mrr = [s["avg_mrr_cm3_min"] for s in part_summaries]
    parts = np.arange(1, len(mrr) + 1)
    ax.plot(parts, mrr, "o-", color=COLORS["adaptive"], lw=1.8, ms=6)
    for p, m in zip(parts, mrr):
        ax.annotate(f"{m:.1f}", xy=(p, m), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8)
    gain = 100.0 * (mrr[-1] / mrr[0] - 1.0)
    ax.set_xlabel("Workpiece number (identical parts)")
    ax.set_ylabel("Average MRR (cm³/min)")
    ax.set_xticks(parts)
    ax.set_title(f"GP boundary learning across parts: {gain:+.1f}% MRR "
                 "(paper reports +8 % by the later parts)", loc="left",
                 fontsize=10)
    fig.savefig(path)
    plt.close(fig)
