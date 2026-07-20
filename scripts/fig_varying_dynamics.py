"""Figure: the scheduling-parameter dependence that motivates LPV control.

(a) modal participation factors b_f,i(x_T) of the milling point along the
    feed path (position dependence);
(b) milling-point receptance magnitude at the dominant modes vs position;
(c) natural-frequency drift with material removal (multi-pass);
(d) time-scale separation: |d theta/dt| vs plant dynamics (text box).
"""
import numpy as np

from common import DC, cached, models, savefig
import matplotlib.pyplot as plt

from avc.modal import bf_sweep, reduce_model
from avc.params import DEFAULT


def main():
    mdl = models(removals=(0.0,))[0.0]
    mm3 = mdl["design"]
    xs, B = bf_sweep(mm3, n_pos=61)

    def removal_build():
        recs = np.linspace(0.0, DEFAULT.design.removal_total, 6)
        fr = []
        for r in recs:
            mm = reduce_model(DEFAULT, 3, height_removed=r)
            fr.append(mm.freqs_hz.copy())
        return recs, np.array(fr)

    recs, fr = cached("removal_freqs", removal_build)

    fig, axs = plt.subplots(1, 3, figsize=(DC, 2.3))

    ax = axs[0]
    for i in range(3):
        ax.plot(xs * 1e3, B[:, i], label=f"mode {i+1}")
    ax.set_xlabel(r"tool position $x_T$ [mm]")
    ax.set_ylabel(r"$b_{f,i}(x_T)$ [kg$^{-1/2}$]")
    ax.set_title("(a) modal participation along path", fontsize=8.5)
    ax.legend()

    ax = axs[1]
    # static milling-point compliance per mode and total
    comp = (B ** 2) / (mm3.omega[None, :] ** 2)
    ax.plot(xs * 1e3, comp.sum(axis=1) * 1e6, "k", label="total")
    for i in range(3):
        ax.plot(xs * 1e3, comp[:, i] * 1e6, "--", alpha=0.7,
                label=f"mode {i+1}")
    ax.set_xlabel(r"tool position $x_T$ [mm]")
    ax.set_ylabel(r"static compliance [$\mu$m/N]")
    ax.set_title("(b) milling-point compliance", fontsize=8.5)
    ax.legend(fontsize=6.5)

    ax = axs[2]
    for i in range(3):
        ax.plot(recs * 1e3, (fr[:, i] / fr[0, i] - 1) * 100, "-o", ms=3,
                label=f"mode {i+1}")
    ax.set_xlabel("height recession [mm]")
    ax.set_ylabel("frequency shift [%]")
    ax.set_title("(c) material-removal drift", fontsize=8.5)
    ax.legend()

    fig.tight_layout()
    savefig(fig, "fig02_varying_dynamics")

    ratio = B[:, :3].max(axis=0) / np.maximum(np.abs(B[:, :3]).min(axis=0), 1e-12)
    print("bf variation range (max/min |bf|):", np.round(ratio, 2))
    print("compliance variation total: x",
          round(float(comp.sum(1).max() / comp.sum(1).min()), 2))


if __name__ == "__main__":
    main()
