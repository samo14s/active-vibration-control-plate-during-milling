"""Figure: FEM model validation.

(a) sensor-point receptance FRF (design vs full model) with measured
    natural frequencies of the reference paper marked;
(b) voltage-to-sensor-displacement FRF (actuator authority);
(c) bar comparison of natural frequencies: this FEM vs measured vs the
    reference paper's own model.
"""
import numpy as np

from common import DC, models, savefig
import matplotlib.pyplot as plt

from avc.params import DEFAULT


def frf_force(mm, freqs, x_in, z_in, x_out, z_out):
    e_in = mm.Phi.T @ mm.model.interp_w(x_in, z_in)
    e_out = mm.Phi.T @ mm.model.interp_w(x_out, z_out)
    w = 2 * np.pi * freqs
    H = np.zeros(len(freqs), dtype=complex)
    for i in range(mm.n_modes):
        H += e_out[i] * e_in[i] / (mm.omega[i] ** 2 - w ** 2
                                   + 2j * mm.zeta[i] * mm.omega[i] * w)
    return H


def main():
    mdl = models(removals=(0.0,))[0.0]
    mm3, mm12 = mdl["design"], mdl["full"]
    p = DEFAULT
    xs, zs = p.sensor.xs, p.sensor.zs

    freqs = np.linspace(50, 5000, 4000)
    Hf3 = frf_force(mm3, freqs, xs, zs, xs, zs)
    Hf12 = frf_force(mm12, freqs, xs, zs, xs, zs)
    Hv12 = mm12.frf_voltage(freqs, xs, zs)

    fig, axs = plt.subplots(1, 3, figsize=(DC, 2.3))

    ax = axs[0]
    ax.semilogy(freqs, np.abs(Hf12) * 1e6, "k", label="full model ($n=12$)")
    ax.semilogy(freqs, np.abs(Hf3) * 1e6, "--", color="#1f77b4",
                label="design model ($n=3$)")
    for f in p.plate.f_measured:
        ax.axvline(f, color="#d62728", alpha=0.35, lw=0.8)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"receptance [$\mu$m/N]")
    ax.set_title("(a) tip receptance", fontsize=8.5)
    ax.legend(loc="upper right")

    ax = axs[1]
    ax.semilogy(freqs, np.abs(Hv12) * 1e6, "k")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"$|w_s/u|$ [$\mu$m/V]")
    ax.set_title("(b) actuator authority", fontsize=8.5)

    ax = axs[2]
    idx = np.arange(5)
    width = 0.27
    ax.bar(idx - width, mm12.freqs_hz[:5], width, label="present FEM",
           color="#1f77b4")
    ax.bar(idx, p.plate.f_measured, width, label="measured (ref.)",
           color="#2ca02c")
    ax.bar(idx + width, p.plate.f_theoretical, width,
           label="ref. model", color="#d62728")
    ax.set_xticks(idx, [f"{i+1}" for i in idx])
    ax.set_xlabel("mode")
    ax.set_ylabel("frequency [Hz]")
    ax.set_title("(c) natural frequencies", fontsize=8.5)
    ax.legend(fontsize=6.5)

    fig.tight_layout()
    savefig(fig, "fig01_model_validation")

    err = (mm12.freqs_hz[:5] - np.array(p.plate.f_measured)) \
        / np.array(p.plate.f_measured) * 100
    print("freq errors vs measured [%]:", np.round(err, 2))


if __name__ == "__main__":
    main()
