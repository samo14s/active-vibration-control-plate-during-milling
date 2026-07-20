"""Figure 13: FRF-level validation against the digitized measured curves
of the reference rig (Fig. 12 of Du et al. 2024; data/ *.csv).

(a) receptance at the right-upper corner: measured points vs model,
    with a single fitted backbone (off-resonance) gain offset - the
    digitized axis origin of the receptance plot carries a reference
    ambiguity, so only the frequency-dependent STRUCTURE is compared;
(b) voltage -> displacement: measured vs model in ABSOLUTE units
    (no fitting): the low-frequency levels agree within ~35 % with
    handbook piezo constants; digitized resonance tips are lower
    bounds (a 0.2-0.6 % damping peak is a few Hz wide - unresolvable
    from a plot).

Prints the resonance-frequency comparison table used in the manuscript.
"""
import numpy as np

import matplotlib.pyplot as plt
from common import DC, ROOT, models, savefig

from avc.params import DEFAULT


def model_receptance(mm, freqs):
    p = DEFAULT
    e = mm.Phi.T @ mm.model.interp_w(p.sensor.xs, p.sensor.zs)
    w = 2 * np.pi * freqs
    H = np.zeros(len(freqs), dtype=complex)
    for i in range(mm.n_modes):
        H += e[i] * e[i] / (mm.omega[i] ** 2 - w ** 2
                            + 2j * mm.zeta[i] * mm.omega[i] * w)
    return H


def pick_peaks(f, m, prom=6.0):
    out = []
    for i in range(1, len(m) - 1):
        if m[i] >= m[i - 1] and m[i] >= m[i + 1] \
                and m[i] - min(m[max(0, i - 4):i + 5]) > prom:
            out.append((f[i], m[i]))
    return out


def main():
    rec = np.genfromtxt(ROOT / "data" / "measured_frf_receptance.csv",
                        delimiter=",", skip_header=1)
    vlt = np.genfromtxt(
        ROOT / "data" / "measured_frf_voltage_to_displacement.csv",
        delimiter=",", skip_header=1)
    mm12 = models(removals=(0.0,))[0.0]["full"]
    freqs = np.linspace(20, 5100, 6000)

    # ---- (a) receptance: fitted backbone offset -----------------------
    Hr = model_receptance(mm12, freqs)
    Hr_db = 20 * np.log10(np.abs(Hr) / 1e-6)          # dB re 1 um/N
    interp = np.interp(rec[:, 0], freqs, Hr_db)
    # backbone points: exclude vicinity of the five resonances
    f_res = np.array([536, 1072, 2779, 3359, 4123], float)
    mask = np.all(np.abs(rec[:, 0][:, None] - f_res[None, :]) > 120.0,
                  axis=1)
    offset = float(np.median(rec[mask, 1] - interp[mask]))

    # ---- (b) voltage FRF: absolute ------------------------------------
    Hv = mm12.frf_voltage(freqs, DEFAULT.sensor.xs, DEFAULT.sensor.zs)
    Hv_db = 20 * np.log10(np.abs(Hv) / 0.01e-6)       # dB re 0.01 um/V

    fig, axs = plt.subplots(1, 2, figsize=(DC, 2.8))
    ax = axs[0]
    ax.plot(rec[:, 0], rec[:, 1], "o", ms=2.5, color="#2ca02c",
            label="measured (digitized, ref. Fig. 12a)")
    ax.plot(freqs, Hr_db + offset, "k", lw=0.9,
            label=f"model (+{offset:.1f} dB backbone offset)")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"receptance [dB re 1 $\mu$m/N]")
    ax.set_title("(a) impact receptance, structure match", fontsize=8.5)
    ax.legend(fontsize=6)

    ax = axs[1]
    ax.plot(vlt[:, 0], vlt[:, 1], "o", ms=2.5, color="#2ca02c",
            label="measured (digitized, ref. Fig. 12b)")
    ax.plot(freqs, Hv_db, "k", lw=0.9, label="model (absolute, no fit)")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"$|w_s/u|$ [dB re 0.01 $\mu$m/V]")
    ax.set_title("(b) piezo transfer function, absolute", fontsize=8.5)
    ax.legend(fontsize=6)

    fig.tight_layout()
    savefig(fig, "fig13_frf_match")

    # ---- console metrics ---------------------------------------------
    print(f"receptance backbone offset: {offset:+.1f} dB")
    dc_meas = 10 ** (vlt[0, 1] / 20) * 0.01           # um/V
    dc_model = 10 ** (Hv_db[0] / 20) * 0.01
    print(f"voltage-FRF low-freq level: measured {dc_meas:.4f} um/V, "
          f"model {dc_model:.4f} um/V "
          f"({20*np.log10(dc_model/dc_meas):+.1f} dB)")
    fm = mm12.freqs_hz[:5]
    for nm, dat in (("receptance", rec), ("voltage", vlt)):
        pk = [p for p in pick_peaks(dat[:, 0], dat[:, 1]) if p[1] > 5]
        print(f"{nm} digitized resonances vs model:")
        for (fd, md), fmod in zip(pk, fm):
            print(f"  {fd:7.0f} Hz  (model {fmod:7.1f} Hz, "
                  f"{(fmod/fd-1)*100:+.1f} %)")


if __name__ == "__main__":
    main()
