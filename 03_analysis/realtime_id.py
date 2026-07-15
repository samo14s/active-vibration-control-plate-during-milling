"""
realtime_id.py
==============
Real-time modal-frequency identification for the finishing sequence, via an ACTIVE
PIEZO PROBE. Rebuilds the persistent-excitation idea (documented finding: under
stable periodic cutting the tooth-passing force dominates the innovations, so a
passive innovation/least-squares identifier is BIASED — persistent excitation is
required) as a clean, self-contained estimator.

TWO PROBE MODES:
  (1) TRANSIT PROBE (primary, unbiased) — at each pass boundary the tool is not
      cutting (rapid transit / retract). The controller is idle, so the map from
      the injected probe voltage to the measured tip displacement is the OPEN-LOOP
      plant FRF directly (no controller inversion). A short band-limited chirp is
      injected, Y(f)/U(f) is formed, and the modal resonances are peak-picked and
      parabolically refined. This is the online analogue of an inter-pass tap test.
  (2) CUTTING-LINE MONITOR (secondary) — during cutting a low-amplitude multisine
      on non-tooth-harmonic DFT bins (window = integer number of tooth periods, so
      the forced harmonics leak exactly zero into the probe bins) gives closed-loop
      FRF samples used only to DETECT drift and trigger a re-probe; the unbiased
      frequency estimate always comes from the transit probe.

The estimator is deliberately model-light: it tracks the current modal frequencies
(the dominant, chatter-critical degrees of freedom — modes 1-2), which is exactly
what the downstream controller needs to re-tune its ESO/LQR internal model and
re-phase the harmonic resonant cancellers (HRC).
"""
import numpy as np
from newmark_solver import NewmarkSimulator


# =====================================================================
#  Transit probe (unbiased, open-loop)
# =====================================================================
def transit_probe_identify(plant, dt, ft, tau, f_lo=300.0, f_hi=1500.0,
                           T_probe=60e-3, amp=3.0, meas_noise_std=1e-8,
                           expect_freqs=None, n_modes_id=2, seed=7,
                           return_frf=False):
    """
    Inject a band-limited chirp through the piezo on the (non-cutting) plant and
    identify the dominant modal frequencies from Y(f)/U(f).

    plant : any solver-compatible plant snapshot (MillingWorkpiece.snapshot_plant()
            or PlateModel), carrying H_Pe_modal, D_obs, Mp/Kp/Cp, Dp_array.
    expect_freqs : optional (n_modes_id,) prior locations to disambiguate peaks.
    Returns f_hat (n_modes_id,) [and (f_grid, |G|) if return_frf].
    """
    sim = NewmarkSimulator(plant, dt=dt, T_end=T_probe, ft=ft, tau=tau,
                           verbose=False)
    nstep = sim.nstep
    t = sim.t_vec
    # linear chirp, unit-ish amplitude; controller = a pure open-loop voltage source
    k0 = f_lo
    k1 = (f_hi - f_lo) / T_probe
    phase = 2 * np.pi * (k0 * t + 0.5 * k1 * t**2)
    u_probe = amp * np.sin(phase)

    class _Probe:
        n_x = 2 * plant.n_modes
        def __init__(self): self._k = 0
        def step(self, x_prev, u_prev, y_meas):
            u = u_probe[min(self._k, nstep - 1)]
            self._k += 1
            return np.zeros(self.n_x), float(u)

    # tool parked at start (no cutting: alpha3 = alpha4 = 0)
    kp_idx = np.zeros(nstep, dtype=int)
    a3 = np.zeros(nstep)
    a4 = np.zeros(nstep)
    res = sim.simulate(a3, a4, kp_idx, controller=_Probe(),
                       meas_noise_std=meas_noise_std,
                       rng=np.random.default_rng(seed),
                       stop_threshold=1.0, progress=False)
    y = res['y']
    u = res['u']
    # discard the first tooth-period-ish transient, window the rest
    i0 = int(2e-3 / dt)
    yw = y[i0:] * np.hanning(len(y) - i0)
    uw = u[i0:] * np.hanning(len(u) - i0)
    NFFT = 1 << int(np.ceil(np.log2(len(yw))) + 1)
    Y = np.fft.rfft(yw, NFFT)
    U = np.fft.rfft(uw, NFFT)
    f = np.fft.rfftfreq(NFFT, dt)
    band = (f >= f_lo) & (f <= f_hi)
    G = np.zeros_like(f)
    good = band & (np.abs(U) > 1e-9 * np.abs(U[band]).max())
    G[good] = np.abs(Y[good]) / np.abs(U[good])

    f_hat = _pick_modes(f, G, band, n_modes_id, expect_freqs)
    if return_frf:
        return f_hat, (f[band], G[band])
    return f_hat


def _pick_modes(f, G, band, n_modes_id, expect_freqs):
    """Peak-pick |G| in-band, parabolically refine, return the n strongest peaks
    (optionally the peaks closest to `expect_freqs`)."""
    idx = np.where(band)[0]
    Gb = G[idx]
    # local maxima
    peaks = []
    for j in range(1, len(Gb) - 1):
        if Gb[j] > Gb[j - 1] and Gb[j] >= Gb[j + 1] and Gb[j] > 0:
            # parabolic interpolation on log-magnitude
            lm = np.log(Gb[j - 1:j + 2] + 1e-30)
            denom = (lm[0] - 2 * lm[1] + lm[2])
            delta = 0.5 * (lm[0] - lm[2]) / denom if abs(denom) > 1e-12 else 0.0
            fj = f[idx[j]] + delta * (f[idx[1]] - f[idx[0]])
            peaks.append((Gb[j], fj))
    if not peaks:
        return np.full(n_modes_id, np.nan)
    if expect_freqs is not None:
        out = []
        for fe in expect_freqs[:n_modes_id]:
            fj = min(peaks, key=lambda p: abs(p[1] - fe))[1]
            out.append(fj)
        return np.array(out)
    peaks.sort(reverse=True)             # by magnitude
    fs = sorted(p[1] for p in peaks[:n_modes_id])
    return np.array(fs)


# =====================================================================
#  Passive baseline (documented negative result)
# =====================================================================
def passive_frequency_estimate(y, dt, f_lo=300.0, f_hi=1500.0, n_modes_id=2,
                               expect_freqs=None):
    """Estimate modal frequencies from the CUTTING response spectrum alone (no
    probe). Under stable periodic cutting the spectrum is dominated by the tooth
    harmonics (h*f_tooth), NOT the modal resonances, so the peaks it returns are
    the forcing lines — the documented bias. Provided so the driver can show the
    probe-vs-passive contrast honestly."""
    N = len(y)
    yw = y * np.hanning(N)
    NFFT = 1 << int(np.ceil(np.log2(N)) + 1)
    Y = np.abs(np.fft.rfft(yw, NFFT))
    f = np.fft.rfftfreq(NFFT, dt)
    band = (f >= f_lo) & (f <= f_hi)
    G = np.zeros_like(f)
    G[band] = Y[band]
    return _pick_modes(f, G, band, n_modes_id, expect_freqs)


# =====================================================================
#  Anti-leakage cutting-line probe design (secondary monitor)
# =====================================================================
def design_probe_lines(f_tooth, dt, n_tooth_periods=25, n_lines=5,
                       f_lo=380.0, f_hi=680.0):
    """Choose probe frequencies that sit EXACTLY on DFT bins of a window spanning
    an integer number of tooth periods AND are not tooth harmonics — so the forced
    tooth-passing harmonics leak zero into the probe lines and vice-versa. Returns
    (freqs, window_len_samples, bin_indices)."""
    T_win = n_tooth_periods / f_tooth
    Nwin = int(round(T_win / dt))
    df = 1.0 / (Nwin * dt)                       # DFT bin spacing
    tooth_bins = set(int(round(h * f_tooth / df)) for h in range(1, 12))
    lo_b, hi_b = int(np.ceil(f_lo / df)), int(np.floor(f_hi / df))
    cand = [b for b in range(lo_b, hi_b + 1) if b not in tooth_bins]
    if len(cand) < n_lines:
        sel = cand
    else:
        sel = [cand[int(round(x))] for x in
               np.linspace(0, len(cand) - 1, n_lines)]
    freqs = np.array([b * df for b in sel])
    return freqs, Nwin, np.array(sel)
