"""Performance metrics (Sec. 4.1 evaluation methodology).

* 0.1 s sliding-window RMS vibration
* material removal rate / removed volume
* empirical surface-roughness correlation (Sec. 3.2)
* wall-thickness deviation from static deflection + vibration marks
* tooth-passing spectral coherence (chatter indicator, < 0.7 = chatter)
* specific cutting energy from tangential force x cutting speed
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .engine import IntervalMeasurement
from .parameters import PaperSetup


@dataclass
class RunHistory:
    """Per-control-interval time series for one machining run."""
    t: list = field(default_factory=list)
    rms_um: list = field(default_factory=list)
    n_rpm: list = field(default_factory=list)
    vf: list = field(default_factory=list)
    ap: list = field(default_factory=list)
    mrr: list = field(default_factory=list)
    Vr: list = field(default_factory=list)
    fy: list = field(default_factory=list)
    ftan: list = field(default_factory=list)
    ra_um: list = field(default_factory=list)
    thick_dev_mm: list = field(default_factory=list)
    fn_true: list = field(default_factory=list)
    force_chunks: list = field(default_factory=list)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self.__dict__.items()
                if k != "force_chunks"}


class MetricsCollector:
    def __init__(self, setup: PaperSetup):
        self.setup = setup
        self.h = RunHistory()
        self._rms_buf: list[float] = []
        self._win = max(1, int(round(setup.sim.rms_window_s
                                     * setup.controller.update_rate_hz)))

    def update(self, meas: IntervalMeasurement, fn_true_hz: float,
               k_wall_N_m: float) -> None:
        h = self.h
        self._rms_buf.append(meas.rms_wall_um)
        if len(self._rms_buf) > self._win:
            self._rms_buf.pop(0)
        rms = float(np.sqrt(np.mean(np.square(self._rms_buf))))

        ra = self.setup.roughness.ra_um(meas.ft_mm, rms)

        # wall thickness deviation: quasi-static deflection under the mean
        # normal force plus the fraction of the vibration amplitude that
        # prints onto the finished wall (tooth-phase averaging halves it)
        static_mm = abs(meas.mean_fy_N) / k_wall_N_m * 1000.0
        vib_mm = 0.5 * math.sqrt(2.0) * rms * 1e-3
        thick_dev = static_mm + vib_mm

        h.t.append(meas.t)
        h.rms_um.append(rms)
        h.n_rpm.append(meas.n_rpm)
        h.vf.append(meas.vf_mm_min)
        h.ap.append(meas.ap_mm)
        h.mrr.append(meas.mrr_cm3_min)
        h.Vr.append(meas.Vr)
        h.fy.append(meas.mean_fy_N)
        h.ftan.append(meas.mean_ft_N)
        h.ra_um.append(ra)
        h.thick_dev_mm.append(thick_dev)
        h.fn_true.append(fn_true_hz)
        h.force_chunks.append(meas.force_samples)


def tooth_pass_coherence(force: np.ndarray, fs: float,
                         n_rpm: float, n_teeth: int) -> float:
    """Correlation between successive tooth-passing periods (Sec. 4.1:
    values below 0.7 indicate regenerative chatter onset)."""
    period = 60.0 / (n_rpm * n_teeth)
    npp = int(round(period * fs))
    if npp < 4 or len(force) < 3 * npp:
        return 1.0
    nseg = len(force) // npp
    segs = force[: nseg * npp].reshape(nseg, npp)
    segs = segs - segs.mean(axis=1, keepdims=True)
    cc = []
    for a, b in zip(segs[:-1], segs[1:]):
        da, db = np.linalg.norm(a), np.linalg.norm(b)
        if da > 1e-12 and db > 1e-12:
            cc.append(float(a @ b / (da * db)))
    return float(np.mean(cc)) if cc else 1.0


def specific_energy_W_min_cm3(ftan_N: np.ndarray, n_rpm: np.ndarray,
                              mrr_cm3_min: np.ndarray,
                              tool_diameter_m: float) -> float:
    """Mean specific cutting energy: P = Ft_sum * Vc, E = P / MRR."""
    Vc = math.pi * tool_diameter_m * np.asarray(n_rpm) / 60.0   # m/s
    P = np.asarray(ftan_N) * Vc                                 # W
    mrr = np.maximum(np.asarray(mrr_cm3_min), 1e-9)
    return float(np.mean(P / mrr))


def summarize(h: RunHistory, setup: PaperSetup) -> dict:
    a = h.as_arrays()
    dt_c = 1.0 / setup.controller.update_rate_hz
    removed_cm3 = float(np.sum(a["mrr"]) * dt_c / 60.0)
    # machining time normalised to the full removal volume of the pass
    t_per_volume_min = (len(a["t"]) * dt_c / 60.0) / max(removed_cm3, 1e-9) \
        * setup.sim.removal_volume_cm3

    coh = []
    n_teeth = setup.tool.n_teeth
    fs = setup.sim.force_sample_hz
    step = max(1, len(h.force_chunks) // 80)
    for idx in range(0, len(h.force_chunks) - 5, step):
        force = np.concatenate(h.force_chunks[idx: idx + 5])
        coh.append(tooth_pass_coherence(force, fs, a["n_rpm"][idx], n_teeth))
    coh = np.asarray(coh) if coh else np.array([1.0])

    return {
        "avg_mrr_cm3_min": float(np.mean(a["mrr"])),
        # sustained peak (98th percentile): transient 10 ms spikes during
        # coordinated ap/vf moves are not a meaningful "peak MRR"
        "peak_mrr_cm3_min": float(np.percentile(a["mrr"], 98.0)),
        "removed_cm3": removed_cm3,
        "t_per_volume_min": t_per_volume_min,
        "rms_mean_um": float(np.mean(a["rms_um"])),
        "rms_max_um": float(np.max(a["rms_um"])),
        "pct_time_above_25um": float(np.mean(a["rms_um"] > 25.0) * 100.0),
        "ra_mean_um": float(np.mean(a["ra_um"])),
        "ra_min_um": float(np.min(a["ra_um"])),
        "ra_max_um": float(np.max(a["ra_um"])),
        "ra_std_um": float(np.std(a["ra_um"])),
        "thick_dev_max_mm": float(np.max(a["thick_dev_mm"])),
        "thick_dev_mean_mm": float(np.mean(a["thick_dev_mm"])),
        "coherence_mean": float(np.mean(coh)),
        "coherence_min": float(np.min(coh)),
        "pct_coherence_below_07": float(np.mean(coh < 0.7) * 100.0),
        "specific_energy_W_min_cm3": specific_energy_W_min_cm3(
            a["ftan"], a["n_rpm"], a["mrr"], setup.tool.diameter_m),
        "final_Vr": float(a["Vr"][-1]),
    }
