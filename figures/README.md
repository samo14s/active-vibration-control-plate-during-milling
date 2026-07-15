# Publication figure set

Curated, publication-quality (**vector PDF, 300 DPI**) figures for the
LQG vs ESO-ADRC vs A-ESO-ADRC study. All are **regenerable** — this folder holds a
committed snapshot for convenience; the source of truth is the scripts.

## Provenance & how to regenerate

| File | Source script | Kind |
|---|---|---|
| `00_geometry_3D.pdf` | `04_figures/gen_geometry_figure.py` | setup (qualitative) |
| `01_summary_bar.pdf` | `05_main/main_simulation.py` | **authoritative** |
| `02_temporal_y.pdf` | `05_main/main_simulation.py` | **authoritative** |
| `04_fft_y.pdf` | `05_main/main_simulation.py` | **authoritative** |
| `05_rung_supervision.pdf` | `05_main/main_simulation.py` | **authoritative** (A-ESO-ADRC rung traces, 4 scenarios) |
| `06_certification.pdf` | `05_main/main_simulation.py` | **authoritative** (design-time Floquet map — the case for the ladder) |
| `07_SLD_3panels.pdf` | `05_main/main_simulation.py` | **authoritative** (closed-loop monodromy, worst of 3 positions) |
| `08_SLD_overlay.pdf` | `05_main/main_simulation.py` | **authoritative** (closed-loop monodromy) |
| `09_robustness_montecarlo.pdf` | `05_main/main_robustness_mc.py` | **authoritative** |
| `12_adaptive_removal.pdf` | `05_main/main_adaptive_removal.py` | **authoritative** (drift benchmark + rung supervision) |

```bash
# From a flat working dir (copy all package .py into one folder, see top-level README):
python main_simulation.py          # -> figs_lqg_vs_adrc/*.pdf  (authoritative)
python main_robustness_mc.py       # -> figs_lqg_vs_adrc/fig_robustness_mc.pdf
python main_adaptive_removal.py    # -> figs_lqg_vs_adrc/fig_adaptive_removal.pdf
python gen_geometry_figure.py      # -> figs_article_publication/fig00*_geometry*.pdf
python gen_SLD_academic_style.py   # styled SLD variants (same rigorous monodromy)
```

## Numbers

All **authoritative** figures use the final model: 5-mode plant with 3-mode
controllers (spillover), 10 nm measurement noise, corrected Eq. (3) forces,
Eq. (15) piezo coupling, rigorous closed-loop monodromy SLD at the worst of 3 tool
positions, design-grid + Floquet-certification selection of the ESO-ADRC designs.
Headline numbers (full log: `docs/REPRODUCED_RESULTS.md`):

- Held-out scenarios: LQG best inside its envelope (S1 0.777 µm vs ESO-ADRC 0.826 /
  A-ESO-ADRC 0.783); fixed designs each have one failure mode (LQG diverges beyond
  −10 % drift; certified ESO rings bounded at −8 %; performance ESO diverges at
  a_p = 0.6 mm).
- Drift benchmark: at −12 % drift (static or ramped during the pass) the fixed LQG
  DIVERGES while ESO-ADRC rides through at ~0.9–1.1 µm; **A-ESO-ADRC never
  diverges in any of the 9 scenarios**.
- Monte-Carlo (±3 % freq, ±15 % cutting, 50 samples): all controllers 50/50
  converged; LQG median 0.788 µm best — the ESO family's advantage lives beyond
  the LQG envelope, not inside it.
- SLD @4900 RPM: OL 0.100 mm (= article experiment), LQG 1.075 mm, ESO-ADRC
  certified 0.913 mm (also A-ESO-ADRC's panic-fallback boundary).
