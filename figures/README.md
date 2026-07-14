# Publication figure set

Curated, publication-quality (**vector PDF, 300 DPI**) figures for the LQG vs PALF-LQG
study. All are **regenerable** — this folder holds a committed snapshot for convenience;
the source of truth is the scripts.

## Provenance & how to regenerate

| File | Source script | Kind |
|---|---|---|
| `00_geometry_3D.pdf` | `04_figures/gen_geometry_figure.py` | setup (qualitative) |
| `01_summary_bar.pdf` | `05_main/main_simulation.py` | **authoritative** |
| `02_temporal_y.pdf` | `05_main/main_simulation.py` | **authoritative** |
| `04_fft_y.pdf` | `05_main/main_simulation.py` | **authoritative** |
| `06_modal_damping_poles.pdf` | `05_main/main_simulation.py` | **authoritative** |
| `07_SLD_3panels.pdf` | `05_main/main_simulation.py` | **authoritative** (closed-loop monodromy) |
| `08_SLD_overlay.pdf` | `05_main/main_simulation.py` | **authoritative** (closed-loop monodromy) |
| `09_robustness_montecarlo.pdf` | `05_main/main_robustness_mc.py` | **authoritative** |
| `11_feedforward_decomposition.pdf` | `04_figures/gen_article_complete_figures.py` | illustrative (qualitative) |
| `15_control_architecture.pdf` | `04_figures/gen_control_strategy_diagram.py` | schematic |

```bash
# From a flat working dir (copy all package .py into one folder, see top-level README):
python main_simulation.py          # -> figs_lqg_vs_palf/*.pdf  (authoritative)
python main_robustness_mc.py       # -> figs_lqg_vs_palf/fig_robustness_mc.pdf
python gen_geometry_figure.py      # -> figs_article_publication/fig00*_geometry*.pdf
python gen_control_strategy_diagram.py
python gen_article_complete_figures.py   # -> figs_article_publication/*.pdf (14 illustrative)
```

## Numbers

**Authoritative** figures (from `main_simulation.py` / `main_robustness_mc.py`) use the
final model: 5-mode plant with 3-mode controllers (spillover), 10 nm measurement noise,
corrected Eq. (3) forces, Eq. (15) piezo coupling, rigorous closed-loop monodromy SLD,
train-once/held-out protocol. The headline numbers they show:

- RMS gain vs LQG: S1 +4.8 %, S2 +3.6 %, S3 (ω−8 %) +9.8 %, S4 +4.5 % (avg +5.4 %).
- Monte-Carlo (50 samples): PALF beats LQG 100 %, median +5.05 % [p05 +3.2, p95 +6.9].
- SLD @4900 RPM: OL 0.10 mm (= article experiment), LQG = PALF 1.72 mm (∂u_FF/∂x̂=0).

The two **illustrative/qualitative** figures (`11_...`, `15_...`) and the **setup**
figure (`00_...`) carry no conflicting headline numbers; `11_...` shows the u_LQG+u_FF
decomposition and the learned phase signature u_FF(φ) (from the 3-mode illustrative
script — see its header). Full verification log: `docs/REPRODUCED_RESULTS.md`.
