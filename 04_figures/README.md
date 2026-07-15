# 04_figures — Figure Generators

Scripts that produce **standalone / styled figures**. The authoritative results
figures come from the drivers in `05_main/` (they are generated together with the
numbers they show); this directory holds the setup and style-variant generators.

Output: `figs_article_publication/` with both PNG (300 DPI) + PDF (vector).

## Files

| File | Generates | Time |
|---|---|---:|
| `gen_SLD_academic_style.py` | Academic-style SLD (overlay + 3 panels), **same rigorous monodromy** as `main_simulation.py`, serif styling | ~2 min |
| `gen_geometry_figure.py` | 3 geometry figures (3D, front, top) | ~10 s |
| `gen_geometry_custom.py` | Same with customizable params | ~10 s |

Authoritative results figures (from the drivers, written to `figs_lqg_vs_adrc/`):

- `fig01_bilan` — y_RMS / gain / effort summary bars (4 held-out scenarios)
- `fig02_temporal_y`, `fig03_temporal_u` — time responses
- `fig04_fft_y` — vibration spectra
- `fig05_rung_supervision` — A-ESO-ADRC rung trace per scenario
- `fig06_certification` — Floquet radius vs frequency mismatch (design-time
  certification; complementary robustness of the two rungs)
- `fig07_SLD_3panels`, `fig08_SLD_overlay` — rigorous closed-loop SLD
  (OL / LQG / ESO-ADRC certified), worst of 3 tool positions
- `fig09_metrics_grid` — 6-panel multi-metric grid
- `fig_robustness_mc` — Monte-Carlo box + gain histogram (`main_robustness_mc.py`)
- `fig_adaptive_removal` — drift benchmark + rung supervision traces
  (`main_adaptive_removal.py`)

## Style guide

- **Font**: Times New Roman (serif) in the academic-style variants
- **Resolution**: 300 DPI PNG + vector PDF
- **Colors**: LQG SeaGreen `#2E8B57`, ESO-ADRC blue `#1E5AA8`,
  A-ESO-ADRC Crimson `#DC143C`, Open-Loop gray `#444444`

## Typical usage

```bash
# From a flat working dir (see top-level README):
python main_simulation.py         # authoritative results + SLD figures
python main_robustness_mc.py      # Monte-Carlo figure
python main_adaptive_removal.py   # drift benchmark figure
python gen_geometry_figure.py     # geometry
python gen_SLD_academic_style.py  # styled SLD variants
```

## LaTeX integration

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\linewidth]{figs/fig08_SLD_overlay.pdf}
    \caption{Closed-loop stability lobes...}
    \label{fig:sld}
\end{figure}
```

Use **PDF** (vector) for crisp printing in journal.
