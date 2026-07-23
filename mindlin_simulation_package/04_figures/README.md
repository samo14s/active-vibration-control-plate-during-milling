# 04_figures — Publication Figure Generators

This directory contains scripts that produce **all figures** for the article.
Output: `figs_article_publication/` with both PNG (300 DPI) + PDF (vector).

## Files

| File | Generates | Time |
|---|---|---:|
| `gen_article_complete_figures.py` | 14 main results figures | ~4 min |
| `gen_SLD_academic_style.py` | 3 SLD style variants | ~80 s |
| `gen_geometry_figure.py` | 3 geometry figures (3D, front, top) | ~10 s |
| `gen_geometry_custom.py` | Same with customizable params | ~10 s |
| `gen_control_strategy_diagram.py` | 3 control architecture diagrams | ~5 s |

## Output: 20+ figures total

### Geometry (3 figures)
- `fig00a_geometry_3D` — 3D isometric perspective
- `fig00b_geometry_front` — Front view (Y=0)
- `fig00c_geometry_top` — Top view (Z=H_P) **[KEY view for peripheral milling]**

### Results (14 figures)
- `fig01_bilan_global` — 3-panel global summary
- `fig02_temporal_y_fullpath` — y(t) on 20.4 s, 4 scenarios
- `fig03_temporal_u_fullpath` — u(t) on 20.4 s, 4 scenarios
- `fig04_time_fft_S1` — Time + FFT side-by-side
- `fig05_fft_y_4scenarios` — FFT y annotated
- `fig06_fft_u_4scenarios` — FFT u (log scale)
- `fig07_poles_damping` — Modal damping + complex plane
- `fig08_SLD_3panels` — SLD 3 panels (OL/LQG/DARC)
- `fig09_SLD_overlay` — SLD overlay (key figure)
- `fig09b_SLD_hatched` — SLD hatched alternative
- `fig10_metrics_grid` — 6-panel multi-metric
- `fig11_DARC_internal` — DARC internal blocks
- `fig12_tool_position_full` — Tool position + envelope
- `fig13_zoom_3phases` — 3 phases zoom
- `fig14_robustness` — Robustness comparison

### Control architecture (3 figures)
- `fig15_control_architecture` — Detailed block diagram
- `fig16_algorithm_flow` — Algorithm flowchart
- `fig17_DARC_summary` — Compact 1-page summary

## Style guide

All figures follow:
- **Font**: Times New Roman (Serif)
- **Resolution**: 300 DPI PNG + Vector PDF
- **Colors**: 
  - LQG: SeaGreen `#2E8B57`
  - DARC-MPC: Crimson `#DC143C`
  - Open-Loop: Gray `#888888`
- **Axes**: Hidden top/right spines
- **Subtitles**: (a), (b), (c) for LaTeX referencing

## Typical usage

```bash
# Generate all figures (run once)
python gen_geometry_figure.py            # ~10 s
python gen_article_complete_figures.py   # ~4 min
python gen_SLD_academic_style.py         # ~80 s
python gen_control_strategy_diagram.py   # ~5 s
```

Total: **~6 minutes** to regenerate all 20+ figures.

## Customization

If you want to change figure parameters (colors, sizes, etc.),
all generators have configuration sections at the top of the file.

The most customizable is `gen_geometry_custom.py` which has:
- All physical parameters (LP, HP, BP, ...)
- All colors
- All visibility flags (SHOW_TITLE, SHOW_GRID, ...)
- 3D viewing angles (ISO_ANGLE_1, ISO_ANGLE_2)

## LaTeX integration

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\linewidth]{figs/fig01_bilan_global.pdf}
    \caption{Global performance comparison...}
    \label{fig:bilan}
\end{figure}
```

Use **PDF** (vector) for crisp printing in journal.
