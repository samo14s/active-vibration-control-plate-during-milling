# 04_figures — Publication Figure Generators

This directory contains scripts that produce the figures for the article.
Output: `figs_article_publication/` with both PNG (300 DPI) + PDF (vector).

| File | Generates |
|---|---|
| `gen_article_complete_figures.py` | 14 main results figures |
| `gen_SLD_academic_style.py` | 3 SLD style variants |
| `gen_geometry_figure.py` | 3 geometry figures (3D, front, top) |
| `gen_geometry_custom.py` | Same with customizable params |
| `gen_control_strategy_diagram.py` | 3 control architecture diagrams |

## Honest-protocol guarantees (post-audit)

The figure pipeline now uses **exactly the same experiment** as
`05_main/main_simulation.py`:

- LQG and DARC share the identical optimal base (w_q = 1e14, w_qd = 1e8) —
  the former deliberately "sub-optimal" LQG baseline (w_q = 1e13) was
  removed;
- both controllers are evaluated with the same sensor noise (0.1 µm RMS,
  common random numbers);
- DARC's feedforward/NN are designed from the NOMINAL cutting model with
  train/validation seeds disjoint from evaluation;
- SLD panels are closed-loop Floquet computations
  (`compute_SLD_closed_loop`); the DARC SLD curve equals the LQG curve by
  theory (exogenous feedforward), and the former 1.30× damping multiplier
  is gone;
- modal-damping/pole figures show LQG (= DARC base) — no fabricated
  "DARC effective damping" bars;
- grid-ceiling critical depths are labeled as lower bounds ("> x mm").

Figure numbering/layout is unchanged; "fig14 robustness" remains a
deterministic 4-scenario comparison (it is NOT a Monte Carlo — see
`03_analysis/README.md`).

## Style guide

- **Font**: Times New Roman (Serif); **Resolution**: 300 DPI PNG + PDF
- **Colors**: LQG SeaGreen `#2E8B57` · DARC Crimson `#DC143C` ·
  Open-Loop Gray `#888888`

## Typical usage

```bash
python gen_geometry_figure.py
python gen_article_complete_figures.py    # longest step (NN training)
python gen_SLD_academic_style.py
python gen_control_strategy_diagram.py
```

Use the **PDF** (vector) outputs for the journal manuscript.
