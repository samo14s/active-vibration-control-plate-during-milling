# Manuscript

`main.tex` — elsarticle manuscript, numbered references (`elsarticle-num`)
from `refs.bib`. Figures are expected under `figures/` (PDF, referenced as
`figures/figNN_*.pdf`).

## Compile

Requires a TeX Live (or MiKTeX) installation with the `elsarticle` class.

```sh
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

or, with latexmk:

```sh
cd paper
latexmk -pdf main
```

## Notes

- Placeholders for numbers still pending from the simulation campaign are
  marked with the `\todo{...}` macro (rendered as bold `[TODO: ...]`);
  search for `\todo` before submission.
- Author names, affiliations, and CRediT roles are placeholders marked
  with `% TODO` comments in the frontmatter.
- Only figures `fig01` and `fig02` exist so far; `fig03`–`fig10` are
  referenced in the text and must be produced by the corresponding
  `scripts/fig_*.py` before a full compile will find them (compile will
  fail at `\includegraphics` for missing files — comment those figure
  environments out for partial builds if needed).
