# Positioning against the literature

From a survey across WebSearch, Consensus and Scholar Gateway. Read this
before writing the introduction: roughly half of what feels novel in this work
was established between 2005 and 2020, and claiming it would invite an easy
rejection.

> **Citation caveat.** Several entries below were surfaced by automated
> search and the key ASME/Elsevier/Springer sources returned 403 to automated
> fetch. Author lists for MSSP 159 (2021), JMP 84:1042–1053 (2022),
> *Machines* 13(6):524 (2025) and the SIAM slow-variation paper are
> **unverified**. Obtain every one of these through institutional access and
> check it before citing. Do not copy a citation from this file into a
> manuscript unread.

---

## What is already done — do NOT claim these

| Claim you might be tempted to make | Who did it |
|---|---|
| "Material removal changes the workpiece dynamics during machining" | Thevenot et al. 2006 (two companion papers); Budak, Tunç, Alan, Özgüven, CIRP Annals 2012 |
| "In-process workpiece dynamics can be predicted along the path" | Tuysuz & Altintas, ASME JMSE 2017 (frequency-domain reduced-order substructuring) and 2018 (time-domain, perturbation); Dang et al. IJMS 2019; Yang et al. MSSP 2019; Yang et al. IJMSD 2022 (GPR + POD surrogate) |
| "Stability lobes depend on tool position — 3D SLD (RPM × a_p × position)" | Bravo et al. 2005; Thevenot et al. 2006; Seguy et al. 2008 ("toolpath dependent" — the exact phrase); Tang & Liu 2009; Campa et al. 2011; Wang et al. IJMS 2019 |
| "The modes of a highly flexible workpiece drift enough to matter" | Stépán, Kiss, Ghalamchi, Sopanen, Bachrathy, CIRP Annals 2017; Kiss, Bachrathy, Stépán, ASME 2020 |
| "Frozen-time stability analysis is not exact for a slowly drifting plant" | Dombóvári, Munoa, Kuske, Stépán, Procedia CIRP 77:110–113, 2018 — *proved*, with a rigorous escape estimate in the companion SIAM paper |
| "Robust control can certify chatter-free operation over a region" | van Dijk, van de Wouw, Nijmeijer, IJRNC 2015 — over a region of (a_p, RPM), at **fixed** structural dynamics |

So §3.2 of `ASSESSMENT.md` (the workpiece moves by ~3×) is **motivation, not
contribution**. Present it as a quantified restatement of a known phenomenon
for this specific geometry, and cite Thevenot/Budak/Kiss when doing so.

---

## What is genuinely unclaimed — this is the paper

The survey's own words on the two central questions were *"Direct answer: no"*
both times.

### 1. A closed-loop stability certificate over a tool path

> "The two certificate types in the literature are disjoint. (a) Open-loop
> path-wide: 3D SLDs cover the whole path but contain no controller.
> (b) Closed-loop frozen: van Dijk et al. certifies over a region of
> (a_p, RPM) but at fixed structural dynamics. **The obvious unclaimed
> contribution is the intersection: a closed-loop 3D SLD**, certified
> chatter-free over RPM × a_p × tool-path coordinate with a controller in the
> loop."

That is exactly what `src/closed_loop_sld.py` + `src/machining_path.py`
produce. The distinguishing technical point is that the **controller
state-space is inside the monodromy matrix** — every path-wide chart in the
literature is open-loop, and every closed-loop chart substitutes an
equivalent damping ratio into an open-loop formula.

### 2. No path-parameterised controller exists

> "Every active chatter controller found for thin walls is synthesised at a
> single frozen operating point."

The strongest near-miss states its own method in one sentence: the varying
dynamics are *"overcome by designing controller with the parameters on the
maximum vibration position to stabilize the whole process"* — a worst-case
frozen design (JMP 84:1042–1053, 2022). The only genuinely position-dependent
controller found (Wang, Song, Liu, IJAMT 105:2843–2856, 2019) schedules a PD
gain heuristically on the first mode shape, with no material-removal-updated
model, no LPV synthesis and **no stability guarantee**.

### 3. Actuator placement for path-wide authority

> "As the mode shape migrates along the path, collocation, controllability
> and observability degrade — and no work jointly optimises actuator/sensor
> placement together with a scheduled controller over the whole path, nor
> certifies that authority is retained everywhere. This is a concrete,
> experimentally testable gap."

This is §3.1 of `ASSESSMENT.md`. Wang et al. IJMS 2019 documented that the
in-process mode *shapes* change, and the 2022 active-modal-control paper
optimises patch position at **one** configuration. Nobody has posed the
worst-case-over-path placement problem, and the finding that a single patch
has γ → 0 blind spots that **no** placement removes appears to be new.

### 4. Two further openings this work is positioned to take

- **Quantify the frozen-time error for material removal.** Dombóvári et al.
  proved frozen-time is wrong, but their slow parameter is the drifting
  *machine structure* in heavy-duty milling. Nobody has computed the error
  when the slow parameter is **wall thinning**, where the drift per pass is
  far larger. Self-contained and publishable, and it is the prerequisite for
  arguing a path-wide certificate is *needed*.
- **Use the scheduling-parameter rate bound.** LPV stability needs a bound on
  |dρ/dt|, and in milling that bound is free — it follows from the feed rate
  and the material removal rate, both known from the CL file *before* the
  cut. No milling paper found exploits this.

---

## Consequences for how the paper is written

1. **Lead with the certificate, not the phenomenon.** The title should be
   about closed-loop path-wide certification and actuator authority, not
   about material removal.
2. **Cite the IPW-prediction line as an enabler, not a competitor.** Budak
   2012 / Tuysuz & Altintas 2017–18 / Yang 2022 all output exactly the
   position-parameterised model family a scheduled controller needs, and all
   terminate in an open-loop chart. The natural sentence is: *these models
   exist and nobody has closed a loop around them.*
3. **Do not claim to be first to show the modes move.** Quantify it for this
   geometry and move on within a paragraph.
4. **The `git`-level honesty carries over.** The survey notes the
   IPW-prediction literature never propagates its reduction error into a
   robust synthesis, so any path-wide certificate is "only as good as an
   unquantified model". This work has the same exposure — a 3-mode modal
   truncation — and should say so rather than wait to be asked.

## Validation sources worth chasing

Mostly paywalled; these were flagged as open-access with digitisable data:

- Yang et al., *Int. J. Mech. Syst. Dyn.* 2(1):117–130 (2022) — GPR+POD,
  predicted vs experimental SLD
- *Machines* 13(6):524 (2025) — 3D SLD for Ti thin walls, semi-discretization
  with process damping
- Wang, Wu, Wan, Dulikravich, *Math. Probl. Eng.* (2015) — explicit LPV
  state-space matrices, usable directly as a control benchmark
- Reviews: *Int. J. Extreme Manufacturing* 7(6) (2025); *Machines* 11(3):359
  (2023)
