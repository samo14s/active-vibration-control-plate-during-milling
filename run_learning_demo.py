#!/usr/bin/env python3
"""Cross-part learning demo (paper Sec. 3.3, Eq. (23) / Sec. 4.2).

Machines several identical workpieces in sequence with a shared
Gaussian-process boundary learner.  Where earlier parts ran stably close
to the predicted stability boundary, the learned correction expands the
usable depth of cut, reproducing the paper's progressive MRR improvement
(~8 % between the first and later identical parts).

Usage:  python3 run_learning_demo.py [--parts 5]
"""

from __future__ import annotations

import argparse
import json
import os

from milling_sim.control import AdaptiveController
from milling_sim.learning import BoundaryGP
from milling_sim.parameters import PaperSetup
from milling_sim.plots import fig_learning
from milling_sim.runner import run_pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, default=5)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    setup = PaperSetup()
    learner = BoundaryGP()
    summaries = []
    for part in range(args.parts):
        ctl = AdaptiveController(setup, boundary_learner=learner)
        # identical parts: same seed base, small trial-to-trial scatter
        h, s = run_pass(setup, ctl, seed=903 + part)
        summaries.append(s)
        print(f"part {part + 1}: MRR {s['avg_mrr_cm3_min']:.1f} cm3/min, "
              f"RMS max {s['rms_max_um']:.0f} um, "
              f"GP points {len(learner.X)}")

    gain = 100.0 * (summaries[-1]["avg_mrr_cm3_min"]
                    / summaries[0]["avg_mrr_cm3_min"] - 1.0)
    print(f"\nMRR gain part 1 -> part {args.parts}: {gain:+.1f}% "
          "(paper reports +8%)")

    fig_learning(summaries, os.path.join(args.out, "fig_learning.png"))
    with open(os.path.join(args.out, "learning_demo.json"), "w") as f:
        json.dump({"summaries": summaries, "gain_pct": gain}, f, indent=2)


if __name__ == "__main__":
    main()
