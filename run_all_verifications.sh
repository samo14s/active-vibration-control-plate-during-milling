#!/usr/bin/env bash
# Run every verification script and report a pass/fail summary.
#
# These are the checks behind every number in README.md and docs/ASSESSMENT.md.
# Runtime is roughly 15-25 minutes in total on 4 cores; the two slowest are
# verify_fixed_gain_instability and verify_ppf_certified, which bisect the
# certified stability boundary many times over.
set -u
cd "$(dirname "$0")"

TESTS=(
  verify_evolving                          # element, quadrature, material removal
  verify_leissa_cantilever                 # CFFF benchmark: the BC actually used
  verify_monodromy_equivalence             # fast monodromy == dense product
  verify_nonlinear_force                   # chip-thickness positivity
  verify_size_effect_and_process_damping   # K_t(h), edge forces, process damping
  verify_piezo_patch_structure             # patch as structure, composite arm
  verify_ppf_certified                     # PPF with its filter in the loop
  verify_feedforward_cannot_move_lobes     # feedforward cannot move the boundary
  verify_closed_loop_sld                   # certified lobe vs time domain
  verify_fixed_gain_instability            # the three-way instability check
)

pass=0; fail=0
for t in "${TESTS[@]}"; do
  printf '  %-42s ' "$t"
  if python3 "tests/$t.py" > "/tmp/${t}.log" 2>&1; then
    echo "PASS"; pass=$((pass+1))
  else
    echo "FAIL  (see /tmp/${t}.log)"; fail=$((fail+1))
  fi
done

echo
echo "  $pass passed, $fail failed"
echo
echo "  Analyses that report rather than assert (no pass/fail):"
echo "    python3 tests/analyze_actuator_alignment.py"
echo "    python3 tests/audit_baseline_claims.py"
echo "    python3 tests/verify_substitution_error_along_path.py"
echo
echo "  Studies producing the paper's results:"
echo "    python3 experiments/run_path_study.py"
echo "    python3 experiments/run_benchmark.py"
echo "    python3 experiments/run_discretisation_study.py"
echo "    python3 experiments/run_uncertainty_quantification.py"
exit $((fail > 0))
