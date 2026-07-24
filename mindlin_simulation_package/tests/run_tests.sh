#!/usr/bin/env bash
# Run the Mindlin port validation suite.
# Usage:  bash run_tests.sh
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "###############################################################"
echo "# Mindlin Q8 port — validation suite"
echo "###############################################################"

python3 "$HERE/verify_mindlin.py"
echo
python3 "$HERE/verify_geometry.py"
echo
python3 "$HERE/verify_inprocess.py"
echo
python3 "$HERE/verify_rcsac.py"
echo
python3 "$HERE/verify_digital_twin.py"
echo
python3 "$HERE/verify_integration.py"

echo
echo "All validation scripts finished."
