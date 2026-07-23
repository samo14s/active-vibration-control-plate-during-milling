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
python3 "$HERE/verify_integration.py"

echo
echo "All validation scripts finished."
