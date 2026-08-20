#!/bin/bash
# refresh_all.sh — la chaine complete, de l'optimisation aux tableaux.
#
# UN SEUL FIL BLAS. Les matrices d'etat de ce depot font 16 a 66 lignes :
# a cette taille le multi-fil coute plus qu'il ne rapporte, et plusieurs
# processus qui ouvrent chacun sept fils se disputent le processeur. Mesure
# sur une evaluation complete de l'objectif : 8.155 s par defaut contre
# 0.433 s a un fil pour le FOPID, 9.359 s contre 0.893 s pour le FDOB.
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export CALIB=${CALIB:-measured}
export PROTOCOL=${PROTOCOL:-B}
cd "$(dirname "$0")/.."

STRUCTS="${STRUCTS:-fopid adrc fdob fdob12345 dvf vpa hinf musyn lqg mpc smc nmpdob}"

step () { echo; echo "=== $* ==="; }

if [ "${SKIP_PSO:-0}" != "1" ]; then
  step "optimisation : $STRUCTS"
  # Une file par coeur, chaque structure dans son propre fichier (OUT_TAG) :
  # sans cela deux processus reliraient le meme .npz et le dernier a finir
  # ecraserait le travail de l'autre.
  ncore=$(nproc)
  i=0
  for k in $STRUCTS; do
    if [ "$k" = "fdob12345" ]; then kinds=fdob; modes=12345
    else kinds=$k; modes=12; fi
    FDOB_MODES=$modes KINDS=$kinds OUT_TAG="_$k" \
      python -u control/run_pso.py > "logs/pso_${PROTOCOL}_$k.log" 2>&1 &
    i=$((i + 1))
    [ $((i % ncore)) -eq 0 ] && wait
  done
  wait
fi

step "fusion des fichiers partiels"
python control/merge_pso.py "results/pso_${PROTOCOL}.npz" \
    results/pso_"${PROTOCOL}"_*.npz

step "comparaison a pleine resolution"
python control/run_compare.py

step "etalon temporel (les douze, dont SMC et MPC non lineaires)"
python control/run_time_compare.py

step "robustesse (sept cas), repartie sur les coeurs"
# Sept cas x douze structures x six positions x une bissection a m = 200 :
# environ six mille resolutions de Floquet, l'etape la plus longue de la
# chaine, et sans aucune interaction entre structures.
ROB="boucle ouverte,${STRUCTS// /,}"
ncore=$(nproc)
i=0
IFS=',' read -ra RK <<< "$ROB"
for k in "${RK[@]}"; do
  [ "$k" = "fdob12345" ] && export FDOB_MODES=12345 || export FDOB_MODES=12
  KINDS="$k" OUT_TAG="_${k// /_}" python -u control/robustness_new.py \
      > "logs/robust_${PROTOCOL}_${k// /_}.log" 2>&1 &
  i=$((i + 1))
  [ $((i % ncore)) -eq 0 ] && wait
done
wait
python control/robustness_new.py --merge "results/robust_new_${PROTOCOL}.npz" \
    results/robust_new_"${PROTOCOL}"_*.npz

step "tableaux et figures"
python control/report_tables.py
python control/figures.py

step "invariants"
python -m pytest tests/ -q

echo
echo "=== chaine terminee ==="
