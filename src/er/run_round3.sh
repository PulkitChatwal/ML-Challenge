#!/usr/bin/env bash
# improvement round after run_all.sh: pseudo-label cross-encoder round 3 (from v3 test scores) + early-stopped stage 2 -> output_v4
set -euo pipefail
cd "$(dirname "$0")"; PY=${PYTHON:-python}; L=${ER_ROOT:-/teamspace/studios/this_studio}/logs; mkdir -p "$L"
ER_TAG=v3 $PY -W ignore run_stage3.py test > "$L/r3_v3test.log" 2>&1                     # saves work/test_scores_v3
( ER_PSEUDO=test_scores_v3 ER_CE_INIT=ce2 ER_CE_OUT=ce3 ER_N_FR=1000000 ER_N_OTH=500000 $PY -W ignore train_ce2.py > "$L/r3_ce3.log" 2>&1
  $PY -W ignore ce_infer.py train ce3 > "$L/r3_ce3_train.log" 2>&1
  $PY -W ignore ce_infer.py test ce3 > "$L/r3_ce3_test.log" 2>&1 ) &
G=$!
ER_TAG=v3b ER_S2_ROUNDS=2000 $PY -W ignore run_stage3.py train > "$L/r3_v3b.log" 2>&1       # stronger stage 2, same features
wait $G
ER_TAG=v4 ER_CE_EXTRA=ce3 ER_S2_ROUNDS=2000 $PY -W ignore run_stage3.py train > "$L/r3_v4.log" 2>&1
ER_TAG=v4 ER_CE_EXTRA=ce3 $PY -W ignore run_stage3.py test > "$L/r3_v4test.log" 2>&1
