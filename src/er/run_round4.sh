#!/usr/bin/env bash
# round 4 after run_round3.sh: generic-word features (v5) + cross-encoder round 4 from v4 test scores (v6)
set -euo pipefail
cd "$(dirname "$0")"; PY=${PYTHON:-python}; L=${ER_ROOT:-/teamspace/studios/this_studio}/logs; mkdir -p "$L"
[ -f "${ER_ROOT:-/teamspace/studios/this_studio}/work/test_gen.parquet" ] || $PY -W ignore gen.py test > "$L/r4_gen_test.log" 2>&1
( ER_PSEUDO=test_scores_v4 ER_CE_INIT=ce3 ER_CE_OUT=ce4 ER_N_FR=1000000 ER_N_OTH=1500000 $PY -W ignore train_ce2.py > "$L/r4_ce4.log" 2>&1
  $PY -W ignore ce_infer.py train ce4 > "$L/r4_ce4_train.log" 2>&1
  $PY -W ignore ce_infer.py test ce4 > "$L/r4_ce4_test.log" 2>&1 ) &
G=$!
$PY -W ignore gen.py train > "$L/r4_gen_train.log" 2>&1
ER_TAG=v5 ER_CE_EXTRA=ce3 ER_EXTRA_FEATS=gen ER_S2_ROUNDS=2000 $PY -W ignore run_stage3.py train > "$L/r4_v5.log" 2>&1
ER_TAG=v5 ER_CE_EXTRA=ce3 ER_EXTRA_FEATS=gen $PY -W ignore run_stage3.py test > "$L/r4_v5test.log" 2>&1
wait $G
ER_TAG=v6 ER_CE_EXTRA=ce3,ce4 ER_EXTRA_FEATS=gen ER_S2_ROUNDS=2000 $PY -W ignore run_stage3.py train > "$L/r4_v6.log" 2>&1
ER_TAG=v6 ER_CE_EXTRA=ce3,ce4 ER_EXTRA_FEATS=gen $PY -W ignore run_stage3.py test > "$L/r4_v6test.log" 2>&1
