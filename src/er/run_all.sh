#!/usr/bin/env bash
# End-to-end pipeline: challenge TSVs -> output_v3/matching_results.tsv + candidate_pairs.tsv
# Usage:  ER_ROOT=/path/to/workdir [ER_DATASET=/path/to/dataset] [PYTHON=python] [START=N] [RUN_V2=1] ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}; START=${START:-1}; i=0
run() { i=$((i + 1)); if [ "$i" -lt "$START" ]; then echo "-- skip step $i: $*"; return; fi
        echo "== step $i: $*  ($(date +%H:%M:%S))"; $PY -W ignore "$@"; }
run prep.py                      # 1  TSV -> parquet
run norm_all.py train test       # 2  field normalisation / transliteration
run train_bienc.py 2000000       # 3  fine-tune multilingual-e5-small bi-encoder (EMB folds)
run embed_fast.py train          # 4
run embed_fast.py test           # 5
run retrieve.py train 3 10       # 6  candidates: record top-3 S1 | S1 top-10 records
run retrieve.py test 3 10        # 7
run features.py train            # 8  ~48 pair features
run features.py test             # 9
run run_train.py                 # 10 stage-1/2 LightGBM (LGB folds), VAL score, work/train_scores
run predict_test.py              # 11 v1 output + work/test_scores (used for pseudo-labels)
run train_ce.py 3000000          # 12 cross-encoder
run ce_infer.py train ce         # 13
run ce_infer.py test ce          # 14
if [ "${RUN_V2:-0}" = 1 ]; then
  run run_stage2.py train        # optional v2 (cross-encoder only)
  run run_stage2.py test
fi
run train_ce2.py                 # 15 cross-encoder v2 with test pseudo-labels (France-weighted)
run ce_infer.py train ce2        # 16
run ce_infer.py test ce2         # 17
run sib.py train                 # 18 sibling-consensus features
run sib.py test                  # 19
run run_stage3.py train          # 20 final stage-2 model, VAL score
run run_stage3.py test           # 21 -> $ER_ROOT/output_v3/
echo "done: ${ER_ROOT:-/teamspace/studios/this_studio}/output_v3"
