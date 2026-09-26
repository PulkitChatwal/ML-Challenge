#!/bin/sh
# v3: CE2 scores -> stage-2 v3 (parallel with test CE2) -> predict -> validate
P=/home/zeus/miniconda3/envs/cloudspace/bin/python
while kill -0 258534 2>/dev/null; do sleep 15; done
cd /teamspace/studios/this_studio/er
$P -W ignore ce_infer.py train ce2 > ../logs/ce2_train.log 2>&1
$P -W ignore ce_infer.py test ce2 > ../logs/ce2_test.log 2>&1 &
T=$!
$P -W ignore run_stage3.py train > ../logs/stage3.log 2>&1
wait $T
$P -W ignore run_stage3.py test > ../logs/predict_v3.log 2>&1
cd ../student_resource && $P utils/validate_submission.py --matching ../output_v3/matching_results.tsv --candidate ../output_v3/candidate_pairs.tsv --test-dir dataset/test --check-ids > ../logs/validate_v3.log 2>&1
