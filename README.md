# ML-Challenge — Business Entity Resolution

Match every Source 1 business record to its Source 2 / Source 3 records (zero, one or many), scored by per-entity macro F0.5.

| Version | What it adds | Validation F0.5 | Leaderboard |
|---|---|---|---|
| v1 | Bi-encoder blocking + two-stage LightGBM + per-entity expected-F0.5 decision | 0.9856 | 0.978 |
| v2 | + cross-encoder score and its competition features | 0.9900 | 0.985 |
| v3 | + cross-encoder v2 adapted to France with test pseudo-labels, sibling-consensus features | 0.9903 | — |
| v4 | + third pseudo-label round from v3 test scores (`run_round3.sh`), early-stopped stage 2 | 0.9904 | — |

Validation is a held-out fold of Source 1 entities with 19% of all Source 1 entities removed, so their records become distractors. This matches the test set's record density (5.75 Source 2+3 records per Source 1 entity).

## Approach

1. **Normalisation** (`normalize.py`): Indic scripts transliterated to Latin, accents stripped, street/state abbreviations canonicalised, legal suffixes removed, a no-space name form for website-style names, initials.
2. **Blocking** (`train_bienc.py`, `embed_fast.py`, `retrieve.py`): `intfloat/multilingual-e5-small` (MIT) fine-tuned contrastively on true pairs. Exact GPU search within each country. Candidates are each record's top-3 Source 1 entities plus each Source 1 entity's top-10 records: 99.39% pair recall at about 17 candidates per Source 1 entity.
3. **Pair features** (`features.py`): rapidfuzz similarities on several name/address forms, rarity-weighted token overlap (rarity per country, computed from the provided data only), house-number agreement, record metadata.
4. **Stage 1** (`run_train.py`): LightGBM scores each pair on its own.
5. **Cross-encoders** (`train_ce.py`, `train_ce2.py`, `ce_infer.py`): the same backbone reads both records jointly. v2 continues training on high-confidence test pseudo-labels weighted towards France, the country absent from train.
6. **Sibling consensus** (`sib.py`): how similar a record is to the other likely records of the same Source 1 entity. Separates same-address decoys from true matches.
7. **Stage 2** (`run_stage3.py`): LightGBM over stage-1 score, cross-encoder scores, sibling features and competition context (rank of this Source 1 entity among the record's candidates, margin to the runner-up, candidate counts per entity).
8. **Decision**: each record is assigned to at most one Source 1 entity, its best-scoring one (in train, every record belongs to ≤1 entity). For each entity, the match set is the prefix that maximises expected F0.5.

Folds are split by Source 1 entity: folds 1–2 train the bi-encoder and cross-encoders, folds 3–4 train LightGBM (2-fold out-of-fold), fold 0 is validation. No stage is scored on data it was trained on.

## Requirements

- NVIDIA GPU with at least 24 GB memory (developed on an L4), 32 GB RAM, about 40 GB disk
- Python 3.12, `pip install -r requirements.txt` (install torch from the CUDA 12.8 index, see the file header)
- The pretrained `intfloat/multilingual-e5-small` weights are downloaded from the Hugging Face Hub on first run. No other external data or services are used.

## Run

```bash
export ER_ROOT=/path/to/workdir                               # data/, work/, output*/ are created here
export ER_DATASET=/path/to/student_resource/dataset           # contains train/ and test/
cd src/er && ./run_all.sh
```

- Resume from a step: `START=12 ./run_all.sh`
- Also produce v2: `RUN_V2=1 ./run_all.sh`
- Custom interpreter: `PYTHON=/path/to/python ./run_all.sh`

The final files are `$ER_ROOT/output_v3/matching_results.tsv` and `candidate_pairs.tsv`. Check them with the challenge validator:

```bash
python3 utils/validate_submission.py --matching $ER_ROOT/output_v3/matching_results.tsv \
  --candidate $ER_ROOT/output_v3/candidate_pairs.tsv --test-dir $ER_DATASET/test --check-ids
```

Approximate time on one L4 (8 CPUs): about 8–9 hours end to end. The slowest steps are embedding (≈20 min per split), the cross-encoder trainings (≈70 and ≈55 min), cross-encoder inference (≈30–40 min per split and model) and the stage-1/2 LightGBM fits (≈50–80 min).

## Layout

```
src/er/            pipeline (run_all.sh runs it in order)
src/er/analysis/   data forensics and diagnostics, not part of the pipeline
src/sync.sh        copy src/er to the Lightning studio (needs LIGHTNING_SSH)
```
