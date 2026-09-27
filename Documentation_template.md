# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary

A blocking-plus-matching pipeline built around the structure of the data:
- A fine-tuned multilingual bi-encoder generates candidates (99.4% pair recall at about 20 candidates per Source 1 entity).
- A two-stage LightGBM matcher combines string features, cross-encoder scores and "competition" features, because every Source 2/3 record belongs to at most one Source 1 entity.
- Each Source 1 entity's final match set is the one that maximises its expected F0.5.
- For France, which is absent from training, cross-encoders were adapted with high-confidence pseudo-labels from the test files themselves. No external data was used.

Held-out validation macro F0.5 is 0.990, measured under test-like distractor density.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from EDA on the 2.2M / 5.0M / 5.3M training records:

- **Exclusivity:** every Source 2/3 record appears in at most one Source 1 match list (7.64M matched pairs, zero exceptions). **Country always agrees** within a match.
- **Singletons:** 5.6% of Source 1 entities have no match, equally for US and India. The mean is 3.46 matches per entity (maximum 11).
- **Distractors:** 26% of train Source 2/3 records match nothing. Test has 5.75 Source 2/3 records per Source 1 entity against 4.68 in train, which implies about 40% distractors. The likely cause is that Source 1 entities were removed from test while their records stayed.
- **Names are weak identifiers:** 51% of Source 1 entities share their core name with another entity (chains, generic names such as "Family", "Physical Therapy"). The address carries most of the identity.
- **Noisy but real matches:** true matches can differ in house number (10 vs 326 Pine Tree Loop) and in city (Oyster Bay vs Massapequa). Names can become website forms (`raymason.com`), trade names ("X doing business as Y"), unrelated brands, OCR-style swaps (`Pa1oma`, `8ay`), or native-script transliterations (Devanagari, Telugu, Kannada, Gujarati, Malayalam). 15% of true pairs share no name token at all.
- **Deliberate decoys:** near-copies of a Source 1 entity (same address, one name token changed, e.g. "NWY Rising PC" vs "NWY Rising Downtown") are labelled as non-matches.
- **Blank addresses:** 4.4% of true-pair records have an empty address and carry only a name.

### 2.2 Solution Strategy

**Approach Type:** Hybrid: learned dense blocking, then feature-based GBDT with cross-encoder stacking and set-level decision.  
**Core Innovation:**
1. The matcher explicitly models competition between Source 1 entities for each record (exclusivity), and each entity's output set is chosen to maximise its expected F0.5.
2. Zero-shot France is handled by country-agnostic features plus test-time pseudo-label adaptation of the cross-encoders.

Source 1 entities are split by a hash of their ID into 5 folds with disjoint roles, so no stage is evaluated on data it was trained on:

| Folds | Used for |
|---|---|
| 1, 2 | Bi-encoder and cross-encoder training |
| 3, 4 | LightGBM training (2-fold out-of-fold) |
| 0 | Validation |

Validation also removes 19% of all Source 1 entities at random, turning their records into distractors. This reproduces the test density (5.745 vs 5.754 records per Source 1 entity).

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  - Dense retrieval with `intfloat/multilingual-e5-small` (MIT) fine-tuned on 2M true pairs with a symmetric in-batch contrastive loss. The text is `name | address`.
  - Exact inner-product search on the GPU, restricted to the same country (country agreement is absolute in train).
  - Both directions are kept: each record's top-3 Source 1 entities, and each Source 1 entity's top-10 records.
- **Candidate pairs generated:** 38.4M for train (17 per Source 1 entity) and 34.6M for test (20 per Source 1 entity). `candidate_pairs.tsv` is exactly the set the matcher scores.
- **How you ensured true matches were not lost:**
  - Recall was measured on train for each cut-off. The top-1 alone reaches 97.8% of true pairs, the chosen union 99.39%, and top-20 99.5%.
  - The remaining misses are mostly records with a blank address whose name is shared by many Source 1 entities.

---

## 4. Matching Model

**Features used:**

- **Name features:** measured on raw, normalised, legal-suffix-stripped, generic-word-stripped, no-space and initials forms. They include rapidfuzz ratio, token-set, token-sort, partial ratio and Jaro-Winkler, plus rarity-weighted token Jaccard, containment and maximum rare-token weight. Normalisation transliterates Indic scripts to Latin (rule-based `indic-transliteration`), strips accents, and rewrites `&` as "and" and website suffixes.
- **Address features:** the same string similarities on a canonical address (street types and US states abbreviated, French R./BD/AV forms mapped, leading zeros removed), rarity-weighted token overlap, and house/number agreement: shared-number count, number Jaccard, closest relative number difference, first-number equality.
- **Rarity weights** are computed per country from the provided files only. They therefore adapt to France automatically, with no hand-coded country lists.
- **Other:**
  - bi-encoder cosine and ranks in both directions;
  - record metadata: source, blank address, non-Latin name, website-style name, lengths;
  - sibling consensus: cosine between a record and the score-weighted centroid of the other likely records of the same Source 1 entity;
  - stage-1 competition features: rank of this Source 1 entity among the record's candidates, margin to the runner-up, sum of other probabilities, number of strong candidates per entity;
  - three cross-encoder logits (below) with their per-record rank and gap.

**Model type:**
- **Stage 1:** LightGBM (127 leaves, 500 rounds) on 48 pair features.
- **Cross-encoders:** the fine-tuned e5-small backbone with a classification head, reading both records jointly.
  - Round 1 is trained on 3M hard candidate pairs.
  - Rounds 2 and 3 continue training on train pairs plus high-confidence test pseudo-labels, weighted toward France. Pseudo-matches are a record's best pair with score above 0.97; pseudo-non-matches are those records' other candidates, and the top candidates of records scoring below 0.005.
  - Cross-encoders score only pairs with stage-1 probability above 0.005, which keeps 99.97% of true pairs.
- **Stage 2:** LightGBM on stage-1 output, cross-encoder scores, sibling features and competition context. Early stopping uses the other LGB fold.

**Threshold selection method:**
1. Each record is assigned only to its highest-scoring Source 1 entity (exclusivity).
2. For each Source 1 entity, its candidate records are sorted by probability. The kept prefix size k (including k = 0, i.e. a singleton prediction) maximises expected per-entity F0.5, estimated by Monte Carlo over independent Bernoulli outcomes.
3. On the out-of-fold data used for selection, this matched or beat the best global threshold: 0.98505 vs 0.98485 for v1, and 0.99001 vs 0.98999 for v3.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.9903 on validation (fold 0, test-like distractor density). Public leaderboard: v1 0.978, v2 0.985.

| Version | Change | Validation F0.5 |
|---|---|---|
| v1 | Blocking + stage 1/2 + expected-F0.5 decision | 0.9856 |
| v2 | + cross-encoder | 0.9900 |
| v3 | + France-adapted cross-encoder, sibling consensus | 0.9903 |

- **Common false positives (wrong merges):**
  - Planted decoys: same address, one distinctive name token changed ("CZX Auto Glass" vs "CZXE Anto Glass", "Drayify" vs "Drayique").
  - Blank-address records whose name matches a different Source 1 entity of the same chain.
  - After v3, precision is 99.8% at pair level.
- **Common false negatives (missed matches):**
  - 78% of the remaining missed pairs are blank-address records. With several Source 1 entities sharing the identical name, they are largely unresolvable, and abstaining scores better under F0.5 than guessing.
  - The rest are trade names or unrelated brand names with only a partial address.
  - Loss breakdown (v3, validation): missed matches 0.0061, predicting empty for matched entities 0.0015, wrong merges 0.0015, matching true singletons 0.0005.

---

## 6. Conclusion

The largest gains came from treating the data as the generator built it:
- per-record exclusivity and set-level expected-F0.5 decisions instead of independent pair thresholds;
- a cross-encoder stacked into the GBDT (+0.0044 validation F0.5);
- test-time pseudo-label adaptation for the unseen country, which cut uncertain test decisions for France from 6.1% to 3.7%.

The main lesson: calibrating validation to the test's distractor density was essential for reliable decisions.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- `src/run_all.sh`: the end-to-end entry point, 21 ordered steps from the challenge TSVs to `output_v3/matching_results.tsv` and `candidate_pairs.tsv`.
- `src/run_round3.sh`: the third pseudo-label round, producing `output_v4/`.
- `src/analysis/`: EDA and diagnostics only.
- `README.md`: set-up, environment variables (`ER_ROOT`, `ER_DATASET`), hardware and timings.
- `requirements.txt`: pinned versions (Python 3.12, torch 2.8 / CUDA 12.8).

Hardware: 1× NVIDIA L4 (24 GB), 8 vCPU, 32 GB RAM. End-to-end time is about 8–9 h. Model licences: `intfloat/multilingual-e5-small` (MIT), about 118M parameters, far below the 8B limit. No external data, APIs or geocoding are used. The pseudo-labels come only from the provided test files.

### B. Additional Results

Uncertain test decisions (best score between 0.2 and 0.8), an unsupervised proxy for test difficulty:

| Country | v1 | v3 |
|---|---|---|
| France | 6.1% | 3.7% |
| India | 3.2% | 1.0% |
| US | 2.8% | 1.5% |

Where validation true pairs end up (v3):

| Outcome | Pairs | Blank address |
|---|---|---|
| Correct | 97.5% | 2.5% |
| Assigned to another Source 1 entity | 11.4k | 88% |
| Not in candidates | 7.7k | 86% |
| Correct entity, low score | 11.9k | 60% |
