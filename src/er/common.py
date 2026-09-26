import polars as pl, numpy as np, os, re, unicodedata
# ER_ROOT: working directory for data/, work/ and output*/ (default: the Lightning studio used during the challenge)
ROOT = os.environ.get("ER_ROOT", "/teamspace/studios/this_studio")
# ER_DATASET: the challenge's dataset/ folder containing train/ and test/
DATASET = os.environ.get("ER_DATASET", f"{ROOT}/student_resource/dataset")
DATA = f"{ROOT}/data"; WORK = f"{ROOT}/work"; os.makedirs(DATA, exist_ok=True); os.makedirs(WORK, exist_ok=True)
NFOLD = 5
# fold roles: bi-encoder trains on EMB folds, matcher on LGB folds, VAL held out
EMB_FOLDS, LGB_FOLDS, VAL_FOLDS = (1, 2), (3, 4), (0,)

def load(split):
    s1 = pl.read_parquet(f"{DATA}/{split}_source1.parquet")
    o = pl.concat([pl.read_parquet(f"{DATA}/{split}_source2.parquet"), pl.read_parquet(f"{DATA}/{split}_source3.parquet")])
    return s1, o

def fold_of(col):  # deterministic fold by S1 id
    return (pl.col(col).hash(seed=42) % NFOLD).cast(pl.Int8)

def gt_pairs():
    gt = pl.read_parquet(f"{DATA}/train_ground_truth.parquet")
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
              .filter(pl.col("matched_entity_ids") != "")
              .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("x")))

def text_expr():
    return (pl.col("business_name") + " | " + pl.col("business_address")).alias("text")
