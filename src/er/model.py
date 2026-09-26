# stage-1 pair model + stage-2 context model + per-S1 expected-F0.5 decision
import sys, numpy as np, polars as pl, lightgbm as lgb, time, json, pickle
from common import *
DROP = 0.19   # fraction of S1 removed to mimic test distractor density
P1 = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8)
EXCL = {"s", "x", "y", "f", "keep", "xf"}

def labels(split, cand):
    s1, o = load(split)
    c = cand.with_columns(fold_of_idx(s1).alias("f"))
    if split == "train":
        ids1 = s1.with_row_index("s").select(pl.col("s").cast(pl.Int32), pl.col("entity_id").alias("s1"))
        ido = o.with_row_index("x").select(pl.col("x").cast(pl.Int32), pl.col("entity_id").alias("xid"))
        own = gt_pairs().join(ids1, on="s1").join(ido, left_on="x", right_on="xid", suffix="_i").select("s", pl.col("x_i").alias("x"), pl.lit(1).cast(pl.Int8).alias("y"))
        c = c.join(own, on=["s", "x"], how="left").with_columns(pl.col("y").fill_null(0))
    return c

def fold_of_idx(s1):
    f = s1.select(fold_of("entity_id")).to_series().to_numpy()
    return pl.col("s").map_batches(lambda s: pl.Series(f[s.to_numpy()]), return_dtype=pl.Int8)

def ctx(df, p):
    # competition features around stage-1 prob p
    return (df.with_columns(pl.Series("p", p))
      .with_columns(
        pl.col("p").rank("ordinal", descending=True).over("x").alias("p_rank_x").cast(pl.Int16),
        pl.col("p").max().over("x").alias("p_max_x"),
        pl.col("p").sum().over("x").alias("p_sum_x"),
        pl.len().over("x").alias("n_x"),
        pl.col("p").rank("ordinal", descending=True).over("s").alias("p_rank_s").cast(pl.Int16),
        pl.col("p").max().over("s").alias("p_max_s"),
        (pl.col("p") > 0.5).sum().over("s").alias("n_hi_s"),
        pl.col("p").sum().over("s").alias("p_sum_s"),
        pl.len().over("s").alias("n_s"))
      .with_columns(
        (pl.col("p") - pl.col("p_max_x")).alias("gap_x"),
        (pl.col("p_sum_x") - pl.col("p")).alias("other_x"),
        pl.when(pl.col("p_rank_x") == 2).then(pl.col("p")).otherwise(0.0).max().over("x").alias("p2_x"))
      .with_columns(pl.when(pl.col("p_rank_x") == 1).then(pl.col("p") - pl.col("p2_x")).otherwise(pl.col("p") - pl.col("p_max_x")).alias("margin_x"))
      .drop("p2_x"))

def exp_f05(df, qcol="q", nsamp=64, seed=0):
    # choose per-S1 prefix size maximising expected F0.5 under independent Bernoulli(q)
    df = df.sort(["s", qcol], descending=[False, True])
    s = df["s"].to_numpy(); q = df[qcol].to_numpy().astype(np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]; ends = np.r_[starts[1:], len(s)]
    rng = np.random.default_rng(seed); keep = np.zeros(len(s), bool)
    maxn = int((ends - starts).max()) if len(s) else 0
    for n in range(1, maxn + 1):  # vectorise over groups of equal size
        g = np.flatnonzero(ends - starts == n)
        if len(g) == 0: continue
        Q = q[starts[g][:, None] + np.arange(n)]                        # G x n
        T = rng.random((nsamp, len(g), n)) < Q[None]                    # samples of truth
        tot = T.sum(2)                                                  # true count (within candidates)
        best = np.where(tot == 0, 1.0, 0.0).mean(0); bk = np.zeros(len(g), int)
        tp = np.zeros((nsamp, len(g)))
        for k in range(1, n + 1):
            tp = tp + T[:, :, k - 1]
            P = tp / k; R = np.where(tot > 0, tp / np.maximum(tot, 1), 0)
            f = np.where(tp > 0, 1.25 * P * R / (0.25 * P + R + 1e-12), 0.0).mean(0)
            better = f > best; best = np.where(better, f, best); bk = np.where(better, k, bk)
        for kk in range(1, n + 1):
            sel = bk >= kk; keep[starts[g][sel] + kk - 1] = True
    return df.with_columns(pl.Series("pick", keep))

def score(pred, truth):
    # pred/truth: dict s -> set ; macro F0.5 over truth keys
    tot = 0.0
    for s, T in truth.items():
        P = pred.get(s, set())
        if not T and not P: tot += 1; continue
        if not T or not P: continue
        tp = len(T & P)
        if tp == 0: continue
        p = tp / len(P); r = tp / len(T); tot += 1.25 * p * r / (0.25 * p + r)
    return tot / len(truth)
