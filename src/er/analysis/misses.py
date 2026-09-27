import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# where do validation true pairs get lost? (v3 scores)
import numpy as np, polars as pl
from common import *
c = pl.read_parquet(f"{WORK}/train_scores_v3b.parquet")
ids1 = pl.read_parquet(f"{DATA}/train_source1.parquet", columns=["entity_id"])
ido = pl.concat([pl.read_parquet(f"{DATA}/train_source{i}.parquet", columns=["entity_id"]) for i in (2, 3)])["entity_id"]
fold1 = ids1.select(fold_of("entity_id")).to_series().to_numpy()
dropped = np.random.default_rng(7).random(ids1.height) < 0.19
g = gt_pairs().join(ids1.select(pl.col("entity_id").alias("s1")).with_row_index("s"), on="s1").join(pl.DataFrame({"x": ido}).with_row_index("xi"), on="x")
sidx = np.flatnonzero(np.isin(fold1, VAL_FOLDS) & ~dropped)
T = g.filter(pl.col("s").is_in(sidx)).select(pl.col("s").cast(pl.Int32), pl.col("xi").cast(pl.Int32).alias("x"))
n1 = ids1.height
ae = pl.read_parquet(f"{WORK}/train_norm.parquet", columns=["addr_empty", "nonlatin"]).slice(n1)
best = c.sort("q", descending=True).unique("x", keep="first").select("x", pl.col("s").alias("best_s"), pl.col("q").alias("best_q"))
J = T.join(c.select("s", "x", "q"), on=["s", "x"], how="left").join(best, on="x", how="left")
J = J.with_columns(pl.Series("addr_empty", ae["addr_empty"].to_numpy()[J["x"].to_numpy()]), pl.Series("nonlatin", ae["nonlatin"].to_numpy()[J["x"].to_numpy()]))
J = J.with_columns(pl.when(pl.col("q").is_null()).then(pl.lit("not_candidate"))
    .when(pl.col("best_s") != pl.col("s")).then(pl.lit("lost_to_other_S1"))
    .when(pl.col("q") < 0.3).then(pl.lit("own_best_q<0.3"))
    .when(pl.col("q") < 0.7).then(pl.lit("own_best_q0.3-0.7"))
    .otherwise(pl.lit("own_best_q>=0.7")).alias("cat"))
print("true val pairs", J.height)
print(J.group_by("cat").agg(pl.len(), pl.col("addr_empty").mean().round(3).alias("addr_empty"), pl.col("nonlatin").mean().round(3).alias("nonlatin")).sort("len", descending=True).rows())
print("addr_empty rate among all true pairs", round(J["addr_empty"].mean(), 4))
lost = J.filter(pl.col("cat") == "lost_to_other_S1")
print("lost: best_q quantiles", [round(lost["best_q"].quantile(q), 3) for q in (.1, .5, .9)], " own q quantiles", [round(lost["q"].quantile(q), 3) for q in (.1, .5, .9)])
print("lost & winner dropped-pool? winner fold VAL share", np.isin(fold1[lost["best_s"].to_numpy()], VAL_FOLDS).mean())
