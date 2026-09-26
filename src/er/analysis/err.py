import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, polars as pl, sys
from common import *
from model import exp_f05
col = sys.argv[1] if len(sys.argv) > 1 else "q"; fn = sys.argv[2] if len(sys.argv) > 2 else "train_scores"
c = pl.read_parquet(f"{WORK}/{fn}.parquet")
ids1 = pl.read_parquet(f"{DATA}/train_source1.parquet", columns=["entity_id", "country"])
ido = pl.concat([pl.read_parquet(f"{DATA}/train_source{i}.parquet", columns=["entity_id"]) for i in (2, 3)])["entity_id"]
fold1 = ids1.select(fold_of("entity_id")).to_series().to_numpy()
dropped = np.random.default_rng(7).random(ids1.height) < 0.19
g = gt_pairs().join(ids1.select(pl.col("entity_id").alias("s1")).with_row_index("s"), on="s1").join(pl.DataFrame({"x": ido}).with_row_index("xi"), on="x")
sidx = np.flatnonzero(np.isin(fold1, VAL_FOLDS) & ~dropped)
best = c.filter(pl.col(col) == pl.col(col).max().over("x")).unique("x", keep="first").rename({col: "qq"})
d = best.filter(pl.col("f").is_in(list(VAL_FOLDS)))
pick = exp_f05(d.filter(pl.col("qq") > 0.02), "qq").filter(pl.col("pick"))
T = g.filter(pl.col("s").is_in(sidx)).select("s", pl.col("xi").alias("x")).with_columns(pl.lit(1).alias("t"))
P = pick.select("s", "x").with_columns(pl.lit(1).alias("p"))
cand = c.select("s", "x").filter(pl.col("s").is_in(sidx)).with_columns(pl.lit(1).alias("c"))
J = T.join(P, on=["s", "x"], how="full", coalesce=True).join(cand, on=["s", "x"], how="left").fill_null(0)
per = J.group_by("s").agg(pl.col("t").sum().alias("nt"), pl.col("p").sum().alias("np"), (pl.col("t") * pl.col("p")).sum().alias("tp"),
                          ((pl.col("t") == 1) & (pl.col("p") == 0) & (pl.col("c") == 0)).sum().alias("miss_cand"),
                          ((pl.col("t") == 1) & (pl.col("p") == 0) & (pl.col("c") == 1)).sum().alias("miss_model"),
                          ((pl.col("t") == 0) & (pl.col("p") == 1)).sum().alias("fp"))
allS = pl.DataFrame({"s": sidx.astype(np.uint32)}).join(per.with_columns(pl.col("s").cast(pl.UInt32)), on="s", how="left").fill_null(0)
allS = allS.with_columns(pl.Series("country", ids1["country"].to_numpy()[allS["s"].to_numpy()]))
P_ = pl.col("tp") / pl.col("np"); R_ = pl.col("tp") / pl.col("nt")
allS = allS.with_columns(pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0).when(pl.col("tp") == 0).then(0.0)
                         .otherwise(1.25 * P_ * R_ / (0.25 * P_ + R_)).alias("F"))
print("VAL F0.5", allS["F"].mean(), "n", allS.height)
print(allS.group_by("country").agg(pl.col("F").mean(), pl.len()).rows())
cat = allS.with_columns(pl.when((pl.col("nt") == 0) & (pl.col("np") > 0)).then(pl.lit("singleton_but_predicted"))
    .when((pl.col("nt") > 0) & (pl.col("np") == 0)).then(pl.lit("has_match_predicted_empty"))
    .when(pl.col("fp") > 0).then(pl.lit("has_false_positive"))
    .when(pl.col("miss_cand") + pl.col("miss_model") > 0).then(pl.lit("only_misses"))
    .otherwise(pl.lit("perfect")).alias("cat"))
tot = allS.height
print(cat.group_by("cat").agg(pl.len().alias("n"), ((1 - pl.col("F")).sum() / tot).alias("loss_share")).sort("loss_share", descending=True).rows())
print("fp total", allS["fp"].sum(), "miss_model", allS["miss_model"].sum(), "miss_cand", allS["miss_cand"].sum(), "tp", allS["tp"].sum())
# examples of FPs
S1 = pl.read_parquet(f"{DATA}/train_source1.parquet"); O = pl.concat([pl.read_parquet(f"{DATA}/train_source{i}.parquet") for i in (2, 3)])
ex = J.filter((pl.col("t") == 0) & (pl.col("p") == 1)).sample(12, seed=3)
xs = ex["x"].to_numpy()
own = g.select("s", "xi").filter(pl.col("xi").is_in(xs))
for s, x in ex.select("s", "x").iter_rows():
    r1 = S1.row(int(s)); rx = O.row(int(x)); o = own.filter(pl.col("xi") == x)
    ow = S1.row(int(o["s"][0])) if o.height else None
    print(f"\nS1 : {r1[1]} | {r1[2]}\n rec: {rx[1]} | {rx[2]}\n true owner: {(ow[1] + ' | ' + ow[2] + (' [DROPPED]' if dropped[o['s'][0]] else '')) if ow else 'NONE'}")
