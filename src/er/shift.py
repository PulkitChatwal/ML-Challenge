import numpy as np, polars as pl
from common import *
tr = pl.read_parquet(f"{WORK}/train_scores.parquet"); te = pl.read_parquet(f"{WORK}/test_scores.parquet")
n1tr = pl.read_parquet(f"{DATA}/train_source1.parquet", columns=["country"]).height * 0.81
n1te = pl.read_parquet(f"{DATA}/test_source1.parquet", columns=["country"]).height
cte = pl.read_parquet(f"{DATA}/test_source1.parquet", columns=["country"])["country"].to_numpy()
for name, c, n1 in (("train-sim", tr, n1tr), ("test", te, n1te)):
    bx = c.group_by("x").agg(pl.col("q").max().alias("qm"), pl.col("p1").max().alias("pm"), pl.len().alias("nc"))
    print(f"== {name}: records {bx.height}, per S1 {bx.height/n1:.3f}")
    print("  best-q quantile bins:", [round(float(((bx['qm'] > a) & (bx['qm'] <= b)).mean()), 4) for a, b in ((-1, .02), (.02, .2), (.2, .5), (.5, .8), (.8, .95), (.95, 2))])
    print("  records per S1 with q>0.5:", round(float((bx['qm'] > .5).sum() / n1), 3), " ambiguous (0.2-0.8) per S1:", round(float(((bx['qm'] > .2) & (bx['qm'] < .8)).sum() / n1), 4))
    # competition: records whose 2nd-best S1 also has p1>0.3
    sec = c.filter(pl.col("p1") > 0.3).group_by("x").len().filter(pl.col("len") > 1).height
    print("  records with >=2 S1 candidates p1>0.3 per S1:", round(sec / n1, 4))
    # per S1: count of candidate records with p1>0.5
    hs = c.filter(pl.col("p1") > 0.5).group_by("s").len()
    print("  S1 hi-count dist:", np.bincount(np.minimum(hs["len"].to_numpy(), 12))[1:].round(-2).tolist()[:12])
bx = te.group_by("x").agg(pl.col("q").max().alias("qm"), pl.col("s").sort_by("q").last().alias("s"))
bx = bx.with_columns(pl.Series("country", cte[bx["s"].to_numpy()]))
print(bx.group_by("country").agg(((pl.col("qm") > .2) & (pl.col("qm") < .8)).mean().alias("ambig"), (pl.col("qm") > .5).mean().alias("hi")).sort("country").rows())
