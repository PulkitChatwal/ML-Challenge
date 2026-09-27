import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 1) uncertainty proxy on validation vs test by country; 2) print uncertain France test cases with their top candidates
import numpy as np, polars as pl
from common import *
tr = pl.read_parquet(f"{WORK}/train_scores_v4.parquet")
c1 = pl.read_parquet(f"{DATA}/train_source1.parquet", columns=["country"])["country"].to_numpy()
b = tr.filter(pl.col("f") == 0).sort("q", descending=True).unique("x", keep="first")
b = b.with_columns(pl.Series("country", c1[b["s"].to_numpy()]))
print("VAL proxy", b.group_by("country").agg(((pl.col("q") > .2) & (pl.col("q") < .8)).mean().round(4).alias("ambig")).sort("country").rows())
te = pl.read_parquet(f"{WORK}/test_scores_v4.parquet")
S1 = pl.read_parquet(f"{DATA}/test_source1.parquet"); O = pl.concat([pl.read_parquet(f"{DATA}/test_source{i}.parquet") for i in (2, 3)])
cte = S1["country"].to_numpy()
te = te.with_columns(pl.Series("country", cte[te["s"].to_numpy()]))
fr = te.filter(pl.col("country") == "France")
bx = fr.sort("q", descending=True).unique("x", keep="first")
amb = bx.filter((pl.col("q") > .2) & (pl.col("q") < .8)).sample(25, seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
for x in amb["x"].to_list():
    r = O.row(x); cs = fr.filter(pl.col("x") == x).sort("q", descending=True).head(3)
    print(f"\nREC {r[0][:2]} {r[1]} | {r[2]}")
    for s, q, p1 in cs.select("s", "q", "p1").iter_rows():
        a = S1.row(s); print(f"   q={q:.2f} p1={p1:.2f}  {a[1]} | {a[2]}")
