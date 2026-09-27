import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# unsupervised proxy: share of test records whose best score is uncertain (0.2-0.8), per country and version
import polars as pl
from common import *
cte = pl.read_parquet(f"{DATA}/test_source1.parquet", columns=["country"])["country"].to_numpy()
for name in sys.argv[1:]:
    c = pl.read_parquet(f"{WORK}/{name}.parquet", columns=["s", "x", "q"])
    b = c.sort("q", descending=True).unique("x", keep="first")
    b = b.with_columns(pl.Series("country", cte[b["s"].to_numpy()]))
    print(name, b.group_by("country").agg(((pl.col("q") > .2) & (pl.col("q") < .8)).mean().round(4).alias("ambig"), (pl.col("q") > .5).mean().round(4).alias("hi")).sort("country").rows())
