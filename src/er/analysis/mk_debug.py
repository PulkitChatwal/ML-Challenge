import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, polars as pl
from common import *
s1, o = load("train")
ids1 = s1.with_row_index("s").select(pl.col("s").cast(pl.Int32), pl.col("entity_id").alias("s1"))
ido = o.with_row_index("xi").select(pl.col("xi").cast(pl.Int32), pl.col("entity_id").alias("x"))
g = gt_pairs().sample(200000, seed=0).join(ids1, on="s1").join(ido, on="x").select("s", pl.col("xi").alias("x"))
rng = np.random.default_rng(0)
neg = pl.DataFrame({"s": rng.integers(0, s1.height, 600000).astype(np.int32), "x": np.repeat(g["x"].to_numpy(), 3)})
c = pl.concat([g, neg]).unique(["s", "x"]).with_columns(pl.lit(0.5).cast(pl.Float32).alias("sim"), pl.lit(0).cast(pl.Int8).alias("rx"), pl.lit(0).cast(pl.Int8).alias("rs"))
c.write_parquet(f"{WORK}/train_cand_dbg.parquet"); print(c.height)
