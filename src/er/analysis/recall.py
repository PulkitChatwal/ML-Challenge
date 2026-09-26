import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polars as pl, numpy as np
from common import *
ids1 = pl.read_parquet(f"{DATA}/train_source1.parquet", columns=["entity_id"]).with_row_index("s").select(pl.col("s").cast(pl.Int32), pl.col("entity_id").alias("s1"))
ido = pl.concat([pl.read_parquet(f"{DATA}/train_source{i}.parquet", columns=["entity_id"]) for i in (2, 3)]).with_row_index("xi").select(pl.col("xi").cast(pl.Int32), pl.col("entity_id").alias("x"))
g = gt_pairs().join(ids1, on="s1").join(ido, on="x").select("s", pl.col("xi").alias("x"))
del ido
gk = (g["s"].to_numpy().astype(np.int64) << 24) | g["x"].to_numpy().astype(np.int64)
c = pl.read_parquet(f"{WORK}/train_cand_dense.parquet", columns=["s", "x", "rx", "rs"])
ck = (c["s"].to_numpy().astype(np.int64) << 24) | c["x"].to_numpy().astype(np.int64)
rx = c["rx"].to_numpy(); rs = c["rs"].to_numpy(); del c
o = np.argsort(ck); ck = ck[o]; rx = rx[o]; rs = rs[o]
pos = np.searchsorted(ck, gk); pos = np.minimum(pos, len(ck) - 1); hit = ck[pos] == gk
grx = np.where(hit, rx[pos], 99); grs = np.where(hit, rs[pos], 127)
print("pairs", len(ck), "true pairs", len(gk), "recall any", hit.mean())
for k in (1, 2, 3, 5, 10, 20): print(f"rx<{k}: recall {(grx < k).mean():.5f} pairs {(rx < k).sum()}")
for a, b in ((2, 5), (3, 5), (3, 10), (5, 10), (5, 20)):
    print(f"rx<{a}|rs<{b}: recall {((grx < a) | (grs < b)).mean():.5f} pairs {((rx < a) | (rs < b)).sum()}")
