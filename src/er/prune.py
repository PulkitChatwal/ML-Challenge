# keep candidate pairs worth scoring: record-side top K1 or S1-side top K2 (ranks from dense retrieval)
import sys, polars as pl
from common import *
split = sys.argv[1]; K1 = int(sys.argv[2]); K2 = int(sys.argv[3])
c = pl.read_parquet(f"{WORK}/{split}_cand_dense.parquet")
c = c.filter((pl.col("rx") < K1) | (pl.col("rs") < K2))
c.write_parquet(f"{WORK}/{split}_cand.parquet"); print(split, "pruned pairs", c.height)
