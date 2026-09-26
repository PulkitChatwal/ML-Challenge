import sys, polars as pl, time
from common import *; from normalize import add_norm
for split in sys.argv[1:]:
    t = time.time(); s1, o = load(split)
    N = add_norm(pl.concat([s1.with_columns(pl.lit(1).cast(pl.Int8).alias("src")),
                            o.with_columns(pl.col("entity_id").str.slice(1, 1).cast(pl.Int8).alias("src"))]))
    N.write_parquet(f"{WORK}/{split}_norm.parquet"); print(split, N.shape, time.time() - t, flush=True)
