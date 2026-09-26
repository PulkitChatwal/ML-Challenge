# convert the challenge TSVs to parquet once (all columns as strings, empty fields as "")
import polars as pl
from common import DATASET, DATA
for sp in ("train", "test"):
    for f in ["source1", "source2", "source3"] + (["ground_truth"] if sp == "train" else []):
        df = pl.read_csv(f"{DATASET}/{sp}/{sp}_{f}.tsv", separator="\t", quote_char=None, infer_schema_length=0).fill_null("")
        df.write_parquet(f"{DATA}/{sp}_{f}.parquet"); print(sp, f, df.shape)
