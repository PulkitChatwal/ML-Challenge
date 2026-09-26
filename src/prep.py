# convert TSV -> parquet once
import polars as pl
D="/teamspace/studios/this_studio/student_resource/dataset"; O="/teamspace/studios/this_studio/data"
import os; os.makedirs(O, exist_ok=True)
for sp in ("train","test"):
    for f in ["source1","source2","source3"]+(["ground_truth"] if sp=="train" else []):
        df = pl.read_csv(f"{D}/{sp}/{sp}_{f}.tsv", separator="\t", quote_char=None, infer_schema_length=0).fill_null("")
        df.write_parquet(f"{O}/{sp}_{f}.parquet"); print(sp, f, df.shape)
