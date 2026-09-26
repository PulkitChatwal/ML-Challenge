# apply trained models to test candidates -> output/matching_results.tsv + candidate_pairs.tsv
import glob, json, os, pickle, numpy as np, polars as pl
from common import *
from model import ctx, exp_f05
M = pickle.load(open(f"{WORK}/models.pkl", "rb")); cfg = json.load(open(f"{WORK}/decision.json"))
rows = []
for p in sorted(glob.glob(f"{WORK}/test_feat/part*.parquet")):
    d = pl.read_parquet(p); X = d.select(pl.col(M["F1"]).cast(pl.Float32)).to_numpy()
    p1 = np.mean([m.predict(X) for m in M["m1"].values()], 0).astype(np.float32)
    rows.append(d.select([k for k in M["KEEP"] if k not in ("y", "f")]).with_columns(pl.Series("p1", p1)))
c = pl.concat(rows); del rows
c = ctx(c, c["p1"].to_numpy())
X2 = c.select(pl.col(M["F2"]).cast(pl.Float32)).to_numpy()
c = c.with_columns(pl.Series("q", np.mean([m.predict(X2) for m in M["m2"].values()], 0))); del X2
c.select("s", "x", "p1", "q").write_parquet(f"{WORK}/test_scores.parquet")
ids1 = pl.read_parquet(f"{DATA}/test_source1.parquet", columns=["entity_id"])["entity_id"].to_numpy()
ido = pl.concat([pl.read_parquet(f"{DATA}/test_source{i}.parquet", columns=["entity_id"]) for i in (2, 3)])["entity_id"].to_numpy()
best = c.filter(pl.col("q") == pl.col("q").max().over("x")).unique("x", keep="first")
sel = exp_f05(best.filter(pl.col("q") > 0.02)).filter(pl.col("pick")) if cfg["method"] == "expF" else best.filter(pl.col("q") > cfg["th"])
OUT = f"{ROOT}/output"; os.makedirs(OUT, exist_ok=True)
def write(df, col, path):
    lists = {}
    for s, x in df.select("s", "x").iter_rows(): lists.setdefault(s, []).append(x)
    with open(path, "w") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for i, e in enumerate(ids1): f.write(f"{e}\t{','.join(ido[sorted(set(lists.get(i, [])))])}\n")
    print(path, len(ids1), "non-empty", len(lists) / len(ids1))
write(sel, "matched_entity_ids", f"{OUT}/matching_results.tsv")
write(c, "candidate_entity_ids", f"{OUT}/candidate_pairs.tsv")
print("matched pairs", sel.height, "per S1", sel.height / len(ids1), "config", cfg)
