# re-decide from saved test scores with a per-country threshold (no retraining): python variant.py v5 France=0.5 [US=0.7 ...]
import sys, os, shutil, numpy as np, polars as pl
from common import *
tag = sys.argv[1]; th = {"*": 0.7}
for a in sys.argv[2:]: k, v = a.split("="); th[k] = float(v)
c = pl.read_parquet(f"{WORK}/test_scores_{tag}.parquet", columns=["s", "x", "q"])
S1 = pl.read_parquet(f"{DATA}/test_source1.parquet", columns=["entity_id", "country"]); ids1 = S1["entity_id"].to_numpy(); cty = S1["country"].to_numpy()
ido = pl.concat([pl.read_parquet(f"{DATA}/test_source{i}.parquet", columns=["entity_id"]) for i in (2, 3)])["entity_id"].to_numpy()
best = c.sort("q", descending=True).unique("x", keep="first")
t = np.array([th.get(k, th["*"]) for k in cty[best["s"].to_numpy()]])
sel = best.filter(pl.Series(best["q"].to_numpy() > t))
name = "_".join(f"{k}{v}" for k, v in th.items() if k != "*") or "base"
OUT = f"{os.environ.get('ER_OUT', ROOT)}/output_{tag}_{name}"; os.makedirs(OUT, exist_ok=True)
lists = {}
for s, x in sel.select("s", "x").iter_rows(): lists.setdefault(s, []).append(x)
with open(f"{OUT}/matching_results.tsv", "w") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for i, e in enumerate(ids1): f.write(f"{e}\t{','.join(ido[sorted(lists.get(i, []))])}\n")
shutil.copy(f"{ROOT}/output_{tag}/candidate_pairs.tsv", f"{OUT}/candidate_pairs.tsv")
n = np.bincount(sel["s"].to_numpy(), minlength=len(ids1))
for k in sorted(set(cty)): m = cty == k; print(OUT, k, "matches/S1", round(n[m].mean(), 4), "empty", round((n[m] == 0).mean(), 4))
