# language-agnostic generic-word features: per country, the most frequent Source 1 name tokens are "generic";
# compare only the distinctive remainder of the names, and count Source 1 entities sharing the candidate's address
import sys, numpy as np, polars as pl, time
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from common import *
split = sys.argv[1]; SRC = sys.argv[2] if len(sys.argv) > 2 else f"{split}_scores_v4"; t0 = time.time()
GEN_DF = float(__import__("os").environ.get("ER_GEN_DF", "0.002"))   # token is generic if in >=0.2% of the country's S1 names
n1 = pl.scan_parquet(f"{DATA}/{split}_source1.parquet").select(pl.len()).collect().item()
N = pl.read_parquet(f"{WORK}/{split}_norm.parquet", columns=["country", "nn", "an"]).with_row_index("r")
tok = N.select("r", "country", pl.col("nn").str.split(" ").alias("t")).explode("t").filter(pl.col("t").str.len_chars() > 0).unique(["r", "t"])
s1t = tok.filter(pl.col("r") < n1)
df = s1t.group_by("country", "t").agg(pl.len().alias("df")).join(s1t.group_by("country").agg(pl.col("r").n_unique().alias("N")), on="country")
G = df.filter(pl.col("df") / pl.col("N") >= GEN_DF).select("country", "t", pl.lit(True).alias("g"))
for c in G["country"].unique().sort():
    print(c, "generic tokens", G.filter(pl.col("country") == c).height, df.filter(pl.col("country") == c).sort("df", descending=True).head(25)["t"].to_list(), flush=True)
tok = tok.join(G, on=["country", "t"], how="left").with_columns(pl.col("g").fill_null(False))
parts = tok.sort("r", "t").group_by("r", maintain_order=True).agg(pl.col("t").filter(~pl.col("g")).str.join(" ").alias("d"), pl.col("t").filter(pl.col("g")).str.join(" ").alias("gs"))
N = N.join(parts, on="r", how="left").with_columns(pl.col("d").fill_null(""), pl.col("gs").fill_null(""))
N = N.with_columns(pl.col("an").str.split(" ").list.sort().list.join(" ").alias("ak"))
S1 = N.head(n1).with_columns(pl.len().over("country", "ak").alias("n_same_addr"))
D = N["d"].to_numpy(); GS = N["gs"].to_numpy(); nsa = S1["n_same_addr"].to_numpy().astype(np.float32)
print("tokens", time.time() - t0, flush=True)
c = pl.read_parquet(f"{WORK}/{SRC}.parquet", columns=["s", "x"])
S = c["s"].to_numpy().astype(np.int64); X = c["x"].to_numpy().astype(np.int64) + n1
out = {}
for nm, sc in (("d_ratio", fuzz.ratio), ("d_tset", fuzz.token_set_ratio), ("d_part", fuzz.partial_ratio), ("g_tset", fuzz.token_set_ratio)):
    arr = GS if nm.startswith("g_") else D; o = np.empty(len(S), np.float32)
    for st in range(0, len(S), 4_000_000): o[st:st+4_000_000] = cpdist(arr[S[st:st+4_000_000]], arr[X[st:st+4_000_000]], scorer=sc, workers=-1)
    out[nm] = o
ds, dx = D[S], D[X]
out["d_eq"] = ((ds == dx) & (ds != "")).astype(np.float32); out["d_empty_x"] = (dx == "").astype(np.float32); out["d_empty_s"] = (ds == "").astype(np.float32)
out["g_only_diff"] = (out["d_eq"] * (GS[S] != GS[X])).astype(np.float32)
out["n_same_addr"] = nsa[S]
c.with_columns(**{k: pl.Series(v) for k, v in out.items()}).write_parquet(f"{WORK}/{split}_gen.parquet"); print("done", time.time() - t0)
