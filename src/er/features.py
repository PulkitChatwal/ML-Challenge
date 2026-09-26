import sys, numpy as np, polars as pl, numba as nb, time
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist
from common import *
from normalize import ADDR_STOP
import gc
split = sys.argv[1]; TAG = sys.argv[2] if len(sys.argv) > 2 else ""; t0 = time.time()
n1 = pl.scan_parquet(f"{DATA}/{split}_source1.parquet").select(pl.len()).collect().item()
if not os.path.exists(f"{WORK}/{split}_norm.parquet"): sys.exit("run norm_all.py first")
N = pl.read_parquet(f"{WORK}/{split}_norm.parquet")
print("norm", time.time() - t0, flush=True)
# ---- token CSR with per-country IDF (transductive, uses only provided data) ----
def csr(col, stop=()):
    # tokens hashed to u64; idf per country
    t = N.select(pl.int_range(pl.len(), dtype=pl.Int64).alias("r"), pl.col("country").cast(pl.Categorical), pl.col(col).str.split(" ").alias("t")).explode("t")
    t = t.filter(pl.col("t").str.len_chars() > 0)
    if stop: t = t.filter(~pl.col("t").is_in(list(stop)))
    t = t.with_columns(pl.col("t").hash(seed=1).alias("h")).drop("t").unique(["r", "h"])
    df = t.group_by("country", "h").agg(pl.len().alias("df"))
    tot = t.group_by("country").agg(pl.col("r").n_unique().alias("N"))
    df = df.join(tot, on="country").with_columns((pl.col("N") / pl.col("df")).log().cast(pl.Float32).alias("idf")).select("country", "h", "idf")
    t = t.join(df, on=["country", "h"]).sort("r", "h")
    cnt = np.bincount(t["r"].to_numpy(), minlength=N.height); ptr = np.zeros(N.height + 1, np.int64); ptr[1:] = np.cumsum(cnt)
    ids = t["h"].to_numpy().astype(np.uint64); w = t["idf"].to_numpy().copy(); del t, df
    return ptr, ids, w
@nb.njit(parallel=True, cache=True)
def wover(a, b, ptr, ids, w):  # ids uint64 sorted within row
    n = len(a); out = np.zeros((n, 4), np.float32)
    for k in nb.prange(n):
        i, j = a[k], b[k]; p, pe, q, qe = ptr[i], ptr[i+1], ptr[j], ptr[j+1]
        wi = 0.0; wa = 0.0; wb = 0.0; mx = 0.0
        for u in range(p, pe): wa += w[u]
        for u in range(q, qe): wb += w[u]
        while p < pe and q < qe:
            if ids[p] == ids[q]:
                wi += w[p]; mx = max(mx, w[p]); p += 1; q += 1
            elif ids[p] < ids[q]: p += 1
            else: q += 1
        un = wa + wb - wi
        out[k, 0] = wi / un if un > 0 else 0; out[k, 1] = wi / min(wa, wb) if min(wa, wb) > 0 else 0; out[k, 2] = mx; out[k, 3] = wi
    return out
CN = csr("nn"); CA = csr("an", ADDR_STOP); CC = csr("ncc")
print("csr", time.time() - t0, flush=True)
@nb.njit(parallel=True, cache=True)
def numfeat(a, b, ptr, v):
    n = len(a); out = np.zeros((n, 5), np.float32)
    for k in nb.prange(n):
        i, j = a[k], b[k]; ni = ptr[i+1]-ptr[i]; nj = ptr[j+1]-ptr[j]; inter = 0; best = 1e9
        for p in range(ptr[i], ptr[i+1]):
            for q in range(ptr[j], ptr[j+1]):
                if v[p] == v[q]: inter += 1
                d = abs(v[p]-v[q]) / max(1.0, max(v[p], v[q]))
                if d < best: best = d
        out[k, 0] = ni; out[k, 1] = nj; out[k, 2] = inter; out[k, 3] = inter / (ni + nj - inter) if ni + nj - inter > 0 else -1
        out[k, 4] = best if best < 1e9 else -1
    return out
cand_all = pl.read_parquet(f"{WORK}/{split}_cand{TAG}.parquet")
lens = None
meta = N.select(pl.col("nonlatin").cast(pl.Float32), pl.col("addr_empty").cast(pl.Float32), pl.col("is_domain").cast(pl.Float32), pl.col("src").cast(pl.Float32),
                pl.col("nn").str.len_chars().cast(pl.Float32).alias("ln"), pl.col("an").str.len_chars().cast(pl.Float32).alias("la"),
                pl.col("nn").str.count_matches(" ").cast(pl.Float32).alias("nw"))
fn = N.select(pl.col("an").str.extract(r"(\d+)").fill_null("")).to_series().to_numpy()
nums = N["nums"].to_list()
nlen = np.array([len(x) for x in nums]); nptr = np.zeros(len(nums) + 1, np.int64); nptr[1:] = np.cumsum(nlen)
nflat = np.array([float(y[:12]) for x in nums for y in x], np.float64); del nums
STR = {c: N[c].to_numpy() for c in ("nn", "nc", "ncc", "nsp", "an", "init", "business_name")}
del N; gc.collect()
CHUNK = 5_000_000
os.makedirs(f"{WORK}/{split}_feat{TAG}", exist_ok=True)
for ci, st0 in enumerate(range(0, cand_all.height, CHUNK)):
    cand = cand_all.slice(st0, CHUNK)
    S = cand["s"].to_numpy().astype(np.int64); X = cand["x"].to_numpy().astype(np.int64) + n1
    F = {}
    def fz(name, scorer, col):
        a = STR[col]; F[name] = cpdist(a[S], a[X], scorer=scorer, workers=-1).astype(np.float32)
    fz("n_ratio", fuzz.ratio, "nn"); fz("n_tset", fuzz.token_set_ratio, "nn"); fz("n_tsort", fuzz.token_sort_ratio, "nn"); fz("n_part", fuzz.partial_ratio, "nn")
    fz("n_jw", JaroWinkler.normalized_similarity, "nn"); fz("nc_ratio", fuzz.ratio, "nc"); fz("nc_tset", fuzz.token_set_ratio, "nc")
    fz("ncc_ratio", fuzz.ratio, "ncc"); fz("nsp_ratio", fuzz.ratio, "nsp"); fz("nsp_part", fuzz.partial_ratio, "nsp"); fz("init_ratio", fuzz.ratio, "init")
    fz("raw_ratio", fuzz.ratio, "business_name")
    fz("a_ratio", fuzz.ratio, "an"); fz("a_tset", fuzz.token_set_ratio, "an"); fz("a_tsort", fuzz.token_sort_ratio, "an"); fz("a_part", fuzz.partial_ratio, "an")
    for nm, C in (("wn", CN), ("wa", CA), ("wc", CC)):
        r = wover(S, X, *C)
        for i, suf in enumerate(("jac", "cont", "max", "sum")): F[f"{nm}_{suf}"] = r[:, i]
    r = numfeat(S, X, nptr, nflat)
    for i, nm in enumerate(("num_n1", "num_nx", "num_inter", "num_jac", "num_reldiff")): F[nm] = r[:, i]
    F["fnum_eq"] = ((fn[S] == fn[X]) & (fn[S] != "")).astype(np.float32); F["fnum_miss"] = (fn[X] == "").astype(np.float32)
    for c in meta.columns: F[f"x_{c}"] = meta[c].to_numpy()[X]
    for c in ("ln", "la", "nw"): F[f"s_{c}"] = meta[c].to_numpy()[S]
    cand.with_columns(**{k: pl.Series(v) for k, v in F.items()}).write_parquet(f"{WORK}/{split}_feat{TAG}/part{ci:03d}.parquet")
    print("chunk", ci, time.time() - t0, flush=True)
print("done", time.time() - t0)
