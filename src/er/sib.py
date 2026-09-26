# sibling-consensus features: similarity of record x to the p1-weighted centroid of the OTHER candidate records of s
import sys, numpy as np, polars as pl, torch, time
from common import *
split = sys.argv[1]; t0 = time.time(); torch.set_num_threads(8)
c = pl.read_parquet(f"{WORK}/{split}_scores.parquet", columns=["s", "x", "p1"])
S = torch.from_numpy(c["s"].to_numpy().astype(np.int64)); X = c["x"].to_numpy().astype(np.int64)
p = c["p1"].to_numpy().astype(np.float32); w = torch.from_numpy(np.where(p > 0.1, p, 0).astype(np.float32))
Eo = np.load(f"{WORK}/{split}_o_emb.npy", mmap_mode="r"); E1 = np.load(f"{WORK}/{split}_s1_emb.npy", mmap_mode="r")
n1 = E1.shape[0]; d = E1.shape[1]
C = torch.zeros((n1, d), dtype=torch.float32); W = torch.zeros(n1, dtype=torch.float32)
W.index_add_(0, S, w)
CH = 2_000_000
for st in range(0, len(X), CH):
    e = torch.from_numpy(Eo[X[st:st+CH]].astype(np.float32))
    C.index_add_(0, S[st:st+CH], e * w[st:st+CH, None])
print("centroids", time.time() - t0, flush=True)
out_cos = np.empty(len(X), np.float32); out_w = np.empty(len(X), np.float32); out_s1 = np.empty(len(X), np.float32)
for st in range(0, len(X), CH):
    s = S[st:st+CH]; e = torch.from_numpy(Eo[X[st:st+CH]].astype(np.float32)); ww = w[st:st+CH]
    Cx = C[s] - e * ww[:, None]; Wx = W[s] - ww
    cos = (e * Cx).sum(1) / Cx.norm(dim=1).clamp_min(1e-6)
    out_cos[st:st+CH] = torch.where(Wx > 1e-3, cos, torch.full_like(cos, -1.0)).numpy(); out_w[st:st+CH] = Wx.numpy()
    e1 = torch.from_numpy(E1[s.numpy()].astype(np.float32))
    out_s1[st:st+CH] = ((e1 * Cx).sum(1) / Cx.norm(dim=1).clamp_min(1e-6)).numpy()
c.select("s", "x").with_columns(pl.Series("sib_cos", out_cos), pl.Series("sib_w", out_w), pl.Series("s1_sib_cos", out_s1),
                                (pl.Series("sib_cos", out_cos) - pl.Series("s1_sib_cos", out_s1)).alias("sib_minus_s1")).write_parquet(f"{WORK}/{split}_sib.parquet")
print("done", time.time() - t0)
