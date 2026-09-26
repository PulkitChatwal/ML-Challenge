# dense retrieval: for every S2/S3 record, top-K1 S1 within the same country (+ derived S1-side rank)
import sys, numpy as np, polars as pl, torch, time
from common import *
split = sys.argv[1]; KX = int(sys.argv[2]) if len(sys.argv) > 2 else 99; KS = int(sys.argv[3]) if len(sys.argv) > 3 else 999; K1 = 20; CH = 8192; t0 = time.time()
c1 = pl.read_parquet(f"{DATA}/{split}_source1.parquet", columns=["country"])["country"].to_numpy()
co = pl.concat([pl.read_parquet(f"{DATA}/{split}_source{i}.parquet", columns=["country"]) for i in (2, 3)])["country"].to_numpy()
E1 = np.load(f"{WORK}/{split}_s1_emb.npy", mmap_mode="r"); Eo = np.load(f"{WORK}/{split}_o_emb.npy", mmap_mode="r")
outs = []
for c in sorted(set(c1)):
    i1 = np.where(c1 == c)[0]; io = np.where(co == c)[0]
    A = torch.from_numpy(np.ascontiguousarray(E1[i1])).cuda()
    V = np.empty((len(io), K1), np.float16); I = np.empty((len(io), K1), np.int32)
    for st in range(0, len(io), CH):
        B = torch.from_numpy(np.ascontiguousarray(Eo[io[st:st+CH]])).cuda()
        v, ix = (B @ A.T).topk(K1, dim=1)
        V[st:st+CH] = v.cpu().numpy(); I[st:st+CH] = ix.int().cpu().numpy()
    d = pl.DataFrame({"s": i1[I.ravel()].astype(np.int32), "x": np.repeat(io, K1).astype(np.int32),
                      "sim": V.ravel().astype(np.float32), "rx": np.tile(np.arange(K1, dtype=np.int8), len(io))})
    del V, I
    d = d.with_columns((pl.col("sim").rank("ordinal", descending=True).over("s").clip(0, 127).cast(pl.Int8) - 1).alias("rs"))
    outs.append(d.filter((pl.col("rx") < KX) | (pl.col("rs") < KS))); del d
    print(c, len(i1), len(io), time.time() - t0, flush=True)
    del A; torch.cuda.empty_cache()
cand = pl.concat(outs)
cand.write_parquet(f"{WORK}/{split}_cand" + ("" if KX < 99 else "_dense") + ".parquet"); print("total", cand.height, time.time() - t0)
