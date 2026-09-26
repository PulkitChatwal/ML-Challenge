import sys, numpy as np, polars as pl, torch
from sentence_transformers import SentenceTransformer
from common import *
split = sys.argv[1]
s1, o = load(split)
m = SentenceTransformer(f"{WORK}/bienc", device="cuda"); m.max_seq_length = 72; m.half()
for name, df in (("s1", s1), ("o", o)):
    t = ("query: " + df.select(text_expr())["text"]).to_list()
    # sort by length for speed, then unsort
    order = np.argsort([len(x) for x in t])
    e = m.encode([t[i] for i in order], batch_size=2048, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float16)
    out = np.empty_like(e); out[order] = e
    np.save(f"{WORK}/{split}_{name}_emb.npy", out); print(split, name, out.shape, flush=True)
