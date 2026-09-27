# cross-encoder adaptation (default: v2 from ce.pt; set ER_PSEUDO/ER_CE_INIT/ER_CE_OUT for later rounds): continue on train EMB-fold pairs + high-confidence test pseudo-labels (France-weighted)
import sys, math, numpy as np, polars as pl, torch, time
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from common import *
t0 = time.time(); rng = np.random.default_rng(1)
import os
PSEUDO = os.environ.get("ER_PSEUDO", "test_scores"); CE_INIT = os.environ.get("ER_CE_INIT", "ce"); CE_OUT = os.environ.get("ER_CE_OUT", "ce2")
N_FR = int(os.environ.get("ER_N_FR", "900000")); N_OTH = int(os.environ.get("ER_N_OTH", "600000"))
# --- test pseudo labels ---
te = pl.read_parquet(f"{WORK}/{PSEUDO}.parquet")
cte = pl.read_parquet(f"{DATA}/test_source1.parquet", columns=["country"])["country"].to_numpy()
te = te.with_columns(pl.col("q").max().over("x").alias("qm"), pl.col("q").rank("ordinal", descending=True).over("x").alias("r"))
pos = te.filter((pl.col("r") == 1) & (pl.col("q") > 0.97)).with_columns(pl.lit(1.0).alias("y"))
neg1 = te.filter((pl.col("qm") > 0.97) & (pl.col("r") > 1) & (pl.col("r") <= 3)).with_columns(pl.lit(0.0).alias("y"))
neg2 = te.filter((pl.col("qm") < 0.005) & (pl.col("r") <= 2)).with_columns(pl.lit(0.0).alias("y"))
ps = pl.concat([pos, neg1, neg2]).select("s", "x", "y").with_columns(pl.Series("fr", cte[pl.concat([pos, neg1, neg2])["s"].to_numpy()] == "France"))
fr = ps.filter(pl.col("fr")).sample(min(N_FR, ps.filter(pl.col("fr")).height), seed=1); oth = ps.filter(~pl.col("fr")).sample(min(N_OTH, ps.filter(~pl.col("fr")).height), seed=1)
ps = pl.concat([fr, oth]).select("s", "x", "y").with_columns(pl.lit("test").alias("split"))
print("pseudo", ps.height, "france", fr.height, "pos rate", ps["y"].mean(), flush=True)
# --- train pairs (EMB folds, hard) ---
ids1 = pl.read_parquet(f"{DATA}/train_source1.parquet", columns=["entity_id"])["entity_id"]
fold1 = pl.DataFrame({"e": ids1}).select(fold_of("e")).to_series().to_numpy()
c = pl.read_parquet(f"{WORK}/train_cand.parquet", columns=["s", "x", "rx", "rs"])
c = c.filter(pl.Series(np.isin(fold1[c["s"].to_numpy()], EMB_FOLDS)) & ((pl.col("rx") < 3) | (pl.col("rs") < 5)))
ido = pl.concat([pl.read_parquet(f"{DATA}/train_source{i}.parquet", columns=["entity_id"]) for i in (2, 3)])["entity_id"]
g = gt_pairs().join(pl.DataFrame({"s1": ids1}).with_row_index("s"), on="s1").join(pl.DataFrame({"x": ido}).with_row_index("xi"), on="x")
gk = np.sort((g["s"].to_numpy().astype(np.int64) << 24) | g["xi"].to_numpy().astype(np.int64))
k = (c["s"].to_numpy().astype(np.int64) << 24) | c["x"].to_numpy().astype(np.int64)
c = c.with_columns(pl.Series("y", (gk[np.minimum(np.searchsorted(gk, k), len(gk) - 1)] == k).astype(np.float64))).sample(1_000_000, seed=5)
tr = c.select("s", "x", "y").with_columns(pl.lit("train").alias("split")); del c, g, gk
def texts(split, d):
    T1 = pl.read_parquet(f"{DATA}/{split}_source1.parquet").select(text_expr())["text"]
    TO = pl.concat([pl.read_parquet(f"{DATA}/{split}_source{i}.parquet") for i in (2, 3)]).select(text_expr())["text"]
    return T1.gather(d["s"].to_numpy()), TO.gather(d["x"].to_numpy())
a1, b1 = texts("train", tr); a2, b2 = texts("test", ps)
A = pl.concat([a1, a2]); B = pl.concat([b1, b2]); Y = np.r_[tr["y"].to_numpy(), ps["y"].to_numpy()].astype(np.float32)
perm = rng.permutation(len(Y)); A = A.gather(perm).rechunk(); B = B.gather(perm).rechunk(); Y = Y[perm]
print("total pairs", len(Y), time.time() - t0, flush=True)
tok = AutoTokenizer.from_pretrained(f"{WORK}/bienc"); BS = 256
class DS(Dataset):
    def __len__(s): return math.ceil(len(A) / BS)
    def __getitem__(s, i):
        e = tok(A.slice(i*BS, BS).to_list(), B.slice(i*BS, BS).to_list(), padding=True, truncation="longest_first", max_length=144, return_tensors="pt")
        e["labels"] = torch.from_numpy(Y[i*BS:(i+1)*BS]); return e
class CE(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.enc = AutoModel.from_pretrained(f"{WORK}/bienc"); s.head = torch.nn.Linear(s.enc.config.hidden_size, 1)
    def forward(s, input_ids, attention_mask):
        return s.head(s.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]).squeeze(-1)
m = CE(); m.load_state_dict(torch.load(f"{WORK}/{CE_INIT}.pt")); m = m.cuda()
dl = DataLoader(DS(), batch_size=None, shuffle=True, num_workers=4, prefetch_factor=4)
opt = torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=0.01)
sch = get_cosine_schedule_with_warmup(opt, int(0.03 * len(dl)), len(dl)); scaler = torch.amp.GradScaler(); run = 0
for i, b in enumerate(dl):
    b = {k: v.cuda(non_blocking=True) for k, v in b.items()}; lab = b.pop("labels")
    with torch.autocast("cuda", dtype=torch.float16):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m(b["input_ids"], b["attention_mask"]).float(), lab)
    opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
    run = 0.98 * run + 0.02 * loss.item()
    if i % 500 == 0: print(i, len(dl), round(run, 4), round(time.time() - t0), flush=True)
torch.save(m.state_dict(), f"{WORK}/{CE_OUT}.pt"); print("saved", time.time() - t0)
