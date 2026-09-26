# cross-encoder (init from fine-tuned bi-encoder backbone) on hard candidate pairs of EMB folds
import sys, os, numpy as np, polars as pl, torch, time, math
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from common import *
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000; t0 = time.time()
ids1 = pl.read_parquet(f"{DATA}/train_source1.parquet", columns=["entity_id"])["entity_id"]
fold1 = pl.DataFrame({"e": ids1}).select(fold_of("e")).to_series().to_numpy()
c = pl.read_parquet(f"{WORK}/train_cand.parquet", columns=["s", "x", "rx", "rs"])
c = c.filter(pl.Series(np.isin(fold1[c["s"].to_numpy()], EMB_FOLDS)) & ((pl.col("rx") < 3) | (pl.col("rs") < 5)))
ido = pl.concat([pl.read_parquet(f"{DATA}/train_source{i}.parquet", columns=["entity_id"]) for i in (2, 3)])["entity_id"]
g = gt_pairs().join(pl.DataFrame({"s1": ids1}).with_row_index("s"), on="s1").join(pl.DataFrame({"x": ido}).with_row_index("xi"), on="x")
gk = np.sort((g["s"].to_numpy().astype(np.int64) << 24) | g["xi"].to_numpy().astype(np.int64))
k = (c["s"].to_numpy().astype(np.int64) << 24) | c["x"].to_numpy().astype(np.int64)
y = (gk[np.minimum(np.searchsorted(gk, k), len(gk) - 1)] == k).astype(np.float32)
c = c.with_columns(pl.Series("y", y)).sample(min(N, c.height), seed=0, shuffle=True)
print("ce pairs", c.height, "pos", c["y"].mean(), flush=True)
T1 = pl.read_parquet(f"{DATA}/train_source1.parquet").select(text_expr())["text"]
TO = pl.concat([pl.read_parquet(f"{DATA}/train_source{i}.parquet") for i in (2, 3)]).select(text_expr())["text"]
A = T1.gather(c["s"].to_numpy()).rechunk(); B = TO.gather(c["x"].to_numpy()).rechunk(); Y = c["y"].to_numpy(); del T1, TO, c
tok = AutoTokenizer.from_pretrained(f"{WORK}/bienc")
BS = 256
class DS(Dataset):
    def __len__(s): return math.ceil(len(A) / BS)
    def __getitem__(s, i):
        a = A.slice(i * BS, BS).to_list(); b = B.slice(i * BS, BS).to_list()
        e = tok(a, b, padding=True, truncation="longest_first", max_length=144, return_tensors="pt"); e["labels"] = torch.from_numpy(Y[i*BS:(i+1)*BS]); return e
class CE(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.enc = AutoModel.from_pretrained(f"{WORK}/bienc"); s.head = torch.nn.Linear(s.enc.config.hidden_size, 1)
    def forward(s, input_ids, attention_mask, token_type_ids=None):
        h = s.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return s.head(h).squeeze(-1)
m = CE().cuda()
dl = DataLoader(DS(), batch_size=None, shuffle=True, num_workers=4, prefetch_factor=4)
opt = torch.optim.AdamW(m.parameters(), lr=4e-5, weight_decay=0.01)
sch = get_cosine_schedule_with_warmup(opt, int(0.03 * len(dl)), len(dl)); scaler = torch.amp.GradScaler()
m.train(); run = 0
for i, b in enumerate(dl):
    b = {k: v.cuda(non_blocking=True) for k, v in b.items()}; lab = b.pop("labels")
    with torch.autocast("cuda", dtype=torch.float16):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m(b["input_ids"], b["attention_mask"]).float(), lab)
    opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
    run = 0.98 * run + 0.02 * loss.item()
    if i % 500 == 0: print(i, len(dl), round(run, 4), round(time.time() - t0), flush=True)
torch.save(m.state_dict(), f"{WORK}/ce.pt"); print("saved", time.time() - t0)
