# score candidate pairs (stage-1 p1 > 0.005) with the cross-encoder
import sys, math, numpy as np, polars as pl, torch, time
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from common import *
split = sys.argv[1]; CKPT = sys.argv[2] if len(sys.argv) > 2 else "ce"; t0 = time.time()
c = pl.read_parquet(f"{WORK}/{split}_scores.parquet", columns=["s", "x", "p1"]).filter(pl.col("p1") > 0.005).select("s", "x")
T1 = pl.read_parquet(f"{DATA}/{split}_source1.parquet").select(text_expr())["text"]
TO = pl.concat([pl.read_parquet(f"{DATA}/{split}_source{i}.parquet") for i in (2, 3)]).select(text_expr())["text"]
A = T1.gather(c["s"].to_numpy()); B = TO.gather(c["x"].to_numpy()); del T1, TO
order = (A.str.len_chars() + B.str.len_chars()).arg_sort().to_numpy()
A = A.gather(order).rechunk(); B = B.gather(order).rechunk()
print(split, "pairs", len(A), flush=True)
tok = AutoTokenizer.from_pretrained(f"{WORK}/bienc"); BS = 1024
class DS(Dataset):
    def __len__(s): return math.ceil(len(A) / BS)
    def __getitem__(s, i): return tok(A.slice(i*BS, BS).to_list(), B.slice(i*BS, BS).to_list(), padding=True, truncation="longest_first", max_length=144, return_tensors="pt")
class CE(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.enc = AutoModel.from_pretrained(f"{WORK}/bienc"); s.head = torch.nn.Linear(s.enc.config.hidden_size, 1)
    def forward(s, input_ids, attention_mask):
        return s.head(s.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]).squeeze(-1)
m = CE(); m.load_state_dict(torch.load(f"{WORK}/{CKPT}.pt")); m = m.half().cuda().eval()
out = np.empty(len(A), np.float32); pos = 0
with torch.inference_mode():
    for i, b in enumerate(DataLoader(DS(), batch_size=None, num_workers=5, prefetch_factor=4)):
        l = m(b["input_ids"].cuda(), b["attention_mask"].cuda()).float().cpu().numpy(); out[pos:pos+len(l)] = l; pos += len(l)
        if i % 1000 == 0: print(pos, round(time.time() - t0), flush=True)
res = np.empty_like(out); res[order] = out
c.with_columns(pl.Series(CKPT, res)).write_parquet(f"{WORK}/{split}_{CKPT}.parquet"); print("done", time.time() - t0)
