# faster embedding: parallel tokenisation workers + fp16 mean-pooling (same output as SentenceTransformer e5)
import sys, os, numpy as np, polars as pl, torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from common import *
split = sys.argv[1]
tok = AutoTokenizer.from_pretrained(f"{WORK}/bienc")
model = AutoModel.from_pretrained(f"{WORK}/bienc", torch_dtype=torch.float16).cuda().eval()
class DS(Dataset):
    def __init__(s, t, bs): s.t, s.bs = t, bs
    def __len__(s): return (len(s.t) + s.bs - 1) // s.bs
    def __getitem__(s, i): return tok(s.t.slice(i*s.bs, s.bs).to_list(), padding=True, truncation=True, max_length=72, return_tensors="pt")
s1, o = load(split)
for name, df in (("s1", s1), ("o", o)):
    if os.path.exists(f"{WORK}/{split}_{name}_emb.npy"): print("skip", name); continue
    t = "query: " + df.select(text_expr())["text"]
    order = t.str.len_chars().arg_sort().to_numpy(); ts = t.gather(order).rechunk(); n = len(t); del t
    dl = DataLoader(DS(ts, 2048), batch_size=None, num_workers=6, prefetch_factor=4)
    res = np.empty((n, model.config.hidden_size), np.float16); pos = 0
    with torch.inference_mode():
        for b in dl:
            b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
            h = model(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1); e = torch.nn.functional.normalize(e.float(), dim=-1)
            res[order[pos:pos+len(e)]] = e.half().cpu().numpy(); pos += len(e)
            if (pos // 2048) % 500 == 0: print(name, pos, flush=True)
    np.save(f"{WORK}/{split}_{name}_emb.npy", res); print(split, name, res.shape, flush=True)
