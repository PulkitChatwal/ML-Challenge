# memory-lean: stage-1 on LGB folds (2-fold OOF), stage-2 context model, evaluate VAL with test-like distractors
import sys, glob, json, pickle, time, numpy as np, polars as pl, lightgbm as lgb
from common import *
from model import ctx, exp_f05, score, P1, EXCL
t0 = time.time(); TAG = sys.argv[1] if len(sys.argv) > 1 else ""
parts = sorted(glob.glob(f"{WORK}/train_feat{TAG}/part*.parquet"))
ids1 = pl.read_parquet(f"{DATA}/train_source1.parquet", columns=["entity_id"])["entity_id"]
ido = pl.concat([pl.read_parquet(f"{DATA}/train_source{i}.parquet", columns=["entity_id"]) for i in (2, 3)])["entity_id"]
fold1 = pl.DataFrame({"e": ids1}).select(fold_of("e")).to_series().to_numpy()
g = gt_pairs().join(pl.DataFrame({"s1": ids1}).with_row_index("s"), on="s1").join(pl.DataFrame({"x": ido}).with_row_index("xi"), on="x")
gk = np.sort((g["s"].to_numpy().astype(np.int64) << 24) | g["xi"].to_numpy().astype(np.int64))
def lab(df):
    k = (df["s"].to_numpy().astype(np.int64) << 24) | df["x"].to_numpy().astype(np.int64)
    p = np.minimum(np.searchsorted(gk, k), len(gk) - 1)
    return df.with_columns(pl.Series("y", (gk[p] == k).astype(np.int8)), pl.Series("f", fold1[df["s"].to_numpy()]))
FEATS = [c for c in pl.read_parquet_schema(parts[0]) if c not in EXCL]
print("feats", len(FEATS), flush=True)
fa, fb = LGB_FOLDS
tr = pl.concat([lab(pl.read_parquet(p)).filter(pl.col("f").is_in(list(LGB_FOLDS))) for p in parts])
print("stage1 train rows", tr.height, "pos rate", tr["y"].mean(), time.time() - t0, flush=True)
m1 = {}
for k in (fa, fb):
    d = tr.filter(pl.col("f") == k)
    m1[k] = lgb.train(P1, lgb.Dataset(d.select(pl.col(FEATS).cast(pl.Float32)).to_numpy(), d["y"].to_numpy()), num_boost_round=500)
    print("stage1 model", k, time.time() - t0, flush=True)
del tr, d
imp = sorted(zip(m1[fa].feature_importance("gain"), FEATS), reverse=True); print("top feats", [n for _, n in imp[:15]])
KEEP = ["s", "x", "y", "f", "sim", "rx", "rs", "wa_cont", "wn_cont", "n_jw", "a_tset", "nsp_part", "num_inter", "x_src", "x_addr_empty", "x_nonlatin"]
rows = []
for p in parts:
    d = lab(pl.read_parquet(p)); X = d.select(pl.col(FEATS).cast(pl.Float32)).to_numpy()
    pa, pb = m1[fa].predict(X), m1[fb].predict(X); f = d["f"].to_numpy()
    p1 = np.where(f == fa, pb, np.where(f == fb, pa, (pa + pb) / 2)).astype(np.float32)
    rows.append(d.select(KEEP).with_columns(pl.Series("p1", p1)))
c = pl.concat(rows); del rows
print("stage1 scored", c.height, time.time() - t0, flush=True)
# distractor simulation: drop 19% of S1 (all folds)
dropped = np.random.default_rng(7).random(len(ids1)) < 0.19
c = c.filter(pl.Series(~dropped[c["s"].to_numpy()]))
c = ctx(c, c["p1"].to_numpy())
F2 = [k for k in c.columns if k not in EXCL and k != "p1"]
print("stage2 feats", F2, flush=True)
m2 = {}
for k in (fa, fb):
    d = c.filter(pl.col("f") == k)
    m2[k] = lgb.train(P1, lgb.Dataset(d.select(pl.col(F2).cast(pl.Float32)).to_numpy(), d["y"].to_numpy()), num_boost_round=300)
X2 = c.select(pl.col(F2).cast(pl.Float32)).to_numpy(); qa, qb = m2[fa].predict(X2), m2[fb].predict(X2); del X2
f2 = c["f"].to_numpy(); c = c.with_columns(pl.Series("q", np.where(f2 == fa, qb, np.where(f2 == fb, qa, (qa + qb) / 2))))
print("stage2 done", time.time() - t0, flush=True)
best = c.filter(pl.col("q") == pl.col("q").max().over("x")).unique("x", keep="first").select("s", "x", "q", "f")
def evaluate(folds, tag):
    sidx = np.flatnonzero(np.isin(fold1, folds) & ~dropped)
    truth = {int(s): set() for s in sidx}
    for s, x in g.filter(pl.col("s").is_in(sidx)).select("s", "xi").iter_rows(): truth[s].add(x)
    d = best.filter(pl.col("f").is_in(folds)); res = {}
    for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        pred = {}
        for s, x in d.filter(pl.col("q") > th).select("s", "x").iter_rows(): pred.setdefault(s, set()).add(x)
        res[f"th{th}"] = score(pred, truth)
    pred = {}
    for s, x in exp_f05(d.filter(pl.col("q") > 0.02)).filter(pl.col("pick")).select("s", "x").iter_rows(): pred.setdefault(s, set()).add(x)
    res["expF"] = score(pred, truth)
    print(tag, {k: round(v, 5) for k, v in res.items()}, flush=True); return res
r = evaluate(list(LGB_FOLDS), "LGB-oof"); evaluate(list(VAL_FOLDS), "VAL")
k = max(r, key=r.get)
json.dump({"method": "expF" if k == "expF" else "th", "th": float(k[2:]) if k != "expF" else None}, open(f"{WORK}/decision.json", "w"))
pickle.dump({"m1": m1, "m2": m2, "F1": FEATS, "F2": F2, "KEEP": KEEP}, open(f"{WORK}/models.pkl", "wb"))
c.select("s", "x", "y", "f", "p1", "q").write_parquet(f"{WORK}/train_scores.parquet")
print("decision", k, "done", time.time() - t0)
