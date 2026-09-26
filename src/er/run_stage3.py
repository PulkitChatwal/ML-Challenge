# stage-2 v2: reuse stage-1 p1, add cross-encoder score + its context; evaluate VAL; optionally predict test
import sys, glob, json, pickle, time, numpy as np, polars as pl, lightgbm as lgb
from common import *
from model import ctx, exp_f05, score, P1, EXCL
t0 = time.time(); MODE = sys.argv[1]  # "train" or "test"
KEEP = ["s", "x", "sim", "rx", "rs", "wa_cont", "wn_cont", "n_jw", "a_tset", "nsp_part", "num_inter", "x_src", "x_addr_empty", "x_nonlatin",
        "n_tset", "nc_ratio", "a_ratio", "wa_jac", "num_jac", "num_reldiff", "fnum_eq", "x_is_domain"]
def build(split):
    parts = sorted(glob.glob(f"{WORK}/{split}_feat/part*.parquet"))
    c = pl.concat([pl.read_parquet(p, columns=KEEP) for p in parts])
    sc = pl.read_parquet(f"{WORK}/{split}_scores.parquet")
    c = c.join(sc.drop("q"), on=["s", "x"], how="inner")          # train scores already exclude dropped S1
    c = c.join(pl.read_parquet(f"{WORK}/{split}_ce.parquet"), on=["s", "x"], how="left").with_columns(pl.col("ce").fill_null(-12.0))
    c = c.join(pl.read_parquet(f"{WORK}/{split}_ce2.parquet"), on=["s", "x"], how="left").with_columns(pl.col("ce2").fill_null(-12.0))
    c = c.join(pl.read_parquet(f"{WORK}/{split}_sib.parquet"), on=["s", "x"], how="left")
    c = ctx(c, c["p1"].to_numpy())
    c = c.with_columns(
        pl.col("ce").rank("ordinal", descending=True).over("x").cast(pl.Int16).alias("ce_rank_x"),
        pl.col("ce").max().over("x").alias("ce_max_x"),
        pl.col("ce").rank("ordinal", descending=True).over("s").cast(pl.Int16).alias("ce_rank_s"),
        (pl.col("ce") > 0).sum().over("s").alias("ce_hi_s"),
        pl.col("ce2").rank("ordinal", descending=True).over("x").cast(pl.Int16).alias("ce2_rank_x"),
        (pl.col("ce2") - pl.col("ce2").max().over("x")).alias("ce2_gap_x"),
        (pl.col("ce2") > 0).sum().over("s").alias("ce2_hi_s"),
        pl.col("sib_cos").rank("ordinal", descending=True).over("x").cast(pl.Int16).alias("sib_rank_x"))
    c = c.with_columns((pl.col("ce") - pl.col("ce_max_x")).alias("ce_gap_x"),
                       pl.when(pl.col("ce_rank_x") == 2).then(pl.col("ce")).otherwise(-99.0).max().over("x").alias("ce2_x"))
    return c.with_columns(pl.when(pl.col("ce_rank_x") == 1).then(pl.col("ce") - pl.col("ce2_x")).otherwise(pl.col("ce") - pl.col("ce_max_x")).alias("ce_margin_x")).drop("ce2_x")
fa, fb = LGB_FOLDS
if MODE == "train":
    c = build("train"); F2 = [k for k in c.columns if k not in EXCL and k not in ("p1", "q")]
    print("rows", c.height, "feats", len(F2), time.time() - t0, flush=True)
    m2 = {}
    for k in (fa, fb):
        d = c.filter(pl.col("f") == k)
        m2[k] = lgb.train(P1, lgb.Dataset(d.select(pl.col(F2).cast(pl.Float32)).to_numpy(), d["y"].to_numpy()), num_boost_round=400)
    X2 = c.select(pl.col(F2).cast(pl.Float32)).to_numpy(); qa, qb = m2[fa].predict(X2), m2[fb].predict(X2); del X2
    f2 = c["f"].to_numpy(); c = c.with_columns(pl.Series("q", np.where(f2 == fa, qb, np.where(f2 == fb, qa, (qa + qb) / 2))))
    imp = sorted(zip(m2[fa].feature_importance("gain"), F2), reverse=True); print("top", [n for _, n in imp[:12]], flush=True)
    ids1 = pl.read_parquet(f"{DATA}/train_source1.parquet", columns=["entity_id"])["entity_id"]
    ido = pl.concat([pl.read_parquet(f"{DATA}/train_source{i}.parquet", columns=["entity_id"]) for i in (2, 3)])["entity_id"]
    fold1 = pl.DataFrame({"e": ids1}).select(fold_of("e")).to_series().to_numpy()
    dropped = np.random.default_rng(7).random(len(ids1)) < 0.19
    g = gt_pairs().join(pl.DataFrame({"s1": ids1}).with_row_index("s"), on="s1").join(pl.DataFrame({"x": ido}).with_row_index("xi"), on="x")
    best = c.filter(pl.col("q") == pl.col("q").max().over("x")).unique("x", keep="first").select("s", "x", "q", "f")
    def evaluate(folds, tag):
        sidx = np.flatnonzero(np.isin(fold1, folds) & ~dropped); truth = {int(s): set() for s in sidx}
        for s, x in g.filter(pl.col("s").is_in(sidx)).select("s", "xi").iter_rows(): truth[s].add(x)
        d = best.filter(pl.col("f").is_in(folds)); res = {}
        for th in (0.5, 0.6, 0.7, 0.8):
            pred = {}
            for s, x in d.filter(pl.col("q") > th).select("s", "x").iter_rows(): pred.setdefault(s, set()).add(x)
            res[f"th{th}"] = score(pred, truth)
        pred = {}
        for s, x in exp_f05(d.filter(pl.col("q") > 0.02)).filter(pl.col("pick")).select("s", "x").iter_rows(): pred.setdefault(s, set()).add(x)
        res["expF"] = score(pred, truth); print(tag, {k: round(v, 5) for k, v in res.items()}, flush=True); return res
    r = evaluate(list(LGB_FOLDS), "LGB-oof"); evaluate(list(VAL_FOLDS), "VAL")
    k = max(r, key=r.get)
    pickle.dump({"m2": m2, "F2": F2, "cfg": {"method": "expF" if k == "expF" else "th", "th": float(k[2:]) if k != "expF" else None}}, open(f"{WORK}/models_v3.pkl", "wb"))
    print("decision", k, time.time() - t0)
else:
    M = pickle.load(open(f"{WORK}/models_v3.pkl", "rb")); cfg = M["cfg"]
    c = build("test"); X2 = c.select(pl.col(M["F2"]).cast(pl.Float32)).to_numpy()
    c = c.with_columns(pl.Series("q", np.mean([m.predict(X2) for m in M["m2"].values()], 0))); del X2
    ids1 = pl.read_parquet(f"{DATA}/test_source1.parquet", columns=["entity_id"])["entity_id"].to_numpy()
    ido = pl.concat([pl.read_parquet(f"{DATA}/test_source{i}.parquet", columns=["entity_id"]) for i in (2, 3)])["entity_id"].to_numpy()
    best = c.filter(pl.col("q") == pl.col("q").max().over("x")).unique("x", keep="first")
    sel = exp_f05(best.filter(pl.col("q") > 0.02)).filter(pl.col("pick")) if cfg["method"] == "expF" else best.filter(pl.col("q") > cfg["th"])
    OUT = f"{ROOT}/output_v3"; import os; os.makedirs(OUT, exist_ok=True)
    def write(df, col, path):
        lists = {}
        for s, x in df.select("s", "x").iter_rows(): lists.setdefault(s, []).append(x)
        with open(path, "w") as f:
            f.write(f"source1_entity_id\t{col}\n")
            for i, e in enumerate(ids1): f.write(f"{e}\t{','.join(ido[sorted(set(lists.get(i, [])))])}\n")
        print(path, "non-empty", len(lists) / len(ids1))
    write(sel, "matched_entity_ids", f"{OUT}/matching_results.tsv"); write(c, "candidate_entity_ids", f"{OUT}/candidate_pairs.tsv")
    print("matched per S1", sel.height / len(ids1), cfg)
