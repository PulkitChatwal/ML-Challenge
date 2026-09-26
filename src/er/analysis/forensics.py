import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polars as pl, re, sys, random
from common import DATASET as D
rd = lambda p: (pl.read_csv(p, separator="\t", quote_char=None, infer_schema_length=0, missing_utf8_is_empty_string=True).fill_null(""))
tr = {i: rd(f"{D}/train/train_source{i}.tsv") for i in (1,2,3)}
te = {i: rd(f"{D}/test/test_source{i}.tsv") for i in (1,2,3)}
gt = rd(f"{D}/train/train_ground_truth.tsv")
print("== shapes"); [print(k, i, d.shape) for k, s in (("train",tr),("test",te)) for i, d in s.items()]
print("== country by source")
for k, s in (("train",tr),("test",te)):
    for i, d in s.items(): print(k, i, d["country"].value_counts().sort("count", descending=True).rows())
dev = r"[ऀ-ॿ]"
print("== devanagari / empty rates")
for k, s in (("train",tr),("test",te)):
    for i, d in s.items():
        print(k, i, d.group_by("country").agg(
            pl.col("business_name").str.contains(dev).mean().alias("name_dev"),
            pl.col("business_address").str.contains(dev).mean().alias("addr_dev"),
            (pl.col("business_address")=="").mean().alias("addr_empty"),
            (pl.col("business_name")=="").mean().alias("name_empty")).sort("country").rows())
# ground truth
g = gt.with_columns(pl.col("matched_entity_ids").str.split(",").list.eval(pl.element().filter(pl.element()!="")).alias("m"))
g = g.with_columns(pl.col("m").list.len().alias("n"),
    pl.col("m").list.eval(pl.element().str.starts_with("S2-")).list.sum().alias("n2"))
g = g.with_columns((pl.col("n")-pl.col("n2")).alias("n3"))
print("== GT rows", g.height, "S1 rows", tr[1].height, "S1 in GT", tr[1]["entity_id"].is_in(g["source1_entity_id"]).mean())
print("singleton rate", (g["n"]==0).mean())
print("n dist", g["n"].value_counts().sort("n").rows()[:25])
print("n2 dist", g["n2"].value_counts().sort("n2").rows()[:15])
print("n3 dist", g["n3"].value_counts().sort("n3").rows()[:15])
pairs = g.select("source1_entity_id","m").explode("m").drop_nulls("m").rename({"source1_entity_id":"s1","m":"x"})
print("pairs", pairs.height, "unique x", pairs["x"].n_unique(), "-> exclusivity violations", pairs.height-pairs["x"].n_unique())
all23 = pl.concat([tr[2]["entity_id"], tr[3]["entity_id"]])
print("S2 matched frac", tr[2]["entity_id"].is_in(pairs["x"]).mean(), "S3 matched frac", tr[3]["entity_id"].is_in(pairs["x"]).mean())
print("GT ids missing from sources", (~pairs["x"].is_in(all23)).sum())
s1c = tr[1].select(pl.col("entity_id").alias("s1"), pl.col("country").alias("c1"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"))
o = pl.concat([tr[2], tr[3]]).select(pl.col("entity_id").alias("x"), pl.col("country").alias("cx"), pl.col("business_name").alias("nx"), pl.col("business_address").alias("ax"))
P = pairs.join(s1c, on="s1").join(o, on="x")
print("country agree", (P["c1"]==P["cx"]).mean(), P.filter(pl.col("c1")!=pl.col("cx")).group_by("c1","cx").len().rows()[:10])
g2 = g.join(s1c, left_on="source1_entity_id", right_on="s1")
print("singleton rate by country", g2.group_by("c1").agg((pl.col("n")==0).mean().alias("single"), pl.col("n").mean().alias("mean_n"), pl.len()).rows())
norm = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", "").str.replace_all(r"\s+"," ").str.strip_chars()
P = P.with_columns(src=pl.col("x").str.slice(0,2))
print("exact name eq", P.group_by("src","c1").agg((norm("n1")==norm("nx")).mean().alias("name_eq"), (norm("a1")==norm("ax")).mean().alias("addr_eq"), (pl.col("ax")=="").mean().alias("ax_empty"), pl.col("nx").str.contains(dev).mean().alias("nx_dev"), pl.col("n1").str.contains(dev).mean().alias("n1_dev")).sort("src","c1").rows())
# unmatched S2/S3 name overlap with S1 (distractor check)
random.seed(0)
print("== sample groups")
samp = g2.filter(pl.col("n")>0).sample(12, seed=1)
idx = o.select("x","nx","ax")
for r in samp.iter_rows(named=True):
    print(f"\nS1 [{r['c1']}] {r['n1']} | {r['a1']}")
    for x in r["m"]:
        rr = idx.filter(pl.col("x")==x).row(0)
        print(f"   {x[:2]} {rr[1]} | {rr[2]}")
print("\n== sample singletons")
for r in g2.filter(pl.col("n")==0).sample(8, seed=2).iter_rows(named=True): print(f"S1 [{r['c1']}] {r['n1']} | {r['a1']}")
print("\n== France test samples")
for i in (1,2,3): print(te[i].filter(pl.col("country")=="France").sample(6, seed=3).rows())
print("== all test country labels", sorted(set(pl.concat([te[i]["country"] for i in (1,2,3)]).unique().to_list())))
print("== all train country labels", sorted(set(pl.concat([tr[i]["country"] for i in (1,2,3)]).unique().to_list())))
