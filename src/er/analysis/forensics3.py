import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polars as pl
from common import DATA as O; r=lambda n: pl.read_parquet(f"{O}/{n}.parquet")
s1=r("train_source1"); gt=r("train_ground_truth"); o=pl.concat([r("train_source2"), r("train_source3")])
pairs=gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids")!="").select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("x")).sample(300000,seed=0)
tok=lambda c: pl.col(c).str.to_lowercase().str.extract_all(r"[a-z]{3,}|[0-9]+").list.unique()
A=pairs.join(s1.select(pl.col("entity_id").alias("s1"),tok("business_address").alias("a1"),tok("business_name").alias("n1")),on="s1").join(o.select(pl.col("entity_id").alias("x"),tok("business_address").alias("ax"),tok("business_name").alias("nx"),(pl.col("business_address")=="").alias("ae")),on="x")
num=lambda c: pl.col(c).list.eval(pl.element().filter(pl.element().str.contains(r"^[0-9]+$")))
A=A.with_columns(
 (pl.col("a1").list.set_intersection("ax").list.len()/pl.col("a1").list.set_union("ax").list.len()).alias("aj"),
 (pl.col("n1").list.set_intersection("nx").list.len()/pl.col("n1").list.set_union("nx").list.len()).alias("nj"),
 num("a1").alias("u1"), num("ax").alias("ux"))
A=A.with_columns((pl.col("u1").list.set_intersection("ux").list.len()>0).alias("numshare"), (pl.col("ux").list.len()==0).alias("xnonum"))
f=A.filter(~pl.col("ae"))
print("addr jaccard quantiles (non-empty addr):", [round(f["aj"].quantile(q),3) for q in (.01,.05,.1,.25,.5)])
print("name jaccard quantiles:", [round(A["nj"].quantile(q),3) for q in (.01,.05,.1,.25,.5)])
print("share >=1 number:", f["numshare"].mean(), " x has no number:", f["xnonum"].mean())
print("name jaccard==0:", (A["nj"]==0).mean(), " addr jacc<0.2 among nonempty:", (f["aj"]<0.2).mean(), " both weak (nj<.2 & aj<.3):", ((A["nj"]<0.2)&(A["aj"]<0.3)).mean())
print("\n== weakest true pairs")
S=pl.concat([s1.select("entity_id","business_name","business_address"), o.select("entity_id","business_name","business_address")])
w=A.filter((pl.col("nj")<0.2)&(pl.col("aj")<0.35)).head(15)
for x in w.iter_rows(named=True):
    a=S.filter(pl.col("entity_id")==x["s1"]).row(0); b=S.filter(pl.col("entity_id")==x["x"]).row(0)
    print(f" {a[1]} | {a[2]}\n   -> {b[1]} | {b[2]}")
