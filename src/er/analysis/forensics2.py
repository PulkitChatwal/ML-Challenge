import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polars as pl
from common import DATA as O
r=lambda n: pl.read_parquet(f"{O}/{n}.parquet")
s1=r("train_source1"); gt=r("train_ground_truth")
o=pl.concat([r("train_source2"), r("train_source3")])
pairs=gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids")!="").select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("entity_id"))
o=o.join(pairs, on="entity_id", how="left")
LEG=r"\b(llc|l l c|inc|corp|corporation|co|company|ltd|limited|pvt|private|llp|pllc|lp|plc|sa|sas|sarl|group|services|service|center|centre|the|and|of)\b"
def nk(c): return pl.col(c).str.to_lowercase().str.replace_all(r"[^a-z0-9 ]"," ").str.replace_all(LEG," ").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars()>0)).list.unique().list.sort().list.join(" ")
s1=s1.with_columns(nk("business_name").alias("nk"))
o=o.with_columns(nk("business_name").alias("nk"))
print("== S1 duplicate structure")
print("S1 rows sharing name-key with another S1 (same country):", s1.group_by("country","nk").len().filter(pl.col("len")>1)["len"].sum()/s1.height)
print("S1 exact dup name+addr:", s1.height - s1.select("business_name","business_address").n_unique())
print("S1 rows sharing exact address:", s1.group_by("business_address").len().filter(pl.col("len")>1)["len"].sum()/s1.height)
print("top repeated S1 name-keys:", s1.group_by("nk").len().sort("len",descending=True).head(15).rows())
um=o.filter(pl.col("s1").is_null()); m=o.filter(pl.col("s1").is_not_null())
print("== unmatched S2/S3:", um.height, "by country", um["country"].value_counts().rows())
s1k=s1.select("nk","country").unique()
print("unmatched whose name-key equals some S1 name-key:", um.join(s1k,on=["nk","country"],how="semi").height/um.height)
print("matched whose name-key equals some S1 name-key:", m.join(s1k,on=["nk","country"],how="semi").height/m.height)
print("matched whose name-key equals its OWN S1 name-key:", m.join(s1.select(pl.col("entity_id").alias("s1"),pl.col("nk").alias("nk1")),on="s1").select((pl.col("nk")==pl.col("nk1")).mean()).item())
# do unmatched records cluster among themselves?
uk=um.group_by("country","nk").len()
print("unmatched sharing name-key with another unmatched:", uk.filter(pl.col("len")>1)["len"].sum()/um.height)
print("\n== unmatched samples")
for x in um.sample(25,seed=5).iter_rows(named=True): print(x["entity_id"][:2], x["country"], "|", x["business_name"], "|", x["business_address"])
# singletons: nearest S1-name-key hit in S2/S3?
sing=gt.filter(pl.col("matched_entity_ids")=="").select(pl.col("source1_entity_id").alias("entity_id"))
ss=s1.join(sing,on="entity_id")
print("\n== singleton S1 whose name-key appears in S2/S3:", ss.join(o.select("nk","country").unique(),on=["nk","country"],how="semi").height/ss.height)
print("non-singleton S1 whose name-key appears in S2/S3:", s1.join(sing,on="entity_id",how="anti").join(o.select("nk","country").unique(),on=["nk","country"],how="semi").height/(s1.height-ss.height))
hit=ss.join(o.filter(pl.col("s1").is_not_null()).select("nk","country","business_name","business_address","s1"),on=["nk","country"]).head(12)
print("singleton S1 vs S2/S3 record with same name-key (record belongs to another S1):")
for x in hit.iter_rows(named=True):
    print(" S1:", x["business_name"],"|",x["business_address"]); print("  X:", x["business_name_right"],"|",x["business_address_right"], "-> owner", x["s1"])
# name form stats among matched
print("\n== name forms among matched by source")
m=m.with_columns(src=pl.col("entity_id").str.slice(0,2))
print(m.group_by("src","country").agg(
  pl.col("business_name").str.contains(r"(?i)\.(com|in|net|org|co)\b").mean().alias("domain"),
  pl.col("business_name").str.contains(r"(?i)doing business as|\bdba\b|t/a|trading as").mean().alias("dba"),
  pl.col("business_name").str.contains(r"[ऀ-ॿ]").mean().alias("deva"),
  pl.col("business_name").str.contains(r"^[^a-z]*$").mean().alias("allcaps"),
  pl.col("business_name").str.contains(r"[\[\]\(\)<>#\-]{2,}|^[^A-Za-z0-9ऀ-ॿ]").mean().alias("junkpunct"),
  (pl.col("business_address")=="").mean().alias("addr_empty"),
).sort("src","country").rows())
