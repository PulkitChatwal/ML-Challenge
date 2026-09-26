import sys, polars as pl, random, torch
from sentence_transformers import SentenceTransformer, losses, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from datasets import Dataset
from common import *
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
s1, o = load("train")
p = gt_pairs().with_columns(fold_of("s1").alias("f")).filter(pl.col("f").is_in(list(EMB_FOLDS)))
p = p.join(s1.select(pl.col("entity_id").alias("s1"), text_expr().alias("a")), on="s1").join(o.select(pl.col("entity_id").alias("x"), text_expr().alias("b")), on="x")
p = p.sample(min(N, p.height), seed=0, shuffle=True)
print("train pairs", p.height, flush=True)
ds = Dataset.from_dict({"anchor": ("query: " + p["a"]).to_list(), "positive": ("query: " + p["b"]).to_list()})
m = SentenceTransformer("intfloat/multilingual-e5-small", device="cuda"); m.max_seq_length = 72
loss = losses.MultipleNegativesSymmetricRankingLoss(m, scale=30)
args = SentenceTransformerTrainingArguments(output_dir=f"{WORK}/bienc_ckpt", num_train_epochs=1, per_device_train_batch_size=512,
    learning_rate=1e-4, warmup_ratio=0.03, fp16=True, batch_sampler=BatchSamplers.NO_DUPLICATES, logging_steps=200,
    save_strategy="no", report_to="none", dataloader_num_workers=4, lr_scheduler_type="cosine")
SentenceTransformerTrainer(model=m, args=args, train_dataset=ds, loss=loss).train()
m.save(f"{WORK}/bienc")
print("saved")
