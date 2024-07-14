import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import config, load_dataset

config.HF_DATASETS_CACHE = "./cache"
import json

ds = load_dataset("fancyzhx/amazon_polarity")

print(ds["train"][0])

with open("dataset/train.jsonl", "w", encoding="utf-8") as fout:
    for i in list(ds["train"])[:200000]:
        fout.write(json.dumps(i) + "\n")

with open("dataset/test.jsonl", "w", encoding="utf-8") as fout:
    for i in list(ds["test"])[:20000]:
        fout.write(json.dumps(i) + "\n")
