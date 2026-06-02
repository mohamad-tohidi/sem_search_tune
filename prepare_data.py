import json
import random
from pathlib import Path

SRC_DIR = Path("dataset")
DATA_DIR = Path("data")
FILES = ["001.json", "002.json", "003.json"]
SPLIT = 0.9

random.seed(42)
DATA_DIR.mkdir(exist_ok=True)

all_records = []
for fname in FILES:
    with open(SRC_DIR / fname) as f:
        records = json.load(f)
    print(f"{fname}: {len(records)} records")
    all_records.extend(records)

print(f"Total: {len(all_records)} records")

random.shuffle(all_records)

split_idx = int(len(all_records) * SPLIT)
train = all_records[:split_idx]
eval = all_records[split_idx:]


def write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


write_jsonl(DATA_DIR / "train.jsonl", train)
write_jsonl(DATA_DIR / "val.jsonl", eval)

print(f"Train: {len(train)} records -> data/train.jsonl")
print(f"Eval:  {len(eval)} records -> data/val.jsonl")
