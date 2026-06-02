# Semantic Search Fine-Tuning

Fine-tune sentence embedding models for semantic search.

## Commands

### Dry run (quick sanity check)

Run a fast verification with 64 training and 16 evaluation samples:

```bash
# Set dry_run: true in config.yaml, then:
python train.py --config config.yaml
```

Set `dry_run: false` in config.yaml (or remove it) to run the full training.

### Full training

```bash
python train.py --config config.yaml
```

## Configuration

Edit `config.yaml` to control model, data paths, hyperparameters, and benchmark settings.

## Dataset format

Training expects **JSONL** files (one JSON object per line). Each record must have at least `"query"` and `"passage"` fields:

```jsonl
{"query": "sample query", "passage": "relevant passage"}
```

Extra fields (`id`, `title`, etc.) are allowed — they get stripped automatically by the trainer.

### Prepare data from raw JSON files

```bash
python prepare_data.py
```

This reads `dataset/001.json`, `002.json`, `003.json`, shuffles, splits 90/10, and writes `data/train.jsonl` and `data/val.jsonl`.
