import argparse
import json
import tempfile
from pathlib import Path

import tqdm
import yaml
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer,
)

from sentence_transformers.sentence_transformer import losses

import plotly.graph_objects as go

from metrics import MetricsCalculator


def load_data(config, dry_run=False):
    dataset = load_dataset(
        "json",
        data_files={
            "train": config["train_data"],
            "validation": config["eval_data"],
        },
    )
    if dry_run:
        n_train = config.get("dry_run_samples", 64)
        n_eval = config.get("dry_run_eval_samples", 16)
        dataset["train"] = dataset["train"].select(
            range(min(n_train, len(dataset["train"])))
        )
        dataset["validation"] = dataset["validation"].select(
            range(min(n_eval, len(dataset["validation"])))
        )
    return dataset


def build_model(config):
    model = SentenceTransformer(config["model_name"])
    model.max_seq_length = config.get("max_seq_length", 512)

    loss_name = config["loss"]
    if loss_name == "mnrl":
        train_loss = losses.MultipleNegativesRankingLoss(model)
    elif loss_name == "cosine":
        train_loss = losses.CosineSimilarityLoss(model)
    elif loss_name == "triplet":
        train_loss = losses.TripletLoss(model)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    return model, train_loss


def _chunk(text: str) -> list[str]:
    text = " ".join(text.split(" ")[:250])
    return [text]


def _eval_model(
    model,
    raw_benchmarks: list[dict],
    raw_docs: list[dict],
    benchmark_cfg: dict,
) -> tuple:
    k = benchmark_cfg.get("k", 10)
    query_prefix = benchmark_cfg.get("query_prefix", "")
    passage_prefix = benchmark_cfg.get("passage_prefix", "")

    def embed_passage(text: str) -> list[float]:
        return (
            model.encode(
                f"{passage_prefix}{text}",
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            .astype(float)
            .tolist()
        )

    def embed_query(text: str) -> list[float]:
        return (
            model.encode(
                f"{query_prefix}{text}",
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            .astype(float)
            .tolist()
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        client = QdrantClient(path=tmpdir)
        collection_name = "eval"

        sample_vec = embed_passage(raw_docs[0]["body"])
        vector_size = len(sample_vec)

        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size, distance=Distance.COSINE
            ),
        )

        qdrant_data = [
            (doc["id"], embed_passage(chunk), chunk)
            for doc in tqdm.tqdm(
                raw_docs, desc="Chunking & embedding docs"
            )
            for chunk in _chunk(doc["body"])
        ]

        print("Indexing in local Qdrant...")
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=idx,
                    vector=vector,
                    payload={
                        "original_id": str(doc_id),
                        "text": text,
                    },
                )
                for idx, (doc_id, vector, text) in enumerate(
                    qdrant_data
                )
            ],
        )

        calculator = MetricsCalculator()
        queries_data = []
        eval_per_item = []

        print("Evaluating benchmark queries...")
        for bench in tqdm.tqdm(raw_benchmarks, desc="Evaluating"):
            query = bench["query"]
            relevant_ids = [
                str(id) for id in bench["rank"][0]["ranker"]["ranked"]
            ]

            query_vec = embed_query(query)
            search_result = client.query_points(
                collection_name=collection_name,
                query=query_vec,
                limit=k * 10,
            )
            retrieved_ids = [
                point.payload["original_id"]
                for point in search_result.points
            ]
            retrieved_ids = [str(id) for id in retrieved_ids]

            queries_data.append((retrieved_ids, relevant_ids))

            eval_per_item.append(
                {
                    "f@k": calculator.f1_score(
                        retrieved_ids, relevant_ids, k
                    ),
                    "average_precision": calculator.average_precision(
                        retrieved_ids, relevant_ids, k=k
                    ),
                    "reciprocal_rank": calculator.reciprocal_rank(
                        retrieved_ids, relevant_ids
                    ),
                    "ndcg@k": calculator.ndcg_at_k(
                        retrieved_ids, relevant_ids, k=k
                    ),
                }
            )

        total_eval = {
            "mean_average_precision": calculator.mean_average_precision(
                queries_data, k=k
            ),
            "mean_reciprocal_rank": calculator.mean_reciprocal_rank(
                queries_data
            ),
        }

    return total_eval, eval_per_item, queries_data


def plot_benchmark_comparison(
    base_total: dict,
    ft_total: dict,
    results_dir: Path,
    run_name: str,
) -> None:
    base_map = base_total["mean_average_precision"][0]
    base_mrr = base_total["mean_reciprocal_rank"][0]
    ft_map = ft_total["mean_average_precision"][0]
    ft_mrr = ft_total["mean_reciprocal_rank"][0]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Base Model", x=["MAP", "MRR"], y=[base_map, base_mrr])
    )
    fig.add_trace(
        go.Bar(name="Fine-tuned", x=["MAP", "MRR"], y=[ft_map, ft_mrr])
    )
    fig.update_layout(
        title=f"Benchmark Comparison — {run_name}",
        barmode="group",
        template="plotly_white",
        yaxis_title="Score",
    )
    path = results_dir / "benchmark_comparison.html"
    fig.write_html(str(path))
    print(f"Benchmark comparison plot saved to {path}")


def plot_training_history(trainer, output_dir: Path, run_name: str) -> None:
    log_history = trainer.state.log_history
    if not log_history:
        return

    train_steps = [
        e["step"] for e in log_history if "loss" in e and "eval_loss" not in e
    ]
    train_losses = [
        e["loss"] for e in log_history if "loss" in e and "eval_loss" not in e
    ]
    eval_steps = [e["step"] for e in log_history if "eval_loss" in e]
    eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]

    if not train_losses and not eval_losses:
        return

    fig = go.Figure()
    if train_losses:
        fig.add_trace(
            go.Scatter(
                x=train_steps,
                y=train_losses,
                mode="lines",
                name="Training Loss",
            )
        )
    if eval_losses:
        fig.add_trace(
            go.Scatter(
                x=eval_steps,
                y=eval_losses,
                mode="lines+markers",
                name="Eval Loss",
            )
        )
    fig.update_layout(
        title=f"Training History — {run_name}",
        xaxis_title="Step",
        yaxis_title="Loss",
        template="plotly_white",
    )
    path = output_dir / "training_history.html"
    fig.write_html(str(path))
    print(f"Training history plot saved to {path}")


def evaluate_benchmark(config: dict, model_dir: Path) -> Path:
    benchmark_cfg = config.get("benchmark", {})

    results_dir = model_dir.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    model_name = config.get("model_name", "unknown")
    run_name = config.get("run_name", model_dir.parent.name)

    data_dir = benchmark_cfg.get("data_dir")
    if not data_dir:
        print("No benchmark.data_dir in config, skipping evaluation.")
        return results_dir

    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"Benchmark data not found at {data_dir}, skipping evaluation.")
        return results_dir

    with open(data_dir / "benchmark.json") as f:
        raw_benchmarks = json.load(f)
    with open(data_dir / "rawdata.json") as f:
        raw_docs = json.load(f)

    # Evaluate base model
    print(f"Loading base model: {model_name}")
    base_model = SentenceTransformer(model_name, trust_remote_code=True)
    base_total, _, _ = _eval_model(base_model, raw_benchmarks, raw_docs, benchmark_cfg)
    del base_model

    # Evaluate fine-tuned model
    print(f"Loading fine-tuned model from {model_dir}")
    ft_model = SentenceTransformer(str(model_dir), trust_remote_code=True)
    ft_total, ft_per_item, ft_queries_data = _eval_model(ft_model, raw_benchmarks, raw_docs, benchmark_cfg)
    del ft_model

    # Save fine-tuned metrics
    metrics = {
        "experiment_name": run_name,
        "description": f"Fine-tuned model: {model_name}",
        "total_eval": ft_total,
        "eval_per_item": ft_per_item,
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(results_dir / "report.md", "w") as f:
        f.write(f"# Benchmark: {run_name}\n\n")
        f.write(f"**Model:** {model_name}\n\n")
        f.write(f"**Total Score:**\n{json.dumps(ft_total, indent=4)}\n\n")

    # Comparison plot
    plot_benchmark_comparison(base_total, ft_total, results_dir, run_name)

    # Qualitative samples (from fine-tuned only)
    k = benchmark_cfg.get("k", 10)
    doc_lookup = {str(d["id"]): d["body"] for d in raw_docs}
    samples = sorted(
        [
            {
                "score": item["ndcg@k"],
                "benchmark": bench,
                "metrics": item,
                "retrieved_ids": qdata[0],
                "relevant_ids": set(qdata[1]),
            }
            for item, bench, qdata in zip(
                ft_per_item, raw_benchmarks, ft_queries_data
            )
        ],
        key=lambda x: x["score"][0],
        reverse=True,
    )

    with open(results_dir / "qualitative_samples.md", "w", encoding="utf-8") as f:
        f.write(f"# Qualitative Analysis — {run_name}\n\n")
        f.write(f"**Model:** {model_name}\n\n")
        f.write("Samples ranked by NDCG@k.\n\n")

        for label, subset in [
            ("Top 5 — Best Retrievals", samples[:5]),
            ("Bottom 5 — Worst Retrievals", samples[-5:]),
        ]:
            f.write(f"## {label}\n\n")
            for sample in subset:
                bench = sample["benchmark"]
                m = sample["metrics"]
                rids = sample["retrieved_ids"]
                rel_set = sample["relevant_ids"]

                f.write(f"### Query: {bench['query']}\n\n")
                f.write(
                    f"- NDCG@{k}: `{m['ndcg@k'][0]:.4f}`  "
                    f"F1@{k}: `{m['f@k'][0]:.4f}`  "
                    f"AP: `{m['average_precision'][0]:.4f}`  "
                    f"RR: `{m['reciprocal_rank'][0]:.4f}`\n\n"
                )
                f.write("**Top 10 retrieved:**\n\n")
                for rank, doc_id in enumerate(rids[:10], 1):
                    status = "✅" if doc_id in rel_set else "❌"
                    snippet = (doc_lookup.get(doc_id, "") or "")[:200]
                    f.write(f"{rank:2d}. `{doc_id}` {status}  {snippet}\n")
                f.write(f"\n**Ground truth:** {sorted(rel_set)}\n\n---\n\n")

    print(f"Benchmark results saved to {results_dir}")
    return results_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dry_run = config.get("dry_run", False)

    dataset = load_data(config, dry_run=dry_run)
    for split in dataset:
        dataset[split] = dataset[split].select_columns(
            ["query", "passage"]
        )
    model, loss = build_model(config)

    if dry_run:
        output_dir = Path("results/dry_run")
    else:
        output_dir = config.get("output_dir")
        if output_dir:
            output_dir = Path(output_dir)
        else:
            run_name = config.get("run_name", "default")
            output_dir = Path("results") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        learning_rate=float(config["lr"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=str(output_dir / "logs"),
    )

    if dry_run:
        training_args.max_steps = 1

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        loss=loss,
    )
    trainer.train()

    plot_training_history(trainer, output_dir, output_dir.name)

    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))

    if dry_run:
        print("Dry run completed successfully — all systems go.")
        return

    print(f"Training complete. Model saved to {final_dir}")

    if "benchmark" in config:
        evaluate_benchmark(config, model_dir=final_dir)


if __name__ == "__main__":
    main()
