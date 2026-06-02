from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score as ap_score,
    dcg_score,
    f1_score as f1,
    ndcg_score,
    precision_score as prec,
    recall_score as rec,
)


def _uniq(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in items if not (x in seen or seen.add(x))]


class MetricsCalculator:
    @staticmethod
    def precision_at_k(
        retrieved_items: list[str],
        relevant_items: list[str],
        k: int,
        *,
        deduplicate_retrieved: bool = True,
    ) -> tuple[float, dict[str, Any]]:
        relevant_set = set(relevant_items)
        if k <= 0:
            return 0.0, {}
        topk = _uniq(retrieved_items[:k]) if deduplicate_retrieved else retrieved_items[:k]
        u = list(dict.fromkeys(retrieved_items + list(relevant_set)))
        yt = np.array([1 if i in relevant_set else 0 for i in u])
        yp = np.array([1 if i in topk else 0 for i in u])
        rr = int(yt @ yp)
        return float(prec(yt, yp, zero_division=0.0)), {
            "meta": {"k": k, "retrieved_len": len(retrieved_items), "relevant_total": len(relevant_set), "relevant_retrieved": rr}
        }

    @staticmethod
    def recall_at_k(
        retrieved_items: list[str],
        relevant_items: list[str],
        k: int,
        *,
        deduplicate_retrieved: bool = True,
    ) -> tuple[float, dict[str, Any]]:
        relevant_set = set(relevant_items)
        total = len(relevant_set)
        if total == 0 or k <= 0:
            return 0.0, {}
        topk = _uniq(retrieved_items[:k]) if deduplicate_retrieved else retrieved_items[:k]
        u = list(dict.fromkeys(retrieved_items + list(relevant_set)))
        yt = np.array([1 if i in relevant_set else 0 for i in u])
        yp = np.array([1 if i in topk else 0 for i in u])
        rr = int(yt @ yp)
        return float(rec(yt, yp, zero_division=0.0)), {
            "meta": {"k": k, "retrieved_len": len(retrieved_items), "relevant_total": total, "relevant_retrieved": rr}
        }

    @staticmethod
    def f1_score(
        retrieved_items: list[str],
        relevant_items: list[str],
        k: int,
        *,
        deduplicate_retrieved: bool = True,
    ) -> tuple[float, dict[str, Any]]:
        relevant_set = set(relevant_items)
        if k <= 0:
            return 0.0, {}
        topk = _uniq(retrieved_items[:k]) if deduplicate_retrieved else retrieved_items[:k]
        u = list(dict.fromkeys(retrieved_items + list(relevant_set)))
        yt = np.array([1 if i in relevant_set else 0 for i in u])
        yp = np.array([1 if i in topk else 0 for i in u])
        rr = int(yt @ yp)
        p = float(prec(yt, yp, zero_division=0.0))
        r = float(rec(yt, yp, zero_division=0.0))
        return float(f1(yt, yp, zero_division=0.0)), {
            "meta": {"k": k, "retrieved_len": len(retrieved_items), "relevant_total": len(relevant_set), "relevant_retrieved": rr},
            "precision": p, "recall": r,
        }

    @staticmethod
    def average_precision(
        retrieved: list[str],
        relevant: list[str],
        *,
        k: int | None = None,
        deduplicate_retrieved: bool = True,
    ) -> tuple[float, dict[str, Any]]:
        relevant_set = set(relevant)
        if not relevant_set:
            return 0.0, {}
        ranked = _uniq(retrieved[:k]) if deduplicate_retrieved else (retrieved if k is None else retrieved[:k])
        if not ranked:
            return 0.0, {}
        u = list(dict.fromkeys(retrieved + list(relevant_set)))
        idx = {i: p for p, i in enumerate(u)}
        yt = np.array([1 if i in relevant_set else 0 for i in u])
        ys = np.zeros(len(u))
        for p, item in enumerate(ranked, 1):
            ys[idx[item]] = len(ranked) - p + 1
        return float(ap_score(yt, ys)), {
            "k": k, "relevant_count": len(relevant_set), "counted_relevant": len(relevant_set & set(ranked)),
        }

    def mean_average_precision(
        self,
        queries_data: list[tuple[list[str], list[str]]],
        *,
        k: int | None = None,
        deduplicate_retrieved: bool = True,
    ) -> tuple[float, dict[str, Any]]:
        if not queries_data:
            return 0.0, {}
        scores, per_q = [], []
        for ret, rel in queries_data:
            ap, meta = self.average_precision(ret, rel, k=k, deduplicate_retrieved=deduplicate_retrieved)
            scores.append(ap)
            per_q.append(meta)
        mean = float(np.mean(scores))
        return mean, {
            "map": mean, "ap_scores": scores, "num_queries": len(scores), "k": k,
            "min_ap": float(np.min(scores)), "max_ap": float(np.max(scores)), "per_query_metadata": per_q,
        }

    @staticmethod
    def reciprocal_rank(
        retrieved: list[str],
        relevant: list[str],
        *,
        deduplicate_retrieved: bool = True,
    ) -> tuple[float, dict[str, Any]]:
        ranked = _uniq(retrieved) if deduplicate_retrieved else retrieved
        relevant_set = set(relevant)
        arr = np.array(ranked)
        matches = np.isin(arr, list(relevant_set))
        first = int(np.argmax(matches)) if np.any(matches) else -1
        rr = 1.0 / (first + 1) if first >= 0 else 0.0
        return rr, {"first_relevant_position": first + 1 if first >= 0 else None}

    def mean_reciprocal_rank(
        self,
        queries_data: list[tuple[list[str], list[str]]],
        *,
        deduplicate_retrieved: bool = True,
    ) -> tuple[float, dict[str, Any]]:
        if not queries_data:
            return 0.0, {}
        rrs, positions = [], []
        for ret, rel in queries_data:
            rr, meta = self.reciprocal_rank(ret, rel, deduplicate_retrieved=deduplicate_retrieved)
            rrs.append(rr)
            positions.append(meta["first_relevant_position"])
        mrr = float(np.mean(rrs))
        valid = [p for p in positions if p is not None]
        avg = float(np.mean(valid)) if valid else None
        return mrr, {
            "mrr": mrr, "rr_scores": rrs, "first_positions": positions, "num_queries": len(rrs),
            "queries_with_results": sum(1 for s in rrs if s > 0), "avg_first_position": avg,
        }

    @staticmethod
    def ndcg_at_k(
        retrieved_items: list[str],
        relevance: dict[str, float] | list[str] | set[str],
        k: int,
        *,
        gain_scheme: str = "exp2",
        deduplicate_retrieved: bool = True,
    ) -> tuple[float, dict[str, Any]]:
        if k <= 0:
            return 0.0, {}
        if isinstance(relevance, dict):
            rel = {str(d): float(s) for d, s in relevance.items()}
        else:
            rel = {str(d): 1.0 for d in set(relevance)}
        ranked = _uniq(retrieved_items[:k]) if deduplicate_retrieved else retrieved_items[:k]
        u = list(dict.fromkeys(retrieved_items + list(rel.keys())))
        idx = {d: i for i, d in enumerate(u)}
        n = len(u)
        yt = np.zeros((1, n))
        ys = np.zeros((1, n))
        for d, r in rel.items():
            yt[0, idx[d]] = r
        for p, d in enumerate(ranked, 1):
            ys[0, idx[d]] = n - p
        ndcg = float(ndcg_score(yt, ys, k=k))
        dcg_val = float(dcg_score(yt, ys, k=k))
        ideal = np.argsort(-yt[0])
        yi = np.zeros((1, n))
        for i, ix in enumerate(ideal):
            yi[0, ix] = n - i
        idcg_val = float(dcg_score(yt, yi, k=k))
        return ndcg, {
            "k": k, "gain_scheme": gain_scheme,
            "dcg": dcg_val, "idcg": idcg_val,
            "used_rels": [rel.get(d, 0.0) for d in ranked],
            "ideal_rels": sorted(rel.values(), reverse=True)[:k],
        }
