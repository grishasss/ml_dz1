"""Common full-catalog top-K metrics and paired user-bootstrap uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    per_user: pd.DataFrame
    recommendations: np.ndarray


def ranking_metrics(recommendations: np.ndarray, users: np.ndarray,
                    relevance: sparse.csr_matrix, k: int = 10) -> pd.DataFrame:
    """Macro binary NDCG, recall, hit rate and precision; user IDs are explicit.

    Padded recommendation IDs (-1) are counted as misses. Evaluation users must
    have at least one relevant candidate; otherwise recall is not well defined.
    """
    users = np.asarray(users, dtype=np.int64)
    recommendations = np.asarray(recommendations, dtype=np.int64)
    if recommendations.shape != (len(users), k):
        raise ValueError("recommendations must have one K-sized row per evaluation user")
    if len(np.unique(users)) != len(users):
        raise ValueError("Evaluation users must be unique")
    discount = 1 / np.log2(np.arange(2, k + 2))
    ideal = np.cumsum(discount)
    result = []
    for user, ranking in zip(users, recommendations):
        truth = relevance.indices[relevance.indptr[user]:relevance.indptr[user + 1]]
        if len(truth) == 0:
            raise ValueError(f"User {user} has no relevant candidates")
        valid = ranking[ranking >= 0]
        if len(np.unique(valid)) != len(valid):
            raise ValueError("A recommendation list contains duplicate items")
        hits = np.isin(ranking, truth) & (ranking >= 0)
        count = int(hits.sum())
        result.append((int(user), float((hits * discount).sum() / ideal[min(len(truth), k) - 1]),
                       count / len(truth), float(count > 0), count / k, len(truth)))
    return pd.DataFrame(result, columns=["user", f"NDCG@{k}", f"Recall@{k}",
                                         f"HitRate@{k}", f"Precision@{k}", "n_relevant"])


def evaluate(model, users: np.ndarray, relevance: sparse.csr_matrix,
             seen: sparse.csr_matrix, k: int = 10, batch_size: int = 128) -> EvaluationResult:
    """Evaluate every eligible user over every train-catalog item, in batches."""
    users = np.asarray(users, dtype=np.int64)
    if len(users) == 0:
        raise ValueError("No eligible users to evaluate")
    if relevance.shape != seen.shape:
        raise ValueError("Relevance and history must use identical user/item vocabularies")
    candidates = np.ones(seen.shape[1], dtype=bool)
    parts = []
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        ranking = np.asarray(model.recommend(batch, seen=seen, candidates=candidates, k=k))
        # Guards catch accidental leakage in any model's recommendation path.
        for user, row in zip(batch, ranking):
            valid = row[row >= 0]
            if np.any(valid >= seen.shape[1]) or len(np.unique(valid)) != len(valid):
                raise ValueError("Model returned invalid/duplicate candidates")
            if seen[user, valid].nnz:
                raise ValueError("Model recommended an item already observed by the user")
        parts.append(ranking)
    recommendations = np.concatenate(parts, axis=0)
    per_user = ranking_metrics(recommendations, users, relevance, k)
    columns = [f"NDCG@{k}", f"Recall@{k}", f"HitRate@{k}", f"Precision@{k}"]
    metrics = {name: float(per_user[name].mean()) for name in columns}
    metrics[f"Coverage@{k}"] = float(len(np.unique(recommendations[recommendations >= 0])) / seen.shape[1])
    metrics["n_users"] = len(users)
    return EvaluationResult(metrics, per_user, recommendations)


def paired_bootstrap(first: pd.DataFrame, second: pd.DataFrame, metric: str = "NDCG@10",
                     seed: int = 42, n_resamples: int = 2000,
                     confidence: float = 0.95) -> dict[str, float]:
    """User bootstrap for mean(first - second), pairing the same user IDs."""
    a = first.set_index("user")[metric].sort_index()
    b = second.set_index("user")[metric].sort_index()
    if not a.index.equals(b.index) or len(a) == 0:
        raise ValueError("Paired bootstrap requires the same nonempty user set")
    differences = (a - b).to_numpy()
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, 100):
        count = min(100, n_resamples - start)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        means[start:start + count] = differences[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return {"difference": float(differences.mean()), "ci_low": float(low), "ci_high": float(high),
            "confidence": confidence, "n_resamples": n_resamples, "seed": seed}
