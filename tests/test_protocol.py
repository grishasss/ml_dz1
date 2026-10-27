"""Analytical checks for temporal leakage and manually calculated top-K metrics."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from recsys.data import build_protocol
from recsys.evaluation import paired_bootstrap, ranking_metrics


def test_temporal_split_train_vocabulary_and_ties():
    # Day 8 is shared by users: global timestamps must not straddle the cutoff.
    rows = [("u", "a", 5, 1), ("v", "b", 1, 2), ("w", "a", 5, 2), ("u", "b", 4, 3),
            ("v", "a", 4, 4), ("u", "c", 5, 5), ("v", "c", 5, 6),
            ("u", "d", 4, 7), ("v", "d", 5, 8), ("new", "a", 5, 8),
            ("v", "e", 5, 9), ("w", "d", 5, 10), ("new", "new-item", 5, 11)]
    reviews = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    metadata = pd.DataFrame({"asin": ["a"], "title": ["A"], "category": [["Pop"]], "brand": [""]})
    protocol = build_protocol(reviews, metadata, train_fraction=0.65, validation_fraction=0.2)
    assert protocol.train.timestamp.max() < protocol.validation.timestamp.min()
    assert protocol.validation.timestamp.max() < protocol.test.timestamp.min()
    assert "new" not in protocol.user_ids
    assert "e" not in protocol.item_ids
    assert "new-item" not in protocol.item_ids
    assert protocol.seen_train.nnz == len(protocol.train)
    assert protocol.seen_final.nnz == len(protocol.train) + len(protocol.validation)
    assert protocol.positive_train.nnz == (protocol.train.rating >= 4).sum()
    assert protocol.ratings_train.shape == protocol.ratings_final.shape
    assert protocol.audit["splits"]["validation"]["dropped_cold_user_rows"] == 1


def test_metrics_match_hand_calculation_and_users_are_macro_averaged():
    truth = sparse.csr_matrix(([1, 1, 1], ([0, 0, 1], [0, 2, 1])), shape=(2, 4))
    recommendations = np.array([[0, 1], [2, 1]])
    metrics = ranking_metrics(recommendations, np.array([0, 1]), truth, k=2)
    discount = 1 / np.log2(3)
    np.testing.assert_allclose(metrics["NDCG@2"], [1 / (1 + discount), discount])
    np.testing.assert_allclose(metrics["Recall@2"], [0.5, 1.0])
    assert metrics["Recall@2"].mean() == 0.75


def test_duplicate_recommendations_rejected():
    truth = sparse.csr_matrix(([1], ([0], [0])), shape=(1, 3))
    with pytest.raises(ValueError, match="duplicate"):
        ranking_metrics(np.array([[0, 0]]), np.array([0]), truth, k=2)


def test_paired_bootstrap_identical_models_exact_zero():
    first = pd.DataFrame({"user": [1, 0], "NDCG@10": [0.2, 0.8]})
    result = paired_bootstrap(first, first.iloc[::-1], n_resamples=100)
    assert result["difference"] == result["ci_low"] == result["ci_high"] == 0
