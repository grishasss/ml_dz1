"""Analytical FM equivalence and strict negative-sampling checks."""

import unittest

import numpy as np
import pandas as pd
from scipy import sparse
import torch

from models.fm import FM, _FMParameters, metadata_features, sample_unobserved


class TestFM(unittest.TestCase):
    def test_metadata_ablation_preserves_common_id_initialization(self):
        ids_only = _FMParameters(3, 4, 5, seed=17, n_items=4)
        with_metadata = _FMParameters(3, 132, 5, seed=17, n_items=4)
        torch.testing.assert_close(ids_only.users.weight, with_metadata.users.weight,
                                   rtol=0, atol=0)
        torch.testing.assert_close(ids_only.features.weight, with_metadata.features.weight[:4],
                                   rtol=0, atol=0)

    def test_explicit_second_order_formula(self):
        model = _FMParameters(2, 5, 3)
        with torch.no_grad():
            model.users.weight.copy_(torch.arange(6).reshape(2, 3) / 10)
            model.features.weight.copy_(torch.arange(15).reshape(5, 3) / 7)
            model.linear.weight.copy_(torch.arange(5).reshape(5, 1) / 11)
        indices = torch.tensor([[1, 3, 4]])
        values = torch.tensor([[1.0, 0.5, 0.25]])
        vector, bias = model.item_terms(indices, values)
        collapsed = (model.users.weight[1] * vector[0]).sum() + bias[0]
        embeddings = torch.vstack([model.users.weight[1], model.features.weight[indices[0]]])
        x = torch.tensor([1.0, 1.0, 0.5, 0.25])
        explicit = (model.linear(indices).squeeze(-1) * values).sum()
        for a in range(4):
            for b in range(a + 1, 4):
                explicit = explicit + x[a] * x[b] * (embeddings[a] * embeddings[b]).sum()
        torch.testing.assert_close(collapsed, explicit)

    def test_negatives_exclude_all_observed_including_dislikes(self):
        users = np.repeat([0, 1], 200)
        seen = np.array([0, 1, 3, 4, 5, 6])  # each user has exactly one unseen item
        samples = sample_unobserved(users, 4, seen, np.random.default_rng(42))
        np.testing.assert_array_equal(samples[:200], 2)
        np.testing.assert_array_equal(samples[200:], 3)

    def test_metadata_is_stable_and_fields_are_normalized(self):
        metadata = pd.DataFrame({"title": ["A blue song", ""], "category": [["Pop", "Pop"], []]})
        indices, values, n = metadata_features(metadata, 2, hash_bins=64)
        indices2, values2, _ = metadata_features(metadata, 2, hash_bins=64)
        np.testing.assert_array_equal(indices, indices2)
        np.testing.assert_array_equal(values, values2)
        self.assertEqual(n, 130)
        np.testing.assert_allclose((values ** 2).sum(axis=1), [3, 3], rtol=1e-6)

    def test_deterministic_training_and_dense_scoring(self):
        torch.set_num_threads(1)
        ratings = sparse.csr_matrix(np.array([[5, 1, 0, 0], [0, 4, 5, 0], [1, 0, 0, 5]], dtype=np.float32))
        metadata = pd.DataFrame({"title": ["One", "Two", "Three", "Four"], "category": [["Pop"]] * 4})
        a = FM(n_factors=4, epochs=2, seed=7).fit(ratings, metadata)
        b = FM(n_factors=4, epochs=2, seed=7).fit(ratings, metadata)
        np.testing.assert_array_equal(a.score([0, 2]), b.score([0, 2]))
        self.assertEqual(a.score([0, 2]).shape, (2, 4))
        self.assertTrue(np.isfinite(a.score([0, 2])).all())


if __name__ == "__main__":
    unittest.main()
