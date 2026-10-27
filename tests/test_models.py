"""Analytical checks of implementations; run python -m unittest discover -s tests."""

import unittest

import numpy as np
from scipy import sparse

from models.base import BaseRecommender
from models.custom_als import CustomALS
from models.ease import EASE
from models.ials import IALS
from models.item_knn import ItemKNN


class FixedScores(BaseRecommender):
    def score(self, users):
        return np.tile([0.0, 3.0, 3.0, 9.0, 3.0], (len(users), 1))


class ModelMathTests(unittest.TestCase):
    def test_als_half_step_equals_dense_weighted_least_squares(self):
        ratings = sparse.csr_matrix([[5.0, 0.0, 2.0], [0.0, 4.0, 0.0]])
        fixed = np.array([[0.2, 0.5], [0.4, -0.3], [0.1, 0.6]])
        for model in [IALS(n_factors=2, alpha=3, reg=0.7),
                      CustomALS(n_factors=2, alpha=3, reg=0.7)]:
            actual = model._solve_rows(ratings, fixed)
            for user, row in enumerate(ratings.toarray()):
                confidence = np.diag(1 + 3 * (row != 0))
                target = model.targets(row)
                expected = np.linalg.solve(
                    fixed.T @ confidence @ fixed + 0.7 * np.eye(2),
                    fixed.T @ confidence @ target,
                )
                np.testing.assert_allclose(actual[user], expected, atol=1e-12)

    def test_custom_changes_target_with_same_confidence(self):
        np.testing.assert_allclose(IALS.targets([1, 2, 3, 4, 5]), [0, 0, 0, 1, 1])
        np.testing.assert_allclose(
            CustomALS.targets([1, 2, 3, 4, 5]), [0, 0, 1/3, 2/3, 1]
        )

    def test_ease_satisfies_constrained_normal_equations(self):
        x = np.array([[1., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
        model = EASE(reg=2).fit(sparse.csr_matrix(x))
        np.testing.assert_array_equal(np.diag(model.coefficients), 0)
        gram = x.T @ x
        for item in range(x.shape[1]):
            others = np.arange(x.shape[1]) != item
            expected = np.linalg.solve(
                gram[np.ix_(others, others)] + 2 * np.eye(others.sum()),
                gram[others, item],
            )
            np.testing.assert_allclose(
                model.coefficients[others, item], expected, rtol=1e-6
            )

    def test_knn_cosine_and_shrink(self):
        x = sparse.csr_matrix([[1., 1., 0.], [1., 0., 1.]])
        model = ItemKNN(n_neighbors=2, shrink=1).fit(x)
        expected = 0.5 / np.sqrt(2)
        self.assertAlmostEqual(model.similarity[0, 1], expected)
        self.assertEqual(model.similarity[1, 2], 0)
        np.testing.assert_array_equal(model.similarity.diagonal(), 0)

    def test_common_ranking_exclusions_and_boundary_ties(self):
        seen = sparse.csr_matrix([[0, 1, 0, 0, 0]])
        recs = FixedScores().recommend(
            [0], seen, np.array([True, True, True, False, True]), k=3
        )
        np.testing.assert_array_equal(recs, [[2, 4, 0]])
        recs = FixedScores().recommend([0], seen, np.ones(5, dtype=bool), k=2)
        np.testing.assert_array_equal(recs, [[3, 2]])

    def test_deterministic_als_and_cold_users(self):
        ratings = sparse.csr_matrix([[5., 0, 2.], [0, 4., 5.], [0., 0., 0.]])
        one = IALS(n_factors=2, n_iters=2, seed=13).fit(ratings)
        two = IALS(n_factors=2, n_iters=2, seed=13).fit(ratings)
        np.testing.assert_array_equal(one.score([0, 1]), two.score([0, 1]))
        np.testing.assert_array_equal(one.score([2]), [[0, 0, 0]])


if __name__ == "__main__":
    unittest.main()
