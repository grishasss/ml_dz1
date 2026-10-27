"""Weighted implicit alternating least squares, without recsys libraries."""

import numpy as np
from scipy import sparse
from scipy.linalg import solve

from .base import BaseRecommender


class IALS(BaseRecommender):
    """Minimise sum_ui c_ui(p_ui-x_u^T y_i)^2 + reg*(||X||²+||Y||²).

    p_ui=1[r_ui>=4], missing p_ui=0. Every observed review (including a low
    rating) receives c_ui=1+alpha; unobserved pairs have c_ui=1. Passing the
    SAME rating matrix to CustomALS isolates the effect of the target alone.
    The confidence support is kept separate from the possibly zero target.
    """

    def __init__(self, n_factors=32, reg=1.0, alpha=20.0, n_iters=12, seed=42):
        if n_factors < 1 or reg <= 0 or alpha < 0 or n_iters < 1:
            raise ValueError("Invalid ALS hyperparameters")
        self.n_factors = int(n_factors)
        self.reg = float(reg)
        self.alpha = float(alpha)
        self.n_iters = int(n_iters)
        self.seed = int(seed)

    @staticmethod
    def targets(ratings):
        return (np.asarray(ratings) >= 4).astype(np.float64)

    def _solve_rows(self, observations, fixed):
        """Solve the exact normal equations for one alternating half-step.

        A = Y'Y + alpha*Y_obs'Y_obs + reg*I,
        b = (1+alpha)*Y_obs'p_obs.
        """
        gram = fixed.T @ fixed + self.reg * np.eye(self.n_factors)
        updated = np.zeros((observations.shape[0], self.n_factors))
        for row in range(observations.shape[0]):
            start, stop = observations.indptr[row : row + 2]
            if start == stop:
                continue
            selected = fixed[observations.indices[start:stop]]
            target = self.targets(observations.data[start:stop])
            normal = gram + self.alpha * (selected.T @ selected)
            rhs = (1.0 + self.alpha) * (selected.T @ target)
            updated[row] = solve(
                normal, rhs, assume_a="pos", check_finite=False
            )
        return updated

    def fit(self, ratings):
        """Fit to a CSR matrix containing observed ratings 1..5, not labels."""
        ratings = sparse.csr_matrix(ratings, dtype=np.float64).copy()
        ratings.sum_duplicates()
        ratings.eliminate_zeros()
        ratings.sort_indices()
        if np.any((ratings.data < 1) | (ratings.data > 5)):
            raise ValueError("ALS expects observed ratings in [1,5]")
        self.n_users, self.n_items = ratings.shape
        rng = np.random.default_rng(self.seed)
        self.item_factors = rng.normal(
            scale=0.01, size=(self.n_items, self.n_factors)
        )
        transposed = ratings.T.tocsr()
        for _ in range(self.n_iters):
            self.user_factors = self._solve_rows(ratings, self.item_factors)
            self.item_factors = self._solve_rows(transposed, self.user_factors)
        return self

    def score(self, users):
        return self.user_factors[np.asarray(users)] @ self.item_factors.T
