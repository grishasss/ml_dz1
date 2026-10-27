"""Embarrassingly Shallow Autoencoder (EASE), solved in closed form."""

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve

from .base import BaseRecommender


class EASE(BaseRecommender):
    """Minimise ||X-XB||²_F + reg*||B||²_F subject to diag(B)=0.

    P=(X'X+reg*I)^(-1), B_ij=-P_ij/P_jj, B_jj=0. Dense item-item
    parameters require O(n_items²) memory and a cubic Cholesky factorisation.
    """

    def __init__(self, reg=100.0):
        if reg <= 0:
            raise ValueError("reg must be positive")
        self.reg = float(reg)

    def fit(self, interactions):
        self.interactions = sparse.csr_matrix(interactions, dtype=np.float64).copy()
        self.interactions.eliminate_zeros()
        self.interactions.data[:] = 1.0
        self.n_users, self.n_items = self.interactions.shape
        gram = (self.interactions.T @ self.interactions).toarray()
        gram.flat[:: self.n_items + 1] += self.reg
        factor = cho_factor(gram, lower=True, overwrite_a=True, check_finite=False)
        precision = cho_solve(
            factor, np.eye(self.n_items), overwrite_b=True, check_finite=False
        )
        precision /= -np.diag(precision).copy()[None, :]
        np.fill_diagonal(precision, 0.0)
        self.coefficients = precision.astype(np.float32)
        return self

    def score(self, users):
        return self.interactions[np.asarray(users)] @ self.coefficients
