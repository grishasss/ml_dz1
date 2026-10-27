"""Item-based cosine nearest neighbours implemented with sparse arithmetic."""

import numpy as np
from scipy import sparse

from .base import BaseRecommender


class ItemKNN(BaseRecommender):
    """s_ij = cosine(i,j) * n_ij/(n_ij+shrink), with top neighbours per i.

    Input is a binary user-item matrix of positive feedback. Scores are the
    sum of similarities from the user's positive history, X @ S.
    """

    def __init__(self, n_neighbors=50, shrink=10.0):
        if n_neighbors < 1 or shrink < 0:
            raise ValueError("Invalid neighbourhood or shrinkage")
        self.n_neighbors = int(n_neighbors)
        self.shrink = float(shrink)

    def fit(self, interactions):
        self.interactions = sparse.csr_matrix(interactions, dtype=np.float64).copy()
        self.interactions.eliminate_zeros()
        self.interactions.data[:] = 1.0
        self.n_users, self.n_items = self.interactions.shape
        counts = np.asarray(self.interactions.sum(axis=0)).ravel()
        cooccurrence = (self.interactions.T @ self.interactions).tocsr()
        cooccurrence.setdiag(0)
        cooccurrence.eliminate_zeros()
        values, columns, indptr = [], [], [0]
        for item in range(self.n_items):
            start, stop = cooccurrence.indptr[item : item + 2]
            neighbours = cooccurrence.indices[start:stop]
            common = cooccurrence.data[start:stop]
            similarity = common / np.sqrt(counts[item] * counts[neighbours])
            if self.shrink:
                similarity *= common / (common + self.shrink)
            order = np.lexsort((neighbours, -similarity))[: self.n_neighbors]
            columns.extend(neighbours[order])
            values.extend(similarity[order])
            indptr.append(len(values))
        self.similarity = sparse.csr_matrix(
            (values, columns, indptr), shape=(self.n_items, self.n_items)
        )
        return self

    def score(self, users):
        return (self.interactions[np.asarray(users)] @ self.similarity).toarray()
