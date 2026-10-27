"""Shared full-catalogue recommendation, with identical exclusion and ties."""

from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse


class BaseRecommender(ABC):
    """Models expose scores; this class owns the common top-K protocol."""

    @abstractmethod
    def score(self, users):
        """Return a dense (len(users), n_items) score array."""

    def recommend(self, users, seen, candidates, k=10):
        """Rank candidates and exclude every previously reviewed item.

        Ties are broken by ascending encoded item ID. If fewer than K eligible
        items exist, remaining positions contain -1 (never a relevant item).
        Candidate eligibility depends only on the supplied snapshot.
        """
        if k < 1:
            raise ValueError("k must be positive")
        users = np.asarray(users, dtype=np.int64)
        seen = sparse.csr_matrix(seen)
        candidates = np.asarray(candidates, dtype=bool)
        scores = np.asarray(self.score(users), dtype=np.float64)
        if scores.shape != (len(users), seen.shape[1]):
            raise ValueError("Unexpected score matrix shape")
        if candidates.shape != (seen.shape[1],):
            raise ValueError("candidates must be a catalogue-length boolean mask")
        if not np.isfinite(scores).all():
            raise ValueError("Scores contain NaN or infinity")
        scores[:, ~candidates] = -np.inf
        result = np.full((len(users), k), -1, dtype=np.int64)
        for row, user in enumerate(users):
            start, stop = seen.indptr[user : user + 2]
            scores[row, seen.indices[start:stop]] = -np.inf
            eligible = np.flatnonzero(np.isfinite(scores[row]))
            if not len(eligible):
                continue
            # Partition reduces sorting to K items. Keep ALL boundary ties so
            # argpartition's arbitrary tie handling cannot affect the ranking.
            take = min(k, len(eligible))
            values = scores[row, eligible]
            threshold = np.partition(values, len(values) - take)[-take]
            selected = eligible[values >= threshold]
            order = np.lexsort((selected, -scores[row, selected]))[:take]
            result[row, :take] = selected[order]
        return result
