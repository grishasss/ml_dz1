"""A second-order factorization machine trained with pairwise BPR loss.

Only user/item identifiers and static product title/category are features. For
each example x, f(x) = w^T x + 1/2 sum_f[(sum_j v_jf x_j)^2
 - sum_j v_jf^2 x_j^2]. The user linear term cancels in pairwise training and
ranking, so it is fixed to zero. No ready-made recommender is used.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseRecommender


def _stable_bucket(value: str, bins: int) -> int:
    """Stable across machines and PYTHONHASHSEED values."""
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "little") % bins


def _category_strings(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [part for entry in value for part in _category_strings(entry)]
    if isinstance(value, str) and value.strip():
        return [value.strip().lower()]
    return []


def metadata_features(metadata, n_items: int, hash_bins: int = 512,
                      use_metadata: bool = True) -> tuple[np.ndarray, np.ndarray, int]:
    """Build item-ID, category-bag and title-token-bag features.

    Each metadata field has squared feature norm one; duplicate hashed tokens
    within a field are removed. The two fields use disjoint hash spaces. Empty
    fields receive an explicit missing token. No vocabulary is fit on holdout.
    """
    if hash_bins < 1:
        raise ValueError("hash_bins must be positive")
    if use_metadata and len(metadata) != n_items:
        raise ValueError("metadata rows must align exactly with item columns")
    rows: list[list[int]] = []
    weights: list[list[float]] = []
    for item in range(n_items):
        ids, vals = [item], [1.0]
        if use_metadata:
            row = metadata.iloc[item] if hasattr(metadata, "iloc") else metadata[item]
            categories = sorted(set(_category_strings(row.get("category", []))))[:16]
            title = row.get("title", "")
            title = title if isinstance(title, str) else ""
            tokens = sorted(set(re.findall(r"[a-z0-9]+", title.lower())))[:16]
            for offset, values, missing in [
                (n_items, categories, "<missing-category>"),
                (n_items + hash_bins, tokens, "<missing-title>"),
            ]:
                buckets = sorted({_stable_bucket(s, hash_bins) for s in (values or [missing])})
                ids.extend(offset + b for b in buckets)
                vals.extend([1.0 / np.sqrt(len(buckets))] * len(buckets))
        rows.append(ids)
        weights.append(vals)
    width = max(map(len, rows), default=1)
    indices = np.zeros((n_items, width), dtype=np.int64)
    values = np.zeros((n_items, width), dtype=np.float32)
    for i, (ids, vals) in enumerate(zip(rows, weights)):
        indices[i, :len(ids)] = ids
        values[i, :len(vals)] = vals
    n_features = n_items + (2 * hash_bins if use_metadata else 0)
    return indices, values, n_features


def sample_unobserved(users: np.ndarray, n_items: int, seen_codes: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
    """Uniform rejection sampling from each user's unobserved catalog.

    ``seen_codes`` is sorted and contains user*n_items+item for ALL ratings,
    including dislikes. Callers must exclude users with an entirely seen catalog.
    """
    users = np.asarray(users, dtype=np.int64)
    negatives = rng.integers(n_items, size=len(users), dtype=np.int64)
    pending = np.arange(len(users))
    while len(pending):
        codes = users[pending] * n_items + negatives[pending]
        positions = np.searchsorted(seen_codes, codes)
        present = positions < len(seen_codes)
        if len(seen_codes):
            present &= seen_codes[np.minimum(positions, len(seen_codes) - 1)] == codes
        pending = pending[present]
        negatives[pending] = rng.integers(n_items, size=len(pending), dtype=np.int64)
    return negatives


class _FMParameters(nn.Module):
    def __init__(self, n_users: int, n_features: int, n_factors: int,
                 seed: int = 42, n_items: int | None = None):
        super().__init__()
        self.users = nn.Embedding(n_users, n_factors)
        self.features = nn.Embedding(n_features, n_factors)
        self.linear = nn.Embedding(n_features, 1)
        # Embedding constructors consume global RNG according to table size.
        # A separate generator and same-shaped ID initialization make the
        # metadata ablation start with EXACTLY the same user/item-ID factors.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        n_items = n_features if n_items is None else n_items
        nn.init.normal_(self.users.weight, std=0.05, generator=generator)
        nn.init.normal_(self.features.weight[:n_items], std=0.05, generator=generator)
        if n_items < n_features:
            nn.init.normal_(self.features.weight[n_items:], std=0.05, generator=generator)
        nn.init.zeros_(self.linear.weight)

    def item_terms(self, indices: torch.Tensor, values: torch.Tensor):
        weighted = self.features(indices) * values.unsqueeze(-1)
        summed = weighted.sum(dim=1)
        within = 0.5 * (summed.square() - weighted.square().sum(dim=1)).sum(dim=1)
        linear = (self.linear(indices).squeeze(-1) * values).sum(dim=1)
        return summed, linear + within


class FM(BaseRecommender):
    """Factorization machine using BPR, implemented directly in PyTorch.

    Args:
        use_metadata: False removes category/title features for a controlled
            user/item-only ablation, with the same training/negative protocol.

    ``fit`` accepts a CSR of observed 1..5 ratings (not a binary matrix) and a
    metadata DataFrame in exactly the item-column order. All computations use
    deterministic CPU operations. Optimizer steps are fixed by ``epochs``;
    validation selects hyperparameters outside this class.
    """

    def __init__(self, n_factors=32, lr=0.01, reg=1e-5, epochs=10,
                 batch_size=2048, seed=42, use_metadata=True, hash_bins=512):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.use_metadata = use_metadata
        self.hash_bins = hash_bins

    def fit(self, ratings: sparse.csr_matrix, metadata):
        ratings = ratings.tocsr(copy=True)
        ratings.sort_indices()
        self.n_users, self.n_items = ratings.shape
        if self.n_users == 0 or self.n_items == 0:
            raise ValueError("ratings must have nonempty dimensions")
        if self.n_factors < 1 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("n_factors, epochs and batch_size must be positive")
        if not np.all((ratings.data >= 1) & (ratings.data <= 5)):
            raise ValueError("FM.fit expects observed ratings in [1, 5]")
        indices, values, n_features = metadata_features(
            metadata, self.n_items, self.hash_bins, self.use_metadata)
        self.feature_indices_ = indices
        self.feature_values_ = values
        item_indices = torch.from_numpy(indices)
        item_values = torch.from_numpy(values)
        all_users = np.repeat(np.arange(self.n_users), np.diff(ratings.indptr))
        eligible = (ratings.data >= 4) & (np.diff(ratings.indptr)[all_users] < self.n_items)
        users = all_users[eligible]
        positives = ratings.indices[eligible].astype(np.int64, copy=False)
        if len(users) == 0:
            raise ValueError("No positive pairs with an available negative")
        seen_codes = np.sort(all_users.astype(np.int64) * self.n_items + ratings.indices)
        rng = np.random.default_rng(self.seed)
        # Isolate torch's RNG so fitting does not alter unrelated model seeds.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            parameters = _FMParameters(self.n_users, n_features, self.n_factors,
                                       seed=self.seed, n_items=self.n_items)
        optimizer = torch.optim.Adam(parameters.parameters(), lr=self.lr,
                                     weight_decay=self.reg)
        self.loss_history_ = []
        for _ in range(self.epochs):
            permutation = rng.permutation(len(users))
            negatives = sample_unobserved(users, self.n_items, seen_codes, rng)
            epoch_loss = 0.0
            for start in range(0, len(users), self.batch_size):
                batch = permutation[start:start + self.batch_size]
                u = torch.from_numpy(users[batch].astype(np.int64, copy=False))
                p = torch.from_numpy(positives[batch])
                n = torch.from_numpy(negatives[batch])
                p_vector, p_bias = parameters.item_terms(item_indices[p], item_values[p])
                n_vector, n_bias = parameters.item_terms(item_indices[n], item_values[n])
                difference = (parameters.users(u) * (p_vector - n_vector)).sum(dim=1)
                difference = difference + p_bias - n_bias
                loss = F.softplus(-difference).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item() * len(batch)
            self.loss_history_.append(epoch_loss / len(users))
        # Collapse static item fields once, so inference is a matrix multiply.
        with torch.no_grad():
            vectors, biases = [], []
            for start in range(0, self.n_items, self.batch_size):
                vector, bias = parameters.item_terms(
                    item_indices[start:start + self.batch_size],
                    item_values[start:start + self.batch_size])
                vectors.append(vector.numpy())
                biases.append(bias.numpy())
            self.user_factors_ = parameters.users.weight.detach().numpy().copy()
            self.item_factors_ = np.concatenate(vectors)
            self.item_bias_ = np.concatenate(biases)
        self.n_features_ = n_features
        return self

    def score(self, users):
        users = np.asarray(users, dtype=np.int64)
        return self.user_factors_[users] @ self.item_factors_.T + self.item_bias_[None, :]
