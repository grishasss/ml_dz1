"""Reproducible Amazon Musical Instruments preparation without recommender libraries.

Review pairs are deduplicated using the earliest observation. Vocabulary fitting
and eligibility are based on the training period; future activity never changes
the training catalog. Product metadata is a historical snapshot, not an event log.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gzip
import hashlib
import html
import json
import re
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy import sparse


SOURCE_PAGE = "https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/index.html"
DATASET_NAME = "Amazon Reviews 2018 — Musical Instruments (5-core)"
DATA_URLS = {
    "Musical_Instruments_5.json.gz": [
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Musical_Instruments_5.json.gz",
        "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Musical_Instruments_5.json.gz",
    ],
    "meta_Musical_Instruments.json.gz": [
        "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Musical_Instruments.json.gz",
        "https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_Musical_Instruments.json.gz",
    ],
}
EXPECTED_SHA256 = {
    "Musical_Instruments_5.json.gz": "e3ae8a3f0d70bf63242a49539a8847d30216a3aa2938c049c7f43d0473fef1bc",
    "meta_Musical_Instruments.json.gz": "8e3f0dbe4c6c1aa5b32d350aa26072232beb6160ff3c5dadb44bca2c473a8b41",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_dataset(cache_dir: str | Path = "cache") -> dict[str, Any]:
    """Fetch the two official gzip files atomically and record their hashes.

    Files are checked against versioned source hashes and the local manifest.
    The experiment saves the manifest in its small public run report as well.
    No archives are unpacked and no strings are evaluated as Python code.
    """
    raw = Path(cache_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest: dict[str, Any] = {}
    for name, urls in DATA_URLS.items():
        target = raw / name
        used_url = old.get(name, {}).get("url", urls[0])
        if not target.exists():
            failures = []
            for url in urls:
                partial = target.with_name(target.name + ".part")
                try:
                    with requests.get(url, stream=True, timeout=(20, 120)) as response:
                        response.raise_for_status()
                        with partial.open("wb") as stream:
                            for chunk in response.iter_content(1024 * 1024):
                                if chunk:
                                    stream.write(chunk)
                    # A short HTML error page should not become a cached archive.
                    with gzip.open(partial, "rb") as stream:
                        stream.read(1)
                    partial.replace(target)
                    used_url = url
                    break
                except (requests.RequestException, OSError, EOFError) as exc:
                    partial.unlink(missing_ok=True)
                    failures.append(f"{url}: {exc}")
            else:
                raise RuntimeError("Dataset download failed:\n" + "\n".join(failures))
        digest = sha256_file(target)
        if digest != EXPECTED_SHA256[name]:
            raise ValueError(f"Source checksum mismatch for {target}; the file is corrupt or upstream data changed.")
        expected = old.get(name, {}).get("sha256")
        if expected is not None and digest != expected:
            raise ValueError(f"Checksum mismatch for {target}; remove the corrupt file and retry.")
        manifest[name] = {"url": used_url, "sha256": digest, "bytes": target.stat().st_size}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _categories(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for part in value:
        if isinstance(part, list):
            result.extend(_categories(part))
        elif part is not None:
            text = _clean_text(part)
            if text:
                result.append(text)
    return sorted(set(result))


def load_data(cache_dir: str | Path = "cache") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Download, validate and deduplicate reviews; stream matching metadata.

    Returns reviews with ``user_id,item_id,rating,timestamp``; item metadata
    with ``asin,title,category,brand``; and a JSON-serializable audit report.
    Missing metadata does not remove a review or modify model eligibility.
    """
    manifest = download_dataset(cache_dir)
    raw = Path(cache_dir) / "raw"
    records = []
    malformed = 0
    raw_rows = 0
    with gzip.open(raw / "Musical_Instruments_5.json.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            raw_rows += 1
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            records.append((value.get("reviewerID"), value.get("asin"),
                            value.get("overall"), value.get("unixReviewTime")))
    frame = pd.DataFrame(records, columns=["user_id", "item_id", "rating", "timestamp"])
    missing = {key: int(value) for key, value in frame.isna().sum().items()}
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    valid = (
        frame["user_id"].map(lambda x: isinstance(x, str) and bool(x.strip()))
        & frame["item_id"].map(lambda x: isinstance(x, str) and bool(x.strip()))
        & frame["rating"].isin([1, 2, 3, 4, 5])
        & frame["timestamp"].notna()
        & np.isfinite(frame["timestamp"])
        & frame["timestamp"].gt(0)
        & frame["timestamp"].mod(1).eq(0)
    )
    invalid_rows = int((~valid).sum())
    frame = frame.loc[valid].copy()
    frame["timestamp"] = frame["timestamp"].astype("int64")
    frame["rating"] = frame["rating"].astype("float64")
    # Stable sort preserves source line order when the same pair has same-day ties.
    frame = frame.sort_values("timestamp", kind="stable")
    duplicate_pairs = int(frame.duplicated(["user_id", "item_id"]).sum())
    frame = frame.drop_duplicates(["user_id", "item_id"], keep="first").reset_index(drop=True)
    wanted = set(frame["item_id"])
    metadata_records = {}
    metadata_rows = 0
    malformed_metadata = 0
    with gzip.open(raw / "meta_Musical_Instruments.json.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            metadata_rows += 1
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed_metadata += 1
                continue
            asin = value.get("asin")
            if asin in wanted and asin not in metadata_records:
                metadata_records[asin] = {
                    "asin": asin,
                    "title": _clean_text(value.get("title")),
                    "category": _categories(value.get("category", value.get("categories", []))),
                    "brand": _clean_text(value.get("brand")),
                }
    metadata = pd.DataFrame(list(metadata_records.values()), columns=["asin", "title", "category", "brand"])
    audit = {
        "dataset": DATASET_NAME, "source": SOURCE_PAGE, "files": manifest,
        "raw_rows": raw_rows, "malformed_rows": malformed,
        "missing_fields": missing, "invalid_rows": invalid_rows,
        "duplicate_user_item_rows": duplicate_pairs, "clean_rows": len(frame),
        "n_users": int(frame["user_id"].nunique()), "n_items": int(frame["item_id"].nunique()),
        "timestamp_min": int(frame["timestamp"].min()), "timestamp_max": int(frame["timestamp"].max()),
        "metadata_source_rows": metadata_rows, "malformed_metadata_rows": malformed_metadata,
        "metadata_matched_items": len(metadata), "metadata_missing_items": len(wanted - set(metadata["asin"])),
        "missing_title_items": int(metadata["title"].eq("").sum()),
        "missing_category_items": int(metadata["category"].map(len).eq(0).sum()),
        "missing_brand_items": int(metadata["brand"].eq("").sum()),
    }
    return frame, metadata, audit


def interaction_matrix(frame: pd.DataFrame, shape: tuple[int, int], mode: str = "ratings") -> sparse.csr_matrix:
    """Create ratings, all-observation history, or positive (rating >= 4) CSR."""
    if mode == "positive":
        frame = frame.loc[frame["rating"] >= 4]
    if mode not in {"ratings", "seen", "positive"}:
        raise ValueError(f"Unknown interaction matrix mode: {mode}")
    values = frame["rating"].to_numpy(dtype=np.float64) if mode == "ratings" else np.ones(len(frame))
    result = sparse.csr_matrix((values, (frame["user"], frame["item"])), shape=shape)
    result.sort_indices()
    return result


@dataclass
class Protocol:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    user_ids: np.ndarray
    item_ids: np.ndarray
    item_metadata: pd.DataFrame
    ratings_train: sparse.csr_matrix
    ratings_final: sparse.csr_matrix
    positive_train: sparse.csr_matrix
    positive_final: sparse.csr_matrix
    seen_train: sparse.csr_matrix
    seen_final: sparse.csr_matrix
    val_relevance: sparse.csr_matrix
    test_relevance: sparse.csr_matrix
    val_users: np.ndarray
    test_users: np.ndarray
    audit: dict

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


def build_protocol(reviews: pd.DataFrame, metadata: pd.DataFrame,
                   train_fraction: float = 0.8, validation_fraction: float = 0.1,
                   max_items: int | None = None) -> Protocol:
    """Build global temporal warm-start evaluation, with a fixed train catalog.

    ``max_items`` optionally selects a catalog by *training-only* review counts,
    with ascending ASIN breaking ties. The default keeps every training item.
    Cutoffs are computed before catalog selection. No future rating, activity,
    popularity or metadata occurrence is used to choose the retained catalog.
    """
    if not (0 < train_fraction < train_fraction + validation_fraction < 1):
        raise ValueError("Invalid train/validation fractions")
    if max_items is not None and max_items < 10:
        raise ValueError("max_items must be at least 10")
    columns = {"user_id", "item_id", "rating", "timestamp"}
    if not columns.issubset(reviews.columns) or len(reviews) == 0:
        raise ValueError("Nonempty cleaned review frame is required")
    if reviews.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Reviews must be deduplicated before splitting")
    cut_validation, cut_test = np.quantile(
        reviews["timestamp"], [train_fraction, train_fraction + validation_fraction], method="nearest"
    ).astype(np.int64)
    raw_train = reviews.loc[reviews["timestamp"] < cut_validation].copy()
    train_rows_before_catalog_filter = len(raw_train)
    raw_validation = reviews.loc[(reviews["timestamp"] >= cut_validation) & (reviews["timestamp"] < cut_test)].copy()
    raw_test = reviews.loc[reviews["timestamp"] >= cut_test].copy()
    if any(len(part) == 0 for part in [raw_train, raw_validation, raw_test]):
        raise ValueError("Time cutoffs produced an empty split; timestamps need more distinct dates")
    if max_items is not None:
        counts = raw_train.groupby("item_id").size().rename("count").reset_index()
        selected = counts.sort_values(["count", "item_id"], ascending=[False, True]).head(max_items)["item_id"]
        raw_train = raw_train.loc[raw_train["item_id"].isin(selected)].copy()
    user_ids = np.sort(raw_train["user_id"].unique())
    item_ids = np.sort(raw_train["item_id"].unique())
    user_map = {value: index for index, value in enumerate(user_ids)}
    item_map = {value: index for index, value in enumerate(item_ids)}
    split_audit = {}

    def encode(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        encoded = frame.copy()
        encoded["user"] = encoded["user_id"].map(user_map)
        encoded["item"] = encoded["item_id"].map(item_map)
        cold_user = encoded["user"].isna()
        cold_item = encoded["item"].isna()
        keep = ~(cold_user | cold_item)
        positives = encoded["rating"] >= 4
        split_audit[label] = {
            "raw_rows": len(frame), "retained_rows": int(keep.sum()),
            "dropped_cold_user_rows": int(cold_user.sum()),
            "dropped_cold_item_rows": int(cold_item.sum()),
            "dropped_any_cold_rows": int((~keep).sum()),
            "cold_users": int(frame.loc[cold_user, "user_id"].nunique()),
            "cold_items": int(frame.loc[cold_item, "item_id"].nunique()),
            "raw_positive_rows": int(positives.sum()),
            "retained_positive_rows": int((positives & keep).sum()),
            "positive_retention": float((positives & keep).sum() / max(1, positives.sum())),
        }
        encoded = encoded.loc[keep].copy()
        encoded[["user", "item"]] = encoded[["user", "item"]].astype("int64")
        return encoded.reset_index(drop=True)

    train = encode(raw_train, "train")
    validation = encode(raw_validation, "validation")
    test = encode(raw_test, "test")
    final = pd.concat([train, validation], ignore_index=True)
    shape = (len(user_ids), len(item_ids))
    ratings_train = interaction_matrix(train, shape)
    ratings_final = interaction_matrix(final, shape)
    seen_train = interaction_matrix(train, shape, "seen")
    seen_final = interaction_matrix(final, shape, "seen")
    positive_train = interaction_matrix(train, shape, "positive")
    positive_final = interaction_matrix(final, shape, "positive")
    val_relevance = interaction_matrix(validation, shape, "positive")
    test_relevance = interaction_matrix(test, shape, "positive")
    # Explicitly enforce candidate novelty, even if a caller changes preprocessing.
    val_relevance = val_relevance - val_relevance.multiply(seen_train)
    test_relevance = test_relevance - test_relevance.multiply(seen_final)
    val_relevance.eliminate_zeros()
    test_relevance.eliminate_zeros()
    val_users = np.flatnonzero(np.diff(val_relevance.indptr) > 0)
    test_users = np.flatnonzero(np.diff(test_relevance.indptr) > 0)
    for name, frame, eligible, rel in [
        ("validation", validation, val_users, val_relevance),
        ("test", test, test_users, test_relevance),
    ]:
        split_audit[name]["evaluation_users"] = len(eligible)
        split_audit[name]["users_with_no_eligible_positive"] = int(frame["user"].nunique() - len(eligible))
        split_audit[name]["eligible_positive_rows"] = int(rel.nnz)
    indexed_metadata = pd.DataFrame({"asin": item_ids}).merge(metadata.drop_duplicates("asin"), on="asin", how="left")
    for name in ["title", "brand"]:
        indexed_metadata[name] = indexed_metadata[name].fillna("")
    indexed_metadata["category"] = indexed_metadata["category"].map(lambda x: x if isinstance(x, list) else [])
    indexed_metadata["item"] = np.arange(len(item_ids))
    audit = {
        "validation_cutoff_timestamp": int(cut_validation), "test_cutoff_timestamp": int(cut_test),
        "train_fraction_requested": train_fraction, "validation_fraction_requested": validation_fraction,
        "catalog_max_items": max_items, "n_train_users": len(user_ids), "n_train_items": len(item_ids),
        "train_rows_before_catalog_filter": train_rows_before_catalog_filter,
        "train_rows_removed_by_catalog_filter": train_rows_before_catalog_filter - len(train),
        "retained_rows_all_splits": len(train) + len(validation) + len(test),
        "splits": split_audit,
        "train_sparsity": float(1 - ratings_train.nnz / (shape[0] * shape[1])),
        "train_positive_sparsity": float(1 - positive_train.nnz / (shape[0] * shape[1])),
        "candidate_policy": "all items observed in train; exclude every previously rated item",
        "relevance": "held-out rating >= 4; candidate must belong to fixed train catalog",
    }
    return Protocol(train, validation, test, user_ids, item_ids, indexed_metadata,
                    ratings_train, ratings_final, positive_train, positive_final,
                    seen_train, seen_final, val_relevance.tocsr(), test_relevance.tocsr(),
                    val_users, test_users, audit)
