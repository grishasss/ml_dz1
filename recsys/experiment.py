"""Validation selection followed by a separate, frozen final test phase."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from models.item_knn import ItemKNN
from models.ials import IALS
from models.ease import EASE
from models.fm import FM
from models.custom_als import CustomALS
from recsys.evaluation import evaluate, paired_bootstrap


MODEL_CLASSES = {"ItemKNN": ItemKNN, "IALS": IALS, "EASE": EASE,
                 "FM": FM, "CustomALS": CustomALS}


def make_model(name, params, seed=42, use_metadata=True):
    params = dict(params)
    if name in {"IALS", "FM", "CustomALS"}:
        params["seed"] = seed
    if name == "FM":
        params["use_metadata"] = use_metadata
    return MODEL_CLASSES[name](**params)


def fit_model(model, name, protocol, final=False):
    ratings = protocol.ratings_final if final else protocol.ratings_train
    positive = protocol.positive_final if final else protocol.positive_train
    if name == "FM":
        return model.fit(ratings, protocol.item_metadata)
    return model.fit(positive if name in {"ItemKNN", "EASE"} else ratings)


def select_hyperparameters(protocol, config):
    """Test matrices/users are never accessed here. Ties keep first grid entry."""
    rows, selected = [], {}
    metric = config["primary_metric"]
    for name, grid in config["search"].items():
        best_value = -np.inf
        for trial, params in enumerate(grid):
            start = perf_counter()
            model = make_model(name, params, config["seed"])
            fit_model(model, name, protocol)
            fit_seconds = perf_counter() - start
            result = evaluate(model, protocol.val_users, protocol.val_relevance,
                              protocol.seen_train, k=config["k"])
            rows.append({"model": name, "trial": trial,
                         "params": json.dumps(params, sort_keys=True),
                         "fit_seconds": fit_seconds, **result.metrics})
            if result.metrics[metric] > best_value:
                best_value = result.metrics[metric]
                selected[name] = dict(params)
            print(f"validation {name} {trial + 1}/{len(grid)}: "
                  f"{metric}={result.metrics[metric]:.6f}; fit={fit_seconds:.1f}s", flush=True)
            Path('cache').mkdir(exist_ok=True)
            Path('cache/progress.json').write_text(json.dumps({
                'phase': 'validation', 'model': name, 'trial': trial + 1,
                'completed_trials': len(rows), 'metrics': result.metrics}))
            del model, result
            gc.collect()
    return pd.DataFrame(rows), selected


def final_comparison(protocol, selected, config):
    """Refit frozen settings on train+validation, then evaluate test exactly once.

    Two prespecified controls isolate metadata and target effects. Test results
    never feed model selection, epoch choice or feature design.
    """
    rows, evaluations, examples, learning = [], {}, [], {}
    specifications = [(name, name, params, True) for name, params in selected.items()]
    specifications.append(("FM IDs only", "FM", selected["FM"], False))
    matched_same = selected["CustomALS"] == selected["IALS"]
    if not matched_same:
        specifications.append(("CustomALS matched", "CustomALS", selected["IALS"], True))
    example_user = int(protocol.test_users[0])
    for label, name, params, use_metadata in specifications:
        start = perf_counter()
        model = make_model(name, params, config["seed"], use_metadata)
        fit_model(model, name, protocol, final=True)
        fit_seconds = perf_counter() - start
        start = perf_counter()
        result = evaluate(model, protocol.test_users, protocol.test_relevance,
                          protocol.seen_final, k=config["k"])
        eval_seconds = perf_counter() - start
        evaluations[label] = result.per_user
        rows.append({"model": label, "params": json.dumps(params, sort_keys=True),
                     "fit_seconds": fit_seconds, "eval_seconds": eval_seconds,
                     **result.metrics})
        if hasattr(model, "loss_history_"):
            learning[label] = model.loss_history_
        ranking = result.recommendations[0]
        truth = set(protocol.test_relevance[example_user].indices)
        for rank, item in enumerate(ranking, start=1):
            if item < 0:
                continue
            meta = protocol.item_metadata.iloc[item]
            examples.append({"model": label, "user_index": example_user,
                             "rank": rank, "title": meta["title"],
                             "relevant": item in truth})
        print(f"test {label}: {config['primary_metric']}="
              f"{result.metrics[config['primary_metric']]:.6f}; fit={fit_seconds:.1f}s", flush=True)
        Path('cache/progress.json').write_text(json.dumps({
            'phase': 'test', 'model': label, 'completed_models': len(rows),
            'metrics': result.metrics}))
        del model, result
        gc.collect()
    controls = []
    custom_label = "CustomALS" if matched_same else "CustomALS matched"
    for first, second, purpose in [(custom_label, "IALS", "graded target - binary target"),
                                    ("FM", "FM IDs only", "metadata - IDs only")]:
        for metric in ["NDCG@10", "Recall@10"]:
            estimate = paired_bootstrap(evaluations[first], evaluations[second], metric,
                                        seed=config["seed"],
                                        n_resamples=config["bootstrap_samples"])
            controls.append({"comparison": purpose, "first": first, "second": second,
                             "metric": metric, **estimate})
    return pd.DataFrame(rows), pd.DataFrame(controls), pd.DataFrame(examples), learning


def save_results(validation, selected, final, controls, data_audit, protocol, config, learning,
                 output_dir="results"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    validation.to_csv(out / "validation.csv", index=False)
    final.to_csv(out / "test.csv", index=False)
    controls.to_csv(out / "controls.csv", index=False)
    versions = {name: importlib.metadata.version(name) for name in
                ["numpy", "scipy", "pandas", "torch", "nbformat", "nbclient", "threadpoolctl"]}
    report = {"config": config, "selected": selected,
              "dataset": data_audit, "protocol": protocol.audit,
              "learning_curves": learning,
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "packages": versions}}
    (out / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
