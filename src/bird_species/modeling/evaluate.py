"""
Scoring classifiers on the combined test set, and taking their errors apart.

Every classifier is reduced to a score matrix (photographs by classes), so the
few-shot prototypes, the trained heads and the fine-tuned network all go
through the same metric code.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from bird_species.config import (
    CLOSENESS_BINS,
    CONFUSED_PAIRS,
    EMBED_BATCH,
    LOADER_WORKERS,
    SIZE_BINS,
    TIMING_SAMPLE,
)
from bird_species.dataset import load_image
from bird_species.duplicates import unit_rows
from bird_species.embeddings import ImageFiles, device
from bird_species.fewshot import classification_metrics
from bird_species.statistics import accuracy_interval


@torch.no_grad()
def predict_probabilities(model: torch.nn.Module, filenames: Sequence[str], transform,
                          batch_size: int = EMBED_BATCH, workers: int = LOADER_WORKERS) -> np.ndarray:
    """Softmax probabilities for every photograph, in the order given."""
    target = device()
    model = model.to(target).eval()
    loader = DataLoader(ImageFiles(filenames, transform), batch_size=batch_size, num_workers=workers,
                        persistent_workers=workers > 0)
    chunks = [torch.softmax(model(batch.to(target)), dim=1).float().cpu().numpy() for batch in loader]
    return np.concatenate(chunks)


def cached_probabilities(path: Path, compute: Callable[[], np.ndarray]) -> np.ndarray:
    """Run `compute` once and keep its result on disk; later calls read the file."""
    if path.exists():
        return np.load(path)
    probabilities = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, probabilities)
    return probabilities


def summarize(labels: np.ndarray, scores: np.ndarray, species_of_class: np.ndarray) -> dict[str, float]:
    """
    Accuracy with its interval, top-5 accuracy, macro F1 and species-level accuracy.

    Species-level accuracy counts a prediction as right when it names the
    correct species but the wrong plumage variant, which separates "confused a
    male with a female" from "confused two species".
    """
    predicted = scores.argmax(axis=1)
    correct = predicted == labels
    low, high = accuracy_interval(correct)
    metrics = classification_metrics(labels, scores)
    return {**metrics, "accuracy_low": low, "accuracy_high": high,
            "species_accuracy": float((species_of_class[predicted] == species_of_class[labels]).mean())}


def error_kinds(labels: np.ndarray, predicted: np.ndarray, species_of_class: np.ndarray) -> pd.Series:
    """How many predictions are right, wrong only in the variant, or wrong in the species."""
    same_species = species_of_class[predicted] == species_of_class[labels]
    kinds = np.where(predicted == labels, "correct",
                     np.where(same_species, "wrong variant, right species", "wrong species"))
    return pd.Series(kinds).value_counts()


def per_class_accuracy(labels: np.ndarray, predicted: np.ndarray, n_classes: int) -> np.ndarray:
    """Recall of each class, read off the diagonal of the confusion matrix."""
    matrix = confusion_matrix(labels, predicted, labels=np.arange(n_classes))
    return np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)


def confused_pairs(labels: np.ndarray, predicted: np.ndarray, class_names: Sequence[str],
                   species_of_class: np.ndarray, top: int = CONFUSED_PAIRS) -> pd.DataFrame:
    """The most frequent (true class, predicted class) mistakes."""
    wrong = predicted != labels
    pairs = pd.DataFrame({"true": labels[wrong], "predicted": predicted[wrong]})
    # Ties are broken by class index, so the table is the same under any pandas version.
    counts = (pairs.value_counts().reset_index(name="errors")
              .sort_values(["errors", "true", "predicted"], ascending=[False, True, True])
              .head(top).reset_index(drop=True))
    names = np.asarray(class_names)
    counts["true_class"] = names[counts["true"]]
    counts["predicted_class"] = names[counts["predicted"]]
    counts["same_species"] = species_of_class[counts["true"]] == species_of_class[counts["predicted"]]
    return counts


def accuracy_by_size(box_fraction: np.ndarray, correct: dict[str, np.ndarray],
                     bins: int = SIZE_BINS) -> pd.DataFrame:
    """Each classifier's accuracy within quartiles of how much of the frame the bird fills."""
    quartile = pd.qcut(box_fraction, bins, labels=[f"Q{k + 1}" for k in range(bins)])
    table = pd.DataFrame({name: pd.Series(hits).groupby(quartile, observed=True).mean()
                          for name, hits in correct.items()})
    bounds = pd.Series(box_fraction).groupby(quartile, observed=True).agg(["min", "max"])
    table.insert(0, "bird_share_of_frame", [f"{low:.0%} to {high:.0%}" for low, high in bounds.to_numpy()])
    return table.reset_index(names="size_quartile")


def best_and_worst(probabilities: np.ndarray, labels: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """
    The most confident correct predictions and the most confident wrong ones.

    A confident mistake is the informative failure: the model was sure, so its
    error says something about what it learned rather than about an image it
    found ambiguous.
    """
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    order = np.argsort(-confidence, kind="stable")
    best = order[predicted[order] == labels[order]][:count]
    worst = order[predicted[order] != labels[order]][:count]
    return best, worst


def time_per_image(predict_one: Callable[[str], np.ndarray], filenames: Sequence[str],
                   sample: int = TIMING_SAMPLE) -> dict[str, float]:
    """
    Wall-clock latency of classifying one photograph, file on disk to class scores.

    Measured one image at a time after a warm-up call, because that is how a
    labeling tool or a field app would call it.
    """
    predict_one(filenames[0])
    times = []
    for name in filenames[:sample]:
        start = time.perf_counter()
        predict_one(name)
        times.append(1000 * (time.perf_counter() - start))
    return {"images": len(times), "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95))}


def image_predictor(network: torch.nn.Module, transform,
                    finish: Callable[[np.ndarray], np.ndarray] | None = None) -> Callable[[str], np.ndarray]:
    """
    A function from one filename to class scores, for timing the whole path from disk.

    `finish` turns the network's output into scores; the few-shot classifier
    uses it to compare the embedding with the prototypes.
    """
    target = device()
    network = network.to(target).eval()

    @torch.no_grad()
    def predict(filename: str) -> np.ndarray:
        output = network(transform(load_image(filename))[None].to(target)).float().cpu().numpy()
        return output if finish is None else finish(output)

    return predict


def score_models(models: dict[str, tuple[np.ndarray, int, str]], labels: np.ndarray,
                 species_of_class: np.ndarray) -> pd.DataFrame:
    """One summary row per classifier, from its test scores, labels used and approach."""
    rows = [{"model": name, "approach": approach, "labels_used": used, **summarize(labels, scores, species_of_class)}
            for name, (scores, used, approach) in models.items()]
    return pd.DataFrame(rows)


def nearest_reference_similarity(query: np.ndarray, reference: np.ndarray, block: int = 2048) -> np.ndarray:
    """Each query photograph's cosine similarity to its most similar reference photograph."""
    query_unit, reference_unit = unit_rows(query), unit_rows(reference)
    return np.concatenate([(query_unit[start:start + block] @ reference_unit.T).max(axis=1)
                           for start in range(0, len(query_unit), block)])


def accuracy_by_closeness(closeness: dict[str, np.ndarray], correct: dict[str, np.ndarray],
                          bins=CLOSENESS_BINS) -> pd.DataFrame:
    """
    Share of photographs and accuracy per band of similarity to the nearest training photograph.

    If two partitions score the same within each band but fill the bands
    differently, the difference in their overall accuracy is a difference in
    how close they sit to the training data, not in how the model behaves.
    """
    tables = []
    for name, values in closeness.items():
        band = pd.cut(values, bins, right=False)
        frame = pd.DataFrame({"band": band, "correct": correct[name]})
        table = frame.groupby("band", observed=False).agg(photographs=("correct", "size"),
                                                         accuracy=("correct", "mean"))
        table["share"] = table["photographs"] / table["photographs"].sum()
        tables.append(table.add_prefix(f"{name}_"))
    return pd.concat(tables, axis=1).reset_index()


def reweighted_accuracy(source: np.ndarray, target: np.ndarray, target_correct: np.ndarray,
                        bins=CLOSENESS_BINS) -> float:
    """The target partition's accuracy if its photographs sat as close to the training data as the source's."""
    weights = pd.Series(pd.cut(source, bins, right=False)).value_counts(normalize=True, sort=False)
    rates = pd.Series(target_correct).groupby(pd.cut(target, bins, right=False), observed=False).mean()
    shared = rates.dropna().index
    return float((rates[shared] * weights[shared]).sum() / weights[shared].sum())


def class_mix_accuracy(correct: np.ndarray, labels: np.ndarray, reference_labels: np.ndarray) -> float:
    """Accuracy if every class weighed as much as it does in the reference partition."""
    per_class = pd.Series(correct).groupby(labels).mean()
    weights = pd.Series(reference_labels).value_counts(normalize=True)
    shared = weights.index.intersection(per_class.index)
    return float((per_class[shared] * weights[shared]).sum() / weights[shared].sum())
