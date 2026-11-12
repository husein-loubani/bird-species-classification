"""Tests for the evaluation helpers, pinned against hand counts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bird_species.embeddings import eval_transform
from bird_species.modeling.evaluate import (
    accuracy_by_size,
    best_and_worst,
    cached_probabilities,
    confused_pairs,
    error_kinds,
    per_class_accuracy,
    predict_probabilities,
    summarize,
    time_per_image,
)
from tests.conftest import make_photo

SPECIES = np.array(["mallard", "mallard", "loon"])
# Three toy classes are fewer than top-5 needs; sklearn warns the top-5 score is trivially perfect.
FEW_CLASSES = pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")


@FEW_CLASSES
def test_summary_separates_variant_mistakes_from_species_mistakes():
    labels = np.array([0, 1, 2, 2])
    scores = np.eye(3)[[0, 0, 2, 1]]
    summary = summarize(labels, scores, SPECIES)
    assert summary["accuracy"] == 0.5
    assert summary["species_accuracy"] == 0.75
    assert summary["accuracy_low"] < 0.5 < summary["accuracy_high"]


def test_error_kinds_count_each_outcome():
    kinds = error_kinds(np.array([0, 1, 2, 2]), np.array([0, 0, 2, 1]), SPECIES)
    assert kinds.to_dict() == {"correct": 2, "wrong variant, right species": 1, "wrong species": 1}


def test_per_class_accuracy_reads_the_diagonal():
    accuracy = per_class_accuracy(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 0]), 3)
    np.testing.assert_allclose(accuracy, [0.5, 1.0, 0.0])


def test_confused_pairs_rank_the_most_common_mistake_first():
    table = confused_pairs(np.array([0, 0, 0, 2]), np.array([1, 1, 0, 0]), ["a", "b", "c"], SPECIES)
    first = table.iloc[0]
    assert (first["true_class"], first["predicted_class"], first["errors"]) == ("a", "b", 2)
    assert bool(first["same_species"])


def test_accuracy_by_size_bins_into_quartiles():
    fraction = np.linspace(0.01, 0.99, 8)
    table = accuracy_by_size(fraction, {"model": np.array([0, 0, 1, 1, 1, 1, 1, 1], dtype=bool)})
    assert table["model"].tolist() == [0.0, 1.0, 1.0, 1.0]
    assert table["size_quartile"].tolist() == ["Q1", "Q2", "Q3", "Q4"]


def test_best_and_worst_are_the_most_confident_hits_and_misses():
    probabilities = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.99, 0.01]])
    best, worst = best_and_worst(probabilities, np.array([0, 0, 0, 1]), count=2)
    assert best.tolist() == [0, 1]
    assert worst.tolist() == [3, 2]


def test_cached_probabilities_compute_once_per_key(tmp_path):
    calls = []
    path = tmp_path / "scores.npy"
    for key in ("model-a", "model-a", "model-b"):
        cached_probabilities(path, lambda: calls.append(1) or np.ones((2, 2)), key)
    assert len(calls) == 2, "the same key reads the file, a new model recomputes"


def test_cached_predictions_follow_the_model_fingerprint(tmp_path, raw_dir, monkeypatch):
    from bird_species.embeddings import ImageFiles
    from bird_species.modeling import evaluate

    monkeypatch.setattr(
        evaluate,
        "ImageFiles",
        lambda names, transform: ImageFiles(names, transform, raw_dir=raw_dir),
    )
    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 4)
    )
    model.cache_key = "first"
    transform = eval_transform((0.5,) * 3, (0.5,) * 3)
    names = ["train/00000.jpg", "test/00000.jpg"]
    path = tmp_path / "scores.npy"
    first = evaluate.cached_predictions(model, names, transform, path, workers=0)
    with torch.no_grad():
        model[2].weight.mul_(3.0)
    assert np.array_equal(
        evaluate.cached_predictions(model, names, transform, path, workers=0), first
    )
    model.cache_key = "retrained"
    assert not np.array_equal(
        evaluate.cached_predictions(model, names, transform, path, workers=0), first
    )


def test_predicted_probabilities_sum_to_one(raw_dir):
    model = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 4)
    )
    names = ["train/00000.jpg", "train/00001.jpg", "test/00000.jpg"]
    probabilities = predict_probabilities(
        model, names, eval_transform((0.5,) * 3, (0.5,) * 3), batch_size=2, workers=0
    )
    assert probabilities.shape == (3, 4)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)


def test_time_per_image_reports_the_sample_size():
    timing = time_per_image(lambda name: np.zeros(1), ["a", "b", "c"], sample=2)
    assert timing["images"] == 2 and timing["median_ms"] >= 0
    assert timing["p95_ms"] >= timing["median_ms"] or timing["p95_ms"] == pytest.approx(
        timing["median_ms"]
    )


@FEW_CLASSES
def test_score_models_gives_one_row_per_classifier():
    from bird_species.modeling.evaluate import score_models

    labels = np.array([0, 1, 2])
    table = score_models(
        {"perfect": (np.eye(3), 3, "finetuned"), "constant": (np.eye(3)[[0, 0, 0]], 0, "baseline")},
        labels,
        SPECIES,
    )
    assert table.set_index("model").loc["perfect", "accuracy"] == 1.0
    assert table.set_index("model").loc["constant", "species_accuracy"] == pytest.approx(2 / 3)


def test_image_predictor_runs_from_a_filename(raw_dir, monkeypatch):
    from bird_species.modeling import evaluate

    monkeypatch.setattr(evaluate, "load_image", lambda name: make_photo())
    network = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    predict = evaluate.image_predictor(
        network, eval_transform((0.5,) * 3, (0.5,) * 3), finish=lambda out: out * 2
    )
    assert predict("any.jpg").shape == (1, 3)


def test_nearest_reference_similarity_finds_the_closest_reference():
    from bird_species.modeling.evaluate import nearest_reference_similarity

    reference = np.array([[1.0, 0.0], [0.0, 1.0]])
    query = np.array([[2.0, 0.1], [0.0, -1.0]])
    np.testing.assert_allclose(
        nearest_reference_similarity(query, reference, block=1),
        [2 / np.hypot(2, 0.1), 0.0],
        atol=1e-6,
    )


def test_closeness_table_and_reweighting_match_a_hand_count():
    from bird_species.modeling.evaluate import accuracy_by_closeness, reweighted_accuracy

    bins = (0.0, 0.5, 1.0)
    closeness = {
        "validation": np.array([0.9, 0.9, 0.9, 0.2]),
        "test": np.array([0.9, 0.2, 0.2, 0.2]),
    }
    correct = {
        "validation": np.array([1, 1, 1, 0], dtype=bool),
        "test": np.array([1, 0, 0, 1], dtype=bool),
    }
    table = accuracy_by_closeness(closeness, correct, bins)
    assert table["validation_share"].tolist() == [0.25, 0.75]
    assert table["test_accuracy"].tolist() == [pytest.approx(1 / 3), 1.0]
    # test rates (1/3 far, 1.0 near) weighted by validation's mix (1/4 far, 3/4 near)
    assert reweighted_accuracy(
        closeness["validation"], closeness["test"], correct["test"], bins
    ) == pytest.approx(1 / 12 + 3 / 4)


def test_class_mix_accuracy_reweights_per_class_rates():
    from bird_species.modeling.evaluate import class_mix_accuracy

    correct = np.array([1, 1, 0, 0], dtype=bool)
    labels = np.array([0, 0, 1, 1])
    assert class_mix_accuracy(correct, labels, np.array([0, 0, 0, 1])) == pytest.approx(0.75)


def test_confused_pairs_break_ties_by_class_index():
    labels = np.array([2, 2, 0, 0, 1, 1])
    predicted = np.array([0, 0, 1, 1, 2, 2])
    table = confused_pairs(labels, predicted, ["a", "b", "c"], np.array(["x", "y", "z"]), top=2)
    assert table[["true", "predicted"]].to_numpy().tolist() == [[0, 1], [1, 2]]
