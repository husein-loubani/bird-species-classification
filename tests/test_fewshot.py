"""Tests for the prototype classifier, the proposal rule and the labeling loop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bird_species.fewshot import (
    PrototypeClassifier,
    center_and_normalize,
    classification_metrics,
    labeling_loop,
    margin,
    propose,
    prototype_neighbors,
    references_before,
    shot_curve,
    zero_shot_scores,
)


def test_centered_features_have_unit_length():
    features = np.array([[3.0, 4.0], [1.0, 1.0]])
    normalized = center_and_normalize(features, np.zeros(2))
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0)
    np.testing.assert_allclose(normalized[0], [0.6, 0.8])


def test_prototypes_classify_separated_clusters(clusters):
    features, labels = clusters
    model = PrototypeClassifier.fit(features[::4], labels[::4], 3)
    assert (model.predict(features) == labels).mean() == 1.0
    np.testing.assert_allclose(np.linalg.norm(model.prototypes, axis=1), 1.0)


def test_prototypes_refuse_a_class_without_examples(clusters):
    features, labels = clusters
    with pytest.raises(ValueError, match="at least one"):
        PrototypeClassifier.fit(features[labels < 2], labels[labels < 2], 3)


def test_margin_is_the_gap_between_the_two_best_scores():
    np.testing.assert_allclose(margin(np.array([[0.9, 0.2, 0.5], [0.1, 0.1, 0.3]])), [0.4, 0.2])


def test_metrics_match_a_hand_count():
    scores = np.array([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
    metrics = classification_metrics(np.array([0, 1, 1, 2]), scores, top_k=2)
    assert metrics["accuracy"] == 0.75
    assert metrics["top2_accuracy"] == 1.0
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)


def test_zero_shot_scores_are_cosines():
    scores = zero_shot_scores(np.array([[2.0, 0.0]]), np.array([[1.0, 0.0], [0.0, 3.0]]))
    np.testing.assert_allclose(scores, [[1.0, 0.0]])


def test_proposals_respect_the_quota_the_labeled_set_and_past_rejections():
    scores = np.array([[0.9, 0.1], [0.8, 0.1], [0.7, 0.6], [0.2, 0.9], [0.95, 0.0]])
    unlabeled = np.array([True, True, True, True, False])
    batch = propose(scores, unlabeled, rejected={(0, 0)}, per_class=1, rank_by="margin")
    assert set(zip(batch["position"], batch["predicted"], strict=True)) == {(1, 0), (3, 1)}
    by_similarity = propose(scores, unlabeled, rejected=set(), per_class=1, rank_by="similarity")
    assert by_similarity.loc[by_similarity["predicted"] == 0, "position"].item() == 0


def test_the_loop_adds_only_confirmed_labels_and_never_reproposes_a_rejection(clusters):
    """Photographs whose answer disagrees with the prediction are rejected, stay unlabeled, and are not shown again."""
    features, labels = clusters
    answers = labels.copy()
    disputed = [1, 21, 41]
    answers[disputed] = (labels[disputed] + 1) % 3
    initial = np.zeros(len(labels), dtype=bool)
    initial[[0, 20, 40]] = True
    result = labeling_loop(features, answers, initial, {"test": (features, labels)},
                           target=1.1, per_class=8, max_rounds=3)
    accepted = result.proposals[result.proposals["accepted"]]
    assert (accepted["true"] == accepted["predicted"]).all()
    assert (result.labeled & ~initial).sum() == len(accepted)
    assert not result.labeled[disputed].any(), "a disputed photograph was labeled"
    assert set(result.proposals.loc[~result.proposals["accepted"], "position"]) == set(disputed)
    assert not result.proposals.duplicated(["position", "predicted"]).any()


def test_the_loop_stops_as_soon_as_the_monitor_reaches_the_target(clusters):
    features, labels = clusters
    initial = np.zeros(len(labels), dtype=bool)
    initial[[0, 20, 40]] = True
    noisy = labels.copy()
    noisy[:6] = (noisy[:6] + 1) % 3
    result = labeling_loop(features, labels, initial, {"test": (features, noisy)},
                           target=0.85, per_class=4, max_rounds=10)
    reached = result.history["test_accuracy"] >= 0.85
    assert reached.iloc[-1] and not reached.iloc[:-1].any()


def test_the_loop_proposes_nothing_when_the_start_already_meets_the_target(clusters):
    features, labels = clusters
    initial = np.zeros(len(labels), dtype=bool)
    initial[[0, 20, 40]] = True
    result = labeling_loop(features, labels, initial, {"test": (features, labels)})
    assert len(result.history) == 1 and result.proposals.empty
    assert "accepted" in result.proposals.columns


def test_monitor_labels_cannot_change_which_photographs_get_labeled(clusters):
    """Scrambling the monitored labels moves only the accuracy column, never the proposals."""
    features, labels = clusters
    initial = np.zeros(len(labels), dtype=bool)
    initial[[0, 20, 40]] = True
    scrambled = np.random.default_rng(0).permutation(labels)
    runs = [labeling_loop(features, labels, initial, {"test": (features, monitor)},
                          target=1.1, per_class=2, max_rounds=3) for monitor in (labels, scrambled)]
    pd.testing.assert_frame_equal(runs[0].proposals, runs[1].proposals)
    assert not runs[0].history["test_accuracy"].equals(runs[1].history["test_accuracy"])


def test_shot_curve_has_one_row_per_shot_and_seed(clusters):
    features, labels = clusters

    def draw(shots, seed):
        rng = np.random.default_rng(seed)
        return np.concatenate([rng.choice(np.flatnonzero(labels == c), shots, replace=False) for c in range(3)])

    curve = shot_curve(features, labels, draw, features, labels, (1, 2), (0, 1))
    assert len(curve) == 4 and set(curve["labeled"]) == {3, 6}


def test_prototype_neighbors_flag_same_species_pairs():
    prototypes = np.array([[1.0, 0.0], [0.99, 0.141], [0.0, 1.0]])
    table = prototype_neighbors(prototypes, np.array(["mallard", "mallard", "loon"]))
    assert table["nearest_class"].tolist() == [1, 0, 1]
    assert table["same_species"].tolist() == [True, True, False]


def test_references_come_only_from_earlier_rounds(clusters):
    features, labels = clusters
    initial = np.zeros(len(labels), dtype=bool)
    initial[[0, 20, 40]] = True
    result = labeling_loop(features, labels, initial, {"test": (features, labels)},
                           target=1.1, per_class=2, max_rounds=2)
    first_round = references_before(result, initial, labels, 1, 0, 5)
    assert first_round.tolist() == [0]
    later = references_before(result, initial, labels, 2, 0, 5)
    assert len(later) > 1 and set(labels[later]) == {0}


def test_compare_rankings_reports_one_row_per_ranking(clusters):
    from bird_species.fewshot import compare_rankings

    features, labels = clusters
    initial = np.zeros(len(labels), dtype=bool)
    initial[[0, 20, 40]] = True
    table = compare_rankings(features, labels, initial, {"validation": (features, labels)},
                             ("margin", "similarity"), stop_on="validation")
    assert table["ranking"].tolist() == ["margin", "similarity"]
    assert {"rounds", "labeled", "reviewed", "confirmed_share", "validation_accuracy"} <= set(table.columns)


def test_verification_cases_pair_each_proposal_with_earlier_labels(clusters):
    from bird_species.fewshot import verification_cases

    features, labels = clusters
    answers = labels.copy()
    answers[[1, 21]] = (labels[[1, 21]] + 1) % 3
    initial = np.zeros(len(labels), dtype=bool)
    initial[[0, 20, 40]] = True
    result = labeling_loop(features, answers, initial, {"test": (features, labels)},
                           target=1.1, per_class=8, max_rounds=2)
    names = np.array([f"file{k}" for k in range(len(labels))])
    cases = verification_cases(result, initial, answers, names, np.array(["a", "b", "c"]), 1, 2, 3)
    assert {case["accepted"] for case in cases} == {True, False}
    assert all(case["references"] for case in cases)
    assert any("rejected" in case["title"] for case in cases)


def test_centering_is_optional_and_changes_only_the_reference_point(clusters):
    features, labels = clusters
    plain = PrototypeClassifier.fit(features, labels, 3)
    centered = PrototypeClassifier.fit(features, labels, 3, features.mean(axis=0))
    assert plain.center is None and centered.center is not None
    assert (plain.predict(features) == labels).all() and (centered.predict(features) == labels).all()
