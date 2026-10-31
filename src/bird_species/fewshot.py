"""
Few-shot classification with class prototypes, and the labeling loop built on it.

A prototype is the average embedding of a class's labeled photographs. A new
photograph gets the class whose prototype it is most similar to, by cosine
similarity. This is the classifier a prototypical network ends with, and it
is what a Siamese network does once it has more than one reference image per
class: compare embeddings, never train a class-specific weight. With one
labeled photograph per class it is one-shot; with a handful, few-shot.

Features are scaled to unit length before averaging, so a prototype is a
direction and the cosine compares directions only. Centering them on the mean
of the training pool first (SimpleShot, Wang et al. 2019) is supported, since
removing the direction every photograph shares can sharpen the comparison; the
notebook tests it and finds no difference with these features, so it is off
by default.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

from bird_species.config import MAX_ROUNDS, PROPOSALS_PER_CLASS, TARGET_ACCURACY, TOP_K


def center_and_normalize(features: np.ndarray, center: np.ndarray | None = None) -> np.ndarray:
    shifted = features if center is None else features - center
    return shifted / np.clip(np.linalg.norm(shifted, axis=1, keepdims=True), 1e-12, None)


@dataclass
class PrototypeClassifier:
    prototypes: np.ndarray
    center: np.ndarray | None = None

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray, n_classes: int,
            center: np.ndarray | None = None) -> PrototypeClassifier:
        """One unit-length prototype per class from its labeled photographs."""
        normalized = center_and_normalize(features, center)
        sums = np.zeros((n_classes, features.shape[1]))
        np.add.at(sums, labels, normalized)
        if (np.bincount(labels, minlength=n_classes) == 0).any():
            raise ValueError("every class needs at least one labeled photograph")
        return cls(sums / np.linalg.norm(sums, axis=1, keepdims=True), center)

    def scores(self, features: np.ndarray) -> np.ndarray:
        """Cosine similarity of every photograph to every prototype."""
        return center_and_normalize(features, self.center) @ self.prototypes.T

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.scores(features).argmax(axis=1)


def margin(scores: np.ndarray) -> np.ndarray:
    """
    How much closer a photograph is to its best class than to the runner-up.

    A photograph near two prototypes is ambiguous even when both similarities
    are high, so the gap, not the top similarity, is the confidence used to
    rank predictions.
    """
    top_two = np.partition(scores, -2, axis=1)[:, -2:]
    return top_two[:, 1] - top_two[:, 0]


def classification_metrics(labels: np.ndarray, scores: np.ndarray, top_k: int = TOP_K) -> dict[str, float]:
    """Accuracy (the brief's metric), top-k accuracy and macro F1 from a score matrix."""
    predicted = scores.argmax(axis=1)
    return {"accuracy": float(accuracy_score(labels, predicted)),
            f"top{top_k}_accuracy": float(top_k_accuracy_score(labels, scores, k=top_k,
                                                               labels=np.arange(scores.shape[1]))),
            "macro_f1": float(f1_score(labels, predicted, average="macro"))}


def zero_shot_scores(image_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
    """Cosine similarity between CLIP image and caption embeddings; no labeled photograph is used."""
    images = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
    texts = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
    return images @ texts.T


def shot_curve(pool_features: np.ndarray, pool_labels: np.ndarray,
               draw: Callable[[int, int], np.ndarray], eval_features: np.ndarray,
               eval_labels: np.ndarray, shots: tuple[int, ...], seeds: tuple[int, ...],
               center: np.ndarray | None = None) -> pd.DataFrame:
    """Accuracy of the prototype classifier for each number of shots and each sampling seed."""
    n_classes = int(pool_labels.max()) + 1
    rows = []
    for shot in shots:
        for seed in seeds:
            chosen = draw(shot, seed)
            model = PrototypeClassifier.fit(pool_features[chosen], pool_labels[chosen], n_classes, center)
            rows.append({"shots": shot, "seed": seed, "labeled": len(chosen),
                         **classification_metrics(eval_labels, model.scores(eval_features))})
    return pd.DataFrame(rows)


def propose(scores: np.ndarray, unlabeled: np.ndarray, rejected: set[tuple[int, int]],
            per_class: int = PROPOSALS_PER_CLASS, rank_by: str = "similarity") -> pd.DataFrame:
    """
    The best predictions to show the labeler: the most confident few per predicted class.

    Ranking the whole pool at once would spend every round on the easiest
    classes, so the quota is per class. A photograph already rejected for a
    class is not proposed for that class again. Confidence is the cosine
    similarity to the predicted class's prototype; `rank_by="margin"` ranks by
    the gap to the runner-up class instead, the alternative it was chosen over.
    """
    positions = np.flatnonzero(unlabeled)
    candidates = pd.DataFrame({"position": positions,
                               "predicted": scores[positions].argmax(axis=1),
                               "margin": margin(scores[positions]),
                               "similarity": scores[positions].max(axis=1)})
    if rejected:
        seen = pd.MultiIndex.from_tuples(sorted(rejected), names=["position", "predicted"])
        keep = ~pd.MultiIndex.from_frame(candidates[["position", "predicted"]]).isin(seen)
        candidates = candidates[keep]
    ranked = candidates.sort_values(rank_by, ascending=False, kind="stable")
    return ranked.groupby("predicted", sort=False).head(per_class).reset_index(drop=True)


@dataclass
class LabelingResult:
    history: pd.DataFrame
    proposals: pd.DataFrame
    labeled: np.ndarray
    model: PrototypeClassifier


def labeling_loop(pool_features: np.ndarray, pool_labels: np.ndarray, initial: np.ndarray,
                  monitors: Mapping[str, tuple[np.ndarray, np.ndarray]], center: np.ndarray | None = None,
                  stop_on: str = "test", target: float = TARGET_ACCURACY,
                  per_class: int = PROPOSALS_PER_CLASS, max_rounds: int = MAX_ROUNDS,
                  rank_by: str = "similarity") -> LabelingResult:
    """
    Grow the labeled set from the classifier's own best predictions until it is accurate enough.

    Each round fits prototypes on the labeled photographs, proposes the most
    confident predictions from the unlabeled pool, and keeps only the ones the
    labeler confirms. `pool_labels` plays the labeler: it answers yes or no for
    a proposed photograph and is never read for anything else. The `monitors`
    are scored after every round and only decide when to stop; proposals never
    depend on them, so the stopping rule the brief sets on the test set cannot
    change which photographs get labeled.
    """
    n_classes = int(pool_labels.max()) + 1
    labeled = initial.copy()
    rejected: set[tuple[int, int]] = set()
    history, proposal_log = [], []
    batch: pd.DataFrame | None = None
    for round_number in range(max_rounds + 1):
        model = PrototypeClassifier.fit(pool_features[labeled], pool_labels[labeled], n_classes, center)
        row = {"round": round_number, "labeled": int(labeled.sum())}
        for name, (features, labels) in monitors.items():
            row[f"{name}_accuracy"] = float(accuracy_score(labels, model.predict(features)))
        if batch is None:
            row.update(proposed=np.nan, accepted=np.nan, precision=np.nan)
        else:
            row.update(proposed=len(batch), accepted=int(batch["accepted"].sum()),
                       precision=float(batch["accepted"].mean()))
        history.append(row)
        if row[f"{stop_on}_accuracy"] >= target or round_number == max_rounds:
            break
        batch = propose(model.scores(pool_features), ~labeled, rejected, per_class, rank_by)
        batch["true"] = pool_labels[batch["position"]]
        batch["accepted"] = batch["predicted"] == batch["true"]
        batch["round"] = round_number + 1
        labeled[batch.loc[batch["accepted"], "position"]] = True
        rejected.update(zip(batch.loc[~batch["accepted"], "position"],
                            batch.loc[~batch["accepted"], "predicted"], strict=True))
        proposal_log.append(batch)
    columns = ["position", "predicted", "margin", "similarity", "true", "accepted", "round"]
    proposals = pd.concat(proposal_log, ignore_index=True) if proposal_log else pd.DataFrame(columns=columns)
    return LabelingResult(pd.DataFrame(history), proposals, labeled, model)


def prototype_neighbors(prototypes: np.ndarray, species_of_class: np.ndarray) -> pd.DataFrame:
    """For every class, its most similar other class prototype and whether that class is the same species."""
    similarity = prototypes @ prototypes.T
    np.fill_diagonal(similarity, -np.inf)
    nearest = similarity.argmax(axis=1)
    return pd.DataFrame({"class_index": np.arange(len(prototypes)), "nearest_class": nearest,
                         "similarity": similarity[np.arange(len(prototypes)), nearest],
                         "same_species": species_of_class == species_of_class[nearest]})


def references_before(result: LabelingResult, initial: np.ndarray, pool_labels: np.ndarray,
                      round_number: int, class_index: int, count: int) -> np.ndarray:
    """Pool positions of photographs of `class_index` that were labeled before `round_number`."""
    labeled = initial.copy()
    earlier = result.proposals[(result.proposals["round"] < round_number) & result.proposals["accepted"]]
    labeled[earlier["position"].to_numpy()] = True
    return np.flatnonzero(labeled & (pool_labels == class_index))[:count]


def compare_backbones(features: Mapping[str, np.ndarray], evaluation: np.ndarray,
                      labels: np.ndarray, draw: Callable[[int, int], np.ndarray], shots: tuple[int, ...],
                      seeds: tuple[int, ...]) -> pd.DataFrame:
    """
    The shot curve of every backbone on the same draws of labeled photographs.

    `pool` and `evaluation` are row positions into each feature matrix and
    `draw(shots, seed)` returns pool row positions, so every backbone sees
    exactly the same labeled photographs and only the features differ.
    """
    curves = []
    for name, matrix in features.items():
        curve = shot_curve(matrix, labels, draw, matrix[evaluation], labels[evaluation], shots, seeds)
        curves.append(curve.assign(backbone=name))
    return pd.concat(curves, ignore_index=True)


def verification_cases(result: LabelingResult, initial: np.ndarray, pool_labels: np.ndarray,
                       pool_filenames: np.ndarray, class_names: np.ndarray, round_number: int,
                       per_outcome: int, references: int) -> list[dict]:
    """Confirmed and rejected proposals from one round, one per proposed class, each with labeled examples of it."""
    batch = result.proposals[result.proposals["round"] == round_number]
    shown = pd.concat([batch[batch["accepted"] == outcome].drop_duplicates("predicted").head(per_outcome)
                       for outcome in (True, False)])
    cases = []
    for row in shown.itertuples(index=False):
        verdict = "confirmed" if row.accepted else f"rejected, is {class_names[row.true]}"
        cases.append({"target": pool_filenames[row.position], "accepted": bool(row.accepted),
                      "title": f"proposed {class_names[row.predicted]}\n({verdict})",
                      "references": pool_filenames[references_before(result, initial, pool_labels, round_number,
                                                                     row.predicted, references)].tolist()})
    return cases


def compare_rankings(pool_features: np.ndarray, pool_labels: np.ndarray, initial: np.ndarray,
                     monitors: Mapping[str, tuple[np.ndarray, np.ndarray]], rankings: tuple[str, ...],
                     stop_on: str) -> pd.DataFrame:
    """
    Run the labeling loop once per confidence ranking and summarize what each cost.

    The cost of a ranking is the number of proposals the labeler reviews and
    the labels it takes to reach the target; the loop that gets there with
    fewer is the better way to choose "best predictions".
    """
    rows = []
    for ranking in rankings:
        history = labeling_loop(pool_features, pool_labels, initial, monitors, stop_on=stop_on,
                                rank_by=ranking).history
        reviewed = int(history["proposed"].sum())
        rows.append({"ranking": ranking, "rounds": int(history["round"].max()),
                     "labeled": int(history["labeled"].iloc[-1]), "reviewed": reviewed,
                     "confirmed_share": float(history["accepted"].sum() / reviewed) if reviewed else np.nan,
                     f"{stop_on}_accuracy": float(history[f"{stop_on}_accuracy"].iloc[-1])})
    return pd.DataFrame(rows)
