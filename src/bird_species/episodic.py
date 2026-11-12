"""
Training the embedding itself on few-shot practice tasks, the way a prototypical network does.

The prototype classifier in `fewshot` compares photographs in a frozen
embedding: DINOv2 learned its features without labels and never saw a bird
class. A prototypical network (Snell et al. 2017) learns the embedding for the
comparison instead. Training is a stream of small practice tasks, or episodes:
pick a few classes, a few labeled photographs of each (the support set) and a
few more to classify (the query set), build prototypes from the support set,
and update the embedding so each query lands nearest its own class's
prototype. A Siamese network (Koch et al. 2015) learns from pairs instead,
same class or not. Either way the embedding is trained on base classes with
plenty of labels and judged on novel classes it never saw, with only a few
labels each.

Here the learned part is a small residual layer on top of the frozen DINOv2
features. Its last layer starts at zero, so before training it is exactly the
identity and the classifier is the frozen one; training moves away from that
only where the episodes reward it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional

from bird_species.config import (
    EPISODE_CHECK,
    EPISODE_QUERIES,
    EPISODE_SHOTS,
    EPISODE_WAYS,
    EPISODES,
    PROJECTION_HIDDEN,
    PROJECTION_LR,
    PROJECTION_WEIGHT_DECAY,
    RANDOM_SEED,
    SPECIES_SPLIT,
)
from bird_species.fewshot import PrototypeClassifier

ROLES = ("base", "validation", "novel")


def split_species(
    species_of_class: np.ndarray,
    fractions: tuple[float, ...] = SPECIES_SPLIT,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    The role of every class (base, validation or novel), assigned by species.

    All plumage classes of one species share a role, so the novel classes are
    new birds, not the female of a male the embedding was trained on.
    """
    species = np.unique(species_of_class)
    shuffled = np.random.default_rng(seed).permutation(species)
    bounds = np.round(np.cumsum(fractions)[:-1] * len(species)).astype(int)
    role_of_species = {
        name: role
        for role, group in zip(ROLES, np.split(shuffled, bounds), strict=True)
        for name in group
    }
    return np.array([role_of_species[name] for name in species_of_class])


class ResidualProjection(nn.Module):
    """x + W2 relu(W1 x), with W2 and its bias at zero, so training starts from the identity."""

    def __init__(self, dim: int, hidden: int = PROJECTION_HIDDEN):
        super().__init__()
        self.up = nn.Linear(dim, hidden)
        self.down = nn.Linear(hidden, dim)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        # A cosine between unit vectors lies in [-1, 1]; the learned scale turns it into a
        # logit sharp enough for the softmax over the episode's classes.
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.down(functional.relu(self.up(features)))


def sample_episode(
    rows_by_class: list[np.ndarray], ways: int, shots: int, queries: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of one episode: a (ways, shots) support set and a (ways, queries) query set."""
    classes = rng.choice(len(rows_by_class), size=ways, replace=False)
    picks = np.stack(
        [rng.choice(rows_by_class[c], size=shots + queries, replace=False) for c in classes]
    )
    return picks[:, :shots], picks[:, shots:]


def episode_loss(
    projection: ResidualProjection, features: torch.Tensor, support: np.ndarray, queries: np.ndarray
) -> torch.Tensor:
    """Cross-entropy of each query against the episode's prototypes, by scaled cosine similarity."""
    ways, shots = support.shape
    embedded = functional.normalize(projection(features[torch.from_numpy(support.ravel())]), dim=1)
    prototypes = functional.normalize(embedded.view(ways, shots, -1).mean(dim=1), dim=1)
    asked = functional.normalize(projection(features[torch.from_numpy(queries.ravel())]), dim=1)
    targets = torch.arange(ways).repeat_interleave(queries.shape[1])
    return functional.cross_entropy(projection.scale * asked @ prototypes.T, targets)


@torch.no_grad()
def project(projection: ResidualProjection | None, features: np.ndarray) -> np.ndarray:
    """Features passed through the trained layer, or unchanged when there is none."""
    if projection is None:
        return features
    return projection.eval()(
        torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
    ).numpy()


def train_projection(
    features: np.ndarray,
    labels: np.ndarray,
    monitor: Callable[[ResidualProjection], float],
    ways: int = EPISODE_WAYS,
    shots: int = EPISODE_SHOTS,
    queries: int = EPISODE_QUERIES,
    episodes: int = EPISODES,
    check_every: int = EPISODE_CHECK,
    lr: float = PROJECTION_LR,
    weight_decay: float = PROJECTION_WEIGHT_DECAY,
    hidden: int = PROJECTION_HIDDEN,
    seed: int = RANDOM_SEED,
) -> tuple[ResidualProjection, pd.DataFrame]:
    """
    Train the residual layer on episodes drawn from `features` and `labels`.

    `monitor` scores a projection on held-out classes or photographs, higher
    is better. It runs before training and every `check_every` episodes, and
    the best-scoring weights are returned, so a layer that only overfits the
    episodes comes back as the identity it started from.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rows_by_class = [
        rows
        for rows in (np.flatnonzero(labels == c) for c in np.unique(labels))
        if len(rows) >= shots + queries
    ]
    projection = ResidualProjection(features.shape[1], hidden)
    optimizer = torch.optim.AdamW(projection.parameters(), lr=lr, weight_decay=weight_decay)
    data = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
    best_score, best_state = (
        monitor(projection),
        {k: v.clone() for k, v in projection.state_dict().items()},
    )
    history = [{"episode": 0, "loss": np.nan, "monitor": best_score}]
    losses = []
    for episode in range(1, episodes + 1):
        projection.train()
        support, asked = sample_episode(
            rows_by_class, min(ways, len(rows_by_class)), shots, queries, rng
        )
        loss = episode_loss(projection, data, support, asked)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if episode % check_every == 0:
            score = monitor(projection)
            history.append({"episode": episode, "loss": float(np.mean(losses)), "monitor": score})
            losses = []
            if score > best_score:
                best_score, best_state = (
                    score,
                    {k: v.clone() for k, v in projection.state_dict().items()},
                )
    projection.load_state_dict(best_state)
    return projection.eval(), pd.DataFrame(history)


def prototype_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    support: np.ndarray,
    asked: np.ndarray,
    projection: ResidualProjection | None = None,
) -> float:
    """
    Accuracy of prototypes built from the `support` rows on the `asked` rows.

    Only the classes present in the support set compete, so the score on the
    novel classes is an N-way task over the novel classes alone.
    """
    classes, support_labels = np.unique(labels[support], return_inverse=True)
    embedded = project(projection, features)
    model = PrototypeClassifier.fit(embedded[support], support_labels, len(classes))
    lookup = {label: index for index, label in enumerate(classes)}
    truth = np.array([lookup[label] for label in labels[asked]])
    return float((model.predict(embedded[asked]) == truth).mean())
